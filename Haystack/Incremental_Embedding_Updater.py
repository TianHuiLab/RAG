#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import hashlib
from typing import Dict, List, Any

from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder


# =========================
# 配置
# =========================
DATA_DIR = "/mnt/a100b/default/chengxi/Haystack/data"
EMB_PKL_PATH = os.path.join(DATA_DIR, "embeddings.pkl")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 你本次要同步的输入文件
INPUT_JSONL = "/mnt/a100b/default/chengxi/Haystack/data/1.jsonl"

# dataset 名（分区关键）
DATASET = "Command_Reference_400"

# 仅删除当前 dataset 的“缺失记录”
DELETE_REMOVED_IN_DATASET = True

BATCH_SIZE = 128


# =========================
# 工具函数
# =========================
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_existing_embeddings(pkl_path: str) -> List[Document]:
    if not os.path.exists(pkl_path):
        return []
    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)
    return docs if isinstance(docs, list) else []


def save_embeddings(pkl_path: str, docs: List[Document]) -> None:
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(docs, f)


def iter_rag_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as fin:
        for file_line_no, raw in enumerate(fin, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield file_line_no, obj


def normalize_content(s: str) -> str:
    return (s or "").strip()


def make_content_hash(content: str) -> str:
    return sha256_hex(normalize_content(content))


def make_record_key(obj: Dict[str, Any], file_path: str, file_line_no: int) -> str:
    """
    Stable identity for a record.
    Priority:
      1) meta.id
      2) sha256(source_file + line_no)
      3) sha256(abs_file_path + file_line_no) fallback
    """
    meta = obj.get("meta") or {}
    if isinstance(meta, dict):
        mid = meta.get("id")
        if isinstance(mid, str) and mid.strip():
            return mid.strip()

        src = meta.get("source_file")
        lno = meta.get("line_no")
        if isinstance(src, str) and src.strip() and isinstance(lno, int):
            return sha256_hex(f"{src.strip()}\n{lno}")

    return sha256_hex(f"{os.path.abspath(file_path)}\n{file_line_no}")


def build_existing_index(docs: List[Document]) -> Dict[str, Document]:
    """
    Index by record_key. record_key is stored at doc.meta["record_key"].
    """
    idx: Dict[str, Document] = {}
    for d in docs:
        meta = d.meta or {}
        rk = meta.get("record_key")
        if isinstance(rk, str) and rk:
            idx[rk] = d
            continue
        mid = meta.get("id")
        if isinstance(mid, str) and mid:
            idx[mid] = d
    return idx


# =========================
# 核心逻辑：按 dataset 分区删除
# =========================
def update_embeddings_from_rag_jsonl_partitioned(
    input_jsonl: str,
    pkl_path: str,
    embedding_model: str,
    dataset: str,
    delete_removed_in_dataset: bool = True,
    batch_size: int = 128,
) -> None:
    os.makedirs(os.path.dirname(pkl_path), exist_ok=True)

    # 1) load existing
    existing_docs = load_existing_embeddings(pkl_path)
    existing_index = build_existing_index(existing_docs)

    # 2) parse input -> determine which to embed
    target_keys = set()
    to_embed: List[Document] = []

    unchanged = 0
    added = 0
    changed = 0
    malformed = 0

    for file_line_no, obj in iter_rag_jsonl(input_jsonl):
        content = normalize_content(obj.get("content", ""))
        if not content:
            malformed += 1
            continue

        record_key = make_record_key(obj, input_jsonl, file_line_no)
        target_keys.add(record_key)

        # merge meta, inject partition + hashes
        meta_in = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
        meta: Dict[str, Any] = dict(meta_in)

        # 强制写入 dataset 分区字段
        meta["dataset"] = dataset

        # 增量更新跟踪字段
        meta["record_key"] = record_key
        meta["content_hash"] = make_content_hash(content)
        meta.setdefault("_input_file", input_jsonl)
        meta.setdefault("_input_file_line", file_line_no)

        old_doc = existing_index.get(record_key)
        if old_doc is None:
            added += 1
            to_embed.append(Document(content=content, meta=meta))
            continue

        old_meta = old_doc.meta or {}
        # 只有 dataset 相同才认为是“同一分区记录”，否则视为新增（避免跨数据集 key 冲突）
        if old_meta.get("dataset") != dataset:
            added += 1
            to_embed.append(Document(content=content, meta=meta))
            continue

        old_hash = old_meta.get("content_hash")
        if isinstance(old_hash, str) and old_hash == meta["content_hash"]:
            unchanged += 1
            continue

        changed += 1
        to_embed.append(Document(content=content, meta=meta))

    # 3) embed updates
    embedded_updates: Dict[str, Document] = {}
    if to_embed:
        embedder = SentenceTransformersDocumentEmbedder(model=embedding_model)
        embedder.warm_up()

        for i in range(0, len(to_embed), batch_size):
            batch = to_embed[i : i + batch_size]
            result = embedder.run(documents=batch)
            for d in result["documents"]:
                rk = (d.meta or {}).get("record_key")
                if isinstance(rk, str) and rk:
                    embedded_updates[rk] = d

    # 4) removal keys: only within this dataset
    removed_keys = set()
    if delete_removed_in_dataset:
        for rk, doc in existing_index.items():
            meta = doc.meta or {}
            if meta.get("dataset") == dataset:
                if rk not in target_keys:
                    removed_keys.add(rk)

    # 5) build new_docs
    new_docs: List[Document] = []

    # replace/keep/remove existing
    for rk, old_doc in existing_index.items():
        if rk in removed_keys:
            continue
        if rk in embedded_updates:
            new_docs.append(embedded_updates[rk])  # replace
        else:
            new_docs.append(old_doc)  # keep

    # add brand-new keys (not in existing_index OR existed but different dataset treated as added)
    existing_keys = set(existing_index.keys())
    for rk, doc in embedded_updates.items():
        if rk not in existing_keys:
            new_docs.append(doc)

    # 6) save
    save_embeddings(pkl_path, new_docs)

    # 7) report
    print("========== Embedding Update Report (Partitioned) ==========")
    print(f"Dataset            : {dataset}")
    print(f"Input JSONL         : {input_jsonl}")
    print(f"PKL path            : {pkl_path}")
    print(f"Model               : {embedding_model}")
    print("----------------------------------------------------------")
    print(f"Existing docs       : {len(existing_docs)}")
    print(f"Target records      : {len(target_keys)}")
    print(f"Unchanged           : {unchanged}")
    print(f"Added (new)         : {added}")
    print(f"Changed (re-embed)  : {changed}")
    print(f"Malformed/skipped   : {malformed}")
    print(f"Removed in dataset  : {len(removed_keys)} (enabled={delete_removed_in_dataset})")
    print(f"Final docs in PKL   : {len(new_docs)}")
    print("===========================================================")


if __name__ == "__main__":
    update_embeddings_from_rag_jsonl_partitioned(
        input_jsonl=INPUT_JSONL,
        pkl_path=EMB_PKL_PATH,
        embedding_model=EMBEDDING_MODEL,
        dataset=DATASET,
        delete_removed_in_dataset=DELETE_REMOVED_IN_DATASET,
        batch_size=BATCH_SIZE,
    )
