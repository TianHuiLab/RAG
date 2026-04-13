#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import hashlib
from pathlib import Path


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def convert_q_a_jsonl_to_rag_jsonl(
    input_path: str,
    output_path: str,
) -> None:
    """
    Input (JSONL):
      {"q": "...", "a": "..."}
    Output (JSONL):
      {"content": "...(contains q and a)...", "meta": {"source_file": "...", "line_no": N, "id": "..."}}
    """

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_file_in_meta = str(in_path)

    written = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw_idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 坏行跳过（你也可以改成 raise 让程序中断）
                continue

            q = str(obj.get("q", "")).strip()
            a = str(obj.get("a", "")).strip()

            # content 必须同时包含 q 与 a
            if not q and not a:
                continue
            content = f"{q}\nCommand: {a}" if (q and a) else (q or f"Command: {a}")

            written += 1
            line_no = written  # 输出文件中的行号（从1开始）

            doc_id = sha256_hex(f"{content}\n{source_file_in_meta}\n{line_no}")

            rag_obj = {
                "content": content,
                "meta": {
                    "source_file": source_file_in_meta,
                    "line_no": line_no,
                    "id": doc_id,
                },
            }

            fout.write(json.dumps(rag_obj, ensure_ascii=False) + "\n")

    print(f"Done. Read lines (incl blank/bad): {raw_idx}, Written docs: {written}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    convert_q_a_jsonl_to_rag_jsonl(
        input_path="/data/chengxi/Haystack/data/source/Command-Reference.jsonl",
        output_path="/data/chengxi/Haystack/data/Command_Reference_400.jsonl",
    )
