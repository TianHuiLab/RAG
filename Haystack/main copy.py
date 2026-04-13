#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
from config.environment import set_environment
from config.settings import DOCUMENT_EMBEDDER_MODEL, TEXT_EMBEDDER_MODEL, LLM_MODEL

from embeddings.document_embedder import create_document_embedder  # 可选：仅用于校验维度/不运行embedding
from embeddings.text_embedder import create_text_embedder

from pipeline.chat_prompt_builder import build_chat_prompt_template
from pipeline.retriever import create_retriever
from pipeline.chat_generator import create_chat_generator
from pipeline.rag_pipeline import create_rag_pipeline

from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

import os
import pickle
from typing import List


# =========================
# 0) Environment
# =========================
set_environment()

# =========================
# 1) Paths
# =========================
EMB_PKL_PATH = "/mnt/a100b/default/chengxi/Haystack/data/embeddings.pkl"
 

# =========================
# 2) PKL helper (只读)
# =========================
def load_docs_pkl(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding PKL not found: {file_path}")

    with open(file_path, "rb") as f:
        docs = pickle.load(f)

    if not isinstance(docs, list) or (len(docs) > 0 and not isinstance(docs[0], Document)):
        raise TypeError(
            "PKL content is not a List[haystack.Document]. "
            "Please ensure your embedding generator stored Haystack Document objects."
        )

    print(f"Embeddings loaded from {file_path} ({len(docs)})")
    return docs


def _assert_docs_have_embeddings(docs: List[Document]) -> None:
    missing = 0
    for d in docs:
        # haystack Document: d.embedding is typically a list/np array
        if getattr(d, "embedding", None) is None:
            missing += 1
    if missing > 0:
        raise ValueError(
            f"{missing}/{len(docs)} documents have no embedding. "
            "Your PKL must store Document objects with precomputed embeddings."
        )


# =========================
# 3) Main (只读pkl -> RAG)
# =========================
def main():

    # 1) load embeddings (全量) and build store
    docs_emb = load_docs_pkl(EMB_PKL_PATH)
    if not docs_emb:
        raise RuntimeError(f"No documents found in {EMB_PKL_PATH}.")

    _assert_docs_have_embeddings(docs_emb)

    document_store = InMemoryDocumentStore()
    document_store.write_documents(docs_emb)
    print(f"Number of documents in the document store: {document_store.count_documents()}")

    # 2) create query embedder / retriever / prompt builder / llm
    # query embedder: 把用户问题编码成向量
    text_embedder = create_text_embedder(TEXT_EMBEDDER_MODEL)

    # retriever: 依赖 document_store 中每个 doc 的 embedding
    retriever = create_retriever(document_store)

    prompt_builder = build_chat_prompt_template()
    chat_generator = create_chat_generator(LLM_MODEL, timeout=60)

    # 3) build pipeline
    rag = create_rag_pipeline()
    rag.add_component("text_embedder", text_embedder)
    rag.add_component("retriever", retriever)
    rag.add_component("prompt_builder", prompt_builder)
    rag.add_component("llm", chat_generator)

    rag.connect("text_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever", "prompt_builder")
    rag.connect("prompt_builder.prompt", "llm.messages")

    # 4) run query
    question = " 俄罗斯参与欧洲 Galileo 导航卫星系统开发时，主要采取了哪三项措施来促进 GLONASS 与 Galileo 系统的兼容与互操作？"

    try:
        response = rag.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )
        print("Response from LLM:", response["llm"]["replies"][0].text)
    except Exception as e:
        print(f"An error occurred: {e}")
        print(
            "Hint: if you see unexpected keyword argument 'text', "
            "your text_embedder.run signature differs. Align create_text_embedder()."
        )





if __name__ == "__main__":
    main()
