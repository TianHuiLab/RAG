#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import hashlib
from pathlib import Path


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def convert_command_reference_to_rag_json(
    input_path: str,
    output_path: str,
    source_file_in_meta: str,
):
    """
    Convert Command-Reference.jsonl to a JSON array for RAG usage.
    """

    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rag_docs = []
    line_no = 0

    with in_path.open("r", encoding="utf-8") as fin:
        for raw_line in fin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            obj = json.loads(raw_line)

            content = str(obj.get("q", "")).strip()
            if not content:
                continue

            line_no += 1
            doc_id = sha256_hex(f"{content}\n{source_file_in_meta}\n{line_no}")

            rag_docs.append(
                {
                    "content": content,
                    "meta": {
                        "source_file": source_file_in_meta,
                        "line_no": line_no,
                        "id": doc_id,
                    },
                }
            )

    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(rag_docs, fout, ensure_ascii=False, indent=2)

    print(f"Converted {line_no} entries → {out_path}")


if __name__ == "__main__":
    convert_command_reference_to_rag_json(
        input_path="/data/chengxi/Haystack/test/Command-Reference.jsonl",
        output_path="/data/chengxi/Haystack/test/Command_Reference_400.jsonl",
        source_file_in_meta="/data/chengxi/Haystack/test/Command-Reference.jsonl",  # 你想要写进meta的路径
    )





