#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
from typing import List, Dict

INPUT_MD = r"/data/chengxi/Haystack/test/Command-Reference.md"
OUTPUT_JSONL = r"/data/chengxi/Haystack/test/Command-Reference.jsonl"




TITLE_RE = re.compile(r"^\*\*(.+?)\*\*\s*$")     # **show ipv6 route**
USAGE_RE = re.compile(r"^\s*-\s*Usage:\s*$")    # - Usage:
FENCE_RE = re.compile(r"^\s*```")               # ``` 或 ```bash


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_md_to_qa(md_text: str) -> List[Dict[str, str]]:
    lines = md_text.splitlines()
    out: List[Dict[str, str]] = []

    i = 0
    n = len(lines)

    while i < n:
        # 1. 匹配标题
        m_title = TITLE_RE.match(lines[i])
        if not m_title:
            i += 1
            continue

        i += 1  # 移到标题下一行

        # 2. 跳过标题后的空行
        while i < n and lines[i].strip() == "":
            i += 1

        # 3. 读取第一段文字（直到空行或 Usage）
        desc_lines: List[str] = []
        while i < n:
            if lines[i].strip() == "":
                break
            if USAGE_RE.match(lines[i]):
                break
            desc_lines.append(lines[i].strip())
            i += 1

        description = " ".join(desc_lines).strip()

        # 4. 向下寻找 - Usage:
        while i < n and not USAGE_RE.match(lines[i]):
            i += 1

        if i >= n:
            break

        i += 1  # 跳过 - Usage:

        # 5. 跳过空行
        while i < n and lines[i].strip() == "":
            i += 1

        # 6. 代码块开始
        if i >= n or not FENCE_RE.match(lines[i]):
            continue

        i += 1  # 跳过 opening fence

        code_lines: List[str] = []
        while i < n and not FENCE_RE.match(lines[i]):
            code_lines.append(lines[i])
            i += 1

        if i >= n:
            break

        i += 1  # 跳过 closing fence

        usage_code = "\n".join(code_lines).strip()

        if description and usage_code:
            out.append({
                "q": description,
                "a": usage_code
            })

    return out


def write_jsonl(path: str, records: List[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    md_text = read_text(INPUT_MD)
    records = parse_md_to_qa(md_text)
    write_jsonl(OUTPUT_JSONL, records)
    print(f"[OK] Parsed {len(records)} items")
    print(f"[OK] Saved to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
