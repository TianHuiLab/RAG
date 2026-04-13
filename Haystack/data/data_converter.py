import json
import hashlib
from pathlib import Path

input_path = Path("/mnt/a100b/default/chengxi/Base_LLM/Datasets-LLM/corpus/all_sat_book_raw_chunks_by_word.jsonl")
output_path = Path("/mnt/a100b/default/chengxi/Base_LLM/Datasets-LLM/corpus/all_sat_book_raw_chunks_by_word_rag.jsonl")


def make_id(record: dict, line_no: int) -> str:
    """
    生成稳定的唯一 id。
    用 book/title/chunk/line_no 组合后做 sha1。
    """
    book = record.get("book", "")
    title = record.get("title", [])
    chunk = record.get("chunk", "")

    if not isinstance(title, list):
        title = [str(title)]

    raw = json.dumps(
        {
            "book": book,
            "title": title,
            "chunk": chunk,
            "line_no": line_no,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def convert_jsonl_to_rag(input_path: Path, output_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                skipped += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[跳过] 第 {line_no} 行 JSON 解析失败: {e}")
                skipped += 1
                continue

            chunk = record.get("chunk")
            if not isinstance(chunk, str) or not chunk.strip():
                print(f"[跳过] 第 {line_no} 行缺少有效的 chunk 字段")
                skipped += 1
                continue

            # 将除 chunk 外的原始字段全部放入 meta
            meta = {k: v for k, v in record.items() if k != "chunk"}

            # 补充标准元信息
            meta["source_file"] = str(input_path)
            meta["line_no"] = line_no
            meta["id"] = make_id(record, line_no)

            new_record = {
                "content": chunk,
                "meta": meta
            }

            fout.write(json.dumps(new_record, ensure_ascii=False) + "\n")
            total += 1

    print(f"转换完成: {total} 条写入 {output_path}")
    print(f"跳过记录: {skipped} 条")


if __name__ == "__main__":
    convert_jsonl_to_rag(input_path, output_path)