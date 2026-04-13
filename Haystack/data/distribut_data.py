import json
import statistics
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

input_path = Path("/mnt/a100b/default/chengxi/Base_LLM/Datasets-LLM/corpus/all_sat_book_raw_chunks_by_word_rag.jsonl")
output_dir = input_path.parent / "content_length_stats"
output_dir.mkdir(parents=True, exist_ok=True)

lengths = []
bad_lines = []

with input_path.open("r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_lines.append(line_no)
            continue

        content = obj.get("content", "")
        if not isinstance(content, str):
            content = str(content)

        lengths.append(len(content))

if not lengths:
    raise ValueError("没有读取到有效的 content 数据")

# 转成 DataFrame
df = pd.DataFrame({
    "content_length": lengths
})

# 基本统计
stats = {
    "count": len(lengths),
    "min": min(lengths),
    "max": max(lengths),
    "mean": sum(lengths) / len(lengths),
    "median": statistics.median(lengths),
    "p10": df["content_length"].quantile(0.10),
    "p25": df["content_length"].quantile(0.25),
    "p50": df["content_length"].quantile(0.50),
    "p75": df["content_length"].quantile(0.75),
    "p90": df["content_length"].quantile(0.90),
    "p95": df["content_length"].quantile(0.95),
    "p99": df["content_length"].quantile(0.99),
}

print("=== content 字符数统计 ===")
for k, v in stats.items():
    if isinstance(v, float):
        print(f"{k}: {v:.2f}")
    else:
        print(f"{k}: {v}")

if bad_lines:
    print(f"\nJSON 解析失败的行数: {len(bad_lines)}")
    print(f"前20个异常行号: {bad_lines[:20]}")

# 保存详细长度表
length_csv = output_dir / "content_lengths.csv"
df.to_csv(length_csv, index=False, encoding="utf-8-sig")

# 保存统计摘要
stats_df = pd.DataFrame([stats])
stats_csv = output_dir / "content_length_summary.csv"
stats_df.to_csv(stats_csv, index=False, encoding="utf-8-sig")

# 直方图
plt.figure(figsize=(10, 6))
plt.hist(df["content_length"], bins=50, edgecolor="black")
plt.xlabel("Content Length (characters)")
plt.ylabel("Frequency")
plt.title("Distribution of Content Lengths")
plt.tight_layout()
hist_path = output_dir / "content_length_histogram.png"
plt.savefig(hist_path, dpi=200)
plt.close()

# 箱线图
plt.figure(figsize=(10, 4))
plt.boxplot(df["content_length"], vert=False)
plt.xlabel("Content Length (characters)")
plt.title("Boxplot of Content Lengths")
plt.tight_layout()
box_path = output_dir / "content_length_boxplot.png"
plt.savefig(box_path, dpi=200)
plt.close()

print("\n=== 输出文件 ===")
print(f"详细长度表: {length_csv}")
print(f"统计摘要:   {stats_csv}")
print(f"直方图:     {hist_path}")
print(f"箱线图:     {box_path}")