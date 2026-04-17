import json
from collections import Counter

file_path = "/mnt/a100b/default/chengxi/RAG/RAG_Sirchmunk/sat_mc_output_thesis_v3_km_test_pred_with_reason_evidence_llm_time.jsonl"

counter = Counter()
total_lines = 0
total_with_is_correct = 0
bad_lines = 0

def normalize_value(v):
    """
    统一处理 is_correct 的取值：
    - JSON 布尔 true/false -> Python True/False
    - 字符串 "true"/"false" -> 统一转为 True/False
    - 其他值保持原样
    """
    if isinstance(v, str):
        s = v.strip().lower()
        if s == "true":
            return True
        elif s == "false":
            return False
        elif s == "null":
            return None
    return v

with open(file_path, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        total_lines += 1

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            print(f"第 {line_num} 行 JSON 解析失败")
            continue

        if "is_correct" in obj:
            total_with_is_correct += 1
            value = normalize_value(obj["is_correct"])
            counter[value] += 1

# 统计 true 数量和比例
true_count = counter[True]
true_ratio = true_count / total_with_is_correct if total_with_is_correct > 0 else 0

print("===== 统计结果 =====")
print(f"总行数: {total_lines}")
print(f"包含 is_correct 字段的样本数: {total_with_is_correct}")
print(f"JSON 解析失败行数: {bad_lines}")
print()

print("is_correct 各属性值数量：")
for k, v in counter.items():
    print(f"  {repr(k)}: {v}")

print()
print(f"is_correct = true 的总数: {true_count}")
print(f"is_correct = true 的比例: {true_ratio:.4%}")