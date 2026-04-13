import csv
import json

# 输入文件路径
input_file_path = '/mnt/a100b/default/chengxi/RAG/Haystack/data/source/在轨风险处置方案_整理版v2.csv'

# 输出文件路径
output_file_path = '/mnt/a100b/default/chengxi/RAG/Haystack/data/dangerousdealing.jsonl'

# 打开 CSV 文件并读取内容
with open(input_file_path, mode='r', encoding='utf-8') as infile:
    reader = csv.reader(infile)
    # 跳过表头
    next(reader)
    
    # 打开 JSONL 文件以写入数据
    with open(output_file_path, mode='w', encoding='utf-8') as outfile:
        # 从第二行开始遍历 CSV 内容
        for line_no, row in enumerate(reader, start=2):
            # 从每一行获取相关内容
            fault_item = row[0] if len(row) > 0 else ""
            fault_module = row[1] if len(row) > 1 else ""
            fault_description = row[2] if len(row) > 2 else ""
            disposal_plan = row[3] if len(row) > 3 else ""
            problem_source = row[4] if len(row) > 4 else ""

            # 将“故障项”、“故障模组”、“故障描述”、“处置方案”内容连接起来
            content = f"{fault_item}{fault_module}{fault_description}，处置措施为：{disposal_plan}"

            # 创建 JSON 对象
            json_obj = {
                "content": content,
                "meta": {
                    "source_file": input_file_path,
                    "line_no": line_no,
                    "id": ""  # 如果没有问题来源，字段为空
                }
            }

            # 写入 JSONL 文件
            outfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

print(f"数据已经成功写入 {output_file_path}")


