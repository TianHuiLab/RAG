import json

# 输入文件路径
input_file_path = '/mnt/a100b/default/chengxi/Satellite_Corpus/Satellite_database/SAT_Book_SFT/SAT-1-0/Step4_final/all_cot.json'

# 输出文件路径
output_file_path = '/mnt/a100b/default/chengxi/Haystack/data/1.jsonl'

# 打开并读取输入文件
with open(input_file_path, 'r', encoding='utf-8') as infile:
    data = json.load(infile)

# 准备一个列表来存储转换后的数据
transformed_data = []

# 处理每个条目
for entry in data:
    # 获取每个条目的对话内容
    conversations = entry.get('conversations', [])
    
    if len(conversations) >= 2:
        # 提取第一个问题和第二个回答
        question = conversations[0].get('value', '').strip()
        answer = conversations[1].get('value', '').strip()
        
        # 创建转换后的内容和元数据字典
        transformed_entry = {
            "content": f"Question: {question}\nAnswer: {answer}",
            "meta": {
                "source_file": input_file_path,
                "id": entry.get('id', '')
            }
        }
        transformed_data.append(transformed_entry)

# 将转换后的数据写入输出文件
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for entry in transformed_data:
        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"数据已成功转换并保存到 {output_file_path}")