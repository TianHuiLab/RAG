'''
# 1) 清你的断点续跑记录
rm -f /mnt/a100b/default/chengxi/RAG/RAG_Sirchmunk/output.jsonl

# 2) 清 Sirchmunk 的知识缓存
rm -f /root/.sirchmunk/.cache/knowledge/knowledge_clusters.parquet

# 3) 清 Sirchmunk 的会话历史
rm -f /root/.sirchmunk/.cache/history/chat_history.db

from pathlib import Path

# 1) 清断点续跑记录
output_file = Path("/mnt/a100b/default/chengxi/RAG/RAG_Sirchmunk/output.jsonl")
output_file.unlink(missing_ok=True)

# 2) 清 Sirchmunk 的知识缓存
knowledge_file = Path("/root/.sirchmunk/.cache/knowledge/knowledge_clusters.parquet")
knowledge_file.unlink(missing_ok=True)

# 3) 清 Sirchmunk 的会话历史
history_file = Path("/root/.sirchmunk/.cache/history/chat_history.db")
history_file.unlink(missing_ok=True)

print("清理完成")



'''


import asyncio
import json
import re
from sirchmunk import AgenticSearch
from sirchmunk.llm import OpenAIChat

llm = OpenAIChat(
    base_url="http://43.159.131.233:3001/v1",
    api_key="*******",
    model="deepseek-chat"   
)

QUERY = """CryoSat-2卫星主体呈长方体，底部一直朝向地球，装有电子设备、无线电通信天线、激光反向反射器(LRR)、两副合成孔径/干涉高度计(SIRAL)卡塞格伦天线和DORIS天线。与CryoSat-1相比，CryoSat-2在性能、测量能力和精度等方面基本一致，但进行了多项改进。下列哪项不属于CryoSat-2相对于CryoSat-1的改进措施？
options: {
  "A": "改进星载软件以利于卫星运行",
  "B": "SIRAL系统设计包含全部备份以增强可靠性",
  "C": "增加热辐射面板保护极端温度条件下的电子设备",
  "D": "将卫星主体形状由长方体改为圆柱体以优化气动性能"
}"""

def extract_choice(text: str) -> str:
    m = re.search(r'\b([ABCD])\b', text.upper())
    if m:
        return m.group(1)
    return "UNKNOWN"

async def main():
    searcher = AgenticSearch(llm=llm)

    # 第一步：Sirchmunk 检索
    ctx = await searcher.search(
        query=QUERY,
        paths=["/mnt/a100b/default/chengxi/Base_LLM/Datasets-LLM/clearned_md"],
        # mode="DEEP",
        enable_dir_scan=True,
        return_context=True,
        llm_fallback=False,
        top_k_files=8,
    )

    # 提取少量证据给第二次生成
    evidence_items = []
    if ctx.cluster and ctx.cluster.evidences:
        for ev in ctx.cluster.evidences[:5]:
            evidence_items.append({
                "file": str(ev.file_or_url),
                "summary": ev.summary,
                "snippets": ev.snippets[:3],
            })

    evidence_text = json.dumps(evidence_items, ensure_ascii=False, indent=2)

    # 第二步：只让模型输出 A/B/C/D
    final_messages = [
        {
            "role": "system",
            "content": (
                "你是选择题判定器。"
                "你只能输出一个大写字母：A、B、C、D。"
                "禁止输出理由，禁止输出标点，禁止输出多余文字。"
            )
        },
        {
            "role": "user",
            "content": f"""题目：
{QUERY}

检索证据：
{evidence_text}

请只输出一个字母：A 或 B 或 C 或 D。"""
        }
    ]

    final_resp = await llm.achat(final_messages, stream=False)

    # 再做一次程序级约束，确保最终只打印 A/B/C/D
    answer = extract_choice(final_resp.content)
    print(answer)

asyncio.run(main())