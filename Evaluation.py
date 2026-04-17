import asyncio
import json
import re
import io
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

from sirchmunk import AgenticSearch
from sirchmunk.llm import OpenAIChat

try:
    from loguru import logger
except ImportError:
    logger = None

# ============ 加载配置 ============
def load_config(config_path="/mnt/a100b/default/chengxi/RAG/config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

config = load_config()

# 获取当前激活的模型配置
active_model = config["active_model"]
model_config = config["models"][active_model]

# 初始化 LLM
llm = OpenAIChat(
    base_url=model_config["base_url"],
    api_key=model_config["api_key"],
    model=model_config["model"]
)

# 动态生成输出文件名
INPUT_JSONL = config["input_jsonl"]
OUTPUT_JSONL = str(Path(config["output_dir"]) / f"{active_model}_sat_mc_output_thesis_v3_km_test_pred_with_reason_evidence_llm_time.jsonl")

SEARCH_PATHS = config["search_paths"]

# 打印当前使用的配置
print(f"当前使用模型: {active_model}")
print(f"Base URL: {model_config['base_url']}")
print(f"输出文件: {OUTPUT_JSONL}")
print("-" * 50)

# ============ 以下代码保持不变 ============
def extract_choice(text: str) -> str:
    """从模型输出中提取 A/B/C/D"""
    if not text:
        return "UNKNOWN"
    m = re.search(r"\b([ABCD])\b", text.upper())
    if m:
        return m.group(1)
    return "UNKNOWN"


def build_query(question: str, options: dict) -> str:
    """
    将 question + options 拼接成 QUERY
    """
    options_text = json.dumps(options, ensure_ascii=False, indent=2)
    return f"""{question}
options: {options_text}"""


def load_jsonl(path: str):
    """
    逐行读取 jsonl
    """
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[跳过] 第 {line_no} 行 JSON 解析失败: {e}")
                continue

            question = obj.get("question", "")
            options = obj.get("options", {})

            if not question or not isinstance(options, dict) or not options:
                print(f"[跳过] 第 {line_no} 行缺少 question 或 options")
                continue

            samples.append({
                "line_no": line_no,
                "raw": obj,
                "question": question,
                "options": options
            })
    return samples


def extract_summary_reason(log_text: str) -> str:
    """
    从日志中提取:
    [role=assistant] <SUMMARY> ... </SUMMARY>
    只保留 SUMMARY 内的内容作为 reason
    """
    if not log_text:
        return ""

    m = re.search(
        r"\[role=assistant\]\s*<SUMMARY>\s*(.*?)\s*</SUMMARY>",
        log_text,
        flags=re.S
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        r"<SUMMARY>\s*(.*?)\s*</SUMMARY>",
        log_text,
        flags=re.S
    )
    if m:
        return m.group(1).strip()

    return ""


async def run_search_and_capture_logs(searcher: AgenticSearch, query: str):
    """
    运行 Sirchmunk 检索，并尽量捕获 stdout / stderr / loguru 日志
    返回: (ctx, captured_logs)
    """
    buffer = io.StringIO()
    sink_id = None

    try:
        if logger is not None:
            sink_id = logger.add(
                buffer,
                level="DEBUG",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} - {message}"
            )

        with redirect_stdout(buffer), redirect_stderr(buffer):
            ctx = await searcher.search(
                query=query,
                paths=SEARCH_PATHS,
                # mode="DEEP",
                enable_dir_scan=True,
                return_context=True,
                llm_fallback=False,
                top_k_files=8,
            )

        captured_logs = buffer.getvalue()
        return ctx, captured_logs

    finally:
        if logger is not None and sink_id is not None:
            try:
                logger.remove(sink_id)
            except Exception:
                pass


async def solve_one(searcher: AgenticSearch, sample: dict) -> dict:
    """
    单条样本流程：
    1. 检索 + 捕获日志
    2. 从日志中提取 reason
    3. 提取 evidence_text
    4. 二次调用大模型判题
    5. 返回 pred / answer / is_correct / reason / evidence / llm_output / time
    """
    start_time = time.perf_counter()

    question = sample["question"]
    options = sample["options"]
    query = build_query(question, options)

    # 第一步：Sirchmunk 检索，并捕获日志
    ctx, search_logs = await run_search_and_capture_logs(searcher, query)

    # 从日志中抽取 reason
    reason = extract_summary_reason(search_logs)

    # 提取少量证据给第二次生成
    evidence_items = []
    if getattr(ctx, "cluster", None) and getattr(ctx.cluster, "evidences", None):
        for ev in ctx.cluster.evidences[:5]:
            evidence_items.append({
                "file": str(getattr(ev, "file_or_url", "")),
                "summary": getattr(ev, "summary", ""),
                "snippets": getattr(ev, "snippets", [])[:3],
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
{query}

检索证据：
{evidence_text}

请只输出一个字母：A 或 B 或 C 或 D。"""
        }
    ]

    final_resp = await llm.achat(final_messages, stream=False)
    llm_output = getattr(final_resp, "content", "") or ""
    pred = extract_choice(llm_output)

    gold = sample["raw"].get("answer")
    elapsed_time = time.perf_counter() - start_time

    result = {
        "pred": pred,
        "answer": gold,
        "is_correct": (pred == gold) if gold in {"A", "B", "C", "D"} else None,
        "reason": reason,
        "evidence": evidence_text,
        "llm_output": llm_output,
        "time": round(elapsed_time, 6)
    }
    return result


async def main():
    samples = load_jsonl(INPUT_JSONL)
    print(f"共读取到 {len(samples)} 条样本")

    searcher = AgenticSearch(llm=llm)

    output_path = Path(OUTPUT_JSONL)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    total_with_answer = 0

    with open(output_path, "w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples, 1):
            row_start_time = time.perf_counter()
            try:
                result = await solve_one(searcher, sample)

                pred = result.get("pred", "UNKNOWN")
                gold = result.get("answer")
                is_correct = result.get("is_correct")
                reason = result.get("reason", "")
                evidence = result.get("evidence", "")
                llm_output = result.get("llm_output", "")
                elapsed_time = result.get("time", 0.0)

                if gold in {"A", "B", "C", "D"}:
                    total_with_answer += 1
                    if is_correct:
                        correct += 1

                print(
                    f"[{idx}/{len(samples)}] "
                    f"pred={pred}, answer={gold}, is_correct={is_correct}, "
                    f"reason_len={len(reason)}, evidence_len={len(evidence)}, "
                    f"llm_output={repr(llm_output)}, time={elapsed_time:.6f}s"
                )

                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

            except Exception as e:
                elapsed_time = time.perf_counter() - row_start_time
                err = {
                    "pred": "ERROR",
                    "answer": sample["raw"].get("answer"),
                    "is_correct": None,
                    "reason": f"处理失败: {str(e)}",
                    "evidence": "",
                    "llm_output": "",
                    "time": round(elapsed_time, 6)
                }
                print(f"[{idx}/{len(samples)}] 处理失败: {e}, time={elapsed_time:.6f}s")
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"\n结果已保存到: {OUTPUT_JSONL}")
    if total_with_answer > 0:
        acc = correct / total_with_answer
        print(f"带标准答案样本数: {total_with_answer}")
        print(f"正确数: {correct}")
        print(f"准确率: {acc:.4f}")


if __name__ == "__main__":
    asyncio.run(main())