# 基于 Sirchmunk 的卫星领域知识库

# 一、知识库工具

## 概述

RAG_Sirchmunk.py是一个 Sirchmunk 知识库使用工工具，可以用检索增强生成（RAG）技术回答单一选择题，专注于单条问题的处理流程，包含知识检索和最终答案生成环节。

## 主要特点

- 使用 `Sirchmunk` 进行本地 Markdown 知识库检索
- 集成 OpenAI 兼容接口（示例使用 DeepSeek-Chat 模型）
- 两步式处理：检索证据 → 约束生成答案
- 程序级正则提取确保最终只输出选项字母（A/B/C/D）

## 输入输出说明

### 输入

脚本内部硬编码了一道关于 CryoSat-2 卫星的选择题：

**题目**：CryoSat-2 卫星主体呈长方体，底部一直朝向地球...（完整题干见代码）  
**选项**：
- A: 改进星载软件以利于卫星运行
- B: SIRAL 系统设计包含全部备份以增强可靠性
- C: 增加热辐射面板保护极端温度条件下的电子设备
- D: 将卫星主体形状由长方体改为圆柱体以优化气动性能

**知识库路径**：`~/clearned_md`

### 输出

脚本最终在控制台打印一个**单字母**（A、B、C 或 D），表示模型对该选择题的预测答案。

示例输出：
```
C
```

若答案提取失败，输出：
```
UNKNOWN
```

## 使用流程

### 1. 环境准备

- Python 3.8+
- 安装依赖：
  ```bash
  pip install sirchmunk openai
  ```
- 确保 LLM 服务地址可访问

### 2. 清理缓存


```bash
# 清理输出记录
rm -f /mnt/a100b/default/chengxi/RAG/RAG_Sirchmunk/output.jsonl

# 清理知识缓存
rm -f /root/.sirchmunk/.cache/knowledge/knowledge_clusters.parquet

# 清理会话历史
rm -f /root/.sirchmunk/.cache/history/chat_history.db
```


### 3. 配置参数

根据需要修改以下配置项：

| 参数 | 位置 | 说明 |
|------|------|------|
| `QUERY` | 第 30-39 行 | 修改为你要测试的题目和选项 |
| `paths` | 第 52 行 | 修改为你的本地知识库路径 |
| `base_url` | 第 18 行 | LLM API 服务地址 |
| `api_key` | 第 19 行 | API 密钥 |
| `model` | 第 20 行 | 模型名称 |

### 4. 运行脚本

```bash
python RAG_Sirchmunk.py
```

### 5. 解读输出

脚本运行后，控制台将打印预测的选项字母。完整的执行流程包括：
1. 初始化 AgenticSearch 实例
2. 在指定知识库中检索相关文档
3. 提取前 5 条证据（文件路径、摘要、片段）
4. 构造严格约束的提示词
5. 调用 LLM 生成答案
6. 正则提取并输出最终选项

## 原理说明

本脚本采用两阶段处理架构，具体流程如下：

### 阶段一：知识检索与证据收集

1. **查询构造**  
   将题干与选项 JSON 拼接为一个完整的自然语言查询字符串，作为 Sirchmunk 的输入。

2. **检索执行**  
   调用 `AgenticSearch.search()` 方法，在指定目录的 Markdown 文件中进行语义搜索。主要参数：
   - `paths`：知识库根目录
   - `enable_dir_scan=True`：启用目录扫描
   - `return_context=True`：返回完整上下文对象
   - `llm_fallback=False`：禁用 LLM 回退（仅依赖检索结果）
   - `top_k_files=8`：最多返回 8 个相关文件

3. **证据提取**  
   从返回的 `ctx.cluster.evidences` 中提取前 5 条证据，每条证据包含：
   - `file`：源文件路径
   - `summary`：证据摘要
   - `snippets`：相关文本片段（最多 3 条）

   证据数据被序列化为 JSON 格式，供下一阶段使用。

### 阶段二：约束生成与答案提取

1. **严格约束提示构造**  
   系统提示明确要求 LLM：
   - 仅输出单个大写字母（A/B/C/D）
   - 禁止输出任何理由、标点或多余文字

   用户消息包含原始题目和第一阶段收集的 JSON 格式证据。

2. **LLM 调用**  
   使用 `llm.achat()` 异步调用大模型，传入构造的消息列表，获取生成的原始文本。

3. **答案提取**  
   通过正则表达式 `\b([ABCD])\b` 从模型输出中提取首个大写字母：
   - 匹配成功：返回字母（如 `C`）
   - 匹配失败：返回 `UNKNOWN`

   该步骤确保了即使模型违反约束输出多余内容，最终结果仍然保持规范的选项格式。


## 自定义与扩展

修改 `QUERY` 变量中的题干和选项 JSON 内容。




## 注意事项

1. **缓存清理**  
   首次运行或更换知识库后，建议清理缓存以确保检索结果不受历史数据影响。

2. **模型兼容性**  
   LLM 服务必须兼容 OpenAI Chat Completion API 格式。若使用其他模型服务，请相应调整 `base_url` 和认证方式。




# 二、评测工具

## 概述

Eval_ds_chat.py是一个基于检索增强生成（RAG）的卫星领域选择题评测工具。脚本读取 JSONL 格式的题目文件，利用 `Sirchmunk` 智能检索框架从知识库路径中搜索相关证据，捕获检索过程的详细日志并提取推理摘要，最后再次调用大语言模型（LLM）输出最终选项（A/B/C/D）。所有预测结果、推理依据、证据及耗时信息均保存为新的 JSONL 文件，便于后续评估与分析。

## 主要特点

- 使用 `Sirchmunk` 进行上下文感知的文档检索与证据聚合
- 支持读取自定义本地 Markdown 格式数据
- 集成 OpenAI 兼容接口 
- 捕获 `loguru` 日志及标准输出/错误流，从中提取 `<SUMMARY>` 标签内容作为推理理由
- 二次 LLM 调用仅输出单字母选项，保证答案格式规整
- 统计准确率并记录每条样本的处理耗时

## 输入文件说明

**`INPUT_JSONL`**  

路径：`~/sat_mc_output_thesis_v3_km03.jsonl`

JSONL 格式，每行一个 JSON 对象，必须包含以下字段：

| 字段名   | 类型   | 描述                   |
|----------|--------|------------------------|
| question | string | 选择题题干             |
| options  | object | 键为选项字母（A/B/C/D），值为选项文本 |
| answer   | string | 标准答案字母，用于评估准确率 |

示例行：
```json
{"question": "地球是太阳系第几大行星？", "options": {"A": "第一", "B": "第二", "C": "第三", "D": "第四"}, "answer": "C"}
```

## 输出文件说明

**`OUTPUT_JSONL`**  
路径：`~/sat_mc_output_thesis_v3_km_test_pred_with_reason_evidence_llm_time.jsonl`

每行 JSON 对象包含以下字段：

| 字段名      | 类型          | 描述                                                         |
|-------------|---------------|--------------------------------------------------------------|
| pred        | string        | 模型预测的选项字母（A/B/C/D/UNKNOWN/ERROR）                  |
| answer      | string/null   | 原始数据中的标准答案（若无则为 null）                        |
| is_correct  | bool/null     | 预测是否正确（仅当标准答案为 A/B/C/D 时有意义）              |
| reason      | string        | 从 Sirchmunk 检索日志 `<SUMMARY>` 标签中提取的推理摘要       |
| evidence    | string        | JSON 字符串，包含前 5 条证据的元数据（文件路径、摘要、片段） |
| llm_output  | string        | 最终 LLM 调用返回的原始内容                                  |
| time        | float         | 处理该样本的总耗时（秒）                                     |

## 使用流程

### 1. 环境准备

- Python 3.8+
- 安装依赖：
  ```bash
  pip install sirchmunk openai loguru
  ```
- 确保 LLM 服务地址可访问

### 2. 配置路径

修改脚本顶部的路径常量：
```python
INPUT_JSONL = "你的输入文件.jsonl"
OUTPUT_JSONL = "你的输出文件.jsonl"
SEARCH_PATHS = ["/你的/知识库/目录"]
```

### 3. 配置 LLM

根据实际使用的模型服务调整 `OpenAIChat` 参数：
```python
llm = OpenAIChat(
    base_url="你的API地址",
    api_key="你的API密钥",
    model="模型名称"
)
```

### 4. 运行脚本

```bash
python Eval_ds_chat.py
```

脚本将依次处理每条样本，控制台实时显示进度和结果摘要。

### 5. 结果解读

- 输出 JSONL 文件可直接用于准确率统计、错误分析或可视化。
- 控制台最后会打印带标准答案样本的总体准确率。

## 原理说明

本管道由两个核心阶段构成：

### 阶段一：检索与推理摘要生成

1. **构建查询**  
   将题干与选项 JSON 拼接为一个自然语言查询字符串。

2. **启动 Sirchmunk 检索**  
   调用 `AgenticSearch.search()` 在指定目录的 Markdown 文件中搜索相关文档。  
   检索过程会输出详细日志（包括 `loguru` 输出及标准输出/错误流），脚本通过重定向捕获这些日志。

3. **提取推理摘要**  
   在捕获的日志中搜索 `<SUMMARY> ... </SUMMARY>` 标签块，将其内容作为模型的“思维链”理由（reason）。该部分反映了 Sirchmunk 在聚合证据后生成的总结性陈述。

4. **收集证据**  
   从检索返回的 `context.cluster.evidences` 中提取前 5 条证据，每条证据包含文件路径、摘要及前 3 条相关片段。这些证据被序列化为 JSON 字符串供下一阶段使用。

### 阶段二：最终答案生成

1. **构造严格约束的提示**  
   系统提示明确要求 LLM 仅输出大写字母 `A`、`B`、`C` 或 `D`，禁止任何额外文字。

2. **传递题目与证据**  
   用户消息包含原始题目和第一阶段收集的证据 JSON，要求模型基于这些信息给出答案。

3. **提取选项字母**  
   通过正则表达式 `\b([ABCD])\b` 从 LLM 返回内容中提取首个大写字母，若提取失败则标记为 `UNKNOWN`。

4. **计算耗时与准确率**  
   记录从开始处理到获得最终答案的总耗时，并与标准答案对比计算正确性。

## 自定义与扩展

- **更换 LLM**：替换 `OpenAIChat` 实例为其他兼容接口的客户端。
  
  Eval_qwen3_4B.py选择qwen3.5_4B模型进行评测

  Eval_qwen3_8B.py选择qwen3_8B模型进行评测

## 注意事项

- 请确保知识库路径可访问且包含有效的 Markdown 文件。
- LLM 服务必须兼容 OpenAI Chat Completion API 格式。




好的，我已经为你整理好测试结果，可以直接添加到 README 的相应章节中。以下是对两种配置的测试结果说明：

---

# 三、测试结果

在相同的 770 道选择题测试集上，分别使用两种不同的后端大模型配合 Sirchmunk RAG 管道进行评测。以下为详细统计结果。

### 配置一：Qwen3.5-4B + RAG

- **模型**：Qwen3.5-4B
- **结果文件**：`Qwen_sat_mc_output_thesis_v3_km_test_pred_with_reason_evidence_llm_time.jsonl`
- **统计摘要**：
  - 总样本数：770
  - 有效样本（含 is_correct 字段）：770
  - JSON 解析失败：0

| 指标 | 数值 |
|------|------|
| 预测正确数 (True) | 683 |
| 预测错误数 (False) | 87 |
| **准确率** | **88.70%** |

### 配置二：DeepSeek-Chat + RAG

- **模型**：DeepSeek-Chat 
- **结果文件**：`sat_mc_output_thesis_v3_km_test_pred_with_reason_evidence_llm_time.jsonl`
- **统计摘要**：
  - 总样本数：770
  - 有效样本（含 is_correct 字段）：770
  - JSON 解析失败：0

| 指标 | 数值 |
|------|------|
| 预测正确数 (True) | 667 |
| 预测错误数 (False) | 100 |
| 答案缺失/无效 (None) | 3 |
| **准确率** | **86.62%** |

### 结果分析

1. **准确率对比**：在完全相同的检索管道和知识库条件下，Qwen3.5-4B + RAG 取得了 **88.70%** 的准确率，略高于 DeepSeek-Chat 的 **86.62%**。两者均展现了 RAG 增强后在专业选择题上的良好表现。

2. **稳定性**：DeepSeek-Chat 有 3 条样本的 `is_correct` 为 `None` 出现异常，而 Qwen3.5-4B  配置下全部 770 条均成功提取到了有效预测。

3. **综合结论**：Sirchmunk RAG 框架能够显著提升中小规模模型在专业知识问答上的准确率。不同基座模型在指令遵循能力和推理效果上存在细微差异，实际应用时可根据场景需求（如响应速度、部署成本、准确率要求）选择合适的后端模型。