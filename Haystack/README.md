# 具身智能卫星模型专业知识库

## Haystack 简介

Haystack 是一个用于构建 **RAG（Retrieval-Augmented Generation，检索增强生成）** 应用的开源框架。  
它允许大模型在回答问题之前，**先从私有数据中检索相关内容**，再基于检索结果生成答案，从而提高回答的准确性和可控性。

---

## Data
路由专业系统配置数据：
```bash
/data/chengxi/Haystack/data/Command_Reference_400.jsonl
/data/chengxi/Haystack/data/rag_router_180.jsonl
```

## main.py

### 功能说明

`main.py` 用于**加载本地已计算好的文档向量（embeddings）**，并基于 Haystack 框架构建一个 **RAG 推理流程**，用于回答用户问题。

这是一个**只读的推理程序**：

- 不会重新计算 embedding  
- 仅在已有向量的基础上进行检索和生成  

---

### 程序功能

#### 1. 加载预计算的向量数据

- 从 `embeddings.pkl` 文件中读取文档
- 每个元素必须是 **已包含 embedding 的 `haystack.Document` 对象**

#### 2. 构建内存文档库

- 使用 `InMemoryDocumentStore`
- 将所有文档及其向量写入文档库

#### 3. 搭建 RAG 推理流水线

- **查询向量编码器（Text Embedder）**  
  将用户问题编码为向量
- **检索器（Retriever）**  
  基于向量相似度检索相关文档
- **提示词构建器（Prompt Builder）**  
  将检索结果整理为大模型输入
- **大模型生成器（LLM）**  
  生成最终回答

#### 4. 执行查询并输出结果

- 运行示例问题
- 打印大模型返回的答案

---

### 用法

```bash
python main.py
```

## 注意事项

- 可在代码中直接修改查询问题，例如：

```python
question = "How to display the uptime of the router system?"
```


## Incremental_Embedding_Updater.py

### 简介

`Incremental_Embedding_Updater.py` 是一个 **RAG 向量库的增量同步器**，  
用于将 **JSONL 文档集按条目生成 embedding**，并**安全地更新到同一个 `embeddings.pkl` 文件中**。

---

### 解决的问题

- 无需每次进行全量 embedding 重算  
- 文档仅修改一条时，只更新对应的一条向量  
- 文档删除后，可同步从向量库中删除  
- 支持 **多个数据集共存在同一个 `embeddings.pkl` 中，互不影响**

---

### 配置说明

```python
INPUT_JSONL = "/data/chengxi/Haystack/data/Command_Reference_400.jsonl"  # 设置源数据路径
DATASET = "command_reference"                                          # 设置数据集分区（非常重要，不可忽略）
EMB_PKL_PATH = "/data/chengxi/Haystack/data/embeddings.pkl"            # 设置 embedding 存储路径
```

#### 注意：
DATASET 用于区分不同来源的数据集，是增量更新和防止误删数据的关键配置项，请务必正确设置。