# RAGAgent 教学文档

> 写给 0 基础同学的完整学习路径。请从第一章开始，按顺序阅读，不要跳跃。

---

## 第一章：先搞懂"它是干什么的"

### 1.1 一个真实场景

你入职一家公司，HR 给你发了 200 页的员工手册 PDF。
你想知道："请假流程是什么？"——你不可能把 200 页都背下来。

**传统做法**：自己翻文档，Ctrl+F 搜关键词，费时费力。
**RAGAgent 的做法**：你直接用中文问，它秒回答，还告诉你答案来自哪一段原文。

这就是这个项目做的事情：**让机器帮你"读"文档，然后你用自然语言问它问题。**

---

### 1.2 RAG 是什么意思

RAG = **R**etrieval-**A**ugmented **G**eneration
中文：检索增强生成

三个词拆开理解：

| 词 | 意思 |
|----|------|
| **检索（Retrieval）** | 从你上传的文档里，找到跟问题最相关的片段 |
| **增强（Augmented）** | 把找到的片段"塞给"大模型，作为参考资料 |
| **生成（Generation）** | 大模型根据参考资料，用自然语言回答你的问题 |

**类比**：就像开卷考试。
- 闭卷 = 纯粹靠大模型的训练记忆（容易"幻觉"、说错话）
- 开卷 = RAG，大模型可以翻"课本"（你上传的文档）再回答，准确率大幅提升

---

### 1.3 整体流程一张图

```
你上传文档                             你提问
     ↓                                   ↓
[文档处理]                          [问题向量化]
切成小块 → 变成向量 → 存入向量库      问题 → 变成向量
                                         ↓
                                    [语义检索]
                                    在向量库里找最相似的片段
                                         ↓
                                    [组合Prompt]
                                    片段 + 问题 → 给大模型
                                         ↓
                                    [大模型回答]
                                    生成自然语言答案 → 返回给你
```

记住这张图，后面所有代码都是围绕这个流程展开的。

---

## 第二章：核心概念逐个击破

### 2.1 向量（Vector）是什么

这是整个系统最重要的基础概念。

**问题**：计算机怎么知道"请假流程"和"如何申请休假"是同一个意思？

**答案**：把文字变成一串数字（向量），语义相近的文字，数字也相近。

```
"如何申请休假"  → [0.12, 0.87, 0.33, 0.56, ...]  ← 1536个数字
"请假流程是什么" → [0.13, 0.85, 0.31, 0.58, ...]  ← 数字非常接近！
"今天天气怎么样" → [0.91, 0.02, 0.78, 0.11, ...]  ← 数字差距大
```

这个"文字 → 向量"的过程叫 **Embedding（向量化）**。
本项目使用 **通义 text-embedding-v3** 模型，每段文字会被转换成 1536 个数字。

**核心文件**：`embeddings/tongyi_embedder.py`

---

### 2.2 向量数据库（Qdrant）是什么

普通数据库存文字，向量数据库存"数字串"。

向量数据库的特殊能力：**给你一串数字，它能在几毫秒内找到最相似的几千万条数据。**

本项目使用 **Qdrant**，它的基本操作：

| 操作 | 含义 |
|------|------|
| `upsert` | 插入/更新向量和对应的文本 |
| `search` | 给一个向量，找最相似的 top-k 条 |
| `delete` | 删除指定向量 |
| `collection` | 类似数据库中的"表"，用于隔离不同知识库 |

**核心文件**：`vectorstore/qdrant_store.py`

---

### 2.3 大语言模型（LLM）在这里做什么

大模型（本项目用通义千问 `qwen-plus`）在 RAG 里只做最后一步：

> "我给你一些参考资料片段，再给你用户的问题，你用自然语言回答。"

它不需要"记住"你的文档，文档内容是每次实时"喂"给它的。

---

## 第三章：文档上传全流程（一步一步拆解）

### 3.1 第一步：接收文件

**入口**：`POST /documents/upload`
**代码**：`api/documents.py` → `upload_document()`

用户上传一个 PDF，系统先做三道校验：

```python
# 校验 1：文件大小（默认 50MB 上限）
if len(content) > MAX_UPLOAD_BYTES:
    raise HTTPException(413, "File too large")

# 校验 2：文件格式（只接受 pdf/docx/md/csv/txt）
if suffix not in SUPPORTED_EXTENSIONS:
    raise HTTPException(400, "Unsupported file type")

# 校验 3：MD5 去重（同一文件内容只入库一次）
file_md5 = compute_md5(content)
if store.md5_exists(file_md5, collection):
    raise HTTPException(409, "Duplicate file")
```

**什么是 MD5？**
MD5 是文件内容的"指纹"，同一个文件无论上传几次，MD5 都一样。
这样即使你重复上传同一份合同，系统不会建立重复数据。

---

### 3.2 第二步：后台异步处理

上传接口立即返回 `task_id`，不等处理完成：

```json
{"task_id": "abc-123", "status": "processing"}
```

为什么？因为一个 100 页 PDF 处理可能要几十秒，
如果接口一直等，用户会以为"超时了"、"请求失败了"。

异步处理的实现靠 FastAPI 的 `BackgroundTasks`：

```python
background_tasks.add_task(_process_document, ...)
return {"task_id": task_id, "status": "processing"}  # 立即返回
```

你可以轮询 `GET /documents/status/{task_id}` 查进度：
- `processing` → 处理中
- `done` → 完成
- `failed` → 失败（含错误信息）

---

### 3.3 第三步：DocumentPipeline（最核心的处理流程）

**代码**：`document_processing/pipeline.py`

```
PDF文件
  ↓
[Loader] 读取文件，提取文本
  ↓
[TextCleaner] 清洗文本（去除多余空格、乱码等）
  ↓
[SmartChunker] 将长文本切成小块（chunk）
  ↓
[注入元数据] 给每个 chunk 加上 file_md5、collection、created_at
  ↓
返回 chunks 列表
```

**为什么要"分块"？**

大模型有输入长度限制（context window）。
一份 100 页文档全塞进去是不可能的。
分块后，每次只把"最相关的几块"塞给大模型。

块的大小由配置控制：
```
chunk_size=1000    # 每块最多 1000 个字符
chunk_overlap=200  # 相邻块重叠 200 字符（防止答案被截断在块边界）
```

**chunk_overlap 为什么重要？**
```
原文：...员工提前3天提交申请，由直属主管审批，
      再由HR确认后生效...

如果恰好从中间切断：
  块A：...员工提前3天提交申请，由直属主管审批，
  块B：再由HR确认后生效...

有 overlap 后：
  块A：...员工提前3天提交申请，由直属主管审批，再由HR确认
  块B：由直属主管审批，再由HR确认后生效...  ← 完整语义保留
```

---

### 3.4 第四步：向量化 + 存入 Qdrant

```python
# 把每个 chunk 的文本转成向量
texts = [c.page_content for c in chunks]
vectors = embedder.embed_documents(texts)

# 存入 Qdrant
store.upsert(chunks, vectors, collection)
```

**存入 Qdrant 的数据结构**：

每个 chunk 存成一条记录：
```json
{
  "id": "随机UUID",
  "vector": [0.12, 0.87, "...1536个数字"],
  "payload": {
    "content": "请假流程：员工需提前3天...",
    "source": "员工手册.pdf",
    "file_md5": "abc123...",
    "collection": "default",
    "created_at": "2026-05-18T..."
  }
}
```

---

### 3.5 第五步：失败回滚

如果处理到一半失败（比如 Qdrant 写入一半断网），
系统会自动删除已写入的部分数据，避免"半截数据"污染知识库：

```python
except Exception as exc:
    store.delete_by_md5(file_md5, collection)  # 回滚！
    tasks[task_id]["status"] = "failed"
```

---

## 第四章：问答全流程（多轮对话）

### 4.1 整体流程

**入口**：`POST /chat`
**代码**：`chat/rag_chain.py` → `ChatEngine.chat()`

```
用户问题（带 session_id）
  ↓
[Step 1] 从 Redis 读取该 session 的历史对话
  ↓
[Step 2] 历史感知改写：
         "他的邮箱是多少？" → "张三的邮箱是多少？"（补全指代）
  ↓
[Step 3] 将改写后的问题向量化
  ↓
[Step 4] 在 Qdrant 中检索最相似的文档片段（top-k 条）
  ↓
[Step 5] 组合 Prompt：检索结果 + 历史 + 问题 → 通义千问
  ↓
[Step 6] 大模型生成答案 → 返回 answer + sources
  ↓
[后台并行]
  → 将本轮对话存入 Redis（下次对话用）
  → 异步写入 MongoDB（永久日志）
  → Q&A 向量化存入 Qdrant _longmem（长期记忆）
```

---

### 4.2 为什么需要"历史感知改写"

**多轮对话的难题**：

```
第1轮：用户问 "张三的邮箱是什么？"  → 可以直接搜索
第2轮：用户问 "他的电话呢？"         → "他"是谁？？
```

直接用"他的电话呢"去搜索向量库，几乎找不到任何结果。

**解决方案**：`create_history_aware_retriever`

先用 LLM 将问题改写：
```
输入：历史[张三的邮箱是什么/答案] + 当前问题[他的电话呢]
输出：张三的电话是什么
```

改写后再搜索，效果大幅提升。

---

### 4.3 三层记忆机制

这个项目实现了三种记忆，理解它们的区别非常重要：

| 记忆类型 | 存储位置 | 生命周期 | 用途 |
|----------|----------|----------|------|
| **会话记忆** | Redis | 7天TTL | 本次对话的上下文（历史感知用） |
| **对话日志** | MongoDB | 永久 | 审计、查看历史记录 |
| **长期记忆** | Qdrant `_longmem` | 永久 | 跨session的向量化知识沉淀 |

**类比**：
- 会话记忆 = 你的短期工作记忆（今天干了什么）
- 对话日志 = 你的日记本（可以翻查）
- 长期记忆 = 你总结提炼的知识笔记（可以被检索）

---

### 4.4 Prompt 工程（大模型如何被"指导"）

系统有两个关键 Prompt，理解它们能帮你更好地调整系统行为。

**Prompt 1：历史感知改写**（让问题独立化）

```
"根据对话历史，将用户问题改写为一个独立完整的问题
（保持原意，无需历史也能理解）。若已独立则直接返回原问题。"
```

**Prompt 2：知识库问答**（让答案精准）

```
"你是知识库问答助手。以下是从知识库中检索到的相关文档片段：
{context}
请根据上述文档内容，直接提取并回答用户的问题。
- 答案要简洁准确，直接给出具体信息（如姓名、邮箱、电话、日期等）
- 只有在文档中确实找不到相关信息时，才回复【知识库中未找到相关信息】"
```

注意最后一条：**不能乱编**，只能回答文档里有的内容，否则返回"未找到"。
这是 RAG 相比纯 LLM 的核心优势：**可控、可溯源、不幻觉**。

---

## 第五章：依赖注入与单例模式

### 5.1 为什么要用依赖注入

**代码**：`core/dependencies.py`

每次有请求进来，都要用到 Qdrant 客户端、Embedder、Retriever、ChatEngine。
如果每次请求都新建一个，会：
1. 重复连接 Qdrant（慢）
2. 重复加载模型配置（浪费）
3. 浪费内存

**解决方案**：`@lru_cache` 单例

```python
@lru_cache(maxsize=1)
def get_store() -> QdrantStore:
    return QdrantStore()  # 只创建一次，后续复用

@lru_cache(maxsize=1)
def get_embedder() -> TongyiEmbedder:
    return TongyiEmbedder()  # 只创建一次
```

FastAPI 通过 `Depends(get_store)` 将单例注入到每个接口函数：

```python
@router.post("/search")
async def search_documents(
    body: SearchRequest,
    retriever: Retriever = Depends(get_retriever),  # 自动注入
):
    ...
```

**好处**：接口函数只关心"我要用 retriever"，
不关心"retriever 怎么创建"——这叫**关注点分离**。

---

## 第六章：配置系统

### 6.1 Pydantic Settings 工作原理

**代码**：`core/config.py`

```python
class Settings(BaseSettings):
    dashscope_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    ...
    model_config = SettingsConfigDict(env_file=".env")
```

Pydantic Settings 的加载优先级（从高到低）：
```
系统环境变量（export QDRANT_URL=xxx）
    > .env 文件中的值
        > 代码中的默认值
```

**好处**：
- 开发时用 `.env` 文件（方便修改）
- 生产环境用系统环境变量（更安全，不会把密钥提交到 git）
- 代码里有合理默认值，不配置也能本地跑通

---

## 第七章：完整数据流串联

读完前面所有章节，现在把整个系统串起来看一遍完整的使用场景。

### 场景：上传员工手册 → 多轮对话问答

```
━━━━━━━━━━━━━━━━━ 阶段一：文档入库 ━━━━━━━━━━━━━━━━━

用户  POST /documents/upload (员工手册.pdf)
        ↓
  api/documents.py
    ├─ 校验大小、格式
    ├─ 计算 MD5，检查是否重复
    ├─ 写入临时文件
    └─ 注册 BackgroundTask，立即返回 task_id
        ↓ (后台执行)
  document_processing/pipeline.py
    ├─ PDFLoader → 提取文本
    ├─ TextCleaner → 清洗
    └─ SmartChunker → N 个 chunk
        ↓
  embeddings/tongyi_embedder.py
    └─ embed_documents([chunk1...N]) → N 个向量
        ↓
  vectorstore/qdrant_store.py
    └─ upsert → 写入 Qdrant collection "default"
        ↓
  task["status"] = "done"

━━━━━━━━━━━━━━━━━ 阶段二：多轮对话 ━━━━━━━━━━━━━━━━━

用户  POST /chat {session_id: "u001", query: "请假流程是什么？"}
        ↓
  api/chat.py → chat/rag_chain.py: ChatEngine.chat()
        ↓
  chat/session_store.py (Redis)
    └─ 读取 "u001" 历史 → 空（第一轮）
        ↓
  历史感知改写（LLM）
    └─ 问题已独立，无需改写，直接用原问题
        ↓
  embeddings/tongyi_embedder.py
    └─ embed_query("请假流程是什么？") → 向量
        ↓
  retrieval/retriever.py (Qdrant)
    └─ search(向量, top_k=5) → 5 个最相关文档片段
        ↓
  通义千问 LLM
    └─ [系统Prompt + 5个片段 + 问题] → 生成答案
        ↓
  返回 {answer: "员工需提前3天...", sources: [...]}
        ↓ (后台并行)
    ├─ Redis: 存入本轮对话记录
    ├─ MongoDB: 异步写入永久日志
    └─ Qdrant _longmem: 存入 Q&A 向量

用户  POST /chat {session_id: "u001", query: "需要谁来审批？"}
        ↓
  Redis: 读取 "u001" 历史 → [上一轮: 请假流程/答案]
        ↓
  历史感知改写（LLM）
    └─ "需要谁来审批？" + 历史 → "请假申请需要谁来审批？"
        ↓
  ... 后续同上，但这次能正确检索到审批相关内容
```

---

## 第八章：常见问题 & 调试指南

### Q1：问题回答不准确，总说"未找到相关信息"

**原因排查**：

1. `score_threshold` 过高（默认 0.7 比较严格）
   - 调用 chat 接口时传入 `"score_threshold": 0.3` 试试

2. 文档分块太大，关键信息被淹没
   - `.env` 里调小 `CHUNK_SIZE=500`，重新上传文档

3. 文档是图片型 PDF（扫描件，无文字层）
   - 症状：上传后 task 状态为 `failed`，错误 `"No extractable text found"`
   - 解法：需要 OCR 预处理（当前版本不支持）

### Q2：上传文档返回 409 Conflict

文件内容已存在（MD5 重复）。
更新文档需要两步：
```bash
# 先删除旧版本
curl -X DELETE "http://localhost:8000/documents/{file_md5}?collection=default"

# 再上传新版本
curl -X POST http://localhost:8000/documents/upload -F "file=@新版手册.pdf"
```

### Q3：多轮对话"忘了"上文

检查 Redis 是否正常：
```bash
docker ps | grep redis      # 确认容器在运行
redis-cli ping              # 应返回 PONG
```

### Q4：健康检查返回 503

```bash
curl http://localhost:8000/health
# → {"status": "error", "qdrant": "Connection refused"}
```
说明 Qdrant 未启动或地址配置错误，检查 `QDRANT_URL`。

---

## 第九章：学习路径建议

### 按顺序读代码（5天计划）

```
Day 1：理解数据结构
  core/config.py                    ← 所有配置项
  vectorstore/qdrant_store.py       ← 向量库增删查

Day 2：理解文档处理
  document_processing/pipeline.py   ← 整体流程
  document_processing/loaders/      ← 各格式解析
  document_processing/chunkers/     ← 分块策略

Day 3：理解检索链路
  embeddings/tongyi_embedder.py     ← 向量化
  retrieval/retriever.py            ← 语义检索
  api/documents.py                  ← 上传接口

Day 4：理解对话系统
  chat/session_store.py             ← Redis 会话记忆
  chat/mongo_logger.py              ← MongoDB 日志
  chat/long_term_memory.py          ← 长期记忆
  chat/rag_chain.py                 ← 核心对话引擎

Day 5：串联全流程
  main.py                           ← 应用入口
  api/chat.py                       ← 对话接口
  core/dependencies.py              ← 依赖注入
  动手跑通：上传一份文档 + 完整问答
```

---

## 总结

这个项目教会你的核心能力：

| 能力 | 对应模块 |
|------|----------|
| 文档解析与处理 | `document_processing/` |
| 向量化（Embedding）原理与实践 | `embeddings/` |
| 向量数据库使用（Qdrant） | `vectorstore/` |
| RAG 检索链路构建 | `retrieval/` + `chat/rag_chain.py` |
| 多轮对话上下文管理 | `chat/session_store.py` |
| LangChain 框架使用 | `chat/rag_chain.py` |
| FastAPI 异步服务设计 | `api/`, `main.py` |
| 生产级工程实践 | MD5去重、异步处理、失败回滚、依赖注入 |

---

> **最后一句话**：RAG 的本质是"给大模型提供上下文"。
> 所有的代码，都是在做一件事：**把正确的文档片段，在正确的时机，塞给大模型。**
> 理解了这句话，你就理解了整个项目。
