# RAGAgent

基于 **FastAPI + LangChain + Qdrant + 通义千问** 构建的企业级 RAG（检索增强生成）知识库 API 服务，支持多格式文档入库、语义检索、多轮对话问答，并集成 Redis 会话记忆、MongoDB 对话日志与 Qdrant 长期向量记忆。

---

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI（异步，BackgroundTasks） |
| AI / 链路 | LangChain（RAG 链、历史感知检索器） |
| 大语言模型 | 通义千问 `qwen-plus`（DashScope） |
| 向量化模型 | 通义 `text-embedding-v3`（1536 维） |
| 向量数据库 | Qdrant |
| 会话记忆 | Redis（RedisChatMessageHistory，TTL 7 天） |
| 对话日志 | MongoDB（motor 异步驱动） |
| 配置管理 | Pydantic Settings（`.env` 文件） |

---

## 项目结构

```
RAGAgent/
├── main.py                     # 应用入口，FastAPI app
├── core/
│   ├── config.py               # 全局配置（Pydantic Settings）
│   └── dependencies.py         # 依赖注入（@lru_cache 单例）
├── api/
│   ├── documents.py            # 文档管理 API
│   └── chat.py                 # 多轮对话 API
├── document_processing/        # 文档处理流水线
│   ├── pipeline.py             # 串联 loader → cleaner → chunker
│   ├── loaders/                # PDF / Word / Markdown / CSV / TXT 加载器
│   ├── cleaners/               # 文本清洗
│   └── chunkers/               # 智能分块
├── embeddings/
│   └── tongyi_embedder.py      # 通义向量化封装
├── vectorstore/
│   └── qdrant_store.py         # Qdrant 增删查封装
├── retrieval/
│   └── retriever.py            # 语义检索封装
└── chat/
    ├── rag_chain.py            # 多轮 RAG 对话引擎（ChatEngine）
    ├── session_store.py        # Redis 会话历史
    ├── mongo_logger.py         # MongoDB 对话日志
    └── long_term_memory.py     # Qdrant 长期向量记忆
```

---

## 核心功能

### 1. 文档处理流水线

文档上传后经过三阶段处理：

```
上传文件
  → Loader（按格式分发）
  → TextCleaner（清洗噪声文本）
  → SmartChunker（按 chunk_size / chunk_overlap 分块）
  → 注入元数据（file_md5、collection、created_at）
  → TongyiEmbedder（向量化）
  → QdrantStore（持久化）
```

- 支持格式：`PDF`、`DOCX`、`Markdown`、`CSV`、`TXT`
- 文件大小上限：50 MB（可配置）
- **MD5 去重**：同一文件内容不会重复入库，返回 `409 Duplicate`
- **并发防竞态**：`in_flight_md5s` 集合防止同一文件同时被多个请求处理
- **异步后台处理**：上传接口立即返回 `task_id`，处理在后台线程执行
- **失败回滚**：处理异常时自动删除已写入的部分 chunks

### 2. 语义检索

基于 Qdrant 向量相似度，支持：
- 指定 `collection`（多知识库隔离）
- `top_k` 返回数量
- `score_threshold` 相似度过滤

### 3. 多轮 RAG 对话

`ChatEngine` 整合四层能力：

```
用户问题
  → [历史感知] 根据对话历史将问题改写为独立完整问题
  → [语义检索] 在 Qdrant 知识库中检索相关文档片段
  → [生成回答] 通义千问基于检索内容回答，返回 answer + sources
  → [会话记忆] Redis 存储本次对话轮次（TTL 7 天）
  → [持久日志] MongoDB 异步写入永久对话记录
  → [长期记忆] 将本轮 Q&A 向量化存入 Qdrant _longmem collection
```

**历史感知检索器**：使用 LangChain `create_history_aware_retriever`，先将带历史上下文的问题改写为独立问题，再执行向量检索，避免多轮对话中指代不清导致的检索偏差。

---

## API 接口

### 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/documents/upload` | 上传文档，异步处理入库 |
| `GET` | `/documents/status/{task_id}` | 查询处理任务状态 |
| `POST` | `/documents/search` | 语义检索 |
| `DELETE` | `/documents/{file_md5}` | 按 MD5 删除文档所有 chunks |
| `GET` | `/documents/collections` | 列出所有 collection |

### 多轮对话

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/chat` | 发起多轮问答 |
| `GET` | `/chat/{session_id}/history` | 查看 Redis 会话历史 |
| `GET` | `/chat/{session_id}/logs` | 查看 MongoDB 永久日志 |
| `DELETE` | `/chat/{session_id}` | 清空会话历史 |

### 系统

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查（验证 Qdrant 连接） |
| `GET` | `/` | 服务状态 |

#### 上传文档示例

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@manual.pdf" \
  -F "collection=default"
# → {"task_id": "xxx", "status": "processing"}

curl http://localhost:8000/documents/status/xxx
# → {"task_id": "xxx", "status": "done"}
```

#### 多轮对话示例

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user-001",
    "query": "文档中提到的联系邮箱是什么？",
    "collection": "default",
    "top_k": 5,
    "score_threshold": 0.3
  }'
# → {"session_id": "user-001", "answer": "...", "sources": [...]}
```

---

## 配置说明

复制 `.env.example` 为 `.env` 并填写：

```env
# 通义 / DashScope
DASHSCOPE_API_KEY=sk-xxx
TONGYI_EMBEDDING_MODEL=text-embedding-v3
TONGYI_LLM_MODEL=qwen-plus

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=              # 云端部署时填写

# Redis（会话记忆）
REDIS_URL=redis://localhost:6379

# MongoDB（对话日志）
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=ragagent

# 检索参数
SCORE_THRESHOLD=0.7
TOP_K=5
VECTOR_SIZE=1536

# 文档处理
MAX_UPLOAD_SIZE_MB=50
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

---

## 快速启动

```bash
# 1. 安装依赖（推荐 uv）
uv sync

# 2. 启动依赖服务
docker run -p 6333:6333 qdrant/qdrant
docker run -p 6379:6379 redis
docker run -p 27017:27017 mongo

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY

# 4. 启动服务
uvicorn main:app --reload --port 8000

# 5. 访问交互文档
open http://localhost:8000/docs
```
