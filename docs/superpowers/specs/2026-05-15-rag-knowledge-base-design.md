# RAG 知识库设计文档

**日期：** 2026-05-15
**项目：** RAGAgent
**状态：** 已审批

---

## 1. 目标

构建一个基于 LangChain + Qdrant 的 RAG 知识库文档处理系统，支持 PDF / Word / Markdown / CSV / Text 五种格式的解析、清洗、结构化切分与向量入库，通过 REST API 对外提供文档上传、去重检测和语义检索能力。首版不含知识图谱，聚焦文档处理流水线与向量检索。

---

## 2. 技术栈

| 组件 | 选型 |
|------|------|
| Web 框架 | FastAPI (Python 3.14+) |
| 文档处理框架 | LangChain (langchain-community) |
| LLM | 千问（Tongyi Qianwen） |
| Embedding | tongyi-embedding-vision-plus |
| 向量数据库 | Qdrant |
| PDF 解析 | pypdf / pymupdf |
| Word 解析 | python-docx |
| 编码检测 | chardet |

---

## 3. 项目结构

```
RAGAgent/
├── main.py                            # FastAPI 入口，注册路由
├── pyproject.toml
├── core/
│   ├── config.py                      # 统一配置（API Key、Qdrant URL、上传限制等）
│   └── dependencies.py               # FastAPI 依赖注入（Qdrant client 单例等）
├── api/
│   └── documents.py                  # 文档上传 / 状态查询 / 检索 / 删除路由
├── document_processing/
│   ├── loaders/
│   │   ├── base.py                   # BaseLoader 抽象接口
│   │   ├── pdf_loader.py             # PDF 加载（pypdf，检测扫描件）
│   │   ├── word_loader.py            # Word 加载（python-docx）
│   │   ├── markdown_loader.py        # Markdown 加载
│   │   ├── csv_loader.py             # CSV 加载（chardet 编码检测）
│   │   └── text_loader.py            # TXT 加载（chardet 编码检测）
│   ├── cleaners/
│   │   └── text_cleaner.py           # 去噪、规范化空白、过滤无效行
│   ├── chunkers/
│   │   └── smart_chunker.py          # RecursiveCharacterTextSplitter + 元数据保留
│   └── pipeline.py                   # 串联：loader → cleaner → chunker，返回 List[Document]
├── embeddings/
│   └── tongyi_embedder.py            # tongyi-embedding-vision-plus 封装，含重试逻辑
├── vectorstore/
│   └── qdrant_store.py               # Qdrant 增删查封装（MD5 过滤、upsert、delete by md5）
└── retrieval/
    └── retriever.py                  # 语义检索：query 向量化 → Qdrant search → 结果排序
```

---

## 4. API 接口

### 4.1 上传文档
```
POST /documents/upload
Content-Type: multipart/form-data

参数：
  file         必填  上传文件（支持 pdf / docx / md / csv / txt）
  collection   选填  目标 collection 名称，默认 "default"

响应 200：{ "task_id": "...", "status": "processing" }
响应 409：{ "detail": "duplicate", "doc_id": "..." }   # MD5 重复
响应 400：{ "detail": "unsupported file type" }
响应 413：{ "detail": "file too large, max 50MB" }
```

### 4.2 查询处理状态
```
GET /documents/status/{task_id}

响应：{ "task_id": "...", "status": "processing|done|failed", "error": null }
```

### 4.3 语义检索
```
POST /documents/search

Body：{
  "query": "查询内容",
  "collection": "default",
  "top_k": 5,
  "score_threshold": 0.7
}

响应：{
  "results": [
    { "content": "...", "source": "file.pdf", "page": 3, "score": 0.92, "metadata": {...} }
  ]
}
```

### 4.4 删除文档
```
DELETE /documents/{file_md5}?collection=default

响应 200：{ "deleted_chunks": 12 }
响应 404：{ "detail": "document not found" }
```

### 4.5 列出 Collections
```
GET /documents/collections

响应：{ "collections": ["default", "manual_v2"] }
```

---

## 5. 数据流

### 5.1 上传流程
```
上传文件（multipart）
  └─► 文件大小校验（> 50MB → 413）
        └─► MIME type + 后缀校验（不支持 → 400）
              └─► 计算文件 MD5
                    ├─► Qdrant 查询 file_md5 是否存在
                    │     └─► 存在 → 返回 409 duplicate
                    └─► 不存在 → 写入处理队列（BackgroundTasks）
                          └─► 返回 { task_id, status: "processing" }

后台处理（BackgroundTask）：
  format-specific Loader
    └─► TextCleaner（去噪、规范化）
          └─► SmartChunker（切分 + 元数据注入）
                └─► TongyiEmbedder（批量向量化，失败重试 3 次）
                      └─► Qdrant.upsert（含 file_md5 payload）
                            └─► 更新 task 状态 → "done" 或 "failed"
```

### 5.2 检索流程
```
query 字符串
  └─► TongyiEmbedder.embed_query
        └─► Qdrant.search(top_k, score_threshold, collection_filter)
              └─► 过滤低相似度结果
                    └─► 返回 List[{ content, source, score, metadata }]
```

---

## 6. Chunk 元数据结构（Qdrant payload）

```json
{
  "source": "业务手册.pdf",
  "file_md5": "d41d8cd98f00b204e9800998ecf8427e",
  "file_type": "pdf",
  "page": 3,
  "chunk_index": 12,
  "collection": "default",
  "created_at": "2026-05-15T10:00:00Z"
}
```

---

## 7. 边界处理

### 7.1 文件输入层

| 场景 | 处理方式 |
|------|---------|
| 文件 > 50MB | 返回 413，不进入处理流程 |
| 格式不支持 / 后缀伪造 | MIME type 二次校验，返回 400 |
| 文件损坏无法解析 | Loader 内 try/catch，BackgroundTask 中标记 task 为 "failed"，记录错误信息 |
| 扫描版 PDF（无文本层） | 解析后检测文本量，低于 50 字符时返回提示"需要 OCR 支持" |
| CSV / TXT 非 UTF-8 编码 | chardet 自动检测编码（支持 GBK / GB2312 / UTF-8），转换后处理 |

### 7.2 去重与更新层

| 场景 | 处理方式 |
|------|---------|
| 同文件并发上传 | MD5 检查 + Qdrant upsert 幂等，重复请求返回 409 |
| 同名不同内容（文件更新） | MD5 不同视为新版本，先删除旧 MD5 所有 chunks，再入库新版本 |

### 7.3 处理流程层

| 场景 | 处理方式 |
|------|---------|
| 解析后内容为空 | Pipeline 早检测，标记 task 失败，不进入 Embedder |
| Chunk 超出 Embedding token 限制 | SmartChunker 二次切分兜底（chunk_size 动态缩小至 512） |
| Tongyi Embedding API 限流 / 超时 | 指数退避重试（1s / 2s / 4s），3 次失败后标记 task 为 "failed" |
| 部分 chunk 入库失败 | 全量失败回滚：删除当前文件已写入的所有 chunks，保持库中数据一致 |

### 7.4 检索层

| 场景 | 处理方式 |
|------|---------|
| Collection 不存在 | 搜索前自动创建（auto-create），或返回 404 明确提示（可配置） |
| 检索相似度低于阈值 | score_threshold 默认 0.7，低于阈值返回空数组，不返回噪音结果 |
| Qdrant 服务不可用 | `/health` 端点启动时验证连接；请求时捕获异常返回 503 |

---

## 8. 依赖清单（待加入 pyproject.toml）

```
langchain
langchain-community
langchain-text-splitters
langchain-qdrant
qdrant-client
pypdf
pymupdf
python-docx
chardet
dashscope          # 千问 / Tongyi API SDK
python-multipart   # FastAPI 文件上传
```

---

## 9. Task 状态存储

首版使用 **FastAPI app.state 内存字典** 存储 task 状态：

```python
# 格式：{ task_id: { "status": "processing|done|failed", "error": None } }
app.state.tasks: dict[str, dict] = {}
```

重启后任务状态清空（可接受，首版范围内）。后续可替换为 Redis 或数据库持久化。

---

## 10. 范围说明

**首版包含：**
- 五种格式文档处理 Pipeline
- MD5 去重
- Qdrant 向量存储与检索
- REST API（上传 / 状态 / 检索 / 删除 / collections）
- 完整边界处理与错误响应

**首版不含（后续迭代）：**
- 知识图谱关联
- OCR 支持
- 多租户 / 权限控制
- LangGraph Agent 集成
