# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """统一配置，所有值均可通过环境变量或 .env 文件覆盖。"""

    # Tongyi / DashScope
    dashscope_api_key: str = ""
    tongyi_embedding_model: str = "text-embedding-v3"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    default_collection: str = "default"

    # 上传限制
    max_upload_size_mb: int = 50

    # 检索参数
    score_threshold: float = 0.7
    top_k: int = 5

    # 向量维度（tongyi embedding model 实际维度，以 DashScope 文档为准）
    vector_size: int = 1536

    # 分块参数
    chunk_size: int = 1000
    chunk_overlap: int = 200
    # 扫描版 PDF 判断阈值：解析后总字符数低于此值视为扫描件
    min_text_length_for_scanned_detection: int = 50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
