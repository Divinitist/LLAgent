from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RAGConfig:
    # ── 数据路径 ──
    chunks_file: str = "first10_chunks.json"      # 预处理后的分块 JSON

    # ── 分块参数 ──
    chunk_size: int = 700
    chunk_overlap: int = 150
    max_chapters: int = 0                         # 0 = 全部

    # ── 检索参数 ──
    bm25_top_k: int = 30
    faiss_top_k: int = 30
    final_top_k: int = 5
    rrf_k: int = 60

    # ── 向量模型 ──
    embed_model: str = "BAAI/bge-small-zh-v1.5"

    # ── 生成模型 ──
    llm_provider: str = "openai"                  # ollama | openai
    llm_model: str = "doubao-seed-1-8-251228"
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_api_key: str = "2fc9822a-18f8-42e5-a9d6-3aea2849d4f6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_json_mode: bool = False

    # ── 评估 ──
    eval_llm_model: str = "doubao-seed-1-8-251228"

    # ── 运行时 ──
    device: str = "cuda"
    debug: bool = True
