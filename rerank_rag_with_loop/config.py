from dataclasses import dataclass
from typing import Optional


@dataclass
class RerankRAGConfig:
    # ── 数据路径 ──
    chunks_file: str = "first10_chunks.json"

    # ── 检索参数 ──
    bm25_top_k: int = 30
    faiss_top_k: int = 30
    rrf_top_k: int = 20
    rrf_k: int = 60

    # ── 重排参数 ──
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = 3
    rerank_device: str = "cuda"

    # ── 向量模型 ──
    embed_model: str = "BAAI/bge-small-zh-v1.5"

    # ── 生成模型 ──
    llm_provider: str = "openai"
    llm_model: str = "doubao-seed-1-8-251228"
    llm_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    llm_api_key: str = "2fc9822a-18f8-42e5-a9d6-3aea2849d4f6"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096

    # ── 反馈循环 ──
    max_loops: int = 3

    # ── 评估 ──
    eval_llm_model: str = "doubao-seed-1-8-251228"

    # ── 运行时 ──
    debug: bool = True
