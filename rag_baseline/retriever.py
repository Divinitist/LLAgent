import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bm25 import BM25Retriever
from vdb import FaissRetriever
from rrf import rrf_fusion


class BaselineRetriever:
    def __init__(self, chunks_file: str, embed_model: str, top_k: int = 20):
        self.bm25 = BM25Retriever(chunks_file=chunks_file)
        self.faiss = FaissRetriever(chunks_file=chunks_file, model_name=embed_model)
        self.top_k = top_k

    def retrieve(self, query: str, bm25_top: int = 30,
                 faiss_top: int = 30, rrf_k: int = 60,
                 final_top: int = 5) -> list[dict]:
        bm25_res = self.bm25.retrieve(query, top_k=bm25_top)
        faiss_res = self.faiss.retrieve(query, top_k=faiss_top)
        fused = rrf_fusion([bm25_res, faiss_res], k=rrf_k, top_k=final_top)
        return fused
