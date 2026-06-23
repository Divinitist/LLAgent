import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bm25 import BM25Retriever
from vdb import FaissRetriever
from rrf import rrf_fusion


class Retriever:
    def __init__(self, chunks_file: str, embed_model: str):
        self.bm25 = BM25Retriever(chunks_file=chunks_file)
        self.faiss = FaissRetriever(chunks_file=chunks_file, model_name=embed_model)

    def retrieve(self, query: str, bm25_top=30, faiss_top=30,
                 rrf_k=60, final_top=20) -> list[dict]:
        bm25_res = self.bm25.retrieve(query, top_k=bm25_top)
        faiss_res = self.faiss.retrieve(query, top_k=faiss_top)
        return rrf_fusion([bm25_res, faiss_res], k=rrf_k, top_k=final_top)
