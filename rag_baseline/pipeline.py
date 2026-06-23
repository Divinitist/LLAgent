from config import RAGConfig
from retriever import BaselineRetriever
from generator import BaselineGenerator


class RAGPipeline:
    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.retriever = BaselineRetriever(
            chunks_file=cfg.chunks_file,
            embed_model=cfg.embed_model,
            top_k=cfg.final_top_k,
        )
        self.generator = BaselineGenerator(cfg)

    def answer(self, query: str, verbose: bool = False) -> dict:
        docs = self.retriever.retrieve(
            query,
            bm25_top=self.cfg.bm25_top_k,
            faiss_top=self.cfg.faiss_top_k,
            rrf_k=self.cfg.rrf_k,
            final_top=self.cfg.final_top_k,
        )
        if verbose:
            print(f"[pipeline] 检索到 {len(docs)} 个 chunk:")
            for d in docs:
                txt = d["chunk_text"][:80].replace("\n", " ")
                print(f"  第{d['chpt_id']}章 chunk{d['chunk_id']} (score={d['score']:.4f}) | {txt}...")
        answer = self.generator.generate(query, docs)
        return {"query": query, "answer": answer, "documents": docs}
