from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cuda", top_k: int = 3):
        self.model = CrossEncoder(model_name, device=device)
        self.top_k = top_k

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        texts = [d["chunk_text"] for d in docs]
        pairs = [(query, t) for t in texts]
        scores = self.model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return docs[:self.top_k]
