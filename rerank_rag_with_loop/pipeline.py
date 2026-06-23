import json
from config import RerankRAGConfig
from retriever import Retriever
from reranker import Reranker
from generator import Generator
from feedback import Feedback


class RerankRAGPipeline:
    def __init__(self, cfg: RerankRAGConfig):
        self.cfg = cfg
        self.retriever = Retriever(cfg.chunks_file, cfg.embed_model)
        self.reranker = Reranker(cfg.rerank_model, cfg.rerank_device, cfg.rerank_top_k)
        self.generator = Generator(cfg)
        self.feedback = Feedback(cfg)

    def answer(self, query: str) -> dict:
        # 1. 检索
        if self.cfg.debug:
            print(f"[pipeline] 检索: {query}")
        docs = self.retriever.retrieve(
            query,
            bm25_top=self.cfg.bm25_top_k,
            faiss_top=self.cfg.faiss_top_k,
            rrf_k=self.cfg.rrf_k,
            final_top=self.cfg.rrf_top_k,
        )

        # 2. 重排
        if self.cfg.debug:
            print(f"[pipeline] 重排: {len(docs)} → {self.cfg.rerank_top_k}")
        docs = self.reranker.rerank(query, docs)

        # 3. 生成 + feedback 循环
        loop_count = 0
        last_answer = ""
        last_feedback = ""

        while loop_count < self.cfg.max_loops:
            if self.cfg.debug:
                print(f"[pipeline] 生成 (loop {loop_count + 1})")
            raw = self.generator.generate(query, docs, last_answer, last_feedback)

            try:
                parsed = json.loads(raw)
                answer = parsed.get("answer", raw)
            except json.JSONDecodeError:
                answer = raw

            if loop_count == self.cfg.max_loops - 1:
                break

            if self.cfg.debug:
                print(f"[pipeline] 审查 (loop {loop_count + 1})")
            verdict = self.feedback.judge(query, docs, answer, last_feedback)

            if not verdict["should_generate_again"]:
                if self.cfg.debug:
                    print(f"[pipeline] 审查通过")
                break

            if self.cfg.debug:
                print(f"[pipeline] 需重生成，原因: {verdict['reason']}")
            last_answer = answer
            last_feedback = verdict["reason"]
            loop_count += 1

        return {
            "query": query,
            "answer": answer,
            "loop_count": loop_count,
            "documents": docs,
        }
