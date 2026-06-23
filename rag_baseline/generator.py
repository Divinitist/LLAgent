from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from config import RAGConfig


class BaselineGenerator:
    def __init__(self, cfg: RAGConfig):
        if cfg.llm_provider == "ollama":
            self.llm = ChatOllama(
                model=cfg.llm_model,
                base_url=cfg.llm_base_url,
                temperature=cfg.llm_temperature,
                num_predict=cfg.llm_max_tokens,
            )
        else:
            self.llm = ChatOpenAI(
                model=cfg.llm_model,
                base_url=cfg.llm_base_url,
                api_key=cfg.llm_api_key,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
            )

    @staticmethod
    def _fmt_docs(docs: list[dict]) -> str:
        return "\n\n".join(
            f"--- 第 {d['chpt_id']} 章 (chunk {d['chunk_id']}) ---\n{d['chunk_text']}"
            for d in docs
        )

    def generate(self, query: str, docs: list[dict]) -> str:
        context = self._fmt_docs(docs)
        prompt = (
            "你是一个熟读玄幻小说的专家。请根据以下参考资料回答用户问题。\n"
            "如果参考资料不足以回答问题，请如实说不知道，不要编造。\n\n"
            f"【参考资料】\n{context}\n\n"
            f"【问题】\n{query}\n\n"
            "【回答】"
        )
        msg = self.llm.invoke([HumanMessage(content=prompt)])
        return msg.content
