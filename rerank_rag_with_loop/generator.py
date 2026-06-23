from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import RerankRAGConfig
import json


class Generator:
    def __init__(self, cfg: RerankRAGConfig):
        if cfg.llm_provider == "ollama":
            self.llm = ChatOllama(
                model=cfg.llm_model,
                base_url=cfg.llm_base_url,
                temperature=cfg.llm_temperature,
                num_predict=cfg.llm_max_tokens,
                format="json",
            )
        else:
            self.llm = ChatOpenAI(
                model=cfg.llm_model,
                base_url=cfg.llm_base_url,
                api_key=cfg.llm_api_key,
                temperature=cfg.llm_temperature,
                max_tokens=cfg.llm_max_tokens,
                model_kwargs={"response_format": {"type": "json_object"}},
            )

    @staticmethod
    def fmt_docs(docs: list[dict]) -> str:
        return "\n\n".join(
            f"--- 第 {d['chpt_id']} 章 (chunk {d['chunk_id']}) ---\n{d['chunk_text']}"
            for d in docs
        )

    def generate(self, query: str, docs: list[dict],
                 last_answer: str = "", last_feedback: str = "") -> str:
        context = self.fmt_docs(docs)
        feedback_block = ""
        if last_feedback:
            feedback_block = (
                "\n【上一次生成的答案】:\n"
                f"{last_answer}\n"
                "【对上一次生成的答案的审查意见】:\n"
                f"{last_feedback}\n"
            )

        prompt = (
            "你是一个熟读玄幻网络小说的专家，能够根据提供的上下文信息回答用户的问题。"
            "请仔细阅读以下提供的上下文内容，然后根据用户的问题生成准确且简洁的回答。\n"
            "【上下文内容】:\n"
            f"{context}\n"
            "【用户问题】:\n"
            f"{query}\n"
            f"{feedback_block}"
            "【输出要求】:\n"
            "请直接输出 JSON 字符串，不得包含任何解释或 ```json 代码块。\n"
            '格式示例: {"answer": "阅读后产生的回答"}\n'
            "【给出回答】:\n"
        )
        msg = self.llm.invoke([HumanMessage(content=prompt)])
        return msg.content
