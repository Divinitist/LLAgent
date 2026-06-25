from openai import OpenAI
from config import RerankRAGConfig
from generator import Generator
import json


class Feedback:
    def __init__(self, cfg: RerankRAGConfig):
        if cfg.llm_provider != "openai":
            raise ValueError("Only OpenAI-compatible providers are supported.")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    def judge(self, query: str, docs: list[dict],
              answer: str, last_feedback: str = "") -> dict:
        context = Generator.fmt_docs(docs)
        prompt = (
            "你是一个严谨的文档审查员。\n"
            "你的任务是核对【模型回答】是否忠实于【参考资料】。\n"
            "【判断准则】:\n"
            "1. 【严防幻觉】：如果回答中包含参考资料中明确没有、或与资料内容直接矛盾的信息，"
            '设 should_generate_again 为 true。\n'
            "2. 【核心缺失】：仅当回答漏掉了解决用户问题的“关键核心信息”时"
            "（例如问“谁治的”却没提医师），才设 should_generate_again 为 true。"
            "不要因为漏掉背景环境描写（如香炉、地毯）而要求重发。\n"
            "3. 【容忍冗余】：如果回答已经准确回答了问题，即便语言不够优美或包含少量无关但正确的废话，"
            "只要不违背事实，应设 should_generate_again 为 false。\n"
            "4. 【一致性优先】：如果本次回答已经根据之前的“审查意见”进行了修正，"
            "且修正后的内容符合参考资料，应设 should_generate_again 为 false，"
            "避免陷入无限循环。\n"
            "【参考资料】:\n"
            f"{context}\n"
            "【模型回答】:\n"
            f"{answer}\n"
            "【输出要求】：\n"
            "请直接输出 JSON 字符串，不得包含任何解释或 ```json 代码块。\n"
            '格式示例：{"should_generate_again": false, "reason": "回答准确覆盖了关键情节"}\n'
        )
        resp = self.client.chat.completions.create(
            model=self.cfg.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.llm_temperature,
            max_tokens=self.cfg.llm_max_tokens,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        try:
            result = json.loads(raw)
            return {
                "should_generate_again": bool(result.get("should_generate_again", False)),
                "reason": str(result.get("reason", "")),
            }
        except (json.JSONDecodeError, KeyError):
            return {
                "should_generate_again": False,
                "reason": f"反馈解析失败，默认通过: {raw[:100]}",
            }
