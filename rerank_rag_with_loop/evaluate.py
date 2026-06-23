import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from config import RerankRAGConfig
from pipeline import RerankRAGPipeline


def run_evaluation(cfg: RerankRAGConfig, queries_path: str, output_path: str = None):
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    pipe = RerankRAGPipeline(cfg)

    results = []
    for q in queries:
        print(f"[eval] 处理: {q['query'][:40]}...")
        res = pipe.answer(q["query"])
        results.append({
            "query": res["query"],
            "answer": res["answer"],
            "reference": q.get("reference", ""),
            "loop_count": res["loop_count"],
            "retrieved_chapters": list({d["chpt_id"] for d in res["documents"]}),
        })

    ds = Dataset.from_list(results)
    eval_llm = LangchainLLMWrapper(ChatOllama(model=cfg.eval_llm_model))
    score = evaluate(ds, metrics=[faithfulness], llm=eval_llm)

    result_df = score.to_pandas()
    print("\n===== Faithfulness 评分 =====")
    print(result_df)
    print(f"\n均值: {result_df['faithfulness'].mean():.4f}")

    if output_path:
        out = {
            "avg_faithfulness": float(result_df["faithfulness"].mean()),
            "details": results,
            "scores": result_df.to_dict(orient="records"),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[eval] 结果 → {output_path}")

    return score
