import json, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RAGConfig
from pipeline import RAGPipeline


def run_benchmark(cfg: RAGConfig, queries_path: str, output_path: str):
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    pipe = RAGPipeline(cfg)
    results = []

    for i, q in enumerate(queries):
        print(f"[{i+1}/{len(queries)}] {q['query'][:40]}...")
        t0 = time.time()
        try:
            res = pipe.answer(q["query"])
        except Exception as e:
            res = {"query": q["query"], "answer": f"ERROR: {e}", "documents": []}
        elapsed = time.time() - t0

        results.append({
            "difficulty": q["difficulty"],
            "query": res["query"],
            "answer": res["answer"],
            "reference": q.get("reference", ""),
            "elapsed": round(elapsed, 2),
            "retrieved_chunks": [(d["chpt_id"], d["chunk_id"]) for d in res["documents"]],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n完成 {len(results)} 条 → {output_path}")


if __name__ == "__main__":
    cfg = RAGConfig(chunks_file="first10_chunks.json")
    run_benchmark(cfg, "queries.json", "benchmark_results.json")
