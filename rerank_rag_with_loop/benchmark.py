import json, sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RerankRAGConfig
from pipeline import RerankRAGPipeline


def load_existing_results(output_path: str) -> list[dict]:
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError) as e:
        backup_path = f"{output_path}.broken.{int(time.time())}"
        os.replace(output_path, backup_path)
        print(f"[resume] 结果文件损坏，已备份为 {backup_path}: {e}")
    return []


def save_results(output_path: str, results: list[dict]):
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, output_path)


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "too many requests" in text


def ordered_results(queries: list[dict], result_by_query: dict[str, dict]) -> list[dict]:
    ordered = [result_by_query[q["query"]] for q in queries if q["query"] in result_by_query]
    extra = [r for key, r in result_by_query.items() if key not in {q["query"] for q in queries}]
    return ordered + extra


def run_benchmark(cfg: RerankRAGConfig, queries_path: str, output_path: str):
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    cfg.debug = False
    pipe = RerankRAGPipeline(cfg)
    results = load_existing_results(output_path)
    result_by_query = {r.get("query"): r for r in results}
    completed = sum(
        1
        for q in queries
        if q["query"] in result_by_query
        and not str(result_by_query[q["query"]].get("answer", "")).startswith("ERROR:")
    )
    if completed:
        print(f"[resume] 已完成 {completed}/{len(queries)} 条，将跳过成功项。")

    for i, q in enumerate(queries):
        existing = result_by_query.get(q["query"])
        if existing and not str(existing.get("answer", "")).startswith("ERROR:"):
            print(f"[{i+1}/{len(queries)}] 跳过: {q['query'][:40]}...")
            continue

        print(f"[{i+1}/{len(queries)}] {q['query'][:40]}...")
        t0 = time.time()
        try:
            res = pipe.answer(q["query"])
        except Exception as e:
            if is_rate_limit_error(e):
                results = ordered_results(queries, result_by_query)
                save_results(output_path, results)
                print(f"[rate-limit] 遇到限流，已保存 {len(results)} 条，停止运行: {e}")
                break
            res = {"query": q["query"], "answer": f"ERROR: {e}", "documents": [], "loop_count": 0}
        elapsed = time.time() - t0

        row = {
            "difficulty": q["difficulty"],
            "query": res["query"],
            "answer": res["answer"],
            "reference": q.get("reference", ""),
            "elapsed": round(elapsed, 2),
            "loop_count": res.get("loop_count", 0),
            "retrieved_chunks": [(d["chpt_id"], d["chunk_id"]) for d in res["documents"]],
        }

        if existing:
            idx = results.index(existing)
            results[idx] = row
        else:
            results.append(row)
        result_by_query[q["query"]] = row
        results = ordered_results(queries, result_by_query)
        save_results(output_path, results)
        print(f"[save] 已保存 {len(results)} 条 → {output_path}")

    save_results(output_path, results)
    print(f"\n完成 {len(results)} 条 → {output_path}")


if __name__ == "__main__":
    cfg = RerankRAGConfig(chunks_file="first10_chunks.json")
    run_benchmark(cfg, "queries.json", "benchmark_results.json")
