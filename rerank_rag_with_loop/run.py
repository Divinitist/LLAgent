"""
rerank_rag_with_loop — 入口脚本

用法:
  python run.py answer "你的问题"
  python run.py eval
  python run.py interact
"""

import argparse, sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import RerankRAGConfig
from pipeline import RerankRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Rerank RAG with Loop")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("answer", help="单次问答")
    a.add_argument("query", type=str)

    sub.add_parser("eval", help="评估 queries.json")
    sub.add_parser("interact", help="交互模式")

    args = parser.parse_args()

    cfg = RerankRAGConfig(chunks_file="first10_chunks.json")
    pipe = RerankRAGPipeline(cfg)

    if args.cmd == "answer":
        cfg.debug = True
        res = pipe.answer(args.query)
        print(f"\n问题: {res['query']}")
        print(f"回答: {res['answer']}")
        print(f"循环次数: {res['loop_count']}")

    elif args.cmd == "eval":
        from evaluate import run_evaluation
        run_evaluation(cfg, "queries.json", "eval_result.json")

    elif args.cmd == "interact":
        print("输入问题 (Ctrl+C 退出)\n")
        try:
            while True:
                q = input("> ")
                if not q:
                    continue
                res = pipe.answer(q)
                print(f"回答: {res['answer']}\n")
        except KeyboardInterrupt:
            print("\n再见。")


if __name__ == "__main__":
    main()
