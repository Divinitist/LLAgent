"""
RAG Baseline — 入口脚本

用法:
  python run_baseline.py answer "你的问题"
  python run_baseline.py eval                     # 运行 queries.json 评估
  python run_baseline.py interact                  # 交互模式
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import RAGConfig
from pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG Baseline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("answer", help="单次问答")
    a.add_argument("query", type=str)

    sub.add_parser("eval", help="评估 queries.json")
    sub.add_parser("interact", help="交互模式")

    args = parser.parse_args()

    cfg = RAGConfig(chunks_file="first10_chunks.json")

    pipe = RAGPipeline(cfg)

    if args.cmd == "answer":
        res = pipe.answer(args.query, verbose=True)
        print(f"\n问题: {res['query']}")
        print(f"回答: {res['answer']}")
        print(f"来源章节: {sorted(set(d['chpt_id'] for d in res['documents']))}")

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
