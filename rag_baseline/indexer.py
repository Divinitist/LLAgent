import json
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import RAGConfig


def parse_chapters(raw_text: str) -> list[dict]:
    parts = raw_text.split("\n\n")
    chapters = []
    cnt = 0
    for para in parts:
        m = re.match(rf"第{cnt + 1}章", para.strip())
        if m:
            cnt += 1
            title = para.strip()[len(m.group(0)):].strip()
            chapters.append({"id": cnt, "title": title, "lines": []})
        elif cnt > 0:
            chapters[cnt - 1]["lines"].append(para.strip())
    return chapters


def build_chunks(cfg: RAGConfig) -> list[dict]:
    if cfg.chunks_file and os.path.exists(cfg.chunks_file):
        with open(cfg.chunks_file, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(cfg.raw_book_path, "r", encoding="utf-8") as f:
        raw = f.read()

    chapters = parse_chapters(raw)
    if cfg.max_chapters > 0:
        chapters = chapters[:cfg.max_chapters]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " "],
        keep_separator=True,
    )

    chunks = []
    for ch in chapters:
        text = "\n".join(ch["lines"])
        pieces = splitter.split_text(text)
        for j, piece in enumerate(pieces):
            chunks.append({
                "chpt_id": ch["id"],
                "title": ch["title"],
                "chunk_id": j,
                "chunk_text": piece,
            })

    os.makedirs(cfg.chunks_dir, exist_ok=True)
    out_path = os.path.join(cfg.chunks_dir, "chunks.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"[indexer] 共 {len(chunks)} 个 chunk → {out_path}")
    return chunks
