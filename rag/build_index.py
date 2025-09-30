import argparse, json, os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

def iter_docs(root: Path):
    # Gather text from policies, faq, treatments, recent
    # Each chunk = (text, meta)
    for md in (root/"policies").glob("*.md"):
        text = md.read_text(encoding="utf-8")
        yield text, {"doc_id": f"policies/{md.name}", "section": "full"}
    faq = (root/"faq.md")
    yield faq.read_text(encoding="utf-8"), {"doc_id": "faq.md", "section": "full"}
    tr = json.loads((root/"treatments.json").read_text(encoding="utf-8"))
    for t in tr:
        blob = json.dumps(t, ensure_ascii=False)
        yield blob, {"doc_id": "treatments.json", "section": t["code"]}
    recent = (root/"recent_queries.jsonl")
    for line in recent.read_text(encoding="utf-8").splitlines():
        obj = json.loads(line)
        yield f"Q:{obj['q']} A:{obj['a']}", {"doc_id": "recent_queries.jsonl", "section": obj["ts"]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="datasets/clinic")
    ap.add_argument("--outdir", required=True, help="artifacts/rag")
    args = ap.parse_args()
    root = Path(args.root); out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts, metas = [], []
    for txt, meta in iter_docs(root):
        texts.append(txt)
        metas.append(meta)

    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    faiss.write_index(index, str(out/"index.faiss"))
    (out/"meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(texts)} chunks → {out}")

if __name__ == "__main__":
    main()
