import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BACKEND   = os.getenv("BACKEND", "dense")  # "sparse" or "dense"
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/index.faiss"))
META_PATH  = Path(os.getenv("META_PATH",  "/mnt/project/atharva-dental-assistant/artifacts/rag/meta.json"))
MODEL_DIR  = os.getenv("MODEL_DIR")  # optional for dense
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title=f"Atharva Retriever ({BACKEND})")

class SearchRequest(BaseModel):
    query: str
    k: int = 4

_ready_reason = "starting"
_model = None; _index = None; _meta: List[dict] = []
_vec = None; _X = None  # sparse objects


# ------------------ Utils ------------------

def _normalize_meta_loaded(data: Any) -> List[dict]:
    """
    Accepts various shapes of meta.json and returns a list of entries.
    Supported:
      - list[dict]
      - {"items": [...]}  (common pattern)
      - {"hits": [...]}   (fallback)
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "hits" in data and isinstance(data["hits"], list):
            return data["hits"]
    raise ValueError("META_PATH must contain a list or a dict with 'items'/'hits'.")


def _parse_doc_and_section(path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Parse labels from meta.path:
      - 'treatments.json#0' -> ('treatments.json', '0')
      - 'faq.md'            -> ('faq.md', None)
      - 'policies/emergency.md' -> ('policies/emergency.md', None)
    """
    if not path:
        return "unknown", None
    if "#" in path:
        d, s = path.split("#", 1)
        return d, s
    return path, None


def _extract_text(m: dict) -> Optional[str]:
    """
    Try common keys for stored chunk text.
    """
    return m.get("text") or m.get("chunk") or m.get("content")


def _enrich_hit(idx: int, score: float) -> dict:
    """
    Build a single enriched hit from meta[idx].
    """
    if idx < 0 or idx >= len(_meta):
        # Guard against out-of-range
        doc_id, section, path, typ, txt = "unknown", None, None, None, None
    else:
        m   = _meta[idx] or {}
        path = m.get("path")
        typ  = m.get("type")
        doc_id, section = _parse_doc_and_section(path)
        txt = _extract_text(m)

    hit = {
        "score": float(score),
        "meta": {
            "doc_id": doc_id,
            "section": section,
            "path": path,
            "type": typ,
        },
    }
    if txt:
        hit["text"] = txt
    return hit


# ------------------ Loaders ------------------

def _load_dense():
    global _model, _index, _meta
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_DIR) if (MODEL_DIR and Path(MODEL_DIR).exists()) else SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = _normalize_meta_loaded(json.loads(META_PATH.read_text(encoding="utf-8")))
        return None
    except Exception as e:
        return f"dense load error: {e}"


def _load_sparse():
    global _vec, _X, _meta
    try:
        import joblib
        from scipy import sparse
        vec_p = Path(os.getenv("VEC_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/tfidf_vectorizer.joblib"))
        X_p   = Path(os.getenv("MAT_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/tfidf_matrix.npz"))
        _vec = joblib.load(vec_p)
        _X = sparse.load_npz(X_p)  # assume rows L2-normalized; dot == cosine
        _meta = _normalize_meta_loaded(json.loads(META_PATH.read_text(encoding="utf-8")))
        return None
    except Exception as e:
        return f"sparse load error: {e}"


@app.on_event("startup")
def startup():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()


# ------------------ Endpoints ------------------

@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ready")
def ready():
    return {"ready": _ready_reason is None, "reason": _ready_reason}


@app.post("/reload")
def reload_index():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()
    if _ready_reason is not None:
        raise HTTPException(status_code=503, detail=_ready_reason)
    return {"reloaded": True}


@app.post("/search")
def search(req: SearchRequest):
    if _ready_reason is not None:
        raise HTTPException(status_code=503, detail=_ready_reason)

    k = max(1, min(int(req.k), 20))

    if BACKEND == "sparse":
        import numpy as np
        q = _vec.transform([req.query])
        scores = (_X @ q.T).toarray().ravel()  # cosine since rows normalized
        if scores.size == 0:
            return {"hits": []}
        # get top-k indices by score desc
        k_eff = min(k, scores.size)
        top = np.argpartition(-scores, range(k_eff))[:k_eff]
        top = top[np.argsort(-scores[top])]
        hits = [
            _enrich_hit(int(i), float(scores[int(i)]))
            for i in top
            if scores[int(i)] > 0
        ]
        return {"hits": hits}

    # dense
    import faiss
    import numpy as np
    v = _model.encode([req.query], normalize_embeddings=True)  # IP ~ cosine
    D, I = _index.search(v.astype("float32"), k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        hits.append(_enrich_hit(int(idx), float(score)))
    return {"hits": hits}

