import os, json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BACKEND = os.getenv("BACKEND", "dense")  # "sparse" or "dense"
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/index.faiss"))
META_PATH  = Path(os.getenv("META_PATH",  "/mnt/project/atharva-dental-assistant/artifacts/rag/meta.json"))
MODEL_DIR  = os.getenv("MODEL_DIR")  # optional for dense
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title=f"Atharva Retriever ({BACKEND})")

class SearchRequest(BaseModel):
    query: str
    k: int = 4

_ready_reason = "starting"
_model = None; _index = None; _meta = None
_vec = None; _X = None  # sparse objects

def _load_dense():
    global _model, _index, _meta
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_DIR) if (MODEL_DIR and Path(MODEL_DIR).exists()) else SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(str(INDEX_PATH))
        _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
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
        _X = sparse.load_npz(X_p)  # L2-normalized rows; dot = cosine
        _meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return None
    except Exception as e:
        return f"sparse load error: {e}"

@app.on_event("startup")
def startup():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()

@app.get("/health")
def health(): return {"ok": True}

@app.get("/ready")
def ready(): return {"ready": _ready_reason is None, "reason": _ready_reason}

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

    if BACKEND == "sparse":
        import numpy as np
        q = _vec.transform([req.query])
        scores = (_X @ q.T).toarray().ravel()  # cosine since rows are normalized
        top = np.argpartition(-scores, range(min(req.k, scores.size)))[:req.k]
        top = top[np.argsort(-scores[top])]
        hits = [{"score": float(scores[i]), "meta": _meta[i]} for i in top if scores[i] > 0]
        return {"hits": hits}

    # dense
    v = _model.encode([req.query], normalize_embeddings=True)
    import faiss, numpy as np  # faiss used for .search if you kept it in memory
    D, I = _index.search(v.astype("float32"), req.k)
    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1: continue
        hits.append({"score": float(score), "meta": _meta[idx]})
    return {"hits": hits}
