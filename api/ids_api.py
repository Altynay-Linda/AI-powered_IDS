
# ids_api.py — secure batch IDS inference with latency logging
import os
import json
import time
import traceback
import sqlite3
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, RootModel
from sklearn.compose import ColumnTransformer
from typing import List, Dict, Any, Optional

# -------------------------
# Configuration
# -------------------------

MODEL_PATH   = os.getenv("MODEL_PATH", "/app/ids_best_model.pkl")
THRESH_PATH  = os.getenv("THRESHOLD_PATH", "/app/threshold.json")
LOG_PATH     = os.environ.get("IDS_LOG_PATH",     "/data/logs/ids_api.log")
DB_PATH      = os.getenv("IDS_DB_PATH", "/data/ids_events.db")  # if you use SQLite
LOG_PATH = os.getenv("LOG_PATH", "/data/ids_metrics_log.jsonl")

API_KEY = os.getenv("IDS_API_KEY", "my-strong-key")       # set before running!
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# simple in-memory rate limiter (per IP)
RATE_LIMIT_WINDOW = 60.0  # seconds
RATE_LIMIT_COUNT = 60     # max requests per window
_client_hits: Dict[str, deque] = {}

# columns your pipeline expects (best-effort inference;
# if the trained pipeline exposes .feature_names_in_)
EXPECTED_CATEGORICALS = {"proto", "service", "state", "attack_cat"}  # safe defaults as strings

app = FastAPI(title="AI-Powered IDS API", version="1.0.0")

# constants + DB helpers
DB_PATH = Path.home() / "events.db"

def _db_init():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        count INTEGER NOT NULL,
        prediction INTEGER NOT NULL,
        prob_attack REAL NOT NULL,
        prob_normal REAL NOT NULL,
        threshold REAL NOT NULL,
        features_json TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def _db_insert_many(ts_iso: str, threshold: float, rows: list[dict], results: list[dict]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for r_in, r_out in zip(rows, results):
        cur.execute("""
        INSERT INTO predictions (ts, count, prediction, prob_attack, prob_normal, threshold, features_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ts_iso, 1, int(r_out["prediction"]), float(r_out["probability_attack"]),
              float(r_out["probability_normal"]), float(threshold), json.dumps(r_in)))
    conn.commit()
    conn.close()


# -------------------------
# Security & rate limiting
# -------------------------
def require_api_key(api_key: Optional[str] = Depends(api_key_header)) -> bool:
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

def enforce_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    dq = _client_hits.setdefault(client_ip, deque())
    # drop old timestamps
    while dq and now - dq[0] > RATE_LIMIT_WINDOW:
        dq.popleft()
    if len(dq) >= RATE_LIMIT_COUNT:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    dq.append(now)

# -------------------------
# Models (Pydantic v2)
# -------------------------
from pydantic import BaseModel

class PredictResult(BaseModel):
    prediction: int
    label: str
    probability_attack: float
    probability_normal: float

class PredictBatchResponse(BaseModel):
    predictions: List[PredictResult]
    count: int
    alert: Optional[str] = None   # <— ADD THIS

class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)

# Batch payload is a list of row dicts: [{col: value, ...}, ...]
class Batch(RootModel[List[Dict[str, Any]]]):
    pass

# -------------------------
# Globals: model + threshold
# -------------------------
_model = None  # sklearn Pipeline
_threshold: float = 0.5

def _load_model() -> None:
    global _model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)

def _load_threshold() -> float:
    if os.path.exists(THRESH_PATH):
        try:
            with open(THRESH_PATH, "r") as f:
                data = json.load(f)
                return float(data.get("threshold", 0.5))
        except Exception:
            return 0.5
    return 0.5

def _save_threshold(th: float) -> None:
    os.makedirs(os.path.dirname(THRESH_PATH), exist_ok=True)
    with open(THRESH_PATH, "w") as f:
        json.dump({"threshold": float(th)}, f)

def _expected_input_features() -> Optional[List[str]]:
    """
    Try to recover the feature names the pipeline was fit on.
    Works if the final pipeline was fit with a pandas DataFrame (sklearn >=1.0).
    """
    if _model is not None and hasattr(_model, "feature_names_in_"):
        return list(_model.feature_names_in_)
    return None

def _log_prediction(*, count: int, attacks: int, normals: int, latency_ms: float) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    rec = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": int(count),
        "attacks": int(attacks),
        "normals": int(normals),
        "latency_ms": float(latency_ms),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

# -------------------------
# Inference core
# -------------------------
def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has the expected columns. Fill missing numeric with 0, categoricals with '-'.
    Drop extra columns if the model recorded feature_names_in_.
    """
    expected = _expected_input_features()
    if expected is None:
        # No feature_names_in_. Use df as-is.
        X = df.copy()
    else:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]

        if missing:
            # create missing with safe defaults
            for c in missing:
                if c in EXPECTED_CATEGORICALS:
                    df[c] = "-"
                else:
                    df[c] = 0
        # drop extras so ColumnTransformer alignment is correct
        if extra:
            df = df.drop(columns=extra, errors="ignore")
        # reorder to expected order
        X = df[expected].copy()

    # dtype hygiene: object for categoricals, numeric for the rest
    for c in X.columns:
        if c in EXPECTED_CATEGORICALS:
            X[c] = X[c].astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X

def _predict_core(df_in: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns list of dicts with prediction + probabilities.
    Uses global _model and _threshold.
    """
    if _model is None:
        raise RuntimeError("Model is not loaded.")

    X = _prepare_dataframe(df_in)

    # predict_proba -> [:, 1] assumed to be "attack"
    probs = _model.predict_proba(X)
    if probs.shape[1] == 1:
        # some models output only positive class prob
        p_attack = probs[:, 0]
        p_normal = 1.0 - p_attack
    else:
        p_attack = probs[:, 1]
        p_normal = probs[:, 0]

    preds = (p_attack >= _threshold).astype(int)

    out: List[Dict[str, Any]] = []
    for pa, pn, yhat in zip(p_attack, p_normal, preds):
        out.append({
            "prediction": int(yhat),
            "label": "Attack" if int(yhat) == 1 else "Normal",
            "probability_attack": float(pa),
            "probability_normal": float(pn),
        })
    return out


def _get_preprocessor_and_clf(pipeline):
    """Return (preprocessor, classifier) from a sklearn Pipeline, robustly."""
    pre = None
    clf = None

    # Prefer named_steps (it preserves the original names)
    if hasattr(pipeline, "named_steps"):
        # try common names first
        pre = (
            pipeline.named_steps.get("preprocessor")
            or pipeline.named_steps.get("prep")
            or pipeline.named_steps.get("pre")   # 👈 add support for 'pre'
        )
        clf = (
            pipeline.named_steps.get("clf")
            or pipeline.named_steps.get("classifier")
            or pipeline.named_steps.get("model")
        )
        # if we still didn't find a preprocessor, scan for any ColumnTransformer
        if pre is None:
            for name, step in pipeline.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    pre = step
                    break

    # Fallbacks if not found via named_steps
    if (pre is None or clf is None) and hasattr(pipeline, "steps"):
        # last step is almost always the classifier
        if clf is None and pipeline.steps:
            clf = pipeline.steps[-1][1]
        if pre is None:
            for name, step in pipeline.steps:
                if isinstance(step, ColumnTransformer):
                    pre = step
                    break

    return pre, clf


def _get_transformed_and_names(preprocessor, X: pd.DataFrame):
    """Transform X and return (ndarray/array-like, feature_names)."""
    Xt = preprocessor.transform(X)
    feat_names = None
    # sklearn >=1.0 ColumnTransformer can expose output feature names
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            feat_names = list(preprocessor.get_feature_names_out())
        except Exception:
            pass
    # fallback
    if feat_names is None:
        feat_names = [f"f{i}" for i in range(np.asarray(Xt).shape[1])]
    return Xt, feat_names

def _log_event(kind: str, payload: Dict[str, Any]) -> None:
    """Very small logger; safe no-op if print/json fails."""
    try:
        print(json.dumps({"ts": time.time(), "kind": kind, **payload}))
    except Exception:
        pass

def _maybe_alert(attacks: int, total: int, max_prob: float) -> Optional[str]:
    """
    Return an alert message if batch looks suspicious; otherwise None.
    Tweak thresholds to your taste.
    """
    if total <= 0:
        return None
    rate = attacks / total
    # Example policy: alert if ≥10% of rows are attacks OR max prob ≥ 0.95
    if attacks > 0 and (rate >= 0.10 or max_prob >= 0.95):
        msg = (
            f"High alert: {attacks}/{total} predicted attacks "
            f"(rate {rate:.2%}), max probability {max_prob:.3f}"
        )
        _log_event("alert", {"attacks": attacks, "total": total, "max_prob": max_prob})
        return msg
    if attacks > 0:
        msg = f"Alert: {attacks} suspected attacks (max prob {max_prob:.3f})"
        _log_event("alert", {"attacks": attacks, "total": total, "max_prob": max_prob})
        return msg
    return None

# -------------------------
# FastAPI endpoints
# -------------------------
@app.on_event("startup")
def _startup() -> None:
    global _threshold
    _load_model()
    _threshold = _load_threshold()

    # Initialize SQLite event store (new addition)
    _db_init()

    print("✅ Model, threshold, and database initialized.")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "model_loaded": _model is not None,
        "threshold": _threshold,
    }

@app.get("/threshold")
def get_threshold() -> Dict[str, float]:
    return {"threshold": _threshold}

@app.put("/threshold", dependencies=[Depends(require_api_key)])
def set_threshold(req: ThresholdRequest) -> Dict[str, float]:
    global _threshold
    _threshold = float(req.threshold)
    _save_threshold(_threshold)
    return {"threshold": _threshold}

@app.post(
    "/predict_batch",
    response_model=PredictBatchResponse,
    dependencies=[Depends(require_api_key)],
)
def predict_batch(batch: Batch, request: Request) -> PredictBatchResponse:
    try:
        # rate limit
        client_ip = request.client.host if request and request.client else "unknown"
        enforce_rate_limit(client_ip)

        rows = batch.root
        if not rows:
            raise HTTPException(status_code=400, detail="No records provided.")

        df = pd.DataFrame(rows)

        t0 = time.perf_counter()
        results = _predict_core(df)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # summarize for logging
        preds = [r["prediction"] for r in results]
        attacks = sum(1 for p in preds if p == 1)
        normals = len(preds) - attacks
        max_prob = max((r["probability_attack"] for r in results), default=0.0)

        # latency log
        _log_prediction(count=len(results), attacks=attacks, normals=normals, latency_ms=elapsed_ms)

        alert_msg = _maybe_alert(attacks, len(results), max_prob)

        # ✅ persist to SQLite
        ts_iso = datetime.now(timezone.utc).isoformat()
        _db_insert_many(ts_iso, _threshold, rows, results)

        # (optional) Slack alert
        _maybe_alert(attacks, len(results), max_prob)

        return PredictBatchResponse(
            predictions=[PredictResult(**r) for r in results],
            count=len(results),
            alert=alert_msg,
        )
    except HTTPException:
        raise
    except Exception as e:
        expected = _expected_input_features()
        hint = f"\nRequired features: {expected}" if expected else ""
        raise HTTPException(
            status_code=400,
            detail=f"Inference failed: {e}{hint}\n{traceback.format_exc()}",
        )

# Optional single-record endpoint (handy for quick tests)
class Record(RootModel[Dict[str, Any]]):
    pass

@app.post(
    "/predict",
    response_model=PredictResult,
    dependencies=[Depends(require_api_key)],
)
def predict_one(record: Record, request: Request) -> PredictResult:
    try:
        # rate limit
        client_ip = request.client.host if request and request.client else "unknown"
        enforce_rate_limit(client_ip)

        # Build a DataFrame that matches the model’s expected columns
        expected = _expected_input_features()  # list[str] from your trained pipeline
        row_in = dict(record.root)

        # Optional: use categorical-friendly fallbacks for known string cols
        defaults = {"proto": "-", "service": "-", "state": "-", "attack_cat": "-"}

        # Fill any missing columns; numeric cols will safely get 0
        filled = {col: row_in.get(col, defaults.get(col, 0)) for col in expected}
        df = pd.DataFrame([filled], columns=expected)

        # Predict
        t0 = time.perf_counter()
        results = _predict_core(df)  # -> list[dict], length 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        out = results[0]

        # latency log (as single)
        _log_prediction(
            count=1,
            attacks=1 if out["prediction"] == 1 else 0,
            normals=1 if out["prediction"] == 0 else 0,
            latency_ms=elapsed_ms,
        )

        # ✅ persist to SQLite with the normalized row we actually scored
        ts_iso = datetime.now(timezone.utc).isoformat()
        _db_insert_many(ts_iso, _threshold, [filled], results)

        # (optional) Slack alert
        _maybe_alert(attacks=1 if out["prediction"] == 1 else 0,
                     total=1,
                     max_prob=out.get("probability_attack", 0.0))

        return PredictResult(**out)

    except HTTPException:
        raise
    except Exception as e:
        expected = _expected_input_features()
        hint = f"\nRequired features: {expected}" if expected else ""
        raise HTTPException(
            status_code=400,
            detail=f"Inference failed: {e}{hint}\n{traceback.format_exc()}",
        )

from pydantic import BaseModel
from typing import List, Dict, Any

class FeatureContribution(BaseModel):
    name: str
    shap: float


class ExplainResponse(BaseModel):
    base_value: float
    top_features: List[FeatureContribution]

@app.post("/explain", response_model=ExplainResponse, dependencies=[Depends(require_api_key)])
def explain_one(record: Record) -> ExplainResponse:
    if _model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        # prepare one-row DF
        X_in = pd.DataFrame([record.root])
        X = _prepare_dataframe(X_in)
        pre, clf = _get_preprocessor_and_clf(_model)

        # transform features and compute SHAP on classifier
        if pre is not None:
            Xt, feat_names = _get_transformed_and_names(pre, X)
        else:
            Xt = X.values
            feat_names = list(X.columns)

        import shap  # ensure installed
        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(Xt)

        # handle binary class formats
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            base = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            sv = shap_vals
            base = explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[0]

        contrib = sv[0] if hasattr(sv, "__getitem__") else sv
        pairs = list(zip(feat_names, contrib))
        # sort by absolute contribution
        pairs.sort(key=lambda t: abs(float(t[1])), reverse=True)
        top = [{"name": n, "shap": float(v)} for n, v in pairs[:15]]

        return ExplainResponse(base_value=float(np.atleast_1d(base)[0]), top_features=top)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explain failed: {e}")

# useful for the dashboard
class QueryParams(BaseModel):
    limit: int = Field(50, ge=1, le=1000)

@app.get("/events")
def list_events(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"count": len(rows), "items": rows}

@app.post("/events/search")
def search_events(q: QueryParams):
    return list_events(limit=q.limit)

from typing import List, Dict

@app.get("/expected_features")
def expected_features() -> Dict[str, List[str]]:
    return {"features": _expected_input_features()}
