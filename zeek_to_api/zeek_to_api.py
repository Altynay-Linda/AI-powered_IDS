#!/usr/bin/env python3
"""
Tail Zeek conn.log and POST batches to the IDS FastAPI /predict_batch endpoint.
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd

# ===== CONFIG =====
API_URL = os.getenv("IDS_API", "http://api:8000/predict_batch")
CONN_LOG = os.path.expanduser("/logs/conn.log")

BATCH_SIZE = int(os.getenv("ZEEK_BATCH", "5"))
FLUSH_EVERY = 3.0  # seconds
TIMEOUT = 10

API_KEY = os.getenv("IDS_API_KEY", "my-strong-key")
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# --- Required columns (must match the model)
REQ_COLS = [
    "id", "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean",
    "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm",
    "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "attack_cat",
]


def map_zeek_conn_to_features(df_conn: pd.DataFrame) -> pd.DataFrame:
    """
    Map Zeek conn.log columns to the IDS model features with sane defaults:
    - infer service from port if Zeek leaves it '.'
    - avoid infinite/huge engineered values when duration==0
    - cap a few heavy-tailed numeric features
    """
    out = pd.DataFrame(0.0, index=df_conn.index, columns=REQ_COLS, dtype="float64")

    # --- categorical defaults
    out[["proto", "service", "state", "attack_cat"]] = "-"
    out["id"] = np.arange(1, len(df_conn) + 1, dtype="int64")

    # ---- raw Zeek fields (may be missing)
    dur = pd.to_numeric(df_conn.get("duration"), errors="coerce")          # seconds
    spkts = pd.to_numeric(df_conn.get("orig_pkts"), errors="coerce")
    dpkts = pd.to_numeric(df_conn.get("resp_pkts"), errors="coerce")
    sbytes = pd.to_numeric(df_conn.get("orig_ip_bytes"), errors="coerce")
    dbytes = pd.to_numeric(df_conn.get("resp_ip_bytes"), errors="coerce")

    # proto / state
    if "proto" in df_conn.columns:
        out["proto"] = df_conn["proto"].astype(str).str.lower().fillna("-")
    if "conn_state" in df_conn.columns:
        out["state"] = df_conn["conn_state"].astype(str).fillna("-")

    # Ports (for service inference)
    oport = pd.to_numeric(df_conn.get("id.orig_p"), errors="coerce")
    rport = pd.to_numeric(df_conn.get("id.resp_p"), errors="coerce")

    # Start with Zeek service if present
    if "service" in df_conn.columns:
        out["service"] = df_conn["service"].fillna("-").astype(str).str.lower()

    # Fallback: infer service from well-known ports
    def _infer_service(op, rp, proto: str):
        p = rp if np.isfinite(rp) else op
        if not np.isfinite(p):
            return "-"
        p = int(p)
        if proto == "udp":
            if p == 53:
                return "dns"
            if p == 123:
                return "ntp"
            if p == 67 or p == 68:
                return "dhcp"
        # tcp or other
        if p == 80:
            return "http"
        if p == 443:
            return "https"
        if p == 22:
            return "ssh"
        if p == 21:
            return "ftp"
        if p == 25:
            return "smtp"
        if p == 110:
            return "pop3"
        if p == 143:
            return "imap"
        return "-"

    # where service == '-' try to infer
    if "service" not in df_conn.columns or (out["service"] == "-").any():
        proto_series = (
            df_conn["proto"] if "proto" in df_conn.columns else pd.Series("-", index=df_conn.index)
        )
        inferred = [
            _infer_service(
                oport.iloc[i] if oport is not None and len(oport) > i else np.nan,
                rport.iloc[i] if rport is not None and len(rport) > i else np.nan,
                str(proto_series.iloc[i]).lower(),
            )
            for i in range(len(df_conn))
        ]
        out.loc[out["service"] == "-", "service"] = pd.Series(inferred, index=df_conn.index)

    # ---- numeric base columns (0-safe)
    out["dur"] = dur.fillna(0)
    out["spkts"] = spkts.fillna(0)
    out["dpkts"] = dpkts.fillna(0)
    out["sbytes"] = sbytes.fillna(0)
    out["dbytes"] = dbytes.fillna(0)

    # If duration == 0 but we saw packets, give tiny non-zero to stabilise rates
    min_dur = 0.02  # 20 ms
    has_pkts = (out["spkts"] + out["dpkts"]) > 0
    out.loc[(out["dur"] == 0) & has_pkts, "dur"] = min_dur

    # ---- engineered features (stable, capped)
    eps = 1e-6
    dur_nz = out["dur"].where(out["dur"] > 0, eps)
    spkts_nz = out["spkts"].where(out["spkts"] > 0, eps)
    dpkts_nz = out["dpkts"].where(out["dpkts"] > 0, eps)

    out["rate"] = (out["sbytes"] + out["dbytes"]) / dur_nz
    out["smean"] = out["sbytes"] / spkts_nz
    out["dmean"] = out["dbytes"] / dpkts_nz
    out["sload"] = out["sbytes"] / dur_nz
    out["dload"] = out["dbytes"] / dur_nz
    out["sinpkt"] = np.clip(out["dur"] / spkts_nz, 0, 60.0)
    out["dinpkt"] = np.clip(out["dur"] / dpkts_nz, 0, 60.0)

    # Hard caps aligned with realistic network values
    out["smean"] = np.clip(out["smean"], 0, 1518)      # bytes/packet
    out["dmean"] = np.clip(out["dmean"], 0, 1518)
    out["rate"] = np.clip(out["rate"], 0, 1_000_000)   # 1 MB/s cap
    out["sload"] = np.clip(out["sload"], 0, 1_000_000)
    out["dload"] = np.clip(out["dload"], 0, 1_000_000)

    # Features not available in conn.log -> keep zeros
    for col in [
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
        "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
        "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
        "sttl", "dttl", "sloss", "dloss", "swin", "stcpb", "dtcpb", "dwin",
        "tcprtt", "synack", "ackdat", "trans_depth", "response_body_len",
    ]:
        if col not in out.columns:
            out[col] = 0.0

    out = out.fillna(0)
    out = out.reindex(columns=REQ_COLS, fill_value=0)

    for c in ["proto", "service", "state", "attack_cat"]:
        out[c] = out[c].astype(str)

    return out


def read_conn_header(path: str):
    """Read #fields header from Zeek TSV log."""
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#fields"):
                return line.strip().split("\t")[1:]
    return None


def iter_conn_rows(path: str):
    """Yield rows from conn.log continuously."""
    cols = None
    while cols is None:
        if os.path.exists(path):
            cols = read_conn_header(path)
            if cols:
                break
        time.sleep(0.3)

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(cols):
                parts += [""] * (len(cols) - len(parts))
            yield dict(zip(cols, parts))

        while True:
            line = f.readline()
            if not line:
                time.sleep(0.3)
                continue
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(cols):
                parts += [""] * (len(cols) - len(parts))
            yield dict(zip(cols, parts))


def main():
    if not os.path.exists(CONN_LOG):
        print(f"[zeek_to_api] Missing {CONN_LOG}. Start Zeek first.")
        return

    print(f"[zeek_to_api] Watching {CONN_LOG} → POST to {API_URL}")

    # Ask API what column order it expects
    expected = None
    try:
        r = requests.get("http://127.0.0.1:8000/expected_features", timeout=5)
        r.raise_for_status()
        expected = r.json().get("features", [])
        if expected:
            print(f"[zeek_to_api] expected_features: {len(expected)} cols")
    except Exception as e:
        print(
            f"[zeek_to_api] warn: expected_features unavailable ({e}); "
            "using REQ_COLS order"
        )
        expected = list(REQ_COLS)

    buf = []
    last_post = time.time()

    try:
        for conn_row in iter_conn_rows(CONN_LOG):
            df_raw = pd.DataFrame([conn_row])
            feat_df = map_zeek_conn_to_features(df_raw)

            # Align columns exactly as API expects
            feat_df = feat_df.reindex(columns=expected, fill_value=0)

            # Convert numpy → Python types
            row = {}
            for k, v in feat_df.iloc[0].items():
                if hasattr(v, "item"):
                    try:
                        v = v.item()
                    except Exception:
                        pass
                row[k] = v

            # On first row, show feature keys so we can debug mapping
            if not buf:
                debug_keys = list(row.keys())[:20]
                print("[DEBUG first payload keys]", debug_keys)

            buf.append(row)
            print(f"[DEBUG] buffered={len(buf)}")
            now = time.time()

            if len(buf) >= BATCH_SIZE or (now - last_post) >= FLUSH_EVERY:
                try:
                    r = requests.post(
                        API_URL, headers=HEADERS, json=buf, timeout=TIMEOUT
                    )
                    if r.ok:
                        j = r.json()
                        # Inspect probabilities for debugging
                        probs = [
                            p.get("probability_attack", 0.0)
                            for p in j.get("predictions", [])
                        ]
                        if probs:
                            p_min = min(probs)
                            p_max = max(probs)
                            p_med = float(np.median(probs))
                            p_q25 = float(np.percentile(probs, 25))
                            print(
                                f"[PROBS] min={p_min:.3f}  p25={p_q25:.3f}  "
                                f"median={p_med:.3f}  max={p_max:.3f}"
                            )
                        print(
                            f"[ALERT] {j.get('alert', 'no alert')} "
                            f"(count={j.get('count')})"
                        )
                    else:
                        print(
                            f"[POST] sent={len(buf)} status={r.status_code} "
                            f"body={r.text[:200]}"
                        )
                except Exception as e:
                    print(f"[POST ERROR] {e}")
                buf = []
                last_post = now

    except KeyboardInterrupt:
        if buf:
            try:
                r = requests.post(
                    API_URL, headers=HEADERS, json=buf, timeout=30
                )
                print(
                    f"[POST] final flush sent={len(buf)} "
                    f"status={r.status_code}"
                )
            except Exception as e:
                print(f"[POST ERROR final] {e}")


if __name__ == "__main__":
    main()
