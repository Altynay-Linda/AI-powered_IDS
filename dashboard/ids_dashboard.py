# =========================
# AI-Powered IDS Dashboard
# =========================
import os, json, time, requests
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import shap

# --- PDF report ---
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
)
from reportlab.lib import colors


# -------------------------
# API & logging config
# -------------------------
API_URL = os.getenv("DASHBOARD_API_URL", "http://api:8000")  # docker service name
API_KEY = os.getenv("DASHBOARD_API_KEY", "my-strong-key")
LOG_FILE = os.getenv("LOG_FILE", "/data/ids_metrics_log.jsonl")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def api_get(path: str, **kwargs):
    return requests.get(f"{API_URL}{path}", headers=HEADERS, **kwargs)


def api_post(path: str, payload: dict, **kwargs):
    return requests.post(f"{API_URL}{path}", headers=HEADERS, json=payload, **kwargs)


# ------------------------
# Helper to build the PDF
# ------------------------
def build_pdf_report(
    out_path: str,
    title: str,
    author: str,
    notes: str,
    metrics_txt_path: str,
    threshold_value: float,
    image_paths: dict,
):
    """
    Create a PDF report with metrics, threshold and any images that exist.

    image_paths: {
        "cm": ".../confusion_matrix.png",
        "roc": ".../roc_curve.png",
        "shap_global": ".../shap_global.png",
        "shap_bee": ".../shap_beeswarm.png",
        "shap_row": ".../shap_row.png"
    }
    """
    styles = getSampleStyleSheet()
    story = []

    doc = SimpleDocTemplate(out_path, pagesize=letter)

    story.append(Paragraph(f"<b>{title}</b>", styles["Title"]))
    story.append(Paragraph(f"Author: {author}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Metrics from metrics_txt_path if present
    acc = prec = rec = f1 = roc = None
    if os.path.exists(metrics_txt_path):
        try:
            with open(metrics_txt_path, "r") as f:
                acc, prec, rec, f1, roc = map(float, f.read().split(","))
        except Exception:
            pass

    data = [
        ["Metric", "Value"],
        ["Accuracy", f"{acc:.4f}" if acc is not None else "—"],
        ["Precision", f"{prec:.4f}" if prec is not None else "—"],
        ["Recall", f"{rec:.4f}" if rec is not None else "—"],
        ["F1 Score", f"{f1:.4f}" if f1 is not None else "—"],
        ["ROC-AUC", f"{roc:.4f}" if roc is not None else "—"],
        ["Decision Threshold", f"{threshold_value:.4f}"],
    ]
    tbl = Table(data, hAlign="LEFT")
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(tbl)
    story.append(Spacer(1, 18))

    if notes.strip():
        story.append(Paragraph("<b>Notes:</b>", styles["Heading2"]))
        story.append(Paragraph(notes, styles["Normal"]))
        story.append(Spacer(1, 12))

    # Attach images if they exist
    def add_img(path, caption):
        if path and os.path.exists(path):
            story.append(Spacer(1, 8))
            story.append(Paragraph(f"<b>{caption}</b>", styles["Heading3"]))
            story.append(RLImage(path, width=480, height=320))
            story.append(Spacer(1, 10))

    add_img(image_paths.get("cm"), "Confusion Matrix")
    add_img(image_paths.get("roc"), "ROC Curve")
    add_img(image_paths.get("shap_global"), "SHAP Global (bar)")
    add_img(image_paths.get("shap_bee"), "SHAP Beeswarm")
    add_img(image_paths.get("shap_row"), "SHAP Waterfall (Single Row)")

    doc.build(story)


# -------------------------
# Model & preprocessing helpers
# -------------------------
MODEL_PATHS = [
    os.path.expanduser("/app/ids_best_model.pkl"),
    os.path.expanduser("/app/ids_xgboost_model.pkl"),
]
METRICS_TXT = os.path.expanduser("~/metrics.txt")

st.set_page_config(page_title="AI-Powered IDS", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            return joblib.load(p), p
    raise FileNotFoundError(
        "No trained model file found (ids_best_model.pkl or ids_xgboost_model.pkl)."
    )


def api_get_threshold(timeout=3):
    r = api_get("/threshold", timeout=timeout)
    r.raise_for_status()
    return float(r.json().get("threshold", 0.5))


def api_set_threshold(x: float, timeout=5):
    r = api_post("/threshold", {"threshold": float(x)}, timeout=timeout)
    r.raise_for_status()
    return float(r.json().get("threshold", x))


def get_pre_and_estimator(model):
    """Return (preprocessor_or_None, estimator_for_shap, predict_proba_callable)."""
    pre, est = None, model
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline):
            pre = model.named_steps.get("pre", None)
            est = model.named_steps.get("clf", list(model.named_steps.values())[-1])
    except Exception:
        pass

    try:
        from sklearn.calibration import CalibratedClassifierCV

        if isinstance(est, CalibratedClassifierCV):
            base = getattr(est, "estimator", None)
            if base is not None:
                est = base
    except Exception:
        pass

    def predict_proba_fn(X):
        return model.predict_proba(X)

    return pre, est, predict_proba_fn


def transform_for_model(pre, df):
    """Apply training-time preprocessing if present."""
    if pre is None:
        return df, df.columns.to_numpy()
    Xt = pre.transform(df)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(Xt.shape[1])])
    return Xt, feat_names


def ensure_no_label(df):
    return df.drop(columns=[c for c in df.columns if c.strip().lower() == "label"], errors="ignore")


def safe_sample(df, n=200, seed=42):
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).copy()


def _get_threshold_for_report():
    """Get threshold from API or from local session as fallback."""
    try:
        return api_get_threshold()
    except Exception:
        return float(st.session_state.get("server_threshold", 0.80))


# Common paths that other scripts create
cm_path = os.path.expanduser("~/confusion_matrix.png")
roc_path = os.path.expanduser("~/roc_curve.png")
sg_path = os.path.expanduser("~/shap_global.png")
sb_path = os.path.expanduser("~/shap_beeswarm.png")
sr_path = os.path.expanduser("~/shap_row.png")
pdf_out = os.path.expanduser("~/ids_report.pdf")


# -------------------------
# Layout (Title + Tabs)
# -------------------------
st.title("AI-Powered Intrusion Detection System")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Dashboard",
        "Explainability",
        "Analytics",
        "Help / About",
    ]
)

# =========================
# TAB 1: DASHBOARD
# =========================
with tab1:
    st.subheader("📊 Model Status & Threshold")
    st.subheader("👋 Welcome")
    st.markdown(
        """
        **How to use this IDS (no command line needed):**

        1. **Check Status**  
           Look at the *Model Status & Threshold* section – it should show that the model is loaded.

        2. **Upload Traffic Data (CSV)**  
           Scroll down to **“🧪 Quick Test via API (logs + threshold)”** and upload a CSV containing network flows.  
           The app will:
           - send them to the IDS API,
           - show **how many are Normal vs Attack**,
           - and let you **download only the suspicious ones**.

        3. **Watch Analytics**  
           Open the **Analytics** tab to see:
           - total events,
           - detected attacks vs normal traffic,
           - trends over time.

        You don’t need to run any terminal commands – just keep this page open.
        """
    )

    st.info(
        """
        **Live monitoring mode**

        When an administrator starts the background Zeek → IDS connector,
        this dashboard will automatically:

        - receive live network events, and  
        - update the Analytics tab with new totals and graphs.

        As an end user, you **don’t** need to start or stop anything – just keep this page open.
        """
    )

    # ---- Training metrics & threshold ----
    colA, colB = st.columns([2, 2])

    # Metrics from training (if present)
    with colA:
        if os.path.exists(METRICS_TXT):
            try:
                with open(METRICS_TXT, "r") as f:
                    acc, prec, rec, f1, roc = map(float, f.read().split(","))
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("Precision", f"{prec:.4f}")
                st.metric("Recall", f"{rec:.4f}")
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("ROC-AUC", f"{roc:.4f}")
            except Exception:
                st.info("metrics.txt present but unreadable. It should contain 'acc,prec,rec,f1,roc'.")

    # Threshold control (talks to FastAPI, falls back if API down)
    with colB:
        st.subheader("🎛 Decision Threshold (Server-wide)")
        api_ok = True
        try:
            current_th = api_get_threshold()
        except Exception as e:
            api_ok = False
            current_th = float(st.session_state.get("server_threshold", 0.80))
            st.warning(f"API not reachable at {API_URL}. Start it and refresh. ({e})")

        c1, c2 = st.columns([3, 1])
        with c1:
            new_th = st.slider(
                "Alert threshold (higher = fewer alerts, lower = more alerts)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                value=current_th,
                help="This controls the cutoff for predicting 'Attack'.",
            )
        with c2:
            if st.button("Apply"):
                if api_ok:
                    try:
                        applied = api_set_threshold(new_th)
                        st.session_state.server_threshold = applied
                        st.success(f"Updated API threshold to {applied:.2f}")
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to update threshold: {e}")
                else:
                    st.session_state.server_threshold = new_th
                    st.info("Saved locally. Start the API, then click Apply to push value to server.")

        if api_ok:
            try:
                live_th = api_get_threshold()
                st.caption(f"Current server threshold: **{live_th:.2f}**")
            except Exception:
                st.caption("Current server threshold: (unavailable)")
        else:
            st.caption(
                f"(Offline) Local threshold: **{float(st.session_state.get('server_threshold', new_th)):.2f}**"
            )

    st.divider()

    # ---- Quick Test on CSV (local model) ----
    st.subheader("🧪 Quick Test on CSV (local)")
    uploaded_file = st.file_uploader(
        "Upload a CSV with network flows (local model)", type=["csv"], key="local_csv"
    )
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        # Drop label column if it exists (avoid mismatch)
        if "label" in df.columns:
            df = df.drop(columns=["label"])

        df = pd.DataFrame(df)

        try:
            local_model = joblib.load("ids_best_model.pkl")
            preds = local_model.predict(df)
            df["Prediction"] = preds
            df["Prediction"] = df["Prediction"].map({0: "Normal", 1: "Attack"})
            st.success("✅ Predictions complete!")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()

    # ---- Generate Project Report (PDF) ----
    st.subheader("🧾 Generate Project Report (PDF)")
    default_title = "AI-Powered Network Intrusion Detection – Report"
    default_author = "Altynay"
    title_input = st.text_input("Report title", value=default_title)
    author_input = st.text_input("Author", value=default_author)
    notes_input = st.text_area("Additional notes (optional)", height=120)

    if st.button("Generate PDF"):
        try:
            th = _get_threshold_for_report()
            build_pdf_report(
                out_path=pdf_out,
                title=title_input,
                author=author_input,
                notes=notes_input,
                metrics_txt_path=METRICS_TXT,
                threshold_value=th,
                image_paths={
                    "cm": cm_path,
                    "roc": roc_path,
                    "shap_global": sg_path,
                    "shap_bee": sb_path,
                    "shap_row": sr_path,
                },
            )
            st.success(f"Report created: {pdf_out}")
            with open(pdf_out, "rb") as f:
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f.read(),
                    file_name="ids_report.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"Report generation failed: {e}")
            st.info(
                "Tip: make sure confusion_matrix.png, roc_curve.png and SHAP images exist."
            )

    st.divider()

    # ---- Quick Test via API (logs + threshold) ----
    st.subheader("🧪 Quick Test via API (logs + threshold)")
    with st.expander("How this works (click to expand)", expanded=True):
        st.markdown(
            """
            Upload a CSV file with one row per network flow (for example, exported from
            a SIEM, firewall logs, or preprocessed Zeek logs).

            The IDS will:
            - align your columns to its expected schema,
            - run the model on all rows,
            - show a summary: total rows, attacks, and attack rate,
            - and let you download a CSV containing *only* the rows flagged as attacks.
            """
        )

    uploaded_api = st.file_uploader(
        "Upload a CSV (sent to FastAPI)", type=["csv"], key="api_csv"
    )
    if uploaded_api is not None:
        df_api = pd.read_csv(uploaded_api)
        st.write("Preview:")
        st.dataframe(df_api.head())

        if "label" in df_api.columns:
            df_api = df_api.drop(columns=["label"])

        # Ask API what it expects and align columns
        try:
            r = requests.get(f"{API_URL}/expected_features", headers=HEADERS, timeout=10)
            r.raise_for_status()
            required = r.json().get("features", [])
        except Exception as e:
            st.error(f"Could not fetch expected features from API: {e}")
            required = list(df_api.columns)

        df_api = df_api.reindex(columns=required, fill_value=0)
        payload = df_api.to_dict(orient="records")

        st.write("Posting", len(payload), "rows to API …")
        try:
            resp = requests.post(
                f"{API_URL}/predict_batch",
                headers=HEADERS,
                json=payload,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()

            preds = data.get("predictions", [])
            n = len(preds)
            attacks = sum(1 for p in preds if p.get("prediction", 0) == 1)
            normals = n - attacks
            attack_rate = (attacks / n) if n else 0.0
            max_prob = max(
                (p.get("probability_attack", 0.0) for p in preds), default=0.0
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{n:,}")
            c2.metric("Attacks", f"{attacks:,}")
            c3.metric("Attack rate", f"{attack_rate:.2%}")
            c4.metric("Max attack prob", f"{max_prob:.3f}")

            alert_msg = data.get("alert")
            if alert_msg:
                st.warning(alert_msg)

            st.subheader("Sample predictions")
            sample_df = pd.DataFrame(preds[:50])
            if not sample_df.empty:
                st.dataframe(sample_df, use_container_width=True)
            else:
                st.info("No predictions returned.")

            # Probability histogram
            probs = [p.get("probability_attack", 0.0) for p in preds]
            if probs:
                fig = plt.figure(figsize=(6, 3))
                plt.hist(probs, bins=30)
                plt.title("Distribution of attack probability")
                plt.xlabel("p(Attack)")
                plt.ylabel("Count")
                st.pyplot(fig, use_container_width=True)

            # Attack-only CSV download
            attacks_df = pd.DataFrame(
                [p for p in preds if p.get("prediction", 0) == 1]
            )
            if not attacks_df.empty:
                csv_bytes = attacks_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download attack-only predictions (CSV)",
                    data=csv_bytes,
                    file_name="attacks_only.csv",
                    mime="text/csv",
                )

            with st.expander("Raw API response (debug)"):
                st.json(data)

        except requests.HTTPError as e:
            st.error(f"API call failed: {e}")
            try:
                st.code(resp.text, language="json")
            except Exception:
                pass
        except Exception as e:
            st.error(f"Unexpected error calling API: {e}")


# =========================
# TAB 2: EXPLAINABILITY (SHAP)
# =========================
with tab2:
    st.header("🧠 Explainability (SHAP)")
    st.markdown(
        """
        This tab is for **analysts** who want to understand *why* the model
        labels a flow as Attack or Normal. If you're just using the IDS day-to-day,
        you can ignore this tab.
        """
    )

    # Load model once
    try:
        model, model_path = load_model()
        pre, est_for_shap, predict_proba_fn = get_pre_and_estimator(model)
        st.caption(f"Loaded model: `{os.path.basename(model_path)}`")
    except Exception as e:
        st.error(f"Cannot load model: {e}")
        st.stop()

    src = st.radio(
        "Explain using:", ["Uploaded CSV (if any)", "UNSW_NB15_testing-set.csv"],
        horizontal=True,
    )
    df_explain = None
    uploaded_explain = st.file_uploader(
        "Optionally upload a CSV to explain",
        type=["csv"],
        accept_multiple_files=False,
        key="explain_upl",
    )
    if src.startswith("Uploaded") and uploaded_explain is not None:
        df_explain = pd.read_csv(uploaded_explain)
    else:
        test_path = os.path.expanduser("~/UNSW_NB15_testing-set.csv")
        if os.path.exists(test_path):
            df_explain = pd.read_csv(test_path)
        else:
            st.warning(
                "No uploaded CSV and ~/UNSW_NB15_testing-set.csv not found. "
                "Please upload a CSV."
            )

    if df_explain is not None:
        st.write("Preview:", df_explain.head())
        df_explain = ensure_no_label(df_explain)

        try:
            X_explain, feat_names = transform_for_model(pre, df_explain)
        except Exception as e:
            st.error(f"Preprocessing failed. Ensure columns match training.\n{e}")
            st.stop()

        # -------- Global SHAP --------
        st.subheader("🌍 Global Importance")
        colg1, colg2 = st.columns(2)
        with colg1:
            if st.button("Compute Global SHAP (bar + beeswarm)"):
                with st.spinner("Computing global SHAP on a sample..."):
                    try:
                        explainer = shap.TreeExplainer(est_for_shap)
                        X_bg = safe_sample(pd.DataFrame(X_explain), n=300)
                        sv = explainer.shap_values(X_bg)
                        if isinstance(sv, list):
                            sv = sv[1] if len(sv) > 1 else sv[0]

                        fig1 = plt.figure(figsize=(7, 4))
                        shap.summary_plot(
                            sv,
                            X_bg,
                            feature_names=feat_names,
                            plot_type="bar",
                            show=False,
                        )
                        plt.tight_layout()
                        st.pyplot(fig1)
                        fig1.savefig(
                            os.path.expanduser("~/shap_global.png"), dpi=300
                        )

                        fig2 = plt.figure(figsize=(7, 4))
                        shap.summary_plot(
                            sv, X_bg, feature_names=feat_names, show=False
                        )
                        plt.tight_layout()
                        st.pyplot(fig2)
                        fig2.savefig(
                            os.path.expanduser("~/shap_beeswarm.png"), dpi=300
                        )
                        st.success(
                            "Saved ~/shap_global.png and ~/shap_beeswarm.png"
                        )
                    except Exception:
                        X_bg_df = safe_sample(pd.DataFrame(X_explain), n=200)
                        f = lambda X: predict_proba_fn(
                            pd.DataFrame(X, columns=feat_names)
                        )[:, 1]
                        explainer = shap.Explainer(f, X_bg_df)
                        sv = explainer(X_bg_df)

                        fig1 = plt.figure(figsize=(7, 4))
                        shap.plots.bar(sv, show=False, max_display=20)
                        plt.tight_layout()
                        st.pyplot(fig1)
                        fig1.savefig(
                            os.path.expanduser("~/shap_global.png"), dpi=300
                        )

                        fig2 = plt.figure(figsize=(7, 4))
                        shap.plots.beeswarm(sv, show=False, max_display=20)
                        plt.tight_layout()
                        st.pyplot(fig2)
                        fig2.savefig(
                            os.path.expanduser("~/shap_beeswarm.png"), dpi=300
                        )
                        st.success(
                            "Saved ~/shap_global.png and ~/shap_beeswarm.png"
                        )
        with colg2:
            if os.path.exists(os.path.expanduser("~/shap_global.png")):
                st.image(
                    os.path.expanduser("~/shap_global.png"),
                    caption="Global SHAP (bar)",
                    use_container_width=True,
                )
            if os.path.exists(os.path.expanduser("~/shap_beeswarm.png")):
                st.image(
                    os.path.expanduser("~/shap_beeswarm.png"),
                    caption="Global SHAP (beeswarm)",
                    use_container_width=True,
                )

        # -------- Single-row SHAP --------
        st.subheader("🔎 Explain a Single Row")
        row_idx = st.number_input(
            "Row index to explain",
            min_value=0,
            max_value=max(0, len(df_explain) - 1),
            value=0,
            step=1,
        )
        if st.button("Explain Selected Row (waterfall)"):
            with st.spinner("Computing local SHAP..."):
                try:
                    explainer = shap.TreeExplainer(est_for_shap)
                    xrow = np.array(X_explain[row_idx : row_idx + 1])
                    sv = explainer.shap_values(xrow)
                    if isinstance(sv, list):
                        sv = sv[1] if len(sv) > 1 else sv[0]
                    base_value = getattr(explainer, "expected_value", 0.0)
                    if isinstance(base_value, (list, np.ndarray)):
                        base_value = (
                            base_value[1]
                            if len(np.atleast_1d(base_value)) > 1
                            else base_value[0]
                        )
                    exp = shap.Explanation(
                        values=sv.flatten(),
                        base_values=np.array([base_value]),
                        data=xrow.flatten(),
                        feature_names=feat_names,
                    )
                    fig = plt.figure(figsize=(8, 5))
                    shap.plots.waterfall(exp, max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    outp = os.path.expanduser("~/shap_row.png")
                    fig.savefig(outp, dpi=300)
                    st.success(f"Saved {outp}")
                except Exception:
                    X_bg_df = safe_sample(pd.DataFrame(X_explain), n=200)
                    f = lambda X: predict_proba_fn(
                        pd.DataFrame(X, columns=feat_names)
                    )[:, 1]
                    explainer = shap.Explainer(f, X_bg_df)
                    xrow = np.array(X_explain[row_idx : row_idx + 1])
                    sv = explainer(xrow)
                    fig = plt.figure(figsize=(8, 5))
                    shap.plots.waterfall(sv[0], max_display=15, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    outp = os.path.expanduser("~/shap_row.png")
                    fig.savefig(outp, dpi=300)
                    st.success(f"Saved {outp}")

    # -------- Explain via API (remote model) --------
    st.header("🌐 Explain a Record via API")

    # Prefer the most recent uploaded CSV from dashboard local/API
    if "df" in locals():
        df_source = df.copy()
        st.caption("Using the most recent uploaded CSV from the Dashboard.")
    else:
        st.caption("No uploaded CSV detected — pulling from API event log (if enabled).")
        try:
            r = api_get("/events", timeout=10)
            items = r.json().get("items", []) if r.ok else []
            if items:
                df_source = pd.DataFrame(
                    [json.loads(items[0]["features_json"])]
                )
            else:
                df_source = None
                st.info("No events available yet.")
        except Exception as e:
            df_source = None
            st.error(f"Could not fetch events from {API_URL}: {e}")

    if df_source is not None:
        st.write("Sample features available for explanation:")
        st.dataframe(df_source.head(5), use_container_width=True)

        row_index = st.number_input(
            "Row to explain (0-based index)",
            min_value=0,
            max_value=len(df_source) - 1,
            value=0,
            step=1,
            key="api_explain_row",
        )

        if st.button("Explain via API"):
            payload = df_source.iloc[row_index].to_dict()
            try:
                ex = api_post("/explain", payload, timeout=30)
                ex.raise_for_status()
                data = ex.json()
                st.success("✅ Explanation received")

                feats = data.get("top_features", [])
                names = [f["name"] for f in feats][::-1]
                vals = [f["shap"] for f in feats][::-1]

                fig = plt.figure(figsize=(8, 5))
                plt.barh(names, vals)
                plt.axvline(0, linestyle="--")
                plt.title("Top SHAP Feature Contributions (API)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

                fig.savefig("shap_row_api.png", dpi=300)
                st.caption("Saved as shap_row_api.png for your PDF report.")
            except Exception as e:
                st.error(f"Explain request failed: {e}")


# =========================
# TAB 3: ANALYTICS (real-time)
# =========================
with tab3:
    st.header("📊 Real-Time IDS Analytics")
    st.markdown(
        """
        This page shows **live statistics** based on all predictions the IDS has seen.

        - **Total Events** – how many flows have been analyzed  
        - **Detected Attacks** – flows the model classified as malicious  
        - **Normal Traffic** – everything else  
        - **Events Over Time** – how many attacks vs normal events for each batch
        - **Sends traffic via an automated script (like Zeek → API),

        the analytics will update automatically. You don’t need to press anything.
        """
    )

    refresh = st.slider(
        "Refresh every (seconds)", 2, 30, 5, help="Auto-refresh Analytics tab"
    )

    if os.path.exists(LOG_FILE):
        mtime = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(LOG_FILE))
        )
        st.caption(f"Log file: {LOG_FILE}  •  Last update: {mtime}")
    else:
        st.info("No logs yet — run some predictions via the API or the Dashboard upload.")
        st.stop()

    try:
        df_log = pd.read_json(LOG_FILE, lines=True)
    except ValueError:
        st.warning("Log file is currently being written. Click Rerun in a second.")
        st.stop()

    if df_log.empty:
        st.info("Log is empty. Trigger some predictions to see analytics.")
        st.stop()

    # --- keep only real-time events (small batches, e.g. Zeek → API) ---
    REALTIME_MAX_BATCH = 1000  # anything bigger is treated as a CSV test batch
    df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors="coerce")
    df_log = df_log.dropna(subset=["timestamp"]).sort_values("timestamp")

    df_rt = df_log[df_log["count"] <= REALTIME_MAX_BATCH].copy()

    if df_rt.empty:
        st.info(
            "No real-time events yet. "
            "So far only large CSV test batches were logged. "
            "Run the Zeek → API connector or send smaller batches to see live analytics."
        )
        st.stop()

    total = int(df_rt["count"].sum())
    attacks = int(df_rt["attacks"].sum())
    normals = int(df_rt["normals"].sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Events (real-time)", f"{total:,}")
    c2.metric("Detected Attacks", f"{attacks:,}")
    c3.metric("Normal Traffic", f"{normals:,}")

    st.subheader("Events Over Time (real-time only)")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df_rt["timestamp"], df_rt["attacks"], label="Attacks")
    ax.plot(df_rt["timestamp"], df_rt["normals"], label="Normal")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count per request")
    ax.legend()
    ax.grid(True, alpha=0.25)
    st.pyplot(fig)

    st.caption(
        f"Real-time view: includes only batches with count ≤ {REALTIME_MAX_BATCH}. "
        "Large CSV test runs from the Dashboard are excluded."
    )
   # try:
    #    df_log = pd.read_json(LOG_FILE, lines=True)
    #except ValueError:
     #   st.warning("Log file is currently being written. Click Rerun in a second.")
      #  st.stop()

   # if df_log.empty:
    #    st.info("Log is empty. Trigger some predictions to see analytics.")
     #   st.stop()

   # df_log["timestamp"] = pd.to_datetime(df_log["timestamp"], errors="coerce")
   # df_log = df_log.dropna(subset=["timestamp"]).sort_values("timestamp")

   # total = int(df_log["count"].sum())
   # attacks = int(df_log["attacks"].sum())
   # normals = int(df_log["normals"].sum())

   # c1, c2, c3 = st.columns(3)
   # c1.metric("Total Events", f"{total:,}")
    #c2.metric("Detected Attacks", f"{attacks:,}")
   # c3.metric("Normal Traffic", f"{normals:,}")

   # st.subheader("Events Over Time")
   # fig, ax = plt.subplots(figsize=(8, 3))
   # ax.plot(df_log["timestamp"], df_log["attacks"], label="Attacks")
   # ax.plot(df_log["timestamp"], df_log["normals"], label="Normal")
   # ax.set_xlabel("Time")
   # ax.set_ylabel("Count per request")
   # ax.legend()
   # ax.grid(True, alpha=0.25)
   # st.pyplot(fig)

   # st.caption("Use the slider to choose how often you want to refresh, then click the Rerun icon in the toolbar if needed.")
# Auto-refresh disabled so other tabs can render correctly.
# If you really want auto-refresh, we can add a safer version later.
# time.sleep(refresh)
# st.rerun()
 


# =========================
# TAB 4: HELP / ABOUT
# =========================
with tab4:
    st.header("ℹ️ Help / About this IDS")

    st.markdown(
        """
        This application is an **AI-Powered Network Intrusion Detection System (IDS)**.

        It combines:

        - ✅ A trained machine learning model (XGBoost)  
        - ✅ A FastAPI backend (`/predict` and `/predict_batch` endpoints)  
        - ✅ A Streamlit dashboard (this UI)  
        - ✅ A live streaming from **Zeek** logs

        ---
        ### 🧠 High-Level Architecture

        ```text
        [Network Traffic]
                ↓
             Zeek (conn.log)
                ↓             (optional live streaming)
         zeek_to_api.py  ─────────►  FastAPI IDS API  ──►  Model
                                         │
                                         │ writes logs to /data/ids_metrics_log.jsonl
                                         ↓
                                  Streamlit Dashboard
        ```

        - **Zeek**: observes real network traffic and writes `conn.log`  
        - **zeek_to_api.py**: tails `conn.log` and sends batches of flows to `/predict_batch`  
        - **FastAPI IDS API**: runs the ML model, returns predictions, logs metrics  
        - **Streamlit Dashboard**: visual front-end for non-technical users
        """
    )

    st.subheader("🚀 How a normal user should use this app")
    st.markdown(
        """
        You **do not** need the command line to use the IDS as an operator.

        **1. Open the dashboard**

        - Open your browser and go to: `http://localhost:8501`

        **2. Test a CSV file**

        On the **Dashboard** tab:

        - Go to **“🧪 Quick Test via API (logs + threshold)”**  
        - Click **Browse files** and upload a CSV with network flows  
        - The app will:
          - Send all rows to the IDS API  
          - Show:
            - total rows  
            - how many are **Normal** vs **Attack**  
            - attack probability histogram  
          - Let you **download a CSV containing attack-only rows**

        **3. View live analytics**

        On the **Analytics** tab:

        - See **Total Events**, **Detected Attacks**, **Normal Traffic**  
        - Watch the **Events Over Time** line chart
        """
    )

    st.subheader("👩‍💻 For administrators / power users")
    st.markdown(
        """
        These parts typically require basic command-line skills (already done for this project):

        - Run the **IDS API** and **Dashboard** via Docker:  
          `docker compose up -d`
        - (Optional) Start **Zeek → API streaming**:  
          `python ~/zeek_to_api.py`
        - Ensure the `/data` volume is shared so both API and dashboard can read
          `ids_metrics_log.jsonl`.

        Once this is set up, non-technical users can work **only in the browser**.
        """
    )

    st.subheader("📌 Tabs overview")
    st.markdown(
        """
        - **Dashboard** – Model status, threshold control, CSV tests, and PDF report generation.  
        - **Explainability** – SHAP-based explanations (global importance and individual flow explanations).  
        - **Analytics** – Real-time metrics and trends based on all predictions logged by the API.  
        - **Help / About** – High-level description, user workflow, and admin notes.
        """
    )

    st.subheader("❓ Troubleshooting (common issues)")
    st.markdown(
        """
        **1. Analytics shows “No logs yet”**  
        - Make sure the API is running (`/health` returns ok)  
        - Send some data (Quick Test via API or `zeek_to_api.py`)  
        - Check that `/data/ids_metrics_log.jsonl` exists inside both containers.

        **2. Everything is classified as Attack**  
        - Adjust the **Decision Threshold** slider on the Dashboard tab.  
        - For low-risk / clean data, you might want a higher threshold (closer to 1.0).  
        - For noisy / attack-heavy datasets, a lower threshold may make more sense.

        **3. The app cannot reach the API**  
        - Check that the Docker `api` service is running  
        - In `docker-compose.yml`, ensure:
          - `DASHBOARD_API_URL=http://api:8000`  
          - `DASHBOARD_API_KEY` matches the backend key
        """
    )

    st.caption("Version: Academic demo build – AI-Powered IDS with FastAPI + Streamlit + Zeek")
