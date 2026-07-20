import streamlit as st
import pandas as pd
import requests
from sklearn.metrics import f1_score, precision_recall_fscore_support

st.set_page_config(page_title="Sentiment Evaluator", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F5F0E8; color: #3E2000; }
    .page-eyebrow { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #8B6A4A; margin-bottom: 8px; }
    .page-title { font-size: 36px; font-weight: 700; color: #3E2000; font-family: Georgia, serif; margin-bottom: 4px; }
    .page-divider { width: 48px; height: 2px; background: #C4956A; margin: 16px 0 24px; border-radius: 2px; }
    .info-card { background: white; border: 1.5px solid #D4B896; border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; }
    .card-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #A0845C; margin-bottom: 10px; display: block; }
    .metric-row { display: flex; gap: 16px; margin-top: 24px; }
    .metric-card { flex: 1; background: white; border: 1.5px solid #D4B896; border-radius: 10px; padding: 20px 24px; }
    .metric-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #A0845C; margin-bottom: 8px; display: block; }
    .metric-value { font-size: 32px; font-weight: 700; color: #3E2000; font-family: Georgia, serif; }
    .badge { font-size: 11px; color: #7A5230; background: #EDD9BE; border-radius: 20px; padding: 4px 12px; border: 1px solid #D4B896; display: inline-block; margin-right: 6px; margin-bottom: 6px; }
    div[data-testid="stRadio"] label { color: #6B4C2A !important; font-size: 14px; }
    div[data-testid="stSelectbox"] label { color: #6B4C2A !important; font-size: 13px; }
    div[data-testid="stFileUploader"] label { color: #6B4C2A !important; }
    div.stButton { display: flex; justify-content: flex-start; }
    div.stButton > button { background: #8B5E3C; color: white; border: none; border-radius: 8px; padding: 10px 28px; font-size: 14px; font-weight: 600; margin-top: 8px; }
    div.stButton > button:hover { background: #7A4F30; }
    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 32px 0 8px;">
  <p class="page-eyebrow">TOSC Research · Module 1</p>
  <p class="page-title">F1 Sentiment Evaluation</p>
  <div class="page-divider"></div>
  <div>
    <span class="badge">Weighted F1</span>
    <span class="badge">Precision</span>
    <span class="badge">Recall</span>
    <span class="badge">Accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_MAP = {
    "Gemma 2.2B": "gemma2:2b",
    "Qwen 2.5 3B": "qwen2.5:3b",
    "Llama 3.2 3B": "llama3.2:3b",
}

def get_sentiment(text, model_tag, use_ner=False):
    if use_ner:
        prompt = f"""You are a sentiment analysis model for Taglish political text with named entity awareness.
Classify the sentiment of the following text as Positive, Negative, or Neutral.
Only respond with one word: Positive, Negative, or Neutral.

Text: {text}
Sentiment:"""
    else:
        prompt = f"""You are a sentiment analysis model for Taglish political text.
Classify the sentiment of the following text as Positive, Negative, or Neutral.
Only respond with one word: Positive, Negative, or Neutral.

Text: {text}
Sentiment:"""
    try:
        res = requests.post(OLLAMA_URL, json={
            "model": model_tag,
            "prompt": prompt,
            "stream": False
        }, timeout=30)
        result = res.json().get("response", "").strip().capitalize()
        if result not in ["Positive", "Negative", "Neutral"]:
            return "Neutral"
        return result
    except Exception:
        return "Error"

# ── Config card ───────────────────────────────────────────
st.markdown('<div class="info-card"><span class="card-label">Configuration</span>', unsafe_allow_html=True)
model = st.selectbox("Model:", list(MODEL_MAP.keys()))
dataset_mode = st.radio("Dataset type:", ["Without NER", "With NER"], horizontal=True)
uploaded_file = st.file_uploader("Upload CSV:", type=["csv"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown('<div class="info-card"><span class="card-label">Column Mapping</span>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        text_col = st.selectbox("Text column:", df.columns)
    with col_b:
        true_col = st.selectbox("True label column:", df.columns)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Run Evaluation →"):
        model_tag = MODEL_MAP[model]
        use_ner = dataset_mode == "With NER"

        progress = st.progress(0, text="Running predictions...")
        predictions = []

        for i, row in df.iterrows():
            pred = get_sentiment(str(row[text_col]), model_tag, use_ner)
            predictions.append(pred)
            progress.progress((i + 1) / len(df), text=f"Processing row {i + 1} of {len(df)}...")

        progress.empty()

        df["predicted"] = predictions
        true_labels = df[true_col].astype(str).str.capitalize()
        pred_labels = df["predicted"].astype(str).str.capitalize()

        wf1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        acc = (true_labels == pred_labels).mean()
        precision, recall, _, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="weighted", zero_division=0
        )

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <span class="metric-label">Weighted F1</span>
                <span class="metric-value">{wf1:.4f}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Accuracy</span>
                <span class="metric-value">{acc:.2%}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Precision</span>
                <span class="metric-value">{precision:.4f}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">Recall</span>
                <span class="metric-value">{recall:.4f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 32px;">
          <p class="page-eyebrow">Results</p>
          <div style="width:48px; height:2px; background:#C4956A; margin-bottom:16px; border-radius:2px;"></div>
        </div>
        """, unsafe_allow_html=True)

        results_df = df.copy()
        results_df["Predicted"] = pred_labels.values
        results_df["Correct"] = (true_labels == pred_labels).values
        st.dataframe(results_df, use_container_width=True)

        csv_out = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results CSV", csv_out, "sentiment_results.csv", "text/csv")

st.markdown("""
<div style="text-align:center; margin-top:48px; padding-bottom:24px;">
  <p style="font-size:11px; color:#A08060; letter-spacing:0.04em;">Entity-targeted bias detection · Political NLP · Taglish corpus</p>
</div>
""", unsafe_allow_html=True)