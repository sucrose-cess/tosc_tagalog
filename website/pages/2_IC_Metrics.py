import streamlit as st
import pandas as pd
import requests
from scipy.stats import entropy

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="IC Metrics", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F5F0E8; color: #3E2000; }
    .page-eyebrow { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #8B6A4A; margin-bottom: 8px; }
    .page-title { font-size: 36px; font-weight: 700; color: #3E2000; font-family: Georgia, serif; margin-bottom: 4px; }
    .page-divider { width: 48px; height: 2px; background: #C4956A; margin: 16px 0 24px; border-radius: 2px; }
    .info-card { background: white; border: 1.5px solid #D4B896; border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; }
    .card-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #A0845C; margin-bottom: 10px; display: block; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 32px 0 8px;">
  <p class="page-eyebrow">TOSC Research · Module 2</p>
  <p class="page-title">Inconsistency Metrics</p>
  <div class="page-divider"></div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_MAP = {"Gemma 2.2B": "gemma2:2b", "Qwen 2.5 3B": "qwen2.5:3b", "Llama 3.2 3B": "llama3.2:3b"}

def get_sentiment(text, model_tag):
    prompt = f"Classify the sentiment of this text as Positive, Negative, or Neutral. Respond with one word only: {text}"
    try:
        res = requests.post(OLLAMA_URL, json={"model": model_tag, "prompt": prompt, "stream": False}, timeout=60)
        return res.json().get("response", "").strip().capitalize()
    except Exception: return "Neutral"

# ── UI ────────────────────────────────────────────────────
model = st.selectbox("Model:", list(MODEL_MAP.keys()))
name_mode = st.radio("Entity name mode:", ["Official Names", "With Nicknames"], horizontal=True)

col1, col2 = st.columns(2)
templates_file = col1.file_uploader("Upload sentence_templates.csv", type=["csv"])
entities_file = col2.file_uploader("Upload entity.csv", type=["csv"])

nicknames_file = None
if name_mode == "With Nicknames":
    nicknames_file = st.file_uploader("Upload entity_nicknames.csv", type=["csv"])

if templates_file and entities_file:
    # Load files
    templates = pd.read_csv(templates_file)
    entities = pd.read_csv(entities_file)
    
    # Standardize column headers (1st col for Entities/Templates)
    entities.columns = ['Official_Name' if i == 1 else col for i, col in enumerate(entities.columns)]
    templates.columns = ['Template_Text' if i == 0 else col for i, col in enumerate(templates.columns)]

    if st.button("Run IC Analysis →"):
        # Validate Nickname input if required
        if name_mode == "With Nicknames" and nicknames_file is None:
            st.error("Please upload the entity_nicknames.csv file.")
            st.stop()
        
        nick_df = pd.read_csv(nicknames_file) if nicknames_file else None
        rows = []
        total = len(entities) * len(templates)
        progress_bar = st.progress(0, text="Starting Analysis...")
        current_step = 0
        
        for _, entity in entities.iterrows():
            official_name = entity['Official_Name']
            display_name = official_name
            
            # Nickname logic
            if name_mode == "With Nicknames" and nick_df is not None:
                match = nick_df[nick_df.iloc[:, 0] == official_name]
                if not match.empty:
                    display_name = str(match.iloc[0, 1])

            for _, tmpl in templates.iterrows():
                real_text = str(tmpl['Template_Text']).replace("[ENTITY]", str(display_name))
                prediction = get_sentiment(real_text, MODEL_MAP[model])
                
                rows.append({
                    "entity": display_name,
                    "template": tmpl['Template_Text'],
                    "predicted": prediction
                })
                current_step += 1
                progress_bar.progress(current_step / total, text=f"Processing {current_step}/{total}...")

        progress_bar.empty()
        results_df = pd.DataFrame(rows)
        
        # Entropy Calculation
        ic_list = []
        for tmpl in results_df['template'].unique():
            subset = results_df[results_df['template'] == tmpl]
            ic_list.append({'template': tmpl, 'ic_score': entropy(subset['predicted'].value_counts(normalize=True))})
        
        ic_df = pd.DataFrame(ic_list)
        
        # Display
        st.markdown(f"""
        <div class="info-card">
            <span class="card-label">Global IC Result</span>
            <div style="font-size: 32px; font-weight: 700; color: #3E2000;">{ic_df['ic_score'].mean():.4f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("IC Score per Template")
        st.bar_chart(ic_df.set_index('template'), color="#C4956A")
        st.dataframe(results_df, use_container_width=True)
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "ic_results.csv", "text/csv")