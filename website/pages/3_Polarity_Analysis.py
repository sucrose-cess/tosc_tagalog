import streamlit as st
import pandas as pd
import requests

# ── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="TOSC Analysis Hub", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F5F0E8; color: #3E2000; }
    .page-eyebrow { font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase; color: #8B6A4A; margin-bottom: 8px; }
    .page-title { font-size: 36px; font-weight: 700; color: #3E2000; font-family: Georgia, serif; margin-bottom: 4px; }
    .page-divider { width: 48px; height: 2px; background: #C4956A; margin: 16px 0 24px; border-radius: 2px; }
    .info-card { background: white; border: 1.5px solid #D4B896; border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; }
    .card-label { font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: #A0845C; margin-bottom: 10px; display: block; }
    .badge { font-size: 11px; color: #7A5230; background: #EDD9BE; border-radius: 20px; padding: 4px 12px; border: 1px solid #D4B896; display: inline-block; margin-right: 6px; margin-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 32px 0 8px;">
  <p class="page-eyebrow">TOSC Research · Module 3</p>
  <p class="page-title">Polarity Axes</p>
  <div class="page-divider"></div>
  <div>
    <span class="badge">Political Order</span>
    <span class="badge">Social Orientation</span>
    <span class="badge">Temporal Orientation</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_MAP = {"Gemma 2.2B": "gemma2:2b", "Qwen 2.5 3B": "qwen2.5:3b", "Llama 3.2 3B": "llama3.2:3b"}

def get_sentiment(text, model_tag):
    prompt = f"Classify sentiment of this Taglish text as Positive, Negative, or Neutral. Reply with one word only: {text}"
    try:
        res = requests.post(OLLAMA_URL, json={"model": model_tag, "prompt": prompt, "stream": False}, timeout=60)
        return res.json().get("response", "").strip().capitalize()
    except: return "Neutral"

# ── UI ────────────────────────────────────────────────────
model_name = st.selectbox("Model:", list(MODEL_MAP.keys()))
model_tag = MODEL_MAP[model_name]
name_mode = st.radio("Entity name mode:", ["Official Names", "With Nicknames"], horizontal=True)

col1, col2, col3 = st.columns(3)
templates_file = col1.file_uploader("Upload templates.csv", type=["csv"])
entities_file = col2.file_uploader("Upload entity.csv", type=["csv"])
nicknames_file = col3.file_uploader("Upload entity_nicknames.csv", type=["csv"]) if name_mode == "With Nicknames" else None

if st.button("Run Full Analysis Pipeline →"):
    if not templates_file or not entities_file or (name_mode == "With Nicknames" and not nicknames_file):
        st.error("Please upload all required files.")
        st.stop()

    templates = pd.read_csv(templates_file)
    entities = pd.read_csv(entities_file)
    
    # LOAD NICKNAMES ONCE HERE, BEFORE THE LOOP
    nicks_df = pd.read_csv(nicknames_file) if name_mode == "With Nicknames" else None
    
    results = []
    with st.spinner("Processing generation and metrics..."):
        for _, entity_row in entities.iterrows():
            eid, ename = entity_row['Entity_ID'], entity_row['Official name']
            
            if name_mode == "With Nicknames":
                display_names = nicks_df[nicks_df['Entity_ID'] == eid]['Nickname'].tolist()
            else:
                display_names = [ename]
            
            for name in display_names:
                for _, tmpl in templates.iterrows():
                    real_text = tmpl['template_text'].replace('[ENTITY]', str(name).strip())
                    pred = get_sentiment(real_text, model_tag)
                    
                    results.append({
                        'template_id': tmpl['template_id'],
                        'official_name': ename,
                        'name_used': str(name).strip(),
                        'order_axis': entity_row['Order_Axis'],
                        'social_axis': entity_row['Social_Axis'],
                        'temporal_axis': entity_row['Temporal_Axis'],
                        'predicted': pred
                    })

    df = pd.DataFrame(results)
    df['numeric'] = df['predicted'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    
    st.success("Analysis Complete!")
    
    # ── Global Axis Metrics ──
    st.markdown("### Global Results")
    col_g1, _ = st.columns(2)
    
    with col_g1:
        st.markdown(f"""
        <div class="info-card">
            <span class="card-label">Global Mean Polarity</span>
            <div style="font-size: 32px; font-weight: 700; color: #3E2000;">{df['numeric'].mean():.4f}</div>
        </div>
        """, unsafe_allow_html=True)

    summary_data = []
    for axis in ['order_axis', 'social_axis', 'temporal_axis']:
        axis_summary = df.groupby(axis)['numeric'].agg(['mean', 'count']).reset_index()
        axis_summary.columns = ['Level', 'Mean_Polarity', 'Count']
        axis_summary['Axis'] = axis.replace('_axis', '').replace('_', ' ').title()
        summary_data.append(axis_summary)
    
    summary_df = pd.concat(summary_data)
    st.markdown("#### Mean Polarity by Axis Level")
    st.dataframe(summary_df, use_container_width=True)
    
    st.download_button("Download All Results", df.to_csv(index=False), "full_analysis.csv")