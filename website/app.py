import streamlit as st

st.set_page_config(page_title="TOSC Analysis Hub", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #F5F0E8; color: #3E2000; }

    /* Card-like button */
    div[data-testid="stButton"] > button {
        width: 100% !important;
        background: white !important;
        color: #3E2000 !important;
        border: 1.5px solid #D4B896 !important;
        border-radius: 0 10px 10px 0 !important;
        padding: 14px 18px !important;
        text-align: left !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        line-height: 1.5 !important;
        white-space: pre-wrap !important;
        height: 72px !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: #FAF3EA !important;
        border-color: #C4956A !important;
        color: #3E2000 !important;
    }
    div[data-testid="stButton"] > button:focus {
        box-shadow: none !important;
        color: #3E2000 !important;
    }

    /* Icon box left side */
    .icon-box {
        background: white;
        border: 1.5px solid #D4B896;
        border-right: none;
        border-radius: 10px 0 0 10px;
        height: 72px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 12px;
    }

    /* Remove gap between columns */
    div[data-testid="stColumns"] { gap: 0 !important; }
    div[data-testid="stColumn"] { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 40px 0 10px;">
  <p style="font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:#8B6A4A; margin-bottom:16px;">TOSC Research</p>
  <div style="display:flex; align-items:center; justify-content:center; gap:14px;">
    <span style="font-size:52px; font-weight:700; color:#3E2000; font-family:Georgia,serif; line-height:1;">NER</span>
    <span style="font-size:28px; color:#C4956A; font-family:Georgia,serif;">&amp;</span>
    <span style="font-size:52px; font-weight:700; color:#3E2000; font-family:Georgia,serif; line-height:1;">TSA</span>
  </div>
  <div style="margin-top:8px;">
    <span style="font-size:13px; font-weight:500; color:#8B5E3C; background:#EAD8C0; border-radius:4px; padding:3px 10px;">Let's Taglish!</span>
  </div>
  <div style="width:48px; height:2px; background:#C4956A; margin:20px auto; border-radius:2px;"></div>
  <p style="font-size:14px; color:#6B4C2A; max-width:480px; margin:0 auto 24px; line-height:1.7;">
    In pursuit for a project in Introduction to Artificial Intelligence, NER & TSA: Let's Taglish! explores the application of named entity recognition and sentiment analysis in the context of Taglish political discourse and model evaluation.
  </p>
  <div style="display:flex; gap:8px; justify-content:center; flex-wrap:wrap; margin-bottom:32px;">
    <span style="font-size:11px; color:#7A5230; background:#EDD9BE; border-radius:20px; padding:4px 12px; border:1px solid #D4B896;">Named Entity Recognition</span>
    <span style="font-size:11px; color:#7A5230; background:#EDD9BE; border-radius:20px; padding:4px 12px; border:1px solid #D4B896;">Sentiment Analysis</span>
    <span style="font-size:11px; color:#7A5230; background:#EDD9BE; border-radius:20px; padding:4px 12px; border:1px solid #D4B896;">Bias Detection</span>
  </div>
  <p style="font-size:11px; letter-spacing:0.1em; text-transform:uppercase; color:#A0845C; margin-bottom:16px;">Choose your module</p>
</div>
""", unsafe_allow_html=True)

_, col, _ = st.columns([1, 2, 1])

with col:
    modules = [
        {
            "icon": "https://cdn-icons-png.flaticon.com/64/2920/2920322.png",
            "title": "F1 Sentiment Evaluation",
            "desc": "Run precision, recall, and F1 scoring",
            "page": "pages/1_Sentiment_Evaluator.py",
            "key": "btn1"
        },
        {
            "icon": "https://cdn-icons-png.flaticon.com/64/6569/6569150.png",
            "title": "Inconsistency Metrics",
            "desc": "Analyze labeling and model inconsistencies",
            "page": "pages/2_IC_Metrics.py",
            "key": "btn2"
        },
        {
            "icon": "https://cdn-icons-png.flaticon.com/64/3132/3132680.png",
            "title": "Polarity Axes",
            "desc": "Examine positive–negative sentiment axes",
            "page": "pages/3_Polarity_Analysis.py",
            "key": "btn3"
        },
    ]

    for m in modules:
        st.markdown('<div style="margin-bottom:10px;">', unsafe_allow_html=True)
        icon_col, btn_col = st.columns([1, 5])

        with icon_col:
            st.markdown(f"""
            <div class="icon-box">
                <img src="{m['icon']}" width="32" height="32" style="object-fit:contain;"
                     onerror="this.src='https://cdn-icons-png.flaticon.com/64/1828/1828843.png'" />
            </div>
            """, unsafe_allow_html=True)

        with btn_col:
            if st.button(f"{m['title']}\n{m['desc']}", key=m["key"], use_container_width=True):
                st.switch_page(m["page"])

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; margin-top:32px; padding-bottom:24px;">
  <p style="font-size:11px; color:#A08060; letter-spacing:0.04em;">Entity-targeted bias detection · Political NLP · Taglish corpus</p>
</div>
""", unsafe_allow_html=True)