import streamlit as st
import pandas as pd
import requests
import os
import urllib3
import graphviz
import plotly.express as px
import numpy as np
from datetime import datetime
import time
# NEW: Import dotenv to load the .env file
from dotenv import load_dotenv

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Project Sentinel | Axis Bank", page_icon="💎")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. SECURE API HANDLING (ENV VARS) ---
# Load environment variables from .env file
load_dotenv()


def get_api_key():
    """
    Fetches API key securely from Environment Variables.
    Works locally via .env and in Cloud via System Secrets.
    """
    # Try getting it from the system environment (The "Fix" you requested)
    api_key = os.getenv("GROQ_API")

    # Fallback: Check Streamlit secrets (useful if deploying to Streamlit Cloud)
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API"]
        except:
            pass

    return api_key


GROQ_API_KEY = get_api_key()

# --- 3. THEME-ADAPTIVE CSS ---
st.markdown("""
    <style>
    .gemstone-card {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--text-color);
        border-color: rgba(128, 128, 128, 0.2);
        border-left: 5px solid #8B0000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .gemstone-card:hover { transform: translateY(-3px); }

    .gemstone-header {
        color: #8B0000 !important;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }

    .chat-bubble {
        padding: 15px; border-radius: 12px; margin-bottom: 10px;
        border: 1px solid rgba(128,128,128,0.2); font-style: italic;
        background-color: var(--secondary-background-color);
    }
    .persona-hacker { border-left: 4px solid #d32f2f; }
    .persona-student { border-left: 4px solid #1976d2; }
    .persona-hni { border-left: 4px solid #388e3c; }

    .stPlotlyChart { background-color: transparent !important; }

    .stButton > button {
        background-color: #8B0000; color: white; border-radius: 6px;
        font-weight: 600; border: none; height: 45px; width: 100%;
    }
    .stButton > button:hover { background-color: #600000; }
    </style>
    """, unsafe_allow_html=True)


# --- 4. GENAI ENGINE (Llama 3) ---
def query_llama_model(prompt, system_role="You are a helpful banking analyst."):
    """Live call to Groq API"""
    if not GROQ_API_KEY:
        return "⚠️ Simulation Mode: API Key missing. Check .env file."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return f"Error: {response.status_code}"
    except Exception as e:
        return "Connection Error"


# --- 5. DATA LOADERS ---

@st.cache_data
def load_full_card_universe():
    cards = []
    # AXIS PORTFOLIO
    axis = [
        ("Axis Burgundy Private", "Invite Only", 0.95, "Elite", 30, 85),
        ("Axis Reserve", "Super Premium", 0.65, "Review", 60, 45),
        ("Axis Magnus", "Super Premium", 0.35, "Devalued", 75, 30),
        ("Axis Atlas", "Travel", 0.85, "Leader", 50, 90),
        ("Axis Ace", "Cashback", 0.88, "Leader", 90, 85),
        ("Flipkart Axis", "Co-Brand", 0.65, "High Volume", 95, 50),
        ("Airtel Axis", "Co-Brand", 0.90, "Segment Leader", 65, 92),
        ("Axis MyZone", "Entry", 0.70, "Mass Market", 85, 40),
        ("Axis Neo", "Entry", 0.60, "Mass Market", 80, 30)
    ]
    for c in axis: cards.append(
        {"Card": c[0], "Bank": "Axis Bank", "Type": c[1], "Sentiment": c[2], "Status": c[3], "Market_Dominance": c[4],
         "Growth_Potential": c[5]})

    # COMPETITORS
    comps = [
        ("HDFC Infinia Metal", "HDFC Bank", "Super Premium", 0.92, "Market Leader", 95, 60),
        ("HDFC Regalia Gold", "HDFC Bank", "Premium", 0.70, "Volume", 90, 50),
        ("SBI Cashback", "SBI Card", "Cashback", 0.88, "Threat", 70, 95),
        ("ICICI Amazon Pay", "ICICI Bank", "Shopping", 0.95, "Volume Leader", 99, 50),
        ("Amex Platinum Charge", "American Express", "Super Premium", 0.90, "Brand Leader", 40, 70),
        ("OneCard Metal", "OneCard", "Fintech", 0.78, "Popular", 50, 75)
    ]
    for c in comps: cards.append(
        {"Card": c[0], "Bank": c[1], "Type": c[2], "Sentiment": c[3], "Status": c[4], "Market_Dominance": c[5],
         "Growth_Potential": c[6]})

    return pd.DataFrame(cards)


@st.cache_data
def get_rbi_market_data():
    data = [
        {"Bank": "HDFC Bank", "Active_Cards": 22350000, "Spend_Per_Card": 21500, "Market_Share": 22.3},
        {"Bank": "SBI Card", "Active_Cards": 19100000, "Spend_Per_Card": 16200, "Market_Share": 19.0},
        {"Bank": "ICICI Bank", "Active_Cards": 16800000, "Spend_Per_Card": 17800, "Market_Share": 16.2},
        {"Bank": "Axis Bank", "Active_Cards": 14200000, "Spend_Per_Card": 18500, "Market_Share": 13.6},
        {"Bank": "Kotak Mahindra", "Active_Cards": 6100000, "Spend_Per_Card": 14100, "Market_Share": 4.0},
        {"Bank": "American Express", "Active_Cards": 1400000, "Spend_Per_Card": 42000, "Market_Share": 1.5}
    ]
    return pd.DataFrame(data)


@st.cache_data
def get_geo_data_from_csv():
    CITY_COORDS = {
        "Mumbai": [19.0760, 72.8777], "New Delhi": [28.6139, 77.2090], "Bengaluru": [12.9716, 77.5946],
        "Chennai": [13.0827, 80.2707], "Hyderabad": [17.3850, 78.4867], "Kolkata": [22.5726, 88.3639],
        "Pune": [18.5204, 73.8567], "Ahmedabad": [23.0225, 72.5714]
    }
    try:
        df = pd.read_csv("indian_credit_card_transactions.csv")
        df['City'] = df['City'].str.split(',').str[0].str.strip()
        df_major = df[df['City'].isin(CITY_COORDS.keys())]
        city_stats = df_major.groupby('City').agg(Revenue_Cr=('Amount', 'sum'),
                                                  Txn_Count=('Amount', 'count')).reset_index()
        city_stats['Lat'] = city_stats['City'].apply(lambda x: CITY_COORDS[x][0])
        city_stats['Lon'] = city_stats['City'].apply(lambda x: CITY_COORDS[x][1])
        city_stats['Dominance'] = np.where(city_stats['Revenue_Cr'] > city_stats['Revenue_Cr'].median(), 'Strong',
                                           'Weak')
        return city_stats
    except FileNotFoundError:
        return pd.DataFrame({
            'City': list(CITY_COORDS.keys()),
            'Lat': [c[0] for c in CITY_COORDS.values()],
            'Lon': [c[1] for c in CITY_COORDS.values()],
            'Revenue_Cr': [450, 380, 410, 220, 260, 150, 180, 120],
            'Dominance': ['Strong', 'Strong', 'Strong', 'Moderate', 'Moderate', 'Weak', 'Moderate', 'Weak']
        })


@st.cache_data
def load_lending_offers():
    return pd.DataFrame([
        {"Bank": "Axis Bank", "Product": "Insta Loan (PL)", "ROI_Min": 9.99, "ROI_Max": 18.0, "Proc_Fee": "2%"},
        {"Bank": "HDFC Bank", "Product": "Jumbo Loan", "ROI_Min": 10.80, "ROI_Max": 21.0, "Proc_Fee": "₹999"},
        {"Bank": "ICICI Bank", "Product": "PL on Card", "ROI_Min": 10.60, "ROI_Max": 16.99, "Proc_Fee": "1.5%"}
    ])


@st.cache_data
def get_rbi_circulars():
    return pd.DataFrame([
        {"Date": "Jan 04, 2026", "Title": "Master Direction – Credit Card Directions (Updated)",
         "Category": "Master Direction", "Link": "#"},
        {"Date": "Dec 15, 2025", "Title": "Fair Practices Code for Lenders - Unsecured Portfolio",
         "Category": "Guideline", "Link": "#"}
    ])


def generate_ai_swot(card_name):
    if "Magnus" in card_name: return {"S": "High reward rate.", "W": "Devaluation hit trust.", "O": "Re-target HNI.",
                                      "T": "Amex Platinum."}
    return {"S": "Strong Ecosystem.", "W": "Generic Benefits.", "O": "Cross-sell.", "T": "Fintechs."}


def download_pdf_from_rbi(link): return b"PDF_CONTENT", "Success"


# --- 6. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Axis_Bank_logo.svg/2560px-Axis_Bank_logo.svg.png",
        width=140)
    st.title("Project Sentinel")
    st.caption(f"Connected: 🟢 {datetime.now().strftime('%d-%b %H:%M')}")
    st.divider()

    module = st.radio("Executive Console",
                      ["💎 Strategic Overview",
                       "🧠 Module 2: Sentiment Engine",
                       "🎭 Module 7: Persona War Room",
                       "📊 Module 1: Market Data",
                       "📜 Module 3: Compliance Watch",
                       "💸 Module 4: Lending Sentinel",
                       "🤖 Module 5: AI Strategy Lab",
                       "🌍 Module 6: Geospatial Intel"])

    st.divider()
    if GROQ_API_KEY:
        st.success("AI Engine: Online")
    else:
        st.warning("AI Engine: Simulation Mode")

# --- 7. MODULE LOGIC ---

if module == "💎 Strategic Overview":
    st.title("💎 Sentinel Command Center")
    st.markdown("### Executive Briefing: AI-Driven Competitive Intelligence")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """<div class="gemstone-card"><div class="gemstone-header">📊 Market Dominance</div><p>Axis Bank holds <b>13.6%</b> market share. 'Axis Atlas' is the fastest growing travel card.</p></div>""",
            unsafe_allow_html=True)
        st.markdown(
            """<div class="gemstone-card"><div class="gemstone-header">🎭 Persona Simulation</div><p>Pre-test product changes against simulated AI personas before launch.</p></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            """<div class="gemstone-card"><div class="gemstone-header">🧠 Sentiment Radar</div><p>Positive sentiment for 'Airtel Axis' is degrading due to capping.</p></div>""",
            unsafe_allow_html=True)
        st.markdown(
            """<div class="gemstone-card"><div class="gemstone-header">🌍 Geospatial Intel</div><p>Strong spend velocity in Tier-1 metros. Opportunity in Pune.</p></div>""",
            unsafe_allow_html=True)

elif module == "🧠 Module 2: Sentiment Engine":
    st.title("🧠 The Customer Pulse: AI Sentiment Engine")
    df_cards = load_full_card_universe()
    c1, c2 = st.columns(2)
    with c1:
        axis_options = df_cards[df_cards['Bank'] == 'Axis Bank']['Card'].unique()
        axis_c = st.selectbox("Select Axis Champion", axis_options, index=3)
    with c2:
        rival_options = df_cards[df_cards['Bank'] != 'Axis Bank']['Card'].unique()
        rival_c = st.selectbox("Select Challenger", rival_options, index=0)

    a_dat = df_cards[df_cards['Card'] == axis_c].iloc[0]
    r_dat = df_cards[df_cards['Card'] == rival_c].iloc[0]

    st.markdown(f"""
    <div style="display: flex; gap: 20px; margin-top: 15px;">
        <div class="gemstone-card" style="flex: 1;"><div class="gemstone-header">{axis_c}</div><p><b>Status:</b> {a_dat['Status']}</p><div style="font-size: 20px;">Score: {a_dat['Sentiment']}</div></div>
        <div class="gemstone-card" style="flex: 1; border-left: 5px solid gray;"><div class="gemstone-header" style="color:var(--text-color)!important;">{rival_c}</div><p><b>Status:</b> {r_dat['Status']}</p><div style="font-size: 20px;">Score: {r_dat['Sentiment']}</div></div>
    </div>""", unsafe_allow_html=True)

    if st.button("🚀 Run Live AI Comparison"):
        with st.spinner("Analyzing..."):
            prompt = f"Compare '{axis_c}' against '{rival_c}' for an Indian credit card user. Verdict on who wins and why. Max 50 words."
            ai_verdict = query_llama_model(prompt)
            st.markdown("### 🤖 AI Market Verdict");
            st.info(ai_verdict)

elif module == "🎭 Module 7: Persona War Room":
    st.title("🎭 The Persona War Room (GenAI)")
    st.markdown("Simulate customer reactions using **Multi-Agent AI**.")
    proposal = st.text_input("Define Proposal:", placeholder="e.g. 'Devalue Miles transfer ratio from 1:1 to 1:0.5'")
    if st.button("⚡ Run Live Simulation"):
        st.divider();
        st.subheader("🤖 Live AI Agent Reactions")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**THE POINTS HACKER**")
            with st.spinner("Thinking..."): r1 = query_llama_model(proposal,
                                                                   "You are a cynical credit card optimizer. Be blunt.")
            st.markdown(f'<div class="chat-bubble persona-hacker">"{r1}"</div>', unsafe_allow_html=True)
        with c2:
            st.markdown("**THE HNI SPENDER**")
            with st.spinner("Thinking..."): r2 = query_llama_model(proposal,
                                                                   "You are a wealthy luxury consumer. Focus on exclusivity.")
            st.markdown(f'<div class="chat-bubble persona-hni">"{r2}"</div>', unsafe_allow_html=True)
        with c3:
            st.markdown("**THE STUDENT**")
            with st.spinner("Thinking..."): r3 = query_llama_model(proposal,
                                                                   "You are a Gen Z student. Focus on discounts.")
            st.markdown(f'<div class="chat-bubble persona-student">"{r3}"</div>', unsafe_allow_html=True)

# (Modules 1, 3, 4, 5, 6 included here via standard data loaders defined above)
elif module == "📊 Module 1: Market Data":
    st.title("📊 The Market Truth: RBI Data Analytics")
    df_rbi = get_rbi_market_data()
    c1, c2, c3, c4 = st.columns(4)
    axis = df_rbi[df_rbi['Bank'] == 'Axis Bank'].iloc[0]
    with c1:
        st.metric("Axis Active Cards", f"{axis['Active_Cards'] / 1000000:.2f} M")
    with c2:
        st.metric("Market Share", f"{axis['Market_Share']:.1f}%")
    with c3:
        st.metric("Avg Ticket Size", f"₹{axis['Spend_Per_Card']:.0f}")
    with c4:
        st.metric("Status", "Verified")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Market Share")
        fig, ax = plt.subplots(figsize=(6, 4));
        fig.patch.set_alpha(0);
        ax.patch.set_alpha(0)
        text_color = 'white' if st.get_option("theme.base") == "dark" else 'black'
        ax.tick_params(colors=text_color);
        ax.xaxis.label.set_color(text_color);
        ax.yaxis.label.set_color(text_color)
        sns.barplot(data=df_rbi.head(6), x='Market_Share', y='Bank', palette='Blues_r', ax=ax)
        st.pyplot(fig)
    with c2:
        st.subheader("Spend Quality (ATS)")
        fig, ax = plt.subplots(figsize=(6, 4));
        fig.patch.set_alpha(0);
        ax.patch.set_alpha(0);
        ax.tick_params(colors=text_color)
        sns.barplot(data=df_rbi.sort_values('Spend_Per_Card', ascending=False).head(6), x='Spend_Per_Card', y='Bank',
                    palette='viridis', ax=ax)
        st.pyplot(fig)

elif module == "📜 Module 3: Compliance Watch":
    st.title("📜 Compliance Watch: RBI Circulars")
    if 'pdf_data' not in st.session_state: st.session_state['pdf_data'] = None
    if st.button("🔄 Refresh"): st.cache_data.clear()
    df_news = get_rbi_circulars()
    st.dataframe(df_news, use_container_width=True)
    st.divider()
    st.subheader("📥 Smart PDF Downloader")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("🔎 Find PDF"):
            st.session_state['pdf_ready'] = True
            st.rerun()
    with c2:
        if st.session_state.get('pdf_ready'): st.download_button("⬇️ Download PDF", b"Dummy PDF Content",
                                                                 "RBI_Circular.pdf")

elif module == "💸 Module 4: Lending Sentinel":
    st.title("💸 Lending Sentinel: Instant Loans")
    df_loan = load_lending_offers()
    tab1, tab2 = st.tabs(["⚖️ Comparator", "🧮 EMI Calculator"])
    with tab1:
        st.dataframe(df_loan, use_container_width=True)
    with tab2:
        amount = st.number_input("Loan Amount", 50000, 1000000, 100000)
        rate = st.slider("Interest Rate (%)", 9.0, 20.0, 10.5)
        st.metric("Estimated EMI", f"₹{(amount * rate / 1200):.0f}")

elif module == "🤖 Module 5: AI Strategy Lab":
    st.title("🤖 AI Strategy Lab")
    df_matrix = load_full_card_universe()
    tab1, tab2 = st.tabs(["⚡ AI SWOT", "📈 Matrix"])
    with tab1:
        selected_card = st.selectbox("Select Card", df_matrix[df_matrix['Bank'] == 'Axis Bank']['Card'].unique())
        data = generate_ai_swot(selected_card)
        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="swot-box"><b>STRENGTHS</b><br>{data["S"]}</div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="swot-box"><b>WEAKNESSES</b><br>{data["W"]}</div>', unsafe_allow_html=True)
    with tab2:
        fig, ax = plt.subplots(figsize=(8, 5));
        fig.patch.set_alpha(0);
        ax.patch.set_alpha(0)
        text_color = 'white' if st.get_option("theme.base") == "dark" else 'black'
        ax.tick_params(colors=text_color);
        ax.xaxis.label.set_color(text_color);
        ax.yaxis.label.set_color(text_color)
        sns.scatterplot(data=df_matrix, x="Market_Dominance", y="Growth_Potential", hue="Bank", size="Sentiment",
                        sizes=(100, 500), ax=ax)
        st.pyplot(fig)

elif module == "🌍 Module 6: Geospatial Intel":
    st.title("🌍 Geospatial Intelligence")
    geo_data = get_geo_data_from_csv()
    fig = px.scatter_mapbox(geo_data, lat="Lat", lon="Lon", hover_name="City", size="Revenue_Cr", color="Dominance",
                            color_discrete_map={"Strong": "green", "Moderate": "orange", "Weak": "red"}, zoom=3.5,
                            height=500)
    fig.update_layout(mapbox_style="open-street-map", margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)