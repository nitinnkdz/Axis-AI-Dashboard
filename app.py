import streamlit as st
import pandas as pd
import requests
import os
import urllib3
import graphviz
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt  # <--- FIXED: Missing import added
import seaborn as sns
from datetime import datetime
from dotenv import load_dotenv

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Project Sentinel | Axis Bank", page_icon="💎")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. SECURE API HANDLING ---
load_dotenv()


def get_api_key():
    api_key = os.getenv("GROQ_API")
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
        height: 100%;
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

    .swot-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px;
        border: 1px solid rgba(128,128,128,0.2); background-color: var(--secondary-background-color);
    }
    .stPlotlyChart { background-color: transparent !important; }

    .stButton > button {
        background-color: #8B0000; color: white; border-radius: 6px;
        font-weight: 600; border: none; height: 45px; width: 100%;
    }
    .stButton > button:hover { background-color: #600000; }

    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #8B0000;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        padding: 10px; border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 4. GENAI ENGINE ---
def query_llama_model(prompt, system_role="You are a helpful banking analyst."):
    if not GROQ_API_KEY:
        return "⚠️ Simulation Mode: API Key missing. Check .env file."

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": "llama-3.1-8b-instant",
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
@st.cache_data(ttl=60)
def load_full_card_universe():
    try:
        import sqlite3
        conn = sqlite3.connect('sentinel_data.db')
        df = pd.read_sql("SELECT * FROM card_universe", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_rbi_market_data():
    try:
        import sqlite3
        conn = sqlite3.connect('sentinel_data.db')
        df = pd.read_sql("SELECT * FROM rbi_market_data", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
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


@st.cache_data(ttl=60)
def load_lending_offers():
    try:
        import sqlite3
        conn = sqlite3.connect('sentinel_data.db')
        df = pd.read_sql("SELECT * FROM lending_offers", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


from bs4 import BeautifulSoup

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_rbi_circulars():
    url = "https://rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx"
    try:
        # Using a standard user agent to avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main content table (this is specific to RBI's current layout)
        circulars = []
        
        # Searching for the typical structure: a table with class 'tablebg'
        table = soup.find('table', {'class': 'tablebg'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cols = row.find_all('td')
                if len(cols) >= 2:
                    date_text = cols[0].text.strip()
                    # Skip header rows or malformed dates
                    if "Date" in date_text or not date_text:
                        continue
                        
                    title_elem = cols[1].find('a')
                    if title_elem:
                        title = title_elem.text.strip()
                        href = title_elem.get('href', '')
                        if href.startswith('http'):
                            link = href
                        else:
                            link = "https://rbi.org.in/Scripts/" + href
                            
                        circulars.append({
                            "Date": date_text,
                            "Title": title,
                            "Category": "Press Release", # Generalizing category
                            "Link": link
                        })
                if len(circulars) >= 10: # Only get the top 10 recent
                    break
                    
        # Fallback if the table structure changed but we can find links
        if not circulars:
            links = soup.select('div#example-min a.link2')
            if not links:
               # Alternative selector commonly used by RBI
               links = soup.select('table.tablebg td a')
               
            for item in links[:10]:
                 title = item.text.strip()
                 if title:
                     link = item.get('href', '')
                     if not link.startswith('http'):
                         link = f"https://rbi.org.in/Scripts/{link}"
                     
                     # Extracting date from text if possible, else just use today
                     date_str = datetime.now().strftime("%b %d, %Y")
                     # Often date is in the parent tr's first td
                     parent_tr = item.find_parent('tr')
                     if parent_tr:
                         tds = parent_tr.find_all('td')
                         if tds and tds[0] != item.parent:
                             potential_date = tds[0].text.strip()
                             if len(potential_date) < 15: # basic check if it looks like a short date string
                                 date_str = potential_date
                                 
                     circulars.append({
                         "Date": date_str,
                         "Title": title,
                         "Category": "Update",
                         "Link": link
                     })

        if not circulars: # If scraper totally fails to find elements
            raise ValueError("Could not parse RBI page structure")
            
        return pd.DataFrame(circulars)
        
    except Exception as e:
        # Fallback to mock data so the app doesn't crash completely if RBI is down
        st.warning(f"Failed to fetch live RBI data: {str(e)}. Showing fallback data.")
        return pd.DataFrame([
            {"Date": "Jan 04, 2026", "Title": "Master Direction – Credit Card Directions (Updated) (Mock)",
             "Category": "Master Direction", "Link": "#"},
            {"Date": "Dec 15, 2025", "Title": "Fair Practices Code for Lenders - Unsecured Portfolio (Mock)",
             "Category": "Guideline", "Link": "#"}
        ])


def generate_ai_swot(card_name):
    if "Magnus" in card_name: return {"S": "High reward rate.", "W": "Devaluation hit trust.", "O": "Re-target HNI.",
                                      "T": "Amex Platinum."}
    return {"S": "Strong Ecosystem.", "W": "Generic Benefits.", "O": "Cross-sell.", "T": "Fintechs."}


def download_pdf_from_rbi(link): return b"PDF_CONTENT", "Success"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_credit_card_news():
    import xml.etree.ElementTree as ET
    url = "https://news.google.com/rss/search?q=indian+credit+cards&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        news_items = []
        for item in root.findall('.//item'):
            title = item.find('title').text if item.find('title') is not None else "No Title"
            link = item.find('link').text if item.find('link') is not None else "#"
            pub_date = item.find('pubDate').text if item.find('pubDate') is not None else "Unknown Date"
            news_items.append({"Title": title, "Link": link, "Date": pub_date})
            if len(news_items) >= 60: # Fetching a bit more than 50 just in case
                break
        return news_items
    except Exception as e:
        st.error(f"Failed to fetch news: {e}")
        return []

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
                       "🌍 Module 6: Geospatial Intel",
                       "📰 Module 8: Credit Card News"])

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
    st.dataframe(df_news, width="stretch")  # <--- FIXED: Deprecation
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
        st.dataframe(df_loan, width="stretch")  # <--- FIXED: Deprecation
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
    fig = px.scatter_map(geo_data, lat="Lat", lon="Lon", hover_name="City", size="Revenue_Cr", color="Dominance",
                         # <--- FIXED: Updated from scatter_mapbox
                         color_discrete_map={"Strong": "green", "Moderate": "orange", "Weak": "red"}, zoom=3.5,
                         height=500)
    fig.update_layout(map_style="open-street-map",
                      margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig, width="stretch")

elif module == "📰 Module 8: Credit Card News":
    st.title("📰 Live News: Indian Credit Card Market")
    st.markdown("Real-time feed aggregating the latest news regarding credit cards in India.")
    
    news_data = get_credit_card_news()
    
    if not news_data:
        st.warning("No news fetched. Please check your internet connection or try again later.")
    else:
        # Define CSS and HTML for the news grid
        full_html = """
        <style>
        .news-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
            padding: 10px 0;
            font-family: sans-serif;
        }
        .news-tile {
            background-color: #1e1e1e;
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-bottom: 4px solid #8B0000;
            border-radius: 8px;
            padding: 15px;
            height: 200px; /* Square effect */
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .news-tile:hover {
            transform: translateY(-4px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        }
        .news-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: #ffffff;
            line-height: 1.4;
            display: -webkit-box;
            -webkit-line-clamp: 4; /* Limit to 4 lines */
            -webkit-box-orient: vertical;
            overflow: hidden;
            text-decoration: none;
        }
        .news-date {
            font-size: 0.75rem;
            color: #aaaaaa;
            margin-top: 10px;
        }
        .news-link {
            text-decoration: none;
        }
        .news-link:hover .news-title {
            color: #d32f2f;
        }
        </style>
        <div class="news-grid">
        """
        
        for item in news_data:
            full_html += f"""
            <a href="{item['Link']}" target="_blank" class="news-link">
                <div class="news-tile">
                    <div class="news-title">{item['Title']}</div>
                    <div class="news-date">{item['Date']}</div>
                </div>
            </a>
            """
        full_html += '</div>'
        
        st.markdown(f"**Showing top {len(news_data)} recent articles.**")
        
        import streamlit.components.v1 as components
        # Calculate dynamic height based on items
        rows = (len(news_data) // 4) + 1  # Approximate 4 cols per row
        components.html(full_html, height=rows * 230, scrolling=False)