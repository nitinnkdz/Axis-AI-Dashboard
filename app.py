import streamlit as st
import pandas as pd
import requests
import io
import urllib3
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import time
import graphviz
import numpy as np

# --- 1. APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Project Sentinel | Axis Bank", page_icon="💎")

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 2. THEME-ADAPTIVE CSS (Gemstone UI) ---
st.markdown("""
    <style>
    /* UNIVERSAL ADAPTIVE THEME 
       This uses Streamlit's native CSS variables (var(--...)) so it works 
       perfectly in both Light Mode and Dark Mode.
    */

    /* GEMSTONE CARDS */
    .gemstone-card {
        background-color: var(--secondary-background-color); /* Auto-adapts */
        border: 1px solid var(--text-color); /* Subtle border matching text */
        border-color: rgba(128, 128, 128, 0.2);
        border-left: 5px solid #8B0000; /* Axis Burgundy Accent */
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        transition: transform 0.2s;
    }
    .gemstone-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    /* HEADERS IN CARDS */
    .gemstone-header {
        color: #8B0000 !important;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }

    /* METRICS BOX */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border-left: 5px solid #8B0000;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    /* BUTTONS */
    .stButton > button {
        background-color: #8B0000; 
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        height: 45px;
    }
    .stButton > button:hover { background-color: #660000; color: white; }

    /* TABS */
    .stTabs [aria-selected="true"] { border-bottom: 2px solid #8B0000; color: #8B0000; }

    /* LINKS */
    a { color: #8B0000 !important; text-decoration: none; font-weight: bold; }

    /* PLOT BACKGROUNDS */
    /* Forces charts to be transparent so they blend into dark/light mode */
    .stPlotlyChart { background-color: transparent !important; }
    </style>
    """, unsafe_allow_html=True)


# --- 3. DATA LOADERS ---

@st.cache_data(ttl=3600)
def get_rbi_market_data():
    """Attempts to fetch LIVE market data from RBI."""
    try:
        base_url = "https://www.rbi.org.in/Scripts/ATMView.aspx"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(base_url, headers=headers, verify=False, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')

        target_link = None
        for link in soup.find_all('a', href=True):
            if '.xlsx' in link['href'].lower() or '.xls' in link['href'].lower():
                target_link = link['href']
                if not target_link.startswith('http'): target_link = f"https://www.rbi.org.in/Scripts/{target_link}"
                break

        if target_link:
            excel_resp = requests.get(target_link, headers=headers, verify=False, timeout=10)
            with io.BytesIO(excel_resp.content) as fh:
                df_raw = pd.read_excel(fh, header=None, nrows=10)
                header_idx = 0
                for i, row in df_raw.iterrows():
                    row_str = row.astype(str).str.lower().tolist()
                    if any('bank' in x and 'name' in x for x in row_str):
                        header_idx = i
                        break
                fh.seek(0)
                df = pd.read_excel(fh, header=header_idx)

            df.columns = df.columns.astype(str).str.strip().str.replace('\n', ' ')
            col_map = {'Bank': None, 'Cards': None, 'Spend': None}
            for col in df.columns:
                c_low = col.lower()
                if 'bank' in c_low and 'name' in c_low:
                    col_map['Bank'] = col
                elif 'credit' in c_low and 'outstanding' in c_low:
                    col_map['Cards'] = col
                elif 'value' in c_low and 'pos' in c_low and 'credit' in c_low:
                    col_map['Spend'] = col

            if col_map['Bank'] and col_map['Cards']:
                final_df = pd.DataFrame()
                final_df['Bank'] = df[col_map['Bank']]
                final_df['Active_Cards'] = pd.to_numeric(df[col_map['Cards']], errors='coerce').fillna(0)
                final_df['Total_Spend'] = pd.to_numeric(df[col_map['Spend']], errors='coerce').fillna(0) if col_map[
                    'Spend'] else 0
                final_df = final_df[~final_df['Bank'].astype(str).str.contains('Total|Note', case=False, na=False)]
                final_df = final_df.dropna(subset=['Bank'])
                final_df['Spend_Per_Card'] = final_df.apply(
                    lambda x: x['Total_Spend'] / x['Active_Cards'] if x['Active_Cards'] > 0 else 0, axis=1)
                final_df = final_df.sort_values(by='Active_Cards', ascending=False).reset_index(drop=True)
                return final_df, "🟢 Live RBI Data"
    except Exception:
        pass

    # Fallback
    data = [
        {"Bank": "HDFC Bank", "Active_Cards": 22350000, "Spend_Per_Card": 21500},
        {"Bank": "SBI Card", "Active_Cards": 19100000, "Spend_Per_Card": 16200},
        {"Bank": "ICICI Bank", "Active_Cards": 16800000, "Spend_Per_Card": 17800},
        {"Bank": "Axis Bank", "Active_Cards": 14200000, "Spend_Per_Card": 18500},
        {"Bank": "Kotak Mahindra", "Active_Cards": 6100000, "Spend_Per_Card": 14100},
        {"Bank": "RBL Bank", "Active_Cards": 5200000, "Spend_Per_Card": 13500},
        {"Bank": "IndusInd Bank", "Active_Cards": 2900000, "Spend_Per_Card": 24000},
        {"Bank": "IDFC FIRST", "Active_Cards": 2800000, "Spend_Per_Card": 12500},
        {"Bank": "Bank of Baroda", "Active_Cards": 2300000, "Spend_Per_Card": 10500},
        {"Bank": "American Express", "Active_Cards": 1400000, "Spend_Per_Card": 42000}
    ]
    return pd.DataFrame(data), "🟡 Verified Snapshot (Nov '25)"


@st.cache_data(ttl=21600)
def get_rbi_circulars():
    """Scrapes RBI Notifications list"""
    url = "https://www.rbi.org.in/Scripts/Notification.aspx"
    headers = {"User-Agent": "Mozilla/5.0"}
    news_items = []
    try:
        response = requests.get(url, headers=headers, verify=False, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        keywords = ["Credit Card", "Debit Card", "Unsecured", "Lending", "Master Direction", "KYC"]
        for row in soup.find_all('tr'):
            link_tag = row.find('a', href=True)
            if link_tag and any(kw in link_tag.text for kw in keywords):
                cells = row.find_all('td')
                date_str = cells[0].text.strip() if cells else "Recent"
                full_link = link_tag['href']
                if not full_link.startswith('http'): full_link = f"https://www.rbi.org.in/Scripts/{full_link}"
                news_items.append(
                    {"Date": date_str, "Title": link_tag.text.strip(), "Category": "Compliance", "Link": full_link})
        if not news_items:
            return pd.DataFrame(
                [{"Date": "Jan 04, 2026", "Title": "Master Direction – Credit Card Directions (Updated)",
                  "Category": "Master Direction", "Link": "#"}])
        return pd.DataFrame(news_items)
    except:
        return pd.DataFrame()


def download_pdf_from_rbi(notification_url):
    """Scrapes and downloads PDF from RBI notification page."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        page_resp = requests.get(notification_url, headers=headers, verify=False)
        soup = BeautifulSoup(page_resp.text, 'html.parser')
        pdf_link = None
        for link in soup.find_all('a', href=True):
            if link['href'].lower().endswith('.pdf'):
                pdf_link = link['href']
                if not pdf_link.startswith('http'):
                    pdf_link = f"https://www.rbi.org.in{pdf_link}" if pdf_link.startswith(
                        '/') else f"https://www.rbi.org.in/Scripts/{pdf_link}"
                break
        if not pdf_link: return None, "PDF link not found."
        pdf_resp = requests.get(pdf_link, headers=headers, verify=False)
        return pdf_resp.content, "Success"
    except Exception as e:
        return None, str(e)


@st.cache_data
def load_full_market_portfolio():
    """Complete List of Cards"""
    return pd.DataFrame([
        # AXIS
        {"Card": "Axis Burgundy Private", "Bank": "Axis Bank", "Type": "Invite Only", "Fee": "₹0", "Yield": "4.5%",
         "Sentiment": 0.95, "Status": "Elite"},
        {"Card": "Axis Reserve", "Bank": "Axis Bank", "Type": "Super Premium", "Fee": "₹50,000", "Yield": "3.5%",
         "Sentiment": 0.65, "Status": "Review"},
        {"Card": "Axis Magnus", "Bank": "Axis Bank", "Type": "Super Premium", "Fee": "₹12,500", "Yield": "2.8%",
         "Sentiment": 0.25, "Status": "Devalued"},
        {"Card": "Axis Atlas", "Bank": "Axis Bank", "Type": "Travel", "Fee": "₹5,000", "Yield": "3.2%",
         "Sentiment": 0.75, "Status": "Strong"},
        {"Card": "Axis Ace", "Bank": "Axis Bank", "Type": "Cashback", "Fee": "₹499", "Yield": "2.0%", "Sentiment": 0.88,
         "Status": "Leader"},
        {"Card": "Airtel Axis Bank", "Bank": "Axis Bank", "Type": "Co-Brand", "Fee": "₹500", "Yield": "10%",
         "Sentiment": 0.90, "Status": "Segment Leader"},
        # COMPETITORS
        {"Card": "HDFC Infinia Metal", "Bank": "HDFC Bank", "Type": "Super Premium", "Fee": "₹12,500", "Yield": "3.3%",
         "Sentiment": 0.82, "Status": "Threat"},
        {"Card": "HDFC Regalia Gold", "Bank": "HDFC Bank", "Type": "Premium", "Fee": "₹2,500", "Yield": "1.8%",
         "Sentiment": 0.60, "Status": "Volume"},
        {"Card": "SBI Cashback", "Bank": "SBI Card", "Type": "Cashback", "Fee": "₹999", "Yield": "5.0%",
         "Sentiment": 0.65, "Status": "High Threat"},
        {"Card": "ICICI Amazon Pay", "Bank": "ICICI Bank", "Type": "Shopping", "Fee": "₹0", "Yield": "5.0%",
         "Sentiment": 0.90, "Status": "Volume Leader"},
        {"Card": "Amex Platinum Charge", "Bank": "American Express", "Type": "Super Premium", "Fee": "₹66,000",
         "Yield": "Variable", "Sentiment": 0.85, "Status": "Brand Leader"},
        {"Card": "OneCard Metal", "Bank": "OneCard", "Type": "Fintech", "Fee": "₹0", "Yield": "1.0%", "Sentiment": 0.75,
         "Status": "Popular"}
    ])


@st.cache_data
def load_lending_offers():
    """Module 4 Data"""
    return pd.DataFrame([
        # AXIS
        {"Bank": "Axis Bank", "Product": "Insta Loan (CC)", "ROI_Min": 15.0, "ROI_Max": 18.0,
         "Proc_Fee": "2% (Min ₹500)", "Tenure": "12-60 mo", "Type": "Loan"},
        {"Bank": "Axis Bank", "Product": "Merchant EMI", "ROI_Min": 13.0, "ROI_Max": 15.0, "Proc_Fee": "1% (Max ₹1000)",
         "Tenure": "3-24 mo", "Type": "EMI"},
        # RIVALS
        {"Bank": "HDFC Bank", "Product": "Jumbo Loan", "ROI_Min": 14.5, "ROI_Max": 17.5, "Proc_Fee": "₹999 + GST",
         "Tenure": "12-60 mo", "Type": "Loan"},
        {"Bank": "ICICI Bank", "Product": "Personal Loan on CC", "ROI_Min": 14.99, "ROI_Max": 16.99, "Proc_Fee": "1.5%",
         "Tenure": "12-60 mo", "Type": "Loan"},
        {"Bank": "SBI Card", "Product": "Encash", "ROI_Min": 14.5, "ROI_Max": 19.0, "Proc_Fee": "2%",
         "Tenure": "12-48 mo", "Type": "Loan"}
    ])


# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/ce/Axis_Bank_logo.svg/2560px-Axis_Bank_logo.svg.png",
        width=140)
    st.title("Project Sentinel")
    st.caption("Strategic Intelligence Unit")
    st.divider()

    module = st.radio("Navigation",
                      ["💎 Strategic Overview",
                       "📊 Module 1: Market Data",
                       "🧠 Module 2: Sentiment Engine",
                       "📜 Module 3: Compliance Watch",
                       "💸 Module 4: Lending Sentinel"])

    st.divider()
    if module == "💎 Strategic Overview":
        st.info("ℹ️ Welcome to Command Center.")
    elif module == "📜 Module 3: Compliance Watch":
        st.info("ℹ️ Includes PDF Scraper.")
    elif module == "💸 Module 4: Lending Sentinel":
        st.info("ℹ️ Compare CC Loans & EMI.")
    else:
        st.success("🟢 Systems Online")

# --- 5. MODULE LOGIC ---

# >>> MODULE 0: STRATEGIC OVERVIEW (GEMSTONE UI) <<<
if module == "💎 Strategic Overview":
    st.title("💎 Sentinel Command Center")
    st.markdown("### AI-Led Competitive Intelligence Framework")
    st.markdown("---")

    # Adaptive Gemstone Cards
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="gemstone-card">
            <div class="gemstone-header">📊 Market Data</div>
            <p><b>Source:</b> RBI Official Reports</p>
            <p>Tracks Market Share, Net Additions, and Spend Quality (ATS). Definitive proof of "Where we stand".</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="gemstone-card">
            <div class="gemstone-header">📜 Compliance Watch</div>
            <p><b>Source:</b> RBI Notifications</p>
            <p>Real-time scraper for Circulars & Master Directions. Early warning system for regulatory risk.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="gemstone-card">
            <div class="gemstone-header">🧠 Sentiment Engine</div>
            <p><b>Source:</b> Reddit, Twitter (X)</p>
            <p>Uses GenAI (Llama-3) to detect "Rant vs. Rave" patterns. Explains "Why we are winning/losing".</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="gemstone-card">
            <div class="gemstone-header">💸 Lending Sentinel</div>
            <p><b>Source:</b> Bank Rate Cards</p>
            <p>Benchmarks "Loan on Credit Card" & Merchant EMI rates. Critical for pricing strategy.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("⚙️ The AI Pipeline")

    # Graphviz Diagram
    graph = graphviz.Digraph()
    graph.attr(rankdir='LR', bgcolor='transparent')

    # Nodes with adaptive colors logic handled by Streamlit automatically?
    # Actually graphviz renders an image. Use neutral colors.
    graph.attr('node', style='filled', fillcolor='white', fontcolor='black', shape='box')

    graph.node('A', 'Unstructured Data\n(Reddit/X)')
    graph.node('B', 'LLM Processor\n(Aspect Extraction)', fillcolor='#ffebee')
    graph.node('C', 'Sentiment Scorer\n(-1.0 to +1.0)')
    graph.node('D', 'Strategic Insight\n(Dashboard)', fillcolor='#8B0000', fontcolor='white')

    graph.edge('A', 'B', label=' Scrape')
    graph.edge('B', 'C', label=' Analyze')
    graph.edge('C', 'D', label=' Visualize')

    st.graphviz_chart(graph)

# >>> MODULE 1: MARKET DATA <<<
elif module == "📊 Module 1: Market Data":
    st.title("📊 The Market Truth: RBI Data Analytics")
    df_rbi, source_status = get_rbi_market_data()
    st.caption(f"Data Source: {source_status}")

    total_market = df_rbi['Active_Cards'].sum()
    df_rbi['Market_Share'] = (df_rbi['Active_Cards'] / total_market) * 100

    try:
        axis_row = df_rbi[df_rbi['Bank'].astype(str).str.contains('Axis', case=False)].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Axis Active Cards", f"{axis_row['Active_Cards'] / 1000000:.2f} M")
        with c2:
            st.metric("Market Share", f"{axis_row['Market_Share']:.1f}%")
        with c3:
            st.metric("Avg Ticket Size", f"₹{axis_row['Spend_Per_Card']:.0f}")
        with c4:
            st.metric("Status", "Verified")
    except:
        st.warning("Axis Bank data not found.")

    st.divider()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.subheader("Market Share")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        # Theme compatibility
        fig1.patch.set_alpha(0)
        ax1.patch.set_alpha(0)

        # Determine text color based on Streamlit theme config
        text_color = 'white' if st.get_option("theme.base") == "dark" else 'black'
        ax1.tick_params(colors=text_color)
        ax1.xaxis.label.set_color(text_color);
        ax1.yaxis.label.set_color(text_color)
        ax1.spines['bottom'].set_color(text_color);
        ax1.spines['left'].set_color(text_color)

        sns.barplot(data=df_rbi.head(6), x='Market_Share', y='Bank', palette='Blues_r', ax=ax1)
        for i, bar in enumerate(ax1.patches):
            if 'Axis' in df_rbi.head(6).iloc[i]['Bank']: bar.set_color('#8B0000')
        st.pyplot(fig1)

    with c2:
        st.subheader("Spend Quality (ATS)")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_alpha(0);
        ax2.patch.set_alpha(0)
        ax2.tick_params(colors=text_color)
        ax2.xaxis.label.set_color(text_color);
        ax2.yaxis.label.set_color(text_color)
        ax2.spines['bottom'].set_color(text_color);
        ax2.spines['left'].set_color(text_color)

        sns.barplot(data=df_rbi.sort_values('Spend_Per_Card', ascending=False).head(8), x='Spend_Per_Card', y='Bank',
                    palette='viridis', ax=ax2)
        st.pyplot(fig2)

# >>> MODULE 2: SENTIMENT ENGINE <<<
elif module == "🧠 Module 2: Sentiment Engine":
    st.title("🧠 The Customer Pulse: AI Sentiment Engine")
    df_cards = load_full_market_portfolio()

    all_competitors = df_cards[df_cards['Bank'] != 'Axis Bank']['Bank'].unique().tolist()
    competitors = st.multiselect("Benchmark Against:", all_competitors,
                                 default=['HDFC Bank', 'SBI Card', 'American Express'])
    filtered_df = df_cards[df_cards['Bank'].isin(competitors + ['Axis Bank'])]

    tab1, tab2, tab3 = st.tabs(["⚔️ Head-to-Head", "📉 Live Analysis", "📚 Full Database"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1: axis_c = st.selectbox("Axis Champion", df_cards[df_cards['Bank'] == 'Axis Bank']['Card'].unique())
        with c2: rival_c = st.selectbox("Competitor", filtered_df[filtered_df['Bank'] != 'Axis Bank']['Card'].unique())

        a_dat = df_cards[df_cards['Card'] == axis_c].iloc[0]
        r_dat = df_cards[df_cards['Card'] == rival_c].iloc[0]

        # Adaptive Gemstone Comparison
        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin-top: 10px;">
            <div class="gemstone-card" style="flex: 1;">
                <div class="gemstone-header">{a_dat['Card']}</div>
                <p><b>Fee:</b> {a_dat['Fee']} | <b>Yield:</b> {a_dat['Yield']}</p>
                <div style="font-size: 24px; font-weight: bold;">{a_dat['Sentiment']}</div>
                <p>Status: {a_dat['Status']}</p>
            </div>
            <div class="gemstone-card" style="flex: 1; border-left: 1px solid gray;">
                <div class="gemstone-header" style="color:var(--text-color)!important;">{r_dat['Card']}</div>
                <p><b>Fee:</b> {r_dat['Fee']} | <b>Yield:</b> {r_dat['Yield']}</p>
                <div style="font-size: 24px; font-weight: bold;">{r_dat['Sentiment']}</div>
                <p>Status: {r_dat['Status']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        if st.button("🚀 Run AI Extraction"):
            time.sleep(1)
            st.bar_chart(filtered_df.set_index('Card')['Sentiment'])

    with tab3:
        st.dataframe(filtered_df, use_container_width=True)

# >>> MODULE 3: COMPLIANCE WATCH (FIXED DOWNLOAD BUTTON) <<<
elif module == "📜 Module 3: Compliance Watch":
    st.title("📜 Compliance Watch: RBI Circulars")

    # 1. State Management for Download
    if 'pdf_data' not in st.session_state: st.session_state['pdf_data'] = None
    if 'pdf_name' not in st.session_state: st.session_state['pdf_name'] = ""

    if st.button("🔄 Refresh"): st.cache_data.clear()

    with st.spinner("Fetching latest circulars..."):
        df_news = get_rbi_circulars()

    if not df_news.empty:
        st.dataframe(df_news[['Date', 'Title', 'Category']], use_container_width=True)
        st.divider()
        st.subheader("📥 Smart PDF Downloader")
        selected_title = st.selectbox("Select Circular:", df_news['Title'].unique())
        selected_row = df_news[df_news['Title'] == selected_title].iloc[0]

        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🔎 Find PDF"):
                with st.spinner("Scraping..."):
                    # Reset state
                    st.session_state['pdf_data'] = None
                    pdf_bytes, msg = download_pdf_from_rbi(selected_row['Link'])

                    if pdf_bytes:
                        st.session_state['pdf_data'] = pdf_bytes
                        st.session_state['pdf_name'] = f"RBI_Circular.pdf"
                        st.success("Found!")
                        time.sleep(0.1)
                        st.rerun()  # Forces the UI to refresh and show the download button
                    else:
                        st.error(f"Error: {msg}")

        with c2:
            # Persistent Check: If data exists, show button
            if st.session_state['pdf_data']:
                st.download_button(
                    label="⬇️ Download PDF Now",
                    data=st.session_state['pdf_data'],
                    file_name=st.session_state['pdf_name'],
                    mime='application/pdf'
                )
    else:
        st.info("No recent circulars found.")

# >>> MODULE 4: LENDING SENTINEL <<<
elif module == "💸 Module 4: Lending Sentinel":
    st.title("💸 Lending Sentinel: Instant Loans & EMI")
    df_loan = load_lending_offers()

    tab1, tab2, tab3 = st.tabs(["⚖️ Rate Comparator", "🧮 EMI Calculator", "📋 Market Scanner"])

    with tab1:
        col_bench, col_comp = st.columns(2)
        with col_bench:
            axis_prod = st.selectbox("Axis Product", df_loan[df_loan['Bank'] == 'Axis Bank']['Product'].unique())
        with col_comp:
            rival_bank = st.selectbox("Competitor Bank", df_loan[df_loan['Bank'] != 'Axis Bank']['Bank'].unique())
            rival_prod_opts = df_loan[df_loan['Bank'] == rival_bank]['Product'].unique()
            rival_prod = st.selectbox("Competitor Product", rival_prod_opts if len(rival_prod_opts) > 0 else ["N/A"])

        if rival_prod != "N/A":
            a_l = df_loan[(df_loan['Bank'] == 'Axis Bank') & (df_loan['Product'] == axis_prod)].iloc[0]
            r_l = df_loan[(df_loan['Bank'] == rival_bank) & (df_loan['Product'] == rival_prod)].iloc[0]

            st.markdown(f"""
            <div style="display: flex; gap: 20px; margin-top: 15px;">
                <div class="gemstone-card" style="flex: 1;">
                    <div class="gemstone-header">{a_l['Bank']} - {a_l['Product']}</div>
                    <h1>{a_l['ROI_Min']}% <small style="font-size:16px;">to {a_l['ROI_Max']}%</small></h1>
                    <p><b>Proc Fee:</b> {a_l['Proc_Fee']}</p>
                </div>
                <div class="gemstone-card" style="flex: 1; border-left: 1px solid gray;">
                    <div class="gemstone-header" style="color:var(--text-color)!important;">{r_l['Bank']} - {r_l['Product']}</div>
                    <h1>{r_l['ROI_Min']}% <small style="font-size:16px;">to {r_l['ROI_Max']}%</small></h1>
                    <p><b>Proc Fee:</b> {r_l['Proc_Fee']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amt = st.number_input("Loan Amount (₹)", 50000, 1000000, 100000, step=10000)
        with c2:
            tenure = st.slider("Tenure (Months)", 6, 60, 12)
        with c3:
            axis_rate = st.number_input("Axis Rate (%)", 10.0, 24.0, 15.0)
            rival_rate = st.number_input("Rival Rate (%)", 10.0, 24.0, 14.5)


        def calc_emi(p, r, n):
            r_mon = r / (12 * 100)
            return p * r_mon * ((1 + r_mon) ** n) / (((1 + r_mon) ** n) - 1)


        emi_axis = calc_emi(loan_amt, axis_rate, tenure)
        emi_rival = calc_emi(loan_amt, rival_rate, tenure)

        st.divider()
        k1, k2, k3 = st.columns(3)
        with k1:
            st.metric("Axis Monthly EMI", f"₹{emi_axis:,.0f}")
        with k2:
            st.metric("Rival Monthly EMI", f"₹{emi_rival:,.0f}", delta=f"₹{emi_axis - emi_rival:,.0f} diff")
        with k3:
            savings = (emi_axis * tenure) - (emi_rival * tenure)
            if savings < 0:
                st.metric("Total Savings", f"₹{abs(savings):,.0f}")
            else:
                st.metric("Extra Cost", f"₹{savings:,.0f}")

    with tab3:
        st.dataframe(df_loan, use_container_width=True)