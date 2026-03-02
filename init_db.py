import sqlite3
import pandas as pd
import os

DB_PATH = 'sentinel_data.db'

def init_db():
    print(f"Initializing database at {os.path.abspath(DB_PATH)}")
    
    # Remove existing db if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Removed existing database.")

    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 1. Card Universe Data
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
        for c in axis:
            cards.append({
                "Card": c[0], "Bank": "Axis Bank", "Type": c[1],
                "Sentiment": c[2], "Status": c[3], "Market_Dominance": c[4],
                "Growth_Potential": c[5]
            })

        # COMPETITORS
        comps = [
            ("HDFC Infinia Metal", "HDFC Bank", "Super Premium", 0.92, "Market Leader", 95, 60),
            ("HDFC Diners Club Black", "HDFC Bank", "Super Premium", 0.89, "Strong Contender", 85, 65),
            ("HDFC Regalia Gold", "HDFC Bank", "Premium", 0.70, "Volume", 90, 50),
            ("HDFC Millennia", "HDFC Bank", "Cashback", 0.85, "Mass Market", 92, 80),
            ("HDFC Tata Neu Infinity", "HDFC Bank", "Co-Brand", 0.88, "Rising", 75, 85),
            ("HDFC Swiggy", "HDFC Bank", "Co-Brand", 0.82, "Popular", 70, 90),
            
            ("SBI Cashback", "SBI Card", "Cashback", 0.88, "Threat", 70, 95),
            ("SBI Elite", "SBI Card", "Premium", 0.72, "Stable", 65, 40),
            ("SBI Prime", "SBI Card", "Rewards", 0.68, "Mass Market", 75, 45),
            ("SBI SimplyCLICK", "SBI Card", "Entry", 0.80, "Volume Leader", 95, 60),
            ("SBI SimplySAVE", "SBI Card", "Entry", 0.65, "Mass Market", 85, 35),
            ("SBI Vistara Prime", "SBI Card", "Co-Brand", 0.84, "Niche", 50, 70),
            
            ("ICICI Emeralde Private", "ICICI Bank", "Super Premium", 0.85, "Elite", 30, 70),
            ("ICICI Sapphiro", "ICICI Bank", "Premium", 0.65, "Stable", 60, 40),
            ("ICICI Rubyx", "ICICI Bank", "Rewards", 0.60, "Mass Market", 75, 35),
            ("ICICI Coral", "ICICI Bank", "Entry", 0.55, "Volume", 88, 30),
            ("ICICI Amazon Pay", "ICICI Bank", "Shopping", 0.95, "Volume Leader", 99, 50),
            ("ICICI MakeMyTrip Signature", "ICICI Bank", "Travel", 0.75, "Popular", 65, 55),
            
            ("Amex Platinum Charge", "American Express", "Super Premium", 0.90, "Brand Leader", 40, 70),
            ("Amex Platinum Travel", "American Express", "Travel", 0.88, "Strong", 55, 75),
            ("Amex Gold Charge", "American Express", "Premium", 0.82, "Classic", 45, 50),
            ("Amex Membership Rewards", "American Express", "Entry", 0.85, "Popular", 60, 65),
            
            ("IndusInd Pinnacle", "IndusInd Bank", "Super Premium", 0.78, "Stable", 40, 50),
            ("IndusInd Legend", "IndusInd Bank", "Premium", 0.68, "Mass Market", 55, 45),
            ("IndusInd EazyDiner", "IndusInd Bank", "Co-Brand", 0.86, "Dining Leader", 65, 80),
            
            ("Standard Chartered Ultimate", "Standard Chartered", "Super Premium", 0.80, "Stable", 35, 45),
            ("Standard Chartered Smart", "Standard Chartered", "Cashback", 0.75, "Growing", 50, 60),
            
            ("IDFC FIRST Wealth", "IDFC FIRST Bank", "Premium", 0.84, "Emerging", 55, 80),
            ("IDFC FIRST Select", "IDFC FIRST Bank", "Rewards", 0.78, "Growing", 65, 75),
            ("IDFC FIRST Vistara", "IDFC FIRST Bank", "Travel", 0.82, "Popular", 50, 70),
            
            ("Kotak White Reserve", "Kotak Mahindra", "Super Premium", 0.70, "Elite", 25, 40),
            ("Kotak Zenith", "Kotak Mahindra", "Premium", 0.65, "Stable", 45, 35),
            ("Kotak Myntra", "Kotak Mahindra", "Co-Brand", 0.80, "Shopping", 60, 75),
            
            ("AU Zenith+", "AU Small Finance", "Super Premium", 0.82, "Rising", 30, 85),
            ("AU Vetta", "AU Small Finance", "Premium", 0.75, "Growing", 40, 70),
            
            ("Yes Private", "Yes Bank", "Super Premium", 0.75, "Elite", 20, 30),
            ("Yes Marquee", "Yes Bank", "Premium", 0.80, "Emerging", 35, 60),
            
            ("OneCard Metal", "OneCard", "Fintech", 0.78, "Popular", 50, 75)
        ]
        for c in comps:
            cards.append({
                "Card": c[0], "Bank": c[1], "Type": c[2],
                "Sentiment": c[3], "Status": c[4], "Market_Dominance": c[5],
                "Growth_Potential": c[6]
            })

        # Save to SQLite
        df_cards = pd.DataFrame(cards)
        df_cards.to_sql('card_universe', conn, if_exists='replace', index=False)
        print(f"Inserted {len(df_cards)} records into 'card_universe'")

        # 2. RBI Market Data
        rbi_data = [
            {"Bank": "HDFC Bank", "Active_Cards": 22350000, "Spend_Per_Card": 21500, "Market_Share": 22.3},
            {"Bank": "SBI Card", "Active_Cards": 19100000, "Spend_Per_Card": 16200, "Market_Share": 19.0},
            {"Bank": "ICICI Bank", "Active_Cards": 16800000, "Spend_Per_Card": 17800, "Market_Share": 16.2},
            {"Bank": "Axis Bank", "Active_Cards": 14200000, "Spend_Per_Card": 18500, "Market_Share": 13.6},
            {"Bank": "Kotak Mahindra", "Active_Cards": 6100000, "Spend_Per_Card": 14100, "Market_Share": 4.0},
            {"Bank": "American Express", "Active_Cards": 1400000, "Spend_Per_Card": 42000, "Market_Share": 1.5}
        ]
        df_rbi = pd.DataFrame(rbi_data)
        df_rbi.to_sql('rbi_market_data', conn, if_exists='replace', index=False)
        print(f"Inserted {len(df_rbi)} records into 'rbi_market_data'")

        # 3. Lending Offers Data
        lending_data = [
            {"Bank": "Axis Bank", "Product": "Insta Loan (PL)", "ROI_Min": 9.99, "ROI_Max": 18.0, "Proc_Fee": "2%"},
            {"Bank": "HDFC Bank", "Product": "Jumbo Loan", "ROI_Min": 10.80, "ROI_Max": 21.0, "Proc_Fee": "₹999"},
            {"Bank": "ICICI Bank", "Product": "PL on Card", "ROI_Min": 10.60, "ROI_Max": 16.99, "Proc_Fee": "1.5%"}
        ]
        df_lending = pd.DataFrame(lending_data)
        df_lending.to_sql('lending_offers', conn, if_exists='replace', index=False)
        print(f"Inserted {len(df_lending)} records into 'lending_offers'")

        print("\nDatabase initialization complete! ✅")
        print("You can verify the database schema using tools like DB Browser for SQLite.")
        
    except Exception as e:
        print(f"Error during initialization: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
