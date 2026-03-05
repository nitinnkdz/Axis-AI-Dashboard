import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import joblib

print("Loading dataset...")
df = pd.read_csv('Credit_Card_Dataset.csv')

# --- Feature Engineering & Selection ---
features = [
    'Age', 
    'Annual_Income', 
    'Credit_Score', 
    'Number_of_Credit_Lines', 
    'Credit_Utilization_Ratio', 
    'Debt_To_Income_Ratio', 
    'Total_Spend_Last_Year'
]

# Drop NaNs in our specific features
df_clean = df.dropna(subset=features + ['Defaulted'])
X = df_clean[features]
y = df_clean['Defaulted']

# --- Pipeline 1: Clustering (Unsupervised Segmentation) ---
print("Training KMeans Clustering model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine clusters (4 seems like a good starting point for personas)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate cluster centers to understand the personas
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
persona_df = pd.DataFrame(cluster_centers, columns=features)
print("\nCluster Personas (Averages):")
print(persona_df)

# --- Pipeline 2: Risk / Default Prediction (Supervised) ---
print("\nTraining Random Forest Risk Model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate generic performance
y_pred = rf_model.predict(X_test)
print("\nRisk Model Performance:")
print(classification_report(y_test, y_pred))

# --- Exporting Models & Scaler ---
print("Saving models to disk...")
joblib.dump(scaler, 'digital_twin_scaler.pkl')
joblib.dump(kmeans, 'digital_twin_cluster_model.pkl')
joblib.dump(rf_model, 'digital_twin_risk_model.pkl')

print("✅ Training complete. Scaler, Cluster Model, and Risk Model exported successfully.")
