import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
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
    'Total_Spend_Last_Year',
    'Total_Transactions',
    'Avg_Transaction_Amount',
    'Max_Transaction_Amount',
    'Min_Transaction_Amount',
    'Unique_Merchant_Categories',
    'Unique_Transaction_Cities',
    'sss' 
]

print("Cleaning data...")
df_clean = df.dropna(subset=features + ['Defaulted'])
X = df_clean[features]
y = df_clean['Defaulted']

# --- Pipeline 1: Clustering (Unsupervised Segmentation) ---
print("Training KMeans Clustering model...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate normalized cluster centers for the Radar Chart
cluster_centers = kmeans.cluster_centers_ 
persona_df = pd.DataFrame(scaler.inverse_transform(cluster_centers), columns=features)
print("\nCluster Personas (Averages):")
print(persona_df.head())

# --- Pipeline 2: Deep Learning Risk Prediction (Supervised) ---
print("\nTraining Deep Learning Risk Model (MLP Neural Network)...")

# We will train the MLP to predict Cluster Personas directly instead of default risk
# This makes the "Interpretability" matrix map to our digital twin persona assignments.
y_personas = df_clean['Cluster']

mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=300, activation='relu', random_state=42)
mlp_model.fit(X_scaled, y_personas)

# Evaluate performance (Confusion Matrix for the Interpretability Tab)
print("Evaluating MLP Persona Predictor...")
y_pred = mlp_model.predict(X_scaled)
cm = confusion_matrix(y_personas, y_pred)

# Since MLP is a black box, we extract Feature Importance mathematically using Permutation
print("Calculating Deep Learning Feature Importances via Permutation...")
perm_importance = permutation_importance(mlp_model, X_scaled, y_personas, n_repeats=5, random_state=42)
feature_importances = perm_importance.importances_mean

# --- Exporting Models, Scaler & Plotting Data ---
print("Saving models and mathematical insights to disk for Streamlit...")
joblib.dump(scaler, 'digital_twin_scaler.pkl')
joblib.dump(kmeans, 'digital_twin_cluster_model.pkl')
joblib.dump(mlp_model, 'digital_twin_dl_model.pkl')

visualization_data = {
    'features': features,
    'cluster_centers_normalized': cluster_centers, # For radar charts
    'cluster_centers_actual': persona_df.to_dict(),
    'dl_feature_importances': feature_importances,
    'dl_confusion_matrix': cm
}
joblib.dump(visualization_data, 'digital_twin_viz_data.pkl')

print("✅ Training complete. Deep Learning MLP, K-Means, and Mathematical Insights exported.")
