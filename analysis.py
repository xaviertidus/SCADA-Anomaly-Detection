# SCADA Anomaly Detection Analysis
#
# IMPORTANT NOTICE: If you use, modify, or build upon this code, please credit:
# - The original Git repository: https://github.com/xaviertidus/SCADA-Anomaly-Detection
# - The paper authors: Shirazi et al. (2016). "Evaluation of Anomaly Detection Techniques for SCADA Communication Resilience".
#
# This script replicates anomaly detection from the paper.

import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

# G-mean helper
def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)

# Load data
try:
    data, meta = arff.loadarff('IanArffDataset.arff')
    df = pd.DataFrame(data)
except:
    df = pd.read_csv('IanArffDataset.txt', sep=',', na_values='?')

# Decode if ARFF
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.decode('utf-8')

# Features and binary labels
features = df.drop(['binary result', 'categorized result', 'specific result', 'time'], axis=1, errors='ignore')
labels = df['binary result'].apply(lambda x: 1 if x == "'1'" else 0)

# Impute missing
numerical_cols = features.select_dtypes(include=[np.number]).columns
categorical_cols = features.select_dtypes(exclude=[np.number]).columns
features[numerical_cols] = features[numerical_cols].fillna(features[numerical_cols].mean())
if not categorical_cols.empty:
    features[categorical_cols] = features[categorical_cols].fillna(features[categorical_cols].mode().iloc[0])

# Normalize numerical
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[numerical_cols])

# PCA
pca = PCA(n_components=0.95)
features_pca = pca.fit_transform(features_scaled)

# Split
X_train, X_test, y_train, y_test = train_test_split(features_pca, labels, test_size=0.3, random_state=42)

print("Data preprocessed. Train shape:", X_train.shape)

# Train models
km = KMeans(n_clusters=2, random_state=42).fit(X_train)
km_pred = km.predict(X_test)

gmm = GaussianMixture(n_components=2, random_state=42).fit(X_train)
gmm_scores = gmm.score_samples(X_test)
gmm_pred = (gmm_scores < np.percentile(gmm_scores, 21)).astype(int)

nb = GaussianNB().fit(X_train, y_train)
nb_pred = nb.predict(X_test)

pca_svd = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca_svd.fit_transform(X_train)
X_train_recon = pca_svd.inverse_transform(X_train_pca)
recon_error_train = np.mean((X_train - X_train_recon)**2, axis=1)
threshold = np.mean(recon_error_train) + 3 * np.std(recon_error_train)
X_test_pca = pca_svd.transform(X_test)
X_test_recon = pca_svd.inverse_transform(X_test_pca)
recon_error_test = np.mean((X_test - X_test_recon)**2, axis=1)
pca_pred = (recon_error_test > threshold).astype(int)

iso = IsolationForest(contamination=0.21, random_state=42).fit(X_train)
iso_pred = (iso.predict(X_test) == -1).astype(int)

# Evaluate
models = {'K-means': km_pred, 'NB': nb_pred, 'PCA-SVD': pca_pred, 'GMM': gmm_pred, 'Isolation Forest': iso_pred}
results = {}
for name, pred in models.items():
    results[name] = {
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred),
        'Accuracy': accuracy_score(y_test, pred),
        'F-score': f1_score(y_test, pred),
        'G-mean': g_mean(y_test, pred)
    }

print(pd.DataFrame(results).T)