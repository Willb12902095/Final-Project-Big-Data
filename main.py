# main.py - Final Project Big Data (Fully aligned to TA GitHub Action)
# Author: <your name or student ID>

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ======================
# Step 1: Load Public Data
# ======================

print("🔍 Loading public data...")
df_public = pd.read_csv('public_data.csv')

print("📋 Public data columns:", df_public.columns)
print(df_public.head())

# ======================
# Step 2: Visualize Public Data
# ======================

# Pair plot
sns.pairplot(df_public)
plt.show()

# Dimension 2 vs 3 (hint from PDF)
sns.scatterplot(x='2', y='3', data=df_public)
plt.title('Public Data: Dimension 2 vs 3')
plt.show()

# ======================
# Step 3: K-Means Clustering on Public Data
# ======================

print("🤖 Running K-Means on public data...")

# Prepare features (exclude 'id')
X_public = df_public[['1', '2', '3', '4']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_public)

# Define number of clusters (4n - 1 = 15)
n_clusters_public = 15
kmeans = KMeans(n_clusters=n_clusters_public, random_state=42, n_init=10)
labels_public = kmeans.fit_predict(X_scaled)

# Save in required format: id,label (id must start from 1!)
submission_public = pd.DataFrame({
    'id': df_public['id'],
    'label': labels_public
})

submission_public.to_csv('public_submission.csv', index=False)

print("✅ Public clustering done! Saved as public_submission.csv (ready for GitHub Action grading)")

# ======================
# Step 4: Load Private Data
# ======================

print("🔍 Loading private data...")
df_private = pd.read_csv('private_data.csv')

print("📋 Private data columns:", df_private.columns)
print(df_private.head())

# ======================
# Step 5: K-Means Clustering on Private Data
# ======================

print("🤖 Running K-Means on private data...")

# Prepare features
X_private = df_private[['1', '2', '3', '4', '5', '6']]

# Scale the data
scaler_private = StandardScaler()
X_scaled_private = scaler_private.fit_transform(X_private)

# Define number of clusters (4n - 1 = 23)
n_clusters_private = 23
kmeans_private = KMeans(n_clusters=n_clusters_private, random_state=42, n_init=10)
labels_private = kmeans_private.fit_predict(X_scaled_private)

# Save in required format: id,label
submission_private = pd.DataFrame({
    'id': df_private['id'],
    'label': labels_private
})

submission_private.to_csv('private_submission.csv', index=False)

print("✅ Private clustering done! Saved as private_submission.csv (for final ZIP submission)")

# ======================
# 🎉 DONE!
# ======================
