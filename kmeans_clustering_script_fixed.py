import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Load dataset
df = pd.read_csv("survey.csv")

# Drop irrelevant columns
df = df.drop(columns=['Timestamp', 'state', 'comments'], errors='ignore')

# Filter out invalid age values
df = df[(df['Age'] >= 18) & (df['Age'] <= 100)]

# Clean Gender column
def clean_gender(g):
    g = str(g).strip().lower()
    if g in ['male', 'm', 'man']:
        return 'Male'
    elif g in ['female', 'f', 'woman']:
        return 'Female'
    else:
        return 'Other'

df['Gender'] = df['Gender'].apply(clean_gender)

# Handle missing values safely (avoid chained assignment)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Separate categorical and numerical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Encode categorical columns
if int(sklearn.__version__.split(".")[1]) >= 2:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

encoded_cat = encoder.fit_transform(df[categorical_cols])
encoded_cat_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out(categorical_cols))

# Normalize numerical columns
scaler = MinMaxScaler()
scaled_num = scaler.fit_transform(df[numerical_cols])
scaled_num_df = pd.DataFrame(scaled_num, columns=numerical_cols)

# Combine all processed features
processed_df = pd.concat([encoded_cat_df.reset_index(drop=True), scaled_num_df.reset_index(drop=True)], axis=1)

# Determine optimal number of clusters
inertia = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(processed_df)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(processed_df, labels))

optimal_k = sil_scores.index(max(sil_scores)) + 2

# Final K-Means clustering
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
final_labels = final_kmeans.fit_predict(processed_df)

# PCA for visualization
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(processed_df)
reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
reduced_df['Cluster'] = final_labels

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=reduced_df, x='PC1', y='PC2', hue='Cluster', palette='Set2', s=60)
plt.title(f'K-Means Clustering with k={optimal_k}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
