# Library
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from fastdtw import fastdtw
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from tslearn.metrics import cdist_dtw
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.metrics import calinski_harabasz_score

# Load dynamic data
path_file_filter = ['D:/ITB/Thesis/Preprocessing/Filtered Data/df_飲料零食_volume.csv', 
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_飲料零食_weight.csv', 
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_飲料零食_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_傢俱寢飾_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_大家都買這些_weight.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_嬰童保健_volume.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_嬰童保健_weight.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_嬰童保健_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_日用生活_volume.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_日用生活_weight.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_日用生活_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_日用生活_length.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_服飾鞋包_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_熱門3C_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_生活家電_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_生活家電_volume.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_生鮮冷凍_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_生鮮冷凍_volume.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_生鮮冷凍_weight.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_米油沖泡_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_米油沖泡_weight.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_米油沖泡_volume.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_美妝個清_volume.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_美妝個清_weight.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_美妝個清_count.csv',
                    'D:/ITB/Thesis/Preprocessing/Filtered Data/df_美妝個清_length.csv',
                    ] 

# Function to calculate Dunn Index
def dunn_index(D, labels):
    clusters = np.unique(labels)

    intra = []
    for c in clusters:
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            intra.append(np.max(D[np.ix_(idx, idx)]))

    if len(intra) == 0:
        return 0.0

    max_intra = np.max(intra)

    inter = []
    for i, c1 in enumerate(clusters):
        idx1 = np.where(labels == c1)[0]
        for c2 in clusters[i+1:]:
            idx2 = np.where(labels == c2)[0]
            inter.append(np.min(D[np.ix_(idx1, idx2)]))

    if len(inter) == 0:
        return 0.0

    min_inter = np.min(inter)

    return min_inter / max_intra

# Function to calculate within-cluster SSE for elbow method
def within_cluster_sse(X, labels):
    sse = 0.0
    for c in np.unique(labels):
        Xc = X[labels == c]
        mu = Xc.mean(axis=0, keepdims=True)
        sse += ((Xc - mu) ** 2).sum()
    return sse

path_order = "D:\\ITB\\Thesis\\Preprocessing\\訂單資料_order.csv"
df_order = pd.read_csv(path_order, encoding = 'utf-8-sig', sep=';', decimal=',')
print(f"Data loaded from {path_order}:")

# Reshape order data
    # Build per-product weekly time series matrix
df_order['创建时间'] = pd.to_datetime(df_order['创建时间'], format='%d/%m/%Y %H:%M')    

# Group the sales data by product and week, and calculate the total sales for each product for each week
    # Build week-start timestamp (Monday)
df_order['week_start'] = df_order['创建时间'].dt.to_period('W-MON').dt.start_time

    # Aggregate weekly sales per product
weekly_sales = df_order.groupby(['week_start', '商品名称', '商品编码'], as_index = False)['商品数量'].sum()

    # Wide matrix: rows are products, columns are weeks, values are sales
ts_wide = weekly_sales.pivot(index=['商品名称', '商品编码'], 
                            columns=['week_start'], 
                            values='商品数量').fillna(0.0).sort_index(axis=1)

# Dynamic clustering
# Stage 1: Cheap DTW grouping via MiniBatchKMeans
X = ts_wide.to_numpy(dtype=float)
n_series = X.shape[0]
n_prototypes = int(np.sqrt(n_series))

prototype_model = MiniBatchKMeans(n_clusters=n_prototypes, 
                                random_state=42,
                                batch_size=2048,
                                n_init = 10,
                                )
prototype_labels = prototype_model.fit_predict(X)

# Stage 2: DBA centroid per prototype group
centroids = []
group_ids = [] # prototype label for each centroid

for p in range (n_prototypes): 
    idx = np.where(prototype_labels == p)[0]
    if len(idx) == 0:
        continue
    
    Xi_3d = X[idx, :, np.newaxis]
    centroid = dtw_barycenter_averaging(Xi_3d, max_iter=10)
    centroids.append(centroid[:, 0]) # remove the extra dimension
    group_ids.append(p)
    
C = np.vstack(centroids)
m = C.shape[0]

# Z-normalize each centroid
Cz = (C - C.mean(axis=1, keepdims=True)) / (C.std(axis=1, keepdims=True) + 1e-8) # add small constant to avoid division by zero

# DTW distances matrix between centorids only between prototypes
D_prototype = cdist_dtw(Cz)

# Stage 3: Ward merge on normalized centroid vectors (Euclidean distance)
Z_proto_best = linkage(Cz, method='ward', metric='euclidean')

# Stage 4: choose k by elbow on within-cluster SSE (Ward/Euclidean space)
# The “elbow” is the point where adding clusters stops giving meaningful improvement
# candidate k range
k_candidate = range(2, min(20, m)) 

elbow_rows = []
for k in k_candidate:
    labels_k = fcluster(Z_proto_best, t=k, criterion='maxclust')
    sse_k = within_cluster_sse(Cz, labels_k)
    elbow_rows.append({'k': k, 'sse': sse_k})

df_elbow = pd.DataFrame(elbow_rows)

# automatic elbow: max distance to line between first and last point
x = df_elbow['k'].to_numpy(dtype=float)
y = df_elbow['sse'].to_numpy(dtype=float)

x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

p1 = np.array([x_n[0], y_n[0]])
p2 = np.array([x_n[-1], y_n[-1]])

num = np.abs((p2[1] - p1[1]) * x_n - (p2[0] - p1[0]) * y_n + p2[0] * p1[1] - p2[1] * p1[0])
den = np.sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2) + 1e-12
dist_to_line = num / den

best_idx = np.argmax(dist_to_line)
best_k = int(x[best_idx])

print(f'df_order: {path_order}')
print("Elbow table:")
print(f"\nBest k by elbow: {best_k}")

# Final labels with elbow-selected k
centroid_labels = fcluster(Z_proto_best, t=best_k, criterion='maxclust')

# Performance Evaluation
# DTW: Silhoutte, Dunn
# Euclidean: DB, CH

k_candidate = range(2, min(10, m // 2))
validity_result = []

for k in k_candidate:
    proto_labels_k = fcluster(Z_proto_best, t=k, criterion='maxclust')
    # Performance evaluation
    sil = silhouette_score(D_prototype, proto_labels_k, metric='precomputed')
    dunn = dunn_index(D_prototype, proto_labels_k)
    
    db = davies_bouldin_score(Cz, proto_labels_k)
    ch = calinski_harabasz_score(Cz, proto_labels_k)
    
    validity_result.append({
        'k': k, 
        'silhouette': sil,
        'dunn': dunn,
        'davies_bouldin': db,
        'calinski_harabasz': ch
    })
    
df_proto_validity = pd.DataFrame(validity_result)

print(f"df_path : {path_order}")
print("Prototype-level validity indices:")
print(df_proto_validity)

# Map each product to its prototype cluster via its assigned prototype

group_to_dynamic = {gid: int(centroid_labels[i]) for i, gid in enumerate(group_ids)}

dynamic_cluster_labels = np.array([group_to_dynamic[prototype_labels[prod_idx]] for prod_idx in range(len(prototype_labels))])
ts_wide['dynamic_cluster'] = dynamic_cluster_labels

print(f'Dynamic cluster distribution:')  
print(ts_wide['dynamic_cluster'].value_counts().sort_index())

# Make a new dataframe

# Incorporate the resultant dynamic cluster labels into each static dataframe,
# then create one merged mixed-cluster dataset with globally unique ids.
cluster_map = dict(zip(ts_wide.index.get_level_values('商品编码'), ts_wide['dynamic_cluster']))

merged_results = []
global_cluster_offset = 0

for path in path_file_filter:
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df['dynamic_cluster'] = df['item_code'].map(cluster_map)

    # Keep rows with dynamic cluster available from order history.
    df = df.dropna(subset=['dynamic_cluster']).copy()
    if df.empty:
        print(f"Skip {path}: no matching dynamic cluster labels.")
        continue

    # One-hot encode dynamic cluster labels.
    df_encoded = pd.get_dummies(df, columns=['dynamic_cluster'], prefix='cluster', dtype=int)

    # Normalization of numeric features.
    price_columns = ['price_per_piece', 'original_price_per_piece']
    for price in price_columns:
        df_encoded[price] = pd.to_numeric(df_encoded[price], errors='coerce')
        df_encoded[f'log_{price}'] = np.log1p(df_encoded[price].clip(lower=0))

    cluster_cols = df_encoded.filter(like='cluster_').columns
    cols_to_normalize = [
        'estimation_discount',
        'log_price_per_piece',
        'log_original_price_per_piece',
        'shelf_life',
        'capacity'
    ] + list(cluster_cols)

    for col in cols_to_normalize:
        df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
    df_encoded = df_encoded.dropna(subset=cols_to_normalize).copy()
    '''
    #if len(df_encoded) < 3:
    #    print(f"Skip {path}: not enough rows after preprocessing.")
    #   continue
    '''

    scaler = StandardScaler()
    for col in cols_to_normalize:
        df_encoded[f'{col}_normalized'] = scaler.fit_transform(df_encoded[[col]])

    # K-means clustering (static data) with elbow method.
    features_final = df_encoded.filter(like='_normalized').columns
    X_static = df_encoded[features_final].astype(float)

    # Ensure valid k range for each file.
    k_max = min(9, len(X_static) - 1)
    if k_max < 2:
        print(f"Skip {path}: not enough samples for K-means.")
        continue

    k_values = list(range(2, k_max + 1))
    elbow_rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        km.fit(X_static)
        elbow_rows.append({'k': k, 'sse': km.inertia_})

    df_elbow_static = pd.DataFrame(elbow_rows)
    x = df_elbow_static['k'].to_numpy(dtype=float)
    y = df_elbow_static['sse'].to_numpy(dtype=float)

    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])

    num = np.abs((p2[1] - p1[1]) * x_n - (p2[0] - p1[0]) * y_n + p2[0] * p1[1] - p2[1] * p1[0])
    den = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) + 1e-12
    dist_to_line = num / den

    best_idx = np.argmax(dist_to_line)
    best_k_static = int(x[best_idx])

    best_kmeans = KMeans(n_clusters=best_k_static, random_state=0, n_init=10)
    labels_static = best_kmeans.fit_predict(X_static)

    file_result = df_encoded[['title', 'category', 'capacity_unit', 'item_code']].copy()
    file_result['cluster_local'] = labels_static
    file_result['cluster_global'] = file_result['cluster_local'] + global_cluster_offset

    # Mixed cluster id is global across files.
    file_result['mixed_cluster'] = file_result['cluster_global']

    cluster_sizes_best = file_result['cluster_global'].value_counts()
    file_result['cluster_size'] = file_result['cluster_global'].map(cluster_sizes_best)
    file_result['allocation_type'] = np.where(
        file_result['cluster_size'] > 1,
        'common allocation',
        'random allocation'
    )
    merged_results.append(file_result)

    global_cluster_offset += best_k_static
    print(f"Processed {path}: best_k_static={best_k_static}, next_global_offset={global_cluster_offset}")

if merged_results:
    df_mixed_merged = pd.concat(merged_results, ignore_index=True)
    output_path = 'D:/ITB/Thesis/Clustering/mixed_cluster_results.csv'
    df_mixed_merged.to_csv(output_path, index=False, encoding='utf-8-sig', sep=';', decimal=',')
    print(f"Merged mixed cluster result saved to: {output_path}")
    print(f"Total rows: {len(df_mixed_merged)}, total global clusters: {df_mixed_merged['cluster_global'].nunique()}")
else:
    print('No file produced a valid mixed clustering result.')