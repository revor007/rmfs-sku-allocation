# libary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.sparse import coo_matrix
import time
from sklearn.preprocessing import normalize

# Load the mixed-cluster assignment from CSV
path_cluster = "fcgma\\Clustering\\time-series-clustering-results.csv"
df_mixed = pd.read_csv(path_cluster, sep=';', decimal=',', encoding='utf-8-sig', engine = "python")
df_mixed.info()

# Create a same-cluster indicator matrix from the mixed-cluster assignment
# Pairwise same-cluster indicator matrix
u = df_mixed[['item_code', 'cluster']].set_index('item_code')
c = u['cluster'].to_numpy() # Get the cluster labels as a numpy array
M = (c[:, None] == c[None, :]).astype(int) 

# Same product must be 0
np.fill_diagonal(M, 0)

# Final indicator matrix
same_cluster_matrix = pd.DataFrame(M, index = u.index, columns = u.index)
same_cluster_matrix.rename_axis(index=None, inplace=True)
same_cluster_matrix.head()

# Save same-cluster indicator matrix to CSV
same_cluster_matrix.to_csv("fcgma\\same_cluster_matrix.csv", encoding='utf-8-sig', sep=';', decimal=',')