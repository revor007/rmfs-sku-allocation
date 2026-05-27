from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from experiment_context import load_experiment_context

OUTPUT_PATH = Path(__file__).resolve().parent / "time-series-clustering-results.csv"
SUMMARY_PATH = Path(__file__).resolve().parent / "time-series-clustering-summary.csv"


def build_daily_order_frequency_matrix(order_df: pd.DataFrame, date_col: str = "created_at", order_id_col: str = "order_id",) -> pd.DataFrame:

    order_df = order_df.copy()
    
    order_df[date_col] = pd.to_datetime(
        order_df[date_col],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()
        
    daily_freq = (
        order_df.groupby(["item_code", date_col], as_index=False)[order_id_col]
        .nunique()
        .rename(columns={order_id_col: "daily_order_frequency"})
    )

    pivot = daily_freq.pivot_table(
        index="item_code",
        columns=date_col,
        values="daily_order_frequency",
        fill_value=0.0,
    )
    
    full_dates = pd.date_range(
            order_df[date_col].min(),
            order_df[date_col].max(),
            freq="D",
    )

    pivot = pivot.reindex(columns=full_dates, fill_value=0.0)

    
    return pivot.sort_index()

def build_normalized_frequency_pattern_matrix(freq_matrix: pd.DataFrame) -> np.ndarray:
    values = freq_matrix.to_numpy(dtype=float)
    row_sum = values.sum(axis=1, keepdims=True)
    pattern_values = values / np.maximum(row_sum, 1e-12)
    return pattern_values

def wcsse_score(values: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0
    for cluster_id in np.unique(labels):
        cluster_values = values[labels == cluster_id]
        if cluster_values.size == 0:
            continue
        centroid = cluster_values.mean(axis=0)
        total += float(((cluster_values - centroid) ** 2).sum())
    return total


def choose_cluster_count(values: np.ndarray) -> tuple[np.ndarray, int, pd.DataFrame]:
    max_candidates = min(60, values.shape[0] - 1)
    if max_candidates < 2:
        labels = np.ones(values.shape[0], dtype=np.int32)
        summary = pd.DataFrame(
            [{"actual_n_clusters": 1, "silhouette_score": np.nan, "wcsse": 0.0}]
        )
        return labels, 1, summary

    Z = linkage(values, method="ward")
    records = []
    for k in range(2, max_candidates + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        actual_k = int(np.unique(labels).size)
        if actual_k < 2 or actual_k >= len(values):
            continue

        records.append(
            {
                "actual_n_clusters": actual_k,
                "silhouette_score": float(silhouette_score(values, labels, metric="euclidean")),
                "wcsse": float(wcsse_score(values, labels)),
            }
        )

    if not records:
        labels = np.ones(values.shape[0], dtype=np.int32)
        summary = pd.DataFrame(
            [{"actual_n_clusters": 1, "silhouette_score": np.nan, "wcsse": 0.0}]
        )
        return labels, 1, summary

    summary = pd.DataFrame(records).drop_duplicates(subset=["actual_n_clusters"])
    x = summary["actual_n_clusters"].to_numpy(dtype=float)
    y = summary["wcsse"].to_numpy(dtype=float)
    x_n = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y_n = (y - y.min()) / max(y.max() - y.min(), 1e-12)

    p1 = np.array([x_n[0], y_n[0]])
    p2 = np.array([x_n[-1], y_n[-1]])
    numerator = np.abs(
        (p2[1] - p1[1]) * x_n
        - (p2[0] - p1[0]) * y_n
        + p2[0] * p1[1]
        - p2[1] * p1[0]
    )
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2) + 1e-12
    distances = numerator / denominator
    best_idx = int(np.argmax(distances))
    best_k = int(x[best_idx])

    labels = fcluster(Z, t=best_k, criterion="maxclust")
    return labels.astype(np.int32), best_k, summary


def main():
    context = load_experiment_context(BASE_DIR)
    historical_orders = context.train_eligible_df[
        context.train_eligible_df["item_code"].isin(context.historical_skus)
    ].copy()
    if historical_orders.empty:
        raise ValueError("No historical pre-T orders remain for time-series clustering.")

    freq_matrix = build_daily_order_frequency_matrix(
        historical_orders,
        date_col="created_at",
        order_id_col="order_id",
    )
    
    pattern_matrix = build_normalized_frequency_pattern_matrix(freq_matrix)

    if pattern_matrix.shape[0] < 2:
        cluster_labels = np.ones(pattern_matrix.shape[0], dtype=np.int32)
        best_k = 1
        summary = pd.DataFrame(
            [{"actual_n_clusters": 1, "silhouette_score": np.nan, "wcsse": 0.0}]
        )
    else:
        cluster_labels, best_k, summary = choose_cluster_count(pattern_matrix)

    cluster_df = pd.DataFrame(
        {
            "item_code": freq_matrix.index.astype(str),
            "cluster": cluster_labels,
            "total_order_frequency": freq_matrix.sum(axis=1).to_numpy(dtype=float),
            "active_days": (freq_matrix > 0).sum(axis=1).to_numpy(dtype=int),
        }
    )
    cluster_df.to_csv(OUTPUT_PATH, index=False, sep=";", encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, sep=";", encoding="utf-8-sig")

    print(f"Cutoff ratio: {context.cutoff_ratio:.2f}")
    print(f"Cutoff timestamp: {context.cutoff_time}")
    print(f"Historical SKUs clustered: {len(cluster_df):,}")
    print(f"Best k: {best_k}")
    print(f"Cluster results saved to: {OUTPUT_PATH}")
    print(f"Cluster summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
