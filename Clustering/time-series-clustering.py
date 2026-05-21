from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from experiment_context import load_experiment_context

OUTPUT_PATH = Path(__file__).resolve().parent / "time-series-clustering-results.csv"
SUMMARY_PATH = Path(__file__).resolve().parent / "time-series-clustering-summary.csv"


def build_daily_demand_matrix(order_df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        order_df.groupby(["item_code", "order_date"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "daily_demand"})
    )
    pivot = daily.pivot_table(
        index="item_code",
        columns="order_date",
        values="daily_demand",
        fill_value=0.0,
    )
    return pivot.sort_index()


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

    demand_matrix = build_daily_demand_matrix(historical_orders)
    log_demand_matrix = np.log1p(demand_matrix.to_numpy(dtype=float))

    if demand_matrix.shape[0] < 2:
        cluster_labels = np.ones(demand_matrix.shape[0], dtype=np.int32)
        best_k = 1
        summary = pd.DataFrame(
            [{"actual_n_clusters": 1, "silhouette_score": np.nan, "wcsse": 0.0}]
        )
    else:
        cluster_labels, best_k, summary = choose_cluster_count(log_demand_matrix)

    cluster_df = pd.DataFrame(
        {
            "item_code": demand_matrix.index.astype(str),
            "cluster": cluster_labels,
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
