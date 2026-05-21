from pathlib import Path

import numpy as np
import pandas as pd

from experiment_context import load_experiment_context


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "Clustering" / "time-series-clustering-results.csv"
OUTPUT_PATH = BASE_DIR / "same_cluster_matrix.csv"


def main():
    context = load_experiment_context(BASE_DIR)
    cluster_df = pd.read_csv(INPUT_PATH, sep=";", encoding="utf-8-sig", decimal=",")
    cluster_df["item_code"] = cluster_df["item_code"].astype(str).str.strip()
    cluster_df["cluster"] = pd.to_numeric(cluster_df["cluster"], errors="coerce")
    cluster_df = cluster_df.dropna(subset=["cluster"]).copy()
    cluster_df["cluster"] = cluster_df["cluster"].astype(np.int32)

    cluster_map = cluster_df.set_index("item_code")["cluster"].to_dict()
    all_skus = context.eligible_skus
    same_cluster = pd.DataFrame(0, index=all_skus, columns=all_skus, dtype=np.int32)

    historical_skus = [sku for sku in all_skus if sku in cluster_map]
    if historical_skus:
        labels = np.asarray([cluster_map[sku] for sku in historical_skus], dtype=np.int32)
        membership = (labels[:, None] == labels[None, :]).astype(np.int32)
        np.fill_diagonal(membership, 0)
        same_cluster.loc[historical_skus, historical_skus] = membership

    same_cluster.to_csv(OUTPUT_PATH, encoding="utf-8-sig", sep=";", decimal=",")

    print(f"Eligible master SKUs: {len(all_skus):,}")
    print(f"Historical SKUs with clusters: {len(historical_skus):,}")
    print(f"Same-cluster matrix saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
