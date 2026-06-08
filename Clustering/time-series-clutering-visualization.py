import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ORDERS_PATH = r"fcgma\Preprocessing\訂單資料_train.csv"
RESULTS_PATH = r"fcgma\Clustering\time-series-clustering-results.csv"
SUMMARY_PATH = r"fcgma\Clustering\time-series-clustering-summary.csv"


def find_elbow_k(summary_df: pd.DataFrame) -> tuple[int, float]:
    x = summary_df["actual_n_clusters"].to_numpy(dtype=float)
    y = summary_df["dtw_dispersion"].to_numpy(dtype=float)

    x_norm = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y_norm = (y - y.min()) / max(y.max() - y.min(), 1e-12)

    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])

    numerator = np.abs(
        (p2[1] - p1[1]) * x_norm
        - (p2[0] - p1[0]) * y_norm
        + p2[0] * p1[1]
        - p2[1] * p1[0]
    )

    denominator = np.sqrt(
        (p2[1] - p1[1]) ** 2
        + (p2[0] - p1[0]) ** 2
    ) + 1e-12

    distances = numerator / denominator
    best_idx = int(np.argmax(distances))

    best_k = int(x[best_idx])
    best_wcss = float(y[best_idx])

    return best_k, best_wcss


def plot_wcss_elbow(summary_df: pd.DataFrame, best_k: int, best_wcss: float) -> None:
    plt.figure(figsize=(9, 5))

    plt.plot(
        summary_df["actual_n_clusters"],
        summary_df["dtw_dispersion"],
        marker="o",
        linewidth=2,
    )

    plt.axvline(
        x=best_k,
        linestyle="--",
        linewidth=2,
        label=f"Best k = {best_k}",
    )

    plt.scatter(
        [best_k],
        [best_wcss],
        s=120,
        zorder=5,
    )

    plt.title("Elbow Method: WCSS vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS / WCSSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_sku_distribution(cluster_df: pd.DataFrame) -> pd.DataFrame:
    cluster_distribution = (
        cluster_df.groupby("cluster", as_index=False)
        .agg(n_sku=("item_code", "count"))
        .sort_values("cluster")
    )

    cluster_distribution["percentage"] = (
        cluster_distribution["n_sku"]
        / cluster_distribution["n_sku"].sum()
        * 100
    )

    plt.figure(figsize=(9, 5))

    plt.bar(
        cluster_distribution["cluster"].astype(str),
        cluster_distribution["n_sku"],
    )

    plt.title("Distribution of SKUs Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Number of SKUs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return cluster_distribution


def summarize_raw_order_distribution(cluster_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["cluster", "item_code", "total_order_frequency", "active_days"]

    missing_columns = [
        col for col in required_columns
        if col not in cluster_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns for raw interpretation: {missing_columns}"
        )

    agg_dict = {
        "n_sku": ("item_code", "count"),
        "mean_total_order_frequency": ("total_order_frequency", "mean"),
        "median_total_order_frequency": ("total_order_frequency", "median"),
        "mean_active_days": ("active_days", "mean"),
        "median_active_days": ("active_days", "median"),
    }

    if "total_quantity" in cluster_df.columns:
        agg_dict["mean_total_quantity"] = ("total_quantity", "mean")
        agg_dict["median_total_quantity"] = ("total_quantity", "median")

    if "mean_quantity_per_order" in cluster_df.columns:
        agg_dict["mean_quantity_per_order"] = ("mean_quantity_per_order", "mean")

    raw_summary = (
        cluster_df.groupby("cluster", as_index=False)
        .agg(**agg_dict)
        .sort_values("cluster")
    )

    raw_summary["sku_percentage"] = (
        raw_summary["n_sku"] / raw_summary["n_sku"].sum() * 100
    )

    return raw_summary


def build_daily_order_frequency_matrix(
    order_df: pd.DataFrame,
    date_col: str = "created_at",
    order_id_col: str = "order_id",
) -> pd.DataFrame:
    order_df = order_df.copy()
    order_df["item_code"] = order_df["item_code"].astype(str)

    order_df[date_col] = pd.to_datetime(
        order_df[date_col],
        dayfirst=True,
        errors="coerce",
    ).dt.normalize()

    order_df = order_df.dropna(subset=[date_col, order_id_col, "item_code"])

    daily_freq = (
        order_df.groupby(["item_code", date_col], as_index=False)[order_id_col]
        .nunique()
        .rename(columns={order_id_col: "daily_order_frequency"})
    )

    freq_matrix = daily_freq.pivot_table(
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

    freq_matrix = freq_matrix.reindex(columns=full_dates, fill_value=0.0)

    return freq_matrix.sort_index()


def row_sum_normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    row_sum = matrix.sum(axis=1).replace(0, np.nan)
    normalized = matrix.div(row_sum, axis=0).fillna(0.0)
    return normalized


def calculate_cluster_mean_pattern(
    matrix: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    cluster_df = cluster_df.copy()
    cluster_df["item_code"] = cluster_df["item_code"].astype(str)

    cluster_map = cluster_df.set_index("item_code")["cluster"]

    common_items = matrix.index.intersection(cluster_map.index)

    matrix = matrix.loc[common_items]
    cluster_map = cluster_map.loc[common_items]

    cluster_pattern = matrix.groupby(cluster_map).mean()
    cluster_pattern.index.name = "cluster"

    return cluster_pattern.sort_index()


def plot_cluster_pattern_heatmap(
    cluster_pattern: pd.DataFrame,
    title: str,
    colorbar_label: str,
) -> None:
    plt.figure(figsize=(14, 6))

    plt.imshow(
        cluster_pattern.to_numpy(dtype=float),
        aspect="auto",
    )

    plt.colorbar(label=colorbar_label)

    plt.yticks(
        ticks=np.arange(len(cluster_pattern.index)),
        labels=[f"Cluster {c}" for c in cluster_pattern.index],
    )

    plt.xticks(
        ticks=np.arange(len(cluster_pattern.columns)),
        labels=[d.strftime("%Y-%m-%d") for d in cluster_pattern.columns],
        rotation=45,
        ha="right",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.show()


def plot_top_cluster_patterns(
    cluster_pattern: pd.DataFrame,
    cluster_df: pd.DataFrame,
    title_prefix: str,
    ylabel: str,
    top_n: int = 5,
) -> None:
    cluster_sizes = (
        cluster_df.groupby("cluster", as_index=False)
        .agg(n_sku=("item_code", "count"))
        .sort_values("n_sku", ascending=False)
    )

    top_clusters = cluster_sizes.head(top_n)["cluster"].tolist()

    for cluster_id in top_clusters:
        if cluster_id not in cluster_pattern.index:
            continue

        row = cluster_pattern.loc[cluster_id]

        plt.figure(figsize=(10, 4))

        plt.plot(
            row.index,
            row.values,
            marker="o",
            linewidth=2,
        )

        plt.title(f"{title_prefix}, Cluster {cluster_id}")
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    cluster_df = pd.read_csv(RESULTS_PATH, sep=";")
    summary_df = pd.read_csv(SUMMARY_PATH, sep=";")
    orders_df = pd.read_csv(ORDERS_PATH, sep=";")

    cluster_df["item_code"] = cluster_df["item_code"].astype(str)
    orders_df["item_code"] = orders_df["item_code"].astype(str)

    best_k, best_wcss = find_elbow_k(summary_df)

    print(f"Best k from elbow method: {best_k}")
    print(f"WCSS at best k: {best_wcss:,.4f}")

    plot_wcss_elbow(summary_df, best_k, best_wcss)

    cluster_distribution = plot_sku_distribution(cluster_df)
    print("\nSKU distribution per cluster:")
    print(cluster_distribution)

    raw_summary = summarize_raw_order_distribution(cluster_df)
    print("\nRaw order distribution per cluster:")
    print(raw_summary)

    freq_matrix = build_daily_order_frequency_matrix(
        orders_df,
        date_col="created_at",
        order_id_col="order_id",
    )
    
    normalized_freq_matrix = row_sum_normalize_matrix(freq_matrix)

    normalized_order_freq_pattern = calculate_cluster_mean_pattern(
        normalized_freq_matrix,
        cluster_df,
    )

    print("\nNormalized cluster pattern:")
    print(normalized_order_freq_pattern)

    plot_top_cluster_patterns(
        normalized_order_freq_pattern,
        cluster_df,
        title_prefix="Normalized Daily Order-Frequency Pattern",
        ylabel="Mean normalized order frequency",
        top_n=5,
    )


if __name__ == "__main__":
    main()