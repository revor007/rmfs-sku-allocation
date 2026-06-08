from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PREPROCESSING_DIR = Path("fcgma") / "Preprocessing"
ORDERS_PATH = PREPROCESSING_DIR / "訂單資料_train.csv"
RESULTS_PATH = Path("fcgma") / "Clustering" / "time-series-clustering-results.csv"
NORMALIZED_FREQ_MATRIX_PATH = (
    Path("fcgma") / "Clustering" / "daily-order-frequency-matrix-normalized.csv"
)
N_REPRESENTATIVE_SAMPLES = 2

OUTPUT_DIR = Path("cluster_pattern_plots")
OUTPUT_DIR.mkdir(exist_ok=True)


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


def row_sum_normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    row_sum = matrix.sum(axis=1).replace(0, np.nan)
    normalized = matrix.div(row_sum, axis=0).fillna(0.0)
    return normalized


def calculate_cluster_mean_patterns(
    freq_matrix: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_df = cluster_df.copy()
    cluster_df["item_code"] = cluster_df["item_code"].astype(str)

    cluster_map = cluster_df.set_index("item_code")["cluster"]

    common_items = freq_matrix.index.intersection(cluster_map.index)
    freq_matrix = freq_matrix.loc[common_items]
    cluster_map = cluster_map.loc[common_items]

    raw_cluster_pattern = freq_matrix.groupby(cluster_map).mean()

    normalized_freq_matrix = row_sum_normalize_matrix(freq_matrix)
    normalized_cluster_pattern = normalized_freq_matrix.groupby(cluster_map).mean()

    return raw_cluster_pattern, normalized_cluster_pattern


def load_normalized_frequency_matrix() -> pd.DataFrame:
    normalized_df = pd.read_csv(
        NORMALIZED_FREQ_MATRIX_PATH,
        sep=";",
        dtype={"item_code": str},
    )

    if "item_code" not in normalized_df.columns:
        first_column = normalized_df.columns[0]
        normalized_df = normalized_df.rename(columns={first_column: "item_code"})

    normalized_df["item_code"] = normalized_df["item_code"].astype(str)
    normalized_df = normalized_df.set_index("item_code")
    normalized_df.columns = pd.to_datetime(normalized_df.columns, errors="coerce")
    normalized_df = normalized_df.loc[:, normalized_df.columns.notna()]

    return normalized_df.sort_index().sort_index(axis=1)


def save_and_close(output_path: Path) -> None:
    plt.savefig(output_path, dpi=300)
    plt.close()


def clear_old_representative_outputs() -> None:
    for stale_file in OUTPUT_DIR.glob("representative_sample_*"):
        if stale_file.is_file():
            stale_file.unlink()
    for stale_file in OUTPUT_DIR.glob("sample_pattern_*"):
        if stale_file.is_file():
            stale_file.unlink()
    sample_selection_path = OUTPUT_DIR / "sample_selection.csv"
    if sample_selection_path.exists():
        sample_selection_path.unlink()


def plot_each_cluster_pattern(
    cluster_pattern: pd.DataFrame,
    ylabel: str,
    title_prefix: str,
    filename_prefix: str,
) -> None:
    for cluster_id, row in cluster_pattern.iterrows():
        plt.figure(figsize=(10, 5))

        plt.plot(
            row.index,
            row.values,
            marker="o",
            linewidth=2,
        )

        plt.title(f"{title_prefix} Cluster {cluster_id}")
        plt.xlabel("Date")
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f"{filename_prefix}_cluster_{cluster_id}.png"
        save_and_close(output_path)


def plot_all_cluster_patterns(
    cluster_pattern: pd.DataFrame,
    ylabel: str,
    title: str,
    output_filename: str,
) -> None:
    plt.figure(figsize=(12, 6))

    for cluster_id, row in cluster_pattern.iterrows():
        plt.plot(
            row.index,
            row.values,
            marker="o",
            linewidth=2,
            label=f"Cluster {cluster_id}",
        )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    save_and_close(output_path)


def select_representative_samples_for_one_cluster(
    normalized_freq_matrix: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cluster_id: int | None,
    top_n: int,
) -> pd.DataFrame:
    cluster_df = cluster_df.copy()
    cluster_df["item_code"] = cluster_df["item_code"].astype(str)
    cluster_map = cluster_df.set_index("item_code")["cluster"]

    common_items = normalized_freq_matrix.index.intersection(cluster_map.index)
    normalized_freq_matrix = normalized_freq_matrix.loc[common_items]
    cluster_map = cluster_map.loc[common_items]

    if cluster_id is None:
        cluster_id = int(cluster_map.value_counts().idxmax())

    item_codes = cluster_map[cluster_map == cluster_id].index
    if len(item_codes) == 0:
        return pd.DataFrame(
            columns=[
                "cluster",
                "item_code",
                "cluster_size",
                "distance_to_cluster_mean",
                "selection_rank",
            ]
        )

    cluster_matrix = normalized_freq_matrix.loc[item_codes]
    mean_row = cluster_matrix.mean(axis=0)
    distance_to_mean = ((cluster_matrix - mean_row) ** 2).sum(axis=1) ** 0.5
    distance_to_mean = distance_to_mean.sort_values()

    representative_df = pd.DataFrame(
        {
            "cluster": int(cluster_id),
            "item_code": distance_to_mean.index.astype(str),
            "cluster_size": int(len(item_codes)),
            "distance_to_cluster_mean": distance_to_mean.to_numpy(dtype=float),
        }
    ).head(top_n).reset_index(drop=True)

    representative_df["selection_rank"] = representative_df.index + 1
    return representative_df


def plot_sample_pattern(
    raw_freq_matrix: pd.DataFrame,
    item_code: str,
    cluster_id: int,
    cluster_size: int,
    distance_to_cluster_mean: float,
    output_filename: str,
) -> None:
    if item_code not in raw_freq_matrix.index:
        return

    row = raw_freq_matrix.loc[item_code]

    plt.figure(figsize=(12, 5))

    plt.plot(
        row.index,
        row.values,
        marker="o",
        linewidth=2,
        label=f"SKU {item_code}",
    )

    plt.title(
        f"SKU Sample Pattern\n"
        f"SKU {item_code}, Cluster {cluster_id}, Cluster size {cluster_size}, "
        f"distance {distance_to_cluster_mean:.4f}"
    )
    plt.xlabel("Date")
    plt.ylabel("Daily Order Frequency")
    plt.xticks(rotation=45)
    plt.ylim(bottom=0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    save_and_close(output_path)


def plot_cluster_pattern_heatmap(
    cluster_pattern: pd.DataFrame,
    title: str,
    output_filename: str,
) -> None:
    plt.figure(figsize=(12, 6))

    plt.imshow(
        cluster_pattern.to_numpy(dtype=float),
        aspect="auto",
    )

    plt.colorbar(label="Mean normalized order frequency")
    plt.yticks(
        ticks=np.arange(len(cluster_pattern.index)),
        labels=cluster_pattern.index.astype(str),
    )
    plt.xticks(
        ticks=np.arange(len(cluster_pattern.columns)),
        labels=[col.strftime("%Y-%m-%d") for col in cluster_pattern.columns],
        rotation=45,
        ha="right",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Cluster")
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    save_and_close(output_path)


def summarize_cluster_distribution(cluster_df: pd.DataFrame) -> pd.DataFrame:
    distribution_df = (
        cluster_df.assign(
            item_code=cluster_df["item_code"].astype(str),
            cluster=cluster_df["cluster"].astype(int),
        )
        .groupby("cluster", as_index=False)
        .agg(n_skus=("item_code", "nunique"))
        .sort_values(["n_skus", "cluster"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total_skus = int(distribution_df["n_skus"].sum())
    distribution_df["share_of_skus"] = distribution_df["n_skus"] / max(total_skus, 1)
    distribution_df["cluster_rank_by_size"] = distribution_df.index + 1
    return distribution_df


def summarize_cluster_statistics(
    freq_matrix: pd.DataFrame,
    cluster_df: pd.DataFrame,
) -> pd.DataFrame:
    cluster_map_df = cluster_df.assign(
        item_code=cluster_df["item_code"].astype(str),
        cluster=cluster_df["cluster"].astype(int),
    )[["item_code", "cluster"]]

    common_items = freq_matrix.index.intersection(cluster_map_df["item_code"])
    if len(common_items) == 0:
        return pd.DataFrame()

    aligned_freq_matrix = freq_matrix.loc[common_items].copy()
    nonzero_mean = (
        aligned_freq_matrix.replace(0.0, np.nan)
        .mean(axis=1)
        .fillna(0.0)
    )

    item_metric_df = pd.DataFrame(
        {
            "item_code": aligned_freq_matrix.index.astype(str),
            "total_order_frequency": aligned_freq_matrix.sum(axis=1).to_numpy(dtype=float),
            "active_days": (aligned_freq_matrix > 0).sum(axis=1).to_numpy(dtype=int),
            "mean_daily_order_frequency": aligned_freq_matrix.mean(axis=1).to_numpy(dtype=float),
            "median_daily_order_frequency": aligned_freq_matrix.median(axis=1).to_numpy(dtype=float),
            "peak_daily_order_frequency": aligned_freq_matrix.max(axis=1).to_numpy(dtype=float),
            "mean_nonzero_daily_order_frequency": nonzero_mean.to_numpy(dtype=float),
        }
    )

    combined_df = item_metric_df.merge(cluster_map_df, on="item_code", how="left")
    stats_df = (
        combined_df.groupby("cluster", as_index=False)
        .agg(
            n_skus=("item_code", "nunique"),
            mean_total_order_frequency=("total_order_frequency", "mean"),
            median_total_order_frequency=("total_order_frequency", "median"),
            std_total_order_frequency=("total_order_frequency", "std"),
            min_total_order_frequency=("total_order_frequency", "min"),
            max_total_order_frequency=("total_order_frequency", "max"),
            mean_active_days=("active_days", "mean"),
            median_active_days=("active_days", "median"),
            std_active_days=("active_days", "std"),
            mean_daily_order_frequency=("mean_daily_order_frequency", "mean"),
            median_daily_order_frequency=("median_daily_order_frequency", "median"),
            mean_peak_daily_order_frequency=("peak_daily_order_frequency", "mean"),
            median_peak_daily_order_frequency=("peak_daily_order_frequency", "median"),
            mean_nonzero_daily_order_frequency=("mean_nonzero_daily_order_frequency", "mean"),
            median_nonzero_daily_order_frequency=("mean_nonzero_daily_order_frequency", "median"),
        )
        .sort_values(["n_skus", "cluster"], ascending=[False, True])
        .reset_index(drop=True)
    )

    total_skus = int(stats_df["n_skus"].sum())
    stats_df["share_of_skus"] = stats_df["n_skus"] / max(total_skus, 1)
    stats_df["cluster_rank_by_size"] = stats_df.index + 1
    stats_df = stats_df.fillna(0.0)
    return stats_df


def plot_cluster_distribution(
    distribution_df: pd.DataFrame,
    output_filename: str,
) -> None:
    if distribution_df.empty:
        return

    plt.figure(figsize=(12, 6))
    colors = ["#1f77b4"] + ["#9ecae1"] * max(len(distribution_df) - 1, 0)
    x_labels = distribution_df["cluster"].astype(str)
    bars = plt.bar(
        x_labels,
        distribution_df["n_skus"],
        color=colors[: len(distribution_df)],
        width=0.7,
    )

    max_count = max(distribution_df["n_skus"].max(), 1)
    for bar, row in zip(bars, distribution_df.itertuples(index=False), strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max_count * 0.02),
            f"{int(row.n_skus)}\n({row.share_of_skus:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title("Distribution of SKUs Across Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Number of SKUs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / output_filename
    save_and_close(output_path)


def plot_cluster_statistics_overview(
    stats_df: pd.DataFrame,
    output_filename: str,
) -> None:
    if stats_df.empty:
        return

    plot_df = stats_df.sort_values(["n_skus", "cluster"], ascending=[False, True]).copy()
    x_labels = plot_df["cluster"].astype(str).tolist()
    metrics = [
        ("n_skus", "Number of SKUs"),
        ("mean_total_order_frequency", "Mean Total Order Frequency"),
        ("median_total_order_frequency", "Median Total Order Frequency"),
        ("mean_active_days", "Mean Active Days"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (column, title) in zip(axes, metrics, strict=False):
        bars = ax.bar(x_labels, plot_df[column], color="#4c78a8", width=0.7)
        ax.set_title(title)
        ax.set_xlabel("Cluster")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        y_max = max(float(plot_df[column].max()), 1.0)
        ax.set_ylim(0, y_max * 1.18)

        for bar, row in zip(bars, plot_df.itertuples(index=False), strict=False):
            if column == "n_skus":
                label = f"{int(row.n_skus)}\n({row.share_of_skus:.1%})"
            else:
                label = f"{getattr(row, column):.2f}"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (y_max * 0.02),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[0].set_ylabel("Count")
    axes[1].set_ylabel("Orders")
    axes[2].set_ylabel("Orders")
    axes[3].set_ylabel("Days")

    fig.suptitle("Cluster-Level Statistics Overview", fontsize=14)
    fig.tight_layout()
    output_path = OUTPUT_DIR / output_filename
    save_and_close(output_path)


def main() -> None:
    orders_df = pd.read_csv(ORDERS_PATH, sep=";")
    cluster_df = pd.read_csv(RESULTS_PATH, sep=";")

    freq_matrix = build_daily_order_frequency_matrix(
        orders_df,
        date_col="created_at",
        order_id_col="order_id",
    )
    normalized_freq_matrix = row_sum_normalize_matrix(freq_matrix)

    if NORMALIZED_FREQ_MATRIX_PATH.exists():
        normalized_freq_matrix = load_normalized_frequency_matrix()

    raw_cluster_pattern, normalized_cluster_pattern = calculate_cluster_mean_patterns(
        freq_matrix,
        cluster_df,
    )

    raw_cluster_pattern.to_csv(
        OUTPUT_DIR / "raw_mean_daily_order_frequency_per_cluster.csv",
        sep=";",
        encoding="utf-8-sig",
    )

    normalized_cluster_pattern.to_csv(
        OUTPUT_DIR / "normalized_mean_daily_order_frequency_per_cluster.csv",
        sep=";",
        encoding="utf-8-sig",
    )

    cluster_distribution_df = summarize_cluster_distribution(cluster_df)
    cluster_distribution_df.to_csv(
        OUTPUT_DIR / "sku_distribution_per_cluster.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    cluster_stats_df = summarize_cluster_statistics(
        freq_matrix=freq_matrix,
        cluster_df=cluster_df,
    )
    cluster_stats_df.to_csv(
        OUTPUT_DIR / "cluster_statistics_per_cluster.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    plot_all_cluster_patterns(
        raw_cluster_pattern,
        ylabel="Mean Daily Order Frequency",
        title="Raw Mean Daily Order Frequency Pattern by Cluster",
        output_filename="all_clusters_raw_mean_pattern.png",
    )

    plot_each_cluster_pattern(
        raw_cluster_pattern,
        ylabel="Mean Daily Order Frequency",
        title_prefix="Raw Mean Daily Order Frequency Pattern",
        filename_prefix="raw_mean_pattern",
    )

    plot_cluster_pattern_heatmap(
        normalized_cluster_pattern,
        title="Normalized Mean Order-Frequency Pattern by Cluster",
        output_filename="normalized_cluster_pattern_heatmap.png",
    )

    plot_cluster_distribution(
        cluster_distribution_df,
        output_filename="sku_distribution_per_cluster.png",
    )

    plot_cluster_statistics_overview(
        cluster_stats_df,
        output_filename="cluster_statistics_overview.png",
    )

    clear_old_representative_outputs()

    representative_samples = select_representative_samples_for_one_cluster(
        normalized_freq_matrix=normalized_freq_matrix,
        cluster_df=cluster_df,
        cluster_id=None,
        top_n=N_REPRESENTATIVE_SAMPLES,
    )
    representative_samples.to_csv(
        OUTPUT_DIR / "sample_selection.csv",
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    for _, sample_row in representative_samples.iterrows():
        item_code = str(sample_row["item_code"])
        cluster_id = int(sample_row["cluster"])
        cluster_size = int(sample_row["cluster_size"])
        distance_to_cluster_mean = float(sample_row["distance_to_cluster_mean"])
        selection_rank = int(sample_row["selection_rank"])

        plot_sample_pattern(
            raw_freq_matrix=freq_matrix,
            item_code=item_code,
            cluster_id=cluster_id,
            cluster_size=cluster_size,
            distance_to_cluster_mean=distance_to_cluster_mean,
            output_filename=(
                f"sample_pattern_{selection_rank}_cluster_{cluster_id}_"
                f"sku_{item_code}.png"
            ),
        )

    print("Saved pattern visualizations to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
