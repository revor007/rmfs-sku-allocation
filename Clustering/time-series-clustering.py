from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from dtaidistance import dtw


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from experiment_context import load_experiment_context

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "time-series-clustering-results.csv"
SUMMARY_PATH = OUTPUT_DIR / "time-series-clustering-summary.csv"
DAILY_FREQ_TABLE_PATH = OUTPUT_DIR / "daily-order-frequency-long.csv"
RAW_FREQ_MATRIX_PATH = OUTPUT_DIR / "daily-order-frequency-matrix-raw.csv"
NORMALIZED_FREQ_MATRIX_PATH = OUTPUT_DIR / "daily-order-frequency-matrix-normalized.csv"
DTW_DISTANCE_MATRIX_PATH = OUTPUT_DIR / "dtw-sku-distance-matrix.csv"
SAMPLE_WARPING_PATH_PATH = OUTPUT_DIR / "sample-dtw-warping-path-matrix.csv"
DENDROGRAM_PATH = OUTPUT_DIR / "time-series-clustering-dendrogram.png"
SAMPLE_ALIGNMENT_PLOT_PATH = OUTPUT_DIR / "sample-dtw-alignment-plot.png"
SAMPLE_BEST_PATH_PATH = OUTPUT_DIR / "sample-dtw-best-path.csv"

DTW_WINDOW = 1
MAX_CANDIDATE_K = 60
DENDROGRAM_LAST_P = 50


def build_daily_order_frequency_table(
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

    if order_df.empty:
        raise ValueError("No valid dated order rows remain after parsing created_at.")

    daily_freq = (
        order_df.groupby(["item_code", date_col], as_index=False)[order_id_col]
        .nunique()
        .rename(
            columns={
                date_col: "order_date",
                order_id_col: "daily_order_frequency",
            }
        )
        .sort_values(["item_code", "order_date"])
        .reset_index(drop=True)
    )
    return daily_freq


def build_daily_order_frequency_matrix(
    daily_freq: pd.DataFrame,
) -> pd.DataFrame:
    if daily_freq.empty:
        raise ValueError("Daily order-frequency table is empty.")

    pivot = daily_freq.pivot_table(
        index="item_code",
        columns="order_date",
        values="daily_order_frequency",
        fill_value=0.0,
    )

    full_dates = pd.date_range(
        daily_freq["order_date"].min(),
        daily_freq["order_date"].max(),
        freq="D",
    )

    pivot = pivot.reindex(columns=full_dates, fill_value=0.0)

    return pivot.sort_index()


def row_sum_normalize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    row_sum = matrix.sum(axis=1).replace(0.0, np.nan)
    normalized = matrix.div(row_sum, axis=0).fillna(0.0)
    return normalized


def ensure_writable_double_array(values: np.ndarray) -> np.ndarray:
    return np.require(
        np.array(values, dtype=np.double, copy=True),
        dtype=np.double,
        requirements=["C", "W"],
    )


def convert_matrix_to_dtw_series(freq_matrix: pd.DataFrame) -> np.ndarray:
    return ensure_writable_double_array(freq_matrix.to_numpy(dtype=np.double))

def plot_sample_dtw_alignment(
    series_i: np.ndarray,
    series_j: np.ndarray,
    best_path: list[tuple[int, int]],
    sample_item_i: str,
    sample_item_j: str,
    dtw_distance: float,
) -> None:
    series_i = np.asarray(series_i, dtype=float)
    series_j = np.asarray(series_j, dtype=float)

    n = len(series_i)
    m = len(series_j)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 9),
        gridspec_kw={"height_ratios": [1, 1.4]},
    )

    ax_matrix = axes[0]
    ax_series = axes[1]

    path_i = [p[0] for p in best_path]
    path_j = [p[1] for p in best_path]

    ax_matrix.plot(path_i, path_j, linewidth=2)
    ax_matrix.scatter(path_i, path_j, s=20)

    ax_matrix.set_title(
        f"DTW Warping Path Matrix\n"
        f"SKU {sample_item_i} vs SKU {sample_item_j}, "
        f"DTW distance = {dtw_distance:.6f}, window = {DTW_WINDOW}"
    )
    ax_matrix.set_xlabel(f"Time index, SKU {sample_item_i}")
    ax_matrix.set_ylabel(f"Time index, SKU {sample_item_j}")
    ax_matrix.set_xlim(-0.5, n - 0.5)
    ax_matrix.set_ylim(-0.5, m - 0.5)
    ax_matrix.grid(True, alpha=0.3)

    x_i = np.arange(n)
    x_j = np.arange(m)

    ax_series.plot(
        x_i,
        series_i,
        marker="o",
        linewidth=2,
        label=f"SKU {sample_item_i}",
    )

    ax_series.plot(
        x_j,
        series_j,
        marker="o",
        linewidth=2,
        label=f"SKU {sample_item_j}",
    )

    for i, j in best_path:
        ax_series.plot(
            [i, j],
            [series_i[i], series_j[j]],
            color="black",
            alpha=0.45,
            linewidth=1,
        )

    ax_series.set_title("Matched Points Based on DTW Alignment")
    ax_series.set_xlabel("Time index")
    ax_series.set_ylabel("Normalized daily order frequency")
    ax_series.grid(True, alpha=0.3)
    ax_series.legend()

    plt.tight_layout()
    plt.savefig(SAMPLE_ALIGNMENT_PLOT_PATH, dpi=300)
    plt.close()

def select_representative_dtw_sample_pair(
    series: np.ndarray,
    item_index: pd.Index,
    labels: np.ndarray,
) -> tuple[str, str]:
    item_codes = pd.Index(item_index.astype(str))
    series_df = pd.DataFrame(series, index=item_codes)
    cluster_df = pd.DataFrame(
        {
            "item_code": item_codes,
            "cluster": labels.astype(int),
        }
    )

    cluster_sizes = cluster_df["cluster"].value_counts()
    target_cluster = None
    for cluster_id in cluster_sizes.index:
        if int(cluster_sizes.loc[cluster_id]) >= 2:
            target_cluster = int(cluster_id)
            break

    if target_cluster is None:
        return str(item_codes[0]), str(item_codes[1])

    cluster_item_codes = cluster_df.loc[
        cluster_df["cluster"] == target_cluster,
        "item_code",
    ]
    cluster_matrix = series_df.loc[cluster_item_codes]
    mean_row = cluster_matrix.mean(axis=0)
    distance_to_mean = ((cluster_matrix - mean_row) ** 2).sum(axis=1) ** 0.5
    selected_items = distance_to_mean.sort_values().index.astype(str).tolist()

    if len(selected_items) < 2:
        return str(item_codes[0]), str(item_codes[1])

    return selected_items[0], selected_items[1]


def save_sample_warping_path_matrix(
    series: np.ndarray,
    item_index: pd.Index,
    sample_item_i: str,
    sample_item_j: str,
) -> None:
    if series.shape[0] < 2:
        return

    item_codes = pd.Index(item_index.astype(str))
    if sample_item_i not in item_codes or sample_item_j not in item_codes:
        raise ValueError(
            f"Sample items must exist in the clustered series. "
            f"Missing: {sample_item_i}, {sample_item_j}."
        )

    sample_i = int(item_codes.get_loc(sample_item_i))
    sample_j = int(item_codes.get_loc(sample_item_j))

    series_i = ensure_writable_double_array(series[sample_i])
    series_j = ensure_writable_double_array(series[sample_j])

    distance, paths = dtw.warping_paths_fast(
        series_i,
        series_j,
        window=DTW_WINDOW,
        use_pruning=False,
    )

    best_path = dtw.best_path(paths)

    path_df = pd.DataFrame(
        best_path,
        columns=["series_i_time_index", "series_j_time_index"],
    )

    path_df.insert(0, "sample_item_i", str(item_index[sample_i]))
    path_df.insert(1, "sample_item_j", str(item_index[sample_j]))
    path_df.insert(2, "dtw_distance", float(distance))

    path_df.to_csv(
        SAMPLE_BEST_PATH_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    paths_df = pd.DataFrame(paths)
    paths_df.insert(0, "sample_item_i", str(item_index[sample_i]))
    paths_df.insert(1, "sample_item_j", str(item_index[sample_j]))
    paths_df.insert(2, "dtw_distance", float(distance))
    paths_df.to_csv(
        SAMPLE_WARPING_PATH_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    plot_sample_dtw_alignment(
        series_i=series_i,
        series_j=series_j,
        best_path=best_path,
        sample_item_i=str(item_index[sample_i]),
        sample_item_j=str(item_index[sample_j]),
        dtw_distance=float(distance),
    )


def compute_dtw_distance_matrices(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    series = ensure_writable_double_array(series)
    condensed_dist = dtw.distance_matrix_fast(
        series,
        window=DTW_WINDOW,
        use_pruning=False,
        parallel=True,
        compact=True,
    )

    condensed_dist = np.asarray(condensed_dist, dtype=float)
    square_dist = squareform(condensed_dist)

    return condensed_dist, square_dist


def save_full_dtw_distance_matrix(
    distance_matrix: np.ndarray,
    item_index: pd.Index,
) -> None:
    item_codes = pd.Index(item_index.astype(str))
    distance_df = pd.DataFrame(
        distance_matrix,
        index=item_codes,
        columns=item_codes,
    )
    distance_df.to_csv(
        DTW_DISTANCE_MATRIX_PATH,
        sep=";",
        encoding="utf-8-sig",
        float_format="%.6f",
    )


def within_cluster_dtw_dispersion(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    total = 0.0

    for cluster_id in np.unique(labels):
        idx = np.where(labels == cluster_id)[0]

        if len(idx) <= 1:
            continue

        cluster_dist = distance_matrix[np.ix_(idx, idx)]
        medoid_local_idx = int(np.argmin(cluster_dist.sum(axis=1)))
        distances_to_medoid = cluster_dist[medoid_local_idx]
        total += float((distances_to_medoid ** 2).sum())

    return total


def calculate_elbow_distances(summary: pd.DataFrame) -> np.ndarray:
    x = summary["actual_n_clusters"].to_numpy(dtype=float)
    y = summary["dtw_dispersion"].to_numpy(dtype=float)

    # Use the raw elbow geometry on the original axes instead of
    # re-scaling k and dispersion into [0, 1].
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    numerator = np.abs(
        (p2[1] - p1[1]) * x
        - (p2[0] - p1[0]) * y
        + p2[0] * p1[1]
        - p2[1] * p1[0]
    )

    denominator = (
        np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
        + 1e-12
    )

    return numerator / denominator


def plot_dendrogram_tree(Z: np.ndarray) -> None:
    plt.figure(figsize=(14, 7))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=DENDROGRAM_LAST_P,
        leaf_rotation=90,
        leaf_font_size=8,
        show_contracted=True,
    )
    plt.title("Hierarchical Clustering Dendrogram, Truncated View")
    plt.xlabel("Clustered SKU group")
    plt.ylabel("DTW distance")
    plt.tight_layout()
    plt.savefig(DENDROGRAM_PATH, dpi=300)
    plt.close()


def choose_cluster_count_dtw(
    series: np.ndarray,
    item_index: pd.Index,
) -> tuple[np.ndarray, int, pd.DataFrame]:
    max_candidates = min(MAX_CANDIDATE_K, series.shape[0] - 1)

    if max_candidates < 2:
        labels = np.ones(series.shape[0], dtype=np.int32)
        sample_item_i, sample_item_j = select_representative_dtw_sample_pair(
            series,
            item_index,
            labels,
        )
        save_sample_warping_path_matrix(
            series,
            item_index,
            sample_item_i,
            sample_item_j,
        )
        summary = pd.DataFrame(
            [
                {
                    "requested_k": 1,
                    "actual_n_clusters": 1,
                    "silhouette_score": np.nan,
                    "dtw_dispersion": 0.0,
                    "elbow_distance": 0.0,
                    "selected": True,
                }
            ]
        )
        return labels, 1, summary

    condensed_dist, distance_matrix = compute_dtw_distance_matrices(series)
    save_full_dtw_distance_matrix(distance_matrix, item_index)

    Z = linkage(condensed_dist, method="average")
    plot_dendrogram_tree(Z)

    records = []

    for requested_k in range(2, max_candidates + 1):
        labels = fcluster(Z, t=requested_k, criterion="maxclust")
        actual_k = int(np.unique(labels).size)

        if actual_k < 2 or actual_k >= len(series):
            continue

        records.append(
            {
                "requested_k": requested_k,
                "actual_n_clusters": actual_k,
                "silhouette_score": float(
                    silhouette_score(distance_matrix, labels, metric="precomputed")
                ),
                "dtw_dispersion": float(
                    within_cluster_dtw_dispersion(distance_matrix, labels)
                ),
            }
        )

    if not records:
        labels = np.ones(series.shape[0], dtype=np.int32)
        sample_item_i, sample_item_j = select_representative_dtw_sample_pair(
            series,
            item_index,
            labels,
        )
        save_sample_warping_path_matrix(
            series,
            item_index,
            sample_item_i,
            sample_item_j,
        )
        summary = pd.DataFrame(
            [
                {
                    "requested_k": 1,
                    "actual_n_clusters": 1,
                    "silhouette_score": np.nan,
                    "dtw_dispersion": 0.0,
                    "elbow_distance": 0.0,
                    "selected": True,
                }
            ]
        )
        return labels, 1, summary

    summary = pd.DataFrame(records)
    summary = summary.drop_duplicates(subset=["actual_n_clusters"]).reset_index(drop=True)

    distances = calculate_elbow_distances(summary)
    best_idx = int(np.argmax(distances))

    best_requested_k = int(summary.loc[best_idx, "requested_k"])
    best_actual_k = int(summary.loc[best_idx, "actual_n_clusters"])

    final_labels = fcluster(Z, t=best_requested_k, criterion="maxclust")
    final_actual_k = int(np.unique(final_labels).size)

    summary["elbow_distance"] = distances
    summary["selected"] = False
    summary.loc[best_idx, "selected"] = True

    if final_actual_k != best_actual_k:
        raise ValueError(
            f"Cluster mismatch: selected actual_n_clusters={best_actual_k}, "
            f"but final labels contain {final_actual_k} clusters. "
            f"requested_k={best_requested_k}."
        )

    sample_item_i, sample_item_j = select_representative_dtw_sample_pair(
        series,
        item_index,
        final_labels,
    )
    save_sample_warping_path_matrix(
        series,
        item_index,
        sample_item_i,
        sample_item_j,
    )

    return final_labels.astype(np.int32), best_actual_k, summary


def main() -> None:
    context = load_experiment_context(BASE_DIR)

    historical_orders = context.train_eligible_df[
        context.train_eligible_df["item_code"].isin(context.historical_skus)
    ].copy()

    if historical_orders.empty:
        raise ValueError("No historical pre-T orders remain for time-series clustering.")

    daily_freq_table = build_daily_order_frequency_table(
        historical_orders,
        date_col="created_at",
        order_id_col="order_id",
    )
    freq_matrix = build_daily_order_frequency_matrix(daily_freq_table)

    normalized_freq_matrix = row_sum_normalize_matrix(freq_matrix)

    daily_freq_table.to_csv(
        DAILY_FREQ_TABLE_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    freq_matrix.to_csv(RAW_FREQ_MATRIX_PATH, sep=";", encoding="utf-8-sig")
    normalized_freq_matrix.to_csv(
        NORMALIZED_FREQ_MATRIX_PATH,
        sep=";",
        encoding="utf-8-sig",
    )

    dtw_series = convert_matrix_to_dtw_series(normalized_freq_matrix)

    if dtw_series.shape[0] < 2:
        trivial_distance_matrix = np.zeros(
            (dtw_series.shape[0], dtw_series.shape[0]),
            dtype=float,
        )
        save_full_dtw_distance_matrix(trivial_distance_matrix, freq_matrix.index)
        cluster_labels = np.ones(dtw_series.shape[0], dtype=np.int32)
        best_k = 1
        summary = pd.DataFrame(
            [
                {
                    "requested_k": 1,
                    "actual_n_clusters": 1,
                    "silhouette_score": np.nan,
                    "dtw_dispersion": 0.0,
                    "elbow_distance": 0.0,
                    "selected": True,
                }
            ]
        )
    else:
        cluster_labels, best_k, summary = choose_cluster_count_dtw(
            dtw_series,
            freq_matrix.index,
        )
    cluster_df = pd.DataFrame(
        {
            "item_code": freq_matrix.index.astype(str),
            "cluster": cluster_labels,
            "total_order_frequency": freq_matrix.sum(axis=1).to_numpy(dtype=float),
            "active_days": (freq_matrix > 0).sum(axis=1).to_numpy(dtype=int),
        }
    )

    summary["clustering_method"] = f"dtw_window_{DTW_WINDOW}_average_linkage"
    summary["dtw_window"] = DTW_WINDOW
    summary["feature_used"] = "row_sum_normalized_daily_unique_order_frequency"
    summary["normalization_applied_to_clustering_input"] = "row_sum"
    summary["auxiliary_matrix_export"] = "raw_daily_unique_order_frequency"
    summary["distance_input"] = "precomputed_dtw_distance_matrix"
    summary["cluster_selection"] = "maximum_elbow_distance_from_raw_dtw_dispersion_curve"
    summary["silhouette_usage"] = "validation_only"

    cluster_df.to_csv(OUTPUT_PATH, index=False, sep=";", encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, sep=";", encoding="utf-8-sig")

    print(f"Cutoff ratio: {context.cutoff_ratio:.2f}")
    print(f"Cutoff timestamp: {context.cutoff_time}")
    print(f"Historical SKUs clustered: {len(cluster_df):,}")
    print(f"Best actual number of clusters: {best_k}")
    print(f"Actual clusters in saved labels: {cluster_df['cluster'].nunique()}")
    print("DTW clustering input used row-sum-normalized daily order-frequency values.")
    print(f"Daily order-frequency table saved to: {DAILY_FREQ_TABLE_PATH}")
    print(f"Raw daily order-frequency matrix saved to: {RAW_FREQ_MATRIX_PATH}")
    print(f"Normalized daily order-frequency matrix saved to: {NORMALIZED_FREQ_MATRIX_PATH}")
    print(f"Full DTW SKU distance matrix saved to: {DTW_DISTANCE_MATRIX_PATH}")
    print(f"Sample DTW warping path matrix saved to: {SAMPLE_WARPING_PATH_PATH}")
    print(f"Dendrogram saved to: {DENDROGRAM_PATH}")
    print(f"Cluster results saved to: {OUTPUT_PATH}")
    print(f"Cluster summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
