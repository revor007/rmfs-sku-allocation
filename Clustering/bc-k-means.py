from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from experiment_context import find_column, load_experiment_context, normalize_item_code

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "bc-k-means-results.csv"
ALLOCATION_SUMMARY_PATH = OUTPUT_DIR / "bc-k-means-allocation-summary.csv"
ALLOCATION_CHART_PATH = OUTPUT_DIR / "bc-k-means-allocation-summary.png"
NEW_PRODUCTS_ONLY_PATH = OUTPUT_DIR / "bc-k-means-new-products-only.csv"
COMMON_SKU_FILTER_SUMMARY_PATH = OUTPUT_DIR / "bc-k-means-common-sku-filter-summary.csv"
COMMON_SKU_FILTER_DETAILS_PATH = OUTPUT_DIR / "bc-k-means-common-sku-filter-details.csv"
FIRST_EXAMPLE_CANDIDATE_PATH = OUTPUT_DIR / "bc-k-means-first-example-candidate-dataset.csv"
FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH = OUTPUT_DIR / "bc-k-means-first-example-diagnostic-summary.csv"
FIRST_EXAMPLE_FIRST_PASS_DATASET_PATH = OUTPUT_DIR / "bc-k-means-first-example-first-pass-encoded-dataset.csv"
FIRST_EXAMPLE_STANDARDIZED_ATTRIBUTES_PATH = (
    OUTPUT_DIR / "bc-k-means-first-example-standardized-attributes.csv"
)
FIRST_EXAMPLE_SKU_DISTANCE_PATH = OUTPUT_DIR / "bc-k-means-first-example-sku-distance-matrix.csv"
FIRST_EXAMPLE_CLUSTER_DISTANCE_PATH = OUTPUT_DIR / "bc-k-means-first-example-first-pass-cluster-distance-matrix.csv"
FIRST_EXAMPLE_CLUSTER_MEMBERSHIP_PATH = OUTPUT_DIR / "bc-k-means-first-example-first-pass-cluster-membership.csv"
FIRST_EXAMPLE_CLUSTER_SUMMARY_PATH = OUTPUT_DIR / "bc-k-means-first-example-first-pass-cluster-summary.csv"
FIRST_EXAMPLE_DENDROGRAM_DATASET_PATH = OUTPUT_DIR / "bc-k-means-first-example-cluster-dendrogram-dataset.csv"
FIRST_EXAMPLE_DENDROGRAM_PATH = OUTPUT_DIR / "bc-k-means-first-example-cluster-dendrogram.png"
DIAGNOSTIC_SAMPLE_NEW_SKU = "10002911001"

MAX_CLUSTER_SIZE = 2


def load_product_metadata(product_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        product_path,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        engine="python",
    ).copy()
    df["item_code"] = df["item_code"].map(normalize_item_code)
    return df


def resolve_minimum_inventory_path(base_dir: Path) -> Path:
    candidates = [
        Path(base_dir) / "minimum_inventory.csv",
        Path(base_dir) / "minimum_inventory_latest.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Could not locate minimum inventory input for BC-k-means filtering. Searched: {searched}"
        )
    return max(existing, key=lambda path: (path.stat().st_mtime, path.name))


def load_pairwise_sku_set(path: Path) -> set[str]:
    df = pd.read_csv(path, sep=";", decimal=",", engine="python", index_col=0)
    index_skus = {
        normalize_item_code(value)
        for value in df.index
        if normalize_item_code(value)
    }
    column_skus = {
        normalize_item_code(value)
        for value in df.columns
        if normalize_item_code(value)
    }
    return index_skus & column_skus


def load_minimum_inventory_sku_set(path: Path) -> set[str]:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig", engine="python")
    sku_col = find_column(df.columns, ["item_code", "Item Code"])
    return {
        sku
        for sku in df[sku_col].map(normalize_item_code)
        if sku
    }


def load_max_capacity_sku_set(path: Path) -> set[str]:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    sku_col = find_column(df.columns, ["item_code"])
    return {
        sku
        for sku in df[sku_col].map(normalize_item_code)
        if sku
    }


def build_common_optimizer_sku_filter(context, product_df: pd.DataFrame) -> dict:
    eligible_skus = {normalize_item_code(code) for code in context.eligible_skus if normalize_item_code(code)}
    jaccard_skus = load_pairwise_sku_set(BASE_DIR / "jaccard_similarity_matrix.csv")
    minimum_inventory_skus = load_minimum_inventory_sku_set(resolve_minimum_inventory_path(BASE_DIR))
    max_capacity_skus = load_max_capacity_sku_set(context.paths.max_capacity_path)

    common_optimizer_skus = (
        eligible_skus
        & jaccard_skus
        & minimum_inventory_skus
        & max_capacity_skus
    )

    historical_skus = sorted(
        code
        for code in context.historical_skus
        if normalize_item_code(code) in common_optimizer_skus
    )
    new_skus = sorted(
        code
        for code in context.new_skus
        if normalize_item_code(code) in common_optimizer_skus
    )
    filtered_product_df = product_df[product_df["item_code"].isin(common_optimizer_skus)].copy()

    original_historical_skus = sorted(
        normalize_item_code(code)
        for code in context.historical_skus
        if normalize_item_code(code)
    )
    original_new_skus = sorted(
        normalize_item_code(code)
        for code in context.new_skus
        if normalize_item_code(code)
    )

    status_rows = []
    for sku_code in sorted(set(original_historical_skus) | set(original_new_skus)):
        in_eligible = sku_code in eligible_skus
        in_jaccard = sku_code in jaccard_skus
        in_minimum_inventory = sku_code in minimum_inventory_skus
        in_max_capacity = sku_code in max_capacity_skus
        in_common_optimizer_universe = sku_code in common_optimizer_skus
        missing_sources = []
        if not in_eligible:
            missing_sources.append("eligible_skus")
        if not in_jaccard:
            missing_sources.append("jaccard_similarity_matrix")
        if not in_minimum_inventory:
            missing_sources.append("minimum_inventory")
        if not in_max_capacity:
            missing_sources.append("max_capacity")

        status_rows.append(
            {
                "item_code": sku_code,
                "sku_role_pre_filter": (
                    "historical" if sku_code in original_historical_skus else "new"
                ),
                "in_eligible_skus": int(in_eligible),
                "in_jaccard_similarity_matrix": int(in_jaccard),
                "in_minimum_inventory": int(in_minimum_inventory),
                "in_max_capacity": int(in_max_capacity),
                "in_common_optimizer_universe": int(in_common_optimizer_universe),
                "filter_status": "kept" if in_common_optimizer_universe else "dropped",
                "missing_sources": "|".join(missing_sources),
            }
        )

    status_df = pd.DataFrame(status_rows)
    summary_df = pd.DataFrame(
        [
            {
                "original_historical_skus": int(len(original_historical_skus)),
                "filtered_historical_skus": int(len(historical_skus)),
                "dropped_historical_skus": int(len(original_historical_skus) - len(historical_skus)),
                "original_new_skus": int(len(original_new_skus)),
                "filtered_new_skus": int(len(new_skus)),
                "dropped_new_skus": int(len(original_new_skus) - len(new_skus)),
                "common_optimizer_skus": int(len(common_optimizer_skus)),
                "filtered_product_rows": int(len(filtered_product_df)),
            }
        ]
    )

    return {
        "common_optimizer_skus": common_optimizer_skus,
        "historical_skus": historical_skus,
        "new_skus": new_skus,
        "product_df": filtered_product_df,
        "status_df": status_df,
        "summary_df": summary_df,
    }


def export_common_sku_filter_artifacts(status_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    summary_df.to_csv(
        COMMON_SKU_FILTER_SUMMARY_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    status_df.to_csv(
        COMMON_SKU_FILTER_DETAILS_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )


def prepare_candidate_frame(
    product_df: pd.DataFrame,
    new_code: str,
    historical_skus: set[str],
) -> pd.DataFrame:
    new_row = product_df[product_df["item_code"] == new_code].copy()
    if new_row.empty:
        return pd.DataFrame()

    category = new_row["category"].iloc[0]
    capacity_category = new_row["capacity_category"].iloc[0]

    historical_candidates = product_df[
        (product_df["item_code"].isin(historical_skus))
        & (product_df["category"] == category)
        & (product_df["capacity_category"] == capacity_category)
    ].copy()

    combined = pd.concat([new_row, historical_candidates], ignore_index=True)
    combined["is_new_product"] = np.where(combined["item_code"] == new_code, 1, 0)
    combined["item_role"] = np.where(combined["is_new_product"] == 1, "new", "historical")
    return combined


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Keep preprocessed_final.csv on its raw scale and transform only the
    # candidate set being clustered for the current new product.
    encoded = df.copy()
    raw_feature_columns = [
        "Promo",
        "capacity",
        "price_per_piece",
        "estimation_discount",
        "shelf_life",
    ]

    for column in raw_feature_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")

    encoded["shelf_life"] = np.ceil(encoded["shelf_life"])
    encoded["log_capacity"] = np.log1p(encoded["capacity"].clip(lower=0))
    encoded["log_price_per_piece"] = np.log1p(encoded["price_per_piece"].clip(lower=0))
    encoded["log_estimation_discount"] = np.log1p(
        encoded["estimation_discount"].clip(lower=0)
    )
    encoded["log_shelf_life"] = np.log1p(encoded["shelf_life"].clip(lower=0))

    feature_input_columns = [
        "Promo",
        "log_capacity",
        "log_price_per_piece",
        "log_estimation_discount",
        "log_shelf_life",
    ]

    encoded = encoded.dropna(subset=feature_input_columns).copy()
    if encoded.empty:
        return encoded, []

    scaler = StandardScaler()
    feature_columns = [
        "Promo_normalized",
        "log_capacity_normalized",
        "log_price_per_piece_normalized",
        "log_estimation_discount_normalized",
        "log_shelf_life_normalized",
    ]
    encoded[feature_columns] = scaler.fit_transform(encoded[feature_input_columns])

    return encoded, feature_columns


def build_feature_matrix_for_fallback(candidate_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    encoded = candidate_df.copy()
    raw_feature_columns = [
        "Promo",
        "capacity",
        "price_per_piece",
        "estimation_discount",
        "shelf_life",
    ]

    for column in raw_feature_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")

    historical_mask = encoded["is_new_product"] == 0
    for column in raw_feature_columns:
        fill_value = encoded.loc[historical_mask, column].median()
        if pd.isna(fill_value):
            fill_value = encoded[column].median()
        if pd.isna(fill_value):
            return pd.DataFrame(), []
        encoded[column] = encoded[column].fillna(fill_value)

    encoded["shelf_life"] = np.ceil(encoded["shelf_life"])
    encoded["log_capacity"] = np.log1p(encoded["capacity"].clip(lower=0))
    encoded["log_price_per_piece"] = np.log1p(encoded["price_per_piece"].clip(lower=0))
    encoded["log_estimation_discount"] = np.log1p(
        encoded["estimation_discount"].clip(lower=0)
    )
    encoded["log_shelf_life"] = np.log1p(encoded["shelf_life"].clip(lower=0))

    feature_input_columns = [
        "Promo",
        "log_capacity",
        "log_price_per_piece",
        "log_estimation_discount",
        "log_shelf_life",
    ]

    scaler = StandardScaler()
    feature_columns = [
        "Promo_normalized",
        "log_capacity_normalized",
        "log_price_per_piece_normalized",
        "log_estimation_discount_normalized",
        "log_shelf_life_normalized",
    ]
    encoded[feature_columns] = scaler.fit_transform(encoded[feature_input_columns])

    return encoded, feature_columns


def find_nearest_historical_candidate(candidate_df: pd.DataFrame, new_code: str) -> str | None:
    encoded_df, feature_columns = build_feature_matrix(candidate_df)
    if encoded_df.empty or not feature_columns:
        encoded_df, feature_columns = build_feature_matrix_for_fallback(candidate_df)

    new_rows = encoded_df[encoded_df["item_code"] == new_code]
    historical_rows = encoded_df[encoded_df["is_new_product"] == 0]
    if new_rows.empty or historical_rows.empty:
        encoded_df, feature_columns = build_feature_matrix_for_fallback(candidate_df)
        if encoded_df.empty or not feature_columns:
            return None
        new_rows = encoded_df[encoded_df["item_code"] == new_code]
        historical_rows = encoded_df[encoded_df["is_new_product"] == 0]

    if encoded_df.empty or not feature_columns:
        return None
    if new_rows.empty or historical_rows.empty:
        return None

    distances = cdist(
        new_rows[feature_columns].to_numpy(dtype=float),
        historical_rows[feature_columns].to_numpy(dtype=float),
        metric="euclidean",
    ).flatten()
    return historical_rows.iloc[int(np.argmin(distances))]["item_code"]


def run_kmeans_pass(
    encoded_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, KMeans | None, int, int]:
    if encoded_df.empty or not feature_columns:
        return encoded_df.copy(), None, 0, 0

    n_clusters = int(np.ceil(len(encoded_df) / MAX_CLUSTER_SIZE))
    n_unique_points = np.unique(
        encoded_df[feature_columns].to_numpy(dtype=float),
        axis=0,
    ).shape[0]

    if n_unique_points < n_clusters or n_clusters <= 0:
        return encoded_df.copy(), None, n_clusters, n_unique_points

    clustered_df = encoded_df.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustered_df["cluster"] = kmeans.fit_predict(clustered_df[feature_columns])

    return clustered_df, kmeans, n_clusters, n_unique_points


def export_new_products_only_dataset(
    product_df: pd.DataFrame,
    new_skus: list[str],
) -> pd.DataFrame:
    new_products_df = product_df[product_df["item_code"].isin(new_skus)].copy()
    new_products_df["is_new_product"] = 1
    new_products_df["item_role"] = "new"
    new_products_df.to_csv(
        NEW_PRODUCTS_ONLY_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    return new_products_df


def export_allocation_statistics(result_df: pd.DataFrame) -> pd.DataFrame:
    allocation_order = ["common_allocation", "random_allocation"]
    allocation_labels = {
        "common_allocation": "Common allocation",
        "random_allocation": "Random allocation",
    }
    allocation_colors = {
        "common_allocation": "#1f77b4",
        "random_allocation": "#d62728",
    }

    total_new_products = int(len(result_df))
    count_series = (
        result_df["allocation_type"].value_counts()
        if not result_df.empty
        else pd.Series(dtype=int)
    )

    summary_rows = []
    for allocation_type in allocation_order:
        count = int(count_series.get(allocation_type, 0))
        share = (count / total_new_products) if total_new_products else 0.0
        summary_rows.append(
            {
                "allocation_type": allocation_type,
                "allocation_label": allocation_labels[allocation_type],
                "new_product_count": count,
                "share_of_new_products": share,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        ALLOCATION_SUMMARY_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        summary_df["allocation_label"],
        summary_df["new_product_count"],
        color=[allocation_colors[row["allocation_type"]] for _, row in summary_df.iterrows()],
        width=0.55,
    )
    ax.set_title("BC-k-means allocation outcomes")
    ax.set_xlabel("Allocation outcome")
    ax.set_ylabel("Number of new products")
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    y_max = max(summary_df["new_product_count"].max(), 1)
    ax.set_ylim(0, y_max * 1.2)

    for bar, row in zip(bars, summary_df.itertuples(index=False), strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (y_max * 0.03),
            f"{int(row.new_product_count)}\n({row.share_of_new_products:.1%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(ALLOCATION_CHART_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary_df


def select_diagnostic_sample_new_sku(new_skus: list[str], preferred_code: str) -> str | None:
    if not new_skus:
        return None
    if preferred_code in set(new_skus):
        return preferred_code
    return new_skus[0]


def _build_display_label(row: pd.Series) -> str:
    label = f"{row['item_code']} [{str(row['item_role']).upper()}]"
    if "cluster" in row.index and pd.notna(row["cluster"]):
        label += f" | cluster_{int(row['cluster'])}"
    return label


def export_first_example_cluster_dendrogram(
    working_df: pd.DataFrame,
    feature_columns: list[str],
    new_code: str,
    new_product_cluster: int | None,
) -> None:
    if working_df.empty or not feature_columns:
        return

    dendrogram_df = working_df.drop_duplicates(subset="item_code").copy()
    dendrogram_df = dendrogram_df.sort_values(
        by=[
            "new_product_first",
            "same_cluster_as_new",
            "distance_to_new_product",
            "item_code",
        ]
    ).reset_index(drop=True)

    dendrogram_df.to_csv(
        FIRST_EXAMPLE_DENDROGRAM_DATASET_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    if len(dendrogram_df) < 2:
        return

    linkage_matrix = linkage(
        dendrogram_df[feature_columns].to_numpy(dtype=float),
        method="ward",
    )

    labels = []
    for row in dendrogram_df.itertuples(index=False):
        if row.item_code == new_code:
            labels.append(f"{row.item_code} [NEW]")
        elif hasattr(row, "cluster") and new_product_cluster is not None and row.cluster == new_product_cluster:
            labels.append(f"{row.item_code} [SAME_CLUSTER]")
        else:
            labels.append(f"{row.item_code} [OTHER_HISTORICAL]")

    figure_height = max(14, 0.16 * len(dendrogram_df))
    fig, ax = plt.subplots(figsize=(18, figure_height))
    dendrogram(
        linkage_matrix,
        labels=labels,
        orientation="left",
        leaf_font_size=5,
        ax=ax,
    )
    ax.set_title(f"Full dendrogram for diagnostic sample {new_code}")
    ax.set_xlabel("Ward linkage distance")
    ax.set_ylabel("SKU")

    for tick in ax.get_ymajorticklabels():
        label_text = tick.get_text()
        if "[NEW]" in label_text:
            tick.set_color("#b00020")
            tick.set_fontweight("bold")
        elif "[SAME_CLUSTER]" in label_text:
            tick.set_color("#005f73")
        else:
            tick.set_color("#4a4a4a")

    fig.tight_layout()
    fig.savefig(FIRST_EXAMPLE_DENDROGRAM_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_first_example_diagnostics(
    candidate_df: pd.DataFrame,
    new_code: str,
) -> dict:
    if candidate_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "new_product_code": new_code,
                    "status": "empty_candidate_dataset",
                    "candidate_rows": 0,
                    "historical_candidate_rows": 0,
                    "encoded_rows": 0,
                    "first_pass_requested_clusters": 0,
                    "first_pass_unique_points": 0,
                    "new_product_cluster": None,
                }
            ]
        )
        summary_df.to_csv(
            FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )
        return {"status": "empty_candidate_dataset"}

    candidate_export_df = candidate_df.copy()
    candidate_export_df.to_csv(
        FIRST_EXAMPLE_CANDIDATE_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    encoded_df, feature_columns = build_feature_matrix(candidate_df)
    clustered_first_pass_df, kmeans, requested_clusters, unique_points = run_kmeans_pass(
        encoded_df,
        feature_columns,
    )

    status = "ok"
    if encoded_df.empty or not feature_columns:
        status = "empty_encoded_dataset"
    elif kmeans is None:
        status = "first_pass_skipped_insufficient_unique_points"

    new_product_cluster = None
    if "cluster" in clustered_first_pass_df.columns:
        new_rows = clustered_first_pass_df[clustered_first_pass_df["item_code"] == new_code]
        if not new_rows.empty:
            new_product_cluster = int(new_rows["cluster"].iloc[0])

    summary_df = pd.DataFrame(
        [
            {
                "new_product_code": new_code,
                "status": status,
                "candidate_rows": int(len(candidate_df)),
                "historical_candidate_rows": int((candidate_df["is_new_product"] == 0).sum()),
                "encoded_rows": int(len(encoded_df)),
                "first_pass_requested_clusters": int(requested_clusters),
                "first_pass_unique_points": int(unique_points),
                "new_product_cluster": new_product_cluster,
            }
        ]
    )
    summary_df.to_csv(
        FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    if encoded_df.empty or not feature_columns:
        return {"status": status}

    standardized_columns = [
        "item_code",
        "item_role",
        "is_new_product",
        "log_capacity",
        "log_price_per_piece",
        "log_estimation_discount",
        "log_shelf_life",
        *feature_columns,
    ]
    standardized_export_df = encoded_df[standardized_columns].copy()
    standardized_export_df.to_csv(
        FIRST_EXAMPLE_STANDARDIZED_ATTRIBUTES_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    working_df = clustered_first_pass_df.copy()
    new_encoded_rows = working_df[working_df["item_code"] == new_code]
    if new_encoded_rows.empty:
        return {"status": "new_product_missing_after_encoding"}

    new_vector = new_encoded_rows[feature_columns].to_numpy(dtype=float)
    working_df["distance_to_new_product"] = cdist(
        working_df[feature_columns].to_numpy(dtype=float),
        new_vector,
        metric="euclidean",
    ).flatten()

    if "cluster" in working_df.columns and new_product_cluster is not None:
        working_df["same_cluster_as_new"] = np.where(
            working_df["cluster"] == new_product_cluster,
            0,
            1,
        )
    else:
        working_df["same_cluster_as_new"] = 1

    working_df["new_product_first"] = np.where(
        working_df["item_code"] == new_code,
        0,
        1,
    )

    working_df = working_df.sort_values(
        by=[
            "new_product_first",
            "same_cluster_as_new",
            "distance_to_new_product",
            "item_code",
        ]
    ).reset_index(drop=True)

    working_df.to_csv(
        FIRST_EXAMPLE_FIRST_PASS_DATASET_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    display_labels = working_df.apply(_build_display_label, axis=1)
    distance_matrix = cdist(
        working_df[feature_columns].to_numpy(dtype=float),
        working_df[feature_columns].to_numpy(dtype=float),
        metric="euclidean",
    )
    sku_distance_df = pd.DataFrame(
        distance_matrix,
        index=display_labels,
        columns=display_labels,
    )
    sku_distance_df.to_csv(
        FIRST_EXAMPLE_SKU_DISTANCE_PATH,
        sep=";",
        encoding="utf-8-sig",
    )

    membership_columns = [
        "item_code",
        "item_role",
        "distance_to_new_product",
    ]
    if "cluster" in working_df.columns:
        membership_columns.insert(2, "cluster")
    cluster_membership_df = working_df[membership_columns].copy()
    cluster_membership_df.to_csv(
        FIRST_EXAMPLE_CLUSTER_MEMBERSHIP_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    if "cluster" in working_df.columns:
        cluster_summary_df = (
            working_df.groupby("cluster", as_index=False)
            .agg(
                n_items=("item_code", "size"),
                n_new_items=("is_new_product", "sum"),
                item_codes=("item_code", lambda s: " | ".join(s.astype(str))),
            )
            .sort_values("cluster")
        )
        cluster_summary_df["contains_example_new_product"] = (
            cluster_summary_df["cluster"] == new_product_cluster
        )
        cluster_summary_df.to_csv(
            FIRST_EXAMPLE_CLUSTER_SUMMARY_PATH,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )

    if kmeans is not None:
        centroid_labels = [f"cluster_{cluster_id}" for cluster_id in range(kmeans.n_clusters)]
        centroid_distance_df = pd.DataFrame(
            cdist(kmeans.cluster_centers_, kmeans.cluster_centers_, metric="euclidean"),
            index=centroid_labels,
            columns=centroid_labels,
        )
        centroid_distance_df.to_csv(
            FIRST_EXAMPLE_CLUSTER_DISTANCE_PATH,
            sep=";",
            encoding="utf-8-sig",
        )

    export_first_example_cluster_dendrogram(
        working_df,
        feature_columns,
        new_code,
        new_product_cluster,
    )

    return {"status": status}


def classify_new_product(candidate_df: pd.DataFrame, new_code: str) -> dict:
    if candidate_df.empty:
        return {
            "new_product_code": new_code,
            "allocation_type": "random_allocation",
            "corresponding_historical_product": None,
        }

    fallback_hist_code = find_nearest_historical_candidate(candidate_df, new_code)
    nearest_hist_code = None
    current_df = candidate_df.copy()

    while True:
        historical_candidates = current_df[current_df["is_new_product"] == 0].copy()
        if historical_candidates.empty:
            return {
                "new_product_code": new_code,
                "allocation_type": (
                    "common_allocation" if fallback_hist_code is not None else "random_allocation"
                ),
                "corresponding_historical_product": fallback_hist_code,
            }

        if len(current_df) <= MAX_CLUSTER_SIZE:
            final_cluster = current_df.copy()
            break

        encoded_df, feature_columns = build_feature_matrix(current_df)
        if encoded_df.empty or not feature_columns:
            final_cluster = current_df.copy()
            break

        new_encoded_rows = encoded_df[encoded_df["item_code"] == new_code]
        if new_encoded_rows.empty:
            final_cluster = current_df.copy()
            break

        clustered_df, kmeans, _, _ = run_kmeans_pass(encoded_df, feature_columns)
        if kmeans is None:
            final_cluster = encoded_df.copy()
            new_row = final_cluster[final_cluster["item_code"] == new_code]
            hist_rows = final_cluster[final_cluster["is_new_product"] == 0]
            if not new_row.empty and not hist_rows.empty:
                distances = cdist(
                    new_row[feature_columns].to_numpy(dtype=float),
                    hist_rows[feature_columns].to_numpy(dtype=float),
                    metric="euclidean",
                ).flatten()
                nearest_hist_code = hist_rows.iloc[int(np.argmin(distances))]["item_code"]
            break

        new_cluster = clustered_df.loc[new_encoded_rows.index, "cluster"].iloc[0]
        problem_cluster = clustered_df[clustered_df["cluster"] == new_cluster].copy()

        if len(problem_cluster) <= MAX_CLUSTER_SIZE:
            final_cluster = problem_cluster.copy()
            break

        if len(problem_cluster) >= len(current_df):
            final_cluster = problem_cluster.copy()
            hist_rows = final_cluster[final_cluster["is_new_product"] == 0]
            new_row = final_cluster[final_cluster["item_code"] == new_code]
            if not new_row.empty and not hist_rows.empty:
                distances = cdist(
                    new_row[feature_columns].to_numpy(dtype=float),
                    hist_rows[feature_columns].to_numpy(dtype=float),
                    metric="euclidean",
                ).flatten()
                nearest_hist_code = hist_rows.iloc[int(np.argmin(distances))]["item_code"]
            break

        current_df = problem_cluster.drop(columns=["cluster"], errors="ignore").copy()

    historical_rows = final_cluster[final_cluster["is_new_product"] == 0].copy()
    if len(final_cluster) < 2:
        allocation_type = "random_allocation"
        corresponding_historical_product = None
    elif len(final_cluster) == 2 and len(historical_rows) == 1:
        allocation_type = "common_allocation"
        corresponding_historical_product = historical_rows["item_code"].iloc[0]
    elif nearest_hist_code is not None:
        allocation_type = "common_allocation"
        corresponding_historical_product = nearest_hist_code
    else:
        allocation_type = "random_allocation"
        corresponding_historical_product = None

    if allocation_type == "random_allocation" and fallback_hist_code is not None:
        allocation_type = "common_allocation"
        corresponding_historical_product = fallback_hist_code

    return {
        "new_product_code": new_code,
        "allocation_type": allocation_type,
        "corresponding_historical_product": corresponding_historical_product,
    }


def main():
    context = load_experiment_context(BASE_DIR)
    product_df = load_product_metadata(context.paths.product_path)
    product_df = product_df[product_df["item_code"].isin(context.eligible_skus)].copy()

    common_filter = build_common_optimizer_sku_filter(context, product_df)
    export_common_sku_filter_artifacts(
        common_filter["status_df"],
        common_filter["summary_df"],
    )

    product_df = common_filter["product_df"]
    historical_skus = set(common_filter["historical_skus"])
    new_skus = list(common_filter["new_skus"])

    new_products_df = export_new_products_only_dataset(product_df, new_skus)

    if not new_skus:
        result_df = pd.DataFrame(
            columns=[
                "new_product_code",
                "allocation_type",
                "corresponding_historical_product",
            ]
        )
    else:
        diagnostic_new_code = select_diagnostic_sample_new_sku(
            new_skus,
            DIAGNOSTIC_SAMPLE_NEW_SKU,
        )
        first_candidate_df = prepare_candidate_frame(product_df, diagnostic_new_code, historical_skus)
        export_first_example_diagnostics(first_candidate_df, diagnostic_new_code)

        results = []
        for new_code in new_skus:
            candidate_df = prepare_candidate_frame(product_df, new_code, historical_skus)
            results.append(classify_new_product(candidate_df, new_code))
        result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_PATH, index=False, sep=";", encoding="utf-8-sig")
    allocation_summary_df = export_allocation_statistics(result_df)

    common_count = int(
        allocation_summary_df.loc[
            allocation_summary_df["allocation_type"] == "common_allocation",
            "new_product_count",
        ].iloc[0]
    )
    random_count = int(
        allocation_summary_df.loc[
            allocation_summary_df["allocation_type"] == "random_allocation",
            "new_product_count",
        ].iloc[0]
    )
    print(f"Cutoff ratio: {context.cutoff_ratio:.2f}")
    print(f"Cutoff timestamp: {context.cutoff_time}")
    print(f"Eligible master SKUs: {len(context.eligible_skus):,}")
    print(
        "Common-SKU filter: "
        f"historical {len(context.historical_skus):,}->{len(historical_skus):,}, "
        f"new {len(context.new_skus):,}->{len(new_skus):,}",
    )
    print(f"Historical SKUs: {len(context.historical_skus):,}")
    print(f"New SKUs: {len(context.new_skus):,}")
    print(f"New-products-only dataset saved to: {NEW_PRODUCTS_ONLY_PATH}")
    if new_skus:
        diagnostic_new_code = select_diagnostic_sample_new_sku(
            new_skus,
            DIAGNOSTIC_SAMPLE_NEW_SKU,
        )
        print(f"Preferred diagnostic sample new SKU: {DIAGNOSTIC_SAMPLE_NEW_SKU}")
        print(f"Diagnostic sample new SKU processed: {diagnostic_new_code}")
        if diagnostic_new_code != DIAGNOSTIC_SAMPLE_NEW_SKU:
            print("Preferred diagnostic sample was not available in this context, so the first available new SKU was used instead.")
        print(f"First example candidate dataset saved to: {FIRST_EXAMPLE_CANDIDATE_PATH}")
        print(f"First example diagnostic summary saved to: {FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH}")
        print(f"First example first-pass encoded dataset saved to: {FIRST_EXAMPLE_FIRST_PASS_DATASET_PATH}")
        print(f"First example standardized attributes saved to: {FIRST_EXAMPLE_STANDARDIZED_ATTRIBUTES_PATH}")
        print(f"First example SKU distance matrix saved to: {FIRST_EXAMPLE_SKU_DISTANCE_PATH}")
        print(f"First example cluster distance matrix saved to: {FIRST_EXAMPLE_CLUSTER_DISTANCE_PATH}")
        print(f"First example cluster membership saved to: {FIRST_EXAMPLE_CLUSTER_MEMBERSHIP_PATH}")
        print(f"First example cluster summary saved to: {FIRST_EXAMPLE_CLUSTER_SUMMARY_PATH}")
        print(f"First example cluster dendrogram dataset saved to: {FIRST_EXAMPLE_DENDROGRAM_DATASET_PATH}")
        print(f"First example cluster dendrogram saved to: {FIRST_EXAMPLE_DENDROGRAM_PATH}")
    print(f"Allocation summary saved to: {ALLOCATION_SUMMARY_PATH}")
    print(f"Allocation summary chart saved to: {ALLOCATION_CHART_PATH}")
    print(f"Common-allocation new SKUs: {common_count:,}")
    print(f"Random-allocation new SKUs: {random_count:,}")
    print(f"BC-k-means results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
