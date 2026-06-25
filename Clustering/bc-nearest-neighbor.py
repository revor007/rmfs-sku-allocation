from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from experiment_context import find_column, load_experiment_context, normalize_item_code

OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = OUTPUT_DIR / "bc-nearest-neighbor-results.csv"
ALLOCATION_SUMMARY_PATH = OUTPUT_DIR / "bc-nearest-neighbor-allocation-summary.csv"
ALLOCATION_CHART_PATH = OUTPUT_DIR / "bc-nearest-neighbor-allocation-summary.png"
NEW_PRODUCTS_ONLY_PATH = OUTPUT_DIR / "bc-nearest-neighbor-new-products-only.csv"
COMMON_SKU_FILTER_SUMMARY_PATH = OUTPUT_DIR / "bc-nearest-neighbor-common-sku-filter-summary.csv"
COMMON_SKU_FILTER_DETAILS_PATH = OUTPUT_DIR / "bc-nearest-neighbor-common-sku-filter-details.csv"
FIRST_EXAMPLE_CANDIDATE_PATH = OUTPUT_DIR / "bc-nearest-neighbor-first-example-candidate-dataset.csv"
FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH = OUTPUT_DIR / "bc-nearest-neighbor-first-example-diagnostic-summary.csv"
FIRST_EXAMPLE_FIRST_PASS_DATASET_PATH = OUTPUT_DIR / "bc-nearest-neighbor-first-example-nearest-ranked-candidates.csv"
FIRST_EXAMPLE_STANDARDIZED_ATTRIBUTES_PATH = (
    OUTPUT_DIR / "bc-nearest-neighbor-first-example-standardized-attributes.csv"
)
FIRST_EXAMPLE_SKU_DISTANCE_PATH = OUTPUT_DIR / "bc-nearest-neighbor-first-example-sku-distance-matrix.csv"
FIRST_EXAMPLE_NEAREST_HISTORICAL_RANKING_PATH = (
    OUTPUT_DIR / "bc-nearest-neighbor-first-example-nearest-historical-ranking.csv"
)
FIRST_EXAMPLE_NEAREST_HISTORICAL_SUMMARY_PATH = (
    OUTPUT_DIR / "bc-nearest-neighbor-first-example-nearest-historical-summary.csv"
)
DIAGNOSTIC_SAMPLE_NEW_SKU = "10002911001"


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
            f"Could not locate minimum inventory input for nearest-neighbor filtering. Searched: {searched}"
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
    ax.set_title("BC-nearest-neighbor allocation outcomes")
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
                    "used_fallback_features": False,
                    "nearest_historical_product": None,
                    "nearest_distance_to_new_product": None,
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
    if encoded_df.empty or not feature_columns:
        encoded_df, feature_columns = build_feature_matrix_for_fallback(candidate_df)
        used_fallback_features = True
    else:
        used_fallback_features = False

    status = "ok"
    nearest_historical_product = None
    nearest_distance = None

    if encoded_df.empty or not feature_columns:
        status = "empty_encoded_dataset"
        summary_df = pd.DataFrame(
            [
                {
                    "new_product_code": new_code,
                    "status": status,
                    "candidate_rows": int(len(candidate_df)),
                    "historical_candidate_rows": int((candidate_df["is_new_product"] == 0).sum()),
                    "encoded_rows": int(len(encoded_df)),
                    "used_fallback_features": bool(used_fallback_features),
                    "nearest_historical_product": nearest_historical_product,
                    "nearest_distance_to_new_product": nearest_distance,
                }
            ]
        )
        summary_df.to_csv(
            FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )
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

    new_rows = encoded_df[encoded_df["item_code"] == new_code].copy()
    historical_rows = encoded_df[encoded_df["is_new_product"] == 0].copy()
    if new_rows.empty:
        status = "new_product_missing_after_encoding"
    elif historical_rows.empty:
        status = "no_historical_candidates_after_encoding"

    if status != "ok":
        summary_df = pd.DataFrame(
            [
                {
                    "new_product_code": new_code,
                    "status": status,
                    "candidate_rows": int(len(candidate_df)),
                    "historical_candidate_rows": int((candidate_df["is_new_product"] == 0).sum()),
                    "encoded_rows": int(len(encoded_df)),
                    "used_fallback_features": bool(used_fallback_features),
                    "nearest_historical_product": nearest_historical_product,
                    "nearest_distance_to_new_product": nearest_distance,
                }
            ]
        )
        summary_df.to_csv(
            FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH,
            index=False,
            sep=";",
            encoding="utf-8-sig",
        )
        return {"status": status}

    new_vector = new_rows[feature_columns].to_numpy(dtype=float)
    distances = cdist(
        new_vector,
        historical_rows[feature_columns].to_numpy(dtype=float),
        metric="euclidean",
    ).flatten()
    nearest_idx = int(np.argmin(distances))
    nearest_historical_product = historical_rows.iloc[nearest_idx]["item_code"]
    nearest_distance = float(distances[nearest_idx])

    ranked_historical_df = historical_rows.copy()
    ranked_historical_df["distance_to_new_product"] = distances
    ranked_historical_df = ranked_historical_df.sort_values(
        by=["distance_to_new_product", "item_code"],
    ).reset_index(drop=True)
    ranked_historical_df["rank"] = np.arange(1, len(ranked_historical_df) + 1)
    ranked_historical_df["is_nearest_historical"] = np.where(
        ranked_historical_df["item_code"] == nearest_historical_product,
        1,
        0,
    )

    new_export_df = new_rows.copy()
    new_export_df["distance_to_new_product"] = 0.0
    new_export_df["rank"] = 0
    new_export_df["is_nearest_historical"] = 0

    ranked_export_columns = [
        "item_code",
        "item_role",
        "is_new_product",
        "distance_to_new_product",
        "rank",
        "is_nearest_historical",
        *feature_columns,
    ]
    ranking_df = pd.concat(
        [
            new_export_df[ranked_export_columns],
            ranked_historical_df[ranked_export_columns],
        ],
        ignore_index=True,
    )
    ranking_df.to_csv(
        FIRST_EXAMPLE_FIRST_PASS_DATASET_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    display_labels = ranking_df.apply(
        lambda row: f"{row['item_code']} [{str(row['item_role']).upper()}]",
        axis=1,
    )
    distance_matrix = cdist(
        ranking_df[feature_columns].to_numpy(dtype=float),
        ranking_df[feature_columns].to_numpy(dtype=float),
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

    ranked_historical_df[
        ["item_code", "distance_to_new_product", "rank", "is_nearest_historical"]
    ].to_csv(
        FIRST_EXAMPLE_NEAREST_HISTORICAL_RANKING_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {
                "new_product_code": new_code,
                "nearest_historical_product": nearest_historical_product,
                "nearest_distance_to_new_product": nearest_distance,
                "historical_candidate_rows": int(len(historical_rows)),
            }
        ]
    ).to_csv(
        FIRST_EXAMPLE_NEAREST_HISTORICAL_SUMMARY_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    summary_df = pd.DataFrame(
        [
            {
                "new_product_code": new_code,
                "status": status,
                "candidate_rows": int(len(candidate_df)),
                "historical_candidate_rows": int((candidate_df["is_new_product"] == 0).sum()),
                "encoded_rows": int(len(encoded_df)),
                "used_fallback_features": bool(used_fallback_features),
                "nearest_historical_product": nearest_historical_product,
                "nearest_distance_to_new_product": nearest_distance,
            }
        ]
    )
    summary_df.to_csv(
        FIRST_EXAMPLE_DIAGNOSTIC_SUMMARY_PATH,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    return {"status": status}


def classify_new_product(candidate_df: pd.DataFrame, new_code: str) -> dict:
    if candidate_df.empty:
        return {
            "new_product_code": new_code,
            "allocation_type": "random_allocation",
            "corresponding_historical_product": None,
        }

    nearest_hist_code = find_nearest_historical_candidate(candidate_df, new_code)
    allocation_type = "common_allocation" if nearest_hist_code is not None else "random_allocation"
    corresponding_historical_product = nearest_hist_code

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
        print(f"First example nearest-neighbor ranking saved to: {FIRST_EXAMPLE_FIRST_PASS_DATASET_PATH}")
        print(f"First example standardized attributes saved to: {FIRST_EXAMPLE_STANDARDIZED_ATTRIBUTES_PATH}")
        print(f"First example SKU distance matrix saved to: {FIRST_EXAMPLE_SKU_DISTANCE_PATH}")
        print(f"First example nearest-historical ranking saved to: {FIRST_EXAMPLE_NEAREST_HISTORICAL_RANKING_PATH}")
        print(f"First example nearest-historical summary saved to: {FIRST_EXAMPLE_NEAREST_HISTORICAL_SUMMARY_PATH}")
    print(f"Allocation summary saved to: {ALLOCATION_SUMMARY_PATH}")
    print(f"Allocation summary chart saved to: {ALLOCATION_CHART_PATH}")
    print(f"Common-allocation new SKUs: {common_count:,}")
    print(f"Random-allocation new SKUs: {random_count:,}")
    print(f"BC-nearest-neighbor results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
