from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from experiment_context import load_experiment_context, normalize_item_code

OUTPUT_PATH = Path(__file__).resolve().parent / "bc-k-means-results.csv"
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


def prepare_candidate_frame(product_df: pd.DataFrame, new_code: str, historical_skus: set[str]) -> pd.DataFrame:
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
    return combined


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # Keep preprocessed_final.csv on its raw scale and transform only the
    # candidate set being clustered for the current new product.
    encoded = df.copy()
    raw_feature_columns = [
        "Promo",
        "capacity",
        "original_price",
        "estimation_discount",
        "shelf_life",
    ]

    for column in raw_feature_columns:
        encoded[column] = pd.to_numeric(encoded[column], errors="coerce")

    encoded["shelf_life"] = np.ceil(encoded["shelf_life"])
    encoded["log_capacity"] = np.log1p(encoded["capacity"].clip(lower=0))
    encoded["log_original_price"] = np.log1p(encoded["original_price"].clip(lower=0))
    encoded["log_estimation_discount"] = np.log1p(
        encoded["estimation_discount"].clip(lower=0)
    )
    encoded["log_shelf_life"] = np.log1p(encoded["shelf_life"].clip(lower=0))

    feature_input_columns = [
        "Promo",
        "log_capacity",
        "log_original_price",
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
        "log_original_price_normalized",
        "log_estimation_discount_normalized",
        "log_shelf_life_normalized",
    ]
    encoded[feature_columns] = scaler.fit_transform(encoded[feature_input_columns])

    return encoded, feature_columns


def classify_new_product(candidate_df: pd.DataFrame, new_code: str) -> dict:
    if candidate_df.empty:
        return {
            "new_product_code": new_code,
            "allocation_type": "random_allocation",
            "corresponding_historical_product": None,
        }

    nearest_hist_code = None
    current_df = candidate_df.copy()

    while True:
        historical_candidates = current_df[current_df["is_new_product"] == 0].copy()
        if historical_candidates.empty:
            return {
                "new_product_code": new_code,
                "allocation_type": "random_allocation",
                "corresponding_historical_product": None,
            }

        if len(current_df) <= MAX_CLUSTER_SIZE:
            final_cluster = current_df.copy()
            break

        encoded_df, feature_columns = build_feature_matrix(current_df)
        if encoded_df.empty or not feature_columns:
            final_cluster = current_df.copy()
            break

        n_clusters = int(np.ceil(len(encoded_df) / MAX_CLUSTER_SIZE))
        n_unique_points = np.unique(
            encoded_df[feature_columns].to_numpy(dtype=float),
            axis=0,
        ).shape[0]

        if n_unique_points < n_clusters:
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

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        encoded_df["cluster"] = km.fit_predict(encoded_df[feature_columns])
        new_cluster = encoded_df.loc[
            encoded_df["item_code"] == new_code, "cluster"
        ].iloc[0]
        problem_cluster = encoded_df[encoded_df["cluster"] == new_cluster].copy()

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

    return {
        "new_product_code": new_code,
        "allocation_type": allocation_type,
        "corresponding_historical_product": corresponding_historical_product,
    }


def main():
    context = load_experiment_context(BASE_DIR)
    product_df = load_product_metadata(context.paths.product_path)
    product_df = product_df[product_df["item_code"].isin(context.eligible_skus)].copy()

    historical_skus = set(context.historical_skus)
    new_skus = list(context.new_skus)
    if not new_skus:
        result_df = pd.DataFrame(
            columns=[
                "new_product_code",
                "allocation_type",
                "corresponding_historical_product",
            ]
        )
    else:
        results = []
        for new_code in new_skus:
            candidate_df = prepare_candidate_frame(product_df, new_code, historical_skus)
            results.append(classify_new_product(candidate_df, new_code))
        result_df = pd.DataFrame(results)

    result_df.to_csv(OUTPUT_PATH, index=False, sep=";", encoding="utf-8-sig")

    common_count = int((result_df["allocation_type"] == "common_allocation").sum()) if not result_df.empty else 0
    random_count = int((result_df["allocation_type"] == "random_allocation").sum()) if not result_df.empty else 0
    print(f"Cutoff ratio: {context.cutoff_ratio:.2f}")
    print(f"Cutoff timestamp: {context.cutoff_time}")
    print(f"Eligible master SKUs: {len(context.eligible_skus):,}")
    print(f"Historical SKUs: {len(context.historical_skus):,}")
    print(f"New SKUs: {len(context.new_skus):,}")
    print(f"Common-allocation new SKUs: {common_count:,}")
    print(f"Random-allocation new SKUs: {random_count:,}")
    print(f"BC-k-means results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
