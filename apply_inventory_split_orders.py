from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiment_context import (
    ORDER_ID_CANDIDATES,
    ORDER_QTY_CANDIDATES,
    ORDER_SKU_CANDIDATES,
    ORDER_TIME_CANDIDATES,
    compute_cutoff_time,
    find_column,
    find_order_data_path,
    find_preprocessing_dir,
    load_semicolon_csv,
    normalize_item_code,
    resolve_cutoff_ratio,
)


BASE_DIR = Path(__file__).resolve().parent
PREPROCESSING_DIR = find_preprocessing_dir(BASE_DIR)
ORDER_FINAL_PATH = find_order_data_path(PREPROCESSING_DIR)
SPLIT_ORDER_OUTPUT_PATH = PREPROCESSING_DIR / "inventory_threshold_split_orders.csv"
SUMMARY_OUTPUT_PATH = PREPROCESSING_DIR / "inventory_threshold_split_summary.csv"
DATA_CLEANING_SPLIT_SUMMARY_PATH = PREPROCESSING_DIR / "data_cleaning_split_summary.csv"
PRODUCT_METADATA_CANDIDATES = [
    PREPROCESSING_DIR / "preprocessed_final_latest.csv",
    PREPROCESSING_DIR / "preprocessed_final.csv",
]


def derive_split_export_path(order_final_path: Path, suffix: str) -> Path:
    if order_final_path.name.endswith("_final.csv"):
        return order_final_path.with_name(
            order_final_path.name.replace("_final.csv", f"_{suffix}.csv")
        )
    return order_final_path.with_name(f"{order_final_path.stem}_{suffix}{order_final_path.suffix}")


def get_writable_output_path(path: Path) -> Path:
    if not path.exists():
        return path

    try:
        with open(path, "a", encoding="utf-8"):
            return path
    except PermissionError:
        return path.with_name(f"{path.stem}_latest{path.suffix}")


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    output_path = get_writable_output_path(path)
    df.to_csv(
        output_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    return output_path


def build_metric_records(records: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(records, columns=["metric", "value"])


def safe_console_text(value: object) -> str:
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


def find_product_metadata_path() -> Path:
    for path in PRODUCT_METADATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not locate preprocessed_final.csv. Searched: "
        + ", ".join(str(path) for path in PRODUCT_METADATA_CANDIDATES)
    )


def main() -> None:
    cutoff_ratio = resolve_cutoff_ratio()

    order_df = load_semicolon_csv(ORDER_FINAL_PATH).copy()
    order_id_col = find_column(order_df.columns, ORDER_ID_CANDIDATES)
    sku_col = find_column(order_df.columns, ORDER_SKU_CANDIDATES)
    qty_col = find_column(order_df.columns, ORDER_QTY_CANDIDATES)
    time_col = find_column(order_df.columns, ORDER_TIME_CANDIDATES)

    order_df["_order_id"] = order_df[order_id_col].astype(str).str.strip()
    order_df["_item_code"] = order_df[sku_col].map(normalize_item_code)
    order_df["_quantity"] = pd.to_numeric(order_df[qty_col], errors="coerce").fillna(0.0)
    order_df["_created_at"] = pd.to_datetime(
        order_df[time_col],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    product_metadata_path = find_product_metadata_path()
    product_metadata = load_semicolon_csv(product_metadata_path).copy()
    code_col = find_column(product_metadata.columns, ["item_code"])
    ceiling_col = find_column(
        product_metadata.columns,
        ["inventory_quantity", "庫存量", "庫存量 (Inventory Quantity)"],
    )
    product_metadata = product_metadata[[code_col, ceiling_col]].copy()
    product_metadata.columns = ["item_code", "inventory_quantity_ceiling"]
    product_metadata["item_code"] = product_metadata["item_code"].map(normalize_item_code)
    product_metadata["inventory_quantity_ceiling"] = pd.to_numeric(
        product_metadata["inventory_quantity_ceiling"], errors="coerce"
    ).fillna(0).astype(int)

    threshold_by_sku = dict(
        zip(
            product_metadata["item_code"],
            product_metadata["inventory_quantity_ceiling"],
        )
    )

    order_df["_inventory_quantity_ceiling"] = (
        order_df["_item_code"].map(threshold_by_sku).fillna(-1).astype(int)
    )
    order_df["_exceeds_inventory_threshold"] = (
        (order_df["_inventory_quantity_ceiling"] >= 0)
        & (order_df["_quantity"] > order_df["_inventory_quantity_ceiling"])
    )

    flagged_lines = order_df[order_df["_exceeds_inventory_threshold"]].copy()
    flagged_order_ids = set(flagged_lines["_order_id"])

    removed_orders_df = order_df[order_df["_order_id"].isin(flagged_order_ids)].copy()
    filtered_order_df = order_df[~order_df["_order_id"].isin(flagged_order_ids)].copy()

    helper_columns = [
        "_order_id",
        "_item_code",
        "_quantity",
        "_created_at",
        "_inventory_quantity_ceiling",
        "_exceeds_inventory_threshold",
    ]

    cutoff_time = compute_cutoff_time(filtered_order_df[["_created_at"]].rename(columns={"_created_at": "created_at"}), cutoff_ratio)
    train_df = filtered_order_df[filtered_order_df["_created_at"] <= cutoff_time].copy()
    test_df = filtered_order_df[filtered_order_df["_created_at"] > cutoff_time].copy()

    order_train_path = derive_split_export_path(ORDER_FINAL_PATH, "train")
    order_test_path = derive_split_export_path(ORDER_FINAL_PATH, "test")

    final_output_path = save_dataframe(
        filtered_order_df.drop(columns=helper_columns),
        ORDER_FINAL_PATH,
    )
    train_output_path = save_dataframe(
        train_df.drop(columns=helper_columns),
        order_train_path,
    )
    test_output_path = save_dataframe(
        test_df.drop(columns=helper_columns),
        order_test_path,
    )

    removed_output_path = save_dataframe(removed_orders_df, SPLIT_ORDER_OUTPUT_PATH)

    overlap_item_codes = (
        set(train_df["_item_code"].unique()) & set(test_df["_item_code"].unique())
    )
    summary_records = [
        ("cutoff_ratio", cutoff_ratio),
        ("cutoff_timestamp", str(cutoff_time)),
        ("source_order_rows", int(len(order_df))),
        ("source_unique_orders", int(order_df["_order_id"].nunique())),
        ("source_unique_item_codes", int(order_df["_item_code"].nunique())),
        ("threshold_skus_available", int(len(threshold_by_sku))),
        ("inventory_threshold_blocked_lines", int(len(flagged_lines))),
        ("inventory_threshold_blocked_unique_orders", int(len(flagged_order_ids))),
        ("inventory_threshold_blocked_unique_item_codes", int(flagged_lines["_item_code"].nunique())),
        ("remaining_order_rows", int(len(filtered_order_df))),
        ("remaining_unique_orders", int(filtered_order_df["_order_id"].nunique())),
        ("remaining_unique_item_codes", int(filtered_order_df["_item_code"].nunique())),
        ("train_order_rows", int(len(train_df))),
        ("test_order_rows", int(len(test_df))),
        ("train_unique_orders", int(train_df["_order_id"].nunique())),
        ("test_unique_orders", int(test_df["_order_id"].nunique())),
        ("train_unique_item_codes", int(train_df["_item_code"].nunique())),
        ("test_unique_item_codes", int(test_df["_item_code"].nunique())),
        ("product_overlap_item_codes", int(len(overlap_item_codes))),
    ]
    summary_df = build_metric_records(summary_records)
    summary_output_path = save_dataframe(summary_df, SUMMARY_OUTPUT_PATH)
    data_cleaning_summary_path = save_dataframe(summary_df, DATA_CLEANING_SPLIT_SUMMARY_PATH)

    print(f"Order file updated: {safe_console_text(final_output_path)}")
    print(f"Train export updated: {safe_console_text(train_output_path)}")
    print(f"Test export updated: {safe_console_text(test_output_path)}")
    print(f"Split orders saved to: {safe_console_text(removed_output_path)}")
    print(f"Summary saved to: {safe_console_text(summary_output_path)}")
    print(
        "Data cleaning split summary synced to: "
        f"{safe_console_text(data_cleaning_summary_path)}"
    )
    print(
        "Orders rerouted to split-order handling because at least one line exceeded "
        f"the SKU inventory quantity ceiling: {len(flagged_order_ids):,}"
    )


if __name__ == "__main__":
    main()
