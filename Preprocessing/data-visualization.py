from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

DEMAND_SUMMARY_OUTPUT_PATH = BASE_DIR / "demand_summary_statistics.csv"
DEMAND_SERIES_DETAIL_OUTPUT_PATH = BASE_DIR / "demand_series_detail.csv"

ORDER_ID_CANDIDATES = [
    "\u8ba2\u5355\u53f7",
    "order_id",
]
SKU_CANDIDATES = [
    "\u5546\u54c1\u7f16\u7801",
    "item_code",
]
PRODUCT_NAME_CANDIDATES = [
    "\u5546\u54c1\u540d\u79f0",
    "item_name",
]
QUANTITY_CANDIDATES = [
    "\u5546\u54c1\u6570\u91cf",
    "quantity",
]
CREATED_TIME_CANDIDATES = [
    "\u521b\u5efa\u65f6\u95f4",
    "created_at",
]


def find_preprocessing_dir():
    candidates = [
        BASE_DIR / "Preprocessing",
        BASE_DIR.parent / "Preprocessing",
        BASE_DIR.parent.parent / "Preprocessing",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not locate the 'Preprocessing' directory.")


def find_order_data_path(preprocessing_dir):
    candidates = sorted(
        path
        for path in preprocessing_dir.glob("*_final.csv")
        if path.name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No order data file ending with '_final.csv' was found in {preprocessing_dir}."
        )

    return candidates[0]


def normalize_column_name(name):
    return str(name).replace("\ufeff", "").strip()


def find_column(columns, candidates):
    normalized = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def get_writable_output_path(path):
    if not path.exists():
        return path

    try:
        with open(path, "a", encoding="utf-8"):
            return path
    except PermissionError:
        return path.with_name(f"{path.stem}_latest{path.suffix}")


def save_dataframe(df, path, **kwargs):
    output_path = get_writable_output_path(path)
    df.to_csv(output_path, **kwargs)
    return output_path


def load_order_data(path):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", decimal=",").copy()

    columns = {
        "order_id": find_column(df.columns, ORDER_ID_CANDIDATES),
        "sku": find_column(df.columns, SKU_CANDIDATES),
        "product_name": find_column(df.columns, PRODUCT_NAME_CANDIDATES),
        "quantity": find_column(df.columns, QUANTITY_CANDIDATES),
        "created_at": find_column(df.columns, CREATED_TIME_CANDIDATES),
    }

    df[columns["order_id"]] = df[columns["order_id"]].astype(str).str.strip()
    df[columns["sku"]] = df[columns["sku"]].astype(str).str.strip()
    df[columns["product_name"]] = df[columns["product_name"]].astype(str).str.strip()
    df["_created_timestamp"] = pd.to_datetime(
        df[columns["created_at"]],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    df["_created_date"] = df["_created_timestamp"].dt.normalize()
    df = df.dropna(subset=["_created_date"]).copy()

    return df, columns


def build_daily_demand_matrix(df, columns):
    sku_col = columns["sku"]
    product_col = columns["product_name"]
    quantity_col = columns["quantity"]

    daily_demand = (
        df.groupby([sku_col, "_created_date"], as_index=False)[quantity_col]
        .sum()
        .rename(columns={quantity_col: "daily_demand"})
    )

    all_dates = pd.date_range(
        daily_demand["_created_date"].min(),
        daily_demand["_created_date"].max(),
        freq="D",
    )
    sku_index = (
        df[[sku_col]]
        .drop_duplicates()
        .sort_values(sku_col)
        .set_index(sku_col)
        .index
    )

    demand_matrix = (
        daily_demand.pivot(index=sku_col, columns="_created_date", values="daily_demand")
        .reindex(index=sku_index, columns=all_dates, fill_value=0)
        .fillna(0)
        .sort_index()
    )

    product_lookup = (
        df[[sku_col, product_col]]
        .drop_duplicates()
        .sort_values([sku_col, product_col])
        .drop_duplicates(subset=[sku_col], keep="first")
        .set_index(sku_col)[product_col]
    )

    return demand_matrix, product_lookup


def build_series_detail(demand_matrix, product_lookup):
    nonzero_counts = demand_matrix.gt(0).sum(axis=1)
    zero_pct = demand_matrix.eq(0).mean(axis=1) * 100
    nonzero_only = demand_matrix.where(demand_matrix > 0)
    avg_nonzero_demand = nonzero_only.mean(axis=1)
    std_nonzero_demand = nonzero_only.std(axis=1)
    cv_nonzero_demand = (
        std_nonzero_demand / avg_nonzero_demand.replace(0, pd.NA) * 100
    ).fillna(0)

    detail = pd.DataFrame(
        {
            "sku": demand_matrix.index,
            "product_name": product_lookup.reindex(demand_matrix.index).fillna(""),
            "observations": demand_matrix.shape[1],
            "nonzero_observations": nonzero_counts,
            "zero_value_pct": zero_pct.round(4),
            "avg_nonzero_demand": avg_nonzero_demand.round(4),
            "std_nonzero_demand": std_nonzero_demand.fillna(0).round(4),
            "cv_nonzero_demand_pct": cv_nonzero_demand.round(4),
        }
    )
    return detail


def summarize_series(series, section_name):
    clean = pd.Series(series, dtype="float64").dropna()
    return pd.DataFrame(
        [
            (section_name, "Mean", clean.mean()),
            (section_name, "S.D.", clean.std()),
            (section_name, "Maximum", clean.max()),
            (section_name, "75%ile", clean.quantile(0.75)),
            (section_name, "50%ile", clean.quantile(0.50)),
            (section_name, "25%ile", clean.quantile(0.25)),
            (section_name, "Minimum", clean.min()),
        ],
        columns=["section", "statistic", "value"],
    )


def build_demand_summary(detail, demand_matrix, source_path):
    summary_parts = [
        pd.DataFrame(
            [
                ("Dataset", "Source file", source_path.name),
                ("Data features", "No. series", demand_matrix.shape[0]),
                ("Data features", "No. obs./series", demand_matrix.shape[1]),
            ],
            columns=["section", "statistic", "value"],
        ),
        summarize_series(detail["zero_value_pct"], "% Zero values"),
        summarize_series(detail["avg_nonzero_demand"], "Average of nonzero demands"),
        summarize_series(detail["cv_nonzero_demand_pct"], "CV of nonzero demands"),
    ]

    summary = pd.concat(summary_parts, ignore_index=True)
    numeric_mask = pd.to_numeric(summary["value"], errors="coerce").notna()
    summary.loc[numeric_mask, "value"] = (
        pd.to_numeric(summary.loc[numeric_mask, "value"]).round(4)
    )
    return summary


def main():
    preprocessing_dir = find_preprocessing_dir()
    source_path = find_order_data_path(preprocessing_dir)
    df_order, columns = load_order_data(source_path)
    demand_matrix, product_lookup = build_daily_demand_matrix(df_order, columns)
    detail = build_series_detail(demand_matrix, product_lookup)
    summary = build_demand_summary(detail, demand_matrix, source_path)

    summary_path = save_dataframe(
        summary,
        DEMAND_SUMMARY_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )
    detail_path = save_dataframe(
        detail,
        DEMAND_SERIES_DETAIL_OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    print("Demand summary assumption: daily demand is the summed quantity per SKU per day.")
    print(f"Source file: {source_path}")
    print(f"Demand summary saved to: {summary_path}")
    print(f"Demand series detail saved to: {detail_path}")
    print("\nDemand summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
