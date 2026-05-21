from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CUTOFF_RATIO = 0.70

ORDER_ID_CANDIDATES = ["订单号", "order_id"]
ORDER_SKU_CANDIDATES = ["商品编码", "item_code"]
ORDER_NAME_CANDIDATES = ["商品名称", "item_name", "title"]
ORDER_QTY_CANDIDATES = ["商品数量", "quantity", "qty", "item_quantity"]
ORDER_TIME_CANDIDATES = ["创建时间", "created_at", "order_date"]

PRODUCT_SKU_CANDIDATES = ["item_code", "商品编码"]
TRANSLATED_SKU_CANDIDATES = ["item code", "item_code"]


@dataclass(frozen=True)
class ExperimentPaths:
    base_dir: Path
    preprocessing_dir: Path
    order_path: Path
    product_path: Path
    translated_info_path: Path
    max_capacity_path: Path


@dataclass
class ExperimentContext:
    paths: ExperimentPaths
    cutoff_ratio: float
    cutoff_time: pd.Timestamp
    order_df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_eligible_df: pd.DataFrame
    test_eligible_df: pd.DataFrame
    eligible_skus: list[str]
    historical_skus: list[str]
    new_skus: list[str]
    removed_post_t_zero_history_skus: list[str]
    sku_status_df: pd.DataFrame


def normalize_column_name(name) -> str:
    return str(name).replace("\ufeff", "").strip()


def normalize_item_code(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""

    text = text.replace(",", ".")
    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if np.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return text


def find_column(columns, candidates: list[str]):
    normalized = {normalize_column_name(col).lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]

    for candidate in candidates:
        key = candidate.lower()
        for normalized_name, original_name in normalized.items():
            if key in normalized_name:
                return original_name

    raise KeyError(
        f"Could not find any of {candidates}. Available columns: {list(columns)}"
    )


def _existing_path(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate {description}. Searched: {searched}")


def find_preprocessing_dir(base_dir: Path) -> Path:
    base_dir = Path(base_dir).resolve()
    return _existing_path(
        [
            base_dir / "Preprocessing",
            base_dir.parent / "Preprocessing",
            base_dir.parent.parent / "Preprocessing",
        ],
        "the 'Preprocessing' directory",
    )


def find_order_data_path(preprocessing_dir: Path) -> Path:
    candidates = sorted(
        path
        for path in Path(preprocessing_dir).glob("*_final.csv")
        if path.name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            f"No order data file ending with '_final.csv' was found in {preprocessing_dir}."
        )
    return candidates[0]


def find_product_data_path(preprocessing_dir: Path) -> Path:
    return _existing_path(
        [Path(preprocessing_dir) / "preprocessed_final.csv"],
        "preprocessed_final.csv",
    )


def find_translated_info_path(preprocessing_dir: Path) -> Path:
    return _existing_path(
        [
            Path(preprocessing_dir) / "儲格設計_原檔(商品資訊)(Translated).csv",
            Path(preprocessing_dir) / "储格设计_原档(商品资讯)(Translated).csv",
        ],
        "the translated RMFS item master CSV",
    )


def find_max_capacity_path(base_dir: Path) -> Path:
    base_dir = Path(base_dir).resolve()
    return _existing_path(
        [
            base_dir / "max_comp_number.csv",
            base_dir.parent / "max_comp_number.csv",
        ],
        "max_comp_number.csv",
    )


def load_semicolon_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        decimal=",",
        engine="python",
    )


def load_order_data(order_path: Path) -> pd.DataFrame:
    df = load_semicolon_csv(order_path).copy()
    order_col = find_column(df.columns, ORDER_ID_CANDIDATES)
    sku_col = find_column(df.columns, ORDER_SKU_CANDIDATES)
    qty_col = find_column(df.columns, ORDER_QTY_CANDIDATES)
    time_col = find_column(df.columns, ORDER_TIME_CANDIDATES)
    name_col = None
    try:
        name_col = find_column(df.columns, ORDER_NAME_CANDIDATES)
    except KeyError:
        pass

    keep_cols = [order_col, sku_col, qty_col, time_col]
    if name_col is not None:
        keep_cols.append(name_col)
    df = df[keep_cols].copy()

    rename_map = {
        order_col: "order_id",
        sku_col: "item_code",
        qty_col: "quantity",
        time_col: "created_at",
    }
    if name_col is not None:
        rename_map[name_col] = "item_name"
    df = df.rename(columns=rename_map)

    df["item_code"] = df["item_code"].map(normalize_item_code)
    df["order_id"] = df["order_id"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )

    if "item_name" in df.columns:
        df["item_name"] = df["item_name"].astype(str).str.strip()
    else:
        df["item_name"] = ""

    df = df.dropna(subset=["item_code", "quantity", "created_at"]).copy()
    df = df[(df["item_code"] != "") & (df["order_id"] != "")]
    df["quantity"] = df["quantity"].astype(float)
    df["order_date"] = df["created_at"].dt.normalize()
    df = df.sort_values(["created_at", "order_id", "item_code"]).reset_index(drop=True)
    return df


def load_sku_set_from_file(path: Path, candidates: list[str], sep: str | None = ";") -> set[str]:
    if sep is None:
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        df = load_semicolon_csv(path)

    sku_col = find_column(df.columns, candidates)
    return {
        sku
        for sku in df[sku_col].map(normalize_item_code)
        if sku
    }


def resolve_cutoff_ratio(cutoff_ratio: float | None = None) -> float:
    if cutoff_ratio is None:
        env_value = os.getenv("FCGMA_CUTOFF_RATIO", "").strip()
        cutoff_ratio = float(env_value) if env_value else DEFAULT_CUTOFF_RATIO

    cutoff_ratio = float(cutoff_ratio)
    if not 0.0 < cutoff_ratio < 1.0:
        raise ValueError("cutoff_ratio must be between 0 and 1.")
    return cutoff_ratio


def compute_cutoff_time(order_df: pd.DataFrame, cutoff_ratio: float) -> pd.Timestamp:
    min_time = order_df["created_at"].min()
    max_time = order_df["created_at"].max()
    if pd.isna(min_time) or pd.isna(max_time):
        raise ValueError("Order data does not contain valid timestamps.")
    if min_time == max_time:
        return min_time
    return min_time + (max_time - min_time) * cutoff_ratio


def expand_pairwise_matrix(matrix_df: pd.DataFrame, sku_codes: list[str]) -> pd.DataFrame:
    expanded = pd.DataFrame(0.0, index=sku_codes, columns=sku_codes, dtype=np.float32)
    overlap = [sku for sku in sku_codes if sku in matrix_df.index and sku in matrix_df.columns]
    if overlap:
        expanded.loc[overlap, overlap] = matrix_df.loc[overlap, overlap].to_numpy(dtype=np.float32)
    values = expanded.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    expanded.iloc[:, :] = values
    return expanded


def load_experiment_paths(base_dir: Path) -> ExperimentPaths:
    base_dir = Path(base_dir).resolve()
    preprocessing_dir = find_preprocessing_dir(base_dir)
    return ExperimentPaths(
        base_dir=base_dir,
        preprocessing_dir=preprocessing_dir,
        order_path=find_order_data_path(preprocessing_dir),
        product_path=find_product_data_path(preprocessing_dir),
        translated_info_path=find_translated_info_path(preprocessing_dir),
        max_capacity_path=find_max_capacity_path(base_dir),
    )


def load_experiment_context(base_dir: Path, cutoff_ratio: float | None = None) -> ExperimentContext:
    cutoff_ratio = resolve_cutoff_ratio(cutoff_ratio)
    paths = load_experiment_paths(base_dir)
    order_df = load_order_data(paths.order_path)

    cutoff_time = compute_cutoff_time(order_df, cutoff_ratio)
    train_df = order_df[order_df["created_at"] <= cutoff_time].copy()
    test_df = order_df[order_df["created_at"] > cutoff_time].copy()

    metadata_skus = load_sku_set_from_file(
        paths.product_path,
        PRODUCT_SKU_CANDIDATES,
        sep=";",
    )
    translated_skus = load_sku_set_from_file(
        paths.translated_info_path,
        TRANSLATED_SKU_CANDIDATES,
        sep=";",
    )
    max_capacity_skus = load_sku_set_from_file(
        paths.max_capacity_path,
        ["item_code"],
        sep=None,
    )

    aligned_support_skus = metadata_skus & translated_skus & max_capacity_skus
    train_aligned_df = train_df[train_df["item_code"].isin(aligned_support_skus)].copy()
    train_order_counts = (
        train_aligned_df.groupby("item_code")["order_id"].nunique().astype(np.int32)
    )

    eligible_skus = sorted(train_order_counts.index.tolist())
    historical_skus = sorted(train_order_counts[train_order_counts > 1].index.tolist())
    new_skus = sorted(train_order_counts[train_order_counts <= 1].index.tolist())

    train_eligible_df = train_aligned_df[train_aligned_df["item_code"].isin(eligible_skus)].copy()
    test_eligible_df = test_df[test_df["item_code"].isin(eligible_skus)].copy()

    removed_post_t_zero_history_skus = sorted(
        (set(test_df["item_code"].unique()) & aligned_support_skus) - set(eligible_skus)
    )

    sku_status_df = pd.DataFrame({"item_code": eligible_skus})
    sku_status_df["pre_t_order_count"] = (
        sku_status_df["item_code"].map(train_order_counts).fillna(0).astype(np.int32)
    )
    sku_status_df["sku_status"] = np.where(
        sku_status_df["pre_t_order_count"] > 1,
        "historical",
        "new",
    )

    return ExperimentContext(
        paths=paths,
        cutoff_ratio=cutoff_ratio,
        cutoff_time=cutoff_time,
        order_df=order_df,
        train_df=train_df,
        test_df=test_df,
        train_eligible_df=train_eligible_df,
        test_eligible_df=test_eligible_df,
        eligible_skus=eligible_skus,
        historical_skus=historical_skus,
        new_skus=new_skus,
        removed_post_t_zero_history_skus=removed_post_t_zero_history_skus,
        sku_status_df=sku_status_df,
    )


def save_cutoff_artifacts(context: ExperimentContext, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eligible_path = output_dir / "eligible_master_skus.csv"
    status_path = output_dir / "sku_status_by_cutoff.csv"

    pd.DataFrame({"item_code": context.eligible_skus}).to_csv(
        eligible_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    context.sku_status_df.to_csv(
        status_path,
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )
    return eligible_path, status_path
