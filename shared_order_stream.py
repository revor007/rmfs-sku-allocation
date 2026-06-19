from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
SHARED_BOOTSTRAP_DIR = ROOT_DIR / "_shared_bootstrap_orders"
RAW_ORDER_ID_CANDIDATES = ["订单号", "order_id", "Order ID"]
RAW_ITEM_CODE_CANDIDATES = ["商品编码", "item_code", "Item Code"]
RAW_QUANTITY_CANDIDATES = ["商品数量", "item_quantity", "quantity", "qty", "Item Quantity"]
RAW_CREATED_AT_CANDIDATES = ["创建时间", "order_date", "created_at", "order_time", "Order Date"]


def normalize_item_code(value) -> str:
    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def normalize_column_name(name) -> str:
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


def find_actual_order_data_path(run_root: Path) -> Path:
    prepared_cutoff_orders = run_root / "data" / "input" / "cutoff_test_orders.csv"
    if prepared_cutoff_orders.exists():
        return prepared_cutoff_orders

    preprocessing_dir = ROOT_DIR / "Preprocessing"
    candidates = sorted(
        path
        for path in preprocessing_dir.glob("*_final.csv")
        if path.name != "preprocessed_final.csv"
    )
    if not candidates:
        raise FileNotFoundError(
            "No cutoff_test_orders.csv or *_final.csv order source was found for shared bootstrap generation."
        )
    return candidates[0]


def load_raw_orders(source_path: Path) -> pd.DataFrame:
    raw_orders = pd.read_csv(
        source_path,
        sep=None,
        engine="python",
        encoding="utf-8-sig",
        decimal=",",
    )

    order_col = find_column(raw_orders.columns, RAW_ORDER_ID_CANDIDATES)
    item_code_col = find_column(raw_orders.columns, RAW_ITEM_CODE_CANDIDATES)
    qty_col = find_column(raw_orders.columns, RAW_QUANTITY_CANDIDATES)
    created_col = find_column(raw_orders.columns, RAW_CREATED_AT_CANDIDATES)

    raw_orders = raw_orders[[order_col, item_code_col, qty_col, created_col]].copy()
    raw_orders.columns = ["source_order_id", "item_code", "item_quantity", "created_at"]
    raw_orders["source_order_id"] = raw_orders["source_order_id"].astype(str).str.strip()
    raw_orders["item_code"] = raw_orders["item_code"].map(normalize_item_code)
    raw_orders["item_quantity"] = pd.to_numeric(raw_orders["item_quantity"], errors="coerce")
    raw_orders["created_at"] = pd.to_datetime(
        raw_orders["created_at"],
        errors="coerce",
        dayfirst=True,
    )
    raw_orders = raw_orders.dropna(
        subset=["source_order_id", "item_code", "item_quantity", "created_at"]
    ).copy()
    raw_orders = raw_orders[raw_orders["item_code"] != ""].copy()
    raw_orders["item_quantity"] = np.ceil(raw_orders["item_quantity"]).astype(int)
    raw_orders = raw_orders[raw_orders["item_quantity"] > 0].copy()

    if raw_orders.empty:
        raise ValueError(f"No valid order rows remained after parsing {source_path}.")

    return raw_orders.sort_values(
        ["created_at", "source_order_id", "item_code", "item_quantity"],
        kind="stable",
    ).reset_index(drop=True)


def build_empirical_order_table(raw_orders: pd.DataFrame) -> pd.DataFrame:
    order_table = (
        raw_orders.groupby("source_order_id", sort=False)
        .agg(created_at=("created_at", "min"))
        .reset_index()
        .sort_values(["created_at", "source_order_id"], kind="stable")
        .reset_index(drop=True)
    )
    base_time = order_table["created_at"].min()
    order_table["arrival_seconds"] = (
        order_table["created_at"] - base_time
    ).dt.total_seconds().round().astype(int)
    return order_table


def bootstrap_arrivals(
    order_table: pd.DataFrame,
    sample_size: int,
    rng: np.random.Generator,
    arrival_mode: str,
) -> np.ndarray:
    if sample_size <= 0:
        return np.asarray([], dtype=np.int64)

    empirical_arrivals = order_table["arrival_seconds"].to_numpy(dtype=np.int64)
    if empirical_arrivals.size == 0:
        return np.zeros(sample_size, dtype=np.int64)

    if arrival_mode == "sample_original_times":
        sampled_arrivals = rng.choice(empirical_arrivals, size=sample_size, replace=True)
        return np.sort(sampled_arrivals.astype(np.int64))

    if arrival_mode != "empirical_interarrival":
        raise ValueError(
            "Unsupported bootstrap arrival mode: "
            f"{arrival_mode}. Expected 'empirical_interarrival' or 'sample_original_times'."
        )

    horizon = int(empirical_arrivals.max())
    if empirical_arrivals.size == 1 or horizon <= 0:
        return np.zeros(sample_size, dtype=np.int64)

    empirical_gaps = np.diff(empirical_arrivals)
    if empirical_gaps.size == 0:
        return np.zeros(sample_size, dtype=np.int64)

    sampled_gaps = rng.choice(empirical_gaps, size=max(sample_size - 1, 0), replace=True)
    sampled_arrivals = np.concatenate(
        [np.asarray([0], dtype=np.float64), np.cumsum(sampled_gaps, dtype=np.float64)]
    )
    if sampled_arrivals[-1] > 0:
        sampled_arrivals = (sampled_arrivals / sampled_arrivals[-1]) * horizon

    sampled_arrivals = np.rint(sampled_arrivals).astype(np.int64)
    return np.maximum.accumulate(sampled_arrivals)


def build_generated_orders(
    sampled_order_ids: np.ndarray,
    raw_orders: pd.DataFrame,
    sampled_arrivals: np.ndarray,
) -> pd.DataFrame:
    order_lines = []
    sequence_id = 0

    for generated_order_id, (source_order_id, order_arrival) in enumerate(
        zip(sampled_order_ids, sampled_arrivals)
    ):
        source_lines = raw_orders.loc[
            raw_orders["source_order_id"] == source_order_id,
            ["item_code", "item_quantity"],
        ]
        for line in source_lines.itertuples(index=False):
            order_lines.append(
                {
                    "sequence_id": int(sequence_id),
                    "order_id": int(generated_order_id),
                    "order_type": 1,
                    "item_code": str(line.item_code),
                    "item_quantity": int(line.item_quantity),
                    "order_arrival": int(order_arrival),
                    "source_order_id": str(source_order_id),
                }
            )
            sequence_id += 1

    generated_order = pd.DataFrame(order_lines)
    if generated_order.empty:
        raise ValueError("Bootstrap generation produced no order lines.")
    return generated_order


def ensure_shared_bootstrap_order_file(
    run_root: Path,
    seed: int,
    n_orders: int | None = None,
    arrival_mode: str = "empirical_interarrival",
) -> Path:
    source_path = find_actual_order_data_path(run_root)
    source_name = source_path.stem.replace(" ", "_")
    n_orders_tag = "default" if n_orders is None else str(int(n_orders))

    SHARED_BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SHARED_BOOTSTRAP_DIR / (
        f"{source_name}_bootstrap_seed_{int(seed)}_n_{n_orders_tag}_{arrival_mode}.csv"
    )
    meta_path = output_path.with_suffix(".json")

    if output_path.exists():
        return output_path

    raw_orders = load_raw_orders(source_path)
    order_table = build_empirical_order_table(raw_orders)
    source_order_ids = order_table["source_order_id"].to_numpy(dtype=object)
    source_unique_orders = int(source_order_ids.size)
    if source_unique_orders <= 0:
        raise ValueError("Bootstrap source contains no valid unique orders.")

    resolved_n_orders = source_unique_orders if n_orders is None else int(n_orders)
    if resolved_n_orders <= 0:
        raise ValueError("Bootstrap n_orders must be positive.")

    rng = np.random.default_rng(int(seed))
    sampled_order_ids = rng.choice(
        source_order_ids,
        size=resolved_n_orders,
        replace=True,
    )
    sampled_arrivals = bootstrap_arrivals(
        order_table=order_table,
        sample_size=resolved_n_orders,
        rng=rng,
        arrival_mode=arrival_mode,
    )
    generated_order = build_generated_orders(
        sampled_order_ids=sampled_order_ids,
        raw_orders=raw_orders,
        sampled_arrivals=sampled_arrivals,
    )
    generated_order.to_csv(output_path, index=False)

    metadata = {
        "generator": "full_postt_shared_bootstrap",
        "source_path": str(source_path),
        "seed": int(seed),
        "n_orders": int(resolved_n_orders),
        "arrival_mode": arrival_mode,
        "source_unique_orders": source_unique_orders,
        "sampled_unique_source_orders": int(pd.Series(sampled_order_ids).nunique()),
        "generated_unique_orders": int(generated_order["order_id"].nunique()),
        "generated_order_lines": int(len(generated_order)),
    }
    with meta_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    return output_path
