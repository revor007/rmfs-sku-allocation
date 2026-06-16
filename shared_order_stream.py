from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
BOOTSTRAP_CACHE_DIR = SCRIPT_DIR / "bootstrap_order_cache"

ACTUAL_ORDER_ID_CANDIDATES = ["订单号", "order_id", "Order ID"]
ACTUAL_SKU_CANDIDATES = ["商品编码", "item_code", "Item Code"]
ACTUAL_QUANTITY_CANDIDATES = ["商品数量", "item_quantity", "quantity", "qty", "Item Quantity"]
ACTUAL_CREATED_TIME_CANDIDATES = ["创建时间", "order_date", "created_at", "order_time", "Order Date"]


def _normalize_item_code(value) -> str:
    text = str(value).replace("\ufeff", "").strip()
    if not text or text.lower() in {"nan", "none"}:
        return ""
    return text[:-2] if text.endswith(".0") else text


def _normalize_column_name(name) -> str:
    return str(name).replace("\ufeff", "").strip()


def _find_column(columns, candidates):
    normalized = {_normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    raise KeyError(
        f"Could not find any of the expected columns {candidates}. "
        f"Available columns: {list(columns)}"
    )


def _read_csv_auto(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    frame.columns = [_normalize_column_name(col) for col in frame.columns]
    return frame


def _stable_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _mapping_signature(items_path: Path) -> tuple[pd.DataFrame, str]:
    items = _read_csv_auto(items_path)
    if "item_id" not in items.columns or "item_code" not in items.columns:
        raise ValueError(f"{items_path} must contain item_id and item_code columns.")

    mapping = items[["item_id", "item_code"]].copy()
    mapping["item_id"] = pd.to_numeric(mapping["item_id"], errors="coerce")
    mapping = mapping.dropna(subset=["item_id"]).copy()
    mapping["item_id"] = mapping["item_id"].astype(int)
    mapping["item_code"] = mapping["item_code"].map(_normalize_item_code)
    mapping = mapping[mapping["item_code"] != ""].copy()
    mapping = mapping.drop_duplicates(subset=["item_id", "item_code"]).sort_values(
        ["item_id", "item_code"],
        kind="stable",
    )

    signature = hashlib.sha256()
    for row in mapping.itertuples(index=False):
        signature.update(f"{row.item_id},{row.item_code}\n".encode("utf-8"))
    return mapping.reset_index(drop=True), signature.hexdigest()[:12]


def find_actual_order_data_path(run_root: Path) -> Path:
    direct_path = run_root / "data" / "input" / "cutoff_test_orders.csv"
    if direct_path.exists():
        return direct_path

    preprocessing_dir = SCRIPT_DIR / "Preprocessing"
    if preprocessing_dir.exists():
        candidates = sorted(
            path
            for path in preprocessing_dir.glob("*_final.csv")
            if path.name != "preprocessed_final.csv"
        )
        if candidates:
            return candidates[0]

    raise FileNotFoundError(
        f"Could not find cutoff_test_orders.csv under {run_root} or a *_final.csv order source in {preprocessing_dir}."
    )


def _load_raw_orders(source_path: Path, items_path: Path) -> pd.DataFrame:
    raw_orders = _read_csv_auto(source_path)

    order_col = _find_column(raw_orders.columns, ACTUAL_ORDER_ID_CANDIDATES)
    sku_col = _find_column(raw_orders.columns, ACTUAL_SKU_CANDIDATES)
    quantity_col = _find_column(raw_orders.columns, ACTUAL_QUANTITY_CANDIDATES)
    created_col = _find_column(raw_orders.columns, ACTUAL_CREATED_TIME_CANDIDATES)

    raw_orders = raw_orders[[order_col, sku_col, quantity_col, created_col]].copy()
    raw_orders.columns = ["source_order_id", "item_code", "item_quantity", "created_at"]
    raw_orders["source_order_id"] = raw_orders["source_order_id"].astype(str).str.strip()
    raw_orders["item_code"] = raw_orders["item_code"].map(_normalize_item_code)
    quantity_series = raw_orders["item_quantity"].astype(str).str.replace(",", ".", regex=False)
    raw_orders["item_quantity"] = pd.to_numeric(quantity_series, errors="coerce")

    created_series = pd.to_datetime(
        raw_orders["created_at"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
        dayfirst=True,
    )
    created_fallback = pd.to_datetime(
        raw_orders["created_at"],
        errors="coerce",
    )
    raw_orders["created_at"] = created_series.fillna(created_fallback)

    raw_orders = raw_orders.dropna(
        subset=["source_order_id", "item_code", "item_quantity", "created_at"]
    ).copy()
    raw_orders = raw_orders[raw_orders["item_code"] != ""].copy()
    raw_orders["item_quantity"] = np.ceil(raw_orders["item_quantity"]).astype(int)
    raw_orders = raw_orders[raw_orders["item_quantity"] > 0].copy()

    item_lookup, _ = _mapping_signature(items_path)
    raw_orders = raw_orders.merge(item_lookup, on="item_code", how="left")

    missing_mask = raw_orders["item_id"].isna()
    if missing_mask.any():
        missing_codes = sorted(
            raw_orders.loc[missing_mask, "item_code"].astype(str).unique().tolist()
        )
        dropped_lines = int(missing_mask.sum())
        raw_orders = raw_orders.loc[~missing_mask].copy()
        if raw_orders.empty:
            raise ValueError(
                "No order lines remain after filtering to SKUs present in items.csv."
            )
        print(
            f"[BOOTSTRAP] Skipping {dropped_lines:,} order lines across {len(missing_codes):,} "
            f"SKUs missing from items.csv. Sample missing codes: {', '.join(missing_codes[:10])}"
        )

    raw_orders["item_id"] = raw_orders["item_id"].astype(int)
    raw_orders = raw_orders.sort_values(
        ["created_at", "source_order_id", "item_id"],
        kind="stable",
    ).reset_index(drop=True)
    return raw_orders


def _build_empirical_order_table(raw_orders: pd.DataFrame) -> pd.DataFrame:
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


def _bootstrap_arrivals(
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
        sampled_arrivals = np.sort(sampled_arrivals.astype(np.int64))
        return sampled_arrivals

    if arrival_mode != "empirical_interarrival":
        raise ValueError(
            f"Unsupported bootstrap arrival mode '{arrival_mode}'. "
            "Expected 'empirical_interarrival' or 'sample_original_times'."
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
    sampled_arrivals = np.maximum.accumulate(sampled_arrivals)
    return sampled_arrivals


def _build_generated_orders(
    raw_orders: pd.DataFrame,
    sampled_order_ids: np.ndarray,
    sampled_arrivals: np.ndarray,
) -> pd.DataFrame:
    order_lines = []
    sequence_id = 0

    for generated_order_id, (source_order_id, order_arrival) in enumerate(
        zip(sampled_order_ids, sampled_arrivals)
    ):
        source_lines = raw_orders.loc[
            raw_orders["source_order_id"] == source_order_id,
            ["item_id", "item_quantity"],
        ]
        for line in source_lines.itertuples(index=False):
            order_lines.append(
                {
                    "sequence_id": int(sequence_id),
                    "order_id": int(generated_order_id),
                    "order_type": 1,
                    "item_id": int(line.item_id),
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
    run_root = run_root.resolve()
    source_path = find_actual_order_data_path(run_root)
    items_path = run_root / "data" / "output" / "items.csv"
    if not items_path.exists():
        raise FileNotFoundError(f"items.csv not found under {items_path}")

    _, mapping_hash = _mapping_signature(items_path)
    source_hash = _stable_file_hash(source_path)
    order_count_token = "default" if n_orders is None else str(int(n_orders))

    cache_dir = BOOTSTRAP_CACHE_DIR / f"{source_hash}_{mapping_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = f"orders_seed_{seed}_n_{order_count_token}_{arrival_mode}.csv"
    cache_path = cache_dir / cache_name
    metadata_path = cache_dir / f"{cache_path.stem}.json"
    if cache_path.exists():
        return cache_path

    raw_orders = _load_raw_orders(source_path, items_path)
    order_table = _build_empirical_order_table(raw_orders)
    source_order_ids = order_table["source_order_id"].to_numpy(dtype=object)
    source_unique_orders = int(source_order_ids.size)
    resolved_n_orders = source_unique_orders if n_orders is None else int(n_orders)
    if resolved_n_orders <= 0:
        raise ValueError("Bootstrap n_orders must be positive.")

    rng = np.random.default_rng(int(seed))
    sampled_order_ids = rng.choice(
        source_order_ids,
        size=int(resolved_n_orders),
        replace=True,
    )
    sampled_arrivals = _bootstrap_arrivals(
        order_table=order_table,
        sample_size=int(resolved_n_orders),
        rng=rng,
        arrival_mode=arrival_mode,
    )
    generated_order = _build_generated_orders(
        raw_orders=raw_orders,
        sampled_order_ids=sampled_order_ids,
        sampled_arrivals=sampled_arrivals,
    )
    generated_order.to_csv(cache_path, index=False)

    metadata = {
        "generator": "shared_bootstrap_actual",
        "seed": int(seed),
        "n_orders": int(resolved_n_orders),
        "arrival_mode": arrival_mode,
        "source_path": str(source_path),
        "items_path": str(items_path),
        "source_hash": source_hash,
        "mapping_hash": mapping_hash,
        "source_unique_orders": source_unique_orders,
        "sampled_unique_source_orders": int(pd.Series(sampled_order_ids).nunique()),
        "generated_unique_orders": int(generated_order["order_id"].nunique()),
        "generated_order_lines": int(len(generated_order)),
        "generated_max_arrival": int(generated_order["order_arrival"].max()),
        "output_path": str(cache_path),
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    return cache_path
