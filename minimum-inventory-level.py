from pathlib import Path

import numpy as np
import pandas as pd

from experiment_context import load_experiment_context


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "minimum_inventory.csv"
DETAIL_PATH = BASE_DIR / "minimum_inventory_evaluation.csv"

LEAD_TIME_DAYS = 1
TARGET_SERVICE_LEVEL = 0.95


def build_daily_demand_series(order_df: pd.DataFrame) -> dict[str, np.ndarray]:
    daily = (
        order_df.groupby(["item_code", "order_date"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"quantity": "daily_demand"})
        .sort_values(["item_code", "order_date"])
    )

    series_by_sku: dict[str, np.ndarray] = {}
    for sku, sku_frame in daily.groupby("item_code", sort=True):
        date_index = pd.date_range(
            sku_frame["order_date"].min(),
            sku_frame["order_date"].max(),
            freq="D",
        )
        dense_series = (
            sku_frame.set_index("order_date")["daily_demand"]
            .reindex(date_index, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        series_by_sku[str(sku)] = dense_series
    return series_by_sku


def build_lead_time_demand_sample(daily_demand: np.ndarray, lead_time_days: int) -> np.ndarray:
    values = np.asarray(daily_demand, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot build a lead-time demand sample from an empty daily demand series.")
    if lead_time_days <= 0:
        raise ValueError("lead_time_days must be positive.")
    if lead_time_days == 1:
        return values.copy()
    if values.size < lead_time_days:
        return np.asarray([values.sum()], dtype=float)
    window = np.ones(lead_time_days, dtype=float)
    return np.convolve(values, window, mode="valid")


def empirical_percentile_n_plus_one(values: np.ndarray, alpha: float) -> tuple[float, float, int, float]:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    if sorted_values.size == 0:
        raise ValueError("Cannot compute the percentile from an empty lead-time demand sample.")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in the interval (0, 1].")

    n = sorted_values.size
    position = float((n + 1) * alpha)
    if position > n:
        return float(sorted_values[-1]), position, n, 0.0
    if position <= 1.0:
        return float(sorted_values[0]), position, 1, 0.0

    rank = int(np.floor(position))
    weight = float(position - rank)
    if np.isclose(weight, 0.0):
        return float(sorted_values[rank - 1]), position, rank, 0.0

    lower_value = float(sorted_values[rank - 1])
    upper_value = float(sorted_values[rank])
    interpolated_value = (1.0 - weight) * lower_value + weight * upper_value
    return float(interpolated_value), position, rank, weight


def main():
    context = load_experiment_context(BASE_DIR)
    training_orders = context.train_eligible_df.copy()
    daily_demand_by_sku = build_daily_demand_series(training_orders)
    if not daily_demand_by_sku:
        raise ValueError("No training demand remains to estimate minimum inventory.")

    pre_t_order_counts = (
        context.sku_status_df.set_index("item_code")["pre_t_order_count"].astype(int).to_dict()
    )

    inventory_records = []
    for sku in context.eligible_skus:
        demand_series = daily_demand_by_sku.get(sku)
        if demand_series is None:
            raise ValueError(f"Missing daily demand series for SKU '{sku}'.")

        lead_time_sample = build_lead_time_demand_sample(demand_series, LEAD_TIME_DAYS)
        reorder_point, percentile_position, percentile_rank, percentile_weight = (
            empirical_percentile_n_plus_one(lead_time_sample, TARGET_SERVICE_LEVEL)
        )
        minimum_inventory_ceiling = int(np.ceil(max(0.0, reorder_point)))

        inventory_records.append(
            {
                "item_code": sku,
                "minimum_inventory": max(0.0, reorder_point),
                "minimum_inventory_ceiling": minimum_inventory_ceiling,
                "inventory_basis_source": "lead_time_percentile_n_plus_one",
                "paired_historical_product": "",
                "pre_t_order_count": int(pre_t_order_counts.get(sku, 0)),
                "daily_observation_count": int(demand_series.size),
                "lead_time_sample_size": int(lead_time_sample.size),
                "percentile_position": percentile_position,
                "percentile_rank": percentile_rank,
                "percentile_weight": percentile_weight,
            }
        )

    result_df = pd.DataFrame(inventory_records)
    if result_df.empty:
        raise ValueError("Minimum inventory results are empty.")

    result_df = result_df.sort_values("item_code").reset_index(drop=True)
    result_df.to_csv(OUTPUT_PATH, index=False, sep=";", encoding="utf-8-sig", decimal=",")

    detail_df = pd.DataFrame(
        [
            {"metric": "cutoff_ratio", "value": context.cutoff_ratio},
            {"metric": "cutoff_timestamp", "value": str(context.cutoff_time)},
            {"metric": "lead_time_days", "value": LEAD_TIME_DAYS},
            {"metric": "target_service_level", "value": TARGET_SERVICE_LEVEL},
            {
                "metric": "percentile_method",
                "value": "n_plus_1_linear_interpolation",
            },
            {"metric": "eligible_master_skus", "value": len(context.eligible_skus)},
            {"metric": "historical_skus", "value": len(context.historical_skus)},
            {"metric": "new_skus", "value": len(context.new_skus)},
            {"metric": "minimum_inventory_skus", "value": len(result_df)},
        ]
    )
    detail_df.to_csv(DETAIL_PATH, index=False, sep=";", encoding="utf-8-sig", decimal=",")

    print(f"Cutoff ratio: {context.cutoff_ratio:.2f}")
    print(f"Cutoff timestamp: {context.cutoff_time}")
    print(f"Lead time days: {LEAD_TIME_DAYS}")
    print(f"Target service level: {TARGET_SERVICE_LEVEL:.2%}")
    print("Percentile method: (n + 1)p with linear interpolation")
    print(f"Eligible SKUs: {len(context.eligible_skus):,}")
    print(f"Historical SKUs: {len(context.historical_skus):,}")
    print(f"New SKUs: {len(context.new_skus):,}")
    print(f"Minimum inventory saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
