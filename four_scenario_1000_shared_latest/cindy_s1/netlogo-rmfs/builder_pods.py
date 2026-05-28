from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from prepare_static_21day_inputs import (
    FCGMA_DIR,
    find_existing_directory,
    normalize_item_code,
    prepare_inputs,
)


def load_cindy_candidate_items(
    items_path: Path,
    max_comp_path: Path,
    eligible_master_path: Path | None = None,
) -> pd.DataFrame:
    items = pd.read_csv(items_path)
    max_comp = pd.read_csv(max_comp_path)

    items["item_code"] = items["item_code"].map(normalize_item_code)
    items["item_initial_quantity_inventory"] = pd.to_numeric(
        items["item_initial_quantity_inventory"], errors="coerce"
    ).fillna(0)
    items["item_order_frequency"] = pd.to_numeric(
        items.get("item_order_frequency", 0), errors="coerce"
    ).fillna(0)

    max_comp["item_code"] = max_comp["item_code"].map(normalize_item_code)
    max_comp_col = (
        "max_fit"
        if "max_fit" in max_comp.columns
        else "max_comp_number"
        if "max_comp_number" in max_comp.columns
        else max_comp.columns[-1]
    )
    max_comp = max_comp[["item_code", max_comp_col]].copy()
    max_comp.columns = ["item_code", "standard_slot_capacity"]
    max_comp["standard_slot_capacity"] = pd.to_numeric(
        max_comp["standard_slot_capacity"], errors="coerce"
    )

    if eligible_master_path is None:
        eligible_master_path = FCGMA_DIR / "eligible_master_skus.csv"
    eligible_skus: set[str] | None = None
    if eligible_master_path.exists():
        eligible_master = pd.read_csv(eligible_master_path)
        eligible_skus = set(
            eligible_master.iloc[:, 0].map(normalize_item_code)
        )

    items = items.merge(max_comp, on="item_code", how="inner")
    items = items[
        (items["item_code"].astype(str).str.strip() != "")
        & (items["item_initial_quantity_inventory"] > 0)
        & (items["standard_slot_capacity"] > 0)
    ].copy()
    if eligible_skus is not None:
        items = items[items["item_code"].isin(eligible_skus)].copy()

    return items


def load_target_budget_from_allocation(
    allocation_path: Path,
) -> dict[str, int]:
    allocation = pd.read_csv(allocation_path)
    required_columns = {"item", "quantity_in_that_slot"}
    missing_columns = required_columns - set(allocation.columns)
    if missing_columns:
        raise KeyError(
            f"Allocation file {allocation_path} is missing required columns: {sorted(missing_columns)}"
        )

    allocation["item"] = allocation["item"].map(normalize_item_code)
    allocation["quantity_in_that_slot"] = pd.to_numeric(
        allocation["quantity_in_that_slot"], errors="coerce"
    ).fillna(0)
    allocation = allocation[
        (allocation["item"].astype(str).str.strip() != "")
        & (allocation["quantity_in_that_slot"] > 0)
    ].copy()

    target_budget = (
        allocation.groupby("item", as_index=False)["quantity_in_that_slot"]
        .sum()
        .rename(columns={"item": "item_code", "quantity_in_that_slot": "target_quantity"})
    )
    target_budget["target_quantity"] = np.ceil(
        pd.to_numeric(target_budget["target_quantity"], errors="coerce").fillna(0)
    ).astype(int)

    return dict(zip(target_budget["item_code"], target_budget["target_quantity"]))


def normalize_target_budget(
    target_quantity_by_sku: dict[str, int] | None,
) -> dict[str, int]:
    if target_quantity_by_sku is None:
        return {}

    return {
        normalize_item_code(item_code): int(np.ceil(quantity))
        for item_code, quantity in target_quantity_by_sku.items()
        if normalize_item_code(item_code) and quantity is not None
    }


def build_slot_quantities(total_quantity: int, slot_capacity: int, slots_needed: int) -> list[int]:
    total_quantity = int(max(0, total_quantity))
    slot_capacity = int(max(1, slot_capacity))
    slots_needed = int(max(0, slots_needed))
    if slots_needed <= 0:
        return []

    full_slots, remainder = divmod(total_quantity, slot_capacity)
    quantities = [slot_capacity] * int(full_slots)
    if remainder > 0:
        quantities.append(int(remainder))

    if len(quantities) < slots_needed:
        quantities.extend([slot_capacity] * (slots_needed - len(quantities)))
    elif len(quantities) > slots_needed:
        quantities = quantities[:slots_needed]

    return quantities


def build_cindy_baseline_allocation(
    items_path: Path,
    max_comp_path: Path,
    output_path: Path,
    required_item_codes: set[str],
    target_quantity_by_sku: dict[str, int] | None = None,
    class_slot_counts: dict[str, int] | None = None,
    eligible_master_path: Path | None = None,
) -> Path:
    if class_slot_counts is None:
        class_slot_counts = {"A": 12, "B": 21, "C": 7}

    items = load_cindy_candidate_items(items_path=items_path, max_comp_path=max_comp_path)
    required_item_codes = {normalize_item_code(code) for code in required_item_codes if normalize_item_code(code)}
    items = items[items["item_code"].isin(required_item_codes)].copy()

    if target_quantity_by_sku is not None:
        normalized_budget = {
            normalize_item_code(item_code): int(np.ceil(quantity))
            for item_code, quantity in target_quantity_by_sku.items()
            if normalize_item_code(item_code) and quantity is not None
        }
        items["item_initial_quantity_inventory"] = items["item_code"].map(
            normalized_budget
        )
        items = items[
            pd.to_numeric(items["item_initial_quantity_inventory"], errors="coerce").fillna(0) > 0
        ].copy()

    missing_required = sorted(required_item_codes - set(items["item_code"]))
    if missing_required:
        preview = ", ".join(missing_required[:10])
        raise ValueError(
            f"Cindy baseline rebuild is missing {len(missing_required)} required sampled SKUs, for example: {preview}"
        )

    items["item_class"] = items["item_class"].astype(str).str.strip()
    items["slots_needed"] = np.ceil(
        items["item_initial_quantity_inventory"] / items["standard_slot_capacity"]
    ).astype(int)

    items = items.sort_values(
        ["item_class", "item_order_frequency", "item_code"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()

    class_totals = (
        items.groupby("item_class")["slots_needed"].sum().to_dict()
    )
    pods_required = max(
        1,
        max(
            int(-(-class_totals.get(item_class, 0) // slot_count))
            for item_class, slot_count in class_slot_counts.items()
            if slot_count > 0
        ),
    )

    class_slot_offsets: dict[str, tuple[int, int]] = {}
    cursor = 1
    for item_class, slot_count in class_slot_counts.items():
        class_slot_offsets[item_class] = (cursor, cursor + slot_count - 1)
        cursor += slot_count

    class_pools: dict[str, list[tuple[int, int]]] = {
        item_class: [] for item_class in class_slot_counts
    }
    for pod in range(1, pods_required + 1):
        for item_class, (slot_start, slot_end) in class_slot_offsets.items():
            class_pools[item_class].extend(
                [(pod, slot) for slot in range(slot_start, slot_end + 1)]
            )

    records: list[dict[str, int | str]] = []
    for item_class in class_slot_counts:
        pool = class_pools[item_class]
        pool_index = 0
        items_of_class = items[items["item_class"] == item_class]
        for _, row in items_of_class.iterrows():
            slots_needed = int(row["slots_needed"])
            if pool_index + slots_needed > len(pool):
                raise ValueError(
                    f"Baseline rebuild ran out of {item_class}-class slots for SKU {row['item_code']}."
                )

            qty_per_slot = int(row["standard_slot_capacity"])
            for pod, slot in pool[pool_index : pool_index + slots_needed]:
                records.append(
                    {
                        "pod": int(pod),
                        "slot": int(slot),
                        "item": row["item_code"],
                        "quantity_in_that_slot": qty_per_slot,
                    }
                )
            pool_index += slots_needed

    allocation = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allocation.to_csv(output_path, index=False)
    return output_path


def build_cindy_scenario2_allocation(
    items_path: Path,
    max_comp_path: Path,
    output_path: Path,
    required_item_codes: set[str],
    target_quantity_by_sku: dict[str, int] | None = None,
    slots_per_pod: int = 40,
    eligible_master_path: Path | None = None,
) -> Path:
    items = load_cindy_candidate_items(
        items_path=items_path,
        max_comp_path=max_comp_path,
        eligible_master_path=eligible_master_path,
    )
    required_item_codes = {
        normalize_item_code(code) for code in required_item_codes if normalize_item_code(code)
    }
    items = items[items["item_code"].isin(required_item_codes)].copy()

    if target_quantity_by_sku is not None:
        normalized_budget = normalize_target_budget(target_quantity_by_sku)
        items["item_initial_quantity_inventory"] = items["item_code"].map(normalized_budget)
        items = items[
            pd.to_numeric(items["item_initial_quantity_inventory"], errors="coerce").fillna(0) > 0
        ].copy()

    missing_required = sorted(required_item_codes - set(items["item_code"]))
    if missing_required:
        preview = ", ".join(missing_required[:10])
        raise ValueError(
            f"Cindy Scenario 2 rebuild is missing {len(missing_required)} required sampled SKUs, for example: {preview}"
        )

    items["item_class"] = items["item_class"].astype(str).str.strip()
    items["slots_needed"] = np.ceil(
        items["item_initial_quantity_inventory"] / items["standard_slot_capacity"]
    ).astype(int)
    items = items.sort_values(
        ["item_class", "item_order_frequency", "item_code"],
        ascending=[True, False, True],
        kind="stable",
    ).copy()

    records: list[dict[str, int | str]] = []
    next_pod_id = 1
    for item_class in ["A", "B", "C"]:
        class_items = items[items["item_class"] == item_class].copy()
        if class_items.empty:
            continue

        pod_pool: list[tuple[int, int]] = []
        class_total_slots = int(class_items["slots_needed"].sum())
        pods_required = max(1, int(np.ceil(class_total_slots / slots_per_pod)))
        for pod in range(next_pod_id, next_pod_id + pods_required):
            pod_pool.extend((pod, slot) for slot in range(1, slots_per_pod + 1))
        next_pod_id += pods_required

        pool_index = 0
        for _, row in class_items.iterrows():
            slots_needed = int(row["slots_needed"])
            if pool_index + slots_needed > len(pod_pool):
                raise ValueError(
                    f"Scenario 2 rebuild ran out of {item_class}-class slots for SKU {row['item_code']}."
                )

            slot_quantities = build_slot_quantities(
                total_quantity=int(row["item_initial_quantity_inventory"]),
                slot_capacity=int(row["standard_slot_capacity"]),
                slots_needed=slots_needed,
            )
            for (pod, slot), qty_per_slot in zip(
                pod_pool[pool_index : pool_index + slots_needed], slot_quantities
            ):
                records.append(
                    {
                        "pod": int(pod),
                        "slot": int(slot),
                        "item": row["item_code"],
                        "quantity_in_that_slot": int(qty_per_slot),
                    }
                )
            pool_index += slots_needed

    allocation = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allocation.to_csv(output_path, index=False)
    return output_path


def build_sample_orders(
    full_cutoff_orders_path: Path,
    sample_orders_path: Path,
    sample_unique_orders: int,
    allowed_item_codes: set[str] | None = None,
) -> dict:
    orders = pd.read_csv(full_cutoff_orders_path, sep=";", encoding="utf-8-sig")
    if "order_id" not in orders.columns:
        raise KeyError(
            f"Expected 'order_id' in {full_cutoff_orders_path}, found {list(orders.columns)}"
        )
    if "item_code" not in orders.columns:
        raise KeyError(
            f"Expected 'item_code' in {full_cutoff_orders_path}, found {list(orders.columns)}"
        )

    orders["item_code"] = orders["item_code"].map(normalize_item_code)
    if allowed_item_codes is not None:
        allowed_item_codes = set(allowed_item_codes)
        fully_covered_order_ids = orders.groupby("order_id")["item_code"].apply(
            lambda series: set(series).issubset(allowed_item_codes)
        )
        valid_order_ids = fully_covered_order_ids[fully_covered_order_ids].index
        orders = orders[orders["order_id"].isin(valid_order_ids)].copy()

    unique_order_ids = orders["order_id"].drop_duplicates().iloc[:sample_unique_orders]
    sampled = orders[orders["order_id"].isin(unique_order_ids)].copy()
    if sampled.empty:
        raise ValueError("The requested sample produced no post-cutoff order rows.")

    sample_orders_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(sample_orders_path, sep=";", index=False, encoding="utf-8-sig")

    created_col = "created_at" if "created_at" in sampled.columns else sampled.columns[-1]
    created = pd.to_datetime(sampled[created_col], errors="coerce")
    span_minutes = 0.0
    if created.notna().any():
        span_minutes = (created.max() - created.min()).total_seconds() / 60.0

    return {
        "sample_unique_orders": int(sampled["order_id"].nunique()),
        "sample_order_lines": int(len(sampled)),
        "sample_arrival_span_minutes": float(span_minutes),
        "sample_unique_skus": int(sampled["item_code"].nunique()),
    }


def summary_to_dict(summary_df: pd.DataFrame) -> dict[str, float]:
    return {
        str(row["metric"]): row["value"]
        for _, row in summary_df.iterrows()
    }


def run_experiment_capped(
    sim,
    max_ticks: int,
    record_every: int,
    progress_every: int,
    suppress_stdout: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if suppress_stdout:
        devnull_handle = open(os.devnull, "w")
        stdout_context = contextlib.redirect_stdout(devnull_handle)
    else:
        devnull_handle = None
        stdout_context = contextlib.nullcontext()

    with stdout_context:
        setup_result = sim.setup()
    if isinstance(setup_result, str) and "error" in setup_result.lower():
        if devnull_handle is not None:
            devnull_handle.close()
        raise RuntimeError(setup_result)

    records = []
    stopped_cleanly = False
    next_record_step = int(record_every)
    next_progress_step = int(progress_every) if progress_every else None
    last_recorded_tick = 0

    with stdout_context:
        for _ in range(max_ticks):
            tick_result = sim.tick()
            if tick_result == "STOP":
                stopped_cleanly = True
                break

            current_step = int(sim.warehouse._step)
            current_tick = float(sim.warehouse._tick)

            if next_progress_step is not None and current_step >= next_progress_step:
                warehouse = sim.warehouse
                progress_line = (
                    f"[progress] allocation={getattr(sim, '_comparison_label', 'unknown')} "
                    f"step={int(warehouse._step)} "
                    f"tick={int(warehouse._tick)} "
                    f"last_arrival={int(getattr(warehouse, 'last_order_arrival', 0))} "
                    f"fulfilled={int(warehouse.orders_fulfilled)}/"
                    f"{int(getattr(warehouse, 'total_orders_expected', 0))} "
                    f"job_queue={len(warehouse.job_queue)} "
                    f"sku_queue={len(warehouse.sku_picking_queue)} "
                    f"pod_visits={int(warehouse.pod_visit_to_station)} "
                    f"replenishment_trips={int(warehouse.replenishment_trips)}"
                )
                print(progress_line, file=sys.__stdout__, flush=True)
                next_progress_step += int(progress_every)

            if isinstance(tick_result, list) and current_step >= next_record_step:
                records.append(
                    {
                        "step": int(sim.warehouse._step),
                        "tick": int(sim.warehouse._tick),
                        "total_energy": tick_result[1],
                        "job_queue_length": tick_result[2],
                        "stop_and_go": tick_result[3],
                        "total_turning": tick_result[4],
                        "replenishment_count": tick_result[5],
                        "replenishment_trips": tick_result[6],
                        "pod_visit_to_station": tick_result[7],
                        "orders_fulfilled": tick_result[8],
                        "average_inventory_level": tick_result[9],
                        "energy_per_order": tick_result[10],
                        "average_pod_inventory_level": tick_result[11],
                        "average_weighted_pod_utilization": tick_result[12],
                        "total_fixed_load_energy": tick_result[13],
                        "fixed_energy_per_order": tick_result[14],
                    }
                )
                last_recorded_tick = int(sim.warehouse._tick)
                next_record_step += int(record_every)

    if devnull_handle is not None:
        devnull_handle.close()

    warehouse = sim.warehouse
    elapsed_hours = float(warehouse._tick) / 60.0 if warehouse._tick else 0.0
    throughput_per_hour = (
        warehouse.orders_fulfilled / elapsed_hours if elapsed_hours > 0 else 0.0
    )

    summary = pd.DataFrame(
        [
            {"metric": "stopped_cleanly", "value": int(stopped_cleanly)},
            {"metric": "ticks_elapsed", "value": int(warehouse._tick)},
            {"metric": "steps_elapsed", "value": int(warehouse._step)},
            {"metric": "record_every_steps", "value": int(record_every)},
            {"metric": "last_recorded_tick", "value": int(last_recorded_tick)},
            {"metric": "orders_expected", "value": int(getattr(warehouse, "total_orders_expected", 0))},
            {"metric": "orders_fulfilled", "value": int(warehouse.orders_fulfilled)},
            {
                "metric": "completion_rate",
                "value": (
                    warehouse.orders_fulfilled / float(getattr(warehouse, "total_orders_expected", 1))
                    if getattr(warehouse, "total_orders_expected", 0)
                    else 0.0
                ),
            },
            {"metric": "last_order_arrival_tick", "value": int(getattr(warehouse, "last_order_arrival", 0))},
            {"metric": "total_energy", "value": float(warehouse.total_energy)},
            {"metric": "total_fixed_load_energy", "value": float(warehouse.total_fixed_load_energy)},
            {
                "metric": "energy_per_fulfilled_order",
                "value": (
                    warehouse.total_energy / warehouse.orders_fulfilled
                    if warehouse.orders_fulfilled > 0
                    else 0.0
                ),
            },
            {
                "metric": "fixed_energy_per_fulfilled_order",
                "value": (
                    warehouse.total_fixed_load_energy / warehouse.orders_fulfilled
                    if warehouse.orders_fulfilled > 0
                    else 0.0
                ),
            },
            {"metric": "throughput_orders_per_hour", "value": throughput_per_hour},
            {"metric": "stop_and_go", "value": int(warehouse.stop_and_go)},
            {"metric": "total_turning", "value": float(warehouse.total_turning)},
            {"metric": "replenishment_count", "value": int(warehouse.replenishment_count)},
            {"metric": "replenishment_trips", "value": int(warehouse.replenishment_trips)},
            {"metric": "pod_visit_to_station", "value": int(warehouse.pod_visit_to_station)},
            {"metric": "average_inventory_level", "value": float(warehouse.average_inventory_level)},
            {"metric": "average_pod_inventory_level", "value": float(warehouse.average_pod_inventory_level)},
            {
                "metric": "average_weighted_pod_utilization",
                "value": float(warehouse.average_weighted_pod_utilization),
            },
            {"metric": "job_queue_length", "value": int(len(warehouse.job_queue))},
            {"metric": "sku_queue_length", "value": int(len(warehouse.sku_picking_queue))},
            {"metric": "all_robots_idle", "value": int(warehouse.allRobotsIdle())},
        ]
    )

    return pd.DataFrame(records), summary


def run_one_sample(
    scenario_root: Path,
    allocation_path: Path,
    metadata_path: Path,
    translated_info_path: Path,
    max_comp_path: Path,
    sample_orders_path: Path,
    required_coverage_skus: set[str],
    max_ticks: int,
    record_every: int,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepare_inputs(
        scenario_root=scenario_root,
        allocation_path=allocation_path,
        metadata_path=metadata_path,
        translated_info_path=translated_info_path,
        max_comp_path=max_comp_path,
        required_coverage_skus=required_coverage_skus,
    )

    replay_path = scenario_root / "data" / "input" / "cutoff_test_orders.csv"
    replay_path.write_bytes(sample_orders_path.read_bytes())

    os.chdir(scenario_root)
    if str(scenario_root) not in sys.path:
        sys.path.insert(0, str(scenario_root))

    if "netlogo" in sys.modules:
        sim = importlib.reload(sys.modules["netlogo"])
    else:
        sim = importlib.import_module("netlogo")
    sim._comparison_label = allocation_path.stem

    metrics_df, summary_df = run_experiment_capped(
        sim=sim,
        max_ticks=max_ticks,
        record_every=record_every,
        progress_every=progress_every,
        suppress_stdout=True,
    )
    return metrics_df, summary_df


def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parents[1]
    preprocessing_dir = find_existing_directory(
        [
            FCGMA_DIR / "Preprocessing",
            workspace_dir / "Preprocessing",
            workspace_dir.parent / "Preprocessing",
        ],
        required_files=["preprocessed_final.csv"],
    )
    cindy_root = find_existing_directory(
        [
            workspace_dir / "netlogoCindy" / "netlogo-rmfs-Skenario-3-Cindy-revisi" / "netlogo-rmfs-Skenario-3-Cindy-revisi",
            workspace_dir / "netlogo-rmfs-Skenario-3-Cindy-revisi" / "netlogo-rmfs-Skenario-3-Cindy-revisi",
            workspace_dir / "netlogo-rmfs-Skenario-3-Cindy-revisi",
            workspace_dir.parent / "netlogoCindy" / "netlogo-rmfs-Skenario-3-Cindy-revisi" / "netlogo-rmfs-Skenario-3-Cindy-revisi",
            workspace_dir.parent / "netlogo-rmfs-Skenario-3-Cindy-revisi" / "netlogo-rmfs-Skenario-3-Cindy-revisi",
            workspace_dir.parent / "netlogo-rmfs-Skenario-3-Cindy-revisi",
        ],
        required_files=["data/output/items.csv", "data/output/pods.csv"],
    )

    parser = argparse.ArgumentParser(
        description="Compare sample train/test RMFS runs between the user's allocation and Cindy Scenario 3 allocation."
    )
    parser.add_argument(
        "--my-allocation",
        type=Path,
        default=FCGMA_DIR / "results_fcgma" / "Z_trial_1.csv",
        help="Path to the user's FCGMA allocation CSV.",
    )
    parser.add_argument(
        "--sample-orders",
        type=int,
        default=200,
        help="Number of unique post-cutoff orders to keep in the sample replay.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=12000,
        help="Safety cap for each sample run.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=100,
        help="Record metrics every N simulation steps.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N simulation steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--baseline-target-source",
        choices=["cindy", "my_allocation"],
        default="cindy",
        help="Use Cindy's original stock targets or copy per-SKU stock targets from the user's allocation.",
    )
    args = parser.parse_args()

    my_allocation_df_prefilter = pd.read_csv(args.my_allocation)
    my_allocation_skus_prefilter = set(
        my_allocation_df_prefilter["item"].map(normalize_item_code)
    )
    baseline_candidate_items_prefilter = load_cindy_candidate_items(
        items_path=cindy_root / "data" / "output" / "items.csv",
        max_comp_path=FCGMA_DIR / "max_comp_number.csv",
    )
    baseline_candidate_skus_prefilter = set(
        baseline_candidate_items_prefilter["item_code"]
    )
    common_covered_skus_prefilter = {
        sku
        for sku in (
            my_allocation_skus_prefilter & baseline_candidate_skus_prefilter
        )
        if sku
    }

    full_cutoff_orders_path = script_dir / "data" / "input" / "cutoff_test_orders.csv"
    if True:
        prepare_inputs(
            scenario_root=script_dir,
            allocation_path=args.my_allocation,
            metadata_path=preprocessing_dir / "preprocessed_final.csv",
            translated_info_path=preprocessing_dir / "儲格設計_原檔(商品資訊)(Translated).csv",
            max_comp_path=FCGMA_DIR / "max_comp_number.csv",
            required_coverage_skus=common_covered_skus_prefilter,
        )

    my_allocation_df = pd.read_csv(args.my_allocation)
    my_allocation_skus = set(my_allocation_df["item"].map(normalize_item_code))
    baseline_candidate_items = load_cindy_candidate_items(
        items_path=cindy_root / "data" / "output" / "items.csv",
        max_comp_path=FCGMA_DIR / "max_comp_number.csv",
    )
    baseline_candidate_skus = set(baseline_candidate_items["item_code"])
    common_covered_skus = {
        sku for sku in (my_allocation_skus & baseline_candidate_skus) if sku
    }

    sample_orders_path = script_dir / "data" / "input" / "cutoff_test_orders_sample.csv"
    sample_info = build_sample_orders(
        full_cutoff_orders_path=full_cutoff_orders_path,
        sample_orders_path=sample_orders_path,
        sample_unique_orders=args.sample_orders,
        allowed_item_codes=common_covered_skus,
    )
    sampled_orders_df = pd.read_csv(sample_orders_path, sep=";", encoding="utf-8-sig")
    required_sample_skus = set(sampled_orders_df["item_code"].map(normalize_item_code))

    baseline_target_quantity_by_sku = None
    if args.baseline_target_source == "my_allocation":
        baseline_target_quantity_by_sku = load_target_budget_from_allocation(args.my_allocation)

    baseline_allocation_path = script_dir / "data" / "input" / "scenario3_baseline_allocation.csv"
    build_cindy_baseline_allocation(
        items_path=cindy_root / "data" / "output" / "items.csv",
        max_comp_path=FCGMA_DIR / "max_comp_number.csv",
        output_path=baseline_allocation_path,
        required_item_codes=required_sample_skus,
        target_quantity_by_sku=baseline_target_quantity_by_sku,
    )

    allocations = [
        ("my_allocation", args.my_allocation),
        ("scenario3_baseline", baseline_allocation_path),
    ]

    result_rows: list[dict] = []
    for label, allocation_path in allocations:
        metrics_df, summary_df = run_one_sample(
            scenario_root=script_dir,
            allocation_path=allocation_path,
            metadata_path=preprocessing_dir / "preprocessed_final.csv",
            translated_info_path=preprocessing_dir / "儲格設計_原檔(商品資訊)(Translated).csv",
            max_comp_path=FCGMA_DIR / "max_comp_number.csv",
            sample_orders_path=sample_orders_path,
            required_coverage_skus=required_sample_skus,
            max_ticks=args.max_ticks,
            record_every=args.record_every,
            progress_every=args.progress_every,
        )

        metrics_output = script_dir / "data" / "output" / f"{label}_sample_tick_metrics.csv"
        summary_output = script_dir / "data" / "output" / f"{label}_sample_summary.csv"
        metrics_df.to_csv(metrics_output, index=False)
        summary_df.to_csv(summary_output, index=False)

        summary = summary_to_dict(summary_df)
        pod_visits = float(summary.get("pod_visit_to_station", 0.0) or 0.0)
        orders_fulfilled = float(summary.get("orders_fulfilled", 0.0) or 0.0)
        average_pile_on = orders_fulfilled / pod_visits if pod_visits > 0 else 0.0

        result_rows.append(
            {
                "allocation_label": label,
                "allocation_path": str(allocation_path),
                **sample_info,
                "ticks_elapsed": summary.get("ticks_elapsed", 0),
                "orders_expected": summary.get("orders_expected", 0),
                "orders_fulfilled": summary.get("orders_fulfilled", 0),
                "completion_rate": summary.get("completion_rate", 0.0),
                "throughput_orders_per_hour": summary.get("throughput_orders_per_hour", 0.0),
                "energy_per_fulfilled_order": summary.get("energy_per_fulfilled_order", 0.0),
                "fixed_energy_per_fulfilled_order": summary.get("fixed_energy_per_fulfilled_order", 0.0),
                "total_energy": summary.get("total_energy", 0.0),
                "stop_and_go": summary.get("stop_and_go", 0),
                "replenishment_trips": summary.get("replenishment_trips", 0),
                "pod_visit_to_station": summary.get("pod_visit_to_station", 0),
                "average_pile_on": average_pile_on,
                "average_inventory_level": summary.get("average_inventory_level", 0.0),
                "average_pod_inventory_level": summary.get("average_pod_inventory_level", 0.0),
                "average_weighted_pod_utilization": summary.get("average_weighted_pod_utilization", 0.0),
            }
        )

    comparison_df = pd.DataFrame(result_rows)
    comparison_path = script_dir / "data" / "output" / "sample_allocation_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("Sample comparison completed.")
    print(f"Comparison: {comparison_path}")
    print(comparison_df.to_string(index=False))


def builder_only_main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parents[1]
    preprocessing_dir = find_existing_directory(
        [
            FCGMA_DIR / "Preprocessing",
            workspace_dir / "Preprocessing",
            workspace_dir.parent / "Preprocessing",
        ],
        required_files=["preprocessed_final.csv"],
    )
    translated_info_default = next(
        preprocessing_dir.glob("*Translated*.csv"),
        preprocessing_dir / "å„²æ ¼è¨­è¨ˆ_åŽŸæª”(å•†å“è³‡è¨Š)(Translated).csv",
    )

    parser = argparse.ArgumentParser(
        description="Build a selected allocation and prepare RMFS inputs only. No simulation is run."
    )
    parser.add_argument(
        "--scenario",
        choices=["my_allocation", "scenario2_baseline", "scenario3_baseline"],
        default="my_allocation",
        help="Which allocation should be prepared into RMFS inputs.",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=script_dir,
        help="RMFS scenario root where items.csv, pods.csv, and replay inputs will be written.",
    )
    parser.add_argument(
        "--my-allocation",
        type=Path,
        default=FCGMA_DIR / "results_fcgma" / "Z_trial_1.csv",
        help="Path to the user's FCGMA allocation CSV.",
    )
    parser.add_argument(
        "--candidate-items",
        type=Path,
        default=script_dir / "data" / "output" / "items.csv",
        help="Candidate items file used when building Cindy baselines.",
    )
    parser.add_argument(
        "--eligible-master",
        type=Path,
        default=FCGMA_DIR / "eligible_master_skus.csv",
        help="Eligible SKU universe used for the baseline rebuild and RMFS coverage check.",
    )
    parser.add_argument(
        "--output-allocation",
        type=Path,
        default=None,
        help="Optional path for the built baseline allocation CSV. Defaults to the scenario root input folder.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=preprocessing_dir / "preprocessed_final.csv",
        help="Path to the aligned product metadata CSV.",
    )
    parser.add_argument(
        "--translated-info",
        type=Path,
        default=translated_info_default,
        help="Path to the translated item dimension/weight master CSV.",
    )
    parser.add_argument(
        "--max-comp",
        type=Path,
        default=FCGMA_DIR / "max_comp_number.csv",
        help="Path to the slot-capacity-by-SKU CSV.",
    )
    parser.add_argument(
        "--minimum-inventory",
        type=Path,
        default=FCGMA_DIR / "minimum_inventory.csv",
        help="Path to minimum_inventory.csv used by prepare_static_21day_inputs.py.",
    )
    parser.add_argument(
        "--cutoff-ratio",
        type=float,
        default=None,
        help="Optional cutoff ratio override for prepare_static_21day_inputs.py.",
    )
    parser.add_argument(
        "--baseline-target-source",
        choices=["current_items", "my_allocation"],
        default="current_items",
        help="Keep per-SKU quantities from --candidate-items, or copy them from --my-allocation.",
    )
    parser.add_argument(
        "--pod-id-policy",
        choices=["identity", "seeded_shuffle"],
        default="identity",
        help="How logical pod labels should be mapped to physical RMFS pod ids.",
    )
    parser.add_argument(
        "--pod-id-seed",
        type=int,
        default=42,
        help="Seed used when --pod-id-policy is seeded_shuffle.",
    )
    args = parser.parse_args()

    scenario_root = args.scenario_root.resolve()
    required_item_codes = set(
        pd.read_csv(args.eligible_master).iloc[:, 0].map(normalize_item_code)
    ) - {""}

    allocation_path = args.my_allocation
    if args.scenario != "my_allocation":
        baseline_target_quantity_by_sku = None
        if args.baseline_target_source == "my_allocation":
            baseline_target_quantity_by_sku = load_target_budget_from_allocation(args.my_allocation)

        default_name = (
            "scenario2_baseline_allocation.csv"
            if args.scenario == "scenario2_baseline"
            else "scenario3_baseline_allocation.csv"
        )
        output_allocation_path = (
            args.output_allocation
            if args.output_allocation is not None
            else scenario_root / "data" / "input" / default_name
        )

        if args.scenario == "scenario2_baseline":
            allocation_path = build_cindy_scenario2_allocation(
                items_path=args.candidate_items,
                max_comp_path=args.max_comp,
                output_path=output_allocation_path,
                required_item_codes=required_item_codes,
                target_quantity_by_sku=baseline_target_quantity_by_sku,
                eligible_master_path=args.eligible_master,
            )
        else:
            allocation_path = build_cindy_baseline_allocation(
                items_path=args.candidate_items,
                max_comp_path=args.max_comp,
                output_path=output_allocation_path,
                required_item_codes=required_item_codes,
                target_quantity_by_sku=baseline_target_quantity_by_sku,
                eligible_master_path=args.eligible_master,
            )

    result = prepare_inputs(
        scenario_root=scenario_root,
        allocation_path=allocation_path,
        metadata_path=args.metadata,
        translated_info_path=args.translated_info,
        max_comp_path=args.max_comp,
        minimum_inventory_path=args.minimum_inventory,
        cutoff_ratio=args.cutoff_ratio,
        required_coverage_skus=required_item_codes,
        pod_id_policy=args.pod_id_policy,
        pod_id_seed=args.pod_id_seed if args.pod_id_policy == "seeded_shuffle" else None,
    )

    print("Allocation build + RMFS input preparation completed.")
    print(f"Scenario root:    {scenario_root}")
    print(f"Prepared from:    {allocation_path}")
    print(f"Live items.csv:   {result['items_live_path']}")
    print(f"Live pods.csv:    {result['pods_live_path']}")
    print(f"Replay orders:    {result['replay_orders_path']}")
    print(f"Pod id mapping:   {result['pod_mapping_path']}")
    print(
        f"Coverage: {result['allocated_skus']} allocated SKUs, "
        f"{result['eligible_ordered_skus']} eligible ordered SKUs, "
        f"{result['pods_used']} pods used, "
        f"{result['occupied_slots']} occupied slots"
    )


if __name__ == "__main__":
    builder_only_main()
