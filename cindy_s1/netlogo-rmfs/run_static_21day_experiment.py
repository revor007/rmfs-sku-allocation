from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path

import pandas as pd

from prepare_static_21day_inputs import find_existing_directory, prepare_inputs


def run_experiment(
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

    if record_every <= 0:
        raise ValueError("record_every must be a positive integer.")
    if progress_every < 0:
        raise ValueError("progress_every must be zero or a positive integer.")

    records = []
    last_recorded_tick = None
    next_record_step = int(record_every)
    next_progress_step = int(progress_every) if progress_every else None
    with stdout_context:
        for _ in range(max_ticks):
            tick_result = sim.tick()
            if tick_result == "STOP":
                break

            current_step = int(sim.warehouse._step)
            current_tick = float(sim.warehouse._tick)

            if next_progress_step is not None and current_step >= next_progress_step:
                warehouse = sim.warehouse
                progress_line = (
                    f"[progress] step={int(warehouse._step)} "
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
        else:
            raise RuntimeError(
                f"Simulation did not stop within {max_ticks} ticks. Increase --max-ticks or inspect the replay state."
            )
    if devnull_handle is not None:
        devnull_handle.close()

    warehouse = sim.warehouse
    elapsed_hours = float(warehouse._tick) / 60.0 if warehouse._tick else 0.0
    throughput_per_hour = (
        warehouse.orders_fulfilled / elapsed_hours if elapsed_hours > 0 else 0.0
    )
    delivered_order_lines = int(getattr(warehouse, "delivered_order_lines", 0))
    picked_units = int(getattr(warehouse, "total_picked_units", 0))
    pod_visits = int(warehouse.pod_visit_to_station)
    total_energy = float(warehouse.total_energy)
    total_fixed_load_energy = float(warehouse.total_fixed_load_energy)
    variable_energy = max(0.0, total_energy - total_fixed_load_energy)

    summary = pd.DataFrame(
        [
            {"metric": "ticks_elapsed", "value": int(warehouse._tick)},
            {"metric": "record_every_ticks", "value": int(record_every)},
            {"metric": "last_recorded_tick", "value": int(last_recorded_tick or 0)},
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
            {"metric": "total_energy", "value": total_energy},
            {"metric": "total_fixed_load_energy", "value": total_fixed_load_energy},
            {"metric": "variable_energy", "value": variable_energy},
            {
                "metric": "energy_per_fulfilled_order",
                "value": (
                    total_energy / warehouse.orders_fulfilled
                    if warehouse.orders_fulfilled > 0
                    else 0.0
                ),
            },
            {
                "metric": "fixed_energy_per_fulfilled_order",
                "value": (
                    total_fixed_load_energy / warehouse.orders_fulfilled
                    if warehouse.orders_fulfilled > 0
                    else 0.0
                ),
            },
            {
                "metric": "variable_energy_per_delivered_line",
                "value": (
                    variable_energy / delivered_order_lines
                    if delivered_order_lines > 0
                    else 0.0
                ),
            },
            {"metric": "throughput_orders_per_hour", "value": throughput_per_hour},
            {"metric": "stop_and_go", "value": int(warehouse.stop_and_go)},
            {"metric": "total_turning", "value": float(warehouse.total_turning)},
            {"metric": "replenishment_count", "value": int(warehouse.replenishment_count)},
            {"metric": "replenishment_trips", "value": int(warehouse.replenishment_trips)},
            {"metric": "pod_visit_to_station", "value": pod_visits},
            {"metric": "delivered_order_lines", "value": delivered_order_lines},
            {"metric": "picked_units", "value": picked_units},
            {
                "metric": "delivered_order_lines_per_pod_visit",
                "value": (
                    delivered_order_lines / pod_visits
                    if pod_visits > 0
                    else 0.0
                ),
            },
            {
                "metric": "picked_units_per_pod_visit",
                "value": (
                    picked_units / pod_visits
                    if pod_visits > 0
                    else 0.0
                ),
            },
            {
                "metric": "energy_per_pod_visit",
                "value": (total_energy / pod_visits) if pod_visits > 0 else 0.0,
            },
            {
                "metric": "variable_energy_per_pod_visit",
                "value": (
                    variable_energy / pod_visits
                    if pod_visits > 0
                    else 0.0
                ),
            },
            {"metric": "average_inventory_level", "value": float(warehouse.average_inventory_level)},
            {"metric": "average_pod_inventory_level", "value": float(warehouse.average_pod_inventory_level)},
            {
                "metric": "average_weighted_pod_utilization",
                "value": float(warehouse.average_weighted_pod_utilization),
            },
        ]
    )

    return pd.DataFrame(records), summary


def main():
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parents[1]
    fcgma_dir = find_existing_directory(
        [
            workspace_dir,
            workspace_dir / "fcgma",
            workspace_dir / "revision-fcgma-copy",
            workspace_dir / "revision-fcgma - Copy" / "rmfs-sku-allocation",
            workspace_dir.parent / "fcgma",
            workspace_dir.parent / "revision-fcgma-copy",
            workspace_dir.parent / "revision-fcgma - Copy" / "rmfs-sku-allocation",
        ],
        required_files=["max_comp_number.csv"],
    )
    preprocessing_dir = find_existing_directory(
        [
            fcgma_dir / "Preprocessing",
            workspace_dir / "Preprocessing",
            workspace_dir.parent / "Preprocessing",
        ],
        required_files=["preprocessed_final.csv"],
    )
    output_dir = script_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Run the static 21-day RMFS Scenario 3 experiment with actual orders and collect energy metrics."
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Prepare RMFS items.csv and pods.csv from the frozen FCGMA allocation before running.",
    )
    parser.add_argument(
        "--allocation",
        type=Path,
        default=fcgma_dir / "results_fcgma" / "Z_trial_1.csv",
        help="Path to the frozen Z allocation CSV.",
    )
    parser.add_argument(
        "--orders",
        type=Path,
        default=preprocessing_dir / "訂單資料_final.csv",
        help="Path to the 21-day order horizon CSV.",
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
        default=preprocessing_dir / "儲格設計_原檔(商品資訊)(Translated).csv",
        help="Path to the translated item dimension/weight master CSV.",
    )
    parser.add_argument(
        "--max-comp",
        type=Path,
        default=fcgma_dir / "max_comp_number.csv",
        help="Path to the slot-capacity-by-SKU CSV.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=250000,
        help="Safety cap for the simulation loop.",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=10,
        help="Record tick-level metrics every N ticks to reduce logging overhead.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a lightweight progress line every N ticks. Use 0 to disable.",
    )
    args = parser.parse_args()

    if args.prepare:
        prepare_inputs(
            scenario_root=script_dir,
            allocation_path=args.allocation,
            order_path=args.orders,
            metadata_path=args.metadata,
            translated_info_path=args.translated_info,
            max_comp_path=args.max_comp,
        )

    import netlogo as sim

    metrics_df, summary_df = run_experiment(
        sim=sim,
        max_ticks=args.max_ticks,
        record_every=args.record_every,
        progress_every=args.progress_every,
    )

    tick_metrics_path = output_dir / "static_21day_tick_metrics.csv"
    summary_path = output_dir / "static_21day_summary.csv"
    metrics_df.to_csv(tick_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Static 21-day experiment completed.")
    print(f"Tick metrics: {tick_metrics_path}")
    print(f"Summary:      {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
