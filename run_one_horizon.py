from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
from shared_order_stream import ensure_shared_bootstrap_order_file


CSV_SEPARATOR = os.environ.get("FULL_POSTT_CSV_SEPARATOR", ";")
CSV_ENCODING = "utf-8-sig"
PROGRESS_ENABLED = os.environ.get("FULL_POSTT_ENABLE_PROGRESS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
}
PROGRESS_TICKS = max(1.0, float(os.environ.get("FULL_POSTT_PROGRESS_TICKS", "100")))
PROGRESS_SECONDS = max(1.0, float(os.environ.get("FULL_POSTT_PROGRESS_SECONDS", "30")))
RUN_COUNT_ENV = "FULL_POSTT_RUN_COUNT"
ORDER_MODE_ENV = "FULL_POSTT_ORDER_MODE"
BOOTSTRAP_BASE_SEED_ENV = "FULL_POSTT_BOOTSTRAP_BASE_SEED"
BOOTSTRAP_ARRIVAL_MODE_ENV = "FULL_POSTT_BOOTSTRAP_ARRIVAL_MODE"
BOOTSTRAP_N_ORDERS_ENV = "FULL_POSTT_BOOTSTRAP_N_ORDERS"
BOOTSTRAP_SHARED_ORDER_PATH_ENV = "FULL_POSTT_SHARED_BOOTSTRAP_ORDER_PATH"
POD_LOCATION_MODE_ENV = "FULL_POSTT_POD_LOCATION_MODE"
POD_LOCATION_BASE_SEED_ENV = "FULL_POSTT_POD_LOCATION_BASE_SEED"
RUNTIME_POD_LOCATION_POLICY_ENV = "RMFS_RUNTIME_POD_LOCATION_POLICY"
RUNTIME_POD_LOCATION_SEED_ENV = "RMFS_RUNTIME_POD_LOCATION_SEED"


def ensure_runtime_input_files(run_root: Path) -> None:
    data_dir = run_root / "data"
    input_dir = data_dir / "input"
    output_dir = data_dir / "output"
    legacy_input_dir = data_dir / "yohana" / "input"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legacy_input_dir.exists():
        return

    for legacy_file in legacy_input_dir.iterdir():
        if not legacy_file.is_file():
            continue

        target = input_dir / legacy_file.name
        if not target.exists():
            shutil.copy2(legacy_file, target)

    # Some clones only contain the older Yohana items_dictionary schema, which
    # lacks the newer `max_fit` column expected by the shared RMFS generator.
    # When that happens, rebuild the live input dictionary from the prepared
    # `data/output/items.csv` artifact instead.
    input_items_dictionary = input_dir / "items_dictionary.csv"
    output_items = output_dir / "items.csv"

    rebuild_items_dictionary = not input_items_dictionary.exists()
    if input_items_dictionary.exists():
        try:
            existing_columns = pd.read_csv(input_items_dictionary, nrows=0).columns
            rebuild_items_dictionary = "max_fit" not in existing_columns
        except Exception:
            rebuild_items_dictionary = True

    if rebuild_items_dictionary and output_items.exists():
        items_df = pd.read_csv(output_items)

        # Older writer variants sometimes persisted `item_id` as an unnamed index.
        unnamed_columns = [col for col in items_df.columns if str(col).startswith("Unnamed:")]
        if unnamed_columns:
            items_df = items_df.drop(columns=unnamed_columns, errors="ignore")

        items_df.to_csv(input_items_dictionary, index=False)


if len(sys.argv) < 5:
    script_name = Path(sys.argv[0]).name if sys.argv else "run_one_horizon.py"
    raise SystemExit(
        f"Usage: python {script_name} <run_dir> <output_csv> <label> <horizon_tick> [run_count]"
    )


run_dir = Path(sys.argv[1]).resolve()
output_csv = Path(sys.argv[2]).resolve()
label = sys.argv[3]
horizon_tick = float(sys.argv[4])
run_count = int(sys.argv[5]) if len(sys.argv) >= 6 else int(os.environ.get(RUN_COUNT_ENV, "1"))
order_mode = os.environ.get(ORDER_MODE_ENV, "fixed_actual").strip().lower()
bootstrap_base_seed = int(os.environ.get(BOOTSTRAP_BASE_SEED_ENV, "42"))
bootstrap_arrival_mode = os.environ.get(
    BOOTSTRAP_ARRIVAL_MODE_ENV,
    "empirical_interarrival",
).strip()
bootstrap_n_orders_raw = os.environ.get(BOOTSTRAP_N_ORDERS_ENV)
bootstrap_n_orders = (
    int(bootstrap_n_orders_raw)
    if bootstrap_n_orders_raw is not None and bootstrap_n_orders_raw.strip() != ""
    else None
)
pod_location_mode = os.environ.get(POD_LOCATION_MODE_ENV, "identity").strip().lower()
pod_location_base_seed = int(os.environ.get(POD_LOCATION_BASE_SEED_ENV, "42"))

if run_count <= 0:
    raise SystemExit("run_count must be a positive integer.")
if order_mode not in {"fixed_actual", "bootstrap_actual"}:
    raise SystemExit("FULL_POSTT_ORDER_MODE must be either 'fixed_actual' or 'bootstrap_actual'.")
if pod_location_mode not in {"identity", "shuffle"}:
    raise SystemExit("FULL_POSTT_POD_LOCATION_MODE must be either 'identity' or 'shuffle'.")

ensure_runtime_input_files(run_dir)
output_csv.parent.mkdir(parents=True, exist_ok=True)

os.chdir(run_dir)
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

devnull = open(os.devnull, "w")


def prepare_order_stream(replication_index: int) -> tuple[str, int | None, Path | None]:
    os.environ[ORDER_MODE_ENV] = order_mode
    if order_mode != "bootstrap_actual":
        os.environ.pop(BOOTSTRAP_SHARED_ORDER_PATH_ENV, None)
        return order_mode, None, None

    current_seed = bootstrap_base_seed + (replication_index - 1)
    shared_order_path = ensure_shared_bootstrap_order_file(
        run_root=run_dir,
        seed=current_seed,
        n_orders=bootstrap_n_orders,
        arrival_mode=bootstrap_arrival_mode,
    )
    os.environ[BOOTSTRAP_SHARED_ORDER_PATH_ENV] = str(shared_order_path)
    return order_mode, current_seed, shared_order_path


def prepare_pod_location_stream(replication_index: int) -> tuple[str, int | None]:
    if pod_location_mode != "shuffle":
        os.environ.pop(RUNTIME_POD_LOCATION_POLICY_ENV, None)
        os.environ.pop(RUNTIME_POD_LOCATION_SEED_ENV, None)
        return pod_location_mode, None

    current_seed = pod_location_base_seed + (replication_index - 1)
    os.environ[RUNTIME_POD_LOCATION_POLICY_ENV] = "shuffle"
    os.environ[RUNTIME_POD_LOCATION_SEED_ENV] = str(current_seed)
    return pod_location_mode, current_seed


def load_simulation_module():
    last_error = None
    for attempt in range(3):
        try:
            if "netlogo" in sys.modules:
                sim_module = importlib.reload(sys.modules["netlogo"])
            else:
                sim_module = importlib.import_module("netlogo")
            with contextlib.redirect_stdout(devnull):
                setup_result = sim_module.setup()
            if isinstance(setup_result, str) and "error" in setup_result.lower():
                raise RuntimeError(setup_result)
            return sim_module
        except Exception as exc:
            last_error = exc
            time.sleep(2)
    raise last_error


def run_single_replication(replication_index: int) -> pd.DataFrame:
    active_order_mode, active_seed, shared_order_path = prepare_order_stream(replication_index)
    active_pod_location_mode, active_pod_location_seed = prepare_pod_location_stream(
        replication_index
    )
    sim = load_simulation_module()
    warehouse = sim.warehouse
    start = time.time()
    stopped_cleanly = False
    last_progress_tick = float(warehouse._tick)
    last_progress_time = start

    print(
        "[START] "
        f"scenario={label} "
        f"run={replication_index}/{run_count} "
        f"order_mode={active_order_mode} "
        f"seed={active_seed if active_seed is not None else 'n/a'} "
        f"pod_location_mode={active_pod_location_mode} "
        f"pod_location_seed={active_pod_location_seed if active_pod_location_seed is not None else 'n/a'} "
        f"horizon_tick={horizon_tick:g} "
        f"run_dir={run_dir}",
        flush=True,
    )
    if shared_order_path is not None:
        print(
            f"[ORDER] scenario={label} run={replication_index}/{run_count} shared_order={shared_order_path}",
            flush=True,
        )
    while float(warehouse._tick) < horizon_tick:
        warehouse.tick()
        current_tick = float(warehouse._tick)
        now = time.time()
        if PROGRESS_ENABLED and (
            (current_tick - last_progress_tick) >= PROGRESS_TICKS
            or (now - last_progress_time) >= PROGRESS_SECONDS
        ):
            elapsed_now = now - start
            print(
                "[PROGRESS] "
                f"scenario={label} "
                f"run={replication_index}/{run_count} "
                f"order_mode={active_order_mode} "
                f"tick={current_tick:.2f}/{horizon_tick:.2f} "
                f"step={int(warehouse._step)} "
                f"elapsed_s={elapsed_now:.1f}",
                flush=True,
            )
            last_progress_tick = current_tick
            last_progress_time = now
        if warehouse.isSimulationComplete():
            stopped_cleanly = True
            break
    elapsed = time.time() - start
    if (
        hasattr(warehouse, "refreshSimulationHealth")
        and int(getattr(warehouse, "health_check_interval", 0)) > 0
    ):
        warehouse.refreshSimulationHealth(force_log=True)
    on_hold = int(
        sum(
            1
            for o in warehouse.order_manager.unfinished_orders
            if getattr(o, "on_hold", False)
        )
    )
    fulfilled = int(warehouse.orders_fulfilled)
    arrived = int(len(warehouse.order_manager.orders))
    delivered_order_lines = int(getattr(warehouse, "delivered_order_lines", 0))
    picked_units = int(getattr(warehouse, "total_picked_units", 0))
    pod_visits = int(warehouse.pod_visit_to_station)
    energy = float(warehouse.total_energy)
    fixed_energy = float(warehouse.total_fixed_load_energy)
    variable_energy = max(0.0, energy - fixed_energy)

    result = pd.DataFrame(
        [
            {
                "scenario": label,
                "replication": replication_index,
                "replications_total": run_count,
                "order_mode": active_order_mode,
                "bootstrap_seed": active_seed,
                "pod_location_mode": active_pod_location_mode,
                "pod_location_seed": active_pod_location_seed,
                "ticks_elapsed": float(warehouse._tick),
                "steps_elapsed": int(warehouse._step),
                "stopped_cleanly_before_horizon": int(stopped_cleanly),
                "arrived_orders_by_horizon": arrived,
                "fulfilled_orders": fulfilled,
                "fulfilled_over_arrived": (fulfilled / arrived) if arrived else 0.0,
                "throughput_orders_per_hour": (
                    fulfilled / (float(warehouse._tick) / 60.0)
                )
                if float(warehouse._tick) > 0
                else 0.0,
                "on_hold_orders": on_hold,
                "unfinished_orders": int(len(warehouse.order_manager.unfinished_orders)),
                "job_queue_length": int(len(warehouse.job_queue)),
                "sku_queue_length": int(len(warehouse.sku_picking_queue)),
                "pod_visits": pod_visits,
                "delivered_order_lines": delivered_order_lines,
                "picked_units": picked_units,
                "delivered_order_lines_per_pod_visit": (
                    delivered_order_lines / pod_visits
                )
                if pod_visits
                else 0.0,
                "picked_units_per_pod_visit": (picked_units / pod_visits) if pod_visits else 0.0,
                "replenishment_count": int(warehouse.replenishment_count),
                "replenishment_trips": int(warehouse.replenishment_trips),
                "health_status_final": getattr(warehouse, "health_status", "unknown"),
                "health_consistency_violations": int(
                    getattr(warehouse, "health_consistency_violations", 0)
                ),
                "health_zombie_orders": int(
                    getattr(warehouse, "health_zombie_order_count", 0)
                ),
                "health_pending_replenishment_count": int(
                    getattr(warehouse, "health_pending_replenishment_count", 0)
                ),
                "health_aged_pending_replenishment_count": int(
                    getattr(warehouse, "health_aged_pending_replenishment_count", 0)
                ),
                "health_oldest_pending_replenishment_age": int(
                    getattr(warehouse, "health_oldest_pending_replenishment_age", 0)
                ),
                "health_last_progress_tick": int(
                    getattr(warehouse, "health_last_progress_tick", 0)
                ),
                "health_progress_gap": max(
                    0,
                    int(float(warehouse._tick))
                    - int(getattr(warehouse, "health_last_progress_tick", 0)),
                ),
                "health_stalled_tick_count": int(
                    getattr(warehouse, "health_stalled_tick_count", 0)
                ),
                "stop_and_go": int(warehouse.stop_and_go),
                "total_energy": energy,
                "total_fixed_load_energy": fixed_energy,
                "variable_energy": variable_energy,
                "energy_per_fulfilled_order": (energy / fulfilled) if fulfilled else 0.0,
                "fixed_energy_per_fulfilled_order": (fixed_energy / fulfilled) if fulfilled else 0.0,
                "variable_energy_per_delivered_line": (
                    variable_energy / delivered_order_lines
                )
                if delivered_order_lines
                else 0.0,
                "energy_per_pod_visit": (energy / pod_visits) if pod_visits else 0.0,
                "variable_energy_per_pod_visit": (
                    variable_energy / pod_visits
                )
                if pod_visits
                else 0.0,
                "wall_clock_seconds": elapsed,
            }
        ]
    )
    if hasattr(warehouse, "getReplenishmentDebugSummary"):
        debug_summary = warehouse.getReplenishmentDebugSummary()
        if debug_summary:
            for key, value in debug_summary.items():
                result.loc[0, key] = value
            print(
                "[REPL_DEBUG] "
                f"scenario={label} "
                f"run={replication_index}/{run_count} "
                f"watchlist_refreshes={debug_summary.get('repldbg_watchlist_refreshes', 0)} "
                f"critical_peak={debug_summary.get('repldbg_critical_sku_peak', 0)} "
                f"enqueue_attempts={debug_summary.get('repldbg_enqueue_attempts', 0)} "
                f"pending_exists={debug_summary.get('repldbg_pending_request_exists_count', 0)} "
                f"no_eligible={debug_summary.get('repldbg_no_eligible_pod_count', 0)} "
                f"qj_gate={debug_summary.get('repldbg_qj_gate_block_count', 0)} "
                f"blocked_no_station={debug_summary.get('repldbg_dispatch_blocked_no_station', 0)} "
                f"blocked_no_robot={debug_summary.get('repldbg_dispatch_blocked_no_robot', 0)} "
                f"blocked_pod_not_idle={debug_summary.get('repldbg_dispatch_blocked_pod_not_idle', 0)} "
                f"max_pending_age={debug_summary.get('repldbg_max_pending_request_age', 0)}",
                flush=True,
            )
            print(
                "[REPL_DEBUG_TOP] "
                f"scenario={label} "
                f"run={replication_index}/{run_count} "
                f"top_no_eligible_skus={debug_summary.get('repldbg_top_no_eligible_skus', '')} "
                f"top_qj_gate_pods={debug_summary.get('repldbg_top_qj_gate_pods', '')} "
                f"top_busy_pods={debug_summary.get('repldbg_top_pod_not_idle_pods', '')}",
                flush=True,
            )
    print(
        "[DONE] "
        f"scenario={label} "
        f"run={replication_index}/{run_count} "
        f"order_mode={active_order_mode} "
        f"seed={active_seed if active_seed is not None else 'n/a'} "
        f"pod_location_mode={active_pod_location_mode} "
        f"pod_location_seed={active_pod_location_seed if active_pod_location_seed is not None else 'n/a'} "
        f"tick={float(warehouse._tick):.2f} "
        f"step={int(warehouse._step)} "
        f"elapsed_s={elapsed:.1f} "
        f"fulfilled={fulfilled} "
        f"arrived={arrived} "
        f"output={output_csv}",
        flush=True,
    )
    return result


try:
    for replication_index in range(1, run_count + 1):
        result = run_single_replication(replication_index)
        write_header = not output_csv.exists() or output_csv.stat().st_size == 0
        result.to_csv(
            output_csv,
            index=False,
            sep=CSV_SEPARATOR,
            encoding=CSV_ENCODING if write_header else "utf-8",
            mode="a",
            header=write_header,
        )
finally:
    devnull.close()
