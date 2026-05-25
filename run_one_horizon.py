from __future__ import annotations

import contextlib
import importlib
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd


CSV_SEPARATOR = os.environ.get("FULL_POSTT_CSV_SEPARATOR", ";")
CSV_ENCODING = "utf-8-sig"


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
        f"Usage: python {script_name} <run_dir> <output_csv> <label> <horizon_tick>"
    )


run_dir = Path(sys.argv[1]).resolve()
output_csv = Path(sys.argv[2]).resolve()
label = sys.argv[3]
horizon_tick = float(sys.argv[4])

ensure_runtime_input_files(run_dir)
output_csv.parent.mkdir(parents=True, exist_ok=True)

os.chdir(run_dir)
if str(run_dir) not in sys.path:
    sys.path.insert(0, str(run_dir))

devnull = open(os.devnull, "w")
last_error = None
sim = None
for attempt in range(3):
    try:
        if "netlogo" in sys.modules:
            sim = importlib.reload(sys.modules["netlogo"])
        else:
            sim = importlib.import_module("netlogo")
        with contextlib.redirect_stdout(devnull):
            setup_result = sim.setup()
        if isinstance(setup_result, str) and "error" in setup_result.lower():
            raise RuntimeError(setup_result)
        last_error = None
        break
    except Exception as exc:
        last_error = exc
        time.sleep(2)
else:
    devnull.close()
    raise last_error

warehouse = sim.warehouse
start = time.time()
stopped_cleanly = False
while float(warehouse._tick) < horizon_tick:
    warehouse.tick()
    if warehouse.isSimulationComplete():
        stopped_cleanly = True
        break
elapsed = time.time() - start
if hasattr(warehouse, "refreshSimulationHealth"):
    warehouse.refreshSimulationHealth(force_log=True)
on_hold = int(
    sum(1 for o in warehouse.order_manager.unfinished_orders if getattr(o, "on_hold", False))
)
fulfilled = int(warehouse.orders_fulfilled)
arrived = int(len(warehouse.order_manager.orders))
delivered_order_lines = int(getattr(warehouse, "delivered_order_lines", 0))
pod_visits = int(warehouse.pod_visit_to_station)
energy = float(warehouse.total_energy)
fixed_energy = float(warehouse.total_fixed_load_energy)
result = pd.DataFrame(
    [
        {
            "scenario": label,
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
            "delivered_order_lines_per_pod_visit": (
                delivered_order_lines / pod_visits
            )
            if pod_visits
            else 0.0,
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
                int(float(warehouse._tick)) - int(getattr(warehouse, "health_last_progress_tick", 0)),
            ),
            "health_stalled_tick_count": int(
                getattr(warehouse, "health_stalled_tick_count", 0)
            ),
            "stop_and_go": int(warehouse.stop_and_go),
            "total_energy": energy,
            "total_fixed_load_energy": fixed_energy,
            "energy_per_fulfilled_order": (energy / fulfilled) if fulfilled else 0.0,
            "fixed_energy_per_fulfilled_order": (
                fixed_energy / fulfilled
            )
            if fulfilled
            else 0.0,
            "wall_clock_seconds": elapsed,
        }
    ]
)
result.to_csv(output_csv, index=False, sep=CSV_SEPARATOR, encoding=CSV_ENCODING)
devnull.close()
