# RMFS Change Log vs Cindy Scenario 3 Base

This note documents the changes made in `revision-fcgma-copy/netlogo-rmfs`
relative to the original folder:

`netlogoCindy/netlogo-rmfs-Skenario-3-Cindy-revisi/netlogo-rmfs-Skenario-3-Cindy-revisi`

The comparison focuses on workflow and code behavior, not generated outputs.

## 1. New top-level scripts

These files do not exist in the original Cindy Scenario 3 folder:

- `prepare_static_21day_inputs.py`
- `run_static_21day_experiment.py`
- `compare_allocation_samples.py`

### `prepare_static_21day_inputs.py`

Purpose:
- convert a frozen FCGMA allocation into live RMFS `items.csv` and `pods.csv`
- export filtered post-cutoff replay orders
- regenerate RMFS physical pod layout to match optimized pod count
- back up previous live RMFS inputs before replacement

Main behavior:
- auto-detects the current FCGMA and preprocessing directories
- loads the cutoff experiment context from the optimizer side
- uses only eligible post-cutoff SKUs
- writes:
  - `data/output/items_cutoff_experiment.csv`
  - `data/output/pods_cutoff_experiment.csv`
  - `data/input/cutoff_test_orders.csv`
  - `data/output/item_code_to_id_cutoff_experiment.csv`
  - `data/output/cutoff_experiment_input_summary.csv`

### `run_static_21day_experiment.py`

Purpose:
- run the RMFS simulation on the cutoff-based replay orders
- record tick-level metrics periodically
- export a summary table for the run

Main behavior:
- uses the live RMFS inputs prepared by `prepare_static_21day_inputs.py`
- supports:
  - `--record-every`
  - `--progress-every`
  - `--max-ticks`
- prints heartbeat/progress lines during execution

### `compare_allocation_samples.py`

Purpose:
- compare two allocation inputs under the same RMFS train/test configuration

Main behavior:
- builds a baseline allocation from Cindy's old `items.csv` + `pods.csv`
- samples a shared set of post-cutoff orders
- runs the RMFS simulation twice:
  - user allocation
  - baseline allocation
- exports side-by-side performance metrics

## 2. Order generation changed from synthetic/default mode to cutoff replay

Modified file:
- `lib/generator/order_generator.py`

Change:
- the current experiment uses `data/input/cutoff_test_orders.csv` as the replay source
- actual timestamps are converted into RMFS arrival ticks using
  `ACTUAL_ORDER_TIME_SCALE_SECONDS = 60`

Effect:
- RMFS now replays real post-cutoff orders from the cutoff experiment
  instead of relying only on the original synthetic order generation workflow

## 3. RMFS now follows optimized pod count

Modified files:
- `world/layout.py`
- `prepare_static_21day_inputs.py`
- `world/warehouse.py`

Change:
- the original Cindy version assumes a fixed warehouse layout / pod geometry
- the current version regenerates `generated_pod.csv` to match the pod count
  required by the current allocation

Effect:
- RMFS can simulate allocations that use a different number of pods than the
  original Scenario 3 warehouse

## 4. Input alignment and eligible SKU filtering were added

Modified / added files:
- `prepare_static_21day_inputs.py`
- `compare_allocation_samples.py`

Change:
- current workflow enforces one eligible SKU universe for the cutoff experiment
- post-cutoff replay orders are filtered to that eligible set
- coverage checks are applied against the eligible master SKU list, not the raw
  order file

Effect:
- RMFS evaluation is now consistent with the optimizer's cutoff-based SKU universe

## 5. Live RMFS input conversion changed

Modified file:
- `prepare_static_21day_inputs.py`

Change:
- `max_qty` in RMFS is now derived from the converted allocation slot quantity
- live `items.csv` and `pods.csv` are rebuilt from the current allocation

Effect:
- RMFS stock initialization is driven by the current experiment policy rather
  than the old static Scenario 3 warehouse files

## 6. Path auto-detection was added

Added/modified file:
- `prepare_static_21day_inputs.py`

Change:
- removed reliance on one hard-coded folder layout
- scripts search for the active FCGMA and preprocessing directories

Effect:
- the experiment pipeline works even if the project folder names move

## 7. Warehouse stop condition changed

Modified files:
- `world/warehouse.py`
- `netlogo.py`

Change:
- current RMFS tracks:
  - total expected orders
  - last order arrival tick
- completion logic now checks:
  - all arrivals have occurred
  - all expected orders are fulfilled
  - no queued work remains
  - robots are idle

Effect:
- the simulation stops based on replay completion logic rather than only the
  original scenario timing assumptions

## 8. Robot graph/pathing bug fixed

Modified file:
- `lib/generator/warehouse_generator.py`

Change:
- station-lane nodes and approach cells are added to the normal travel graph
  as well as the pod graph

Reason:
- the earlier comparison runs failed with
  `networkx.exception.NodeNotFound: Source 2,13 not in G`

Effect:
- robots can route correctly from station-lane starting positions

## 9. Pod inventory loading and picking behavior fixed

Modified files:
- `world/entities/pod.py`
- `world/warehouse.py`
- `world/managers/pod_manager.py`

Changes:
- repeated pod-SKU rows now accumulate instead of overwrite
- `pickSKU` returns the actual picked quantity
- partial picks now requeue the remaining quantity instead of forcing
  impossible fulfillment
- duplicate pod registration for one SKU is prevented
- duplicate pod candidates are filtered during SKU lookup

Effect:
- RMFS no longer corrupts stock state when one SKU occupies multiple rows
  in the same pod
- queued orders do not stall due to silent overdraw / negative pod stock

## 10. Comparison baseline was standardized to the same compartment-capacity model

Modified file:
- `compare_allocation_samples.py`

Change:
- Cindy's old `pods.csv` is converted into a simplified allocation format
- each occupied slot keeps Cindy's placement footprint
- but the slot quantity is replaced with the shared
  `max_comp_number.csv` capacity for that SKU

Effect:
- baseline comparison now uses the same per-compartment capacity model as the
  new allocation experiment

## 11. Progress / sample metric reporting was added

Modified / added files:
- `run_static_21day_experiment.py`
- `compare_allocation_samples.py`

Change:
- periodic progress lines are printed during long runs
- summary CSVs and tick-metric CSVs are exported for sample comparisons

Effect:
- easier debugging and more transparent train/test comparisons

## 12. Output artifacts added by the new workflow

These are generated by the new pipeline and were not part of the original
Cindy Scenario 3 folder structure:

- `data/input/cutoff_test_orders.csv`
- `data/input/cutoff_test_orders_sample.csv`
- `data/input/scenario3_baseline_allocation.csv`
- `data/output/items_cutoff_experiment.csv`
- `data/output/pods_cutoff_experiment.csv`
- `data/output/item_code_to_id_cutoff_experiment.csv`
- `data/output/cutoff_experiment_input_summary.csv`
- `data/output/*sample_summary.csv`
- `data/output/*sample_tick_metrics.csv`
- `data/output/sample_allocation_comparison.csv`
- `data/output/backups/static_21day_*/`

## 13. What did not change conceptually

Some Cindy Scenario 3 ideas are still preserved:

- mixed-class pod placement is still the original baseline concept
- RMFS still uses the same general robot / station / pod simulation engine
- replenishment logic remains operational RMFS logic, separate from FCGMA stock design

