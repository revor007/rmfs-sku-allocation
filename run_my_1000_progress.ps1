$env:FULL_POSTT_ENABLE_PROGRESS = '1'
$env:FULL_POSTT_PROGRESS_TICKS = '100'
$env:FULL_POSTT_PROGRESS_SECONDS = '20'
Set-Location 'd:\ITB\Tugas Akhir\_full_postt_parallel_runs'
python .\run_one_horizon.py 'd:\ITB\Tugas Akhir\_full_postt_parallel_runs\four_scenario_1000_shared_latest\my_scenario\netlogo-rmfs' 'd:\ITB\Tugas Akhir\_full_postt_parallel_runs\four_scenario_1000_shared_latest\my_scenario\result_simplified_reset_1000_no_runtime_logs.csv' 'My scenario' 1000
