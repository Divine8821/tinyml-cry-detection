import os
import pandas as pd
from src import NoiseGenerator
from tests.test_baseline_wander import run_bw_test
from tests.test_muscle_artifact import run_ma_test
from tests.test_electrode_motion import run_em_test
from tests.test_official_nstdb import run_official_nstdb_test

def setup_project():
    """Ensure the results directory exists before running tests."""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created /results directory.")

def run_full_analysis():
    """
    Executes the entire benchmarking suite for the research paper.
    """
    print("="*60)
    print("STARTING ECG ALGORITHM BENCHMARK SUITE")
    print("="*60)

    # 1. Baseline Wander Stress Test
    print("\n[1/4] Running Baseline Wander (BW) Analysis...")
    run_bw_test()

    # 2. Muscle Artifact Stress Test
    print("\n[2/4] Running Muscle Artifact (MA) Analysis...")
    run_ma_test()

    # 3. Electrode Motion Stress Test
    print("\n[3/4] Running Electrode Motion (EM) Analysis...")
    run_em_test()

    # 4. Official NSTDB Benchmark
    print("\n[4/4] Running Official NSTDB (118/119) Benchmark...")
    run_official_nstdb_test()

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE: All results saved to /results folder.")
    print("="*60)

if __name__ == "__main__":
    setup_project()
    run_full_analysis()