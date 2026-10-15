"""Run every experiment (resumable). ~1 h on a single CPU core.

    python -m experiments.run_all
"""
from experiments import ablation_ansatz, circuit_depth, noise_analysis, qubit_count, run_main

if __name__ == "__main__":
    run_main.main()
    qubit_count.main()
    circuit_depth.main()
    ablation_ansatz.main()
    noise_analysis.main()
    print("ALL DONE", flush=True)
