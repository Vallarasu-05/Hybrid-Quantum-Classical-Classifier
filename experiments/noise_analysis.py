"""Experiment 4 - finite-shot / noisy simulation.

(a) TRAIN-time shot noise: train with ~100 or ~1024 shots per expectation value
    (forward pass and parameter-shift gradients), evaluate with the same back-end.
(b) INFERENCE-time noise: models trained ideally (Exp. 1) are re-evaluated on a
    Qiskit-Aer simulator with finite shots and depolarizing noise. These numbers
    are produced automatically inside every ideal hybrid run (see _runner.py).
"""
from experiments._runner import run_grid

HARD = "versicolor_vs_virginica"


def main():
    run_grid([dict(task=HARD, model="hybrid", shots=100),
              dict(task=HARD, model="hybrid", shots=1024)], range(3))


if __name__ == "__main__":
    main()
