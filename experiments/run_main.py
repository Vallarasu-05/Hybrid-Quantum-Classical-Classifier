"""Experiment 1 - classical NN vs hybrid quantum-classical NN (+ logistic-regression reference)."""
from experiments._runner import run_grid

EASY, HARD = "setosa_vs_versicolor", "versicolor_vs_virginica"


def main():
    # cheap models first: 5 seeds on both tasks
    for task in (EASY, HARD):
        run_grid([dict(task=task, model="logreg"), dict(task=task, model="classical")], range(5))
    # hybrid model: 3 seeds on the (saturated) easy task, 5 on the hard task
    run_grid([dict(task=HARD, model="hybrid")], range(5))
    run_grid([dict(task=EASY, model="hybrid")], range(3))


if __name__ == "__main__":
    main()
