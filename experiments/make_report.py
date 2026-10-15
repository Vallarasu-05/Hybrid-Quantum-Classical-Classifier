"""Aggregate results/runs.jsonl -> results/metrics.csv, results/summary.md, figures.

    python -m experiments.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.quantum_circuit import build_circuit

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "results" / "figures"
EASY, HARD = "setosa_vs_versicolor", "versicolor_vs_virginica"
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc", "train_time_s"]
NICE = {"accuracy": "Test accuracy", "precision": "Precision", "recall": "Recall",
        "f1": "F1-score", "roc_auc": "ROC-AUC", "train_time_s": "Training time (s)"}
C = {"classical": "#4C78A8", "hybrid": "#E45756", "logreg": "#9D9D9D"}


def load() -> pd.DataFrame:
    rows = [json.loads(l) for l in (RES / "runs.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["shots"] = df["shots"].fillna(0).astype(int)   # 0 == ideal / exact
    df["label"] = df.apply(_label, axis=1)
    return df


def _label(r) -> str:
    if r.model == "logreg":
        return "Logistic regression"
    if r.model == "classical":
        return f"Classical NN ({r.n_qubits}f)"
    s = f"Hybrid {r.n_qubits}q L={r.n_layers}"
    if not r.final_rotation:
        s += " (literal ansatz)"
    if r.shots:
        s += f" train-shots={r.shots}"
    return s


def pm(x: pd.Series, d=3) -> str:
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["task", "label"], sort=False)
    out = g.agg(n_seeds=("seed", "nunique"), n_params=("n_params", "first"),
                **{m: (m, lambda s, m=m: pm(s, 1 if m == "train_time_s" else 3)) for m in METRICS})
    return out.reset_index()


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def select(df, **kw):
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= df[k] == v
    return df[m]


# ----------------------------- figures -----------------------------
def fig_circuit():
    qc, _, _ = build_circuit(4, 1, True)
    qc.draw("mpl", filename=str(FIG / "circuit_4q_1layer.png"), fold=-1)
    qc2, _, _ = build_circuit(4, 2, True)
    qc2.draw("mpl", filename=str(FIG / "circuit_4q_2layer.png"), fold=-1)
    plt.close("all")


def fig_learning_curves(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, task in zip(axes, (EASY, HARD)):
        for model in ("classical", "hybrid"):
            sub = select(df, task=task, model=model, n_qubits=4, n_layers=1, final_rotation=True, shots=0)
            if sub.empty:
                continue
            L = np.array(sub.train_loss_hist.tolist())
            mu, sd, x = L.mean(0), L.std(0), np.arange(1, L.shape[1] + 1)
            ax.plot(x, mu, color=C[model], label=f"{model} (n={len(sub)})")
            ax.fill_between(x, mu - sd, mu + sd, color=C[model], alpha=0.2)
        ax.set(title=task.replace("_", " "), xlabel="epoch", ylabel="train BCE loss")
        ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "learning_curves.png", dpi=150); plt.close(fig)


def fig_confusion(df):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, model in zip(axes, ("classical", "hybrid")):
        sub = select(df, task=HARD, model=model, n_qubits=4, n_layers=1, final_rotation=True, shots=0)
        if sub.empty:
            continue
        cm = np.sum([np.array(c) for c in sub.confusion_matrix], axis=0)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center", fontsize=14,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set(title=f"{model} (summed over {len(sub)} seeds)", xticks=[0, 1], yticks=[0, 1],
               xlabel="predicted", ylabel="true")
    fig.suptitle("Confusion matrices, versicolor vs virginica (test sets pooled)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png", dpi=150); plt.close(fig)


def _bars(ax, labels, groups, ylabel, ylim=None):
    x = np.arange(len(labels))
    means = [np.mean(g) for g in groups]
    errs = [np.std(g, ddof=1) if len(g) > 1 else 0 for g in groups]
    ax.bar(x, means, yerr=errs, capsize=4, color="#72B7B2")
    for xi, g in zip(x, groups):
        ax.scatter([xi] * len(g), g, color="k", s=12, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def fig_depth_and_ablation(df):
    seeds = [0, 1, 2]  # common seeds for a paired comparison
    specs = [("Hybrid 4q L=1", dict(n_layers=1, final_rotation=True)),
             ("Hybrid 4q L=2", dict(n_layers=2, final_rotation=True)),
             ("literal ansatz L=1", dict(n_layers=1, final_rotation=False)),
             ("literal ansatz L=2", dict(n_layers=2, final_rotation=False))]
    labels, accs, times = [], [], []
    for name, kw in specs:
        sub = select(df, task=HARD, model="hybrid", n_qubits=4, shots=0, **kw)
        sub = sub[sub.seed.isin(seeds)].sort_values("seed")
        if len(sub) == len(seeds):
            labels.append(name); accs.append(sub.accuracy.tolist()); times.append(sub.train_time_s.tolist())
    if not labels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _bars(axes[0], labels, accs, "test accuracy (seeds 0-2)", (0.5, 1.05))
    _bars(axes[1], labels, times, "training time (s)")
    fig.suptitle("Circuit depth and ansatz ablation (versicolor vs virginica)")
    fig.tight_layout(); fig.savefig(FIG / "depth_and_ablation.png", dpi=150); plt.close(fig)


def fig_noise(df):
    h = select(df, task=HARD, model="hybrid", n_qubits=4, n_layers=1, final_rotation=True)
    ideal = h[h.shots == 0]
    labels, groups = [], []
    if len(ideal):
        labels.append("ideal"); groups.append(ideal.accuracy.tolist())
        for k in ["shots_1024", "shots_100", "depol_low", "depol_high"]:
            labels.append(f"infer: {k}")
            groups.append([d[k] for d in ideal.inference_noise_acc])
    for s in (1024, 100):
        sub = h[h.shots == s]
        if len(sub):
            labels.append(f"train+eval: {s} shots"); groups.append(sub.accuracy.tolist())
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    _bars(ax, labels, groups, "test accuracy", (0.5, 1.05))
    ax.set_title("Hybrid model under finite shots / depolarizing noise (versicolor vs virginica)")
    fig.tight_layout(); fig.savefig(FIG / "noise_analysis.png", dpi=150); plt.close(fig)


def fig_qubits(df):
    specs = [("Classical 2f", dict(model="classical", n_qubits=2)),
             ("Hybrid 2q", dict(model="hybrid", n_qubits=2)),
             ("Classical 4f", dict(model="classical", n_qubits=4)),
             ("Hybrid 4q", dict(model="hybrid", n_qubits=4, n_layers=1, final_rotation=True, shots=0))]
    labels, accs, times = [], [], []
    for name, kw in specs:
        sub = select(df, task=HARD, **kw)
        sub = sub[sub.seed < 5]
        if len(sub):
            labels.append(f"{name}\n(n={len(sub)})"); accs.append(sub.accuracy.tolist()); times.append(sub.train_time_s.tolist())
    if not labels:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    _bars(axes[0], labels, accs, "test accuracy", (0.5, 1.05))
    _bars(axes[1], labels, times, "training time (s)")
    fig.suptitle("Number of qubits / features (versicolor vs virginica)")
    fig.tight_layout(); fig.savefig(FIG / "qubit_count.png", dpi=150); plt.close(fig)


# ----------------------------- summary -----------------------------
def paired(df, task, a, b) -> str:
    """Mean paired accuracy difference (b - a) over seeds common to both."""
    A = a.set_index("seed").accuracy; B = b.set_index("seed").accuracy
    common = sorted(set(A.index) & set(B.index))
    if not common:
        return "n/a"
    d = (B.loc[common] - A.loc[common])
    return f"{d.mean():+.3f} (n={len(common)} paired seeds, wins/ties/losses = " \
           f"{(d>0).sum()}/{(d==0).sum()}/{(d<0).sum()})"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    summ = summarize(df)
    summ.to_csv(RES / "metrics.csv", index=False)
    df.drop(columns=["train_loss_hist", "test_loss_hist", "test_acc_hist"]).to_csv(RES / "all_runs.csv", index=False)

    parts = []
    for task in (EASY, HARD):
        parts.append(f"### {task.replace('_', ' ')}\n\n" + md_table(summ[summ.task == task].drop(columns="task")))
    ideal = df[(df.shots == 0) & (df.model == "hybrid") & (df.n_qubits == 4) & (df.final_rotation)]
    inf = ideal[ideal.task == HARD]
    if len(inf):
        rows = []
        for L, g in inf.groupby("n_layers"):
            row = {"model": f"Hybrid 4q L={L}", "n_seeds": len(g), "ideal": pm(g.accuracy)}
            for k in ["shots_1024", "shots_100", "depol_low", "depol_high"]:
                row[k] = pm(pd.Series([d[k] for d in g.inference_noise_acc]))
            rows.append(row)
        parts.append("### inference-time noise (trained ideally, evaluated on Aer; test accuracy)\n\n"
                     + md_table(pd.DataFrame(rows)))

    pair = []
    for task in (EASY, HARD):
        cl = select(df, task=task, model="classical", n_qubits=4)
        hy = select(df, task=task, model="hybrid", n_qubits=4, n_layers=1, final_rotation=True, shots=0)
        pair.append(f"- {task}: hybrid - classical test accuracy = {paired(df, task, cl, hy)}")
    l1 = select(df, task=HARD, model="hybrid", n_qubits=4, n_layers=1, final_rotation=True, shots=0)
    l2 = select(df, task=HARD, model="hybrid", n_qubits=4, n_layers=2, final_rotation=True, shots=0)
    if len(l2):
        pair.append(f"- depth: L=2 - L=1 accuracy = {paired(df, HARD, l1, l2)}")
    parts.append("### paired differences\n\n" + "\n".join(pair))
    (RES / "summary.md").write_text("\n\n".join(parts) + "\n")

    fig_circuit(); fig_learning_curves(df); fig_confusion(df)
    fig_depth_and_ablation(df); fig_noise(df); fig_qubits(df)
    print("\n\n".join(parts))


if __name__ == "__main__":
    main()
