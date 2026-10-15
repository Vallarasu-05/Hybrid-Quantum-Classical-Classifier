"""Shared experiment runner.

Every run (one config x one seed) is appended as a JSON line to
``results/runs.jsonl``. Runs already present are skipped, so the experiment
scripts are resumable and can share identical configs without retraining.

``seed`` controls EVERYTHING random: the stratified train/test split, model
initialisation, minibatch order, and shot noise.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import load_binary_iris
from src.evaluate import evaluate, metrics_from_logits
from src.hybrid_model import ClassicalNN, HybridQNN, count_params
from src.quantum_circuit import build_qnn, shots_to_precision
from src.train import set_seed, train_model

ROOT = Path(__file__).resolve().parents[1]
RUNS_PATH = ROOT / "results" / "runs.jsonl"
MODEL_DIR = ROOT / "results" / "models"

DEFAULTS = dict(
    task="setosa_vs_versicolor", model="hybrid", n_qubits=4, n_layers=1,
    final_rotation=True, shots=None, features=None, hidden=4,
    epochs=20, lr=0.05, batch_size=16,
)
KEY_FIELDS = ["task", "model", "n_qubits", "n_layers", "final_rotation", "shots",
              "features", "hidden", "epochs", "lr", "batch_size", "seed"]

# Inference-time back-ends for the noise study (trained ideally, evaluated noisily).
INFERENCE_BACKENDS = {
    "shots_100": dict(shots=100, p1=0.0, p2=0.0),
    "shots_1024": dict(shots=1024, p1=0.0, p2=0.0),
    "depol_low": dict(shots=1024, p1=0.001, p2=0.01),
    "depol_high": dict(shots=1024, p1=0.005, p2=0.05),
}
INFERENCE_REPEATS = 5


def make_cfg(**kw) -> dict:
    cfg = {**DEFAULTS, **kw}
    cfg["features"] = list(cfg["features"]) if cfg["features"] is not None else None
    return cfg


def run_key(cfg: dict, seed: int) -> str:
    return json.dumps({k: (cfg | {"seed": seed})[k] for k in KEY_FIELDS}, sort_keys=True)


def _load_done() -> set[str]:
    if not RUNS_PATH.exists():
        return set()
    return {json.loads(l)["key"] for l in RUNS_PATH.read_text().splitlines() if l.strip()}


def _noisy_qnn(cfg: dict, spec: dict, seed: int):
    """Qiskit-Aer back-end: real finite-shot sampling (+ optional depolarizing noise)."""
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit_aer.primitives import EstimatorV2 as AerEstimator

    prec = shots_to_precision(spec["shots"])
    opts = {"seed_simulator": seed}
    if spec["p1"] > 0 or spec["p2"] > 0:
        nm = NoiseModel()
        nm.add_all_qubit_quantum_error(depolarizing_error(spec["p1"], 1), ["ry", "rz"])
        nm.add_all_qubit_quantum_error(depolarizing_error(spec["p2"], 2), ["cx"])
        opts["noise_model"] = nm
    est = AerEstimator(options={"backend_options": opts, "default_precision": prec})
    return build_qnn(cfg["n_qubits"], cfg["n_layers"], cfg["final_rotation"],
                     precision=prec, estimator=est)


def _inference_noise_eval(model: HybridQNN, cfg: dict, data, seed: int) -> dict:
    out = {}
    for name, spec in INFERENCE_BACKENDS.items():
        accs = []
        for r in range(INFERENCE_REPEATS):
            alt = _noisy_qnn(cfg, spec, seed * 1000 + r)
            logits = model.forward_with_qnn(data.X_test, alt)
            accs.append(float(((logits > 0).float() == data.y_test).float().mean()))
        out[name] = float(np.mean(accs))
    return out


def run_one(cfg: dict, seed: int, verbose: bool = False) -> dict:
    cfg = make_cfg(**cfg)
    set_seed(seed)
    feats = tuple(cfg["features"]) if cfg["features"] else None
    data = load_binary_iris(cfg["task"], features=feats, seed=seed)
    n_feat = data.X_train.shape[1]
    extras = {}

    if cfg["model"] == "logreg":
        t0 = time.perf_counter()
        clf = LogisticRegression(max_iter=1000).fit(data.X_train.numpy(), data.y_train.numpy())
        hist = {"train_loss": [], "test_loss": [], "test_acc": [], "train_time_s": time.perf_counter() - t0}
        logits = torch.as_tensor(clf.decision_function(data.X_test.numpy()), dtype=torch.float32)
        metrics, n_params = metrics_from_logits(logits, data.y_test), n_feat + 1
    else:
        if cfg["model"] == "classical":
            model = ClassicalNN(n_feat, cfg["hidden"])
        else:
            prec = shots_to_precision(cfg["shots"]) if cfg["shots"] else 0.0
            model = HybridQNN(n_feat, cfg["n_layers"], cfg["final_rotation"], prec, seed=seed)
        n_params = count_params(model)
        hist = train_model(model, data, cfg["epochs"], cfg["lr"], cfg["batch_size"], seed, verbose)
        # Final metrics always use the *training-time* back-end (ideal or shot-noisy).
        metrics = evaluate(model, data.X_test, data.y_test)
        if cfg["model"] == "hybrid" and not cfg["shots"]:
            extras["inference_noise_acc"] = _inference_noise_eval(model, cfg, data, seed)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_DIR / (
            f"{cfg['task']}_{cfg['model']}_q{n_feat}_L{cfg['n_layers']}_fr{int(cfg['final_rotation'])}"
            f"_sh{cfg['shots']}_s{seed}.pt"))

    return {"key": run_key(cfg, seed), **cfg, "seed": seed, "n_params": n_params,
            **{k: v for k, v in metrics.items()}, "train_time_s": hist["train_time_s"],
            "train_loss_hist": hist["train_loss"], "test_loss_hist": hist["test_loss"],
            "test_acc_hist": hist["test_acc"], **extras}


def run_grid(configs: list[dict], seeds: list[int], verbose: bool = True) -> None:
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done()
    for cfg in configs:
        cfg = make_cfg(**cfg)
        for seed in seeds:
            if run_key(cfg, seed) in done:
                continue
            t0 = time.perf_counter()
            row = run_one(cfg, seed)
            with RUNS_PATH.open("a") as f:
                f.write(json.dumps(row) + "\n")
            done.add(row["key"])
            if verbose:
                print(f"[{time.strftime('%H:%M:%S')}] {cfg['task'][:12]} {cfg['model']:9s} "
                      f"q={cfg['n_qubits']} L={cfg['n_layers']} fr={int(cfg['final_rotation'])} "
                      f"shots={cfg['shots']} seed={seed}: acc={row['accuracy']:.3f} "
                      f"auc={row['roc_auc']:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
