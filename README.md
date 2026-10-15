# Hybrid Quantum–Classical Neural Network for Binary Classification

A PyTorch model whose middle layer is a 4-qubit parameterised quantum circuit (Qiskit), trained end-to-end
and compared against classical baselines on binary Iris tasks. Everything runs on **simulators**
(Qiskit `StatevectorEstimator` for ideal runs, Qiskit Aer for noisy evaluation).

**Headline finding: the hybrid model matches the classical network; it does not beat it, and it is ~1000x slower to train.**
This is an honest result on a tiny dataset, not a claim of quantum advantage.

## Architecture
```
x (4 features) -> Linear(4,4) -> pi*tanh -> [Ry encode | Ry,Rz | CNOT chain | Ry] -> <Z_i> (4 values) -> Linear(4,1) -> sigmoid
   classical feature layer        angles in [-pi,pi]      4-qubit PQC                    measurement        classical output layer
```
Trained jointly with Adam + binary cross-entropy; gradients reach the classical layers and the circuit
(parameter-shift via `EstimatorQNN` + `TorchConnector`). Circuit diagram: `results/figures/circuit_4q_1layer.png`.

## Setup
```bash
pip install -r requirements.txt          # tested with Python 3.12
python -m pytest -q                      # 17 tests
python -m experiments.run_all            # all experiments, ~1.5 h on 1 CPU core, resumable
python -m experiments.make_report        # tables + figures -> results/
```
Notebooks (`notebooks/`) are Colab-ready; edit the git-clone URL in the first cell.

## Protocol
Stratified 80/20 split (80 train / 20 test), scaler fitted on train only, Adam lr 0.05, batch 16, 20 epochs,
no early stopping or model selection on the test set. The **seed controls split, initialisation, batch order and shot noise**,
so each seed is a different train/test split; results are mean ± std over seeds (3–5). Classical NN: Linear(4,4)–tanh–Linear(4,1), 25 params;
hybrid: 37 params. Logistic regression is included as a sanity reference.

## Results
Test metrics, mean ± std over seeds. Full tables: `results/metrics.csv`, `results/summary.md`; per-run data: `results/all_runs.csv`.

**Versicolor vs virginica** (overlapping classes, the informative task)

| Model | seeds | params | Accuracy | F1 | ROC-AUC | Train time |
|---|---|---|---|---|---|---|
| Logistic regression | 5 | 5 | 0.970 ± 0.027 | 0.968 | 1.000 | <0.1 s |
| Classical NN | 5 | 25 | 0.970 ± 0.045 | 0.971 | 1.000 | <0.1 s |
| **Hybrid, 4 qubits, 1 layer** | 5 | 37 | 0.970 ± 0.027 | 0.968 | 1.000 | 145.6 ± 4.2 s |

**Setosa vs versicolor** (the task from the original plan) is linearly separable: every model scores 1.000, so it cannot discriminate between models.

### Experiments
| Experiment | Result (versicolor vs virginica) |
|---|---|
| 2: depth, 1 vs 2 layers (seeds 0–2) | Accuracy 0.983 for both; L=2 takes ~2x longer (297 s vs ~145 s). No benefit. |
| 3: 2 qubits (petal length/width) vs 4 | 2q hybrid 0.960 ± 0.042 (23 s) vs 4q 0.970 ± 0.027 (146 s). Classical 2-feature NN: 0.960. |
| 4a: train with finite shots (~100 / ~1024, Gaussian shot-noise model) | 0.950 ± 0.050 for both, vs 0.983 ideal on the same seeds 0–2. |
| 4b: trained ideally, evaluated on Aer with 100/1024 shots or depolarizing noise (1q 0.1-0.5 %, 2q 1-5 %) | Accuracy unchanged (0.970–0.974 vs 0.970 ideal). |
| Ablation: literal ansatz without final Ry layer | 0.983 ± 0.029, same as ours at L=1 and L=2, and faster (93 s). |

Figures: `results/figures/` (learning curves, confusion matrices, depth/ablation, qubit count, noise).

## Findings worth knowing
1. **Dead parameters in the textbook ansatz.** With `Ry,Rz -> CNOT chain -> measure Z`, every <Z_i> pulls back through the
   CNOT chain to a Z-string, which commutes with Rz, so all Rz gradients are exactly 0 (unit-tested). A final trainable Ry layer
   fixes this. In the ablation this made **no measurable accuracy difference** on this task, so it is a correctness fix, not a performance gain.
2. **Qiskit-ML pitfall:** `EstimatorQNN` defaults to precision 1/64, i.e. it silently injects shot noise into "ideal" simulation. We set it explicitly.
3. **Seeding pitfall:** `StatevectorEstimator` with an integer seed replays the *same* noise every call. We pass a persistent `Generator`.

## Limitations
- 20 test samples per split: one sample = 5 % accuracy. Differences of ~0.01–0.03 are **not statistically meaningful**, and with only 3–5 seeds no significance tests are claimed.
- The 100-shot and 1024-shot training runs give identical accuracy/AUC; their loss curves differ, so they are distinct runs. The same two borderline test samples are misclassified in both, which a test set this small cannot resolve.
- Shot noise during training uses `StatevectorEstimator`'s Gaussian approximation (std = 1/sqrt(shots)); the inference-time study uses true sampling on Aer.
- Simulation only; no real hardware. Depolarizing noise levels were chosen by hand. The 4-qubit ideal state-vector simulation is classically trivial, so no speed or capacity advantage is expected.
- Training time reflects Qiskit-ML parameter-shift overhead on a CPU, not quantum hardware.

## Layout
`src/` model + circuit + training + metrics · `experiments/` runner and one script per experiment · `tests/` · `notebooks/` · `results/` metrics, figures, raw runs, saved weights.
