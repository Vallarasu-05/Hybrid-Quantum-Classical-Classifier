### setosa vs versicolor

| label | n_seeds | n_params | accuracy | precision | recall | f1 | roc_auc | train_time_s |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 5 | 5 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 ± 0.0 |
| Classical NN (4f) | 5 | 25 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.0 ± 0.0 |
| Hybrid 4q L=1 | 3 | 37 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 143.5 ± 1.0 |

### versicolor vs virginica

| label | n_seeds | n_params | accuracy | precision | recall | f1 | roc_auc | train_time_s |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 5 | 5 | 0.970 ± 0.027 | 1.000 ± 0.000 | 0.940 ± 0.055 | 0.968 ± 0.029 | 1.000 ± 0.000 | 0.0 ± 0.0 |
| Classical NN (4f) | 5 | 25 | 0.970 ± 0.045 | 0.967 ± 0.075 | 0.980 ± 0.045 | 0.971 ± 0.042 | 1.000 ± 0.000 | 0.0 ± 0.0 |
| Hybrid 4q L=1 | 5 | 37 | 0.970 ± 0.027 | 1.000 ± 0.000 | 0.940 ± 0.055 | 0.968 ± 0.029 | 1.000 ± 0.000 | 145.6 ± 4.2 |
| Classical NN (2f) | 5 | 9 | 0.960 ± 0.022 | 1.000 ± 0.000 | 0.920 ± 0.045 | 0.958 ± 0.024 | 1.000 ± 0.000 | 0.0 ± 0.0 |
| Hybrid 2q L=1 | 5 | 15 | 0.960 ± 0.042 | 1.000 ± 0.000 | 0.920 ± 0.084 | 0.957 ± 0.046 | 1.000 ± 0.000 | 23.4 ± 0.3 |
| Hybrid 4q L=2 | 3 | 45 | 0.983 ± 0.029 | 0.970 ± 0.052 | 1.000 ± 0.000 | 0.984 ± 0.027 | 1.000 ± 0.000 | 297.3 ± 1.7 |
| Hybrid 4q L=1 (literal ansatz) | 3 | 33 | 0.983 ± 0.029 | 0.970 ± 0.052 | 1.000 ± 0.000 | 0.984 ± 0.027 | 1.000 ± 0.000 | 93.4 ± 0.4 |
| Hybrid 4q L=2 (literal ansatz) | 3 | 41 | 0.983 ± 0.029 | 0.970 ± 0.052 | 1.000 ± 0.000 | 0.984 ± 0.027 | 1.000 ± 0.000 | 221.6 ± 0.8 |
| Hybrid 4q L=1 train-shots=100 | 3 | 37 | 0.950 ± 0.050 | 0.944 ± 0.096 | 0.967 ± 0.058 | 0.952 ± 0.046 | 0.997 ± 0.006 | 144.7 ± 0.5 |
| Hybrid 4q L=1 train-shots=1024 | 3 | 37 | 0.950 ± 0.050 | 0.944 ± 0.096 | 0.967 ± 0.058 | 0.952 ± 0.046 | 0.997 ± 0.006 | 144.8 ± 0.9 |

### inference-time noise (trained ideally, evaluated on Aer; test accuracy)

| model | n_seeds | ideal | shots_1024 | shots_100 | depol_low | depol_high |
|---|---|---|---|---|---|---|
| Hybrid 4q L=1 | 5 | 0.970 ± 0.027 | 0.972 ± 0.026 | 0.974 ± 0.025 | 0.972 ± 0.026 | 0.974 ± 0.025 |
| Hybrid 4q L=2 | 3 | 0.983 ± 0.029 | 0.997 ± 0.006 | 0.993 ± 0.012 | 0.987 ± 0.023 | 0.987 ± 0.023 |

### paired differences

- setosa_vs_versicolor: hybrid - classical test accuracy = +0.000 (n=3 paired seeds, wins/ties/losses = 0/3/0)
- versicolor_vs_virginica: hybrid - classical test accuracy = -0.000 (n=5 paired seeds, wins/ties/losses = 1/2/2)
- depth: L=2 - L=1 accuracy = +0.000 (n=3 paired seeds, wins/ties/losses = 1/1/1)
