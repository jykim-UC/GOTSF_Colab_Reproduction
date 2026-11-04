# Reproducing *Goal-Oriented Time-Series Forecasting* (GOTSF) in Colab

An independent reproduction of

> L.-A. Fechete, M. Sana, F. Ayed, N. Piovesan, W. Li, A. De Domenico, T. Si Salem, **Goal-Oriented Time-Series Forecasting: Foundation Framework Design**, NeurIPS 2025 Workshop BERT²S, [arXiv:2504.17493](https://arxiv.org/abs/2504.17493) (2025).
> Official code: [netop-team/gotsf](https://github.com/netop-team/gotsf)

| Notebook | Contents |
|---|---|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jykim-UC/GOTSF_Colab_Reproduction/blob/main/GOTSF_Colab_Reproduction.ipynb) `GOTSF_Colab_Reproduction.ipynb` | Main experiments: Table 1 for BLW-TrafficDS and SynthDS |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jykim-UC/GOTSF_Colab_Reproduction/blob/main/GOTSF_Colab_Additional_Experiments.ipynb) `GOTSF_Colab_Additional_Experiments.ipynb` | Additional experiments: Fig. 2 (E2E policy), Fig. 5, Fig. 6–9, Table 1 for Traffic, Electricity and Weather |

한국어: [README.ko.md](README.ko.md)

> This is an unofficial reproduction and is not affiliated with the authors or Huawei. All rights to the original code and data belong to their owners.

Both notebooks clone the official repository at a fixed commit (`31b17e5`) and apply only the patches needed to run on Colab. The training and evaluation logic is the official code. The notebooks themselves are written in Korean.

## TL;DR

| Experiment | Result |
|---|---|
| **Table 1, BLW-TrafficDS** (the paper's new dataset) | ✅ All 20 cells (4 models × 5 policies) within 0–1% of the paper |
| **Table 1, SynthDS** | ✅ Averages and trends match (57–66% improvement over B vs. 52–63% in the paper). Per-interval values differ by up to 40–60% because the paper's exact synthetic trace is not public |
| **Fig. 2 and the E2E policy** | ✅ Reproduced qualitatively. B matches (34.1 vs. 33.7); the conditioned policies are 10–20% above the paper |
| **Fig. 6–9** (sensitivity analysis) | ✅ All trends reproduced; Fig. 6 also matches closely in value |
| **Table 1, Traffic** | ✅ 15 of 20 cells within ±5%, using two settings chosen by comparison with the paper ([details](#settings-chosen-by-comparison-with-the-paper)) |
| **Table 1, Electricity** | ◐ DLinear in all columns, D_L for three models, and B for iTransformer and DLinear within 0–5%. The patched columns are not reliably comparable |
| **Table 1, Weather** | ◐ Our values are about 10× the paper's. If the table's "×2" scale label is read as "×0.2", B for three models and four of five DLinear columns are within 6% |
| **Fig. 5** (energy saving) | ❌ The Perfect Forecast curve and the B policy match, but D's advantage (≈3× lower energy error) is not reproduced. The public data does not contain the repository's series |

Policy names: **B** = standard training, **C** = continuous random intervals, **D_L** = L discrete intervals (no patching), **D¹₂L / D∞₂L** = trained on 2L intervals, then patched by probability-weighted average / maximum probability. All metrics are the masked MAE used by the paper and the official code: only positions whose ground truth lies inside the interval count.

## Main results (`GOTSF_Colab_Reproduction.ipynb`)

### BLW-TrafficDS: mean MAE over I1–I8, ×10³ (ours / paper)

| Model | D_L | D¹₂L | D∞₂L | B | C ¹ | Improvement over B |
|---|---|---|---|---|---|---|
| iTransformer | 47.6 / 47.8 | 48.6 / 48.6 | 48.8 / 48.6 | 51.6 / 51.5 | 48.7 / 48.5 | 7.7% / 7.2% |
| DLinear | 54.6 / 54.5 ² | 54.9 / 54.8 | 54.3 / 54.3 | 57.6 / 57.6 | 61.7 / 61.6 | 5.7% / 5.7% ² |
| PatchTST | 22.5 / 22.4 | 28.9 / 28.8 | 25.6 / 25.5 | 50.8 / 50.7 | 26.7 / 26.7 | 55.7% / 55.8% |
| TimeMixer | 29.5 / 29.4 | 31.5 / 31.4 | 30.9 / 30.9 | 52.8 / 52.7 | 35.0 / 34.9 | 44.1% / 44.2% |

![BLW-TrafficDS, ours vs. paper](results/figures/table1_BLW_vs_paper.png)

Solid lines (ours) and dashed lines (paper) overlap almost completely.

### SynthDS: mean MAE over I1–I4, ×10³ (ours / paper)

| Model | D_L | D¹₂L | D∞₂L | B | C ¹ | Improvement over B |
|---|---|---|---|---|---|---|
| iTransformer | 10.7 / 10.8 | 10.9 / 11.0 | 11.3 / 12.3 | 25.8 / 29.1 | 12.0 / 13.2 | 58.4% / 62.9% |
| DLinear | 13.8 / 13.8 | 13.4 / 13.6 | 14.7 / 15.1 | 39.2 / 32.3 | 18.6 / 17.1 | 65.8% / 57.9% |
| PatchTST | 10.3 / 10.7 | 10.8 / 11.2 | 11.1 / 12.1 | 26.4 / 29.1 | 11.4 / 12.3 | 61.0% / 63.3% |
| TimeMixer | 14.0 / 13.7 | 13.8 / 14.2 | 15.5 / 16.5 | 32.2 / 28.3 | 14.6 / 14.6 | 57.1% / 51.7% |

![SynthDS forecasts](results/figures/toy_forecasts.png)

Standard training (B) outputs a single compromise among the four possible futures, while the interval-conditioned policies (C, D) follow the hypothesis inside each region of interest.

¹ The C column was trained with `w_hat = 0.1` (the repository's `train.sh` setting). The paper labels this column C₀.₂, but `w_hat = 0.2` deviates by up to +7.6% (TimeMixer) on BLW-TrafficDS, whereas `w_hat = 0.1` matches all four models within 0.2%.

² Table 1 prints 54.0 for DLinear's D_L average and 6.3% for its improvement. The mean of the same table's per-interval values (I1–I8) is 54.5, giving 5.7%; we report the latter.

## Additional experiments (`GOTSF_Colab_Additional_Experiments.ipynb`)

### Fig. 2 and the E2E policy

iTransformer on SynthDS with region of interest I = [0.75, 1], 10 seeds, MAE loss, 66/17/17 split. E2E is implemented as a discrete policy with a single interval [0.75, 1], an indicator loss and the classification loss disabled.

| Policy | Ours (mean ± std, 10 seeds) | Paper (Table 1, SynthDS I4) |
|---|---|---|
| (a) E2E | 14.35 ± 0.29 | – |
| (b) C (w_hat = 0.2) | 16.41 ± 0.24 | 13.4 |
| (c) D₄ | 14.75 ± 0.23 | 13.1 |
| (d) D₈, max (D∞₈) | 15.88 ± 0.54 | 13.6 |
| (e) D₈, expectation (D¹₈) | 14.22 ± 0.26 | 12.4 |
| (f) B | 34.14 ± 0.66 | 33.7 |

![Fig. 2 reproduction](results/extra/figures/fig2.png)

The paper's narrative is reproduced: the D policies come close to the E2E policy, C is the weakest of the conditioned policies, and B averages the possible futures and is more than twice as bad. Absolute values for the conditioned policies are 10–20% above Table 1, which uses a different synthetic trace.

### Fig. 6–9: sensitivity analysis

![Fig. 6–9 reproduction](results/extra/figures/fig6_9_sensitivity.png)

Values below are read from our plots and from the paper's figures, so they are approximate.

- **Fig. 6** (PatchTST, D_L with L ∈ {4, 8, 16, 32}): the error grows with interval width, and larger L is worse at coarse widths. With max patching our values at width 1 are about 0.12–0.17, close to the paper's 0.12–0.155. The paper does not say which patching it used; ours suggests max.
- **Fig. 7** (noise-free SynthDS): the ratio γ = MAE(max) / MAE(expectation) is below 1 for narrow intervals and about 1.27–1.29 at width 1 for L = 8, 16, 32 (paper: about 1.2–1.33). The exception is L = 4 (0.98).
- **Fig. 8** (PatchTST D₈, decay ν): larger ν is better for narrow intervals (ν = ∞ about 0.003 vs. ν = 0 about 0.013 at width 1/8), and all curves converge at width 1, as in the paper.
- **Fig. 9** (C policy, iTransformer, minimum interval width δ):

| δ (%) | 0 | 5 | 10 | 15 | 20 | 25 | 35 |
|---|---|---|---|---|---|---|---|
| Avg. MAE ×10³ | 12.54 | 12.39 | 12.36 | 12.26 | 12.48 | 12.68 | 13.68 |

  Same U shape as the paper. Our minimum is at δ = 15% rather than about 5%, and the value at 35% (13.68) is close to the paper's (about 14.3).

### Table 1: Traffic, Electricity and Weather

Paper averages are the means of the per-interval values in Table 1. **Bold** marks cells within ±5%; other cells show the deviation from the paper.

**Traffic** (×500)

| Model | D_L | D¹₂L | D∞₂L | B | C₀.₂ | Improvement over B |
|---|---|---|---|---|---|---|
| iTransformer | 0.94 / 0.84 (+12%) | 0.93 / 0.85 (+8%) | 0.94 / 0.87 (+8%) | **1.01 / 1.02** | **0.84 / 0.84** | 17.4% / 17.6% |
| DLinear | **1.11 / 1.10** | **0.91 / 0.91** | **0.99 / 0.98** | **1.30 / 1.29** | **1.16 / 1.15** | 29.8% / 29.9% |
| PatchTST | **0.64 / 0.62** | **0.61 / 0.59** | **0.65 / 0.63** | **1.03 / 1.00** | **0.60 / 0.61** | 41.3% / 40.6% |
| TimeMixer | **0.98 / 0.94** | **0.90 / 0.88** | **0.94 / 0.91** | 1.28 / 1.45 (-12%) | 1.02 / 1.14 (-10%) | 29.5% / 39.6% |

**Electricity** (×10⁻¹)

| Model | D_L | D¹₂L | D∞₂L | B | C₀.₂ | Improvement over B |
|---|---|---|---|---|---|---|
| iTransformer | 2.73 / 5.03 (-46%) | 2.69 / 4.08 (-34%) | 2.69 / 4.21 (-36%) | **1.41 / 1.39** | 2.78 / 1.46 (+91%) | 0.0% / 0.0% |
| DLinear | **1.60 / 1.56** | **1.63 / 1.57** | **1.62 / 1.57** | **1.58 / 1.57** | **1.59 / 1.58** | 0.0% / 0.8% |
| PatchTST | **1.29 / 1.29** | 1.30 / 1.75 (-26%) | 1.34 / 1.81 (-26%) | **1.45 / 1.38** | 1.34 / 1.43 (-6%) | 10.6% / 6.8% |
| TimeMixer | **1.53 / 1.51** | 1.73 / 1.52 (+14%) | 1.76 / 1.54 (+14%) | 1.71 / 1.86 (-8%) | 1.72 / 1.90 (-10%) | 10.6% / 18.8% |

**Weather** (our values divided by 10; see below)

| Model | D_L | D¹₂L | D∞₂L | B | C₀.₂ | Improvement over B |
|---|---|---|---|---|---|---|
| iTransformer | 0.61 / 0.29 (+105%) | 0.61 / 0.29 (+109%) | 0.63 / 0.29 (+112%) | 0.41 / 0.39 (+6%) | 0.70 / 0.74 (-6%) | 0.0% / 24.7% |
| DLinear | **0.44 / 0.42** | **0.31 / 0.31** | **0.35 / 0.34** | **0.42 / 0.40** | 0.63 / 0.55 (+14%) | 25.9% / 24.2% |
| PatchTST | 0.24 / 0.20 (+20%) | 0.32 / 0.28 (+13%) | 0.32 / 0.27 (+20%) | **0.38 / 0.37** | 0.37 / 0.40 (-6%) | 37.3% / 45.6% |
| TimeMixer | 0.30 / 0.74 (-60%) | 0.38 / 0.29 (+32%) | 0.38 / 0.30 (+27%) | 0.46 / 0.52 (-11%) | 0.42 / 0.72 (-41%) | 34.6% / 44.4% |

![Table 1, ours vs. paper](results/extra/figures/table1_extra_vs_paper.png)

- **Traffic**: the two remaining gaps are iTransformer's D policies (+8–12%, unrelated to training length) and TimeMixer's B and C (about 11% *better* than the paper, also with full training).
- **Electricity**: in the patched columns, the classification probabilities underflow to exactly 0 for up to 97–99% of the elements (PatchTST, TimeMixer), so expectation patching reduces to a plain average there. The paper does not say how this case is handled, so these columns cannot be compared reliably. iTransformer's D policies are much worse than B, as in the paper.
- **Weather**: compared directly, the B column is 7.1–11.8× the paper's in every model and interval (median 10.1). A modelling problem would not produce such a uniform factor, and our data matches the paper's description (52,696 rows, 21 variables). We therefore believe the scale label "×2" should read "×0.2", and the table above uses that reading. Weather was run with the rules found on Traffic, unchanged, as an out-of-sample test of those rules.

### Fig. 5: energy saving

iTransformer B and D⋆ (L = 4, region [0, 0.5] patched from two intervals), forecasts rolled over 96 hours, thresholds 0–2.5%.

| Target | B: energy error (Wh/day) | D: energy error (Wh/day) |
|---|---|---|
| Paper | 950 | 337 |
| Best match to the paper's Perfect Forecast curve (one beam) | 904 | 932 |
| Average over 100 beams | 1223 | 1445 |

![Fig. 5 reproduction](results/extra/figures/fig5_v6.png)

- **Data scale cannot be recovered.** The repository's `wireless.csv` is divided by one global constant (overall mean 1), so the true utilization is unknown. We searched the paper's public dataset ([netop/Beam-Level-Traffic-Timeseries-Dataset](https://huggingface.co/datasets/netop/Beam-Level-Traffic-Timeseries-Dataset); four modalities, all periods, per-beam and aggregated) for the same series with [`tools/fig5_data_diagnostic.ipynb`](tools/fig5_data_diagnostic.ipynb). The best median correlation was 0.69, reached only at time shifts that are multiples of 24 hours (daily pattern), and exact zeros make up 2.7% of `wireless.csv` versus 20–32% of the public data. The public release does not contain the repository's series.
- **What we did instead.** We normalized each beam by its maximum. The paper's three curves were extracted exactly from the vector paths in the paper's PDF. Because the Perfect Forecast curve depends only on the data, we chose the 96-hour window and target by matching that curve (RMSE 1.8 percentage points), without looking at model output.
- **Result.** The Perfect Forecast curve and B's energy error (904 vs. 950 Wh/day) match the paper, but D is not better than B.

## Settings chosen by comparison with the paper

The paper does not specify every setting. The following were chosen because they reproduce the paper's numbers, and they are what the final notebook uses by default.

| Experiment | Setting | Evidence |
|---|---|---|
| BLW-TrafficDS | Repository `train.sh` (2 layers, `d_model=128`, Adam, `lr=1e-4`, 60 epochs, patience 25) instead of the appendix | B matches Table 1 within 0.2 in every interval |
| BLW-TrafficDS | C trained with `w_hat = 0.1` | Matches all four models within 0.2%; `w_hat = 0.2` does not |
| Traffic, Electricity, Weather | Appendix settings (3 layers, `d_model=256`, `lr=1e-3`, 50 epochs, patience 5) | B matches for iTransformer and DLinear |
| Traffic | Decay rate 37 divided by the value range (185), as the paper's own description implies | D_L went from +13–26% to +1–4% for three models |
| Traffic | Early stopping disabled for D₈ (DLinear, PatchTST, TimeMixer), PatchTST B, TimeMixer B and C | These runs stopped at epochs 6–11; with full training the patched columns went from +7–63% to 0–3% |
| Electricity | Raw decay rate 37, no changes | D_L already matches within 0–2% |
| Weather | Traffic's rules applied unchanged (scaled decay, D₈ without early stopping) | Out-of-sample test of the Traffic rules |
| Fig. 5 | Per-beam max normalization; window and target matched to the Perfect Forecast curve | See above |

## Inconsistencies found in the paper

1. **Table 1, BLW-TrafficDS**: DLinear's D_L average is printed as 54.0, but the per-interval values average to 54.5.
2. **Table 1, BLW-TrafficDS**: the column labelled C₀.₂ matches `w_hat = 0.1` (the repository setting), not 0.2.
3. **Table 1, Weather**: the values are uniformly about 10× smaller than ours; the "×2" scale label most likely should read "×0.2".
4. **Appendix A**: the stated training configuration does not match the BLW-TrafficDS results, which match the repository's `train.sh` instead.
5. **Fig. 8**: the caption says D₈ and the text says D₃₂; the x-axis (widths 1/8 to 1) matches D₈.
6. **SynthDS**: the paper describes 3,456 points and a 66/17/17 split; the repository file has 3,072 points.

## How to run

**Main notebook.** Click its Colab badge, choose a GPU runtime and run all cells. The defaults are the paper's Table 1 conditions (MAE loss, `RUN_MODE = "full"`, 60 epochs); `RUN_MODE = "quick"` (3 epochs) is a fast smoke test. It takes about 15 minutes on an NVIDIA RTX PRO 6000.

**Additional experiments notebook.** Each part has its own switch (`RUN_FIG2`, `RUN_FIG5`, `RUN_SENS`, `RUN_TABLE1`). All data is downloaded automatically: Traffic and Electricity from the LSTNet GitHub mirror, Weather from the [Time-Series-Library Hugging Face copy](https://huggingface.co/datasets/thuml/Time-Series-Library), and DLPRB from the paper's dataset.

| Part | Paper | Runs | Time (A100, approx.) |
|---|---|---|---|
| C | Fig. 2, E2E | 50 | 30–40 min |
| D | Fig. 5 | 2 | 5 min |
| E | Fig. 6–9 | 21 | 30–40 min |
| F | Table 1: Traffic, Electricity, Weather | 60 | more than 10 h (Weather alone took 5.4 h on a T4) |

For part F, set `USE_DRIVE = True`. Results are then saved to Google Drive and finished runs are skipped when the notebook is run again.

## Trained weights

The [Releases](../../releases) page provides the weights of the 133 runs behind the additional experiments' results: Fig. 2 (50), Fig. 5 (2), Fig. 6–9 (21) and Table 1 for Traffic, Electricity and Weather (60).

| File | Contents |
|---|---|
| `gotsf_extra_checkpoints.zip` | `checkpoints_extra/<run>/` with `checkpoint.pth`, `args.pk` and `train_loss.npy`, plus `metrics_extra.csv` |
| `metrics_extra.csv` | Every evaluated metric (MAE, MSE, RMSE) of those runs |

Weights for the main notebook are not provided, since it retrains in about 15 minutes. Runs that were superseded while matching the paper (for example Traffic with the raw decay rate) are not included either.

To evaluate the additional experiments without retraining:
1. Open the additional experiments notebook in Colab and run sections 0–2 (clone and patches).
2. Upload `gotsf_extra_checkpoints.zip` to the session and extract it into the cloned repository:
   ```python
   !unzip -q /content/gotsf_extra_checkpoints.zip -d /content/gotsf
   ```
3. Run the remaining cells with `USE_DRIVE = False`. Finished runs are detected, so training is skipped and only evaluation runs.

`checkpoint.pth` is a PyTorch `state_dict`. `args.pk` is a pickle of the training arguments, so only open it from a source you trust.

## Repository structure

```
├── README.md, README.ko.md
├── GOTSF_Colab_Reproduction.ipynb             # main experiments (executed)
├── GOTSF_Colab_Additional_Experiments.ipynb   # additional experiments (final settings)
├── results/
│   ├── table1_comparison.csv                  # BLW-TrafficDS and SynthDS vs. paper
│   ├── tables/, figures/                      # main notebook outputs
│   └── extra/
│       ├── tables/                            # additional experiments, incl. table1_extra_comparison.csv
│       └── figures/
├── logs/                                      # main notebook logs
│   └── extra/                                 # additional experiments logs
├── tools/
│   ├── fig5_data_diagnostic.ipynb             # is wireless.csv in the public dataset? (executed)
│   └── package_weights.py                     # bundles checkpoints and metrics from a Colab session
└── LICENSE
```

## Patches to the official code

| # | File | Change | Reason |
|---|---|---|---|
| 1 | `utils/tools.py` | `np.Inf` → `np.inf` | Removed in NumPy 2.x |
| 2 | `utils/tools.py` | `plt.close()` after saving in `visual()` | Avoid accumulating figures |
| 3 | `layers/SelfAttention_Family.py` | Optional `reformer_pytorch` import | Unused Reformer dependency |
| 4 | `experiments/exp_basic.py` | Fall back to CPU without a GPU | The original always uses `cuda:0` |
| 5 | `experiments/exp_long_term_forecasting_*.py` | `torch.load(..., map_location=self.device)` | Evaluate GPU checkpoints on CPU |
| 6 | `data_provider/data_loader.py` | `drop(['date'], 1)` → `drop(columns=['date'])` | pandas 2.x API |
| 7 | `script_train.py` | Seed from `GOTSF_SEED` (default 2023) | 10-seed runs for Fig. 2 |
| 8 | `data_provider/data_loader.py` | Split ratios from `GOTSF_PERC_TRAIN`, `GOTSF_PERC_TEST` (default 0.7, 0.2) | The paper's 66/17/17 split |
| 9 | `data_provider/data_factory.py` | Validation and test batch size from `GOTSF_EVAL_BS` (default: whole set) | GPU memory on large datasets |
| 10 | `experiments/exp_long_term_forecasting_*.py` | Slice interval tensors to the batch; concatenate predictions | Needed for 9 |

Patches 1–6 are used by both notebooks and 7–10 only by the additional experiments notebook. Patches 7–10 change nothing unless their environment variables are set. The shell scripts (`train.sh` and others) use `conda activate`, `CUDA_VISIBLE_DEVICES=1` and 16 parallel jobs, so the notebooks run the same arguments sequentially from Python.

The additional experiments notebook also changes the evaluation script around the official `ModelEvaluator`:
- **Vectorized patching.** The official code loops over every element in Python, which takes hours on Traffic and Electricity. The vectorized version is bit-identical for max patching and identical up to floating-point summation order for expectation.
- **No overwriting of source arrays.** When a region of interest equals a trained sub-interval, the official code overwrites that sub-interval's predictions. This silently corrupted results once probabilities underflowed, so the source arrays are kept.
- **All-zero probabilities.** Where all patched probabilities are 0, expectation patching uses a plain average; the official code produces NaN.
- **Array pruning.** For Table 1, prediction arrays are deleted after evaluation and only metrics are kept (about 85 GB otherwise).

Batched evaluation was verified to give bit-identical outputs to full-batch evaluation.

## Observations

- **iTransformer's classification head** is computed from inputs after per-variable normalization (`use_norm`), so it cannot see each beam's absolute level. Its probabilities lean towards the low intervals that are common in the data.
- **Early stopping is fragile.** With patience 5, the D₈ models of PatchTST and TimeMixer on Traffic stopped at epochs 6–10 and were 50–63% off the paper; trained for 50 epochs they were within 2–3%. The stopping point also depends on the GPU: PatchTST C on Traffic stopped at epoch 23 on an L4 (0.68) but ran all 50 epochs on an A100 (0.60).
- **Determinism.** Re-running on the same GPU type reproduced every number bit for bit.

## Environment

| Notebook or part | GPU (Google Colab) | Software |
|---|---|---|
| Main notebook | NVIDIA RTX PRO 6000 Blackwell (an earlier run on an A100 gave matching numbers) | Python 3.13.15, PyTorch 2.11.0+cu130, NumPy 2.1.3, pandas 2.2.3 |
| Fig. 2, Fig. 6–9, Table 1 Traffic and Electricity | NVIDIA A100 40GB (Traffic D₈ retraining on an A100 80GB, three Traffic B/C retrainings on an L4) | Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3, pandas 2.2.3 |
| Fig. 5, Table 1 Weather | NVIDIA T4 | Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.4.6, pandas 2.2.3 |

Official code: [netop-team/gotsf](https://github.com/netop-team/gotsf) @ `31b17e5` (2025-06-05).

## Original paper

```bibtex
@misc{fechete2025goaloriented,
  title         = {Goal-Oriented Time-Series Forecasting: Foundation Framework Design},
  author        = {Fechete, Luca-Andrei and Sana, Mohamed and Ayed, Fadhel and Piovesan, Nicola and Li, Wenjie and De Domenico, Antonio and Si Salem, Tareq},
  year          = {2025},
  eprint        = {2504.17493},
  archivePrefix = {arXiv}
}
```

## License

- **Notebooks, scripts and results in this repository**: MIT License ([LICENSE](LICENSE))
- **Official code**: [netop-team/gotsf](https://github.com/netop-team/gotsf) is MIT-licensed; its copyright notice follows the original repository's LICENSE.
- **Datasets**: subject to the terms of the original repository and the [Hugging Face dataset page](https://huggingface.co/datasets/netop/Beam-Level-Traffic-Timeseries-Dataset).
- **Paper PDF**: not included for copyright reasons.
