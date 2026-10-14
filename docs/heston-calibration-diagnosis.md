# Heston calibration diagnosis — 14 October 2026

The available evidence does **not** establish that differential evolution cannot fit the curve, or that the nonlinear Heston model is intrinsically incapable of doing so. It establishes problems in result provenance, the return observation rule, the first-hitting-time counter, objective resolution, and numerical stability. These should be addressed before another expensive optimizer restart.

This audit uses the saved `conditional_mfht_joint_explore` UN/LN parameters, the raw empirical matrices, the supplied Fortran sources, and fresh simulations. It leaves the original fits, notebook edits, and production model unchanged. No newly calibrated solution is claimed.

## 1. The plotted result is stale

Both `conditional_mfht_joint_explore/mfht_recount/{UN,LN}/summary.json` identify **conditional_mfht_joint_trial** as their source. Their six optimized parameters differ from the current explore parameter CSVs. The notebook selects the explore directory but checks only whether old preprocessing metadata is present; it does not compare the parameters.

A fresh independent-seed recount of the actual explore fit is saved under `data/processed/heston_diagnosis/fresh_recount`. With the notebook's five-bin pooling and minimum 100 events:

| Market | Empirical peak (steps) | Synthetic peak (steps) | Empirical physical-volatility bin | Synthetic bin |
|---|---:|---:|---|---|
| UN | 80.79 | 62.79 | 0.0110–0.0112 | 0.0128–0.0130 |
| LN | 64.20 | 73.80 | 0.0094–0.0096 | 0.0130–0.0132 |

These are descriptive maxima of noisy binned curves, not confidence-bounded peak estimates. Correcting provenance does not eliminate the mismatch. The Fortran reported x-axis is approximately half the physical volatility; this rescales both curves and cannot explain their relative mismatch.

## 2. Resetting the latent state creates artificial financial returns

The fitted runs use `sampling_return_mode='sampled_x'` and convert log increments to simple returns with `expm1`. The simulator first resets an escaped state below -6 to zero, then computes differences between sampled **post-reset** states. Consequently the numerical restart becomes an upward price jump. An observed log jump of 6 means a simple return of approximately 402, or 40,200%.

I isolated this using identical internal shocks: record every pre-reset Euler increment, sum those increments within each observation interval, and compare with the existing post-reset endpoint differences. This is a diagnostic alternative, not a silently adopted new model.

| Same-shock experiment, seed 81001 | UN | LN |
|---|---:|---:|
| Original average per-stock daily SD | 39.53% | 52.04% |
| Average SD after excluding artificial restart jumps | 2.99% | 4.03% |
| Observations containing a reset | 0.0249% | 0.0588% |
| Gains above 100% occurring in reset intervals | 100% | 100% |

Removing reset jumps does **not** by itself repair MFHT, and it changes the mean return and crash distribution. A consistent interpretation of post-escape dynamics is needed before fitting price series. Possibilities include regenerative episodes with a break between them, or an accumulated return process excluding restarts; these imply different financial processes and should be compared explicitly.

The supplied Fortran records `dx=x-x_old` **before** resetting x. Thus the current multi-step endpoint convention differs materially from the supplied simulator, not merely in sampling frequency.

The objective largely overlooks these rare gains. It compares 103 nonzero-return quantiles ending at 0.999, without a moment or extreme-tail constraint. Reset events occur at less than 0.1% frequency; arbitrarily large observations beyond the largest fitted quantile need not change this loss appreciably. A good quantile loss therefore does not imply realistic prices or even a realistic variance.

## 3. The legacy counter has memory that corrupts the intended FHT statistic

`count_fortran_hitting_events(..., reset_on_discard=False)` faithfully retains a problematic Fortran rule: duration and volatility accumulators are cleared only after an **accepted** crossing.

This has two distinct effects:

- Rejected short crossings can be concatenated into later events. Among saved empirical accepted events, FHT differs from `end_step-start_step` in 11.87% of UN events and 10.68% of LN events.
- Once the accumulated duration exceeds 300, it cannot return to the admissible interval. A dense synthetic path then produces no further accepted crossings. A stock-specific NaN clears this state in empirical data, so missingness changes the effective process.

A deterministic check with one overlong episode followed by five ordinary two-step episodes produces **0** accepted events with the legacy counter, **5** with `reset_on_discard=True`, and **4** with the legacy counter after inserting one missing observation. This is a counter-state issue, independent of Heston parameters.

On the full empirical matrices, changing only `reset_on_discard` increases accepted counts from 201,947 to 244,499 (UN) and from 139,027 to 159,539 (LN). These are different target statistics, so a scientifically corrected calibration must recount both datasets and cannot retain the old target unchanged.

Even the clean counter estimates a duration distribution conditional on completed, retained events with 2 <= T <= 300. The current 'survival' objective is the survival function of that truncated sample, not a censoring-aware survival estimate. Finite series, gaps, and maximum duration must be aligned or explicitly modeled. Kaplan–Meier is not automatically appropriate when selection also depends on volatility measured over the completed event.

## 4. The objective scarcely resolves the curve being judged

The optimized loss is

`conditional FHT loss + 0.2 * MFHT loss + nonzero-return quantile loss`.

The four conditional volatility regions are extremely broad. The first ends at 0.02499 for UN and 0.02937 for LN, encompassing the rising branch, peak, and much of the decline. Its averaged survival curve can fit while the detailed MFHT curve does not.

The 30 MFHT bins extend to the empirical 99.5th percentile, which is far beyond the displayed interval:

| Objective property | UN | LN |
|---|---:|---:|
| Upper physical volatility | 0.13369 | 0.24390 |
| Bin width | 0.004456 | 0.008130 |
| Bins intersecting displayed 0–0.04 range | 9 | 5 |

Thus LN's visual peak is compressed into roughly one broad MFHT bin. Most bins concern the unplotted tail, each has equal weight in the shape loss, and the resulting loss has weight only 0.2. Broad-bin means also depend on how events are distributed **within** each bin.

In the saved explore fit diagnostics, the weighted MFHT terms are about 0.0172 (UN) and 0.0500 (LN), versus return terms 0.0641 and 0.1088. These numerical weights are not uncertainty-normalized and need not reflect the scientific priority.

Low simulated counts create another problem: bins with fewer than 30 events receive a fixed error instead of their measured residual. LN's fresh 1071-path recount has MFHT loss 0.319 versus about 0.250 in its saved 1189-path pilot diagnostics. The small 256-path experiments have larger coverage penalties still. Their raw total losses should not be compared directly with full-size optimization losses.

## 5. The numerical variance process is materially different from the intended CIR process

For notation, repository `aa, bb, cc` are CIR parameters kappa, theta, xi; repository `a, b` are potential coefficients. The fitted Feller ratios `2*kappa*theta/xi^2` are only 0.0541 (UN) and 0.0315 (LN). This does not invalidate CIR: it makes the near-zero boundary particularly relevant.

The simulator repeatedly redraws a Gaussian Euler proposal until it is nonnegative. That samples a truncated Gaussian, whose conditional mean exceeds the untruncated Euler mean. If a proposal has mean mu and SD s, its accepted mean is `mu + s*phi(mu/s)/Phi(mu/s)`. This creates an upward drift near zero.

Across three independent realizations, mean saved variances were approximately 0.01212 and 0.01174, compared with fitted theta values 0.002178 and 0.001644. The first-proposal negative probability evaluated at saved states averaged about 7.53% and 6.09%; this diagnostic is not an instrumented count of all internal rejections.

Halving dt while doubling both the sampling interval in internal steps and burn-in steps preserves the physical observation and burn-in durations. It nevertheless changes the MFHT shape appreciably. On a shared set of adequately populated physical-volatility bins, the pooled MFHT RMSE changes:

| Same parameters; three independent seeds | UN (58 common bins) | LN (55 common bins) |
|---|---:|---:|
| Original dt | 5.80 steps | 9.32 steps |
| Half dt | 2.45 steps | 12.61 steps |

This is sensitivity, not proof that half dt is converged or that it solves the calibration. It mixes price integration, variance approximation, and reset-timing effects. It does show that fitted parameters cannot yet be interpreted independently of the numerical grid.

For an SDE-faithful implementation, test the exact noncentral-chi-square CIR transition for variance and convergence of the remaining price integration. Exact variance transitions alone do not make the joint price simulation exact. Retain redraw Euler as a separate historical-code reproduction. See [Malham and Wiese, Chi-square simulation of the CIR process and the Heston model](https://arxiv.org/abs/0802.4411).

## 6. Observation heterogeneity matters, but is not a demonstrated one-parameter fix

Empirical zero-return frequencies are 6.26% for UN and **34.95% for LN**, versus zero for the continuous simulator. Zero returns are removed from the return-distribution objective but retained in empirical FHT counting. Many inactive observations therefore affect one target but not the other.

An independent Bernoulli zero-insertion ablation raises and shifts the LN peak and worsens common-support RMSE from 9.32 to 10.83 steps. Matching zero frequency alone is insufficient. Price rounding, stale quotes, zero-run lengths, trading activity, and differences across stocks need to be characterized before constructing an observation model.

Shuffling returns within each synthetic path preserves its marginal distribution but greatly lowers the peak. This confirms that temporal dependence matters. A smaller hump survives, so the existence of a hump alone does not identify nonlinear dynamics: local volatility is computed on a window whose length is the hitting time itself. Null models should use the identical event measurement.

The current fit also pools decades and heterogeneous stocks into a single parameter vector. This is a plausible remaining source of mismatch, not a proven model inadequacy. The empirical saved moments contain extreme outliers too; raw-price corporate actions and return construction should be inspected before imposing tail constraints indiscriminately.

## Recommended next calibration

1. **Lock down the experiment.** Keep the historical Fortran reproduction separate. For price synthesis, choose explicit post-escape behavior, exclude numerical restart jumps from financial returns, use the clean counter on both datasets, and match observation horizons and gap handling. Save provenance linking parameters, counting rules, and outputs.
2. **Establish numerical convergence.** Compare dt, dt/2, dt/4 with fixed observation duration and multiple seeds. Test an exact CIR variance step. Refit only after choosing a stable numerical specification.
3. **Make MFHT the actual primary target.** Use fixed physical-volatility edges with adequate resolution near the peak, for example initial widths 0.0005–0.001 over 0–0.04, merging poorly supported empirical bins before optimization. Keep outer-tail diagnostics separate. Give the rising branch, peak, and decline explicit balanced influence. A useful residual is `log(M_s/M_e)`, with weights or tolerances estimated by stock/time-block resampling and a separate support penalty. Report absolute errors in days as well. Do not let candidates improve the loss by emptying difficult bins.
4. **Constrain realistic returns.** Retain marginal quantiles, but also inspect mean, robust scale, per-stock scale dispersion, tail exceedance rates, zero-run statistics, return and absolute-return autocorrelations. Fit a Pareto tradeoff or minimize MFHT subject to prespecified tolerances on these quantities, instead of choosing arbitrary loss weights and judging only one component afterward.
5. **Improve search only after those changes.** Search positive scale parameters in log coordinates; use mean-reversion time, variance level, and dimensionless volatility-of-variance to organize bounds. Profile or fix vstart after a burn-in sensitivity check: nominal CIR initial-memory factors after the current burn-in are about 7.6e-7 (UN) and 0.012 (LN). Use several fixed training seeds, independent validation seeds, saved populations, and enough paths to populate the target bins. Redraw-dependent RNG consumption means the current single fixed seed is reproducible but does not provide well-aligned common shocks across parameter candidates.
6. **Test model capacity.** Compare held-out dates/stocks, multiple crash/rally thresholds, and temporal diagnostics. If a persistent residual exceeds empirical and simulation uncertainty after the above corrections, test a small hierarchical stock-volatility mixture or activity process. Do not infer a need for more parameters merely from the existing optimizer's plateau.

The explore runs reached their iteration limits. The earlier UN trial reported convergence, but convergence only means the population met SciPy's stopping rule, not that the model is statistically adequate. See the [SciPy differential-evolution documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).

The [paper's accessible preprint](https://arxiv.org/html/1708.08695v1) demonstrates fixed-parameter agreement for a particular ensemble and measurement protocol; it is not evidence that one homogeneous parameter set must fit every market, period, and observation convention. The journal abstract was accessible; the detailed comparison here used the accessible preprint and the repository's supplied Fortran.

## Reproduce and inspect

Run from the repository root:

```powershell
.venv/Scripts/python.exe scripts/diagnose_heston_fit.py
.venv/Scripts/python.exe scripts/count_optimized_heston_mfht.py --fit-dir data/processed/heston_calibration/conditional_mfht_joint_explore --markets UN LN --output-dir data/processed/heston_diagnosis/fresh_recount
```

The diagnostic uses 256 paths x 3030 observations for seeds 81001, 81002, 81003 per market. Reset exclusion uses the first seed with identical internal shocks. Recounts use 1071 x 3030 and independent seed 523456. Empirical comparisons retain the selected full histories. These are sensitivity experiments, not bootstrap confidence intervals. Common-support RMSE omits unsupported bins, so it must be read together with coverage and the full curves.

Outputs under `data/processed/heston_diagnosis/`: `audit.json`, `experiments.csv`, `common_support_rmse.csv`, `diagnostics.png`, per-market target/ablation CSVs, and the fresh recount. Data outputs are ignored by Git; the diagnostic script and this report are source artifacts. All 43 existing Heston tests passed.
