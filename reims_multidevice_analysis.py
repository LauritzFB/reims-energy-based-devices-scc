# %% [markdown]
# # REIMS with four energy-based devices for SCC detection: analysis workflow
#
# This notebook reproduces the data analysis of *"Comparison of Energy-Based Devices for Mass
# Spectrometric Detection of Skin Cancer During Tissue Removal"*, starting from the peak-picked
# spectral matrix.
#
# **Input** (`data/`):
# * `reims_peakpicked_spectra.npz`: `X` (933 spectra x 10,480 consensus peaks, raw ion counts) and `mz`.
# * `reims_spectrum_metadata.csv`: one row per spectrum (`burn_file`, `scan_index`, `tissue`, `device`, `sample_id`).
# * `X_burn` in the same .npz file and `reims_burn_metadata.csv`: burn-level matrix (100 burns/acquisitions), used only for the
#   spectral-richness count.
#
# **Upstream steps not repeated here** (performed in MS IonVision): conversion of Waters .raw files to
# centroided mzML (ProteoWizard msconvert), selection of the scans belonging to each burn
# (*m/z* 200-1,400, intensity >= 10 counts), consensus peak picking on class-average spectra
# (local-maximum method, window 11 points, minimum height 0.1 % of the class maximum) and extraction of
# peak intensities per spectrum (tolerance 0.01 Da).
#
# **Steps in this notebook**
# 1. Dataset structure (mouse > burn > spectrum)
# 2. Outlier detection (PCA: Hotelling's T2 and Q residuals)
# 3. Mouse-level training / hold-out split
# 4. Feature selection (prevalence filter, FDR, logistic-regression ranking, nested cross-validation)
# 5. Model comparison
# 6. Device-independent model and hold-out evaluation
# 7. Intra- and cross-device models
# 8. Supplementary analyses: unsupervised PCA, feature-level statistics, mean spectra, spectral richness
#
# Requirements: Python >= 3.10, numpy, pandas, scipy, scikit-learn, matplotlib.

# %%
import json, os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
DATA, OUT = "data", "output"
os.makedirs(OUT, exist_ok=True)

DEVICES = ["SP_CO2", "P_CO2", "Cautery", "N2_plasma"]
DEVICE_NAME = {"SP_CO2": "Super-pulsed CO2 laser", "P_CO2": "Pulsed CO2 laser",
               "Cautery": "Electrocautery", "N2_plasma": "Nitrogen plasma"}
DEVICE_LABEL = {"SP_CO2": "Super-pulsed CO$_2$ laser", "P_CO2": "Pulsed CO$_2$ laser",
                "Cautery": "Electrocautery", "N2_plasma": "Nitrogen plasma"}
DEVICE_COLOR = {"SP_CO2": "#C8442C", "P_CO2": "#0E9F78", "Cautery": "#3F51B5", "N2_plasma": "#8E44AD"}
TISSUE_COLOR = {"tumor": "#D81B8C", "non_tumor": "#4CAF50", "background": "#888888"}
TISSUE_NAME = {"tumor": "Tumor", "non_tumor": "Non-tumor", "background": "Background"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "savefig.dpi": 400, "figure.facecolor": "white"})
FACTS = {}   # key numbers, written to output/key_numbers.json at the end

# %% [markdown]
# ## Helper functions

# %%
def tic_log(X):
    """Total-ion-count normalisation followed by log(1 + x)."""
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1
    return np.log1p(X / s)


def prevalence_mask(X, y, threshold=30.0):
    """Keep features detected (> 0) in at least `threshold` % of the spectra of any class."""
    keep = np.zeros(X.shape[1], dtype=bool)
    for c in np.unique(y):
        keep |= (X[y == c] > 0).mean(axis=0) * 100 >= threshold
    return keep


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p values."""
    p = np.nan_to_num(np.asarray(p, dtype=float), nan=1.0)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def class_pvalues(X, y):
    """Welch t test for two classes, one-way ANOVA for more than two."""
    classes = [c for c in np.unique(y) if (y == c).sum() >= 2]
    if len(classes) == 2:
        return stats.ttest_ind(X[y == classes[0]], X[y == classes[1]], equal_var=False).pvalue
    return stats.f_oneway(*[X[y == c] for c in classes]).pvalue


def logreg():
    return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)


def lr_importance(model):
    """Mean absolute coefficient across classes (one row for a binary model)."""
    return np.abs(model.coef_).mean(axis=0)


def binary_metrics(y_true, y_pred, positive="tumor", negative="non_tumor"):
    tp = int(((y_true == positive) & (y_pred == positive)).sum())
    fn = int(((y_true == positive) & (y_pred == negative)).sum())
    tn = int(((y_true == negative) & (y_pred == negative)).sum())
    fp = int(((y_true == negative) & (y_pred == positive)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"n": tp + fn + tn + fp, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "accuracy": (tp + tn) / max(tp + fn + tn + fp, 1), "sensitivity": sens,
            "specificity": spec, "balanced_accuracy": (sens + spec) / 2}


def stick_plot(ax, x, y, color, lw=1.0):
    ax.vlines(x, 0, y, color=color, lw=lw)

# %% [markdown]
# ## 1. Load data and describe the dataset structure
#
# Hierarchy: **mouse > sampling region > burn > spectrum**. One data file corresponds to one continuous
# burn; consecutive spectra (1 s scans) from the same burn are not independent observations.
# `sample_id` is the mouse number for skin spectra; background acquisitions carry their own identifiers.

# %%
npz = np.load(f"{DATA}/reims_peakpicked_spectra.npz")
X_all, mz = npz["X"].astype(float), npz["mz"]
meta = pd.read_csv(f"{DATA}/reims_spectrum_metadata.csv", dtype={"sample_id": str})
assert len(meta) == X_all.shape[0] and len(mz) == X_all.shape[1]

print(f"{X_all.shape[0]} spectra x {X_all.shape[1]} consensus peaks, m/z {mz.min():.2f}-{mz.max():.2f}")
print(pd.crosstab(meta.tissue, meta.device))

is_tissue = meta.tissue.isin(["tumor", "non_tumor"]).values
burns = meta[is_tissue].groupby("burn_file").size()
FACTS["n_spectra_total"] = int(len(meta))
FACTS["n_spectra_by_tissue"] = meta.tissue.value_counts().to_dict()
FACTS["n_mice"] = int(meta[is_tissue].sample_id.nunique())
FACTS["n_burns"] = int(len(burns))
FACTS["spectra_per_burn"] = {"mean": round(float(burns.mean()), 2), "min": int(burns.min()), "max": int(burns.max())}
print(f"\n{FACTS['n_mice']} mice, {len(burns)} burns, {int(is_tissue.sum())} tumor/non-tumor spectra; "
      f"spectra per burn: mean {burns.mean():.1f}, range {burns.min()}-{burns.max()}")

# %% [markdown]
# ## 2. Outlier detection
#
# All 933 spectra (tumor, non-tumor and background) are TIC-normalised, log-transformed and autoscaled. A 10-component PCA gives
# Hotelling's T2 and Q residuals; spectra at or above the 95th percentile of either statistic, computed
# within each tissue x device group, are flagged as outliers and removed.

# %%
Z = np.nan_to_num(StandardScaler().fit_transform(tic_log(X_all)))
pca_out = PCA(n_components=10, random_state=SEED)
scores = pca_out.fit_transform(Z)
t2 = (scores ** 2 / np.maximum(pca_out.explained_variance_, 1e-10)).sum(axis=1)
q_res = ((Z - pca_out.inverse_transform(scores)) ** 2).sum(axis=1)

outlier = np.zeros(len(meta), dtype=bool)
for stat in (t2, q_res):
    for _, idx in meta.groupby(["tissue", "device"]).indices.items():
        if len(idx) >= 3:
            outlier[idx] |= stat[idx] >= np.percentile(stat[idx], 95)
meta["outlier"] = outlier
FACTS["n_outliers"] = int(outlier.sum())
FACTS["n_outliers_by_tissue"] = meta.groupby("tissue").outlier.sum().astype(int).to_dict()
print(f"{outlier.sum()} outliers:", FACTS["n_outliers_by_tissue"])

fig, ax = plt.subplots(figsize=(6.5, 2.6))
ax.scatter(scores[~outlier, 0], scores[~outlier, 1], s=6, c="#1f77b4", alpha=0.6, linewidths=0, label=f"Inliers (n = {(~outlier).sum()})")
ax.scatter(scores[outlier, 0], scores[outlier, 1], s=9, c="#d62728", alpha=0.9, linewidths=0, label=f"Outliers (n = {outlier.sum()})")
ev = pca_out.explained_variance_ratio_ * 100
ax.set_xscale("symlog", linthresh=20); ax.set_yscale("symlog", linthresh=20)      # a few spectra lie far from the bulk
ax.set_xlabel(f"PC1 ({ev[0]:.1f}%), symmetric log scale"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/FigureSM1_outliers.png"); plt.close(fig)

# %% [markdown]
# ## 3. Training / hold-out split at the mouse level
#
# The 12 mice were randomly allocated to training (8) and hold-out (4) sets, and the five background
# acquisitions likewise (4 / 1). The allocation is fixed here so that it is identical to the one used in
# the paper. All spectra of a mouse (all burns, devices and tissue labels) are in the same set.

# %%
kept = meta[~meta.outlier]
test_ids = ["340", "379", "385", "393", "Background4"]
train_ids = sorted(set(meta.sample_id) - set(test_ids))
meta["set"] = np.where(meta.sample_id.isin(test_ids), "hold-out", "training")

mouse_ids = sorted(meta[is_tissue].sample_id.unique())
FACTS["training_mice"] = [m for m in mouse_ids if m in set(train_ids)]
FACTS["holdout_mice"] = [m for m in mouse_ids if m in set(test_ids)]
FACTS["spectra_after_outlier_removal"] = {f"{s}|{t}": int(v) for (s, t), v in kept.assign(set=meta.set[~meta.outlier]).groupby(["set", "tissue"]).size().items()}
print("Training mice:", FACTS["training_mice"]); print("Hold-out mice:", FACTS["holdout_mice"])
print(pd.crosstab(meta[~meta.outlier].set, meta[~meta.outlier].tissue))

# %% [markdown]
# ### Supplementary table: dataset structure per mouse

# %%
rows = []
for mouse, g in meta[is_tissue].groupby("sample_id"):
    row = {"Mouse ID": mouse, "Set": g.set.iloc[0].capitalize()}
    for d in DEVICES:
        for t, short in (("tumor", "T"), ("non_tumor", "NT")):
            s = g[(g.device == d) & (g.tissue == t)]
            row[f"{DEVICE_NAME[d]} | {short} | burns"] = s.burn_file.nunique()
            row[f"{DEVICE_NAME[d]} | {short} | spectra"] = len(s)
            row[f"{DEVICE_NAME[d]} | {short} | spectra after outlier removal"] = int((~s.outlier).sum())
    row["Total burns"] = g.burn_file.nunique(); row["Total spectra"] = len(g)
    row["Total spectra after outlier removal"] = int((~g.outlier).sum())
    rows.append(row)
per_mouse = pd.DataFrame(rows)
total = {"Mouse ID": "Total", "Set": ""}
total.update({c: per_mouse[c].sum() for c in per_mouse.columns[2:]})
per_mouse = pd.concat([per_mouse, pd.DataFrame([total])], ignore_index=True)
per_mouse.to_csv(f"{OUT}/TableSM4_dataset_per_mouse.csv", index=False)
per_mouse[["Mouse ID", "Set", "Total burns", "Total spectra", "Total spectra after outlier removal"]]

# %%
# Working matrices after outlier removal
keep_rows = ~meta.outlier.values
M = meta[keep_rows].reset_index(drop=True)
X = X_all[keep_rows]
train = (M.set == "training").values
test = ~train

# %% [markdown]
# ## 4. Feature selection (training set only)
#
# * Classes: tumor, non-tumor and background.
# * TIC normalisation and log(1 + x) on all peaks; prevalence filter (>= 30 % in at least one class).
# * Ranking: multinomial logistic regression, importance = mean absolute coefficient across classes,
#   averaged over leave-one-sample-out models; features must be significant (one-way ANOVA across
#   classes, Benjamini-Hochberg FDR <= 0.05); score = importance x (1 - q).
# * Nested cross-validation: in each outer fold (one mouse or one background acquisition left out) the
#   ranking is recomputed on the fold's training data only and models with the top-N features are tested
#   on the left-out sample.

# %%
fs_rows = train & M.tissue.isin(["tumor", "non_tumor", "background"]).values
y_fs, g_fs = M.tissue.values[fs_rows], M.sample_id.values[fs_rows]
Xp = tic_log(X[fs_rows])
prev = prevalence_mask(Xp, y_fs, 30)
Xs = StandardScaler().fit_transform(Xp[:, prev])          # scaled once on the training set, as in the original analysis
mz_fs = mz[prev]
FACTS["feature_selection"] = {"n_spectra": int(fs_rows.sum()), "n_features_after_prevalence": int(prev.sum()),
                              "n_features_after_prevalence_tumor_vs_non_tumor_only": int(prevalence_mask(
                                  tic_log(X[train & M.tissue.isin(['tumor', 'non_tumor']).values]),
                                  M.tissue.values[train & M.tissue.isin(['tumor', 'non_tumor']).values], 30).sum())}
print(FACTS["feature_selection"])

logo = LeaveOneGroupOut()
outer = list(logo.split(Xs, y_fs, g_fs))

# Final ranking on the complete training set
fold_importance = np.array([lr_importance(logreg().fit(Xs[a], y_fs[a])) for a, _ in outer])
importance, importance_sd = fold_importance.mean(axis=0), fold_importance.std(axis=0)
q_anova = bh_fdr(class_pvalues(Xs, y_fs))
score = np.where(q_anova <= 0.05, importance * (1 - q_anova), -1)
ranked = [i for i in np.argsort(-score, kind="stable") if score[i] >= 0]
FACTS["feature_selection"]["n_fdr_significant"] = int((q_anova <= 0.05).sum())

# %%
# Nested cross-validation sweep (takes a few minutes)
N_START, STEP = 500, 5
counts = sorted(set(list(range(N_START, 0, -STEP)) + [1]), reverse=True)
acc = {n: [] for n in counts}; bal = {n: [] for n in counts}; fold_ranks = []
for a, b in outer:
    m = logreg().fit(Xs[a], y_fs[a])
    q = bh_fdr(class_pvalues(Xs[a], y_fs[a]))
    s = np.where(q <= 0.05, lr_importance(m) * (1 - q), -1)
    order = [i for i in np.argsort(-s, kind="stable") if s[i] >= 0]
    fold_ranks.append(order)
    for n in counts:
        cols = order[:n]
        pred = logreg().fit(Xs[a][:, cols], y_fs[a]).predict(Xs[b][:, cols])
        acc[n].append((pred == y_fs[b]).mean()); bal[n].append(balanced_accuracy_score(y_fs[b], pred))
sweep = pd.DataFrame({"n_features": counts, "accuracy_mean": [np.mean(acc[n]) for n in counts],
                      "accuracy_sd": [np.std(acc[n]) for n in counts],
                      "balanced_accuracy_mean": [np.mean(bal[n]) for n in counts],
                      "balanced_accuracy_sd": [np.std(bal[n]) for n in counts]}).sort_values("n_features")
sweep.to_csv(f"{OUT}/feature_sweep_nested_cv.csv", index=False)

N_FEATURES = 170
row170 = sweep[sweep.n_features == N_FEATURES].iloc[0]
FACTS["feature_selection"].update({"n_outer_folds": len(outer), "cv_accuracy_at_170": round(float(row170.accuracy_mean), 4),
                                   "cv_balanced_accuracy_at_170": round(float(row170.balanced_accuracy_mean), 4)})
print(f"{len(outer)} outer folds; accuracy with {N_FEATURES} features: {row170.accuracy_mean:.3f} (balanced {row170.balanced_accuracy_mean:.3f})")

fig, ax = plt.subplots(figsize=(6.8, 2.6))
ax.fill_between(sweep.n_features, sweep.accuracy_mean - sweep.accuracy_sd, np.minimum(sweep.accuracy_mean + sweep.accuracy_sd, 1), color="#1f77b4", alpha=0.15, lw=0)
ax.plot(sweep.n_features, sweep.accuracy_mean, color="#1f77b4", lw=1.5, label="Accuracy (mean ± SD)")
ax.plot(sweep.n_features, sweep.balanced_accuracy_mean, color="#ff7f0e", lw=1.5, label="Balanced accuracy (mean)")
ax.axvline(N_FEATURES, color="#d62728", ls="--", lw=1); ax.text(N_FEATURES + 5, 0.42, f"{N_FEATURES} features", color="#d62728", fontsize=8)
ax.set_xlabel("Number of features"); ax.set_ylabel("Cross-validation score"); ax.set_ylim(0.35, 1.02); ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/FigureSM2_feature_sweep.png"); plt.close(fig)

# %%
# The 170-feature panel
panel_local = np.array(ranked[:N_FEATURES])                 # indices into the prevalence-filtered matrix
panel = np.where(prev)[0][panel_local]                      # indices into the full peak list
selection_frequency = np.array([sum(i in set(o[:N_FEATURES]) for o in fold_ranks) for i in panel_local])
full_model = logreg().fit(Xs, y_fs)
favoured = full_model.classes_[np.argmax(full_model.coef_[:, panel_local], axis=0)]
print("Selected m/z range: %.2f-%.2f; below m/z 550: %d" % (mz[panel].min(), mz[panel].max(), (mz[panel] < 550).sum()))

# %% [markdown]
# ## 5. Model comparison (training set, leave-one-sample-out cross-validation)
#
# LDA, logistic regression and random forest at prevalence thresholds of 10, 30 and 50 %.

# %%
def compare_models():
    out = []
    for thr in (10, 30, 50):
        mask = prevalence_mask(Xp, y_fs, thr)
        Xm = StandardScaler().fit_transform(Xp[:, mask])
        for name, make in (("LDA", lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                           ("Logistic regression", logreg),
                           ("Random forest", lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1))):
            fold_acc, yt, yp = [], [], []
            for a, b in outer:
                pred = make().fit(Xm[a], y_fs[a]).predict(Xm[b])
                fold_acc.append((pred == y_fs[b]).mean()); yt.append(y_fs[b]); yp.append(pred)
            out.append({"Prevalence threshold (%)": thr, "Features (n)": int(mask.sum()), "Model": name,
                        "Mean accuracy": round(float(np.mean(fold_acc)), 3), "SD": round(float(np.std(fold_acc)), 3),
                        "Cohen's kappa": round(float(cohen_kappa_score(np.concatenate(yt), np.concatenate(yp))), 3)})
    return pd.DataFrame(out)

model_comparison = compare_models()
model_comparison.to_csv(f"{OUT}/TableSM1_model_comparison.csv", index=False)
model_comparison

# %% [markdown]
# ## 6. Device-independent model
#
# Binary logistic regression (tumor vs macroscopically non-tumor skin) on the 170 selected features:
# TIC normalisation within the selected features, log(1 + x), autoscaling fitted on the training spectra.
# Predictions with a maximum class probability below 0.80 are treated as unclassified.

# %%
THRESHOLD = 0.80
tissue2 = M.tissue.isin(["tumor", "non_tumor"]).values
tr2, te2 = train & tissue2, test & tissue2
cols = np.sort(panel)
A, B = tic_log(X[tr2][:, cols]), tic_log(X[te2][:, cols])
y_tr, y_te = M.tissue.values[tr2], M.tissue.values[te2]
scaler = StandardScaler().fit(A)
final_model = logreg().fit(scaler.transform(A), y_tr)
proba = final_model.predict_proba(scaler.transform(B))
y_pred = final_model.classes_[proba.argmax(axis=1)]
confident = proba.max(axis=1) >= THRESHOLD

FACTS["device_independent_model"] = {"n_training_spectra": int(tr2.sum()), "n_holdout_spectra": int(te2.sum()),
                                     "coverage": [int(confident.sum()), int(len(confident))],
                                     "confident_only": binary_metrics(y_te[confident], y_pred[confident]),
                                     "all_spectra": binary_metrics(y_te, y_pred)}
print(json.dumps(FACTS["device_independent_model"], indent=1))

dev_te = M.device.values[te2]
by_device = []
for d in DEVICES:
    s = dev_te == d
    by_device.append({"Device": DEVICE_NAME[d], "Hold-out spectra (n)": int(s.sum()), "Coverage (%)": round(confident[s].mean() * 100, 1),
                      **{f"{k} (thresholded, %)": round(v * 100, 1) for k, v in binary_metrics(y_te[s & confident], y_pred[s & confident]).items() if k in ("accuracy", "sensitivity", "specificity")},
                      "accuracy (all spectra, %)": round(binary_metrics(y_te[s], y_pred[s])["accuracy"] * 100, 1)})
pd.DataFrame(by_device).to_csv(f"{OUT}/device_independent_model_by_device.csv", index=False)
pd.DataFrame(by_device)

# %% [markdown]
# ### 6b. Sensitivity checks for optimistic bias
#
# * **Outliers in the hold-out set.** Outlier removal used the tissue labels (percentiles within tissue x device
#   groups) and was applied to all spectra, including the hold-out mice. Here the final model is also applied to
#   the hold-out spectra that were removed as outliers.
# * **Burn level.** Spectra from one burn are correlated; each burn is classified by its mean tumor probability.
# * **Mouse level.** Accuracy for each of the four hold-out mice.
# * **Uncertainty.** Wilson 95 % intervals at spectrum level (these ignore within-burn correlation and are too narrow).

# %%
def wilson(k, n, z=1.96):
    p = k / n; c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return [round(m - c, 4), round(m + c, 4)]

te_all = (meta.set == "hold-out").values & meta.tissue.isin(["tumor", "non_tumor"]).values
p_all = final_model.predict_proba(scaler.transform(tic_log(X_all[te_all][:, cols])))
pred_all = final_model.classes_[p_all.argmax(axis=1)]; y_all = meta.tissue.values[te_all]
conf_all = p_all.max(axis=1) >= THRESHOLD
out_te = meta.outlier.values[te_all]
bias = {"including_outliers_all": binary_metrics(y_all, pred_all),
        "including_outliers_confident": binary_metrics(y_all[conf_all], pred_all[conf_all]),
        "including_outliers_coverage": [int(conf_all.sum()), int(len(conf_all))],
        "outliers_only_all": binary_metrics(y_all[out_te], pred_all[out_te]),
        "wilson_95ci_accuracy_confident": wilson(int((y_pred[confident] == y_te[confident]).sum()), int(confident.sum())),
        "wilson_95ci_accuracy_all": wilson(int((y_pred == y_te).sum()), len(y_te))}
burn_te = M.burn_file.values[te2]
votes = pd.DataFrame({"burn": burn_te, "true": y_te, "tumor_prob": proba[:, list(final_model.classes_).index("tumor")]})
burn_level = votes.groupby("burn").agg(true=("true", "first"), tumor_prob=("tumor_prob", "mean"))
burn_level["pred"] = np.where(burn_level.tumor_prob >= 0.5, "tumor", "non_tumor")
bias["burn_level_mean_probability"] = binary_metrics(burn_level.true.values, burn_level.pred.values)
mouse_te = M.sample_id.values[te2]
bias["per_mouse_accuracy_all"] = {m: round(float((y_pred[mouse_te == m] == y_te[mouse_te == m]).mean()), 4) for m in sorted(set(mouse_te))}
FACTS["bias_checks"] = bias
print(json.dumps(bias, indent=1))

# %% [markdown]
# ## 7. Intra- and cross-device models
#
# For every pair of devices a model is trained on the training-mouse spectra of one device and tested on
# the hold-out-mouse spectra of the target device (diagonal = same device). Features: 30 % prevalence
# filter fitted on the training subset (the 170-feature panel is not used here). Logistic regression
# with balanced class weights; metrics at the 80 % confidence threshold and for all spectra.

# %%
def device_pair(train_device, target_device):
    a = tr2 & (M.device == train_device).values
    b = te2 & (M.device == target_device).values
    Xa, Xb, ya, yb = tic_log(X[a]), tic_log(X[b]), M.tissue.values[a], M.tissue.values[b]
    keep = prevalence_mask(Xa, ya, 30)
    sc = StandardScaler().fit(Xa[:, keep])
    model = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xa[:, keep]), ya)
    p = model.predict_proba(sc.transform(Xb[:, keep]))
    pred = model.classes_[p.argmax(axis=1)]
    conf = p.max(axis=1) >= THRESHOLD
    thr, raw = binary_metrics(yb[conf], pred[conf]), binary_metrics(yb, pred)
    return {"Training device": DEVICE_NAME[train_device], "Target device": DEVICE_NAME[target_device],
            "Training spectra (n)": int(a.sum()), "Target spectra (n)": int(b.sum()), "Features (n)": int(keep.sum()),
            "Coverage (%)": round(conf.mean() * 100, 1),
            "Balanced accuracy, thresholded (%)": round(thr["balanced_accuracy"] * 100, 1),
            "Sensitivity, thresholded (%)": round(thr["sensitivity"] * 100, 1),
            "Specificity, thresholded (%)": round(thr["specificity"] * 100, 1),
            "Balanced accuracy, all spectra (%)": round(raw["balanced_accuracy"] * 100, 1),
            "Sensitivity, all spectra (%)": round(raw["sensitivity"] * 100, 1),
            "Specificity, all spectra (%)": round(raw["specificity"] * 100, 1)}

cross = pd.DataFrame([device_pair(a, b) for a in DEVICES for b in DEVICES])
cross.to_csv(f"{OUT}/TableSM6_cross_device.csv", index=False)
names = [DEVICE_NAME[d] for d in DEVICES]
print(cross.pivot(index="Training device", columns="Target device", values="Balanced accuracy, thresholded (%)").loc[names, names])
print(cross.pivot(index="Training device", columns="Target device", values="Coverage (%)").loc[names, names])

# %% [markdown]
# ## 8. Supplementary analyses
# ### 8.1 Unsupervised PCA of all tumor and non-tumor spectra

# %%
T = M[tissue2].reset_index(drop=True)
XT = X[tissue2]
XTp = tic_log(XT)
keep_pca = prevalence_mask(XTp, T.tissue.values, 30)
pca = PCA(n_components=10, random_state=SEED)
S = pca.fit_transform(StandardScaler().fit_transform(XTp[:, keep_pca]))
evr = pca.explained_variance_ratio_ * 100

def eta_squared(values, groups):
    grand = values.mean()
    return float(sum((groups == g).sum() * (values[groups == g].mean() - grand) ** 2 for g in np.unique(groups)) / ((values - grand) ** 2).sum())

FACTS["pca"] = {"n_spectra": int(len(T)), "n_features": int(keep_pca.sum()), "explained_variance_percent": evr[:5].round(1).tolist(),
                "eta_squared": {f"PC{i + 1}": {k: round(eta_squared(S[:, i], T[c].values), 2) for k, c in (("tissue", "tissue"), ("device", "device"), ("mouse", "sample_id"))} for i in range(3)}}
print(json.dumps(FACTS["pca"], indent=1))

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.9), sharex=True, sharey=True)
kw = dict(s=7, alpha=0.75, linewidths=0)
for t in ("non_tumor", "tumor"):
    s = (T.tissue == t).values; axs[0].scatter(S[s, 0], S[s, 1], c=TISSUE_COLOR[t], label=TISSUE_NAME[t], **kw)
short = {"SP_CO2": "SP CO$_2$", "P_CO2": "P CO$_2$", "Cautery": "Cautery", "N2_plasma": "N$_2$ plasma"}
for d in DEVICES:
    s = (T.device == d).values; axs[1].scatter(S[s, 0], S[s, 1], c=DEVICE_COLOR[d], label=short[d], **kw)
cmap = plt.get_cmap("tab20")
for i, mouse in enumerate(sorted(T.sample_id.unique())):
    s = (T.sample_id == mouse).values; axs[2].scatter(S[s, 0], S[s, 1], color=cmap(i % 20), label=mouse, **kw)
for ax, title in zip(axs, ("Tissue class", "Device", "Mouse")):
    ax.set_title(title, fontsize=10); ax.set_xlabel(f"PC1 ({evr[0]:.1f}%)")
    ax.axhline(0, color="0.85", lw=0.5, zorder=0); ax.axvline(0, color="0.85", lw=0.5, zorder=0)
axs[0].set_ylabel(f"PC2 ({evr[1]:.1f}%)")
axs[0].legend(frameon=False, fontsize=7, markerscale=2, handletextpad=0.1); axs[1].legend(frameon=False, fontsize=7, markerscale=2, handletextpad=0.1)
axs[2].legend(frameon=False, fontsize=5.8, markerscale=2, ncol=3, handletextpad=0.05, columnspacing=0.4, title="Mouse ID", title_fontsize=6)
fig.tight_layout(w_pad=0.6); fig.savefig(f"{OUT}/FigureSM4_PCA.png"); plt.close(fig)

# %% [markdown]
# ### 8.2 Feature-level table for the 170 selected features
#
# Per-device statistics are descriptive and use all tumor and non-tumor spectra after outlier removal
# (training and hold-out mice). Annotations are tentative: they are based on *m/z* only (within 0.02 Da
# of the [M-H]- ion of common fatty acids and glycerophospholipids), without MS/MS or reference standards.

# %%
CANDIDATES = {"FA 14:0": 227.2017, "FA 15:0": 241.2173, "FA 16:1": 253.2173, "FA 16:0": 255.2330, "FA 17:0": 269.2486,
              "FA 18:2": 279.2330, "FA 18:1": 281.2486, "FA 18:0": 283.2643, "FA 20:5": 301.2173, "FA 20:4": 303.2330,
              "FA 20:3": 305.2486, "FA 20:2": 307.2643, "FA 20:1": 309.2799, "FA 20:0": 311.2956, "FA 22:6": 327.2330,
              "FA 22:5": 329.2486, "FA 22:4": 331.2643, "FA 22:1": 337.3112, "FA 22:0": 339.3269, "FA 24:1": 365.3425,
              "FA 24:0": 367.3582, "FA 26:1": 393.3738, "FA 26:0": 395.3895, "PA 34:1": 673.4814, "PA 36:2": 699.4970,
              "PE 34:1": 716.5236, "PE 36:2": 742.5392, "PE 36:1": 744.5549, "PG 34:1": 747.5182, "PE 38:4": 766.5392,
              "PG 36:2": 773.5338, "PS 36:1": 788.5447, "PS 38:4": 810.5291, "PI 36:2": 861.5499, "PI 38:4": 885.5499}

pooled_diff = XTp[T.tissue.values == "tumor"][:, panel].mean(axis=0) - XTp[T.tissue.values == "non_tumor"][:, panel].mean(axis=0)
q_tumor = bh_fdr(stats.ttest_ind(Xs[y_fs == "tumor"], Xs[y_fs == "non_tumor"], equal_var=False).pvalue)
features = pd.DataFrame({
    "Rank": np.arange(1, N_FEATURES + 1), "m/z": mz[panel].round(4),
    "Importance (mean |coefficient|)": importance[panel_local].round(4), "Importance SD": importance_sd[panel_local].round(4),
    "Class favoured by model": [TISSUE_NAME[c] for c in favoured],
    "FDR q (ANOVA, three classes)": q_anova[panel_local], "FDR q (tumor vs non-tumor)": q_tumor[panel_local],
    f"Selection frequency (of {len(outer)} outer folds)": selection_frequency,
    "Direction (all devices pooled)": np.where(pooled_diff > 0, "Higher in tumor", "Higher in non-tumor")})
concordant = np.zeros(N_FEATURES, dtype=int)
for d in DEVICES:
    s = (T.device == d).values
    yd, Zd = T.tissue.values[s], XTp[s][:, panel]
    features[f"{DEVICE_NAME[d]}: detected (% of spectra)"] = ((XT[s][:, panel] > 0).mean(axis=0) * 100).round(0).astype(int)
    qd = bh_fdr(stats.ttest_ind(Zd[yd == "tumor"], Zd[yd == "non_tumor"], equal_var=False).pvalue)
    diff = Zd[yd == "tumor"].mean(axis=0) - Zd[yd == "non_tumor"].mean(axis=0)
    sig = qd <= 0.05
    features[f"{DEVICE_NAME[d]}: tumor vs non-tumor"] = np.where(~sig, "n.s.", np.where(diff > 0, "higher", "lower"))
    concordant += sig & (np.sign(diff) == np.sign(pooled_diff))
features["Devices with significant, concordant difference (of 4)"] = concordant
annotation, delta = [], []
for value in mz[panel]:
    name, theo = min(CANDIDATES.items(), key=lambda kv: abs(kv[1] - value))
    hit = abs(value - theo) <= 0.020
    annotation.append(name if hit else ""); delta.append(round((value - theo) * 1000, 1) if hit else np.nan)
features["Tentative annotation ([M-H]-, by m/z only)"] = annotation
features["Mass difference (mDa)"] = delta
features.to_csv(f"{OUT}/TableSM5_170_features.csv", index=False)

# Single-device rankings: how much do they overlap with the pooled panel?
single_top = {}
for d in DEVICES:
    s = tr2 & (M.device == d).values
    yd, Zd = M.tissue.values[s], tic_log(X[s])
    k = prevalence_mask(Zd, yd, 30)
    Zs = StandardScaler().fit_transform(Zd[:, k])
    qd = bh_fdr(stats.ttest_ind(Zs[yd == "tumor"], Zs[yd == "non_tumor"], equal_var=False).pvalue)
    sc = np.where(qd <= 0.05, lr_importance(logreg().fit(Zs, yd)) * (1 - qd), -1)
    top = np.argsort(-sc)[:N_FEATURES]
    single_top[d] = set(np.where(k)[0][top[sc[top] >= 0]])
in_n = pd.Series([f for d in DEVICES for f in single_top[d]]).value_counts()

FACTS["features"] = {"favoured_class": features["Class favoured by model"].value_counts().to_dict(),
                     "n_below_mz_550": int((mz[panel] < 550).sum()), "mz_range": [round(float(mz[panel].min()), 2), round(float(mz[panel].max()), 2)],
                     "selected_in_all_folds": int((selection_frequency == len(outer)).sum()),
                     "selected_in_at_least_9_folds": int((selection_frequency >= 9).sum()),
                     "concordant_in_4_devices": int((concordant == 4).sum()), "concordant_in_at_least_3": int((concordant >= 3).sum()),
                     "concordant_in_0": int((concordant == 0).sum()),
                     "overlap_single_device_top170_with_panel": {DEVICE_NAME[d]: len(single_top[d] & set(panel)) for d in DEVICES},
                     "features_shared_by_all_four_single_device_rankings": int((in_n == 4).sum()),
                     "tentative_annotations": features.loc[features.iloc[:, -2] != "", ["Rank", "m/z", "Tentative annotation ([M-H]-, by m/z only)", "Mass difference (mDa)"]].values.tolist()}
print(json.dumps(FACTS["features"], indent=1))

fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [2.3, 1, 1]})
for name, key in (("Non-tumor", "non_tumor"), ("Tumor", "tumor"), ("Background", "background")):
    s = features["Class favoured by model"] == name
    axs[0].vlines(features.loc[s, "m/z"], 0, features.loc[s, "Importance (mean |coefficient|)"], color=TISSUE_COLOR[key], lw=1.0, label=f"{name} (n = {s.sum()})")
axs[0].set_xlabel("$m/z$"); axs[0].set_ylabel("Importance (mean |coefficient|)"); axs[0].set_xlim(190, 760); axs[0].set_ylim(0, 0.125)
axs[0].legend(frameon=False, fontsize=6.5, title="Class favoured by model", title_fontsize=6.5, loc="upper right", handlelength=1.2)
axs[1].bar(*np.unique(selection_frequency, return_counts=True), color="#555555", width=0.8)
axs[1].set_xlabel("Outer folds in which\nfeature ranked in top 170"); axs[1].set_ylabel("Features (n)"); axs[1].set_xticks(range(0, 13, 2))
axs[2].bar(*np.unique(concordant, return_counts=True), color="#555555", width=0.7)
axs[2].set_xlabel("Devices with significant,\nconcordant difference"); axs[2].set_ylabel("Features (n)"); axs[2].set_xticks(range(5))
for ax, letter in zip(axs, "ABC"):
    ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontweight="bold", fontsize=11, ha="right")
fig.tight_layout(w_pad=1.0); fig.savefig(f"{OUT}/FigureSM5_features.png"); plt.close(fig)

# %% [markdown]
# ### 8.3 Mean spectra per device and tissue class

# %%
def tic(Xm):
    s = Xm.sum(axis=1, keepdims=True); s[s == 0] = 1
    return Xm / s

XTn = tic(XT)
# Main-text figure: mean spectrum per device (tumor and non-tumor pooled), separate panels,
# each device scaled so that its most intense peak in the displayed range is 1
lo, hi = 525, 905
w = (mz >= lo) & (mz <= hi)
means = {d: XTn[(T.device == d).values].mean(axis=0) for d in DEVICES}
fig, axs = plt.subplots(4, 1, figsize=(7.2, 5.4), sharex=True, sharey=True)
for ax, d in zip(axs, DEVICES):
    stick_plot(ax, mz[w], means[d][w] / means[d][w].max(), DEVICE_COLOR[d], 1.1)
    ax.set_ylim(0, 1.05); ax.set_xlim(lo, hi); ax.set_yticks([0, 0.5, 1])
    ax.text(0.99, 0.9, f"{DEVICE_LABEL[d]}  (n = {int((T.device == d).sum())} spectra)", transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=DEVICE_COLOR[d], fontweight="bold")
axs[-1].set_xlabel("$m/z$"); fig.supylabel("Scaled intensity", fontsize=9.5, x=0.02)
fig.tight_layout(h_pad=0.6); fig.savefig(f"{OUT}/Figure3_mean_spectra_by_device.png"); plt.close(fig)

# Supplementary figure: non-tumor and tumor spectra for each device
lo, hi = 200, 1000
w = (mz >= lo) & (mz <= hi); wi = (mz >= 550) & (mz <= 900)
fig, axs = plt.subplots(4, 2, figsize=(7.2, 7.4), sharex=True)
signal = []
for i, d in enumerate(DEVICES):
    mm = {t: XTn[((T.device == d) & (T.tissue == t)).values].mean(axis=0) for t in ("non_tumor", "tumor")}
    top = max(mm["non_tumor"][w].max(), mm["tumor"][w].max())
    for j, (t, title) in enumerate((("non_tumor", "Macroscopically non-tumor skin"), ("tumor", "Tumor (SCC)"))):
        s = ((T.device == d) & (T.tissue == t)).values
        ax = axs[i, j]
        stick_plot(ax, mz[w], mm[t][w] / top * 100, TISSUE_COLOR[t], 0.9)
        ax.set_ylim(0, 185); ax.set_yticks([0, 50, 100]); ax.set_xlim(lo, hi)
        ax.text(0.02, 0.95, f"n = {s.sum()} spectra\n{T.burn_file[s].nunique()} burns", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
        if i == 0: ax.set_title(title, fontsize=10, color=TISSUE_COLOR[t], fontweight="bold")
        if j == 0: ax.set_ylabel(DEVICE_LABEL[d] + "\nRel. intensity (%)", fontsize=8.5)
        ins = ax.inset_axes([0.50, 0.63, 0.48, 0.30])
        stick_plot(ins, mz[wi], mm[t][wi] / top * 100, TISSUE_COLOR[t], 0.7)
        ins.set_xlim(550, 900); ins.set_ylim(0, max(mm["non_tumor"][wi].max(), mm["tumor"][wi].max()) / top * 105)
        ins.tick_params(labelsize=6.5, length=2); ins.text(0.98, 0.95, "$m/z$ 550–900", transform=ins.transAxes, ha="right", va="top", fontsize=6.5)
        signal.append({"Device": DEVICE_NAME[d], "Tissue": TISSUE_NAME[t], "Spectra (n)": int(s.sum()), "Burns (n)": int(T.burn_file[s].nunique()),
                       "Median total ion count": float(np.median(XT[s].sum(axis=1))), "Median detected peaks per spectrum": float(np.median((XT[s] > 0).sum(axis=1)))})
for ax in axs[-1]: ax.set_xlabel("$m/z$")
fig.tight_layout(h_pad=0.5, w_pad=1.0); fig.savefig(f"{OUT}/FigureSM3_non_tumor_and_tumor_spectra.png"); plt.close(fig)
signal = pd.DataFrame(signal); signal.to_csv(f"{OUT}/signal_summary.csv", index=False)
signal

# %% [markdown]
# ### 8.4 Spectral richness
#
# Uses the burn-level intensity matrix exported by MS IonVision (`X_burn`: peak intensities extracted
# from the average spectrum of each burn). For each device the tumor and non-tumor burns are averaged;
# a consensus peak is counted when its mean intensity is at least 5 % of the most intense peak.

# %%
X_burn = npz["X_burn"].astype(float)
burn_meta = pd.read_csv(f"{DATA}/reims_burn_metadata.csv", dtype={"sample_id": str})
richness = []
for d in DEVICES:
    s = ((burn_meta.device == d) & burn_meta.tissue.isin(["tumor", "non_tumor"])).values
    avg = X_burn[s].mean(axis=0)
    peak = avg >= 0.05 * avg.max()
    richness.append({"Device": DEVICE_NAME[d], "Burns (n)": int(s.sum()), "Peaks (n)": int(peak.sum()),
                     "Peaks m/z < 550": int((peak & (mz < 550)).sum()), "Peaks m/z >= 550": int((peak & (mz >= 550)).sum())})
richness = pd.DataFrame(richness); richness.to_csv(f"{OUT}/spectral_richness.csv", index=False)
FACTS["spectral_richness"] = richness.to_dict(orient="records")
richness

# %%
with open(f"{OUT}/key_numbers.json", "w") as fh:
    json.dump(FACTS, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("Done. Tables and figures are in", OUT)
