import os
import glob
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SUMMARY_DIR = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_data/summary"
RESULTS_DIR = "/data/zshao/rllm/ArcticInference/tests/extracted_problem_data/results/timeseries_analysis"


def load_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = ["current_step", "avg_accept_toks", "avg_score", "avg_angle"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")
    # Ensure sorted by time
    df = df.sort_values("current_step").reset_index(drop=True)
    return df[["current_step", "avg_accept_toks", "avg_score", "avg_angle"]]


def corr_pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return np.nan
    return x.corr(y, method="pearson")


def corr_spearman(x: pd.Series, y: pd.Series) -> float:
    """Compute Spearman correlation without SciPy by ranking then Pearson."""
    if len(x) < 3:
        return np.nan
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    return xr.corr(yr, method="pearson")


def pearson_spearman(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    return (corr_pearson(x, y), corr_spearman(x, y))


def lag_corr(x: pd.Series, y: pd.Series, max_lag: int = 10) -> Dict[str, Tuple[int, float]]:
    """Return lag k (x leads y positive k) with max absolute Pearson corr and the corr value.

    Positive lag means x(t-k) vs y(t), i.e., x leads y by k steps.
    """
    best = {"pearson": (0, np.nan), "spearman": (0, np.nan)}
    n = len(x)
    if n < 5:
        return best
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            xk, yk = x, y
        elif k > 0:
            xk, yk = x.iloc[:-k], y.iloc[k:]
        else:  # k < 0 => y leads
            xk, yk = x.iloc[-k:], y.iloc[:k]
        if len(xk) < 3:
            continue
        p = corr_pearson(xk, yk)
        s = corr_spearman(xk, yk)
        if not np.isnan(p):
            if np.isnan(best["pearson"][1]) or abs(p) > abs(best["pearson"][1]):
                best["pearson"] = (k, p)
        if not np.isnan(s):
            if np.isnan(best["spearman"][1]) or abs(s) > abs(best["spearman"][1]):
                best["spearman"] = (k, s)
    return best


def analyze_file(path: str, max_lag: int = 10) -> Dict:
    df = load_series(path)
    x = df["avg_accept_toks"]
    score = df["avg_score"]
    angle = df["avg_angle"]

    p_s, s_s = pearson_spearman(x, score)
    p_a, s_a = pearson_spearman(x, angle)

    l_s = lag_corr(x, score, max_lag=max_lag)
    l_a = lag_corr(x, angle, max_lag=max_lag)

    return {
        "file": os.path.basename(path),
        "n": len(df),
        "pearson_accept_score": p_s,
        "spearman_accept_score": s_s,
        "pearson_accept_angle": p_a,
        "spearman_accept_angle": s_a,
        "lag_accept_score": l_s,
        "lag_accept_angle": l_a,
    }


def zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def ccf_max_abs(x: pd.Series, y: pd.Series, max_lag: int = 15) -> Tuple[int, float]:
    xz = zscore(x)
    yz = zscore(y)
    best_k, best_c = 0, np.nan
    for k in range(-max_lag, max_lag + 1):
        if k == 0:
            xk, yk = xz, yz
        elif k > 0:
            xk, yk = xz.iloc[:-k], yz.iloc[k:]
        else:
            xk, yk = xz.iloc[-k:], yz.iloc[:k]
        if len(xk) < 3:
            continue
        c = corr_pearson(xk, yk)
        if not np.isnan(c):
            if np.isnan(best_c) or abs(c) > abs(best_c):
                best_k, best_c = k, c
    return best_k, best_c


def dtw_distance(a: np.ndarray, b: np.ndarray, window: int) -> float:
    n, m = len(a), len(b)
    w = max(window, abs(n - m))
    # Initialize with inf
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = (ai - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def dtw_with_path(a: np.ndarray, b: np.ndarray, window: int):
    n, m = len(a), len(b)
    w = max(window, abs(n - m))
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = (ai - b[j - 1]) ** 2
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    # Backtrack path
    i, j = n, m
    path = [(i, j)]
    while i > 0 or j > 0:
        choices = []
        if i > 0 and j > 0:
            choices.append((dtw[i - 1, j - 1], i - 1, j - 1))
        if i > 0:
            choices.append((dtw[i - 1, j], i - 1, j))
        if j > 0:
            choices.append((dtw[i, j - 1], i, j - 1))
        cost, i, j = min(choices, key=lambda t: t[0])
        path.append((i, j))
        if (i, j) == (0, 0):
            break
    path.reverse()
    return float(dtw[n, m]), dtw, path


def compute_ccf_curve(x: pd.Series, y: pd.Series, max_lag: int = 30):
    xz = zscore(x)
    yz = zscore(y)
    lags = list(range(-max_lag, max_lag + 1))
    corrs = []
    for k in lags:
        if k == 0:
            xk, yk = xz, yz
        elif k > 0:
            xk, yk = xz.iloc[:-k], yz.iloc[k:]
        else:
            xk, yk = xz.iloc[-k:], yz.iloc[:k]
        if len(xk) < 3:
            corrs.append(np.nan)
        else:
            corrs.append(corr_pearson(xk, yk))
    return np.array(lags), np.array(corrs)


def plot_ccf(file_base: str, x: pd.Series, score: pd.Series, angle: pd.Series, out_dir: str):
    lags_s, ccf_s = compute_ccf_curve(x, score, max_lag=30)
    lags_a, ccf_a = compute_ccf_curve(x, angle, max_lag=30)
    plt.figure(figsize=(8, 4))
    plt.plot(lags_s, ccf_s, label="CCF accept vs score")
    plt.plot(lags_a, ccf_a, label="CCF accept vs angle")
    plt.axvline(0, color="k", linewidth=0.8, linestyle="--")
    plt.xlabel("Lag (x leads > 0)")
    plt.ylabel("Correlation")
    plt.title(f"CCF curves: {file_base}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{file_base}_ccf.png"), dpi=150)
    plt.close()


def plot_dtw(file_base: str, x: pd.Series, score: pd.Series, angle: pd.Series, out_dir: str):
    zx = zscore(x).to_numpy()
    zs = zscore(score).to_numpy()
    za = zscore(angle).to_numpy()
    win = max(10, len(zx) // 20)
    d_s, dtw_s, path_s = dtw_with_path(zx, zs, window=win)
    d_a, dtw_a, path_a = dtw_with_path(zx, za, window=win)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(dtw_s[1:, 1:], origin="lower", aspect="auto", cmap="viridis")
    ps = np.array(path_s)
    axes[0].plot(ps[:, 1] - 1, ps[:, 0] - 1, color="w", linewidth=1.0)
    axes[0].set_title(f"DTW accept vs score\nDist={d_s:.1f}")
    axes[0].set_xlabel("score index")
    axes[0].set_ylabel("accept index")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(dtw_a[1:, 1:], origin="lower", aspect="auto", cmap="viridis")
    pa = np.array(path_a)
    axes[1].plot(pa[:, 1] - 1, pa[:, 0] - 1, color="w", linewidth=1.0)
    axes[1].set_title(f"DTW accept vs angle\nDist={d_a:.1f}")
    axes[1].set_xlabel("angle index")
    axes[1].set_ylabel("accept index")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{file_base}_dtw.png"), dpi=150)
    plt.close()


def granger_gain(target: pd.Series, predictor: pd.Series, maxlag: int = 3) -> float:
    y = target.astype(float).to_numpy()
    x = predictor.astype(float).to_numpy()
    n = len(y)
    p = maxlag
    if n <= p + 5:
        return np.nan
    # Build lag matrices
    Ys = y[p:]
    X_self = []
    X_both = []
    for t in range(p, n):
        row_self = [y[t - k] for k in range(1, p + 1)]
        row_pred = [x[t - k] for k in range(1, p + 1)]
        X_self.append(row_self)
        X_both.append(row_self + row_pred)
    X_self = np.asarray(X_self)
    X_both = np.asarray(X_both)
    # Add bias
    ones = np.ones((len(Ys), 1))
    X_self_b = np.concatenate([ones, X_self], axis=1)
    X_both_b = np.concatenate([ones, X_both], axis=1)
    # OLS via lstsq
    beta_self, *_ = np.linalg.lstsq(X_self_b, Ys, rcond=None)
    beta_both, *_ = np.linalg.lstsq(X_both_b, Ys, rcond=None)
    resid_self = Ys - X_self_b @ beta_self
    resid_both = Ys - X_both_b @ beta_both
    mse_self = float(np.mean(resid_self ** 2))
    mse_both = float(np.mean(resid_both ** 2))
    if mse_self <= 0:
        return np.nan
    return (mse_self - mse_both) / mse_self


def analyze_file_extended(path: str) -> Dict:
    df = load_series(path)
    x = df["avg_accept_toks"]
    score = df["avg_score"]
    angle = df["avg_angle"]

    # CCF
    k_s, c_s = ccf_max_abs(x, score, max_lag=15)
    k_a, c_a = ccf_max_abs(x, angle, max_lag=15)

    # DTW on z-scored series with window
    zx = zscore(x).to_numpy()
    zs = zscore(score).to_numpy()
    za = zscore(angle).to_numpy()
    win = max(10, len(zx) // 20)  # ~5% length, min 10
    d_s = dtw_distance(zx, zs, window=win)
    d_a = dtw_distance(zx, za, window=win)

    # Granger-style predictive gain
    g_s = granger_gain(x, score, maxlag=3)
    g_a = granger_gain(x, angle, maxlag=3)

    # Per-metric winner
    win_ccf = "score" if (not np.isnan(c_s) and (np.isnan(c_a) or abs(c_s) >= abs(c_a))) else "angle"
    win_dtw = "score" if (not np.isinf(d_s) and (np.isinf(d_a) or d_s <= d_a)) else "angle"
    win_granger = "score" if (not np.isnan(g_s) and (np.isnan(g_a) or g_s >= g_a)) else "angle"
    wins = {win_ccf, win_dtw, win_granger}
    overall = "score" if list(wins).count("score") >= 2 else "angle"

    return {
        "file": os.path.basename(path),
        "n": len(df),
        "ccf_k_score": k_s,
        "ccf_val_score": c_s,
        "ccf_k_angle": k_a,
        "ccf_val_angle": c_a,
        "dtw_score": d_s,
        "dtw_angle": d_a,
        "granger_gain_score": g_s,
        "granger_gain_angle": g_a,
        "winner_ccf": win_ccf,
        "winner_dtw": win_dtw,
        "winner_granger": win_granger,
        "overall_winner": overall,
    }


def aggregate(rows: List[Dict]) -> Dict[str, float]:
    def collect(key: str) -> List[float]:
        vals = [r[key] for r in rows]
        return [v for v in vals if not (v is None or np.isnan(v))]

    agg = {
        "mean_pearson_accept_score": np.nanmean(collect("pearson_accept_score")) if rows else np.nan,
        "median_pearson_accept_score": np.nanmedian(collect("pearson_accept_score")) if rows else np.nan,
        "mean_spearman_accept_score": np.nanmean(collect("spearman_accept_score")) if rows else np.nan,
        "median_spearman_accept_score": np.nanmedian(collect("spearman_accept_score")) if rows else np.nan,
        "mean_pearson_accept_angle": np.nanmean(collect("pearson_accept_angle")) if rows else np.nan,
        "median_pearson_accept_angle": np.nanmedian(collect("pearson_accept_angle")) if rows else np.nan,
        "mean_spearman_accept_angle": np.nanmean(collect("spearman_accept_angle")) if rows else np.nan,
        "median_spearman_accept_angle": np.nanmedian(collect("spearman_accept_angle")) if rows else np.nan,
    }
    # Leading/lagging summary (pearson only)
    lags_score = [r["lag_accept_score"]["pearson"][0] for r in rows if r["lag_accept_score"]["pearson"][1] == r["lag_accept_score"]["pearson"][1]]
    lags_angle = [r["lag_accept_angle"]["pearson"][0] for r in rows if r["lag_accept_angle"]["pearson"][1] == r["lag_accept_angle"]["pearson"][1]]
    agg["median_best_lag_accept_score"] = np.median(lags_score) if lags_score else np.nan
    agg["median_best_lag_accept_angle"] = np.median(lags_angle) if lags_angle else np.nan
    return agg


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(SUMMARY_DIR, "*.csv")))
    # Basic correlations
    results_basic: List[Dict] = []
    for p in paths:
        try:
            res = analyze_file(p, max_lag=15)
            results_basic.append(res)
        except Exception as e:
            print(f"[WARN] Skipping basic {os.path.basename(p)}: {e}")
    if results_basic:
        dfb = pd.DataFrame(results_basic)
        aggb = aggregate(results_basic)
        print("Per-file correlations (first 10):")
        print(dfb[[
            "file", "n",
            "pearson_accept_score", "spearman_accept_score",
            "pearson_accept_angle", "spearman_accept_angle",
        ]].head(10).to_string(index=False))
        print("\nAggregate correlations:")
        for k, v in aggb.items():
            print(f"{k}: {v}")

    # Extended metrics: CCF, DTW, Granger-like gain
    results_ext: List[Dict] = []
    for p in paths:
        try:
            res = analyze_file_extended(p)
            results_ext.append(res)
        except Exception as e:
            print(f"[WARN] Skipping extended {os.path.basename(p)}: {e}")
    if not results_ext:
        print("No extended results")
        return
    dfe = pd.DataFrame(results_ext)
    # Write CSV
    out_csv = os.path.join(RESULTS_DIR, "extended_metrics.csv")
    dfe.to_csv(out_csv, index=False)
    print(f"\nWrote extended metrics to: {out_csv}")
    print("\nExtended metrics (first 10):")
    print(dfe[[
        "file", "n",
        "ccf_k_score", "ccf_val_score", "ccf_k_angle", "ccf_val_angle",
        "dtw_score", "dtw_angle",
        "granger_gain_score", "granger_gain_angle",
        "winner_ccf", "winner_dtw", "winner_granger", "overall_winner"
    ]].head(10).to_string(index=False))

    # Overall winners count
    print("\nOverall winner counts:")
    print(dfe["overall_winner"].value_counts().to_string())

    # Generate per-file plots
    plots_dir = os.path.join(RESULTS_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for p in paths:
        try:
            df = load_series(p)
            base = os.path.splitext(os.path.basename(p))[0]
            plot_ccf(base, df["avg_accept_toks"], df["avg_score"], df["avg_angle"], plots_dir)
            plot_dtw(base, df["avg_accept_toks"], df["avg_score"], df["avg_angle"], plots_dir)
        except Exception as e:
            print(f"[WARN] Plotting failed for {os.path.basename(p)}: {e}")


if __name__ == "__main__":
    main()


