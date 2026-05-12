"""Methodological improvements: cross-validation, confidence baseline, bootstrap CIs.

Fixes the three gaps identified in the publication readiness assessment.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import (build_agreement_matrix, load_tiered_certification,
                         load_per_question_tiers, DATASETS, H5_ROOT)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

TIER_MAP = {"gold": 3, "silver": 2, "bronze": 1, "reject": 0}
GOLD_THRESHOLD = 0.90


def load_confidence_matrix(meta, models):
    """Load per-item LLM confidence values into a matrix aligned with the agreement matrix."""
    n_items = len(meta)
    n_models = len(models)
    conf_matrix = np.full((n_items, n_models), np.nan)

    model_to_col = {m: i for i, m in enumerate(models)}
    item_id_to_row = {iid: i for i, iid in enumerate(meta["item_id"].values)}

    for ds in DATASETS:
        pred_dir = H5_ROOT / "data" / "public" / ds / "predictions_v2"
        for fpath in sorted(pred_dir.glob("*.json")):
            with open(fpath, encoding="utf-8") as f:
                preds = json.load(f)
            if not preds:
                continue
            model_name = preds[0]["model_name"]
            if model_name not in model_to_col:
                continue
            col = model_to_col[model_name]
            for p in preds:
                row = item_id_to_row.get(p["id"])
                if row is not None:
                    conf_matrix[row, col] = p.get("confidence", np.nan)

    return conf_matrix


def sprt_test_length(outcomes, p0=0.80, p1=0.90, alpha=0.05, beta=0.10):
    """Run Bernoulli SPRT on a binary sequence. Returns test length."""
    log_A = np.log(beta / (1.0 - alpha))
    log_B = np.log((1.0 - beta) / alpha)
    inc_agree = np.log(p1 / p0)
    inc_disagree = np.log((1.0 - p1) / (1.0 - p0))
    cum_lr = 0.0
    for k, u in enumerate(outcomes):
        cum_lr += u * inc_agree + (1 - u) * inc_disagree
        if cum_lr >= log_B or cum_lr <= log_A:
            return k + 1
    return len(outcomes)


def run_h13_improved(matrix, meta, models, conf_matrix):
    """H13 with leave-one-model-out cross-validation + confidence baseline."""
    print("=" * 60)
    print("H13 IMPROVED: Cross-Validated Ordering + Confidence Baseline")
    print("=" * 60)

    per_q_frames = []
    for ds in DATASETS:
        pq = load_per_question_tiers(ds)
        pq["dataset"] = ds
        per_q_frames.append(pq)
    per_q_all = pd.concat(per_q_frames, ignore_index=True)
    gold_cells = per_q_all[per_q_all["tier"] == "gold"][
        ["model_name", "question_id", "dataset"]
    ].copy()

    model_idx = {m: i for i, m in enumerate(models)}
    rng = np.random.default_rng(42)
    N_SHUFFLES = 200

    rows = []
    for _, gcell in gold_cells.iterrows():
        model = gcell["model_name"]
        qid = gcell["question_id"]
        ds = gcell["dataset"]
        if model not in model_idx:
            continue
        m_col = model_idx[model]

        item_mask = (meta["question_id"].values == qid) & (meta["dataset"].values == ds)
        item_indices = np.where(item_mask)[0]
        if len(item_indices) < 20:
            continue

        agreements = matrix[item_indices, m_col]
        confidences = conf_matrix[item_indices, m_col]

        other_cols = [j for j in range(len(models)) if j != m_col]
        loo_difficulty = np.nanmean(matrix[np.ix_(item_indices, other_cols)], axis=1)

        full_difficulty = np.nanmean(matrix[item_indices], axis=1)

        valid = (~np.isnan(agreements) & ~np.isnan(loo_difficulty)
                 & ~np.isnan(confidences))
        if valid.sum() < 20:
            valid_no_conf = ~np.isnan(agreements) & ~np.isnan(loo_difficulty)
            if valid_no_conf.sum() < 20:
                continue
            agr = agreements[valid_no_conf].astype(int)
            diff_loo = loo_difficulty[valid_no_conf]
            diff_full = full_difficulty[valid_no_conf]
            conf = None
        else:
            agr = agreements[valid].astype(int)
            diff_loo = loo_difficulty[valid]
            diff_full = full_difficulty[valid]
            conf = confidences[valid]

        hardest_loo = np.argsort(diff_loo)
        hardest_full = np.argsort(diff_full)

        len_hardest_loo = sprt_test_length(agr[hardest_loo])
        len_hardest_full = sprt_test_length(agr[hardest_full])

        if conf is not None:
            lowest_conf_first = np.argsort(conf)
            len_conf = sprt_test_length(agr[lowest_conf_first])
        else:
            len_conf = np.nan

        random_lengths = []
        for _ in range(N_SHUFFLES):
            perm = rng.permutation(len(agr))
            random_lengths.append(sprt_test_length(agr[perm]))
        len_random = np.mean(random_lengths)

        rows.append({
            "model": model,
            "question_id": qid,
            "dataset": ds,
            "n_items": len(agr),
            "len_random": len_random,
            "len_hardest_full": len_hardest_full,
            "len_hardest_loo": len_hardest_loo,
            "len_confidence": len_conf,
            "saving_full_pct": (len_random - len_hardest_full) / len_random * 100,
            "saving_loo_pct": (len_random - len_hardest_loo) / len_random * 100,
            "saving_conf_pct": (len_random - len_conf) / len_random * 100 if not np.isnan(len_conf) else np.nan,
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("  No Gold cells.")
        return {}

    print(f"\n  Gold cells analyzed: {len(df)}")

    methods = {
        "Random": df["len_random"],
        "Hardest-first (full data)": df["len_hardest_full"],
        "Hardest-first (LOO-CV)": df["len_hardest_loo"],
        "Lowest-confidence-first": df["len_confidence"],
    }

    print(f"\n  Mean SPRT test length:")
    for name, series in methods.items():
        m = series.mean()
        saving = (df["len_random"].mean() - m) / df["len_random"].mean() * 100
        print(f"    {name:35s}  {m:6.1f}  (saving: {saving:+.1f}%)")

    saving_loo = (df["len_random"].mean() - df["len_hardest_loo"].mean()) / df["len_random"].mean() * 100
    saving_conf = (df["len_random"].mean() - df["len_confidence"].dropna().mean()) / df["len_random"].mean() * 100

    print(f"\n  Cross-validated saving (LOO): {saving_loo:.1f}%")
    print(f"  H13 threshold (>= 15%): {'** CONFIRMED **' if saving_loo >= 15 else 'NOT CONFIRMED'}")

    valid_conf = df.dropna(subset=["len_confidence"])
    if len(valid_conf) > 10:
        diff_loo_vs_conf = valid_conf["len_hardest_loo"] - valid_conf["len_confidence"]
        stat, p = stats.wilcoxon(diff_loo_vs_conf, alternative="two-sided")
        mean_diff = diff_loo_vs_conf.mean()
        print(f"\n  LOO difficulty vs confidence ordering:")
        print(f"    Mean length difference: {mean_diff:.2f} (negative = difficulty better)")
        print(f"    Wilcoxon p = {p:.4e}")
        if mean_diff < 0:
            print(f"    -> Difficulty ordering outperforms confidence ordering")
        else:
            print(f"    -> Confidence ordering outperforms difficulty ordering")

    stat_loo, p_loo = stats.wilcoxon(df["len_random"] - df["len_hardest_loo"],
                                      alternative="greater")
    print(f"\n  Wilcoxon (random > LOO hardest): p = {p_loo:.4e}")

    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if len(sub) > 0:
            s_loo = (sub["len_random"].mean() - sub["len_hardest_loo"].mean()) / sub["len_random"].mean() * 100
            s_conf = np.nan
            sub_conf = sub.dropna(subset=["len_confidence"])
            if len(sub_conf) > 0:
                s_conf = (sub_conf["len_random"].mean() - sub_conf["len_confidence"].mean()) / sub_conf["len_random"].mean() * 100
            print(f"  [{ds}]: LOO saving={s_loo:+.1f}%, confidence saving={s_conf:+.1f}%  (N={len(sub)})")

    df.to_csv(RESULTS_DIR / "h13_improved.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    method_names = ["Random", "Difficulty\n(full)", "Difficulty\n(LOO-CV)", "Confidence"]
    method_means = [df["len_random"].mean(), df["len_hardest_full"].mean(),
                    df["len_hardest_loo"].mean(), df["len_confidence"].mean()]
    colors = ["#585961", "#FF8844", "#FF4D00", "#2F99A4"]
    axes[0].bar(method_names, method_means, color=colors, edgecolor="black")
    axes[0].set_ylabel("Mean SPRT Test Length")
    axes[0].set_title("H13: Ordering Strategies\n(lower = fewer annotations needed)")

    axes[1].hist(df["saving_loo_pct"].dropna(), bins=25, alpha=0.6, color="#FF4D00",
                 edgecolor="black", label="Difficulty (LOO)")
    axes[1].hist(df["saving_conf_pct"].dropna(), bins=25, alpha=0.6, color="#2F99A4",
                 edgecolor="black", label="Confidence")
    axes[1].axvline(x=15, color="green", ls="--", lw=1.5, label="H13 threshold")
    axes[1].set_xlabel("Saving vs Random (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Per-Cell Saving Distribution")
    axes[1].legend(fontsize=8)

    valid_both = df.dropna(subset=["saving_loo_pct", "saving_conf_pct"])
    if len(valid_both) > 0:
        axes[2].scatter(valid_both["saving_conf_pct"], valid_both["saving_loo_pct"],
                       alpha=0.4, s=20, c="#2F99A4", edgecolors="black", linewidth=0.3)
        lim = max(abs(valid_both["saving_conf_pct"]).max(),
                  abs(valid_both["saving_loo_pct"]).max()) + 5
        axes[2].plot([-lim, lim], [-lim, lim], "r--", alpha=0.5, label="Equal performance")
        axes[2].set_xlabel("Confidence ordering saving (%)")
        axes[2].set_ylabel("Difficulty ordering saving (LOO, %)")
        axes[2].set_title("Difficulty vs Confidence\n(above diagonal = difficulty wins)")
        axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "h13_improved.png", dpi=150)
    plt.close()

    return {
        "saving_full": (df["len_random"].mean() - df["len_hardest_full"].mean()) / df["len_random"].mean() * 100,
        "saving_loo": saving_loo,
        "saving_conf": saving_conf,
        "wilcoxon_p_loo": p_loo,
        "confirmed_loo": saving_loo >= 15,
    }


def run_bootstrap_cis(matrix, meta, models, n_boot=2000):
    """Bootstrap 95% CIs on all key H12/H14 metrics."""
    print("\n" + "=" * 60)
    print("BOOTSTRAP: 95% Confidence Intervals")
    print("=" * 60)

    per_q_frames = []
    for ds in DATASETS:
        pq = load_per_question_tiers(ds)
        pq["dataset"] = ds
        per_q_frames.append(pq)
    per_q_all = pd.concat(per_q_frames, ignore_index=True)
    per_q_all["tier_numeric"] = per_q_all["tier"].map(TIER_MAP)

    model_ability = {}
    for j, m in enumerate(models):
        model_ability[m] = np.nanmean(matrix[:, j])

    per_q_all["ability"] = per_q_all["model_name"].map(model_ability)
    per_q_all = per_q_all.dropna(subset=["ability", "tier_numeric"])

    model_agg = per_q_all.groupby("model_name").agg(
        ability=("ability", "first"),
        mean_tier=("tier_numeric", "mean"),
        gold_frac=("tier_numeric", lambda x: (x == 3).mean()),
    ).reset_index()

    rng = np.random.default_rng(42)

    def boot_spearman(x, y, n=n_boot):
        rhos = []
        for _ in range(n):
            idx = rng.choice(len(x), size=len(x), replace=True)
            r, _ = stats.spearmanr(x[idx], y[idx])
            if not np.isnan(r):
                rhos.append(r)
        return np.percentile(rhos, [2.5, 97.5]) if rhos else (np.nan, np.nan)

    x = model_agg["ability"].values
    y = model_agg["mean_tier"].values
    rho, _ = stats.spearmanr(x, y)
    ci_lo, ci_hi = boot_spearman(x, y)
    print(f"\n  H12 per-model Spearman rho = {rho:.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")

    y_gold = model_agg["gold_frac"].values
    rho_gold, _ = stats.spearmanr(x, y_gold)
    ci_lo_g, ci_hi_g = boot_spearman(x, y_gold)
    print(f"  H12 ability vs Gold frac rho = {rho_gold:.4f}  95% CI [{ci_lo_g:.4f}, {ci_hi_g:.4f}]")

    gold_cells = per_q_all[per_q_all["tier_numeric"] == 3]
    model_idx_map = {m: i for i, m in enumerate(models)}
    cross_model_rate = np.nanmean(matrix, axis=1)

    deg_samples = []
    for _, gcell in gold_cells.iterrows():
        model = gcell["model_name"]
        qid = gcell["question_id"]
        ds = gcell["dataset"]
        if model not in model_idx_map:
            continue
        m_col = model_idx_map[model]
        item_mask = (meta["question_id"].values == qid) & (meta["dataset"].values == ds)
        idxs = np.where(item_mask)[0]
        if len(idxs) < 20:
            continue
        diffs = cross_model_rate[idxs]
        agrs = matrix[idxs, m_col]
        valid = ~np.isnan(agrs) & ~np.isnan(diffs)
        if valid.sum() < 20:
            continue
        q25 = np.percentile(diffs[valid], 25)
        hardest_mask = diffs[valid] <= q25
        if hardest_mask.sum() < 3:
            continue
        hardest_rate = agrs[valid][hardest_mask].mean()
        deg_samples.append(1 if hardest_rate < GOLD_THRESHOLD else 0)

    deg_arr = np.array(deg_samples)
    deg_pct = deg_arr.mean() * 100

    boot_degs = []
    for _ in range(n_boot):
        idx = rng.choice(len(deg_arr), size=len(deg_arr), replace=True)
        boot_degs.append(deg_arr[idx].mean() * 100)
    ci_lo_d, ci_hi_d = np.percentile(boot_degs, [2.5, 97.5])
    print(f"\n  H14 degradation = {deg_pct:.1f}%  95% CI [{ci_lo_d:.1f}%, {ci_hi_d:.1f}%]")

    results = {
        "h12_rho": rho, "h12_rho_ci": (ci_lo, ci_hi),
        "h12_gold_rho": rho_gold, "h12_gold_ci": (ci_lo_g, ci_hi_g),
        "h14_deg_pct": deg_pct, "h14_deg_ci": (ci_lo_d, ci_hi_d),
    }

    with open(RESULTS_DIR / "bootstrap_cis.txt", "w", encoding="utf-8") as f:
        f.write("Bootstrap 95% Confidence Intervals (n=2000)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"H12 rho (ability vs tier): {rho:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]\n")
        f.write(f"H12 rho (ability vs gold): {rho_gold:.4f} [{ci_lo_g:.4f}, {ci_hi_g:.4f}]\n")
        f.write(f"H14 degradation: {deg_pct:.1f}% [{ci_lo_d:.1f}%, {ci_hi_d:.1f}%]\n")

    return results


def main():
    print("Loading data...")
    matrix, meta, models = build_agreement_matrix()
    print(f"  Matrix: {matrix.shape}")

    print("Loading confidence matrix...")
    conf_matrix = load_confidence_matrix(meta, models)
    conf_coverage = (~np.isnan(conf_matrix)).mean()
    print(f"  Confidence coverage: {conf_coverage:.1%}")

    h13 = run_h13_improved(matrix, meta, models, conf_matrix)
    boot = run_bootstrap_cis(matrix, meta, models)

    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY")
    print("=" * 60)
    print(f"\n  H13 (cross-validated, LOO):")
    print(f"    Full-data saving: {h13['saving_full']:.1f}%")
    print(f"    LOO-CV saving: {h13['saving_loo']:.1f}%")
    print(f"    Confidence saving: {h13['saving_conf']:.1f}%")
    print(f"    H13 confirmed (LOO): {h13['confirmed_loo']}")
    print(f"\n  Bootstrap CIs:")
    print(f"    H12 rho: {boot['h12_rho']:.4f} [{boot['h12_rho_ci'][0]:.4f}, {boot['h12_rho_ci'][1]:.4f}]")
    print(f"    H14 deg: {boot['h14_deg_pct']:.1f}% [{boot['h14_deg_ci'][0]:.1f}%, {boot['h14_deg_ci'][1]:.1f}%]")


if __name__ == "__main__":
    main()
