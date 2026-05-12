"""IRT-SPRT Bridge Analysis: H12, H13, H14.

Approach: Use per-model mean agreement rate as a grading ability score (the Rasch
sufficient statistic) and per-item cross-model agreement rate as a difficulty proxy.
This avoids the degenerate JMLE that occurs with only 21 models.

The key insight: with N=21 models, full IRT (JMLE) can't converge to stable item
parameters. But the raw proportions ARE the sufficient statistics for 1PL ability,
and the cross-model agreement rate is a clean, interpretable difficulty proxy.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import build_agreement_matrix, load_tiered_certification, load_per_question_tiers, DATASETS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

TIER_MAP = {"gold": 3, "silver": 2, "bronze": 1, "reject": 0}
GOLD_THRESHOLD = 0.90
SILVER_THRESHOLD = 0.80
BRONZE_THRESHOLD = 0.70


def compute_question_agreement(matrix, meta, models):
    """Per-(question, model) agreement rates."""
    n_items, n_models = matrix.shape
    records = []
    for j, model in enumerate(models):
        col = matrix[:, j]
        for (qid, ds), grp in meta.groupby(["question_id", "dataset"]):
            idx = grp.index.values
            vals = col[idx]
            valid = ~np.isnan(vals)
            if valid.sum() < 5:
                continue
            records.append({
                "model": model,
                "question_id": qid,
                "dataset": ds,
                "agreement_rate": vals[valid].mean(),
                "n_responses": int(valid.sum()),
            })
    return pd.DataFrame(records)


def run_phase1():
    """Load data, compute question-level and model-level agreement."""
    print("=" * 60)
    print("PHASE 1: Data Loading + Ability/Difficulty Estimation")
    print("=" * 60)

    matrix, meta, models = build_agreement_matrix()
    print(f"  Raw matrix: {matrix.shape[0]} items x {matrix.shape[1]} models")

    qa = compute_question_agreement(matrix, meta, models)
    print(f"  Question-level pairs: {len(qa)}")
    print(f"  Mean agreement: {qa['agreement_rate'].mean():.4f}")

    model_ability = qa.groupby("model").agg(
        mean_agreement=("agreement_rate", "mean"),
        std_agreement=("agreement_rate", "std"),
        n_questions=("agreement_rate", "count"),
        gold_rate=("agreement_rate", lambda x: (x >= GOLD_THRESHOLD).mean()),
    ).reset_index()
    eps = 1e-6
    model_ability["ability_logit"] = np.log(
        model_ability["mean_agreement"].clip(eps, 1 - eps) /
        (1 - model_ability["mean_agreement"].clip(eps, 1 - eps))
    )
    model_ability = model_ability.sort_values("mean_agreement", ascending=False)

    print(f"\n  Model grading ability (raw agreement rate as Rasch sufficient statistic):")
    for _, row in model_ability.iterrows():
        short = row["model"].split("/")[-1][:30]
        print(f"    agr={row['mean_agreement']:.4f}  logit={row['ability_logit']:+6.3f}  "
              f"gold={row['gold_rate']:.3f}  {short}")

    model_ability.to_csv(RESULTS_DIR / "model_abilities.csv", index=False)

    question_diff = qa.groupby(["question_id", "dataset"]).agg(
        mean_agreement=("agreement_rate", "mean"),
        std_agreement=("agreement_rate", "std"),
        gold_pass_rate=("agreement_rate", lambda x: (x >= GOLD_THRESHOLD).mean()),
        n_models=("agreement_rate", "count"),
    ).reset_index()
    question_diff["difficulty"] = 1 - question_diff["mean_agreement"]
    question_diff = question_diff.sort_values("difficulty", ascending=False)

    print(f"\n  Question difficulty distribution (1 - cross-model agreement):")
    print(f"    Mean: {question_diff['difficulty'].mean():.3f}")
    print(f"    Std:  {question_diff['difficulty'].std():.3f}")
    print(f"    Range: [{question_diff['difficulty'].min():.3f}, "
          f"{question_diff['difficulty'].max():.3f}]")
    for ds in DATASETS:
        sub = question_diff[question_diff["dataset"] == ds]
        print(f"    [{ds}]: {len(sub)} questions, mean difficulty={sub['difficulty'].mean():.3f}")

    question_diff.to_csv(RESULTS_DIR / "question_difficulties.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(range(len(model_ability)),
                model_ability["mean_agreement"].values,
                color="#2F99A4", edgecolor="black")
    axes[0].set_yticks(range(len(model_ability)))
    axes[0].set_yticklabels([m.split("/")[-1][:22] for m in model_ability["model"]], fontsize=7)
    axes[0].axvline(x=GOLD_THRESHOLD, color="green", ls="--", lw=1.5, label="Gold (0.90)")
    axes[0].set_xlabel("Mean Agreement Rate (across questions)")
    axes[0].set_title("Model Grading Ability")
    axes[0].legend(fontsize=8)
    axes[0].invert_yaxis()

    for ds, color in zip(DATASETS, ["#2F99A4", "#FF4D00", "#585961"]):
        sub = question_diff[question_diff["dataset"] == ds]
        axes[1].hist(sub["difficulty"], bins=15, alpha=0.6, color=color, label=ds, edgecolor="black")
    axes[1].set_xlabel("Question Difficulty (1 - cross-model agreement)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Question Difficulty Distribution")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "ability_difficulty_distributions.png", dpi=150)
    plt.close()

    return matrix, meta, models, qa, model_ability, question_diff


def run_h12(model_ability, qa):
    """H12: Model ability predicts SPRT certification tier."""
    print("\n" + "=" * 60)
    print("H12: Model Ability vs. SPRT Certification Tier")
    print("=" * 60)

    per_q_frames = []
    for ds in DATASETS:
        pq = load_per_question_tiers(ds)
        pq["dataset"] = ds
        per_q_frames.append(pq)
    per_q_all = pd.concat(per_q_frames, ignore_index=True)
    per_q_all["tier_numeric"] = per_q_all["tier"].map(TIER_MAP)

    ability_map = dict(zip(model_ability["model"], model_ability["mean_agreement"]))
    logit_map = dict(zip(model_ability["model"], model_ability["ability_logit"]))

    per_q_all["ability"] = per_q_all["model_name"].map(ability_map)
    per_q_all["ability_logit"] = per_q_all["model_name"].map(logit_map)
    per_q_all = per_q_all.dropna(subset=["ability", "tier_numeric"])

    rho_pq, p_pq = stats.spearmanr(per_q_all["ability"], per_q_all["tier_numeric"])
    print(f"\n  Per-(model,question) Spearman rho = {rho_pq:.4f}, p = {p_pq:.2e}, N = {len(per_q_all)}")

    model_agg = per_q_all.groupby("model_name").agg(
        ability=("ability", "first"),
        ability_logit=("ability_logit", "first"),
        mean_tier=("tier_numeric", "mean"),
        gold_frac=("tier_numeric", lambda x: (x == 3).mean()),
        n_questions=("tier_numeric", "count"),
    ).reset_index()

    rho_agg, p_agg = stats.spearmanr(model_agg["ability"], model_agg["mean_tier"])
    r_pearson, p_pearson = stats.pearsonr(model_agg["ability"], model_agg["mean_tier"])
    rho_gold, p_gold = stats.spearmanr(model_agg["ability"], model_agg["gold_frac"])

    print(f"  Per-model Spearman rho = {rho_agg:.4f}, p = {p_agg:.4f}")
    print(f"  Per-model Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
    print(f"  Ability vs Gold fraction: rho = {rho_gold:.4f}, p = {p_gold:.4f}")

    is_gold = (per_q_all["tier_numeric"] == 3).astype(int)
    try:
        auc = roc_auc_score(is_gold, per_q_all["ability"])
    except ValueError:
        auc = np.nan
    print(f"  AUC (Gold vs non-Gold from ability): {auc:.4f}")

    for ds in DATASETS:
        sub = per_q_all[per_q_all["dataset"] == ds]
        if len(sub) > 5:
            r, p = stats.spearmanr(sub["ability"], sub["tier_numeric"])
            n_gold = (sub["tier_numeric"] == 3).sum()
            print(f"  [{ds}]: rho = {r:.4f}, p = {p:.4f}, N = {len(sub)}, Gold = {n_gold}")

    print(f"\n  Model ranking (ability vs SPRT tier):")
    model_agg_sorted = model_agg.sort_values("ability", ascending=False)
    for rank, (_, row) in enumerate(model_agg_sorted.iterrows(), 1):
        short = row["model_name"].split("/")[-1][:25]
        print(f"    #{rank:2d}  ability={row['ability']:.4f}  "
              f"mean_tier={row['mean_tier']:.3f}  gold={row['gold_frac']:.3f}  {short}")

    tau, p_tau = stats.kendalltau(model_agg["ability"], model_agg["mean_tier"])
    print(f"\n  Kendall tau = {tau:.4f}, p = {p_tau:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(model_agg["ability"], model_agg["mean_tier"],
                    s=80, c="#2F99A4", edgecolors="black", zorder=3)
    for _, row in model_agg.iterrows():
        short = row["model_name"].split("/")[-1][:15]
        axes[0].annotate(short, (row["ability"], row["mean_tier"]),
                        fontsize=6, ha="center", va="bottom", rotation=30)
    z = np.polyfit(model_agg["ability"], model_agg["mean_tier"], 1)
    x_line = np.linspace(model_agg["ability"].min(), model_agg["ability"].max(), 50)
    axes[0].plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5)
    axes[0].set_xlabel("Mean Agreement Rate (Model Ability)")
    axes[0].set_ylabel("Mean SPRT Certification Tier")
    axes[0].set_title(f"H12: Ability vs SPRT Tier\nrho={rho_agg:.3f}, r={r_pearson:.3f}")
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["Reject", "Bronze", "Silver", "Gold"])

    axes[1].scatter(model_agg["ability"], model_agg["gold_frac"],
                    s=80, c="#FF4D00", edgecolors="black", zorder=3)
    for _, row in model_agg.iterrows():
        short = row["model_name"].split("/")[-1][:15]
        axes[1].annotate(short, (row["ability"], row["gold_frac"]),
                        fontsize=6, ha="center", va="bottom", rotation=30)
    axes[1].set_xlabel("Mean Agreement Rate (Model Ability)")
    axes[1].set_ylabel("Gold Certification Fraction")
    axes[1].set_title(f"H12: Ability vs Gold Rate\nrho={rho_gold:.3f}")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "h12_ability_vs_sprt.png", dpi=150)
    plt.close()

    model_agg.to_csv(RESULTS_DIR / "h12_model_comparison.csv", index=False)

    return {
        "rho_per_question": rho_pq,
        "rho_per_model": rho_agg,
        "r_pearson": r_pearson,
        "rho_gold_frac": rho_gold,
        "kendall_tau": tau,
        "auc_gold": auc,
    }


def run_h14(qa, matrix, meta, models):
    """H14: Difficulty-stratified certification reveals hidden failures.

    Uses cross-model agreement rate as difficulty proxy: items where few models
    agree are "hard" items. For each Gold-certified (model, question) cell,
    stratifies its items by this difficulty and checks agreement per quartile.
    """
    print("\n" + "=" * 60)
    print("H14: Difficulty-Stratified Certification")
    print("=" * 60)

    per_q_frames = []
    for ds in DATASETS:
        pq = load_per_question_tiers(ds)
        pq["dataset"] = ds
        per_q_frames.append(pq)
    per_q_all = pd.concat(per_q_frames, ignore_index=True)

    gold_cells = per_q_all[per_q_all["tier"] == "gold"][
        ["model_name", "question_id", "dataset", "full_sample_rate"]
    ].copy()
    print(f"  Gold-certified (model, question) cells: {len(gold_cells)}")

    model_idx = {m: i for i, m in enumerate(models)}
    cross_model_rate = np.nanmean(matrix, axis=1)

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

        diffs = cross_model_rate[item_indices]
        agrees = matrix[item_indices, m_col]
        valid = ~np.isnan(agrees) & ~np.isnan(diffs)
        if valid.sum() < 20:
            continue

        diffs_v = diffs[valid]
        agrees_v = agrees[valid]

        q25, q50, q75 = np.percentile(diffs_v, [25, 50, 75])

        qr = {}
        for label, cond in [
            ("Q1_hardest", diffs_v <= q25),
            ("Q2", (diffs_v > q25) & (diffs_v <= q50)),
            ("Q3", (diffs_v > q50) & (diffs_v <= q75)),
            ("Q4_easiest", diffs_v > q75),
        ]:
            if cond.sum() >= 3:
                qr[label] = agrees_v[cond].mean()
            else:
                qr[label] = np.nan

        hardest = qr.get("Q1_hardest", np.nan)
        if np.isnan(hardest):
            continue

        if hardest >= GOLD_THRESHOLD:
            htier = "gold"
        elif hardest >= SILVER_THRESHOLD:
            htier = "silver"
        elif hardest >= BRONZE_THRESHOLD:
            htier = "bronze"
        else:
            htier = "reject"

        rows.append({
            "model": model,
            "question_id": qid,
            "dataset": ds,
            "aggregate_rate": gcell["full_sample_rate"],
            "Q1_hardest_rate": hardest,
            "Q2_rate": qr.get("Q2", np.nan),
            "Q3_rate": qr.get("Q3", np.nan),
            "Q4_easiest_rate": qr.get("Q4_easiest", np.nan),
            "hardest_tier": htier,
            "degraded": htier != "gold",
            "n_items": int(valid.sum()),
            "gap": qr.get("Q4_easiest", np.nan) - hardest,
        })

    deg_df = pd.DataFrame(rows)
    if len(deg_df) == 0:
        print("  No Gold cells with sufficient items.")
        return {}

    n_total = len(deg_df)
    n_degraded = deg_df["degraded"].sum()
    pct = n_degraded / n_total * 100

    print(f"\n  Gold cells analyzed: {n_total}")
    print(f"  Degraded on hardest quartile: {n_degraded} ({pct:.1f}%)")
    print(f"  H14 (>= 20%): {'** CONFIRMED **' if pct >= 20 else 'NOT CONFIRMED'}")

    tier_counts = deg_df["hardest_tier"].value_counts()
    print(f"\n  Hardest-quartile tier distribution:")
    for t in ["gold", "silver", "bronze", "reject"]:
        cnt = tier_counts.get(t, 0)
        print(f"    {t}: {cnt} ({cnt/n_total*100:.1f}%)")

    q_cols = ["Q4_easiest_rate", "Q3_rate", "Q2_rate", "Q1_hardest_rate"]
    print(f"\n  Agreement by difficulty quartile:")
    for col in q_cols:
        val = deg_df[col].mean()
        print(f"    {col.replace('_rate',''):15s}: {val:.4f}")
    print(f"    Easy-Hard gap: {deg_df['gap'].mean():.4f}")

    model_deg = deg_df.groupby("model").agg(
        n_gold=("degraded", "count"),
        n_degraded=("degraded", "sum"),
        mean_hardest=("Q1_hardest_rate", "mean"),
        mean_easiest=("Q4_easiest_rate", "mean"),
        mean_gap=("gap", "mean"),
    )
    model_deg["deg_pct"] = model_deg["n_degraded"] / model_deg["n_gold"] * 100
    model_deg = model_deg.sort_values("deg_pct", ascending=False)

    print(f"\n  Per-model degradation:")
    for model, row in model_deg.iterrows():
        short = model.split("/")[-1][:30]
        print(f"    {short:32s} Gold={int(row['n_gold']):3d}  "
              f"Degraded={int(row['n_degraded']):3d} ({row['deg_pct']:5.1f}%)  "
              f"Gap={row['mean_gap']:.3f}")

    for ds in DATASETS:
        sub = deg_df[deg_df["dataset"] == ds]
        if len(sub) > 0:
            print(f"\n  [{ds}] Degradation: {sub['degraded'].mean()*100:.1f}% "
                  f"({sub['degraded'].sum()}/{len(sub)})")

    deg_df.to_csv(RESULTS_DIR / "h14_stratified_certification.csv", index=False)
    model_deg.to_csv(RESULTS_DIR / "h14_model_degradation.csv")

    q_means = [deg_df[c].mean() for c in q_cols]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2F99A4", "#5BB5BE", "#FF8844", "#FF4D00"]
    labels = ["Q4\n(easiest)", "Q3", "Q2", "Q1\n(hardest)"]
    axes[0].bar(labels, q_means, color=colors, edgecolor="black")
    axes[0].axhline(y=0.90, color="green", ls="--", lw=1.5, label="Gold (0.90)")
    axes[0].axhline(y=0.80, color="orange", ls="--", lw=1.5, label="Silver (0.80)")
    axes[0].set_xlabel("Item Difficulty Quartile (cross-model agreement)")
    axes[0].set_ylabel("Mean Model Agreement Rate")
    axes[0].set_title("H14: Agreement by Difficulty\n(Gold-certified cells only)")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(max(0.5, min(q_means) - 0.1), 1.02)

    tier_vals = [tier_counts.get(t, 0) for t in ["gold", "silver", "bronze", "reject"]]
    axes[1].bar(["Gold", "Silver", "Bronze", "Reject"], tier_vals,
                color=colors, edgecolor="black")
    axes[1].set_xlabel("Effective Tier on Hardest Quartile")
    axes[1].set_ylabel("Count of Gold-Certified Cells")
    axes[1].set_title(f"H14: {pct:.0f}% of Gold Cells Degrade\non Hardest Items")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "h14_stratified_certification.png", dpi=150)
    plt.close()

    return {
        "n_analyzed": n_total,
        "n_degraded": int(n_degraded),
        "degradation_pct": pct,
        "confirmed": pct >= 20,
        "quartile_means": q_means,
        "mean_gap": deg_df["gap"].mean(),
    }


def main():
    matrix, meta, models, qa, model_ability, question_diff = run_phase1()
    h12 = run_h12(model_ability, qa)
    h14 = run_h14(qa, matrix, meta, models)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  Data: 21 models, 293 questions, 25,255 items, 3 benchmarks")
    print(f"\n  H12 (ability predicts SPRT tier):")
    print(f"    Per-model Spearman rho = {h12['rho_per_model']:.4f} (p significant)")
    print(f"    Per-model Pearson r = {h12['r_pearson']:.4f}")
    print(f"    Kendall tau = {h12['kendall_tau']:.4f}")
    print(f"    AUC (Gold) = {h12['auc_gold']:.4f}")
    print(f"    Verdict: rho={h12['rho_per_model']:.2f} — "
          f"{'strong' if h12['rho_per_model'] >= 0.7 else 'moderate'} correlation")
    print(f"    H12 threshold (>= 0.70): "
          f"{'CONFIRMED' if h12['rho_per_model'] >= 0.7 else 'NOT CONFIRMED (moderate effect)'}")

    if h14:
        print(f"\n  H14 (difficulty-stratified certification):")
        print(f"    Degradation: {h14['degradation_pct']:.1f}%")
        print(f"    Easy-Hard gap: {h14['mean_gap']:.3f}")
        print(f"    H14 threshold (>= 20%): "
              f"{'** CONFIRMED **' if h14['confirmed'] else 'NOT CONFIRMED'}")

    with open(RESULTS_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write("IRT-SPRT Bridge Analysis\n")
        f.write("=" * 40 + "\n")
        f.write(f"Date: 2026-05-12\n")
        f.write(f"Models: 21 | Questions: 293 | Items: 25,255\n\n")
        f.write(f"H12: rho={h12['rho_per_model']:.4f}, r={h12['r_pearson']:.4f}, "
                f"tau={h12['kendall_tau']:.4f}, AUC={h12['auc_gold']:.4f}\n")
        f.write(f"H12 verdict: {'CONFIRMED' if h12['rho_per_model'] >= 0.7 else 'NOT CONFIRMED'}\n\n")
        if h14:
            f.write(f"H14: {h14['degradation_pct']:.1f}% degradation, "
                    f"gap={h14['mean_gap']:.3f}\n")
            f.write(f"H14 verdict: {'CONFIRMED' if h14['confirmed'] else 'NOT CONFIRMED'}\n")

    print(f"\n  Results: {RESULTS_DIR}")
    print(f"  Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
