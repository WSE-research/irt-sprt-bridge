"""Load and align prediction data, gold labels, and SPRT certification results."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

H5_ROOT = Path("C:/Users/jonas.gwozdz/Git Projekte/h5-sprt-certification")
DATASETS = ["asap_sas", "mohler", "scientsbank"]

AGREEMENT_SPECS = {
    "asap_sas": {"mode": "within_tolerance", "tolerance": 1.0},
    "mohler": {"mode": "within_tolerance", "tolerance": 1.0},
    "scientsbank": {"mode": "exact", "tolerance": None},
}


def load_gold_labels(dataset: str) -> pd.DataFrame:
    path = H5_ROOT / "data" / "public" / dataset / "standardized.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows.append({
                "item_id": rec["id"],
                "question_id": rec["question_id"],
                "gold_score": str(rec["gold_score"]),
                "dataset": dataset,
            })
    return pd.DataFrame(rows)


def load_predictions(dataset: str) -> pd.DataFrame:
    pred_dir = H5_ROOT / "data" / "public" / dataset / "predictions_v2"
    frames = []
    for fpath in sorted(pred_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        frames.append(df[["id", "question_id", "predicted_score", "confidence", "model_name"]])
    if not frames:
        raise FileNotFoundError(f"No prediction files in {pred_dir}")
    combined = pd.concat(frames, ignore_index=True)
    combined.rename(columns={"id": "item_id"}, inplace=True)
    combined["predicted_score"] = combined["predicted_score"].astype(str)
    combined["dataset"] = dataset
    return combined


def load_tiered_certification() -> pd.DataFrame:
    path = H5_ROOT / "results_tiered" / "cross_dataset_model_tier_summary.csv"
    return pd.read_csv(path)


def load_per_question_tiers(dataset: str) -> pd.DataFrame:
    path = H5_ROOT / "results_tiered" / dataset / f"{dataset}_tiered_certification.csv"
    return pd.read_csv(path)


def build_agreement_matrix() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build binary agreement matrix: rows=items, columns=models.

    Returns:
        agreement_df: DataFrame with item_id, question_id, dataset + one bool column per model
        item_meta: DataFrame with item_id, question_id, dataset
        model_names: sorted list of model names
    """
    all_gold = []
    all_preds = []

    for ds in DATASETS:
        gold = load_gold_labels(ds)
        preds = load_predictions(ds)
        all_gold.append(gold)
        all_preds.append(preds)

    gold_df = pd.concat(all_gold, ignore_index=True)
    preds_df = pd.concat(all_preds, ignore_index=True)

    merged = preds_df.merge(
        gold_df[["item_id", "dataset", "gold_score"]],
        on=["item_id", "dataset"],
        how="inner",
    )
    merged["agree"] = 0
    for ds, spec in AGREEMENT_SPECS.items():
        ds_mask = merged["dataset"] == ds
        if not ds_mask.any():
            continue
        if spec["mode"] == "exact":
            merged.loc[ds_mask, "agree"] = (
                merged.loc[ds_mask, "predicted_score"] == merged.loc[ds_mask, "gold_score"]
            ).astype(int)
        else:
            pred_num = pd.to_numeric(merged.loc[ds_mask, "predicted_score"], errors="coerce")
            gold_num = pd.to_numeric(merged.loc[ds_mask, "gold_score"], errors="coerce")
            merged.loc[ds_mask, "agree"] = (
                (pred_num - gold_num).abs() <= spec["tolerance"]
            ).fillna(
                merged.loc[ds_mask, "predicted_score"] == merged.loc[ds_mask, "gold_score"]
            ).astype(int)

    model_names = sorted(merged["model_name"].unique())
    pivoted = merged.pivot_table(
        index=["item_id", "question_id", "dataset"],
        columns="model_name",
        values="agree",
        aggfunc="first",
    )
    pivoted = pivoted.reindex(columns=model_names)

    item_meta = pivoted.index.to_frame(index=False)
    matrix = pivoted.values.astype(float)

    return matrix, item_meta, model_names


if __name__ == "__main__":
    matrix, meta, models = build_agreement_matrix()
    print(f"Agreement matrix: {matrix.shape[0]} items × {matrix.shape[1]} models")
    print(f"Models: {len(models)}")
    print(f"Datasets: {meta['dataset'].value_counts().to_dict()}")
    print(f"NaN fraction: {np.isnan(matrix).mean():.4f}")
    print(f"Overall agreement: {np.nanmean(matrix):.4f}")
