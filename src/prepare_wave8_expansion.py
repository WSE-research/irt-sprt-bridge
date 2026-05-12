"""Prepare for expanding the analysis from 21 (v2) to 29 (wave8) models.

This script checks which additional models have prediction files available
and generates a GPU job script for scoring any models that don't.

Run this locally. If all 29 models have predictions, the expansion can
proceed on CPU. If not, run the generated GPU script on H200.
"""

import json
from pathlib import Path

H5_ROOT = Path("C:/Users/jonas.gwozdz/Git Projekte/h5-sprt-certification")
DATASETS = ["asap_sas", "mohler", "scientsbank"]

WAVE8_MODELS = [
    "allenai/OLMo-2-0325-32B-Instruct",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "google/gemma-3-27b-it",
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-31B-it",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "ibm-granite/granite-3.3-8b-instruct",
    "internlm/internlm2_5-20b-chat",
    "microsoft/phi-4",
    "mistralai/Ministral-3-14B-Instruct-2512",
    "mistralai/Ministral-3-8B-Instruct-2512",
    "mistralai/Mistral-Small-24B-Instruct-2501",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.6-27B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B",
    "RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16",
    "tiiuae/Falcon-H1-34B-Instruct",
    "tiiuae/Falcon-H1-7B-Instruct",
]


def model_to_filename(model: str) -> str:
    return model.replace("/", "__").replace("-", "_").replace(".", "_") + ".json"


def check_predictions():
    """Check which wave8 models have prediction files in predictions_v2."""
    v2_models = set()
    for ds in DATASETS:
        pred_dir = H5_ROOT / "data" / "public" / ds / "predictions_v2"
        for f in pred_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
                if data:
                    v2_models.add(data[0]["model_name"])

    print(f"Models in v2 predictions: {len(v2_models)}")
    print(f"Models in wave8: {len(WAVE8_MODELS)}")

    missing = [m for m in WAVE8_MODELS if m not in v2_models]
    present = [m for m in WAVE8_MODELS if m in v2_models]

    print(f"\nPresent in v2 ({len(present)}):")
    for m in sorted(present):
        print(f"  + {m}")

    print(f"\nMissing from v2 ({len(missing)}):")
    for m in sorted(missing):
        print(f"  - {m}")

    if not missing:
        print("\nAll 29 models have v2 predictions! Expansion can proceed on CPU.")
        print("Run: python src/run_analysis.py  (after updating data_loader to include all models)")
    else:
        print(f"\n{len(missing)} models need scoring. GPU script needed.")
        print("These models had wave8 results but no v2 prediction files.")
        print("Options:")
        print("  1. Check if predictions exist under a different directory (predictions_wave8/)")
        print("  2. Re-run scoring via h5-sprt-certification/run_public.py on H200")

    # Check for wave8-specific prediction directories
    for ds in DATASETS:
        wave8_dir = H5_ROOT / "data" / "public" / ds / "predictions_wave8"
        if wave8_dir.exists():
            files = list(wave8_dir.glob("*.json"))
            print(f"\n  Found predictions_wave8/{ds}/: {len(files)} files")
            for f in files[:5]:
                print(f"    {f.name}")

    return present, missing


if __name__ == "__main__":
    present, missing = check_predictions()
