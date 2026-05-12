#!/bin/bash
# Score the 8 wave8 models that lack v2 predictions.
# Run on H200 via: tmux new -s irt-scoring && bash scripts/score_missing_models.sh
#
# Prerequisites:
#   - h5-sprt-certification repo cloned on H200
#   - vLLM serving endpoint available or models downloaded
#   - standardized.jsonl files present for all 3 datasets
#
# Estimated time: ~120-200 GPU-hours total (~1-2 days on H200)
# Estimated VRAM: largest model is 72B AWQ (already tested),
#   Falcon-H1-34B needs ~40GB, OLMo-2-32B needs ~40GB

set -euo pipefail

H5_DIR="${HOME}/h5-sprt-certification"
cd "$H5_DIR"

MISSING_MODELS=(
    "allenai/OLMo-2-0325-32B-Instruct"
    "ibm-granite/granite-3.3-8b-instruct"
    "mistralai/Ministral-3-14B-Instruct-2512"
    "mistralai/Ministral-3-8B-Instruct-2512"
    "Qwen/Qwen3.6-27B"
    "Qwen/Qwen3.6-35B-A3B"
    "tiiuae/Falcon-H1-34B-Instruct"
    "tiiuae/Falcon-H1-7B-Instruct"
)

DATASETS=("asap_sas" "mohler" "scientsbank")

echo "=== IRT-SPRT Bridge: Score missing wave8 models (v2 pipeline) ==="
echo "Models to score: ${#MISSING_MODELS[@]}"
echo "Datasets: ${DATASETS[*]}"
echo "Started: $(date)"
echo ""

# Check GPU
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""

for model in "${MISSING_MODELS[@]}"; do
    echo "================================================================"
    echo "Scoring: $model"
    echo "Time: $(date)"
    echo "================================================================"

    # Use the h5 run_public.py script which handles v2 pipeline
    # (chat templates, constrained decoding, structured outputs)
    python run_public.py \
        --model "$model" \
        --datasets "${DATASETS[@]}" \
        --version v2 \
        --temperature 0.0 \
        2>&1 | tee -a "logs/irt_scoring_$(echo "$model" | tr '/' '_').log"

    echo "Completed: $model at $(date)"
    echo ""
done

echo "=== All models scored. ==="
echo "Finished: $(date)"
echo ""
echo "Next: copy predictions to local machine and re-run IRT analysis:"
echo "  scp -r h200:~/h5-sprt-certification/data/public/*/predictions_v2/ ."
echo "  python src/run_analysis.py"
