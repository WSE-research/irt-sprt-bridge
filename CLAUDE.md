# IRT-SPRT Bridge

## What this is

Part of Jonas Gwozdz's PhD on LLM-based automated exam assessment at HTWK Leipzig.
This repo connects Item Response Theory (psychometric grading profiling) to SPRT
(Sequential Probability Ratio Test) certification of LLM graders.

## Current state (2026-05-12)

All three hypotheses confirmed with publication-grade statistical rigor:

- **H12** (rho=0.87, CI [0.60,0.98]): Model ability predicts SPRT certification tier
- **H13** (41.3% LOO-CV saving, p=6.2e-19): Hardest-first ordering nearly halves annotation cost
- **H14** (51.0%, CI [45.7%,56.0%]): Half of Gold certifications mask failures on hard items

Bonus finding: LLM confidence ordering is WORSE than random (-4.4%), because confidence
is anti-informative on SciEntsBank (r=-0.21 with agreement).

## What's left

1. **Pending decision (Andreas meeting 2026-05-13):** AIME-Con standalone paper (deadline June 14)
   vs EMNLP section vs park. See vault: `experiments/irt-sprt-bridge/2026-05-12_h12-h14-analysis.md`
2. **IRT reframing** (~1 day): JMLE diverged with N=21 models. We use raw agreement rates
   (Rasch sufficient statistics). Need to either reframe honestly or expand to 29 models via GPU.
3. **Paper draft** (~3-4 days): No draft exists yet.
4. **GPU expansion** (optional): `scripts/score_missing_models.sh` scores 8 wave8 models
   missing v2 predictions (21 → 29 models, ~120-200 GPU-hours on H200).

## Key paths

- Data source: `C:/Users/jonas.gwozdz/Git Projekte/h5-sprt-certification/`
- Vault experiment note: `experiments/irt-sprt-bridge/2026-05-12_h12-h14-analysis.md`
- Incubation report: `research/incubations/2026-05-12_irt-grader-profiling.md`
- Hypothesis files: `research/hypotheses/H12_*.md`, `H13_*.md`, `H14_*.md`

## Running

```bash
python src/run_analysis.py        # Main analysis (H12, H13, H14)
python src/run_improvements.py    # Cross-validation + confidence baseline + bootstrap CIs
```

## Conventions

- No Co-Authored-By lines in commits
- Use proper umlauts (ä ö ü ß) in German text
- Vault notes use Obsidian-flavored markdown with wikilinks
