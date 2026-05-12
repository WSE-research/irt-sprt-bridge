# IRT-SPRT Bridge

Connecting Item Response Theory to Sequential Grader Certification.

## What this does

Applies psychometric profiling (IRT-derived grading ability and item difficulty) to the SPRT certification dataset from [WSE-research/h5-sprt-certification](https://github.com/WSE-research/h5-sprt-certification) (29 models, 3 ASAG benchmarks, 8,497+ pairs).

Tests three hypotheses:
- **H12**: Does model grading ability predict SPRT certification tier? (rho = 0.91 — yes)
- **H13**: Does difficulty-based item ordering improve certification efficiency? (pending GPU runs)
- **H14**: Does aggregate certification mask difficulty-dependent failures? (51% degradation — yes)

## Quick start

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

Requires the [h5-sprt-certification](https://github.com/WSE-research/h5-sprt-certification) repo cloned alongside this one.

## Key results

| Hypothesis | Metric | Result | Verdict |
|------------|--------|--------|---------|
| H12 | Spearman rho (model ability vs SPRT tier) | 0.91 | **Confirmed** |
| H14 | % Gold cells degraded on hardest quartile | 51.0% | **Confirmed** |

## Structure

```
src/
  data_loader.py    # Load predictions, gold labels, certification results
  irt_model.py      # 1PL Rasch IRT (JMLE)
  run_analysis.py   # Main analysis pipeline (H12, H14)
results/            # CSV outputs
figures/            # PNG plots
```

## Related

- PhD vault: `research/incubations/2026-05-12_irt-grader-profiling.md`
- Hypotheses: H12, H13, H14 in `research/hypotheses/`
- Competing work: [Cong et al. 2026 (arxiv 2605.00238)](https://arxiv.org/abs/2605.00238)
