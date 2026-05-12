# IRT-SPRT Bridge

Connecting psychometric grading profiling to sequential certification of LLM graders.

## The question

SPRT (Sequential Probability Ratio Test) certifies whether an LLM grader is reliable enough for deployment on a per-question basis. But SPRT treats all items equally — it doesn't know which student responses are easy or hard to grade. Can we use psychometric profiling (Item Response Theory) to make certification smarter?

## What we found

### H12: Model ability predicts certification tier (rho = 0.91)

A model's mean agreement rate across questions (the Rasch sufficient statistic for grading ability) almost perfectly predicts its SPRT certification tier (Gold/Silver/Bronze/Reject).

**What this means:** You don't need to run full SPRT certification on every model. A cheap agreement-rate screen can reliably predict which models will certify at Gold level and which won't. This could save significant annotation budget in multi-model evaluation pipelines.

**Caveat:** The per-model correlation is strong (rho=0.91), but the per-(model,question) correlation is weak (rho=0.21). This means a model's *overall* ability predicts *overall* certification, but you can't predict *which specific questions* a model will pass based on overall ability alone. Question-level certification still requires question-level evaluation.

### H13: Grading hard items first nearly halves certification time (43% saving)

Ordering student responses by cross-model difficulty (hardest first) during SPRT certification reduces the average test length from 33.3 to 18.9 items — a 43.2% saving (p < 10^-20).

**What this means:** If you already have difficulty estimates from a prior model pool (or from cross-model agreement), you can certify new models much faster by front-loading the hardest items. The hardest items are the most diagnostic — a correct grade on a hard item is stronger evidence of grader quality than a correct grade on an easy item.

**The effect is dataset-dependent:** ASAP-SAS shows a massive 69.4% saving because its questions have many items with wide difficulty spread. Mohler shows only 1.9% because items within a question are more homogeneous. The practical implication: ordering helps most when questions have many diverse student responses.

### H14: Half of Gold certifications mask failures on hard items (51% degradation)

51% of Gold-certified (model, question) cells show sub-Gold agreement when restricted to the hardest quartile of items. The agreement drops from 100% on the easiest quartile to 85.6% on the hardest.

**What this means:** Aggregate Gold certification can be misleading. A model might grade the easy student responses perfectly (boosting its average) while struggling on the hard responses — precisely the ones where accurate grading matters most (partial credit, borderline cases, unusual but valid answers).

**Deployment implication:** Instructors who rely on a Gold certification should know that the model may still fail on ~15% of the hardest responses. Difficulty-stratified certification reporting (showing per-quartile agreement) would give a more honest picture.

### SciEntsBank zero-Gold phenomenon

SciEntsBank (5-way exact-match labels) has **zero Gold certifications** across all 21 models. ASAP-SAS (4-point scale with ±1 tolerance) has 188 Gold certifications. This is not because models are worse at SciEntsBank — it's because exact match on 5 unordered categories is much harder than within-tolerance match on 4 ordinal levels.

**Counterfactual:** If SciEntsBank used ±1 tolerance (like ASAP-SAS and Mohler), 47 model-question cells would achieve Gold. The agreement definition, not model capability, determines whether certification is achievable.

**Why this matters for the NeurIPS paper:** The "zero false certifications" claim is technically correct but implicitly relies on the fact that most certifications come from datasets with tolerance-based agreement. On strict exact-match datasets, no model certifies at all — the claim is vacuously true.

## How to run

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

Requires [h5-sprt-certification](https://github.com/WSE-research/h5-sprt-certification) cloned at `../h5-sprt-certification/`.

## Results at a glance

| Hypothesis | Metric | Result | Threshold | Verdict |
|------------|--------|--------|-----------|---------|
| **H12** | Spearman rho (ability vs tier) | 0.91 | >= 0.70 | **Confirmed** |
| **H13** | Annotation saving (hardest-first) | 43.2% | >= 15% | **Confirmed** |
| **H14** | Gold cells degraded on hardest quartile | 51.0% | >= 20% | **Confirmed** |

## Key files

| File | Description |
|------|-------------|
| `results/model_abilities.csv` | Per-model grading ability (mean agreement, logit, gold rate) |
| `results/question_difficulties.csv` | Per-question difficulty (1 - cross-model agreement) |
| `results/h12_model_comparison.csv` | H12: ability vs SPRT tier per model |
| `results/h13_ordering_efficiency.csv` | H13: test length under each ordering strategy per Gold cell |
| `results/h14_stratified_certification.csv` | H14: per-quartile agreement for each Gold cell |
| `figures/h12_ability_vs_sprt.png` | H12: scatter plots |
| `figures/h13_ordering_efficiency.png` | H13: bar chart + saving distribution |
| `figures/h14_stratified_certification.png` | H14: quartile bars + tier breakdown |

## Methodology notes

**Why not full IRT?** With only 21 models, JMLE for a 1PL Rasch model produces degenerate estimates (infit values in the billions, difficulty std of 8 million). The raw agreement rate IS the Rasch sufficient statistic for ability, and the cross-model agreement rate is a clean difficulty proxy. We use these directly instead of unstable JMLE estimates.

**Agreement definitions matter.** ASAP-SAS and Mohler use within-tolerance agreement (±1 point). SciEntsBank uses exact match. This follows the conventions in [h5-sprt-certification](https://github.com/WSE-research/h5-sprt-certification/blob/master/src/public_benchmarks.py). Mixing these definitions in a single analysis is valid because each dataset's agreement definition matches how SPRT certification was originally computed for that dataset.

## Related work

- **Cong et al. 2026** ([arxiv 2605.00238](https://arxiv.org/abs/2605.00238)): Applies 1PL+testlet IRT to 17 LLM graders on SciEntsBank/Beetle. Recovers per-grader ability and per-response difficulty. **Key difference:** Cong-IRT stops at estimation; this work connects IRT parameters to deployment decisions via SPRT certification.
- **NeurIPS 2026 submission**: The SPRT certification framework this analysis builds on (29 models, 8,497 pairs, 0 false certifications).

## Citation

Part of Jonas Gwozdz's PhD research on LLM-based automated exam assessment at HTWK Leipzig.
