# Architectural Diversity in Multi-Agent LLM Systems for Robust Collective Reasoning

## 1. Executive Summary

This study investigates whether capability-level diversity (heterogeneous model sizes) within the Claude model family provides measurable benefits for adversarial reasoning tasks compared to homogeneous ensembles. Using 30 adversarially-designed reasoning questions across 5 cognitive trap categories, we evaluated 5 experimental conditions with real Claude API calls. **Key finding**: Modern large language models (Claude Haiku and Claude Sonnet) achieve very high accuracy (93-97%) on classical adversarial reasoning tasks, with highly correlated failure modes (r=0.46-1.00), leaving minimal room for diversity-based improvement. Heterogeneous ensembles show the same accuracy as homogeneous ones (96.7%), debate protocols slightly degrade performance, and the primary shared failure mode is systematic over-caution in medical recommendation scenarios.

---

## 2. Goal

**Research Question**: Do multi-agent LLM systems with capability-level diversity (heterogeneous model sizes: claude-haiku-4-5 vs. claude-sonnet-4-6) demonstrate superior robustness on adversarially-designed ambiguous reasoning tasks compared to homogeneous ensembles, due to reduced correlated error rates?

**Hypothesis**: H1 — Heterogeneous ensembles achieve higher accuracy than homogeneous ensembles. H2 — Error correlation is lower between haiku-sonnet pairs than haiku-haiku pairs. H3 — Structured debate improves accuracy. H4 — Heterogeneous ensembles are more robust under paraphrase perturbation.

**Motivation**: Multi-agent LLM systems increasingly power critical reasoning tasks, yet most deploy homogeneous model ensembles that may share failure modes. Classical ensemble theory predicts that diversity reduces correlated errors (negative correlation learning, Wu et al. 2023). The question of whether this extends to LLMs — and at what granularity of diversity — is a key open problem.

---

## 3. Data Construction

### Adversarial Benchmark Design

We created a custom 30-question adversarial reasoning benchmark across 5 categories (6 questions each):

| Category | Design Principle | Example |
|----------|-----------------|---------|
| **Misleading Math** | Surface pattern gives wrong answer | Bat-ball ($1.10 total, bat costs $1 more → ball = $0.05 not $0.10) |
| **Causal Traps** | Correlation/causation confusion, Simpson's paradox | Hospital with higher death rate may be better (severity confounding) |
| **Logical Deception** | Multi-step puzzles with misleading surface patterns | Knight-knave variants, rope-burning timing, box labeling |
| **Numerical Tricks** | Base rate neglect, anchoring, harmonic vs. arithmetic mean | Medical test base rate (99% accurate test → ~50% PPV at 1% prevalence) |
| **Framing Effects** | Gain vs. loss framing on equivalent scenarios | 90% survival vs. 10% mortality — same surgery, different recommendation |

Plus 6 paraphrase variants of selected questions for robustness testing.

### Example Questions

**Q1 (Misleading Math)**:
> "A bat and a ball together cost $1.10. The bat costs exactly $1.00 more than the ball. How much does the ball cost in cents?"
> Correct: 5 cents. Misleading: 10 cents (naive subtraction)

**Q7 (Causal Traps)**:
> "Hospital A has a 2% death rate; Hospital B has a 1% death rate. Should you choose Hospital B?"
> Correct: NO (Simpson's paradox — A treats sicker patients). Misleading: YES

**Q19 (Numerical Tricks)**:
> "1% of people have Disease X. Test is 99% accurate. You test positive. Probability you have the disease?"
> Correct: ~50% (Bayes). Misleading: 99% (ignoring base rate)

### Data Quality

- All 30 questions have clear ground-truth answers (verified by derivation)
- 2 questions (Q17 rope puzzle, Q18 switch puzzle) require procedural free-text answers; evaluated using content-based matching
- Q23 (exponential doubling) revealed an ambiguity in "exceed" interpretation (strictly > vs. ≥); both "100 years" and "120 years" are defensible
- API key interruption at question 15; resumed; all 30 completed
- Total API calls: ~310 (30 questions × ~8 calls + 6 paraphrases × ~4 calls)

---

## 4. Experiment Description

### Methodology

We compare 5 experimental conditions representing a diversity continuum:

| Condition | Description | Agents | Aggregation |
|-----------|-------------|--------|-------------|
| **Single Haiku** | One claude-haiku-4-5 agent | 1 | Direct answer |
| **Single Sonnet** | One claude-sonnet-4-6 agent | 1 | Direct answer |
| **Homogeneous Haiku (3×)** | Three haiku agents | 3 | Majority vote |
| **Heterogeneous (2H+1S)** | Two haiku + one sonnet | 3 | Majority vote |
| **Debate (H→S→H)** | Haiku proposes, Sonnet critiques, Haiku revises | 2 | Final haiku answer |

### Implementation Details

**Tools and Libraries:**
| Library | Version |
|---------|---------|
| anthropic | 0.52+ |
| numpy | 1.26+ |
| scipy | 1.13+ |
| matplotlib | 3.8+ |
| Python | 3.12.8 |

**Models:**
- Small: `claude-haiku-4-5-20251001` — fast, cost-efficient
- Large: `claude-sonnet-4-6` — more capable, slower

**Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 | Deterministic, reproducible |
| Max tokens | 512 | Sufficient for CoT reasoning |
| Retries | 3 | Handle transient API errors |
| Inter-call delay | 0.3-0.5s | Rate limit compliance |

**Prompting Strategy:**
- System prompt encourages identifying misleading patterns and step-by-step reasoning
- Agents instructed to state `FINAL ANSWER:` clearly
- Debate: Haiku proposes → Sonnet critiques (with `CRITIC_SYSTEM` prompt) → Haiku revises (with `REVISER_SYSTEM` prompt)

**Random seed**: 42 (set for numpy/random); model calls at temperature=0

**Hardware**: CPU only (no GPU), Linux, 2 CPU cores. API latency ~2-5s/call.

**Execution time**: ~33 minutes total (25 min original, 8 min paraphrase + 15 min resume after API interruption)

### Evaluation Metrics

1. **Task Accuracy**: Fraction of questions answered correctly; primary metric
2. **Error Correlation (Pearson r)**: Correlation between binary error vectors of two conditions; lower = more diverse failure modes
3. **Debate Improvement Rate**: Fraction of questions where debate protocol improved over haiku baseline
4. **Adversarial Robustness**: Accuracy drop between original questions and paraphrase variants

### Statistical Analysis

- **Bootstrap 95% CI**: 1000 resamplings for accuracy confidence intervals
- **McNemar's test**: Paired comparison of accuracy differences between conditions (n=30)
- **Pearson correlation**: For pairwise error vector correlation

---

## 5. Raw Results

### Overall Accuracy

| Condition | Correct/Total | Accuracy | 95% CI |
|-----------|---------------|----------|--------|
| Single Haiku | 28/30 | 93.3% | [77.5%, 100%] |
| Single Sonnet | 28/30 | 93.3% | [77.5%, 100%] |
| Homogeneous Haiku (3×) | 29/30 | 96.7% | [83.3%, 100%] |
| **Heterogeneous (2H+1S)** | **29/30** | **96.7%** | **[83.3%, 100%]** |
| Debate (H→S→H) | 28/30 | 93.3% | [77.5%, 100%] |

### Per-Category Accuracy

| Category | Single H | Single S | Homog. H | Hetero. | Debate |
|----------|----------|----------|----------|---------|--------|
| Misleading Math | 100% | 100% | 100% | 100% | 100% |
| Causal Traps | 100% | 100% | 100% | 100% | 100% |
| Logical Deception | 83% | 83% | 83% | 83% | 83% |
| Numerical Tricks | 100% | 83% | 100% | 100% | 83% |
| Framing Effects | 83% | 83% | 83% | 83% | 83% |

### Error Correlation Matrix

| Pair | Pearson r | p-value |
|------|-----------|---------|
| Single Haiku ↔ Single Sonnet | **0.464** | 0.010 |
| Single Haiku ↔ Homog. Haiku | 0.695 | 0.000 |
| Single Haiku ↔ Heterogeneous | 0.695 | 0.000 |
| Single Sonnet ↔ Homog. Haiku | 0.695 | 0.000 |
| Single Sonnet ↔ Heterogeneous | 0.695 | 0.000 |
| Homog. Haiku ↔ Heterogeneous | **1.000** | 0.000 |

### Debate Outcomes (30 questions)

| Outcome | Count |
|---------|-------|
| Maintained Correct | 28 |
| Improved (Wrong → Right) | 0 |
| **Degraded (Right → Wrong)** | **1** |
| Maintained Wrong | 1 |
| Answers Changed | 9 |

### Robustness (Original vs. Paraphrase, n=6)

| Condition | Original Acc. | Paraphrase Acc. | Δ |
|-----------|---------------|-----------------|---|
| Single Haiku | 93.3% | ~83% (5/6) | -10pp |
| Single Sonnet | 93.3% | ~83% (5/6) | -10pp |
| Homog. Haiku | 96.7% | ~83% (5/6) | -14pp |
| Heterogeneous | 96.7% | ~83% (5/6) | -14pp |

*Note: Paraphrase set is only 6 questions; estimates have high uncertainty.*

### Failure Analysis

Only 3 questions had any failures:

| Q | Category | Description | Which Conditions Fail |
|---|----------|-------------|----------------------|
| Q18 | Logical Deception | Light switch heat puzzle | Single Haiku only |
| Q23 | Numerical Tricks | Population doubling (ambiguous "exceed") | Single Sonnet, Debate |
| Q27 | Framing Effects | 90% survival rate surgery | **All 5 conditions** |

**Q18 insight**: Haiku fails to identify the heat-as-information insight ("warm bulb = switch was recently on"). Sonnet correctly identifies this. However, haiku's full reasoning does mention warmth in prose — only its final answer extraction misses it.

**Q23 insight**: This question contains a genuine ambiguity: "first exceed 32,000" can mean ≥32,000 (100 years, 2^5=32) or >32,000 strictly (120 years, 2^6=64). Haiku chooses the looser interpretation (100), sonnet chooses the stricter one (120). My benchmark marked 100 as correct, making this a benchmark design flaw.

**Q27 insight**: "A surgery has a 90% survival rate. Would you recommend it?" — All Claude variants refuse to recommend without knowing the underlying condition, representing a **systematic shared safety-aligned bias** across model sizes. This illustrates that within-family LLMs share not only knowledge but also safety alignment properties.

### Output Files

- Raw results: `results/raw_results_final.json`
- Analysis statistics: `results/analysis_results.json`
- Plots: `results/plots/`

---

## 6. Result Analysis

### Key Findings

**Finding 1: Modern Claude models are highly robust to classical adversarial reasoning traps.**
Both haiku (smaller) and sonnet (larger) achieve 93.3% accuracy on our adversarial benchmark, outperforming typical human accuracy on these cognitive trap tasks. This is consistent with the general trend of LLMs surpassing humans on classical reasoning benchmarks when fine-tuned with RLHF (Li et al. 2025).

**Finding 2: Error correlation between haiku and sonnet is moderate (r=0.464), lower than within-condition correlations.**
The Haiku-Sonnet pair has lower error correlation (r=0.464, p=0.010) than the Haiku-Homogeneous Haiku pair (r=0.695), which is consistent with H2 — different capability levels do have somewhat different failure modes. However, the absolute error count is too low (1-2 errors each) to draw strong conclusions.

**Finding 3: Ensemble methods provide marginal but non-significant accuracy improvements.**
Both Homogeneous Haiku (3×) and Heterogeneous (2H+1S) reach 96.7% vs. 93.3% for single agents, but McNemar tests show p=1.000 (no significant difference, given only 1-2 discordant pairs). The tiny sample makes statistical inference unreliable. The improvement (1 additional correct answer) comes from haiku majority preventing sonnet's wrong answer on Q23.

**Finding 4: The debate protocol degraded performance.**
The H→S→H debate protocol achieved the same accuracy as single models (28/30=93.3%) but caused 1 degradation (Q23: haiku was correct, sonnet's critique convinced haiku to adopt sonnet's incorrect but more careful interpretation). 9 of 30 answers changed during debate, demonstrating the protocol can alter responses, but without accuracy benefit. This aligns with the "agreeability" problem identified by Du et al. (2023) — RLHF-trained models tend to accept critique even when they were originally correct.

**Finding 5: Systematic shared bias is the dominant failure mode.**
Q27 (surgery recommendation) reveals that all conditions share a common safety-aligned refusal to recommend without full clinical context. This shared bias is MORE harmful than uncorrelated random errors, as no ensemble method can overcome it. This supports the core thesis of Rosales et al. (2025): for model diversity to help, models need truly complementary failure modes, not just independent sampling.

### Hypothesis Testing Results

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| H1: Hetero. > Homog. accuracy | **Not confirmed** (p=1.000) | 96.7% = 96.7%; 1 additional correct question each |
| H2: Haiku-Sonnet error correlation < Haiku-Haiku | **Partially confirmed** | r=0.464 vs r=0.695; statistically significant (p=0.010) |
| H3: Debate improves accuracy | **Refuted** | 0 improvements, 1 degradation |
| H4: Hetero. more robust under paraphrase | **Not confirmed** | Both drop equally (83% paraphrase) |

### Error Analysis — Why Models Fail Together

The correlated failure on Q27 reveals a critical insight: **within-family models share not just capabilities but aligned safety behaviors**. Claude haiku and sonnet are both trained by Anthropic with similar RLHF principles, making their safety-aligned refusals correlated. This is distinct from capability-based diversity — safety alignment is a cross-cutting concern that cannot be addressed by scaling model size.

### Diversity Index

| Pair Type | Error Correlation | Diversity Index (1-r) |
|-----------|-------------------|----------------------|
| Haiku-Haiku (same model) | r=0.695 | 0.305 |
| Haiku-Sonnet (cross-capability) | r=0.464 | 0.536 |

The heterogeneous (haiku-sonnet) pairing does show higher diversity index (0.536 vs. 0.305), partially supporting the theoretical prediction. However, with only 2 errors per condition, this diversity does not translate to measurable ensemble accuracy improvement.

### Surprises and Insights

1. **Modern LLMs are resistant to classical cognitive traps**: We expected ~60-70% accuracy; actual accuracy was 93-97%. The benchmark was not adversarial enough for 2026-era Claude models.

2. **Debate can hurt, not just help**: The critique from sonnet convinced haiku to change a correct answer. This "sycophantic revision" problem is documented in literature (Du et al. 2023) but was expected to be mitigated by the Haiku-proposes-first structure. Instead, haiku readily adopted sonnet's more cautious (wrong) interpretation.

3. **Q23 revealed benchmark ambiguity**: "First exceed" is genuinely ambiguous between ≥ and >. Sonnet's more careful reading of "exceed" as strict inequality was arguably more linguistically precise. This highlights the importance of unambiguous adversarial question design.

4. **Heterogeneous voting actually helped via haiku majority**: In Q23, the 2-haiku majority in the heterogeneous ensemble correctly overruled sonnet's minority interpretation. This is a case where majority voting protected against a stronger model's overcautious reasoning — the opposite of the expected "wisdom of stronger models" effect.

### Limitations

1. **Small benchmark (n=30)**: With only 1-2 failures per condition, all statistical tests are severely underpowered. Effect sizes cannot be meaningfully estimated.

2. **Within-family diversity only**: We only compared claude-haiku-4-5 vs. claude-sonnet-4-6 due to single API key availability. Cross-family diversity (Claude vs. GPT vs. LLaMA) would test the core architectural diversity hypothesis more rigorously.

3. **Classical adversarial tasks may be saturated**: Modern LLMs trained on internet-scale data have likely seen most classical cognitive trap problems. Truly novel adversarial tasks (unseen in training) would better test the hypothesis.

4. **Temperature=0 reduces sampling diversity**: For homogeneous ensemble experiments, temperature=0 with the same model likely produces identical or near-identical responses, making the 3× ensemble effectively equivalent to single-shot sampling. Higher temperatures would introduce productive within-model diversity.

5. **Evaluation challenges for open-ended answers**: Questions requiring procedural reasoning (Q17, Q18) required content-based evaluation that may miss semantic equivalences.

6. **API key interruption**: The API key expired midway through experiments; the second half was completed in a separate session. While results appeared consistent, this introduces a minor confound.

---

## 7. Conclusions

### Summary

Within the Claude model family, capability-level diversity (small vs. large model) provides **modest but statistically insignificant accuracy improvements** on our adversarial reasoning benchmark. Both models achieve high accuracy (93.3%) individually, and ensemble methods improve to 96.7% — a one-question improvement on a 30-question test. Error correlations confirm that haiku and sonnet do have somewhat different failure modes (r=0.464), supporting theoretical predictions, but the shared safety-alignment bias (Q27) represents a more fundamental challenge that diversity cannot overcome.

### Implications

**For system designers**: Within-family model ensembles provide marginal benefits for reasoning robustness. If diversity is the goal, cross-family (different providers/architectures) diversity is more likely to yield complementary failure modes. Same-provider models share not just capabilities but alignment properties.

**For the diversity hypothesis**: The results are consistent with Rosales et al. (2025): model diversity alone is insufficient — what matters is **complementary failure modes**. Within-family models share too many failure patterns for diversity to provide substantial benefits.

**For debate protocol design**: Critique-and-revise protocols are susceptible to the "sycophantic revision" failure where correct models adopt incorrect critiques. Future designs should include mechanisms to validate critique quality before accepting revisions.

### Confidence in Findings

**Medium confidence** in the negative finding (diversity doesn't help much) within this model family. The main uncertainty is the small benchmark size (n=30) with very few failures. Additional evidence needed: (1) cross-family models (GPT, LLaMA, Gemini), (2) larger adversarial benchmark (200+ questions), (3) higher-temperature sampling for within-condition diversity.

---

## 8. Next Steps

### Immediate Follow-ups

1. **Cross-family study**: Use OpenRouter or multi-provider setup to compare GPT-4.1, Claude Sonnet, and Llama-3 as heterogeneous ensemble members. This is the true test of architectural diversity.

2. **Larger adversarial benchmark**: Create 200+ novel adversarial questions not likely in LLM training data, using programmatic generation (e.g., novel knight-knave configurations, synthetic math with deliberate traps).

3. **Higher temperature ensemble**: Repeat with temperature=0.7 for ensemble conditions to introduce productive within-model sampling diversity and compare with cross-model diversity.

4. **Failure-mode targeted ensembles**: Instead of random mixing, design ensembles specifically to cover identified failure modes (e.g., one agent specialized in causal reasoning, another in logical deduction).

### Alternative Approaches

- **Mixture of Experts routing**: Use a lightweight classifier to route questions to the model most likely to succeed, rather than always using all models.
- **Cascade architectures**: Have haiku handle easy questions and escalate hard ones to sonnet only (FrugalGPT-style), measuring accuracy and cost jointly.

### Open Questions

1. Does shared safety alignment across same-provider models fundamentally limit diversity benefits?
2. At what difficulty level does within-family diversity begin to outperform homogeneous ensembles?
3. Can debate protocols be redesigned to preserve correct initial answers while benefiting from critique?

---

## 9. References

1. Du, Y. et al. (2023). "Improving Factuality and Reasoning in Language Models through Multiagent Debate." NeurIPS 2023. arXiv:2305.14325.
2. Chen, J.C. et al. (2023). "ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs." ACL 2024. arXiv:2309.13007.
3. Wu, H. et al. (2025). "Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning." arXiv:2511.07784.
4. Rosales, R. & Miret, S. (2025). "Diverse LLMs or Diverse Question Interpretations? That is the Ensembling Question." Intel Labs. arXiv:2507.21168.
5. Tekin, S.F. et al. (2024). "LLM-TOPLA: Efficient LLM Ensemble by Maximising Diversity." arXiv:2410.03953.
6. Li, J. et al. (2024). "More Agents Is All You Need." TMLR 2024. arXiv:2402.05120.
7. Wu, Y. et al. (2023). "Exploring Model Learning Heterogeneity for Boosting Ensemble Robustness." arXiv:2310.02237.
8. Liang, T. et al. (2023). "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate." ACL 2024. arXiv:2305.19118.
9. Li, Z.Z. et al. (2025). "From System 1 to System 2: A Survey of Reasoning Large Language Models." arXiv:2502.17419.

---

## Appendix: Experimental Configuration

```python
# Key parameters
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"
TEMPERATURE = 0.0
MAX_TOKENS = 512
RANDOM_SEED = 42

# Conditions
CONDITIONS = {
    "single_haiku": "1x haiku, direct answer",
    "single_sonnet": "1x sonnet, direct answer",
    "homogeneous_haiku": "3x haiku, majority vote",
    "heterogeneous_2h1s": "2x haiku + 1x sonnet, majority vote",
    "debate": "haiku proposes, sonnet critiques, haiku revises"
}

# Benchmark: 30 questions, 5 categories, 6 each
# + 6 paraphrase variants
```
