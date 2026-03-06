# Research Planning: Architectural Diversity in Multi-Agent LLM Systems

## Motivation & Novelty Assessment

### Why This Research Matters
Multi-agent LLM systems are increasingly deployed for complex reasoning tasks, yet most systems use homogeneous model ensembles that share failure modes. If diverse agents make complementary errors, architectural heterogeneity could substantially improve system robustness — particularly on adversarially-designed ambiguous tasks where shared biases are most harmful.

### Gap in Existing Work
Based on `literature_review.md`:
- Du et al. (2023), More Agents (2024): Study homogeneous ensembles only; don't measure error correlation
- ReConcile (2023): Uses heterogeneous models but doesn't systematically ablate diversity level or test adversarial robustness
- Wu et al. (2025): Controlled study of debate factors, finds diversity important, but uses same-family models with different prompts
- Rosales et al. (2025): Challenges naive diversity hypothesis — finds question interpretation diversity often beats model diversity; argues failure mode correlation, not mere independence, is key
- **Critical Gap**: No study systematically compares capability-level diversity (different-size same-family models) with adversarially-designed tasks specifically targeting shared failure modes, while measuring error correlation matrices

### Our Novel Contribution
1. **Custom adversarial benchmark**: Reasoning tasks with surface-pattern misleading cues across 5 categories, designed to expose specific failure modes
2. **Capability diversity study**: Comparing homogeneous (same model size) vs. heterogeneous (mixed model sizes within Claude family) ensembles
3. **Error correlation quantification**: Measuring pairwise error correlation between agent types across task categories
4. **Adversarial robustness**: Testing stability under paraphrase perturbations

### Experiment Justification
- **Exp 1 (Single Agent Baselines)**: Establish individual accuracy and failure mode profiles for each model
- **Exp 2 (Homogeneous Ensembles)**: Test whether scaling same-model sampling provides diversity benefits
- **Exp 3 (Heterogeneous Ensembles)**: Test whether mixing capability levels improves on homogeneous baselines
- **Exp 4 (Debate Protocol)**: Test whether structured debate between different-capability agents improves accuracy beyond voting
- **Exp 5 (Adversarial Robustness)**: Test stability of each condition under question paraphrasing

---

## Research Question
Do multi-agent LLM systems with deliberate capability diversity (heterogeneous model families/sizes) demonstrate superior robustness on adversarially-designed ambiguous reasoning tasks compared to homogeneous ensembles, due to reduced correlated error rates?

## Background and Motivation
Classical ensemble learning theory predicts that diversity reduces correlated errors. For LLMs, different model sizes within the same family exhibit meaningfully different failure modes: smaller models rely more on surface patterns, while larger models apply deeper reasoning. This within-family capability diversity provides a tractable proxy for cross-family architectural diversity, enabling controlled experiments with available API access.

## Hypothesis Decomposition

**H1 (Accuracy)**: Heterogeneous ensembles achieve higher accuracy than same-size homogeneous ensembles on adversarial tasks
**H2 (Error Correlation)**: Error correlation between haiku-haiku agent pairs > error correlation between haiku-sonnet agent pairs
**H3 (Debate Benefit)**: Structured debate (critique + revision) between different-capability agents outperforms mere voting
**H4 (Adversarial Robustness)**: Heterogeneous ensembles degrade less under paraphrase perturbations

**Independent Variables**:
- Ensemble composition (homogeneous-haiku, homogeneous-sonnet, heterogeneous-mixed)
- Aggregation mechanism (majority voting, confidence-weighted voting, debate)
- Task category (5 adversarial categories)

**Dependent Variables**:
- Task accuracy (primary)
- Pairwise error correlation (diversity metric)
- Perturbation stability (robustness metric)

---

## Proposed Methodology

### Approach
Use the Anthropic API with claude-haiku-4-5 (fast, smaller) and claude-sonnet-4-6 (capable, larger) to represent capability-level diversity within the Claude family. Design 30 adversarial reasoning questions across 5 categories. Compare 5 experimental conditions:

1. **Single-Haiku**: One haiku agent, zero-shot CoT
2. **Single-Sonnet**: One sonnet agent, zero-shot CoT
3. **Homogeneous-Haiku**: 3x haiku agents, majority voting
4. **Homogeneous-Sonnet**: 3x sonnet agents, majority voting
5. **Heterogeneous-Mixed**: 2x haiku + 1x sonnet agents, majority voting
6. **Debate-Mixed**: haiku proposes, sonnet critiques, haiku revises (1 round)

### Adversarial Task Categories (6 per category = 30 total)
1. **Misleading Math**: Problems where a shortcut/pattern gives wrong answer (bat-ball, rate problems, compound interest traps)
2. **Causal Traps**: Correlation-causation, confounded causal chains, Simpson's paradox variants
3. **Logical Deception**: Knight-knave-style puzzles with deliberate misleading surface patterns
4. **Numerical Tricks**: Problems using anchoring, base rate neglect, or unit confusion
5. **Framing Effects**: Same problem, misleading framing leads to different reasoning paths; tests consistency

Plus 6 paraphrase variants of 6 original questions for adversarial robustness testing.

### Baselines
- Single best model (sonnet) = performance ceiling individual baseline
- Single smallest model (haiku) = performance floor individual baseline
- Homogeneous 3x majority voting (established as effective baseline in More Agents, 2024)

### Evaluation Metrics
1. **Accuracy**: Exact match to ground truth (0/1 per question)
2. **Pairwise Error Correlation (ρ)**: Cohen's kappa between agent pair error vectors; lower = more diverse
3. **Diversity Index**: Mean pairwise error correlation across ensemble members (lower = more diverse)
4. **Consensus Rate**: Fraction of questions where majority voting reaches clear consensus (>2/3 agreement)
5. **Perturbation Stability**: Accuracy change under paraphrase (robustness score)

### Statistical Analysis Plan
- **Effect size**: Cohen's d for accuracy differences between conditions (n=30 questions)
- **Bootstrap CI**: 1000 bootstrap resamples for 95% confidence intervals on accuracy
- **McNemar's test**: For pairwise comparison of which condition gets more questions right
- **Pearson correlation**: For error correlation between agent pairs
- **Significance level**: α = 0.05

---

## Adversarial Task Design

### Category 1: Misleading Math (6 questions)
Design principle: Problems where the "obvious" arithmetic shortcut gives wrong answer

Example: "A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost?"
- Surface answer: $0.10 (wrong)
- Correct answer: $0.05

### Category 2: Causal Traps (6 questions)
Design principle: Problems with correlation/causation confusion, hidden confounders

Example: "Hospital X has a higher death rate than Hospital Y. Should you avoid Hospital X?"
- Surface answer: Yes
- Correct: No — Hospital X likely receives sicker patients (Simpson's paradox)

### Category 3: Logical Deception (6 questions)
Design principle: Multi-step logical puzzles where one misleading piece of information creates false impression

Example: Knight-Knave variants where the structure suggests one answer but the logical deduction gives another

### Category 4: Numerical Tricks (6 questions)
Design principle: Anchoring, base rate neglect, unit confusion

Example: "If 1% of people have disease X, and the test has 99% accuracy, what is the probability a positive test means you have the disease?"
- Surface answer: 99%
- Correct: ~50% (base rate neglect)

### Category 5: Framing Effects (6 questions)
Design principle: Questions where framing creates systematic bias in reasoning direction

---

## Expected Outcomes

**Supporting H1**: Heterogeneous ensemble accuracy > homogeneous-haiku; possibly > homogeneous-sonnet due to complementary strengths

**Supporting H2**: Haiku-haiku error correlation > haiku-sonnet error correlation (different capability = different failure modes)

**Supporting H3**: Debate condition outperforms majority voting, especially on logic and causal tasks

**Supporting H4**: Heterogeneous ensemble shows < 10% accuracy degradation under paraphrase; homogeneous shows > 15%

**Null result possible**: If sonnet dominates all conditions (anchoring effect in debate), heterogeneous may not outperform homogeneous-sonnet. This would suggest that capability asymmetry leads to dominance rather than complementarity.

---

## Timeline and Milestones

- **Phase 0-1 (Planning)**: 30 min — Complete (this document)
- **Phase 2 (Environment)**: 10 min — Install packages, verify API
- **Phase 3 (Adversarial Task Design)**: 20 min — Write 30 questions with ground truth
- **Phase 4 (Experiment Implementation)**: 20 min — Code multi-agent framework
- **Phase 5 (Experiment Execution)**: 30 min — Run all API calls (~250 calls)
- **Phase 6 (Analysis & Documentation)**: 20 min — Statistics, visualization, report

Total: ~130 min (within 1h if focused; target to complete core in 60 min)

---

## Potential Challenges

1. **API Rate Limits**: Anthropic throttles requests; use async calls with retry backoff
2. **LLM Non-Determinism**: Run with temperature=0 for reproducibility
3. **Answer Extraction**: LLMs give verbose answers; need robust extraction logic
4. **Haiku Limitations**: May fail on all hard tasks, reducing statistical power; include moderate-difficulty questions

## Success Criteria
1. At least 3 of 4 sub-hypotheses show trends in predicted direction
2. Error correlation is measurably lower for haiku-sonnet than haiku-haiku pairs
3. Statistical tests show p < 0.1 for at least one key comparison (given n=30)
4. Results are interpretable and informative even if hypothesis not confirmed
