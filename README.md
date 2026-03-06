# Architectural Diversity in Multi-Agent LLM Systems

## Project Description

An empirical study testing whether **capability-level diversity** (heterogeneous model sizes) in multi-agent LLM systems improves reasoning robustness on adversarially-designed cognitive trap tasks. Uses real Claude API calls (claude-haiku-4-5 and claude-sonnet-4-6) across 5 experimental conditions with 30 custom adversarial questions.

## Key Findings

- **Modern Claude models are highly robust**: Both haiku and sonnet achieve 93.3% accuracy on our adversarial benchmark — far above typical human performance on these cognitive trap tasks
- **Within-family diversity shows limited benefit**: Heterogeneous (2H+1S) and homogeneous (3×H) ensembles achieve the same accuracy (96.7%), not statistically distinguishable from single-model performance (p=1.000, McNemar's test, n=30)
- **Error correlations confirm diversity exists but fails to translate**: Haiku-Sonnet error correlation (r=0.464) is lower than Haiku-Haiku (r=0.695), suggesting different failure modes, but only 1-2 total errors per condition limits measurable impact
- **Debate protocol can hurt**: H→S→H debate caused 1 degradation, 0 improvements; the "sycophantic revision" problem — sonnet's critique convinced haiku to revise a correct answer
- **Shared safety alignment is the dominant shared failure**: All models refuse to recommend surgery with insufficient clinical context (Q27), revealing cross-cutting alignment biases that diversity cannot overcome

## How to Reproduce

### Environment Setup

```bash
cd /workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6
source .venv/bin/activate
export ANTHROPIC_API_KEY="your_key_here"
```

### Run Experiments

```bash
# Full experiment (~310 API calls, ~25 min)
python src/run_experiments.py
# Fix answer extraction
python src/fix_extraction.py
# Generate analysis and plots
python src/analyze_results.py
```

## File Structure

```
├── planning.md                    # Research plan
├── REPORT.md                      # Full research report with findings
├── src/
│   ├── adversarial_benchmark.py   # 30-question adversarial benchmark
│   ├── multi_agent_framework.py   # API wrappers and ensemble logic
│   ├── run_experiments.py         # Main experiment runner
│   ├── resume_experiments.py      # Resume after interruption
│   ├── fix_extraction.py          # Post-processing for answer extraction
│   └── analyze_results.py        # Statistical analysis + visualizations
├── results/
│   ├── raw_results_final.json     # All experiment results
│   ├── analysis_results.json     # Statistics
│   └── plots/                    # Generated figures
├── datasets/                      # Pre-downloaded benchmarks (GSM8K, MMLU, etc.)
├── papers/                        # Research papers (23 downloaded)
├── literature_review.md           # Comprehensive literature review
└── resources.md                   # Resource catalog
```

## Experimental Conditions

| Condition | Agents | Aggregation | Accuracy |
|-----------|--------|-------------|----------|
| Single Haiku | 1× haiku | Direct | 93.3% |
| Single Sonnet | 1× sonnet | Direct | 93.3% |
| Homogeneous (3×H) | 3× haiku | Majority vote | 96.7% |
| **Heterogeneous (2H+1S)** | 2× haiku + 1× sonnet | Majority vote | **96.7%** |
| Debate (H→S→H) | haiku + sonnet | Haiku revises | 93.3% |

See [REPORT.md](REPORT.md) for full analysis, methodology, and implications.
