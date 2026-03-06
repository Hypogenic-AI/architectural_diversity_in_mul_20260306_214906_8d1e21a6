# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project:
**"Architectural Diversity in Multi-Agent LLM Systems for Robust Collective Reasoning"**

Resources were gathered through manual arXiv searches (paper-finder service was unavailable) across multiple targeted query sets, covering 9 query categories and 96+ unique papers before filtering to the most relevant 23.

---

## Papers

**Total papers downloaded: 23**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Improving Factuality via Multiagent Debate | Du, Li, Torralba, Tenenbaum, Mordatch | 2023 | Du2023_multiagent_debate_factuality.pdf | Foundational; NeurIPS 2023 |
| 2 | Encouraging Divergent Thinking via MAD | Liang, He, Jiao et al. | 2023 | Liang2023_divergent_thinking_debate.pdf | DoT problem; ACL 2024 |
| 3 | ChatEval: LLM-based Evaluators via Debate | Chan et al. | 2023 | Chan2023_chateval_multiagent_evaluators.pdf | Evaluation via debate |
| 4 | ReConcile: Diverse LLMs Round-Table | Chen, Saha, Bansal | 2023 | Chen2023_reconcile_diverse_llms.pdf | MOST RELEVANT; +11.4% |
| 5 | Can LLM Agents Really Debate? | Wu, Li, Li | 2025 | Wu2025_can_llm_agents_debate.pdf | Controlled diversity study |
| 6 | Survey on LLM Ensemble | Chen et al. | 2025 | Survey2025_llm_ensemble.pdf | Comprehensive taxonomy |
| 7 | Diverse LLMs or Diverse Interpretations? | Rosales, Miret (Intel) | 2025 | Diverse2025_llms_ensembling.pdf | Model vs. prompt diversity |
| 8 | Stable LLM Ensemble and Diversity | Niimi | 2025 | Stable2025_llm_ensemble_diversity.pdf | Calibrated diversity |
| 9 | LLM-TOPLA: Ensemble via Max Diversity | Tekin, Ilhan, Liu et al. | 2024 | LLM_TOPLA2024_ensemble_diversity.pdf | Focal diversity metric |
| 10 | More Agents Is All You Need | Li, Zhang, Yu et al. (Tencent) | 2024 | More2024_agents_all_you_need.pdf | Agent scaling baseline |
| 11 | Heterogeneous Ensemble Robustness | Wu, Chow, Wei, Liu | 2023 | Wu2023_heterogeneous_ensemble_robustness.pdf | Negative correlation theory |
| 12 | MetaGPT | Hong, Zhuge et al. | 2023 | Hong2023_metagpt.pdf | Multi-agent SOP framework |
| 13 | Cognitive Architectures for Language Agents | Sumers et al. | 2023 | Sumers2023_cognitive_architectures_agents.pdf | Agent architecture survey |
| 14 | From System 1 to System 2: Reasoning Survey | Li, Zhang et al. | 2025 | Li2025_system2_reasoning_survey.pdf | Reasoning survey |
| 15 | Adversarial Robustness of LLMs | Multiple | 2025 | Adversarial2025_robustness_llms.pdf | Adversarial benchmarking |
| 16 | SWE-Debate: Competitive Debate for Code | Li, Shi, Lin et al. | 2025 | SWEDebate2025_competitive_debate.pdf | Debate for software tasks |
| 17 | Tree-of-Debate: Multi-Persona | Kargupta, Agarwal et al. | 2025 | TreeOfDebate2025_multipersona.pdf | Structural debate diversity |
| 18 | TS-Debate: Multimodal Debate | Trirat, Kwak, Heo | 2026 | TSDebate2026_multimodal_debate.pdf | Cross-modal debate |
| 19 | Effective GenAI Multi-Agent Collaboration | Shu, Das et al. (AWS) | 2024 | Enterprise2024_multigenai_collaboration.pdf | Enterprise evaluation |
| 20 | Narrative Priming in LLM Collaboration | Multiple | 2025 | Narrative2025_llm_collaborate.pdf | Social dynamics of agents |
| 21 | Requesting Expert Reasoning | Multiple | 2026 | ExpertReasoning2026_llm_collaborative.pdf | Expert LLM augmentation |
| 22 | Vendi-RAG: Diversity in RAG | Multiple | 2025 | VendiRAG2025_diversity_quality.pdf | Diversity-quality tradeoffs |
| 23 | Small LLMs as Multi-LLM Agent | Multiple | 2024 | MultiLLM2024_agent_tool_learning.pdf | Multi-LLM specialization |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets downloaded: 5**

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| GSM8K | HuggingFace: `openai/gsm8k` | 7.5K train / 1.3K test | Math reasoning | datasets/gsm8k/ | Most used in debate papers |
| MMLU | HuggingFace: `cais/mmlu` (all) | 14K test / 100K aux | General knowledge (57 subjects) | datasets/mmlu/ | Cross-domain diversity testing |
| BoolQ | HuggingFace: `google/boolq` | 9.4K train / 3.3K val | Binary QA | datasets/boolq/ | Used in Rosales et al. 2025 |
| CommonsenseQA | HuggingFace: `tau/commonsense_qa` | 9.7K train / 1.2K val | 5-choice commonsense | datasets/commonsense_qa/ | Related to Chen et al. 2023 |
| ARC-Challenge | HuggingFace: `allenai/ai2_arc` | 1.1K train / 1.2K test | 4-choice science | datasets/arc_challenge/ | Science knowledge diversity |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

**Total repositories cloned: 6**

| Name | URL | Purpose | Location | Key Contribution |
|------|-----|---------|----------|-----------------|
| Multi-Agents-Debate | github.com/Skytliang/Multi-Agents-Debate | Homogeneous debate baseline | code/multi_agents_debate/ | MAD framework with judge |
| ReConcile | github.com/dinobby/ReConcile | Heterogeneous multi-model | code/reconcile/ | Confidence-weighted voting |
| LLM Debate (Du et al.) | github.com/composable-models/llm_debate | Original debate reference | code/llm_debate/ | Society of minds |
| LLM-TOPLA | github.com/git-disl/llm-topla | Diversity-optimized ensemble | code/llm_topla/ | Focal diversity metric |
| AgentForest | github.com/MoreAgentsIsAllYouNeed/AgentForest | Homogeneous scaling | code/agent_forest/ | Agent count scaling |
| HeteRobust | github.com/git-disl/HeteRobust | Heterogeneous robustness theory | code/heterobust/ | Negative correlation analysis |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

Used the arXiv Python API (`arxiv` library) with 9 targeted query categories:
1. "multi-agent LLM diverse models collective reasoning"
2. "heterogeneous ensemble large language models reasoning"
3. "mixture of experts LLM agents collaborative reasoning"
4. "LLM ensemble diversity adversarial robustness"
5. "multi-agent debate LLM reasoning"
6. "multi-LLM ensemble debate society of mind reasoning"
7. "LLM agent diversity correlated errors collective intelligence"
8. "heterogeneous model ensemble robustness reasoning benchmark"
9. "multi-agent LLM collaboration adversarial prompts"
10. "society of mind LLM collaboration diverse agents"
11. "multi-agent LLM debate majority voting self-consistency"
12. "diverse large language models bias reduction ensemble"
13. "collective reasoning multiple LLMs benchmark evaluation"

Also fetched specific arXiv IDs for foundational papers (Du et al., Liang et al., Chen et al., etc.)

### Selection Criteria

Papers selected based on:
1. **Relevance score**: Count of key topic terms in title + abstract (multi-agent, ensemble, diverse, debate, collective, robust, heterogeneous, etc.)
2. **Recency**: 2023-2026 papers preferred (+2 points for ≥2024, +1 for ≥2023)
3. **Direct relation to hypothesis**: Papers specifically about diversity benefits in collective reasoning
4. **Methodology quality**: Controlled experiments with clear baselines preferred
5. **Code availability**: Papers with available implementations prioritized

### Challenges Encountered

1. **Paper-finder service unavailable**: Fell back to manual arXiv search. Manual searches are less systematic but yielded good coverage of the field.
2. **Broad query results**: Many general "multi-agent" papers were about RL, not LLM reasoning; required careful filtering.
3. **Foundational papers with incorrect IDs**: Some initially guessed arXiv IDs (e.g., for "Mixture of Agents" by Together AI) didn't match expected papers; corrected through targeted searches.

### Gaps and Workarounds

1. **StrategyQA**: Not available as standard HuggingFace dataset (loading script issue); substituted CommonsenseQA and note that ReConcile repo includes the StrategyQA dataset directly.
2. **Knight-Knave-Spy dataset** (Wu et al. 2025): Not available as standalone HuggingFace dataset; can be downloaded from the paper's code repository.
3. **LLM-TOPLA pre-computed outputs**: Requires Google Drive download; documented in datasets/README.md.

---

## Recommendations for Experiment Design

### 1. Primary Datasets
- **GSM8K** for math reasoning (clear correctness, widely benchmarked)
- **MMLU** for diverse knowledge (57 subjects let different models shine)
- **Custom adversarial dataset** (must be created by experiment runner) — bias-exploiting questions where homogeneous models fail together

### 2. Baseline Methods (in order of importance)
1. Single best agent (GPT-4o / Claude 3.5 Sonnet)
2. Self-Consistency (Wang et al. 2023) — homogeneous sampling + majority vote
3. Agent Forest / More Agents — homogeneous scaling baseline
4. MAD (Liang et al.) — homogeneous debate baseline
5. **ReConcile** — heterogeneous multi-model discussion (strongest existing baseline to match)

### 3. Evaluation Metrics
1. **Task Accuracy** (primary): Exact match or answer accuracy
2. **Pairwise Error Correlation**: Key diversity metric — lower correlation = more diverse
3. **Convergence Rate**: Rounds to reach consensus in debate settings
4. **Adversarial Robustness**: Accuracy on perturbed/adversarial variants
5. **Focal Diversity Score**: Using LLM-TOPLA's metric

### 4. Experimental Conditions
- **Condition 1**: Homogeneous ensemble (same model family, N agents)
- **Condition 2**: Heterogeneous ensemble (different model families, N agents total)
- **Ablation**: Vary number of agents, diversity level (1-3 different families), debate rounds

### 5. Model Selection for Heterogeneous Condition
Use at minimum 3 architecturally distinct model families:
- **Frontier closed-source**: GPT-4o (OpenAI)
- **Alternative closed-source**: Claude 3.5 Sonnet (Anthropic) — different RLHF approach
- **Open-source**: LLaMA-3 or Mistral/Mixtral (Meta/Mistral) — different pre-training
- Optional: Gemini (Google) as 4th family
