# Code Repositories for Multi-Agent LLM Diversity Research

## Repository 1: Multi-Agents-Debate (MAD)

- **URL**: https://github.com/Skytliang/Multi-Agents-Debate
- **Paper**: Liang et al. (2023), arXiv:2305.19118
- **Location**: `code/multi_agents_debate/`
- **Purpose**: Reference implementation of Multi-Agent Debate (MAD) with affirmative/negative sides and a judge
- **Key Files**:
  - `code/debate4tran.sh`: Main script to run MAD
  - `code/interactive.py`: Interactive debate mode
  - `requirements.txt`: Dependencies (openai, tiktoken, etc.)
- **Key Features**:
  - Two-agent debate in "tit for tat" mode
  - Judge LLM to determine winner and final answer
  - Adaptive break mechanism
  - OpenAI API-based
- **How to Use for Our Research**:
  - Adapt `interactive.py` to support heterogeneous models (different APIs)
  - Use as baseline homogeneous debate framework
  - Modify to add n>2 agents for diversity ablation
- **Requirements**: OpenAI API key

---

## Repository 2: ReConcile

- **URL**: https://github.com/dinobby/ReConcile
- **Paper**: Chen et al. (2023), arXiv:2309.13007
- **Location**: `code/reconcile/`
- **Purpose**: Heterogeneous multi-LLM round-table discussion with confidence-weighted voting
- **Key Files**:
  - `run.py`: Main execution script (`python run.py --num_samples 100 --dataset SQA`)
  - `generation.py`: Response generation from multiple LLMs (GPT, PaLM, Claude)
  - `data_utils.py`: Dataset loading utilities
  - `claude.py`: Claude API integration
  - `dataset/`: Pre-processed datasets (StrategyQA, ECQA, GSM8K, AQuA)
- **Key Features**:
  - Multi-model multi-agent discussion (GPT-3.5, PaLM-2, Claude)
  - Confidence-weighted voting mechanism
  - Convincingness demonstration via few-shot examples
  - Multiple discussion rounds
- **How to Use for Our Research**:
  - Directly use as the **primary heterogeneous baseline**
  - Extend to include more model families (LLaMA, Mistral, etc.)
  - Use confidence-weighted voting implementation as reference
  - Adapt dataset loading for custom adversarial tasks
- **Requirements**: OpenAI, PaLM, Claude API keys

---

## Repository 3: LLM Debate (Du et al.)

- **URL**: https://github.com/composable-models/llm_debate
- **Paper**: Du et al. (2023), arXiv:2305.14325
- **Location**: `code/llm_debate/`
- **Purpose**: Original "Society of Minds" multiagent debate implementation
- **Key Files**: Project website files (HTML/CSS/JS) — code is referenced at project website
- **Notes**: This repo is primarily a project website. The actual code may be at the composable-models GitHub organization.
- **How to Use**: Reference the paper for implementation details; use ReConcile's implementation instead

---

## Repository 4: LLM-TOPLA

- **URL**: https://github.com/git-disl/llm-topla
- **Paper**: Tekin et al. (2024), arXiv:2410.03953
- **Location**: `code/llm_topla/`
- **Purpose**: Diversity-optimized LLM ensemble with focal diversity metric and pruning
- **Key Files**:
  - `topla_weighted.py`: Weighted ensemble for constrained tasks (MMLU, GSM8K)
  - `topla_open_ended.py`: Open-ended generation ensemble (SearchQA)
  - `topla_summary.py`: Summarization ensemble (XSum)
  - `configs.py`: Configuration for model IDs and tasks
  - `helper.py`: Focal diversity metric implementation
  - `data/`: Dataset directory (requires download)
  - `results/`: Pre-computed base model outputs (requires download)
- **Key Features**:
  - **Focal diversity metric**: Core tool for measuring ensemble diversity
  - Diversity-optimized pruning algorithm
  - Learn-to-ensemble for inconsistency resolution
  - Supports 8 diverse LLMs
- **Data Requirements**:
  - MMLU/GSM8K data: Download from Google Drive link in README
  - Pre-computed model outputs: Download from Google Drive link in README
- **How to Use for Our Research**:
  - Use `helper.py` focal diversity metric to measure diversity in our experiments
  - Adapt the pruning algorithm to select optimal heterogeneous subsets
  - Use pre-computed outputs as reference model performance data
- **Requirements**: PyTorch, Transformers, HuggingFace, LLM API access

---

## Repository 5: AgentForest (More Agents Is All You Need)

- **URL**: https://github.com/MoreAgentsIsAllYouNeed/AgentForest
- **Paper**: Li et al. (2024), arXiv:2402.05120
- **Location**: `code/agent_forest/`
- **Purpose**: Sampling-and-voting method demonstrating LLM performance scales with agent count
- **Key Files**:
  - `src/`: Core implementation
  - `script/`: Evaluation scripts
  - `dataset/`: Task datasets
- **Key Features**:
  - Agent Forest (sampling + majority voting)
  - Multiple LLM support (Llama-2, GPT series)
  - Integration with CoT, Zero-Shot CoT, SPP
- **How to Use for Our Research**:
  - Use as the **key homogeneous scaling baseline**
  - Test whether heterogeneous diversity provides gains beyond agent count scaling
  - Compare accuracy curves (agents vs. accuracy) for homogeneous vs. heterogeneous
- **Requirements**: Tencent internal APIs + OpenAI

---

## Repository 6: HeteRobust

- **URL**: https://github.com/git-disl/HeteRobust
- **Paper**: Wu et al. (2023), arXiv:2310.02237
- **Location**: `code/heterobust/`
- **Purpose**: Heterogeneous ensemble robustness via negative correlation analysis
- **Key Files**: Object detection / semantic segmentation ensemble code
- **Key Features**:
  - Negative correlation ensemble theory
  - Connected component labeling for cross-architecture alignment
  - Adversarial robustness evaluation
- **How to Use for Our Research**:
  - Adapt the **negative correlation metric** to measure LLM error correlations
  - Use the theoretical framework to justify expected behavior of heterogeneous LLM ensembles
  - Translate diversity metrics from vision domain to language domain
- **Note**: This is a vision domain repo; methodology transfers conceptually to LLMs

---

## Summary Table

| Name | Purpose | Directly Usable | Key Contribution |
|------|---------|-----------------|-----------------|
| multi_agents_debate | MAD baseline | Yes (with API keys) | Homogeneous debate reference |
| reconcile | Heterogeneous multi-model | Yes (with API keys) | Diverse LLM discussion |
| llm_debate | Du et al. reference | Partial (website only) | Original debate paper |
| llm_topla | Diversity metrics + ensemble | Yes | Focal diversity metric |
| agent_forest | Homogeneous scaling | Yes | Agent count scaling baseline |
| heterobust | Diversity theory | Conceptually | Negative correlation analysis |

---

## Recommended Implementation Strategy

For the experiment runner, we recommend:

1. **Start with ReConcile** as the framework backbone (already supports heterogeneous models)
2. **Add the focal diversity metric** from LLM-TOPLA to measure ensemble diversity
3. **Use AgentForest** as the homogeneous baseline
4. **Adapt the negative correlation analysis** from HeteRobust for LLM error analysis
5. **Extend datasets** with adversarially-designed tasks using the ARC-Challenge and GSM8K formats
