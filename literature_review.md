# Literature Review: Architectural Diversity in Multi-Agent LLM Systems for Robust Collective Reasoning

## Research Area Overview

This review covers the intersection of multi-agent LLM frameworks and diversity-driven robustness in collective reasoning. The core research question is: **Do heterogeneous multi-agent LLM systems (using diverse model families) outperform homogeneous ensembles on adversarially-designed ambiguous tasks, and do they exhibit lower correlated failure rates?**

The field has evolved rapidly since 2023, building on classical ensemble learning theory and extending it to large language models. Key themes include:
1. Multi-agent debate and discussion frameworks
2. Ensemble diversity metrics and their impact on performance
3. Heterogeneous vs. homogeneous model combinations
4. Adversarial robustness of LLM ensembles
5. Convergence and consensus mechanisms in multi-agent reasoning

---

## Key Papers

### Paper 1: Improving Factuality and Reasoning in Language Models through Multiagent Debate

- **Authors**: Yilun Du, Shuang Li, Antonio Torralba, Joshua Tenenbaum, Igor Mordatch
- **Year**: 2023
- **Source**: arXiv:2305.14325 (NeurIPS 2023)
- **Key Contribution**: Foundational "society of minds" approach where multiple LLM instances propose, debate, and update answers over multiple rounds. Demonstrates that multi-agent debate significantly improves mathematical reasoning and factual validity.
- **Methodology**:
  - Multiple instances of the same LLM (homogeneous) generate independent candidate answers
  - Agents then read and critique others' responses
  - Consensus prompts guide agents to update their answers
  - Process iterates until convergence
  - Uses zero-shot chain-of-thought reasoning throughout
- **Datasets Used**: GSM8K (math), Chess (strategic reasoning), Biography (factual verification), MMLU (general reasoning)
- **Baselines**: Single LLM (GPT-3.5, GPT-4), CoT, Self-Consistency
- **Results**: Multi-agent debate improves accuracy by 11% on GSM8K over single models; improves factual validity by 20%; performance scales with number of agents (from 1 to 7)
- **Key Finding**: Agents are "agreeable" due to RLHF training, which limits debate duration; "stubborn" prompts improve debate quality
- **Code Available**: Yes - https://composable-models.github.io/llm_debate/
- **Relevance**: Foundational paper establishing the multi-agent debate paradigm; primarily uses homogeneous agents, which our research will extend to heterogeneous architectures

### Paper 2: Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate

- **Authors**: Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, et al.
- **Year**: 2023
- **Source**: arXiv:2305.19118 (ACL 2024)
- **Key Contribution**: Identifies the "Degeneration-of-Thought (DoT)" problem with single-model self-reflection and introduces Multi-Agent Debate (MAD) as the solution. Key insight: once an LLM establishes confidence, self-reflection fails to generate novel thoughts.
- **Methodology**:
  - Two agents in "tit for tat" debate (affirmative vs. negative)
  - A judge LLM manages the debate and determines final answer
  - Adaptive break mechanism stops debate when consensus emerges
  - Tests heterogeneous models (found that different LLMs as judges can introduce bias)
- **Datasets Used**: Counter-intuitive arithmetic reasoning, Commonsense Machine Translation (trans-MT)
- **Baselines**: Self-reflection, zero-shot CoT, single agent
- **Results**: MAD outperforms self-reflection on both tasks; modest level of debate intensity yields best results
- **Key Finding**: "LLMs might not be a fair judge if different LLMs are used for agents" — diversity can also introduce bias
- **Code Available**: Yes - https://github.com/Skytliang/Multi-Agents-Debate
- **Relevance**: Directly relevant; first systematic study of multi-agent debate with early exploration of heterogeneous models

### Paper 3: ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs

- **Authors**: Justin Chih-Yao Chen, Swarnadeep Saha, Mohit Bansal
- **Year**: 2023
- **Source**: arXiv:2309.13007 (ACL 2024)
- **Key Contribution**: RECONCILE — the first multi-model (heterogeneous) multi-agent framework with confidence-weighted voting, convincingness demonstration, and round-table discussion. Explicitly demonstrates that **diversity originating from different models is critical** to superior performance.
- **Methodology**:
  - Round-table conference with diverse LLM agents (GPT-3.5, PaLM-2, Claude)
  - Discussion prompts include: grouped answers, confidence scores, demonstrations of corrective explanations
  - Confidence-weighted voting mechanism for consensus
  - Multiple rounds of discussion with convincingness learning
- **Datasets Used**: 7 benchmarks: StrategyQA, ECQA, GSM8K, AQuA, Saycan, MATH, plus additional evaluations
- **Baselines**: Self-Refine, Self-Consistency, Du et al. (2023) Debate, GPT-4 single agent
- **Results**: +11.4% over best baseline on CommonsenseQA; outperforms GPT-4 on StrategyQA (+3.4%); +8% on MATH (even beating GPT-4 and DeepSeekMath)
- **Key Finding**: "Diversity originating from different models is critical to its superior performance" — BERTScore-based diversity metric confirms this
- **Code Available**: Yes - https://github.com/dinobby/ReConcile
- **Relevance**: **Most directly relevant** to our hypothesis; proves diverse LLMs outperform homogeneous ensembles and provides the confidence-weighted voting approach we should baseline against

### Paper 4: Can LLM Agents Really Debate? A Controlled Study of Multi-Agent Debate in Logical Reasoning

- **Authors**: Haolun Wu, Zhenkun Li, Lingyao Li
- **Year**: 2025
- **Source**: arXiv:2511.07784
- **Key Contribution**: Controlled study disentangling the effects of various structural and cognitive factors on MAD outcomes using Knight-Knave-Spy logic puzzles with verifiable ground truth.
- **Methodology**:
  - Custom Knight-Knave-Spy dataset: 1,800 puzzles (4-9 players, 300 per size)
  - Six factors studied: team size, composition (diversity), confidence visibility, debate order, debate depth, task difficulty
  - Player-by-player debate loop with initial proposal, debate phase, and reflection phase
  - Process-level analysis of behavioral patterns
- **Datasets Used**: Custom Knight-Knave-Spy dataset (released with paper)
- **Baselines**: Single agents of various models, homogeneous teams, majority voting baselines
- **Results**: Group diversity and intrinsic reasoning strength dominate; structural parameters (order, confidence visibility) offer minimal gains; majority pressure suppresses independent correction
- **Key Finding**: "Intrinsic reasoning strength and group diversity are the dominant drivers of debate success"; diverse teams effectively overturn incorrect consensus while homogeneous teams create echo chambers
- **Code Available**: Yes (link in paper)
- **Relevance**: Provides the most rigorous controlled experimental framework for testing our diversity hypothesis; the Knight-Knave-Spy puzzle is an ideal adversarial task

### Paper 5: Harnessing Multiple Large Language Models: A Survey on LLM Ensemble

- **Authors**: Zhijun Chen, Jingzheng Li, et al.
- **Year**: 2025
- **Source**: arXiv:2502.18036
- **Key Contribution**: First systematic survey of LLM ensemble methods with taxonomy: ensemble-before-inference, ensemble-during-inference, ensemble-after-inference.
- **Taxonomy**:
  - Before: Routing (classification-based, reward-based, assignment-based)
  - During: Token-level, span-level, process-level ensemble
  - After: Non-cascade (selection, selection+regeneration) and Cascade methods
- **Key Methods Surveyed**: LLM-Blender, Mixtral, More Agents (AgentForest), LLM-TOPLA, FrugalGPT, etc.
- **Relevance**: Essential reference for understanding the landscape of LLM ensemble approaches; provides experimental methodology and evaluation metrics used across the field

### Paper 6: Diverse LLMs or Diverse Question Interpretations? That is the Ensembling Question

- **Authors**: Rafael Rosales, Santiago Miret (Intel Labs)
- **Year**: 2025
- **Source**: arXiv:2507.21168
- **Key Contribution**: Compares model diversity (multiple models answering same question) vs. question interpretation diversity (same model answering rephrased question) — finds that question interpretation diversity often outperforms model diversity!
- **Methodology**:
  - Binary question answering with majority voting
  - GPT-4 and Llama model families
  - 3 datasets: BoolQ, StrategyQA, PubMedQA
- **Results**: Question interpretation diversity consistently leads to better accuracy; model diversity typically produces results between best and worst individual members without clear improvement
- **Key Finding**: For model diversity to work, models need truly complementary failure modes (negatively correlated errors), not just independence
- **Datasets Used**: BoolQ, StrategyQA, PubMedQA
- **Relevance**: Critical challenge to the naive model diversity hypothesis; suggests that architectural diversity alone is insufficient — failure mode correlation is key

### Paper 7: LLM-TOPLA: Efficient LLM Ensemble by Maximising Diversity

- **Authors**: Selim Furkan Tekin, Fatih Ilhan, Tiansheng Huang, Sihao Hu, Ling Liu (Georgia Tech)
- **Year**: 2024
- **Source**: arXiv:2410.03953
- **Key Contribution**: Introduces "focal diversity metric" for quantifying diversity-performance correlation; diversity-optimized ensemble pruning to select top-k sub-ensembles; learn-to-ensemble for resolving output inconsistency.
- **Methodology**:
  - Focal diversity metric: captures diversity-performance correlation among ensemble members
  - Pruning algorithm: recommends top-performing sub-ensembles of size S from pool of N LLMs
  - Learn-to-ensemble: detects and resolves output inconsistency
- **Datasets Used**: MMLU, GSM8K, SearchQA, XSum (8 models: phi-2, Mixtral, LLaMA variants, GPT-3.5)
- **Results**: +2.2% on MMLU over Mixtral; +2.1% on GSM8K over AgentForest; +3.9x on SearchQA F1
- **Code Available**: Yes - https://github.com/git-disl/llm-topla
- **Relevance**: Provides the diversity metric we should use to measure ensemble diversity; architecture directly applicable to heterogeneous multi-agent systems

### Paper 8: More Agents Is All You Need

- **Authors**: Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, Deheng Ye (Tencent)
- **Year**: 2024
- **Source**: arXiv:2402.05120 (TMLR 2024)
- **Key Contribution**: Demonstrates that LLM performance scales with the number of agents in a sampling-and-voting framework (Agent Forest). This scaling is orthogonal to other methods and correlates with task difficulty.
- **Methodology**:
  - Sampling-and-voting (Agent Forest) method
  - Evaluate across many LLMs (Llama-2 13B/70B, GPT-3.5, GPT-4) and tasks
  - Integrate with CoT, Zero-Shot CoT, SPP
- **Datasets Used**: GSM8K, MATH, MMLU, Chess (Du et al.), HumanEval
- **Results**: Performance improves monotonically with number of agents; effect is stronger on harder tasks; similar LLMs at larger scale can match heterogeneous ensembles
- **Key Finding**: Agent count matters even for homogeneous agents; provides strong baseline for comparing diverse vs. non-diverse ensembles
- **Code Available**: Yes - https://github.com/MoreAgentsIsAllYouNeed/AgentForest
- **Relevance**: Important homogeneous ensemble baseline; allows measuring marginal benefit of diversity beyond mere scale

### Paper 9: Exploring Model Learning Heterogeneity for Boosting Ensemble Robustness

- **Authors**: Yanzhao Wu, Ka-Ho Chow, Wenqi Wei, Ling Liu (Georgia Tech)
- **Year**: 2023
- **Source**: arXiv:2310.02237
- **Key Contribution**: Formal analysis showing that heterogeneous deep ensembles with high diversity can boost robustness against both natural errors and adversarial attacks via negative correlation analysis.
- **Methodology**:
  - Two-tier heterogeneity: within-task (object detection) and cross-task (detection + segmentation)
  - Weighted bounding box ensemble consensus
  - Connected component labeling (CCL) for cross-task alignment
  - Negative correlation as the formal metric for diversity benefit
- **Datasets Used**: COCO, VOC (object detection/segmentation)
- **Results**: Heterogeneous ensembles demonstrate stronger adversarial robustness and better generalization than homogeneous ones; negative correlation between member errors is key
- **Key Finding**: "Heterogeneous DNN models trained for solving the same learning problem can significantly strengthen performance"; formal analysis via negative correlation learning theory
- **Code Available**: Yes - https://github.com/git-disl/HeteRobust
- **Relevance**: Provides the theoretical foundation for heterogeneous diversity; negative correlation analysis should be applied to LLM ensembles in our experiments

### Paper 10: Stable LLM Ensemble: Interaction between Example Representativeness and Diversity

- **Authors**: Junichiro Niimi (Meijo University)
- **Year**: 2025
- **Source**: arXiv:2510.13143
- **Key Contribution**: Shows that combining representative example selection (centroid-based) with controlled temperature diversity outperforms random selection by +7.6% (macro-F1) and exceeds 5-shot prompting by +21.1%.
- **Methodology**:
  - Centroid-based vs. random one-shot example selection
  - Temperature variation for output diversity
  - Ensemble aggregation with majority voting
- **Datasets Used**: Sentiment analysis, classification (various NLP tasks)
- **Results**: Higher temperature + centroid-based examples >> random selection; controlled diversity is better than unconstrained diversity
- **Relevance**: Highlights that diversity needs to be calibrated; too much diversity (high temperature, random prompts) can hurt; architectural diversity must be paired with appropriate calibration

### Paper 11: MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework

- **Authors**: Sirui Hong, Mingchen Zhuge, Jiaqi Chen et al. (DeepWisdom + KAUST)
- **Year**: 2023 (ICLR 2024)
- **Source**: arXiv:2308.00352
- **Key Contribution**: Introduces SOPs (Standardized Operating Procedures) for multi-agent LLM systems; assembly-line paradigm for complex software engineering tasks.
- **Methodology**:
  - Role-based agent assignment (Product Manager, Developer, QA Engineer, etc.)
  - Standardized output formats for each role
  - Structured message passing between agents
- **Datasets Used**: Software engineering benchmarks (HumanEval, SWE-bench)
- **Code Available**: Yes - https://github.com/geekan/MetaGPT
- **Relevance**: Key reference for multi-agent coordination architectures; role diversity vs. model architecture diversity distinction

### Paper 12: Towards Effective GenAI Multi-Agent Collaboration (Enterprise Applications)

- **Authors**: Raphael Shu, Nilaksh Das, Michelle Yuan, Monica Sunkara, Yi Zhang (AWS Bedrock)
- **Year**: 2024
- **Source**: arXiv:2412.05449
- **Key Contribution**: Empirical evaluation of multi-agent collaboration at enterprise scale showing +70% improvement over single-agent approaches; introduces coordination mode and routing mode.
- **Methodology**:
  - Coordination mode: parallel communication with payload referencing
  - Routing mode: efficient message forwarding
  - 3 enterprise domains with handcrafted scenarios
- **Results**: 90% end-to-end goal success with multi-agent; +70% vs. single-agent; payload referencing improves code tasks by 23%
- **Relevance**: Practical evidence for multi-agent superiority in complex tasks; establishes enterprise use case motivation

### Paper 13: Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking

- **Authors**: Priyanka Kargupta, Ishika Agarwal, Tal August et al.
- **Year**: 2025
- **Source**: arXiv:2502.14767
- **Key Contribution**: Extends debate to a tree structure with multiple personas for scientific comparison tasks; addresses exponential growth of research requiring comparative analysis.
- **Methodology**:
  - Tree of debate with branching arguments
  - Multiple distinct personas per LLM agent
  - Critical thinking elicitation for scientific comparison
- **Relevance**: Shows how structural diversity (roles/personas) complements model diversity

### Paper 14: SWE-Debate: Competitive Multi-Agent Debate for Software Issue Resolution

- **Authors**: Han Li, Yuling Shi, Shaoxin Lin et al.
- **Year**: 2025
- **Source**: arXiv:2507.23348
- **Key Contribution**: Applies competitive debate to software engineering — agents independently generate solutions and critique each other before merging, improving SWE-bench resolution rates.
- **Relevance**: Shows multi-agent debate effectiveness in specialized domains; provides evaluation framework on concrete coding tasks

### Paper 15: From System 1 to System 2: A Survey of Reasoning Large Language Models

- **Authors**: Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang et al.
- **Year**: 2025
- **Source**: arXiv:2502.17419
- **Key Contribution**: Comprehensive survey of reasoning in LLMs from fast intuitive (System 1) to deliberate analytical (System 2) reasoning; covers CoT, tree search, multi-agent approaches.
- **Relevance**: Contextualizes multi-agent debate within the broader reasoning landscape; clarifies where architectural diversity provides unique benefits

---

## Common Methodologies

1. **Multi-Round Debate**: Used in Du et al. (2023), Liang et al. (2023), Chen et al. (2023), Wu et al. (2025). Agents iteratively update answers based on others' responses.

2. **Sampling-and-Voting / Majority Voting**: Used in Wang et al. (2023, Self-Consistency), Li et al. (2024, More Agents), Rosales et al. (2025). Simple ensemble baseline.

3. **Confidence-Weighted Voting**: Used in ReConcile (Chen et al., 2023), Stable LLM Ensemble (Niimi, 2025). Weights agent votes by expressed confidence.

4. **Diversity-Optimized Ensemble Pruning**: Used in LLM-TOPLA (Tekin et al., 2024). Select optimal sub-ensemble from larger pool based on diversity metrics.

5. **Negative Correlation Analysis**: Used in Wu et al. (2023, HeteRobust). Theoretical framework linking diversity to ensemble robustness.

---

## Standard Baselines

- **Single Agent (CoT)**: Standard chain-of-thought with single LLM; ubiquitous baseline
- **Self-Consistency** (Wang et al., 2023): Multiple samples from same model + majority voting; critical homogeneous ensemble baseline
- **Self-Refine/Self-Reflection** (Madaan et al., 2023): Single model iterative refinement with own feedback
- **Homogeneous Debate** (Du et al., 2023): Multiple instances of same model in debate; key comparison for diversity hypothesis
- **Agent Forest / More Agents** (Li et al., 2024): Scaling homogeneous sampling-voting
- **GPT-4 Single Agent**: Strong single-model baseline representing best available model

---

## Evaluation Metrics

- **Accuracy**: Most common for classification/QA tasks (MMLU, ARC, GSM8K)
- **F1 Score**: For open-ended generation tasks
- **ROUGE**: For summarization tasks
- **Error Correlation**: Pairwise agreement/disagreement between ensemble members (diversity metric)
- **BERTScore Diversity**: Used in ReConcile to quantify output diversity
- **Focal Diversity Metric**: Used in LLM-TOPLA (Tekin et al., 2024)
- **Negative Correlation**: Theoretical metric from Wu et al. (2023)
- **Convergence Rate**: Rounds to reach consensus in debate

---

## Datasets Used in Literature

| Dataset | Task | Size | Used In |
|---------|------|------|---------|
| GSM8K | Math reasoning | 7,473 train / 1,319 test | Du2023, Chen2023, Li2024, Tekin2024 |
| MMLU | General knowledge (57 subjects) | 14K test | Du2023, Li2024, Tekin2024 |
| StrategyQA | Commonsense reasoning | ~2K | Chen2023, Rosales2025 |
| BoolQ | Binary QA | 9.4K train / 3.3K val | Rosales2025 |
| ARC-Challenge | Science QA (multi-choice) | 1,119 train / 1,172 test | Standard benchmark |
| CommonsenseQA | Commonsense reasoning | 9.7K train / 1.2K val | Chen2023 |
| AQuA-RAT | Algebraic reasoning | ~100K | Chen2023 |
| HumanEval | Code generation | 164 problems | Li2024 |
| Knight-Knave-Spy | Logical deduction | 1,800 puzzles | Wu2025 |

---

## Gaps and Opportunities

1. **Heterogeneous vs. Homogeneous Comparison at Scale**: Most papers either study homogeneous debate (Du et al., More Agents) or use a handful of heterogeneous models without systematic ablation of diversity level. Rigorous comparison across diversity levels is lacking.

2. **Adversarial/Ambiguous Task Design**: Most benchmarks test factual knowledge or straightforward reasoning. Deliberately adversarial or ambiguous tasks (designed to expose model biases) are underexplored.

3. **Inductive Bias Diversity**: No study specifically examines how diversity in training paradigms (instruction tuning vs. RLHF vs. pre-training only) or architecture family (transformer variants, state space models) affects collective reasoning.

4. **Correlated Error Quantification**: While Wu et al. (2023) provide negative correlation analysis for DNNs, systematic measurement of error correlation among LLM ensembles is missing.

5. **Convergence Speed vs. Accuracy Trade-off**: How does diversity affect convergence speed in debate? Diverse teams might take longer to converge but reach better answers.

6. **Adversarial Robustness**: Chen et al. (2023) and Wu et al. (2025) show improved robustness, but adversarial prompting specifically designed to exploit shared biases of homogeneous models is unstudied.

---

## Recommendations for Our Experiment

**Recommended Datasets**:
1. **Primary**: GSM8K (math reasoning - well-understood, fast evaluation), MMLU (diverse knowledge)
2. **Secondary**: Knight-Knave-Spy-style logic puzzles (controlled, verifiable), adversarially-crafted ambiguous questions
3. **Create adversarial dataset**: Generate questions specifically designed to exploit known biases of individual model families (e.g., positional bias in GPT, format biases in Claude, frequency biases in LLaMA)

**Recommended Baselines** (in order of importance):
1. Single best agent (GPT-4o or Claude 3.5 Sonnet)
2. Self-Consistency (same model, multiple samples, majority vote)
3. Homogeneous debate (Du et al. 2023 approach with same model family)
4. More Agents / Agent Forest (large homogeneous ensemble)
5. ReConcile (heterogeneous, confidence-weighted — closest to our approach)

**Recommended Metrics**:
1. Task accuracy (primary)
2. Pairwise error correlation between agents (key diversity metric)
3. Convergence rate (rounds to consensus)
4. Answer consistency across perturbations (adversarial robustness)
5. Diversity score (focal diversity metric from LLM-TOPLA)

**Methodological Considerations**:
- Use at least 3 distinct model families (GPT, Claude, LLaMA/Mistral) for heterogeneous condition
- Control for total compute: compare N heterogeneous models vs. N*k homogeneous samples
- Measure diversity at multiple levels: answer diversity, reasoning chain diversity, error correlation
- Use adversarial perturbations (paraphrasing, answer-swapping, distractor insertion) to test robustness
- Report both accuracy AND error correlation to support the hypothesis mechanistically
- Consider confidence calibration: heterogeneous models may have different confidence scales
