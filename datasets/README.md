# Datasets for Multi-Agent LLM Diversity Research

This directory contains datasets for evaluating multi-agent LLM systems with architectural diversity.
Data files are excluded from git due to size. Follow download instructions below.

## Dataset 1: GSM8K (Grade School Math)

### Overview
- **Source**: HuggingFace: `openai/gsm8k`
- **Size**: 7,473 train / 1,319 test
- **Format**: HuggingFace Dataset (Parquet)
- **Task**: Mathematical word problem reasoning (chain-of-thought)
- **License**: MIT
- **Used in**: Du et al. (2023), Chen et al. (2023), Li et al. (2024), Tekin et al. (2024)

### Why Relevant
GSM8K is the most commonly used benchmark in multi-agent debate papers. It tests multi-step mathematical reasoning where diverse agents can meaningfully contribute different solution approaches. Clear ground truth enables exact accuracy measurement.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main")
dataset.save_to_disk("datasets/gsm8k")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/gsm8k")
train = dataset['train']
test = dataset['test']
```

### Sample Data
```json
[
  {
    "question": "Natalia sold clips to 48 of her friends in April...",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
  }
]
```

---

## Dataset 2: MMLU (Massive Multitask Language Understanding)

### Overview
- **Source**: HuggingFace: `cais/mmlu`, subset: `all`
- **Size**: 14,042 test / 1,531 validation / 285 dev / 99,842 auxiliary train
- **Format**: HuggingFace Dataset (Parquet)
- **Task**: Multiple-choice knowledge across 57 subjects
- **Used in**: Du et al. (2023), Li et al. (2024), Tekin et al. (2024)

### Why Relevant
MMLU tests breadth of knowledge across diverse domains. Different LLM architectures trained on different data distributions will have different subject-matter strengths — making it ideal for testing whether heterogeneous ensembles can cover each other's blind spots.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("cais/mmlu", "all")
dataset.save_to_disk("datasets/mmlu")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mmlu")
test = dataset['test']
# Fields: question, choices (list of 4), answer (int 0-3), subject
```

---

## Dataset 3: BoolQ (Boolean Question Answering)

### Overview
- **Source**: HuggingFace: `google/boolq`
- **Size**: 9,427 train / 3,270 validation
- **Format**: HuggingFace Dataset (Parquet)
- **Task**: Binary (True/False) question answering with passage context
- **Used in**: Rosales & Miret (2025)

### Why Relevant
Binary QA enables clean measurement of majority voting performance and error correlation between agents. Used in the key Rosales et al. (2025) paper comparing model diversity vs. question interpretation diversity.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("google/boolq")
dataset.save_to_disk("datasets/boolq")
```

### Loading the Dataset
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/boolq")
val = dataset['validation']
# Fields: question, passage, answer (bool)
```

---

## Dataset 4: CommonsenseQA

### Overview
- **Source**: HuggingFace: `tau/commonsense_qa`
- **Size**: 9,741 train / 1,221 validation / 1,140 test
- **Format**: HuggingFace Dataset (Parquet)
- **Task**: 5-choice commonsense reasoning questions
- **Used in**: Chen et al. (2023) via ECQA extension

### Why Relevant
Tests commonsense knowledge where LLMs trained on different data may have different biases and strengths. Multi-choice format allows measuring error correlation directly.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("tau/commonsense_qa")
dataset.save_to_disk("datasets/commonsense_qa")
```

---

## Dataset 5: ARC-Challenge (AI2 Reasoning Challenge)

### Overview
- **Source**: HuggingFace: `allenai/ai2_arc`, config: `ARC-Challenge`
- **Size**: 1,119 train / 1,172 test / 299 validation
- **Format**: HuggingFace Dataset (Parquet)
- **Task**: 4-choice science question answering (challenging set)

### Why Relevant
Tests scientific reasoning requiring genuine knowledge retrieval; harder questions expose differences between model families that may have been pre-trained on different science corpora.

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
dataset.save_to_disk("datasets/arc_challenge")
```

---

## Recommended Custom Dataset: Adversarial Ambiguous Tasks

For testing the core hypothesis about adversarial robustness, we recommend creating an adversarial dataset:

### Design Principles
1. **Bias-exploiting questions**: Questions where model families have known systematic biases (e.g., positional bias, sycophancy, recency bias)
2. **Ambiguous reasoning**: Questions with multiple valid interpretations requiring explicit disambiguation
3. **Adversarial perturbations**: Each question has variants with distractor answers, reversed premises, or misleading context
4. **Ground truth**: Clear, verifiable correct answers despite surface ambiguity

### Creation Script
See `code/create_adversarial_dataset.py` (to be implemented by experiment runner)

### Structure
```json
{
  "question_id": "adv_001",
  "question": "...",
  "correct_answer": "A",
  "adversarial_variant": "...",
  "bias_type": "positional|sycophancy|recency",
  "difficulty": "easy|medium|hard"
}
```

---

## Loading All Datasets

```python
from datasets import load_from_disk

datasets = {
    "gsm8k": load_from_disk("datasets/gsm8k"),
    "mmlu": load_from_disk("datasets/mmlu"),
    "boolq": load_from_disk("datasets/boolq"),
    "commonsense_qa": load_from_disk("datasets/commonsense_qa"),
    "arc_challenge": load_from_disk("datasets/arc_challenge"),
}

# Quick access
for name, ds in datasets.items():
    print(f"{name}: {ds}")
```
