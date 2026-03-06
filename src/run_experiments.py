"""
Main experiment runner.

Runs all 5 experimental conditions on all 30 adversarial questions
plus 6 paraphrase variants.

Saves results to results/raw_results.json.
"""

import os
import json
import time
import logging
import random
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adversarial_benchmark import get_all_original_questions, get_paraphrase_variants
from multi_agent_framework import (
    get_client,
    run_single_agent,
    run_homogeneous_ensemble,
    run_heterogeneous_ensemble,
    run_debate_protocol,
    run_confidence_weighted_heterogeneous,
    HAIKU_MODEL, SONNET_MODEL
)

random.seed(42)

logger = logging.getLogger(__name__)

RESULTS_DIR = "/workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_all_conditions_on_question(client, question: dict, include_debate: bool = True) -> dict:
    """Run all experimental conditions on a single question."""
    qid = question["id"]
    logger.info(f"Q{qid} ({question['category']}): {question['question'][:60]}...")

    result = {
        "question_id": qid,
        "category": question["category"],
        "question": question["question"],
        "correct_answer": question["correct_answer"],
        "conditions": {}
    }

    # 1. Single agent - haiku
    logger.info(f"  [1/5] Single haiku")
    r = run_single_agent(client, HAIKU_MODEL, question)
    result["conditions"]["single_haiku"] = {
        "answer": r["extracted_answer"],
        "is_correct": r["is_correct"],
        "response": r["raw_response"]
    }
    time.sleep(0.3)

    # 2. Single agent - sonnet
    logger.info(f"  [2/5] Single sonnet")
    r = run_single_agent(client, SONNET_MODEL, question)
    result["conditions"]["single_sonnet"] = {
        "answer": r["extracted_answer"],
        "is_correct": r["is_correct"],
        "response": r["raw_response"]
    }
    time.sleep(0.3)

    # 3. Homogeneous ensemble - 3x haiku
    logger.info(f"  [3/5] Homogeneous 3x haiku")
    r = run_homogeneous_ensemble(client, HAIKU_MODEL, question, n=3)
    result["conditions"]["homogeneous_haiku"] = {
        "answer": r["voted_answer"],
        "is_correct": r["is_correct"],
        "individual_answers": r["individual_answers"],
        "consensus": r["consensus"]
    }
    time.sleep(0.5)

    # 4. Heterogeneous ensemble - 2x haiku + 1x sonnet
    logger.info(f"  [4/5] Heterogeneous 2h+1s")
    r = run_heterogeneous_ensemble(client, question, haiku_count=2, sonnet_count=1)
    result["conditions"]["heterogeneous_2h1s"] = {
        "answer": r["voted_answer"],
        "is_correct": r["is_correct"],
        "individual_answers": r["individual_answers"],
        "agent_models": r["agent_models"],
        "consensus": r["consensus"]
    }
    time.sleep(0.5)

    # 5. Debate: haiku proposes, sonnet critiques, haiku revises
    if include_debate:
        logger.info(f"  [5/5] Debate protocol")
        r = run_debate_protocol(client, question)
        result["conditions"]["debate"] = {
            "answer": r["haiku_answer_final"],
            "is_correct": r["is_correct"],
            "initial_answer": r["haiku_answer_initial"],
            "initial_correct": r["initial_was_correct"],
            "changed": r["debate_changed_answer"],
            "critique_excerpt": r["sonnet_critique"][:200]
        }
        time.sleep(0.5)

    return result


def compute_accuracy_summary(all_results: list) -> dict:
    """Compute per-condition accuracy across all questions."""
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku",
                  "heterogeneous_2h1s", "debate"]
    summary = {c: {"correct": 0, "total": 0} for c in conditions}

    for result in all_results:
        for cond, data in result["conditions"].items():
            if cond in summary:
                summary[cond]["total"] += 1
                if data.get("is_correct"):
                    summary[cond]["correct"] += 1

    for cond in summary:
        n = summary[cond]["total"]
        c = summary[cond]["correct"]
        summary[cond]["accuracy"] = c / n if n > 0 else 0

    return summary


def main():
    logger.info("=" * 60)
    logger.info("Starting Multi-Agent Diversity Experiment")
    logger.info(f"Time: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    client = get_client()

    original_questions = get_all_original_questions()
    paraphrase_variants = get_paraphrase_variants()

    logger.info(f"Original questions: {len(original_questions)}")
    logger.info(f"Paraphrase variants: {len(paraphrase_variants)}")

    # =====================================================
    # PHASE 1: Run all conditions on original questions
    # =====================================================
    logger.info("\n=== PHASE 1: Original Questions ===")
    original_results = []

    for i, question in enumerate(original_questions):
        logger.info(f"\n--- Question {i+1}/{len(original_questions)} ---")
        try:
            result = run_all_conditions_on_question(client, question, include_debate=True)
            original_results.append(result)

            # Save incrementally
            with open(f"{RESULTS_DIR}/raw_results_partial.json", "w") as f:
                json.dump(original_results, f, indent=2)

        except Exception as e:
            logger.error(f"Error on question {question['id']}: {e}")
            original_results.append({
                "question_id": question["id"],
                "category": question["category"],
                "error": str(e),
                "conditions": {}
            })

    # =====================================================
    # PHASE 2: Run paraphrase variants (single+homogeneous+heterogeneous only)
    # =====================================================
    logger.info("\n=== PHASE 2: Paraphrase Variants (Robustness Test) ===")
    paraphrase_results = []

    for i, question in enumerate(paraphrase_variants):
        logger.info(f"\n--- Paraphrase {i+1}/{len(paraphrase_variants)} ---")
        try:
            result = run_all_conditions_on_question(client, question, include_debate=False)
            paraphrase_results.append(result)

            # Save incrementally
            with open(f"{RESULTS_DIR}/paraphrase_results_partial.json", "w") as f:
                json.dump(paraphrase_results, f, indent=2)

        except Exception as e:
            logger.error(f"Error on paraphrase {question['id']}: {e}")
            paraphrase_results.append({
                "question_id": question["id"],
                "error": str(e),
                "conditions": {}
            })

    # =====================================================
    # PHASE 3: Compute summaries
    # =====================================================
    logger.info("\n=== PHASE 3: Computing Summaries ===")

    orig_summary = compute_accuracy_summary(original_results)
    para_summary = compute_accuracy_summary(paraphrase_results)

    logger.info("\nOriginal Questions - Accuracy by Condition:")
    for cond, stats in orig_summary.items():
        logger.info(f"  {cond}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")

    logger.info("\nParaphrase Variants - Accuracy by Condition:")
    for cond, stats in para_summary.items():
        if stats['total'] > 0:
            logger.info(f"  {cond}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.1%}")

    # Per-category breakdown for original questions
    categories = list(set(r["category"] for r in original_results if "category" in r))
    cat_results = {}
    for cat in categories:
        cat_qs = [r for r in original_results if r.get("category") == cat]
        cat_results[cat] = compute_accuracy_summary(cat_qs)

    # Compute error correlation matrices
    # For pairs of agents: how often do they both get wrong?
    error_correlations = compute_error_correlations(original_results)

    # Save final results
    final_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "haiku_model": HAIKU_MODEL,
            "sonnet_model": SONNET_MODEL,
            "n_original_questions": len(original_results),
            "n_paraphrase_variants": len(paraphrase_results)
        },
        "original_results": original_results,
        "paraphrase_results": paraphrase_results,
        "summaries": {
            "original": orig_summary,
            "paraphrase": para_summary,
            "by_category": cat_results
        },
        "error_correlations": error_correlations
    }

    output_path = f"{RESULTS_DIR}/raw_results.json"
    with open(output_path, "w") as f:
        json.dump(final_output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info("Experiment complete!")

    return final_output


def compute_error_correlations(results: list) -> dict:
    """
    Compute pairwise error correlation between agent types.
    An error vector is: 1 if wrong, 0 if correct.
    """
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku", "heterogeneous_2h1s"]
    error_vectors = {c: [] for c in conditions}

    for result in results:
        for cond in conditions:
            if cond in result.get("conditions", {}):
                is_correct = result["conditions"][cond].get("is_correct", False)
                error_vectors[cond].append(0 if is_correct else 1)

    # Compute pairwise correlations
    import numpy as np
    correlations = {}
    for i, c1 in enumerate(conditions):
        for c2 in conditions[i:]:
            v1 = error_vectors[c1]
            v2 = error_vectors[c2]
            if len(v1) == len(v2) and len(v1) > 1:
                try:
                    corr = float(np.corrcoef(v1, v2)[0, 1])
                except Exception:
                    corr = 0.0
                correlations[f"{c1}_vs_{c2}"] = corr

    # Also compute individual error rates
    error_rates = {}
    for cond in conditions:
        v = error_vectors[cond]
        error_rates[cond] = sum(v) / len(v) if v else 0

    return {
        "pairwise_correlations": correlations,
        "error_rates": error_rates,
        "error_vectors": {k: v for k, v in error_vectors.items()}
    }


if __name__ == "__main__":
    main()
