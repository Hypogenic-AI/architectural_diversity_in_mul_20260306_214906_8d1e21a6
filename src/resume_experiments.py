"""Resume experiment for questions 15-30 and all paraphrases."""

import os, json, time, logging, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adversarial_benchmark import get_all_original_questions, get_paraphrase_variants
from multi_agent_framework import (
    get_client, run_single_agent, run_homogeneous_ensemble,
    run_heterogeneous_ensemble, run_debate_protocol,
    HAIKU_MODEL, SONNET_MODEL
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6/logs/resume.log"),
        logging.StreamHandler()
    ])

RESULTS_DIR = "/workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6/results"


def run_all_conditions(client, question, include_debate=True):
    qid = question["id"]
    logger.info(f"Q{qid} ({question['category']}): {question['question'][:60]}...")
    result = {
        "question_id": qid, "category": question["category"],
        "question": question["question"], "correct_answer": question["correct_answer"],
        "conditions": {}
    }
    # Single haiku
    r = run_single_agent(client, HAIKU_MODEL, question)
    result["conditions"]["single_haiku"] = {"answer": r["extracted_answer"], "is_correct": r["is_correct"], "response": r["raw_response"]}
    time.sleep(0.3)
    # Single sonnet
    r = run_single_agent(client, SONNET_MODEL, question)
    result["conditions"]["single_sonnet"] = {"answer": r["extracted_answer"], "is_correct": r["is_correct"], "response": r["raw_response"]}
    time.sleep(0.3)
    # Homogeneous 3x haiku
    r = run_homogeneous_ensemble(client, HAIKU_MODEL, question, n=3)
    result["conditions"]["homogeneous_haiku"] = {"answer": r["voted_answer"], "is_correct": r["is_correct"], "individual_answers": r["individual_answers"], "individual_responses": r["individual_responses"], "consensus": r["consensus"]}
    time.sleep(0.5)
    # Heterogeneous 2h+1s
    r = run_heterogeneous_ensemble(client, question, haiku_count=2, sonnet_count=1)
    result["conditions"]["heterogeneous_2h1s"] = {"answer": r["voted_answer"], "is_correct": r["is_correct"], "individual_answers": r["individual_answers"], "individual_responses": r["individual_responses"], "agent_models": r["agent_models"], "consensus": r["consensus"]}
    time.sleep(0.5)
    # Debate
    if include_debate:
        r = run_debate_protocol(client, question)
        result["conditions"]["debate"] = {
            "answer": r["haiku_answer_final"], "is_correct": r["is_correct"],
            "initial_answer": r["haiku_answer_initial"], "initial_correct": r["initial_was_correct"],
            "changed": r["debate_changed_answer"], "critique_excerpt": r["sonnet_critique"][:200]
        }
        time.sleep(0.5)
    return result


def main():
    # Load existing results
    with open(f"{RESULTS_DIR}/raw_results.json") as f:
        data = json.load(f)

    existing_orig = data["original_results"]
    completed_ids = {r["question_id"] for r in existing_orig if "error" not in r and r.get("conditions")}
    logger.info(f"Already completed: {sorted(completed_ids)}")

    client = get_client()

    # Resume original questions 15-30
    all_orig = get_all_original_questions()
    remaining = [q for q in all_orig if q["id"] not in completed_ids]
    logger.info(f"Remaining original questions: {len(remaining)}")

    new_results = []
    for i, q in enumerate(remaining):
        logger.info(f"\n--- Q{q['id']} ({i+1}/{len(remaining)}) ---")
        try:
            result = run_all_conditions(client, q, include_debate=True)
            new_results.append(result)
            # Add to data and save incrementally
            data["original_results"].append(result)
            with open(f"{RESULTS_DIR}/raw_results.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error on Q{q['id']}: {e}")
            data["original_results"].append({"question_id": q["id"], "category": q["category"], "error": str(e), "conditions": {}})

    # Run paraphrase variants
    paraphrases = get_paraphrase_variants()
    logger.info(f"\n=== Running {len(paraphrases)} paraphrase variants ===")
    para_results = []
    for i, q in enumerate(paraphrases):
        logger.info(f"\n--- Para {q['id']} ({i+1}/{len(paraphrases)}) ---")
        try:
            result = run_all_conditions(client, q, include_debate=False)
            para_results.append(result)
            data["paraphrase_results"] = para_results
            with open(f"{RESULTS_DIR}/raw_results.json", "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error on para {q['id']}: {e}")
            para_results.append({"question_id": q["id"], "error": str(e), "conditions": {}})

    # Final save
    data["paraphrase_results"] = para_results
    with open(f"{RESULTS_DIR}/raw_results.json", "w") as f:
        json.dump(data, f, indent=2)

    # Summary
    valid_orig = [r for r in data["original_results"] if "error" not in r and r.get("conditions")]
    logger.info(f"\nTotal valid original: {len(valid_orig)}/30")
    for cond in ["single_haiku", "single_sonnet", "homogeneous_haiku", "heterogeneous_2h1s", "debate"]:
        vec = [r["conditions"][cond]["is_correct"] for r in valid_orig if cond in r.get("conditions", {})]
        if vec:
            logger.info(f"  {cond}: {sum(vec)}/{len(vec)} = {sum(vec)/len(vec):.1%}")

    logger.info("Resume complete!")


if __name__ == "__main__":
    main()
