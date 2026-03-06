"""
Multi-agent LLM experiment framework for testing architectural diversity.

Implements 5 experimental conditions:
1. Single agent (haiku or sonnet)
2. Homogeneous ensemble (3x same model, majority vote)
3. Heterogeneous ensemble (2x haiku + 1x sonnet, majority vote)
4. Confidence-weighted heterogeneous (haiku + sonnet, weighted vote)
5. Debate: haiku proposes, sonnet critiques, haiku revises
"""

import os
import json
import time
import random
import re
import logging
from typing import Optional

import anthropic

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("/workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6/logs/experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

random.seed(42)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"


def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


def call_model(
    client: anthropic.Anthropic,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    retries: int = 3
) -> str:
    """Call a model with retry logic."""
    for attempt in range(retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text.strip()
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 5
            logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1})")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"API error on attempt {attempt+1}: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return ""


# System prompt for reasoning agents
REASONING_SYSTEM = (
    "You are a rigorous analytical reasoner. When given a problem:\n"
    "1. Identify any misleading patterns or surface-level shortcuts\n"
    "2. Reason step-by-step from first principles\n"
    "3. Check your answer against the question carefully\n"
    "4. State your FINAL ANSWER clearly at the end, preceded by 'FINAL ANSWER:'\n"
    "Keep your response concise (under 200 words)."
)

# System prompt for confidence-aware responses
CONFIDENCE_SYSTEM = (
    "You are a rigorous analytical reasoner. When given a problem:\n"
    "1. Reason step-by-step from first principles\n"
    "2. State your FINAL ANSWER clearly preceded by 'FINAL ANSWER:'\n"
    "3. After your answer, state your confidence from 0-100 preceded by 'CONFIDENCE:'\n"
    "Keep your response concise (under 200 words)."
)

# System prompt for debate critic
CRITIC_SYSTEM = (
    "You are a critical reviewer. You will receive a proposed answer to a reasoning problem. "
    "Your job is to:\n"
    "1. Identify any logical errors, false assumptions, or overlooked information\n"
    "2. Point out common cognitive traps the answer may have fallen into\n"
    "3. If the answer is correct, confirm it with 'CRITIQUE: Answer appears correct'\n"
    "4. If incorrect, explain why and provide what you believe is the right answer\n"
    "Be concise and specific."
)

# System prompt for debate reviser
REVISER_SYSTEM = (
    "You are a careful reasoner who receives your initial answer and a critique. "
    "Your job is to:\n"
    "1. Consider the critique carefully\n"
    "2. Determine if the critique is valid\n"
    "3. Either maintain your original answer (if critique is wrong) or revise it\n"
    "4. State your FINAL ANSWER clearly preceded by 'FINAL ANSWER:'\n"
    "Be concise (under 150 words)."
)


def extract_final_answer(text: str, question: dict) -> str:
    """Extract the final answer from model response."""
    # Look for 'FINAL ANSWER:' prefix
    match = re.search(r'FINAL ANSWER:\s*(.+?)(?:\n|CONFIDENCE:|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Take first line only
        answer = answer.split('\n')[0].strip()
        return answer

    # Fallback: look for the answer pattern based on expected answer type
    expected = question.get("correct_answer", "")

    # For YES/NO questions
    if expected.upper() in ["YES", "NO"]:
        if re.search(r'\bYES\b', text, re.IGNORECASE):
            return "YES"
        if re.search(r'\bNO\b', text, re.IGNORECASE):
            return "NO"

    # For A/B/C/D multiple choice
    if expected in ["A", "B", "C", "D"]:
        match = re.search(r'\b([ABCD])\b(?:\s*is|\s*would|\s*seems|[\.\)])', text)
        if match:
            return match.group(1)
        # Last occurrence of A/B/C/D
        matches = re.findall(r'\b([ABCD])\b', text)
        if matches:
            return matches[-1]

    # For KNIGHT/KNAVE or TRUTH-TELLER/LIAR
    if expected in ["KNIGHT", "KNAVE"]:
        if re.search(r'\bKNIGHT\b', text, re.IGNORECASE):
            return "KNIGHT"
        if re.search(r'\bKNAVE\b', text, re.IGNORECASE):
            return "KNAVE"

    if expected in ["TRUTH-TELLER", "LIAR"]:
        if re.search(r'\bTRUTH.TELLER\b', text, re.IGNORECASE):
            return "TRUTH-TELLER"
        if re.search(r'\bLIAR\b', text, re.IGNORECASE):
            return "LIAR"

    # For TRUE/FALSE
    if expected in ["TRUE", "FALSE"]:
        if re.search(r'\bTRUE\b', text, re.IGNORECASE):
            return "TRUE"
        if re.search(r'\bFALSE\b', text, re.IGNORECASE):
            return "FALSE"

    # For numeric answers
    if re.match(r'^\d+$', str(expected)):
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            return numbers[-1]

    # For fractions like 1/6
    if '/' in str(expected):
        fracs = re.findall(r'\d+/\d+', text)
        if fracs:
            return fracs[-1]

    # Return last 50 chars as fallback
    return text.strip()[-100:]


def extract_confidence(text: str) -> float:
    """Extract confidence score from model response (0-1)."""
    match = re.search(r'CONFIDENCE:\s*(\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)) / 100.0
    return 0.5  # default


def is_correct(extracted: str, correct: str, question: dict) -> bool:
    """Check if extracted answer matches correct answer."""
    extracted = extracted.strip().upper()
    correct = correct.strip().upper()

    # Direct match
    if extracted == correct:
        return True

    # Numeric near-match (allow off by 1 for rounding)
    try:
        e_num = float(extracted.replace('$', '').replace('%', '').replace('MPH', '').replace('KM/H', '').strip())
        c_num = float(correct.replace('$', '').replace('%', '').replace('MPH', '').replace('KM/H', '').strip())
        return abs(e_num - c_num) < 0.6  # allow small numeric errors
    except ValueError:
        pass

    # For fraction comparison
    if '/' in correct:
        try:
            parts = correct.split('/')
            c_val = float(parts[0]) / float(parts[1])
            if '/' in extracted:
                eparts = extracted.split('/')
                e_val = float(eparts[0]) / float(eparts[1])
                return abs(e_val - c_val) < 0.01
        except (ValueError, ZeroDivisionError):
            pass

    # Contains check for longer answers (debate)
    if len(correct) > 10 and correct.split()[0] in extracted:
        return True

    return False


def majority_vote(answers: list, question: dict) -> str:
    """Return the majority answer from a list."""
    if not answers:
        return ""
    from collections import Counter
    # Normalize answers
    normalized = [a.strip().upper()[:50] for a in answers]
    counts = Counter(normalized)
    return counts.most_common(1)[0][0]


def confidence_weighted_vote(answers_with_conf: list, question: dict) -> str:
    """Return confidence-weighted majority answer."""
    if not answers_with_conf:
        return ""
    weights: dict = {}
    for answer, conf in answers_with_conf:
        key = answer.strip().upper()[:50]
        weights[key] = weights.get(key, 0) + conf
    return max(weights, key=weights.get)


# ============================================================
# EXPERIMENTAL CONDITIONS
# ============================================================

def run_single_agent(client: anthropic.Anthropic, model: str, question: dict) -> dict:
    """Run single agent on a question."""
    prompt = question["question"]
    response = call_model(client, model, REASONING_SYSTEM, prompt)
    answer = extract_final_answer(response, question)
    correct = is_correct(answer, question["correct_answer"], question)
    return {
        "condition": f"single_{model.split('-')[1]}",
        "model": model,
        "raw_response": response,
        "extracted_answer": answer,
        "correct_answer": question["correct_answer"],
        "is_correct": correct
    }


def run_homogeneous_ensemble(
    client: anthropic.Anthropic, model: str, question: dict, n: int = 3
) -> dict:
    """Run N instances of the same model, majority vote."""
    responses = []
    answers = []
    for i in range(n):
        response = call_model(client, model, REASONING_SYSTEM, question["question"],
                              temperature=0.0)
        answer = extract_final_answer(response, question)
        responses.append(response)
        answers.append(answer)
        logger.debug(f"  Agent {i+1} ({model}): {answer}")

    voted = majority_vote(answers, question)
    correct = is_correct(voted, question["correct_answer"], question)

    return {
        "condition": f"homogeneous_{model.split('-')[1]}_{n}x",
        "model": model,
        "n_agents": n,
        "individual_answers": answers,
        "individual_responses": responses,
        "voted_answer": voted,
        "correct_answer": question["correct_answer"],
        "is_correct": correct,
        "consensus": len(set([a.strip().upper()[:50] for a in answers])) == 1
    }


def run_heterogeneous_ensemble(
    client: anthropic.Anthropic, question: dict,
    haiku_count: int = 2, sonnet_count: int = 1
) -> dict:
    """Run mixed haiku + sonnet agents, majority vote."""
    all_answers = []
    all_responses = []
    agent_models = []

    for i in range(haiku_count):
        response = call_model(client, HAIKU_MODEL, REASONING_SYSTEM, question["question"])
        answer = extract_final_answer(response, question)
        all_answers.append(answer)
        all_responses.append(response)
        agent_models.append(HAIKU_MODEL)
        logger.debug(f"  Haiku {i+1}: {answer}")

    for i in range(sonnet_count):
        response = call_model(client, SONNET_MODEL, REASONING_SYSTEM, question["question"])
        answer = extract_final_answer(response, question)
        all_answers.append(answer)
        all_responses.append(response)
        agent_models.append(SONNET_MODEL)
        logger.debug(f"  Sonnet {i+1}: {answer}")

    voted = majority_vote(all_answers, question)
    correct = is_correct(voted, question["correct_answer"], question)

    return {
        "condition": f"heterogeneous_{haiku_count}h_{sonnet_count}s",
        "agent_models": agent_models,
        "individual_answers": all_answers,
        "individual_responses": all_responses,
        "voted_answer": voted,
        "correct_answer": question["correct_answer"],
        "is_correct": correct,
        "consensus": len(set([a.strip().upper()[:50] for a in all_answers])) == 1
    }


def run_debate_protocol(client: anthropic.Anthropic, question: dict) -> dict:
    """
    Debate: haiku proposes, sonnet critiques, haiku revises.
    Tests whether structured cross-capability debate improves accuracy.
    """
    prompt = question["question"]

    # Step 1: Haiku proposes initial answer
    haiku_initial = call_model(client, HAIKU_MODEL, REASONING_SYSTEM, prompt)
    haiku_answer_initial = extract_final_answer(haiku_initial, question)
    logger.debug(f"  Haiku initial: {haiku_answer_initial}")

    # Step 2: Sonnet critiques haiku's answer
    critique_prompt = (
        f"Problem: {prompt}\n\n"
        f"Proposed answer: {haiku_initial}\n\n"
        "Please critique this answer."
    )
    sonnet_critique = call_model(client, SONNET_MODEL, CRITIC_SYSTEM, critique_prompt)
    logger.debug(f"  Sonnet critique: {sonnet_critique[:100]}...")

    # Step 3: Haiku revises based on critique
    revision_prompt = (
        f"Problem: {prompt}\n\n"
        f"Your initial answer: {haiku_initial}\n\n"
        f"Critique received: {sonnet_critique}\n\n"
        "Please reconsider and provide your final answer."
    )
    haiku_revised = call_model(client, HAIKU_MODEL, REVISER_SYSTEM, revision_prompt)
    haiku_answer_revised = extract_final_answer(haiku_revised, question)
    logger.debug(f"  Haiku revised: {haiku_answer_revised}")

    correct_initial = is_correct(haiku_answer_initial, question["correct_answer"], question)
    correct_final = is_correct(haiku_answer_revised, question["correct_answer"], question)

    return {
        "condition": "debate_haiku_sonnet",
        "haiku_initial": haiku_initial,
        "haiku_answer_initial": haiku_answer_initial,
        "sonnet_critique": sonnet_critique,
        "haiku_revised": haiku_revised,
        "haiku_answer_final": haiku_answer_revised,
        "correct_answer": question["correct_answer"],
        "is_correct": correct_final,
        "initial_was_correct": correct_initial,
        "debate_changed_answer": haiku_answer_initial.strip().upper()[:50] != haiku_answer_revised.strip().upper()[:50]
    }


def run_confidence_weighted_heterogeneous(
    client: anthropic.Anthropic, question: dict
) -> dict:
    """Heterogeneous ensemble with confidence-weighted voting."""
    all_answers_conf = []
    details = []

    for model in [HAIKU_MODEL, SONNET_MODEL]:
        response = call_model(client, model, CONFIDENCE_SYSTEM, question["question"],
                              max_tokens=512)
        answer = extract_final_answer(response, question)
        conf = extract_confidence(response)
        all_answers_conf.append((answer, conf))
        details.append({
            "model": model,
            "response": response,
            "answer": answer,
            "confidence": conf
        })
        logger.debug(f"  {model.split('-')[1]}: {answer} (conf={conf:.2f})")

    voted = confidence_weighted_vote(all_answers_conf, question)
    correct = is_correct(voted, question["correct_answer"], question)

    return {
        "condition": "confidence_weighted",
        "details": details,
        "voted_answer": voted,
        "correct_answer": question["correct_answer"],
        "is_correct": correct
    }
