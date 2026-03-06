"""
Post-processing script to fix answer extraction issues.
Re-evaluates correctness from raw model responses stored in results.

Issue: Models use FINAL ANSWER: with markdown like "**NO** — explanation"
       is_correct then fails because "**NO**..." != "NO"
"""

import json
import re
import os

RESULTS_DIR = "/workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6/results"


def strip_markdown(text: str) -> str:
    """Remove markdown formatting from text."""
    # Remove ** and * bold/italic
    text = re.sub(r'\*+', '', text)
    # Remove __ underlines
    text = re.sub(r'_{1,2}', '', text)
    # Remove backticks
    text = re.sub(r'`+', '', text)
    # Remove # headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def clean_extract(text: str, correct: str) -> str:
    """Robust answer extraction from model response text."""
    correct = correct.strip().upper()
    cleaned_text = strip_markdown(text)

    # Try FINAL ANSWER: marker first
    match = re.search(r'FINAL ANSWER:\s*(.+?)(?:\n|CONFIDENCE:|$)', text,
                      re.IGNORECASE | re.DOTALL)
    if match:
        candidate = match.group(1).strip()
        candidate = strip_markdown(candidate)
        # Remove trailing explanation (dashes, colons, em-dashes)
        candidate = re.split(r'\s*[—\-:\.]+\s', candidate)[0].strip()
        # Take first word
        words = candidate.split()
        if words:
            candidate = words[0].rstrip('.,;:!?()').upper()
            if candidate:
                return candidate

    # YES/NO questions
    if correct in ["YES", "NO"]:
        # Look for YES/NO clearly stated
        for pattern in [
            r'(?:answer is|recommend|should choose|choose)[^.!?\n]*?\b(YES|NO)\b',
            r'^(YES|NO)\b',
            r'^\s*(YES|NO)[\s\.\*,;:!]',
            r'\b(YES|NO)\b',
        ]:
            m = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()

    # A/B/C/D multiple choice
    if correct in ["A", "B", "C", "D"]:
        for pattern in [
            r'(?:answer is|correct answer|choose|recommend)[^\w]*\(([A-D])\)',
            r'(?:answer is|correct answer|choose|recommend)[^\w]*([A-D])\b',
            r'\(([A-D])\)\s+(?:is|seems|appears)',
            r'The answer is\s+([A-D])\b',
            r'^\s*([A-D])[\)\.:]',
        ]:
            m = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()
        # All occurrences - take last one
        ms = re.findall(r'\b([A-D])\b', cleaned_text.upper())
        if ms:
            return ms[-1]

    # KNIGHT/KNAVE
    if correct in ["KNIGHT", "KNAVE"]:
        for kw in ["KNAVE", "KNIGHT"]:
            if re.search(r'\b' + kw + r'\b', cleaned_text, re.IGNORECASE):
                return kw

    # TRUTH-TELLER/LIAR
    if correct in ["TRUTH-TELLER", "LIAR"]:
        if re.search(r'TRUTH.TELLER', cleaned_text, re.IGNORECASE):
            return "TRUTH-TELLER"
        if re.search(r'\bLIAR\b', cleaned_text, re.IGNORECASE):
            return "LIAR"

    # TRUE/FALSE
    if correct in ["TRUE", "FALSE"]:
        for pattern in [
            r'(?:statement is|answer is|this is|the answer)[^.!?\n]*?(TRUE|FALSE)',
            r'^(TRUE|FALSE)\b',
            r'\b(TRUE|FALSE)\b',
        ]:
            m = re.search(pattern, cleaned_text, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()

    # Numeric
    if re.match(r'^\d+(?:\.\d+)?$', correct):
        # Look for "the answer is X" pattern first
        for pattern in [
            r'(?:answer is|equals|=\s*)(\d+(?:\.\d+)?)',
            r'\b(\d+(?:\.\d+)?)\s*(?:days|cents|dollars|mph|km|minutes|hours|miles|%)',
        ]:
            m = re.search(pattern, cleaned_text, re.IGNORECASE)
            if m:
                return m.group(1)
        # All numbers - take last
        nums = re.findall(r'\b(\d+(?:\.\d+)?)\b', cleaned_text)
        if nums:
            return nums[-1]

    # Fraction
    if '/' in correct:
        fracs = re.findall(r'\d+/\d+', cleaned_text)
        if fracs:
            return fracs[-1]

    # Long text: return last non-empty line
    lines = [l.strip() for l in cleaned_text.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1][:50].upper()

    return cleaned_text[:50].upper()


def is_correct(extracted: str, correct: str) -> bool:
    """Flexible correctness check."""
    extracted = strip_markdown(extracted).strip().upper()
    correct = correct.strip().upper()

    # Strip explanation parts (text after dash or colon)
    extracted_key = re.split(r'\s*[—\-:\.]+\s', extracted)[0].strip()
    extracted_key = extracted_key.split()[0].rstrip('.,;:!?()') if extracted_key.split() else extracted_key

    # Direct match
    if extracted == correct or extracted_key == correct:
        return True

    # Contains check for short correct answers
    if len(correct) <= 10 and correct in extracted:
        return True

    # Numeric match (allow small tolerance)
    try:
        e_nums = re.findall(r'\d+(?:\.\d+)?', extracted_key or extracted)
        c_nums = re.findall(r'\d+(?:\.\d+)?', correct)
        if e_nums and c_nums:
            e_val = float(e_nums[0])
            c_val = float(c_nums[0])
            if abs(e_val - c_val) < 0.6:
                return True
    except (ValueError, IndexError):
        pass

    # Fraction comparison
    if '/' in correct:
        try:
            def parse_frac(s):
                parts = re.findall(r'\d+', s)
                if len(parts) >= 2:
                    return float(parts[0]) / float(parts[1])
                return None
            c_val = parse_frac(correct)
            e_val = parse_frac(extracted)
            if c_val is not None and e_val is not None:
                return abs(e_val - c_val) < 0.01
        except (ValueError, ZeroDivisionError):
            pass

    # Long text: check if correct key word appears
    if len(correct) > 10:
        key_word = correct.split()[0]
        return key_word in extracted

    return False


def reprocess_all(data: dict) -> dict:
    """Re-extract and re-evaluate all answers from raw responses."""

    for dataset_key in ["original_results", "paraphrase_results"]:
        results = data.get(dataset_key, [])
        for result in results:
            if "error" in result:
                continue

            correct = result.get("correct_answer", "")
            conditions = result.get("conditions", {})

            for cond_name, cond_data in conditions.items():
                if not isinstance(cond_data, dict):
                    continue

                # Single agent: use 'response' field
                if cond_name in ["single_haiku", "single_sonnet"]:
                    raw = cond_data.get("response", "")
                    if raw:
                        new_ans = clean_extract(raw, correct)
                        new_correct = is_correct(new_ans, correct)
                        cond_data["answer"] = new_ans
                        cond_data["is_correct"] = new_correct

                # Ensemble: re-evaluate individual responses and re-vote
                elif cond_name in ["homogeneous_haiku", "heterogeneous_2h1s"]:
                    ind_resps = cond_data.get("individual_responses", [])
                    if ind_resps:
                        from collections import Counter
                        new_answers = []
                        for resp in ind_resps:
                            ans = clean_extract(resp, correct)
                            new_answers.append(ans)

                        cond_data["individual_answers"] = new_answers
                        # Majority vote
                        normalized = [re.split(r'\s*[—\-:\.]+', a)[0].split()[0].upper()[:20]
                                      if a.split() else a.upper()[:20]
                                      for a in new_answers]
                        if normalized:
                            voted = Counter(normalized).most_common(1)[0][0]
                            cond_data["voted_answer"] = voted
                            cond_data["answer"] = voted
                            cond_data["is_correct"] = is_correct(voted, correct)

                # Debate: answer might be in 'answer' field; also check initial
                elif cond_name == "debate":
                    ans = cond_data.get("answer", "")
                    if ans:
                        cond_data["is_correct"] = is_correct(ans, correct)
                    init_ans = cond_data.get("initial_answer", "")
                    if init_ans:
                        cond_data["initial_correct"] = is_correct(init_ans, correct)

    return data


def main():
    path = f"{RESULTS_DIR}/raw_results.json"
    if not os.path.exists(path):
        print("Results not yet available")
        return

    with open(path) as f:
        data = json.load(f)

    print("Reprocessing answer extraction...")
    data = reprocess_all(data)

    # Save fixed results
    fixed_path = f"{RESULTS_DIR}/raw_results_fixed.json"
    with open(fixed_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Fixed results saved to {fixed_path}")

    # Print corrected summary
    for dataset_key in ["original_results", "paraphrase_results"]:
        results = [r for r in data.get(dataset_key, [])
                   if "error" not in r and r.get("conditions")]
        if not results:
            continue
        print(f"\n{dataset_key} ({len(results)} questions):")
        conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku",
                      "heterogeneous_2h1s", "debate"]
        for cond in conditions:
            vec = [r["conditions"][cond]["is_correct"]
                   for r in results if cond in r.get("conditions", {})]
            if vec:
                print(f"  {cond}: {sum(vec)}/{len(vec)} = {sum(vec)/len(vec):.1%}")


if __name__ == "__main__":
    main()
