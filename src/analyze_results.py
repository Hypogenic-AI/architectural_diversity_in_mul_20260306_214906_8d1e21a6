"""
Analysis script for multi-agent diversity experiment results.

Computes:
1. Accuracy by condition and category
2. Error correlation matrices
3. Diversity indices
4. Debate improvement analysis
5. Adversarial robustness comparison
6. Bootstrap confidence intervals
7. Statistical tests
"""

import json
import os
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

RESULTS_DIR = "/workspaces/architectural_diversity_in_mul_20260306_214906_8d1e21a6/results"
FIGURES_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Condition display names
CONDITION_NAMES = {
    "single_haiku": "Single\nHaiku",
    "single_sonnet": "Single\nSonnet",
    "homogeneous_haiku": "Homog.\nHaiku (3x)",
    "heterogeneous_2h1s": "Hetero.\n(2H+1S)",
    "debate": "Debate\n(H→S→H)"
}

CONDITION_COLORS = {
    "single_haiku": "#4878CF",
    "single_sonnet": "#6ACC65",
    "homogeneous_haiku": "#D65F5F",
    "heterogeneous_2h1s": "#B47CC7",
    "debate": "#C4AD66"
}

CATEGORY_NAMES = {
    "misleading_math": "Misleading\nMath",
    "causal_traps": "Causal\nTraps",
    "logical_deception": "Logical\nDeception",
    "numerical_tricks": "Numerical\nTricks",
    "framing_effects": "Framing\nEffects"
}


def load_results(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def bootstrap_ci(values: list, n_boot: int = 1000, ci: float = 0.95) -> tuple:
    """Bootstrap confidence interval for mean."""
    if not values:
        return 0.0, 0.0, 0.0
    values = np.array(values, dtype=float)
    means = [np.mean(np.random.choice(values, size=len(values), replace=True))
             for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.mean(values)), float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100))


def extract_condition_results(results: list, condition: str) -> list:
    """Extract binary correctness vector for a condition."""
    correct = []
    for r in results:
        if condition in r.get("conditions", {}):
            val = r["conditions"][condition].get("is_correct", False)
            correct.append(1 if val else 0)
    return correct


def compute_accuracy_stats(results: list) -> dict:
    """Compute accuracy with bootstrap CIs for each condition."""
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku",
                  "heterogeneous_2h1s", "debate"]
    stats_dict = {}
    for cond in conditions:
        vec = extract_condition_results(results, cond)
        if vec:
            mean, lo, hi = bootstrap_ci(vec)
            stats_dict[cond] = {
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n": len(vec),
                "correct": sum(vec)
            }
    return stats_dict


def compute_per_category_accuracy(results: list) -> dict:
    """Compute accuracy by category for each condition."""
    categories = list(set(r["category"] for r in results if "category" in r))
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku",
                  "heterogeneous_2h1s", "debate"]

    cat_stats = {}
    for cat in categories:
        cat_results = [r for r in results if r.get("category") == cat]
        cat_stats[cat] = compute_accuracy_stats(cat_results)

    return cat_stats


def compute_error_correlation_matrix(results: list) -> dict:
    """Compute pairwise error correlation matrix between conditions."""
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku", "heterogeneous_2h1s"]
    error_vecs = {}
    for cond in conditions:
        vec = extract_condition_results(results, cond)
        if vec:
            error_vecs[cond] = [1 - v for v in vec]  # error = 1 when wrong

    corr_matrix = {}
    for c1 in conditions:
        for c2 in conditions:
            if c1 in error_vecs and c2 in error_vecs:
                v1, v2 = error_vecs[c1], error_vecs[c2]
                if len(v1) == len(v2) and len(v1) > 2:
                    try:
                        r, p = stats.pearsonr(v1, v2)
                        corr_matrix[f"{c1}_vs_{c2}"] = {"r": float(r), "p": float(p)}
                    except Exception:
                        pass

    return corr_matrix, error_vecs


def compute_diversity_index(error_vecs: dict) -> dict:
    """
    Diversity index: lower pairwise error correlation = higher diversity.
    DI = 1 - mean(pairwise_error_correlation)
    """
    homog_conditions = ["single_haiku", "homogeneous_haiku"]
    hetero_conditions = ["single_haiku", "single_sonnet"]

    def mean_correlation(conds, ev):
        corrs = []
        for i, c1 in enumerate(conds):
            for c2 in conds[i+1:]:
                if c1 in ev and c2 in ev:
                    v1, v2 = ev[c1], ev[c2]
                    if len(v1) == len(v2) and len(v1) > 2:
                        try:
                            r, _ = stats.pearsonr(v1, v2)
                            corrs.append(r)
                        except Exception:
                            pass
        return float(np.mean(corrs)) if corrs else 0.0

    homog_corr = mean_correlation(homog_conditions, error_vecs)
    hetero_corr = mean_correlation(hetero_conditions, error_vecs)

    return {
        "homogeneous_error_correlation": homog_corr,
        "heterogeneous_error_correlation": hetero_corr,
        "diversity_index_homogeneous": 1.0 - homog_corr,
        "diversity_index_heterogeneous": 1.0 - hetero_corr,
    }


def compute_debate_analysis(results: list) -> dict:
    """Analyze debate protocol: how often does debate improve/maintain/hurt."""
    improvements = 0
    degradations = 0
    maintained_correct = 0
    maintained_wrong = 0
    changed = 0
    total = 0

    for r in results:
        debate = r.get("conditions", {}).get("debate", {})
        if debate:
            total += 1
            init_correct = debate.get("initial_correct", False)
            final_correct = debate.get("is_correct", False)
            ch = debate.get("changed", False)

            if ch:
                changed += 1

            if not init_correct and final_correct:
                improvements += 1
            elif init_correct and not final_correct:
                degradations += 1
            elif init_correct and final_correct:
                maintained_correct += 1
            else:
                maintained_wrong += 1

    return {
        "total": total,
        "improvements": improvements,
        "degradations": degradations,
        "maintained_correct": maintained_correct,
        "maintained_wrong": maintained_wrong,
        "changed_count": changed,
        "improvement_rate": improvements / total if total > 0 else 0,
        "degradation_rate": degradations / total if total > 0 else 0,
        "net_improvement": improvements - degradations
    }


def compute_robustness(original_results: list, paraphrase_results: list) -> dict:
    """
    Compare accuracy on original vs paraphrase variants.
    Map paraphrase back to original by original_id.
    """
    # For paraphrases, we need to match them to originals by original_id
    # But since we only have 6 paraphrases, compute aggregate stats
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku", "heterogeneous_2h1s"]
    robustness = {}

    # Original accuracy on paraphrase source questions (ids 1,7,13,19,4,20)
    paraphrase_source_ids = {31: 1, 32: 7, 33: 13, 34: 19, 35: 4, 36: 20}

    for cond in conditions:
        # Original accuracy
        orig_vec = extract_condition_results(original_results, cond)
        orig_acc = np.mean(orig_vec) if orig_vec else 0.0

        # Paraphrase accuracy
        para_vec = extract_condition_results(paraphrase_results, cond)
        para_acc = np.mean(para_vec) if para_vec else 0.0

        # Stability = para_acc / orig_acc (or difference)
        stability = para_acc - orig_acc  # positive = more robust on paraphrases

        robustness[cond] = {
            "original_accuracy": float(orig_acc),
            "paraphrase_accuracy": float(para_acc),
            "stability": float(stability),
            "n_paraphrase": len(para_vec)
        }

    return robustness


def mcnemar_test(v1: list, v2: list) -> dict:
    """McNemar's test for paired comparison of two conditions."""
    if len(v1) != len(v2) or not v1:
        return {"statistic": None, "p_value": None}

    # Build 2x2 contingency table
    # b = v1 correct, v2 wrong; c = v1 wrong, v2 correct
    b = sum(1 for a, b_ in zip(v1, v2) if a == 1 and b_ == 0)
    c = sum(1 for a, b_ in zip(v1, v2) if a == 0 and b_ == 1)

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": 0, "c": 0}

    # McNemar's statistic
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(stat, df=1)

    return {"statistic": float(stat), "p_value": float(p), "b": b, "c": c}


# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def plot_accuracy_comparison(acc_stats: dict, title: str, filename: str):
    """Bar chart comparing accuracy across conditions with CI bars."""
    conditions = [c for c in ["single_haiku", "single_sonnet", "homogeneous_haiku",
                               "heterogeneous_2h1s", "debate"]
                  if c in acc_stats]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    means = [acc_stats[c]["mean"] for c in conditions]
    ci_lo = [acc_stats[c]["mean"] - acc_stats[c]["ci_low"] for c in conditions]
    ci_hi = [acc_stats[c]["ci_high"] - acc_stats[c]["mean"] for c in conditions]
    colors = [CONDITION_COLORS.get(c, "#888888") for c in conditions]
    labels = [CONDITION_NAMES.get(c, c) for c in conditions]

    bars = ax.bar(x, means, color=colors, alpha=0.85, width=0.6,
                  yerr=[ci_lo, ci_hi], capsize=5, error_kw={"linewidth": 2})

    # Add value labels
    for bar, mean, n_correct, n in zip(bars, means, [acc_stats[c]["correct"] for c in conditions],
                                        [acc_stats[c]["n"] for c in conditions]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{mean:.0%}\n({n_correct}/{n})', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance level')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_category_heatmap(cat_stats: dict, filename: str):
    """Heatmap of accuracy by category and condition."""
    categories = list(CATEGORY_NAMES.keys())
    conditions = ["single_haiku", "single_sonnet", "homogeneous_haiku",
                  "heterogeneous_2h1s", "debate"]

    # Build matrix
    matrix = []
    for cond in conditions:
        row = []
        for cat in categories:
            if cat in cat_stats and cond in cat_stats[cat]:
                row.append(cat_stats[cat][cond]["mean"])
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([CATEGORY_NAMES.get(c, c) for c in categories], fontsize=10)
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([CONDITION_NAMES.get(c, c).replace('\n', ' ') for c in conditions], fontsize=10)

    # Add values
    for i in range(len(conditions)):
        for j in range(len(categories)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=11, fontweight='bold',
                        color='black' if 0.3 < val < 0.8 else 'white')

    plt.colorbar(im, ax=ax, label='Accuracy')
    ax.set_title("Accuracy by Condition and Task Category", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_error_correlation_heatmap(error_vecs: dict, filename: str):
    """Heatmap of pairwise error correlations between conditions."""
    conditions = [c for c in ["single_haiku", "single_sonnet", "homogeneous_haiku",
                               "heterogeneous_2h1s"] if c in error_vecs]
    labels = [CONDITION_NAMES.get(c, c).replace('\n', ' ') for c in conditions]

    n = len(conditions)
    corr_matrix = np.eye(n)

    for i, c1 in enumerate(conditions):
        for j, c2 in enumerate(conditions):
            if i != j and c1 in error_vecs and c2 in error_vecs:
                v1, v2 = error_vecs[c1], error_vecs[c2]
                if len(v1) == len(v2) and len(v1) > 2:
                    try:
                        r, _ = stats.pearsonr(v1, v2)
                        corr_matrix[i, j] = r
                    except Exception:
                        pass

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='black' if abs(corr_matrix[i, j]) < 0.7 else 'white')

    plt.colorbar(im, ax=ax, label='Pearson r (error correlation)')
    ax.set_title("Pairwise Error Correlation Between Agent Conditions\n(Higher = More Correlated Errors = Less Diversity)",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_debate_analysis(debate_stats: dict, filename: str):
    """Pie/bar chart of debate outcomes."""
    categories_data = {
        "Maintained\nCorrect": debate_stats["maintained_correct"],
        "Debate\nImproved": debate_stats["improvements"],
        "Debate\nDegraded": debate_stats["degradations"],
        "Maintained\nWrong": debate_stats["maintained_wrong"]
    }

    colors = ["#6ACC65", "#4878CF", "#D65F5F", "#C0C0C0"]
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(categories_data))
    bars = ax.bar(x, categories_data.values(), color=colors, width=0.6, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories_data.keys(), fontsize=11)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Debate Protocol Outcomes (Haiku→Sonnet→Haiku)\nImprovements vs Degradations",
                 fontsize=12, fontweight='bold')

    for bar, val in zip(bars, categories_data.values()):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                    str(val), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_robustness_comparison(robustness: dict, filename: str):
    """Bar chart comparing original vs paraphrase accuracy."""
    conditions = [c for c in ["single_haiku", "single_sonnet", "homogeneous_haiku",
                               "heterogeneous_2h1s"] if c in robustness]
    labels = [CONDITION_NAMES.get(c, c).replace('\n', ' ') for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    orig_vals = [robustness[c]["original_accuracy"] for c in conditions]
    para_vals = [robustness[c]["paraphrase_accuracy"] for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, orig_vals, width, label='Original Questions', alpha=0.85,
                color=['#4878CF', '#6ACC65', '#D65F5F', '#B47CC7'])
    b2 = ax.bar(x + width/2, para_vals, width, label='Paraphrase Variants', alpha=0.85,
                color=['#93B5D9', '#A8DCA5', '#E89393', '#D3A8E6'])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Adversarial Robustness: Original vs Paraphrase Accuracy\n(Stability Test)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.0%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    print("=" * 60)
    print("Analyzing Multi-Agent Diversity Experiment Results")
    print("=" * 60)

    # Load results
    results_path = f"{RESULTS_DIR}/raw_results.json"
    if not os.path.exists(results_path):
        print(f"ERROR: Results file not found at {results_path}")
        return None

    data = load_results(results_path)
    original_results = data["original_results"]
    paraphrase_results = data.get("paraphrase_results", [])

    # Filter out errors
    original_results = [r for r in original_results if "error" not in r and r.get("conditions")]
    paraphrase_results = [r for r in paraphrase_results if "error" not in r and r.get("conditions")]

    print(f"Valid original results: {len(original_results)}/30")
    print(f"Valid paraphrase results: {len(paraphrase_results)}/6")

    # 1. Overall accuracy
    print("\n=== 1. OVERALL ACCURACY ===")
    acc_stats = compute_accuracy_stats(original_results)
    for cond, stats_d in acc_stats.items():
        print(f"  {cond}: {stats_d['correct']}/{stats_d['n']} = {stats_d['mean']:.1%} "
              f"[{stats_d['ci_low']:.1%}, {stats_d['ci_high']:.1%}]")

    # 2. Per-category accuracy
    print("\n=== 2. PER-CATEGORY ACCURACY ===")
    cat_stats = compute_per_category_accuracy(original_results)
    for cat, conds in cat_stats.items():
        print(f"\n  {cat}:")
        for cond, st in conds.items():
            print(f"    {cond}: {st['correct']}/{st['n']} = {st['mean']:.1%}")

    # 3. Error correlations
    print("\n=== 3. ERROR CORRELATIONS ===")
    corr_matrix, error_vecs = compute_error_correlation_matrix(original_results)
    for pair, vals in corr_matrix.items():
        if "_vs_" in pair:
            c1, c2 = pair.split("_vs_")
            if c1 != c2:
                print(f"  {c1} vs {c2}: r={vals['r']:.3f} (p={vals['p']:.3f})")

    # 4. Diversity index
    print("\n=== 4. DIVERSITY INDEX ===")
    div_idx = compute_diversity_index(error_vecs)
    for k, v in div_idx.items():
        print(f"  {k}: {v:.3f}")

    # 5. Debate analysis
    print("\n=== 5. DEBATE ANALYSIS ===")
    debate_stats = compute_debate_analysis(original_results)
    for k, v in debate_stats.items():
        print(f"  {k}: {v}")

    # 6. Robustness
    print("\n=== 6. ADVERSARIAL ROBUSTNESS ===")
    robustness = {}
    if paraphrase_results:
        robustness = compute_robustness(original_results, paraphrase_results)
        for cond, vals in robustness.items():
            print(f"  {cond}: orig={vals['original_accuracy']:.1%}, "
                  f"para={vals['paraphrase_accuracy']:.1%}, "
                  f"stability={vals['stability']:+.1%}")

    # 7. Statistical tests
    print("\n=== 7. STATISTICAL TESTS ===")
    haiku_vec = extract_condition_results(original_results, "single_haiku")
    sonnet_vec = extract_condition_results(original_results, "single_sonnet")
    homo_vec = extract_condition_results(original_results, "homogeneous_haiku")
    hetero_vec = extract_condition_results(original_results, "heterogeneous_2h1s")
    debate_vec = extract_condition_results(original_results, "debate")

    # McNemar tests
    pairs = [
        ("single_haiku", haiku_vec, "homogeneous_haiku", homo_vec),
        ("single_haiku", haiku_vec, "heterogeneous_2h1s", hetero_vec),
        ("homogeneous_haiku", homo_vec, "heterogeneous_2h1s", hetero_vec),
        ("single_sonnet", sonnet_vec, "heterogeneous_2h1s", hetero_vec),
        ("single_sonnet", sonnet_vec, "debate", debate_vec),
        ("heterogeneous_2h1s", hetero_vec, "debate", debate_vec),
    ]

    mcnemar_results = {}
    for n1, v1, n2, v2 in pairs:
        result = mcnemar_test(v1, v2)
        key = f"{n1}_vs_{n2}"
        mcnemar_results[key] = result
        print(f"  McNemar {n1} vs {n2}: chi2={result.get('statistic', 'N/A')}, "
              f"p={result.get('p_value', 'N/A')}")

    # Cohen's d for key comparison
    def cohens_d(v1, v2):
        v1, v2 = np.array(v1, float), np.array(v2, float)
        pooled_std = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
        return float((v1.mean() - v2.mean()) / pooled_std) if pooled_std > 0 else 0.0

    print(f"\n  Cohen's d (homog_haiku vs hetero): {cohens_d(homo_vec, hetero_vec):.3f}")
    print(f"  Cohen's d (single_haiku vs hetero): {cohens_d(haiku_vec, hetero_vec):.3f}")

    # =====================
    # GENERATE VISUALIZATIONS
    # =====================
    print("\n=== GENERATING VISUALIZATIONS ===")

    plot_accuracy_comparison(acc_stats, "Overall Accuracy by Experimental Condition\n(30 Adversarial Reasoning Questions, 95% CI)",
                             "accuracy_comparison.png")

    if cat_stats:
        plot_category_heatmap(cat_stats, "category_heatmap.png")

    if error_vecs:
        plot_error_correlation_heatmap(error_vecs, "error_correlation_heatmap.png")

    plot_debate_analysis(debate_stats, "debate_outcomes.png")

    if robustness:
        plot_robustness_comparison(robustness, "robustness_comparison.png")

    # Save analysis results
    analysis_output = {
        "accuracy_stats": acc_stats,
        "category_stats": cat_stats,
        "error_correlations": {k: v for k, v in corr_matrix.items()},
        "diversity_index": div_idx,
        "debate_analysis": debate_stats,
        "robustness": robustness,
        "statistical_tests": mcnemar_results
    }

    with open(f"{RESULTS_DIR}/analysis_results.json", "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\nAnalysis saved to {RESULTS_DIR}/analysis_results.json")

    return analysis_output


if __name__ == "__main__":
    main()
