"""
Adversarial reasoning benchmark for testing multi-agent LLM diversity.

30 questions across 5 categories, each designed to have a misleading
surface-level pattern and a correct answer requiring deeper reasoning.
Also includes 6 paraphrase variants for robustness testing.
"""

ADVERSARIAL_BENCHMARK = [
    # =========================================================
    # CATEGORY 1: MISLEADING MATH (questions 1-6)
    # Surface patterns give wrong answers; requires careful algebra
    # =========================================================
    {
        "id": 1,
        "category": "misleading_math",
        "question": (
            "A bat and a ball together cost $1.10. "
            "The bat costs exactly $1.00 more than the ball. "
            "How much does the ball cost? "
            "Give your answer in cents (e.g., 5 for 5 cents)."
        ),
        "correct_answer": "5",
        "misleading_answer": "10",
        "explanation": (
            "Let ball=x, bat=x+1.00. Then 2x+1.00=1.10, so 2x=0.10, x=0.05 (5 cents). "
            "Surface pattern: 1.10 - 1.00 = 0.10, which gives 10 cents (wrong)."
        )
    },
    {
        "id": 2,
        "category": "misleading_math",
        "question": (
            "A store reduces a price by 20% and then increases the result by 20%. "
            "If the original price was $100, what is the final price?"
        ),
        "correct_answer": "96",
        "misleading_answer": "100",
        "explanation": (
            "100 * 0.8 * 1.2 = 96. Surface pattern: up 20% then down 20% looks like net zero."
        )
    },
    {
        "id": 3,
        "category": "misleading_math",
        "question": (
            "If you have a 6-sided die and roll it 6 times, the expected number of 6s you will see is 1. "
            "You've rolled 5 times and never gotten a 6. "
            "What is the probability of getting a 6 on your next roll? "
            "Express as a fraction (e.g., 1/6)."
        ),
        "correct_answer": "1/6",
        "misleading_answer": "5/6",
        "explanation": (
            "Dice are memoryless; each roll is independent. The probability is still 1/6. "
            "Gambler's fallacy makes people think it's 'due' so probability is higher."
        )
    },
    {
        "id": 4,
        "category": "misleading_math",
        "question": (
            "A snail is at the bottom of a 10-foot well. "
            "Each day it climbs 3 feet, but each night it slides back 2 feet. "
            "How many days does it take to reach the top?"
        ),
        "correct_answer": "8",
        "misleading_answer": "10",
        "explanation": (
            "Net gain per day: 1 foot. After 7 days: 7 feet. On day 8 it climbs 3 feet to 10 feet (escapes). "
            "Misleading: 10/1 = 10 days (forgets that on the final day it doesn't slide back)."
        )
    },
    {
        "id": 5,
        "category": "misleading_math",
        "question": (
            "Two trains start from cities 300 miles apart, heading toward each other. "
            "Train A goes 60 mph; Train B goes 40 mph. "
            "A fly starts at Train A's front and flies back and forth between the trains at 100 mph until they meet. "
            "How many miles does the fly travel total?"
        ),
        "correct_answer": "300",
        "misleading_answer": "150",
        "explanation": (
            "Trains meet in 300/100 = 3 hours. Fly travels at 100 mph for 3 hours = 300 miles. "
            "Misleading: trying to compute the infinite back-and-forth series gives a complex calculation "
            "but the elegant solution is much simpler."
        )
    },
    {
        "id": 6,
        "category": "misleading_math",
        "question": (
            "A farmer has 17 sheep. All but 9 die. How many sheep are left?"
        ),
        "correct_answer": "9",
        "misleading_answer": "8",
        "explanation": (
            "'All but 9' means 9 survive. People often subtract: 17 - 9 = 8 (wrong)."
        )
    },

    # =========================================================
    # CATEGORY 2: CAUSAL TRAPS (questions 7-12)
    # Correlation-causation confusion, hidden confounders, Simpson's paradox
    # =========================================================
    {
        "id": 7,
        "category": "causal_traps",
        "question": (
            "Hospital A has a 2% death rate; Hospital B has a 1% death rate. "
            "You need surgery. Should you choose Hospital B to maximize your survival chances? "
            "Answer YES or NO and briefly explain why."
        ),
        "correct_answer": "NO",
        "misleading_answer": "YES",
        "explanation": (
            "Hospital A likely receives more severe/critical cases (referral hospital), "
            "explaining the higher crude death rate. For equivalent patients, Hospital A may be superior. "
            "This is Simpson's paradox / confounding by case severity."
        )
    },
    {
        "id": 8,
        "category": "causal_traps",
        "question": (
            "Studies show that people who carry lighters are more likely to develop lung cancer. "
            "Should we ban lighters to reduce lung cancer rates? "
            "Answer YES or NO and briefly explain why."
        ),
        "correct_answer": "NO",
        "misleading_answer": "YES",
        "explanation": (
            "Carrying a lighter is correlated with smoking, which causes lung cancer. "
            "Lighters are a proxy/confounder, not the cause. Banning lighters won't address the actual cause."
        )
    },
    {
        "id": 9,
        "category": "causal_traps",
        "question": (
            "A drug trial shows: Treatment group — 200 mild cases (190 recovered, 10 died); "
            "100 severe cases (30 recovered, 70 died). Total: 220/300 recovered (73%). "
            "Control group — 100 mild cases (81 recovered, 19 died); "
            "200 severe cases (40 recovered, 160 died). Total: 121/300 recovered (40%). "
            "Is the treatment beneficial overall? Answer YES or NO."
        ),
        "correct_answer": "YES",
        "misleading_answer": "NO",
        "explanation": (
            "For mild cases: treatment 95% vs control 81% — treatment better. "
            "For severe cases: treatment 30% vs control 20% — treatment better. "
            "Treatment better in BOTH subgroups, yet overall 73% vs 40% also confirms it. "
            "This question is designed to test whether the model correctly applies subgroup analysis. "
            "The correct answer is YES — the treatment is beneficial."
        )
    },
    {
        "id": 10,
        "category": "causal_traps",
        "question": (
            "Cities with more hospitals have more deaths. "
            "Does this mean hospitals cause deaths? "
            "Answer YES or NO and briefly explain."
        ),
        "correct_answer": "NO",
        "misleading_answer": "YES",
        "explanation": (
            "Cities with more hospitals are larger and have more sick people (confounding). "
            "Correlation ≠ causation. Hospitals treat disease but cannot reverse its cause."
        )
    },
    {
        "id": 11,
        "category": "causal_traps",
        "question": (
            "Studies consistently show that children who attend preschool earn more as adults. "
            "A policymaker concludes that building more preschools will increase adult earnings. "
            "What is the most significant threat to this causal conclusion? "
            "Choose: (A) Sample size too small, (B) Confounding by family socioeconomic status, "
            "(C) Preschool content is outdated, (D) The study only measured salaries not happiness."
        ),
        "correct_answer": "B",
        "misleading_answer": "A",
        "explanation": (
            "Families who send children to preschool tend to be wealthier and more educated — "
            "these same factors predict adult earnings. Without randomization, we can't know if "
            "preschool itself caused the earnings difference."
        )
    },
    {
        "id": 12,
        "category": "causal_traps",
        "question": (
            "A study finds that people who sleep with their shoes on are more likely to wake up with a headache. "
            "What most likely explains this finding? "
            "Choose: (A) Shoes restrict blood flow causing headaches, (B) Both are caused by going to bed drunk, "
            "(C) Shoes cause psychological discomfort during sleep, (D) People who wear shoes to bed are generally unhealthier."
        ),
        "correct_answer": "B",
        "misleading_answer": "A",
        "explanation": (
            "People who sleep with shoes on typically did so because they were intoxicated, "
            "which also causes morning headaches. Classic common cause / confounder scenario."
        )
    },

    # =========================================================
    # CATEGORY 3: LOGICAL DECEPTION (questions 13-18)
    # Multi-step logical puzzles with misleading surface patterns
    # =========================================================
    {
        "id": 13,
        "category": "logical_deception",
        "question": (
            "On an island, knights always tell the truth and knaves always lie. "
            "You meet two islanders, A and B. A says: 'We are both knaves.' "
            "What is A? Answer: KNIGHT or KNAVE."
        ),
        "correct_answer": "KNAVE",
        "misleading_answer": "KNIGHT",
        "explanation": (
            "If A is a knight, A's statement must be true, meaning both are knaves — contradiction "
            "(A can't be knight and knave). So A must be a knave. "
            "As a knave, A's statement 'we are both knaves' is a lie, meaning B is a knight. Consistent."
        )
    },
    {
        "id": 14,
        "category": "logical_deception",
        "question": (
            "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed'. "
            "All three labels are WRONG. You can draw one fruit from one box. "
            "You draw from the 'Mixed' box and get an apple. "
            "What does the 'Apples' box actually contain? "
            "Answer: APPLES, ORANGES, or MIXED."
        ),
        "correct_answer": "ORANGES",
        "misleading_answer": "MIXED",
        "explanation": (
            "The 'Mixed' label is wrong, so the box you drew from is either all-apples or all-oranges. "
            "You got an apple, so it's all-apples. The 'Apples' box can't contain apples (label is wrong), "
            "and can't contain apples (taken). So 'Apples' box contains oranges. "
            "'Oranges' box contains mixed."
        )
    },
    {
        "id": 15,
        "category": "logical_deception",
        "question": (
            "If all Bloops are Razzies, and all Razzies are Lazzies, "
            "which of the following must be true? "
            "(A) All Bloops are Lazzies, (B) All Lazzies are Bloops, "
            "(C) All Razzies are Bloops, (D) Some Lazzies are not Razzies."
        ),
        "correct_answer": "A",
        "misleading_answer": "B",
        "explanation": (
            "Transitivity: Bloops → Razzies → Lazzies, so all Bloops are Lazzies (A is true). "
            "B reverses the direction and is not necessarily true. "
            "C reverses Bloop→Razzie which needn't hold. D might be true but isn't guaranteed."
        )
    },
    {
        "id": 16,
        "category": "logical_deception",
        "question": (
            "A man lives on the 20th floor of an apartment building. "
            "Every morning he takes the elevator down to the ground floor and goes to work. "
            "When he returns in the evening, he takes the elevator to the 10th floor and walks up the stairs to the 20th floor. "
            "Why does he do this? "
            "Choose: (A) He enjoys the exercise, (B) He is too short to reach button 20, "
            "(C) The elevator only goes to the 10th floor at night, (D) He lives on floor 10 and works on floor 20."
        ),
        "correct_answer": "B",
        "misleading_answer": "A",
        "explanation": (
            "Classic lateral thinking puzzle. He can reach the 'G' button (at the bottom) but not '20' "
            "(at the top). He can reach '10' in the middle. He is too short to reach button 20."
        )
    },
    {
        "id": 17,
        "category": "logical_deception",
        "question": (
            "You have two ropes and a lighter. Each rope burns for exactly 1 hour total, "
            "but they burn unevenly (not at constant rate). "
            "How do you measure exactly 45 minutes using only these ropes and lighter?"
        ),
        "correct_answer": (
            "Light rope 1 at both ends and rope 2 at one end simultaneously. "
            "When rope 1 burns out (30 min), light the other end of rope 2. "
            "Rope 2 burns out 15 min later. Total: 45 min."
        ),
        "misleading_answer": (
            "Cut the ropes into pieces proportional to time."
        ),
        "explanation": (
            "The key insight: lighting both ends of rope 1 makes it burn in 30 min regardless of uneven rate. "
            "Then rope 2 has 30 min left; lighting its other end burns it in 15 more min. "
            "Misleading: people try to use length as a proxy for burn time (fails due to uneven burning)."
        )
    },
    {
        "id": 18,
        "category": "logical_deception",
        "question": (
            "There are 3 switches outside a room. One switch controls a light bulb inside the room. "
            "You can't see into the room from outside. "
            "You can flip switches as many times as you want, but can only enter the room once. "
            "How do you determine which switch controls the bulb?"
        ),
        "correct_answer": (
            "Turn on switch 1 for 10 minutes, then turn it off. Turn on switch 2 and enter. "
            "If bulb is on: switch 2. If bulb is off and warm: switch 1. If off and cold: switch 3."
        ),
        "misleading_answer": (
            "Enter the room multiple times to test each switch."
        ),
        "explanation": (
            "The key insight: heat (from switch 1 being on then off) provides information beyond just on/off. "
            "Surface pattern: people try to use multiple entries, but constraint says only once."
        )
    },

    # =========================================================
    # CATEGORY 4: NUMERICAL TRICKS (questions 19-24)
    # Base rate neglect, anchoring, unit confusion, percentages
    # =========================================================
    {
        "id": 19,
        "category": "numerical_tricks",
        "question": (
            "1% of the population has Disease X. A test for Disease X is 99% accurate (99% true positive, "
            "99% true negative). You test positive. "
            "What is the probability you actually have Disease X? "
            "Choose the closest answer: (A) 99%, (B) 50%, (C) 1%, (D) 10%."
        ),
        "correct_answer": "B",
        "misleading_answer": "A",
        "explanation": (
            "Bayes' theorem: P(disease|positive) = P(positive|disease)*P(disease) / P(positive). "
            "= 0.99*0.01 / (0.99*0.01 + 0.01*0.99) = 0.0099 / 0.0198 ≈ 50%. "
            "Base rate (1%) is critically low; most positives are false positives."
        )
    },
    {
        "id": 20,
        "category": "numerical_tricks",
        "question": (
            "A car travels from city A to city B at 60 mph, then returns at 40 mph. "
            "What is the average speed for the entire round trip?"
        ),
        "correct_answer": "48",
        "misleading_answer": "50",
        "explanation": (
            "Harmonic mean: 2*60*40 / (60+40) = 4800/100 = 48 mph. "
            "Arithmetic mean (50 mph) is wrong because the car spends more time at the slower speed."
        )
    },
    {
        "id": 21,
        "category": "numerical_tricks",
        "question": (
            "A jacket is on sale for 40% off, and then an additional 30% off the sale price. "
            "What is the total percentage off from the original price?"
        ),
        "correct_answer": "58",
        "misleading_answer": "70",
        "explanation": (
            "0.60 * 0.70 = 0.42, so 58% off total. "
            "40% + 30% = 70% is wrong; the second discount applies to the already-discounted price."
        )
    },
    {
        "id": 22,
        "category": "numerical_tricks",
        "question": (
            "In a class of 30 students, 60% passed the math test. Of those who passed, 50% also passed "
            "the science test. Of those who failed math, 25% passed the science test. "
            "How many students passed the science test total?"
        ),
        "correct_answer": "12",
        "misleading_answer": "9",
        "explanation": (
            "Math passers: 30*0.6 = 18. Science passers from math passers: 18*0.5 = 9. "
            "Math failers: 12. Science passers from math failers: 12*0.25 = 3. "
            "Total science passers: 9 + 3 = 12. "
            "Misleading: people only count from math passers (9)."
        )
    },
    {
        "id": 23,
        "category": "numerical_tricks",
        "question": (
            "A population doubles every 20 years. If the population was 1,000 in year 2000, "
            "how many years from 2000 will it take for the population to first exceed 32,000?"
        ),
        "correct_answer": "100",
        "misleading_answer": "640",
        "explanation": (
            "2^5 = 32, so 5 doublings needed. 5 * 20 = 100 years. "
            "Misleading: linear thinking gives 31 * 1000/year * 20 = 640 or similar."
        )
    },
    {
        "id": 24,
        "category": "numerical_tricks",
        "question": (
            "If you flip a fair coin 10 times and get 10 heads in a row, what is the probability "
            "that the 11th flip is heads? "
            "Choose: (A) 1/2, (B) 1/2048, (C) 1/1024, (D) Much less than 1/1024 because of the streak."
        ),
        "correct_answer": "A",
        "misleading_answer": "D",
        "explanation": (
            "Each flip is independent. The probability of heads on the 11th flip is always 1/2. "
            "The gambler's fallacy makes people think the streak affects future probability."
        )
    },

    # =========================================================
    # CATEGORY 5: FRAMING EFFECTS (questions 25-30)
    # Same underlying problem framed differently, tests consistency
    # =========================================================
    {
        "id": 25,
        "category": "framing_effects",
        "question": (
            "A disease is expected to kill 600 people. "
            "Program A will save 200 people for certain. "
            "Program B has a 1/3 probability of saving 600 people and 2/3 probability of saving no one. "
            "Which program do you recommend? Answer A or B."
        ),
        "correct_answer": "A",
        "misleading_answer": "B",
        "explanation": (
            "Expected value: Program A saves 200. Program B: (1/3)*600 + (2/3)*0 = 200. "
            "Equal expected value. Program A (risk-averse) vs B (risk-seeking). "
            "Most people choose A here due to certainty framing. "
            "The 'correct' answer accepts A since it's the risk-dominant strategy."
        )
    },
    {
        "id": 26,
        "category": "framing_effects",
        "question": (
            "The same disease scenario: 600 people expected to die. "
            "Program C: 400 people will die for certain. "
            "Program D: 1/3 probability that nobody dies, 2/3 probability that 600 die. "
            "Which program do you recommend? Answer C or D."
        ),
        "correct_answer": "C",
        "misleading_answer": "D",
        "explanation": (
            "Program C: 400 die = 200 saved. Program D: expected (1/3)*0 + (2/3)*600 = 400 die = same. "
            "This is the SAME as questions 25 with negative framing (loss frame vs gain frame). "
            "Logically equivalent: C = A (200 saved), D = B (same expected). "
            "Correct answer C is consistent with choosing A in the gain frame. "
            "Framing effect: people switch to D due to risk-seeking in loss domain."
        )
    },
    {
        "id": 27,
        "category": "framing_effects",
        "question": (
            "A surgery has a 90% survival rate. "
            "Would you recommend it?"
            " Answer YES or NO."
        ),
        "correct_answer": "YES",
        "misleading_answer": "NO",
        "explanation": (
            "90% survival rate (positive framing) is the same as 10% mortality (negative framing). "
            "Without context about the disease being treated, 90% survival is generally favorable. "
            "Framing effect: 'mortality rate' framing causes more refusals."
        )
    },
    {
        "id": 28,
        "category": "framing_effects",
        "question": (
            "A product has a 1% defect rate. A factory produces 10,000 units per day. "
            "How many defective units are produced per day?"
        ),
        "correct_answer": "100",
        "misleading_answer": "1",
        "explanation": (
            "1% of 10,000 = 100. Straightforward calculation, but framing as 'only 1%' "
            "causes anchoring on the small percentage rather than the large absolute number."
        )
    },
    {
        "id": 29,
        "category": "framing_effects",
        "question": (
            "Study A found that a drug reduces heart attack risk by 50%. "
            "Study B found the same drug reduces absolute heart attack risk from 2% to 1%. "
            "Are these studies contradictory? Answer YES or NO."
        ),
        "correct_answer": "NO",
        "misleading_answer": "YES",
        "explanation": (
            "Both describe the same result: 2% baseline * 0.50 = 1% new rate, or equivalently 50% relative reduction. "
            "Relative risk reduction (50%) vs absolute risk reduction (1 percentage point) measure different things "
            "but can describe the same effect. They are not contradictory."
        )
    },
    {
        "id": 30,
        "category": "framing_effects",
        "question": (
            "A lottery ticket costs $1. The jackpot is $10,000,000. There are 20,000,000 tickets sold. "
            "The expected value of a ticket is -$0.50. "
            "Is the following statement true or false? "
            "'Buying two tickets doubles your chances of winning.' "
            "Answer TRUE or FALSE."
        ),
        "correct_answer": "TRUE",
        "misleading_answer": "FALSE",
        "explanation": (
            "If you have 1 ticket out of 20M, probability = 1/20M. With 2 tickets: 2/20M = 1/10M. "
            "That IS double the probability. The expected value framing misleads people into thinking "
            "the doubling claim is false because the expected value is still negative."
        )
    }
]

# Paraphrase variants for robustness testing (questions 31-36)
PARAPHRASE_VARIANTS = [
    {
        "id": 31,
        "original_id": 1,
        "category": "misleading_math",
        "question": (
            "The combined price of a baseball bat and a baseball is $1.10. "
            "The bat's price exceeds the ball's price by exactly one dollar. "
            "What is the price of the baseball in cents?"
        ),
        "correct_answer": "5",
    },
    {
        "id": 32,
        "original_id": 7,
        "category": "causal_traps",
        "question": (
            "Medical Center X has higher patient mortality (2%) compared to Clinic Y (1%). "
            "A patient needing surgery wants to maximize survival. Which should they choose? "
            "Answer YES to Medical Center X, NO to Clinic Y (meaning choose X). "
            "Or answer NO if you'd choose Clinic Y."
        ),
        "correct_answer": "YES",
    },
    {
        "id": 33,
        "original_id": 13,
        "category": "logical_deception",
        "question": (
            "In a village, truth-tellers always tell the truth and liars always lie. "
            "Person A tells you: 'Both I and person B are liars.' "
            "What must A be? Answer TRUTH-TELLER or LIAR."
        ),
        "correct_answer": "LIAR",
    },
    {
        "id": 34,
        "original_id": 19,
        "category": "numerical_tricks",
        "question": (
            "Rare Condition Z affects 1 in 100 people. A screening test correctly identifies 99% of "
            "those with the condition and correctly clears 99% of those without it. "
            "Your test result is positive. What is the approximate probability you have Condition Z? "
            "(A) About 99% (B) About 50% (C) About 1% (D) About 10%"
        ),
        "correct_answer": "B",
    },
    {
        "id": 35,
        "original_id": 4,
        "category": "misleading_math",
        "question": (
            "A caterpillar is climbing a 10-meter tree. Every daytime it advances 3 meters upward. "
            "Every night it slips back 2 meters. Starting from the bottom, "
            "after how many days does it first reach the top?"
        ),
        "correct_answer": "8",
    },
    {
        "id": 36,
        "original_id": 20,
        "category": "numerical_tricks",
        "question": (
            "A cyclist rides to a destination at 30 km/h and returns at 20 km/h. "
            "What is the cyclist's average speed for the entire journey?"
        ),
        "correct_answer": "24",
    }
]

ALL_QUESTIONS = ADVERSARIAL_BENCHMARK + PARAPHRASE_VARIANTS


def get_questions_by_category(category: str) -> list:
    return [q for q in ADVERSARIAL_BENCHMARK if q["category"] == category]


def get_all_original_questions() -> list:
    return ADVERSARIAL_BENCHMARK


def get_paraphrase_variants() -> list:
    return PARAPHRASE_VARIANTS


if __name__ == "__main__":
    from collections import Counter
    cats = Counter(q["category"] for q in ADVERSARIAL_BENCHMARK)
    print("Adversarial Benchmark Summary:")
    print(f"  Total questions: {len(ADVERSARIAL_BENCHMARK)}")
    for cat, count in cats.items():
        print(f"  {cat}: {count}")
    print(f"\nParaphrase variants: {len(PARAPHRASE_VARIANTS)}")
    print(f"Total (including paraphrases): {len(ALL_QUESTIONS)}")
