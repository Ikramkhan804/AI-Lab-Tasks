# Task 1: AI milestones, arranged chronologically
milestones = [
    {
        "year_or_period": "1950",
        "event": "Turing proposes the imitation game",
        "significance": (
            "Turing reframed the question of machine intelligence "
            "as an observable text conversation."
        ),
        "source": "https://academic.oup.com/mind/article/LIX/236/433/986238",
    },
    {
        "year_or_period": "1955–1956",
        "event": "Dartmouth proposal and summer project",
        "significance": (
            "The proposal named artificial intelligence as a research "
            "program and helped establish AI as a field."
        ),
        "source": "https://www-formal.stanford.edu/jmc/history/dartmouth.html",
    },
    {
        "year_or_period": "1966",
        "event": "ELIZA is published",
        "significance": (
            "ELIZA showed how conversational responses could sound "
            "understanding without establishing that the program understood."
        ),
        "source": "https://doi.org/10.1145/365153.365168",
    },
    {
        "year_or_period": "c. 1974–1980",
        "event": "First AI winter",
        "significance": (
            "Expectations and support for AI cooled after the field's "
            "ambitious claims met technical limits and skepticism."
        ),
        "source": (
            "https://publications.parliament.uk/pa/ld201719/ldselect/"
            "ldai/100/10005.htm"
        ),
    },
    {
        "year_or_period": "1980s",
        "event": "Expert systems expand",
        "significance": (
            "Rule-based expert systems brought narrow specialist decision "
            "procedures into commercial use."
        ),
        "source": (
            "https://publications.parliament.uk/pa/ld201719/ldselect/"
            "ldai/100/10005.htm"
        ),
    },
    {
        "year_or_period": "2012",
        "event": "AlexNet uses ImageNet data and GPU compute",
        "significance": (
            "A deep neural network trained on a large labeled image dataset "
            "using GPUs substantially improved image-classification results."
        ),
        "source": (
            "https://papers.nips.cc/paper_files/paper/2012/hash/"
            "c399862d3b9d6b76c8436e924a68c45b-Abstract.html"
        ),
    },
]

for milestone in milestones:
    print(
        f"{milestone['year_or_period']} — {milestone['event']}\n"
        f"Significance: {milestone['significance']}\n"
        f"Source: {milestone['source']}\n"
    )

    # Task 2: Scholarship rules and exactly four core cases
# All thresholds and conditions are synthetic lab assumptions.

def assess(applicant):
    if applicant["score"] < 70:
        return "FAIL", "Score is below the synthetic minimum of 70."

    if not applicant["documents_complete"]:
        return "FAIL", "Required documents are incomplete."

    if not applicant["local_resident"]:
        return "FAIL", "LocalResidency condition is not met."

    if applicant["annual_income"] > 50_000:
        return "FAIL", "IncomeCap condition is not met."

    return "PASS", "All four explicit rules are met."


cases = [
    {
        "name": "Case 1 — pass",
        "input": {
            "score": 82,
            "documents_complete": True,
            "local_resident": True,
            "annual_income": 42_000,
        },
        "expected": "PASS",
    },
    {
        "name": "Case 2 — LocalResidency failure",
        "input": {
            "score": 82,
            "documents_complete": True,
            "local_resident": False,
            "annual_income": 42_000,
        },
        "expected": "FAIL",
    },
    {
        "name": "Case 3 — IncomeCap failure",
        "input": {
            "score": 82,
            "documents_complete": True,
            "local_resident": True,
            "annual_income": 50_001,
        },
        "expected": "FAIL",
    },
    {
        "name": "Case 4 — score threshold boundary",
        "input": {
            "score": 70,
            "documents_complete": True,
            "local_resident": True,
            "annual_income": 50_000,
        },
        "expected": "PASS",
    },
]

for case in cases:
    actual, reason = assess(case["input"])
    print(
        f"{case['name']}\n"
        f"Input:    {case['input']}\n"
        f"Expected: {case['expected']}\n"
        f"Actual:   {actual}\n"
        f"Reason:   {reason}\n"
    )