"""
RecoverAI confidence assessment.

Confidence is derived from the decision structure rather
than requiring a different ML model.

Signals:
    1. Probability distance from 0.5
    2. Difference between the best and second-best action
    3. Number of available actions
"""


def probability_confidence(
    probability: float,
) -> float:
    """
    Convert a probability into confidence.

    A probability close to 0.5 indicates uncertainty.
    A probability close to 0 or 1 indicates stronger
    model confidence.
    """

    probability = max(
        0.0,
        min(
            1.0,
            float(probability),
        ),
    )

    return abs(
        probability - 0.5
    ) * 2.0


def action_margin(
    best_score: float,
    second_score: float,
) -> float:
    """
    Measure separation between the best and
    second-best action.
    """

    return max(
        0.0,
        float(best_score)
        - float(second_score),
    )


def assess_confidence(
    probability: float,
    best_score: float = 0.0,
    second_score: float = 0.0,
    available_actions: int = 1,
) -> dict:
    """
    Produce an interpretable confidence assessment.

    This is intentionally model-agnostic.
    """

    probability_conf = (
        probability_confidence(
            probability
        )
    )

    margin = action_margin(
        best_score,
        second_score,
    )

    if available_actions <= 1:
        margin_conf = 1.0

    else:
        margin_conf = min(
            1.0,
            margin / 0.10,
        )

    confidence = (
        0.6 * probability_conf
        + 0.4 * margin_conf
    )

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    if confidence >= 0.70:
        level = "high"

    elif confidence >= 0.40:
        level = "medium"

    else:
        level = "low"

    return {
        "confidence_score": confidence,
        "confidence_level": level,
        "probability_confidence": (
            probability_conf
        ),
        "action_margin": margin,
        "margin_confidence": margin_conf,
    }