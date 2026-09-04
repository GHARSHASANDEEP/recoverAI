import pytest
from src.engine.confidence import (
    assess_confidence,
    probability_confidence,
    action_margin,
)


def test_probability_near_half_is_uncertain():
    confidence = probability_confidence(
        0.50
    )

    assert confidence == 0.0


def test_probability_near_one_is_confident():
    confidence = probability_confidence(
        0.95
    )

    assert confidence > 0.8


def test_large_action_margin():
    margin = action_margin(
        0.90,
        0.50,
    )

    assert margin == 0.40


def test_small_action_margin():
    margin = action_margin(
        0.57,
        0.56,
    )

    assert margin == pytest.approx(
        0.01
    )


def test_high_confidence_prediction():

    result = assess_confidence(
        probability=0.90,
        best_score=0.90,
        second_score=0.50,
        available_actions=3,
    )

    assert result[
        "confidence_level"
    ] == "high"

    assert result[
        "confidence_score"
    ] >= 0.70


def test_low_confidence_prediction():

    result = assess_confidence(
        probability=0.56,
        best_score=0.57,
        second_score=0.56,
        available_actions=3,
    )

    assert result[
        "confidence_level"
    ] == "low"

    assert result[
        "confidence_score"
    ] < 0.40


def test_medium_confidence_prediction():

    result = assess_confidence(
        probability=0.70,
        best_score=0.70,
        second_score=0.65,
        available_actions=3,
    )

    assert result[
        "confidence_level"
    ] == "medium"


def test_single_action_is_confident_by_margin():

    result = assess_confidence(
        probability=0.60,
        best_score=0.60,
        second_score=0.0,
        available_actions=1,
    )

    assert result[
        "margin_confidence"
    ] == 1.0


def test_probability_is_clamped():

    result = assess_confidence(
        probability=1.5,
        best_score=1.5,
        second_score=0.5,
        available_actions=2,
    )

    assert (
        0.0
        <= result["confidence_score"]
        <= 1.0
    )


def test_confidence_result_is_explainable():

    result = assess_confidence(
        probability=0.80,
        best_score=0.80,
        second_score=0.60,
        available_actions=3,
    )

    assert (
        "probability_confidence"
        in result
    )

    assert (
        "action_margin"
        in result
    )

    assert (
        "margin_confidence"
        in result
    )

    assert (
        "confidence_level"
        in result
    )