import math

import pytest

from dro_intro import (
    AmbiguitySet,
    portfolio_worst_case_return,
    robust_mean_estimate,
)


def test_ambiguity_set_bounds():
    robust = AmbiguitySet(nominal_mean=10.0, radius=2.0)

    assert robust.worst_case_mean() == 8.0
    assert robust.best_case_mean() == 12.0


def test_robust_mean_estimate_is_conservative():
    samples = [5.0, 7.0, 9.0]
    radius = 1.0

    nominal = sum(samples) / len(samples)
    assert robust_mean_estimate(samples, radius) == nominal - radius


def test_portfolio_worst_case_return_applies_margin_to_each_asset():
    weights = [0.5, 0.5]
    nominal_returns = [0.10, 0.20]
    radius = 0.03

    expected = 0.5 * (0.10 - 0.03) + 0.5 * (0.20 - 0.03)
    assert portfolio_worst_case_return(weights, nominal_returns, radius) == pytest.approx(expected)


@pytest.mark.parametrize(
    "samples, radius",
    [([], 0.1), ([1.0, 2.0], -0.1)],
)
def test_invalid_inputs_raise_value_error(samples, radius):
    with pytest.raises(ValueError):
        robust_mean_estimate(samples, radius)


def test_portfolio_input_validation():
    with pytest.raises(ValueError):
        portfolio_worst_case_return([0.5, 0.5], [0.10], 0.02)

    with pytest.raises(ValueError):
        portfolio_worst_case_return([-0.1, 1.1], [0.10, 0.20], 0.02)

    with pytest.raises(ValueError):
        portfolio_worst_case_return([], [], 0.02)


def test_example_round_trip_matches_hand_calculation():
    weights = [0.4, 0.35, 0.25]
    nominal_returns = [0.12, 0.08, 0.06]
    radius = 0.02

    expected = 0.4 * (0.12 - 0.02) + 0.35 * (0.08 - 0.02) + 0.25 * (0.06 - 0.02)
    assert portfolio_worst_case_return(weights, nominal_returns, radius) == pytest.approx(expected)
    assert math.isclose(expected, 0.071, rel_tol=1e-9)
