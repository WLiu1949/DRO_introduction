"""A minimal introduction to distributionally robust optimization.

The example uses a simple ambiguity radius: we assume the true expected return of
an asset may differ from the nominal sample mean by at most ``radius``. The
worst-case estimate is therefore the nominal estimate minus that radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AmbiguitySet:
    """A toy ambiguity set defined around a nominal mean.

    The true expectation is assumed to live within [nominal_mean - radius,
    nominal_mean + radius].
    """

    nominal_mean: float
    radius: float

    def worst_case_mean(self) -> float:
        """Return the conservative lower bound on the expected return."""
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        return self.nominal_mean - self.radius

    def best_case_mean(self) -> float:
        """Return the optimistic upper bound on the expected return."""
        if self.radius < 0:
            raise ValueError("radius must be non-negative")
        return self.nominal_mean + self.radius


def robust_mean_estimate(samples: Iterable[float], radius: float) -> float:
    """Return the worst-case mean estimate under a radius-based ambiguity set."""
    data = list(samples)
    if not data:
        raise ValueError("samples must contain at least one value")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    nominal_mean = sum(data) / len(data)
    return AmbiguitySet(nominal_mean=nominal_mean, radius=radius).worst_case_mean()


def portfolio_worst_case_return(
    weights: Sequence[float],
    nominal_returns: Sequence[float],
    radius: float,
) -> float:
    """Return a conservative portfolio return estimate.

    Each asset's expected return is treated as uncertain up to ``radius``. The
    worst-case estimate applies the same ambiguity margin to every asset, which is
    a light-weight DRO-style approximation.
    """
    if len(weights) != len(nominal_returns):
        raise ValueError("weights and nominal_returns must have the same length")
    if not weights:
        raise ValueError("at least one asset is required")
    if any(weight < 0 for weight in weights):
        raise ValueError("weights must be non-negative")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    return sum(weight * (return_value - radius) for weight, return_value in zip(weights, nominal_returns))


def main() -> None:
    """Print a compact example that explains DRO in plain terms."""
    weights = [0.4, 0.35, 0.25]
    nominal_returns = [0.12, 0.08, 0.06]
    radius = 0.02

    worst_case_return = portfolio_worst_case_return(weights, nominal_returns, radius)
    nominal_portfolio_return = sum(weight * return_value for weight, return_value in zip(weights, nominal_returns))

    print("Distributionally robust optimization (DRO) protects decisions against uncertain data.")
    print(f"Nominal estimate: {nominal_portfolio_return:.2%}")
    print(f"Worst-case estimate with ambiguity radius {radius:.2%}: {worst_case_return:.2%}")
    print("The robust view rewards decision rules that stay safe even under the least favorable distribution.")


if __name__ == "__main__":
    main()
