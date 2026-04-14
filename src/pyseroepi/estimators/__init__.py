"""
Module for estimating trait prevalence, diversity and incidence among isolates.
"""

from pyseroepi.estimators._core import (
    FrequentistPrevalenceEstimator,
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)

__all__ = [
    "FrequentistPrevalenceEstimator",
    "AlphaDiversityEstimator",
    "BetaDiversityEstimator",
]

try:
    from pyseroepi.estimators._modelled import (
        BayesianPrevalenceEstimator,
        RegressionPrevalenceEstimator,
        SpatialPrevalenceEstimator,
        RegressionIncidenceEstimator,
    )
    __all__.extend([
        "BayesianPrevalenceEstimator",
        "RegressionPrevalenceEstimator",
        "SpatialPrevalenceEstimator",
        "RegressionIncidenceEstimator",
    ])
except ImportError:
    pass