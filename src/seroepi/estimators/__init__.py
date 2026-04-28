"""
Module for estimating trait prevalence, diversity and incidence among isolates.
"""

from seroepi.estimators._base import (
    Estimates,
    PrevalenceEstimates,
    IncidenceEstimates,
    AlphaDiversityEstimates,
    BetaDiversityEstimates,
)
from seroepi.estimators._core import (
    FrequentistPrevalenceEstimator,
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)

__all__ = [
    "Estimates",
    "PrevalenceEstimates",
    "IncidenceEstimates",
    "AlphaDiversityEstimates",
    "BetaDiversityEstimates",
    "FrequentistPrevalenceEstimator",
    "AlphaDiversityEstimator",
    "BetaDiversityEstimator",
]

try:
    from seroepi.estimators._modelled import (
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