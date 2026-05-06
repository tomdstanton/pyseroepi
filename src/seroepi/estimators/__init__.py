"""
Module for estimating trait prevalence, diversity and incidence among isolates.
"""

from seroepi.estimators._base import (BaseEstimator, Estimates, PrevalenceEstimates, IncidenceEstimates,
                                      AlphaDiversityEstimates, BetaDiversityEstimates)
from seroepi.estimators._core import (FrequentistPrevalenceEstimator, AlphaDiversityEstimator,
                                      BetaDiversityEstimator)

__all__ = [
    "Estimates",
    "PrevalenceEstimates",
    "IncidenceEstimates",
    "AlphaDiversityEstimates",
    "BetaDiversityEstimates",
    "BaseEstimator",
    "FrequentistPrevalenceEstimator",
    "AlphaDiversityEstimator",
    "BetaDiversityEstimator",
]

try:
    from seroepi.estimators._modelled import (
        ModelledMixin, BayesianMixin,
        BayesianPrevalenceEstimator,
        RegressionPrevalenceEstimator,
        SpatialPrevalenceEstimator,
        RegressionIncidenceEstimator,
    )
    __all__.extend([
        "ModelledMixin", "BayesianMixin"
        "BayesianPrevalenceEstimator",
        "RegressionPrevalenceEstimator",
        "SpatialPrevalenceEstimator",
        "RegressionIncidenceEstimator",
    ])
except ImportError:
    pass
