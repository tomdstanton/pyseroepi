"""
Module for estimating trait prevalence, diversity and incidence among isolates.
"""

from ._base import (
    AlphaDiversityEstimates,
    BaseEstimator,
    BetaDiversityEstimates,
    Estimates,
    IncidenceEstimates,
    PrevalenceEstimates,
)
from ._core import (
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
    UnpooledPrevalenceEstimator,
)

__all__ = (
    "AlphaDiversityEstimates",
    "AlphaDiversityEstimator",
    "BaseEstimator",
    "BetaDiversityEstimates",
    "BetaDiversityEstimator",
    "Estimates",
    "IncidenceEstimates",
    "PrevalenceEstimates",
    "UnpooledPrevalenceEstimator",
)

try:
    from ._modelled import (
        BayesianIncidenceEstimator,
        BayesianMixin,
        BayesianPrevalenceEstimator,
        GLMIncidenceEstimator,
        GLMPrevalenceEstimator,
        ModelledMixin,
        SpatialPrevalenceEstimator,
    )

    __all__ += (
        "BayesianIncidenceEstimator",
        "BayesianMixin",
        "BayesianPrevalenceEstimator",
        "GLMIncidenceEstimator",
        "GLMPrevalenceEstimator",
        "ModelledMixin",
        "SpatialPrevalenceEstimator",
    )
except ImportError:
    pass
