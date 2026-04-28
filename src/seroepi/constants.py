"""
Enums for non-user-facing API constants - mostly to help with the app
"""

from enum import StrEnum, auto


# Enums ----------------------------------------------------------------------------------------------------------------
class _UiEnum(StrEnum):

    @classmethod
    def choices(cls) -> list[str]:
        """Returns a raw list of strings for simple dropdowns."""
        return [e.value for e in cls]

    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        """
        Returns a dictionary mapping the strict value to a pretty UI label.
        e.g., {"transmission_cluster": "Transmission Cluster"}
        """
        return {e.value: e.value.replace('_', ' ').title() for e in cls}



class PlotType(_UiEnum):
    FOREST = auto()
    EPICURVE = auto()
    CHOROPLETH = auto()
    COMPOSITION_BAR = auto()
    COMPOSITION_HEATMAP = auto()
    LONGITUDINAL_PREVALENCE = auto()
    VACCINE_COVERAGE = auto()
    STABILITY_BUMP = auto()
    SPATIAL_SURFACE = auto()
    ALPHA_DIVERSITY = auto()
    BETA_HEATMAP = auto()
    NETWORK = auto()


class HoldoutStrategy(_UiEnum):
    COUNTRY = auto()
    TRANSMISSION_CLUSTER = auto()
    STUDY = auto()


class MetricType(_UiEnum):
    PREVALENCE = auto()
    INCIDENCE = auto()


class AggregationType(_UiEnum):
    TRAIT = auto()
    COMPOSITIONAL = auto()


class EstimatorType(_UiEnum):
    FREQUENTIST = auto()
    BAYESIAN = auto()
    REGRESSION = auto()
    SPATIAL = auto()

    @property
    def class_name(self) -> str:
        return {
            self.FREQUENTIST: "FrequentistPrevalenceEstimator",
            self.BAYESIAN: "BayesianPrevalenceEstimator",
            self.REGRESSION: "RegressionPrevalenceEstimator",
            self.SPATIAL: "SpatialPrevalenceEstimator",
        }[self]


class InferenceMethod(_UiEnum):
    MCMC = "mcmc"
    SVI = "svi"
