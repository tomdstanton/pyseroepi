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
    CUMULATIVE_COVERAGE = auto()
    STABILITY_BUMP = auto()
    SPATIAL_SURFACE = auto()
    ALPHA_DIVERSITY = auto()
    BETA_HEATMAP = auto()
    NETWORK = auto()
    LONGEVITY = auto()


class HoldoutStrategy(_UiEnum):
    COUNTRY = auto()
    TRANSMISSION_CLUSTER = auto()
    STUDY = auto()


class MetricType(_UiEnum):
    PREVALENCE = auto()
    DIVERSITY = auto()
    INCIDENCE = auto()


class AggregationType(_UiEnum):
    TRAIT = auto()
    COMPOSITIONAL = auto()


class Domain(_UiEnum):
    AMR = auto()
    VIRULENCE = auto()
    QC = auto()
    SPATIAL = auto()
    TEMPORAL = auto()
    SPATIAL_RES = auto()
    TEMPORAL_RES = auto()
    GENOTYPE = "geno"
    PHENOTYPE = "pheno"
    CLUSTER = auto()


class DistanceFlavour(_UiEnum):
    PATHOGENWATCH = auto()
    SKA2 = auto()
    NEWICK = auto()


class GenotypeFlavour(_UiEnum):
    PATHOGENWATCH_KLEBORATE = "pathogenwatch-kleborate"


class EstimatorType(_UiEnum):
    UNPOOLED = auto()
    GLM = auto()
    BAYESIAN = auto()
    SPATIAL = auto()

    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        return {
            cls.UNPOOLED.value: "Frequentist (Unpooled CIs)",
            cls.GLM.value: "Frequentist (GLM)",
            cls.BAYESIAN.value: "Bayesian (Hierarchical)",
            cls.SPATIAL.value: "Bayesian (Spatial GP)",
        }

    @property
    def class_name(self) -> str:
        return {
            self.UNPOOLED: "UnpooledPrevalenceEstimator",
            self.BAYESIAN: "BayesianPrevalenceEstimator",
            self.GLM: "GLMPrevalenceEstimator",
            self.SPATIAL: "SpatialPrevalenceEstimator",
        }[self]


class BayesianInferenceMethod(_UiEnum):
    MCMC = auto()
    SVI = auto()


class TemporalResolution(_UiEnum):
    YEAR = auto()
    MONTH = auto()
    WEEK = auto()
    DAY = auto()
    UNKNOWN = auto()

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN

    @property
    def pandas_offset(self) -> str:
        """Returns the modern Pandas 2.2+ start-offset alias."""
        return {
            self.YEAR: 'YS',
            self.MONTH: 'MS',
            self.WEEK: 'W-MON',
            self.DAY: 'D',
            self.UNKNOWN: ''
        }[self]

    @property
    def pandas_period(self) -> str:
        """Returns the Pandas period alias."""
        return {
            self.YEAR: 'Y',
            self.MONTH: 'M',
            self.WEEK: 'W',
            self.DAY: 'D',
            self.UNKNOWN: ''
        }[self]


class SpatialResolution(_UiEnum):
    GLOBAL = auto()
    CONTINENT = auto()
    REGION = auto()
    COUNTRY = auto()
    CITY = auto()
    HOSPITAL = auto()
    EXACT = auto()
    UNKNOWN = auto()

    @classmethod
    def _missing_(cls, value):
        return cls.UNKNOWN


class DistanceMetricType(StrEnum):
    """
    Enumeration of supported metric types for pairwise comparisons.

    These metric types define how to interpret the numerical values
    in a distance/similarity matrix.

    Attributes:
        ABSOLUTE_DISTANCE: An absolute distance measure (e.g., 5 SNPs).
        RELATIVE_DISTANCE: A relative distance measure typically between 0.0 and 1.0 (e.g., 0.05 Hamming).
        ABSOLUTE_SIMILARITY: An absolute similarity measure (e.g., 95 shared nucleotides).
        RELATIVE_SIMILARITY: A relative similarity measure typically between 0.0 and 1.0 (e.g., 0.95 Jaccard).
    """
    ABSOLUTE_DISTANCE = auto()  # e.g., 5 SNPs
    RELATIVE_DISTANCE = auto()  # e.g., 0.05 Hamming
    ABSOLUTE_SIMILARITY = auto()  # e.g., 95 shared nucleotides
    RELATIVE_SIMILARITY = auto()  # e.g., 0.95 Jaccard

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            if v := cls.__members__.get(value.upper().replace(" ", "_").replace("-", "_")):
                return v
        return cls.ABSOLUTE_DISTANCE