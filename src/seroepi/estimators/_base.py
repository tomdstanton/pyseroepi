from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Optional
from dataclasses import dataclass
from warnings import warn
import pandas as pd
from seroepi.constants import PlotType, AggregationType


# Classes --------------------------------------------------------------------------------------------------------------
# Define a TypeVar that represents ANY dataclass you might invent in the future
T_Result = TypeVar('T_Result')

# The BaseEstimator inherits from Generic[T_Result]
class BaseEstimator(ABC, Generic[T_Result]):
    """
    The universal contract for all seroepi statistical models.

    All prevalence, diversity, and incidence estimators must inherit from this
    class and implement the `calculate` method.
    """

    def _extract_strata(self, agg_df: pd.DataFrame, exclude_cols: list[str] = None) -> Tuple[list[str], dict]:
        """
        Extracts stratification columns and metadata from an aggregated DataFrame.

        Args:
            agg_df: The aggregated DataFrame.
            exclude_cols: Column names to exclude from stratification.

        Returns:
            A tuple containing the list of strata columns and the metadata dictionary.
        """
        if exclude_cols is None:
            exclude_cols = []

        meta = agg_df.attrs.get("metric_meta", {})
        inferred_strata = [col for col in agg_df.columns if col not in exclude_cols]
        stratified_by = meta.get("stratified_by", inferred_strata)
        self._validate_suitability(agg_df, stratified_by)
        return stratified_by, meta

    @staticmethod
    def _validate_suitability(df: pd.DataFrame, strata: list[str]):
        """
        Validates that the stratification columns are suitable for modeling.

        Checks for common statistical traps such as continuous variables being
        used as strata or over-stratification by unique identifiers.

        Args:
            df: The DataFrame to check.
            strata: The list of stratification columns.

        Raises:
            ValueError: If a stratum is a continuous float.
        """
        for col in strata:
            # 1. The Continuous Float Trap
            if pd.api.types.is_float_dtype(df[col]):
                raise ValueError(
                    f"Strata column '{col}' is a continuous float. "
                    "Prevalence estimators require discrete categorical groups. "
                    f"Please bin this variable (e.g., using pd.cut()) before aggregating."
                )

            # 2. The Raw Datetime Trap (Checks dtype OR the 'meta_' prefix)
            if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower():
                # We only warn here, because sometimes daily is intentional during a rapid outbreak
                warn(
                    f"You are stratifying on a raw date column '{col}'. "
                    "This will calculate prevalence for every single day. "
                    "Consider bucketing by month/year using .dt.to_period('M') first.",
                    UserWarning
                )

        # 3. The Primary Key / Over-Stratification Trap
        if 'n' in df.columns:
            # If the average group size is close to 1, they shattered the dataset
            avg_group_size = df['n'].mean()
            if len(df) > 1 and avg_group_size < 1.5:
                warn(
                    f"The average group size is {avg_group_size:.2f}. "
                    "Did you accidentally stratify by a unique identifier (like 'sample_id')? "
                    "This will cause your models to overfit or crash.",
                    UserWarning
                )

    @abstractmethod
    def calculate(self, df: pd.DataFrame) -> T_Result:
        """
        Executes the estimator's logic on the provided DataFrame.

        Args:
            df: The input DataFrame (usually aggregated).

        Returns:
            An Estimates object (e.g., PrevalenceEstimates).
        """
        pass


# Result dataclasses ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Estimates:
    """
    Base container for statistical estimates.

    Attributes:
        data: A DataFrame containing the estimates and original strata.
        stratified_by: List of columns used for stratification.
        adjusted_for: Column name used for cluster adjustment, if any.
        target: The target variable for which estimates were calculated.
    """
    data: pd.DataFrame
    stratified_by: list[str]
    adjusted_for: Optional[str]
    target: str  # e.g., "blaKPC" or "Serotype"

    def plot(self, kind: PlotType, **kwargs):
        """
        Renders a visualization of the estimates.

        Args:
            kind: The type of plot to render.
            **kwargs: Additional arguments passed to the plotter.

        Returns:
            A plotly Figure object.
        """
        try:  # LAZY IMPORT: We only try to import the registry when the user actually calls .plot()
            from seroepi.plotting._base import BasePlotter
        except ImportError as e:
            # The Sophisticated Trap: Catch the specific missing dependency and give a beautiful error
            raise ImportError(
                "Plotting features require the optional 'plot' dependencies.\n"
                "Please install them by running: pip install seroepi[plot]"
            ) from None  # 'from None' hides the ugly raw stack trace from the user

        plotter_map = BasePlotter._PLOT_REGISTRY.get(type(self), {})
        if (plotter := plotter_map.get(kind)) is None:
            available = list(plotter_map.keys())
            raise ValueError(f"Plot type '{kind}' is not registered. Available: {available}")

        return plotter.render(self, **kwargs)


@dataclass(frozen=True, slots=True)
class PrevalenceEstimates(Estimates):
    """
    Container for prevalence results.

    Attributes:
        method: The statistical method used (e.g., 'bayesian_mcmc').
        aggregation_type: Either 'trait' or 'compositional'.
    """
    method: str
    aggregation_type: AggregationType  # "trait" or "compositional"


@dataclass(frozen=True, slots=True)
class AlphaDiversityEstimates(Estimates):
    """
    Container for Alpha Diversity results.

    Attributes:
        metrics: List of diversity metrics calculated (e.g., ['shannon', 'simpson']).
    """
    metrics: list[str]


@dataclass(frozen=True, slots=True)
class BetaDiversityEstimates(Estimates):
    """
    Container for Beta Diversity results (distance matrices).

    Attributes:
        metric: The distance metric used (e.g., 'braycurtis').
    """
    metric: str


@dataclass(frozen=True, slots=True)
class IncidenceEstimates(Estimates):
    """
    Container for time-series incidence results.

    Attributes:
        freq: The time resolution used (e.g., 'ME').
        aggregation_type: Either 'trait' or 'compositional'.
        model_results: A DataFrame containing regression outputs (IRR, CIs, P-values).
    """
    freq: str                   # The time resolution (e.g., 'ME')
    aggregation_type: AggregationType         # "trait" or "compositional"
    model_results: pd.DataFrame # The regression outputs (IRR, CIs, P-values)
