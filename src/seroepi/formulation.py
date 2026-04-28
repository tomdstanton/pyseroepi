"""
Module for abstracting a vaccine formulation using trait prevalence and stability.
"""
from dataclasses import dataclass
import pandas as pd
from abc import ABC
from typing import Optional
from joblib import Parallel, delayed
from seroepi.estimators._base import BaseEstimator


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Formulation:
    """
    Represents a proposed vaccine formulation based on target prevalence and stability.

    This class holds the results of a formulation design process, including rankings,
    stability metrics from cross-validation, and permutation history.

    Attributes:
        target: The target target type (e.g., 'K_locus').
        max_valency: The maximum number of targets in the formulation.
        rankings: A DataFrame containing the definitive ranking of targets
            (Antigen, Rank, Prevalence, Cumulative Coverage).
        stability_metrics: A DataFrame containing metrics from LOO stability analysis
            (e.g., Mean LOO Rank, Rank Variance, Probability in Top N).
        permutation_history: A DataFrame containing the full history of ranks
            across all LOO permutations.

    Examples:
        >>> import pandas as pd
        >>> from seroepi.formulation import Formulation
        >>> rankings = pd.DataFrame({'target': ['K1', 'K2'], 'estimate': [0.5, 0.3]})
        >>> formulation = Formulation(
        ...     target='K_locus',
        ...     max_valency=2,
        ...     rankings=rankings,
        ...     stability_metrics=pd.DataFrame(),
        ...     permutation_history=pd.DataFrame()
        ... )
        >>> print(formulation.get_formulation())
        ['K1', 'K2']
    """
    target: str  # e.g., 'K_locus'
    max_valency: int  # e.g., 6 (for a hexavalent vaccine)
    # The definitive ranking matrix (Antigen, Rank, Prevalence, Cumulative Coverage)
    rankings: pd.DataFrame
    # Leave-One-Out (LOO) Stability Data
    # Rows = Antigens, Cols = 'Rank Variance', 'Prob in Top N', etc.
    stability_metrics: pd.DataFrame
    # Full history of ranks across all LOO permutations (used for plotting)
    permutation_history: pd.DataFrame

    @classmethod
    def from_custom(
            cls,
            custom_targets: list[str],
            baseline_result: 'PrevalenceEstimates'
    ) -> 'Formulation':
        """
        Creates a custom Formulation from a user-defined list of targets.
        Calculates the baseline coverage for these specific targets.
        """
        target_col = baseline_result.target
        raw_df = baseline_result.data

        # 1. Calculate the true baseline prevalence for everything
        baseline = raw_df.groupby(target_col)['estimate'].sum().sort_values(ascending=False).reset_index()
        baseline.rename(columns={target_col: 'target'}, inplace=True)

        # 2. Filter and reorder the baseline to match the user's custom list exactly
        # We use pd.Categorical to ensure the dataframe retains the exact order the user requested
        baseline['target_cat'] = pd.Categorical(baseline['target'], categories=custom_targets, ordered=True)
        custom_rankings = baseline.dropna(subset=['target_cat']).sort_values('target_cat').drop(
            columns=['target_cat'])

        # The new rank is simply the order they requested them in
        custom_rankings['baseline_rank'] = range(1, len(custom_targets) + 1)

        # 3. Create empty stability metrics (since this is a manual override, not a LOO calculation)
        empty_df = pd.DataFrame()

        return cls(
            target=target_col,
            max_valency=len(custom_targets),
            rankings=custom_rankings,
            stability_metrics=empty_df,
            permutation_history=empty_df
        )

    def plot(self, kind: str, **kwargs):
        """
        Renders a visualization of the formulation results.

        Args:
            kind: The type of plot to render.
            **kwargs: Additional arguments passed to the plotter.

        Returns:
            A plotly Figure object.

        Raises:
            ImportError: If the 'plot' optional dependencies are not installed.
            ValueError: If the requested plot kind is not registered.
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

    def get_formulation(self) -> list[str]:
        """
        Returns the top N targets for the proposed vaccine.

        Returns:
            A list of target names.
        """
        return self.rankings.head(self.max_valency)['target'].tolist()


class BaseFormulationDesigner(ABC):
    """
    Abstract base class for formulation designers.

    Designers are responsible for evaluating prevalence estimates and generating
    a vaccine formulation with stability metrics.

    Attributes:
        valency: The target valency (number of targets) for the vaccine.
        n_jobs: The number of CPU cores to use for processing (-1 for all).
        formulation_: The resulting Formulation object after fitting.
    """

    def __init__(self, valency: int = 6, n_jobs: int = -1):
        """
        Initializes the designer.

        Args:
            valency: The target valency. Defaults to 6.
            n_jobs: Number of concurrent workers. Defaults to -1 (all available).
        """
        self.valency = valency
        self.n_jobs = n_jobs
        self.formulation_: Optional[Formulation] = None

    def fit(self, *args, **kwargs) -> 'BaseFormulationDesigner':
        """Calculates the formulation and stores it in self.formulation_"""
        pass


class PostHocFormulationDesigner(BaseFormulationDesigner):
    """
    Fast formulation design using post-hoc estimation.

    This method is valid ONLY for Frequentist estimates where retraining is not
    required for Leave-One-Out (LOO) analysis.
    """

    def fit(self, result: 'PrevalenceEstimates', loo_col: str) -> 'PostHocFormulationDesigner':
        """
        Evaluates prevalence results to design a formulation.

        Args:
            result: The prevalence estimates to evaluate.
            loo_col: The column name to use for Leave-One-Out cross-validation.

        Returns:
            The fitted designer instance.
        """
        raw_df = result.data
        target = result.target

        # 1. Baseline
        baseline = _extract_ranks(raw_df, target, 'baseline_rank')

        # 2. Permutations (Vectorized O(N) Subtraction)
        # Pre-calculate global sums and the individual group sums
        total_estimates = raw_df.groupby(target)['estimate'].sum()
        group_target_estimates = raw_df.groupby([loo_col, target])['estimate'].sum().unstack(fill_value=0)

        loo_records = []
        for group in raw_df[loo_col].unique():
            # Subtract the holdout group's contribution from the global total
            if group in group_target_estimates.index:
                loo_estimates = total_estimates - group_target_estimates.loc[group]
            else:
                loo_estimates = total_estimates.copy()

            loo_ranks = loo_estimates.sort_values(ascending=False).reset_index()
            loo_ranks.columns = ['target', 'estimate']
            loo_ranks['loo_rank'] = loo_ranks.index + 1
            loo_ranks['holdout_group'] = group
            loo_records.append(loo_ranks)

        history = pd.concat(loo_records, ignore_index=True)

        # 3. Compile
        self.formulation_ = _compile_stability_metrics(baseline, history, target, self.valency)
        return self


class CVFormulationDesigner(BaseFormulationDesigner):
    """
    Rigorous formulation design using true Leave-One-Out (LOO) cross-validation.

    This method retrains the model for each LOO permutation, which is more
    computationally expensive but necessary for complex models.
    """
    def fit(self, estimator: BaseEstimator, agg_df: pd.DataFrame, loo_col: str) -> 'CVFormulationDesigner':
        """
        Evaluates an estimator using LOO cross-validation to design a formulation.

        Args:
            estimator: The estimator instance to use.
            agg_df: The aggregated data for the estimator.
            loo_col: The column name to use for Leave-One-Out cross-validation.

        Returns:
            The fitted designer instance.
        """
        # 1. Baseline
        baseline_result = _clone_estimator(estimator).calculate(agg_df)
        target = baseline_result.target
        baseline = _extract_ranks(baseline_result.data, target, 'baseline_rank')

        # 2. Permutations (Parallel Processing)
        groups = agg_df[loo_col].unique()
        
        loo_records = Parallel(n_jobs=self.n_jobs)(
            delayed(_run_cv_fold)(estimator, agg_df, loo_col, group, target) 
            for group in groups
        )

        history = pd.concat(loo_records, ignore_index=True)

        # 3. Compile
        self.formulation_ = _compile_stability_metrics(baseline, history, target, self.valency)
        return self


# Functions ------------------------------------------------------------------------------------------------------------
def _extract_ranks(df: pd.DataFrame, target: str, rank_col_name: str) -> pd.DataFrame:
    """Consistently groups, sums, sorts, and ranks targets."""
    ranks = df.groupby(target)['estimate'].sum().sort_values(ascending=False).reset_index()
    ranks[rank_col_name] = ranks.index + 1
    ranks.rename(columns={target: 'target'}, inplace=True)
    return ranks


def _run_cv_fold(estimator: BaseEstimator, agg_df: pd.DataFrame, loo_col: str, group: str, target: str) -> pd.DataFrame:
    """Helper function to execute a single CV fold in parallel."""
    holdout_df = agg_df[agg_df[loo_col] != group]
    # Retrain from scratch on the subset
    loo_result = _clone_estimator(estimator).calculate(holdout_df)
    ranks = _extract_ranks(loo_result.data, target, 'loo_rank')
    ranks['holdout_group'] = group
    return ranks

def _compile_stability_metrics(
        baseline: pd.DataFrame,
        history: pd.DataFrame,
        target: str,
        valency: int
) -> 'Formulation':
    """Compiles the final variance and probability matrix for the formulation."""
    stability = []
    for target in baseline['target']:
        v_hist = history[history['target'] == target]
        stability.append({
            'target': target,
            'mean_loo_rank': v_hist['loo_rank'].mean(),
            'rank_variance': v_hist['loo_rank'].var(),
            'probability_in_top_n': (v_hist['loo_rank'] <= valency).mean()
        })

    return Formulation(
        target=target,
        max_valency=valency,
        rankings=baseline,
        stability_metrics=pd.DataFrame(stability).set_index('target'),
        permutation_history=history
    )


def _clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """
    Safely creates a fresh, unfitted instance of the estimator.
    Avoids deepcopy pickling errors caused by compiled JAX/NumPyro models.
    """
    if hasattr(estimator, 'get_params'):
        # Scikit-learn API compatibility (safely transfers hyperparameters)
        return type(estimator)(**estimator.get_params())
    # Default seroepi fallback
    return type(estimator)()
