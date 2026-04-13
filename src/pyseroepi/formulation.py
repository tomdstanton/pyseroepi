from dataclasses import dataclass
import pandas as pd
import copy
from abc import ABC, abstractmethod
from pyseroepi.estimators._base import BaseEstimator


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Formulation:
    """
    Represents a proposed vaccine formulation based on antigen prevalence and stability.

    This class holds the results of a formulation design process, including rankings,
    stability metrics from cross-validation, and permutation history.

    Attributes:
        target: The target antigen type (e.g., 'K_locus').
        max_valency: The maximum number of antigens in the formulation.
        rankings: A DataFrame containing the definitive ranking of antigens
            (Antigen, Rank, Prevalence, Cumulative Coverage).
        stability_metrics: A DataFrame containing metrics from LOO stability analysis
            (e.g., Mean LOO Rank, Rank Variance, Probability in Top N).
        permutation_history: A DataFrame containing the full history of ranks
            across all LOO permutations.

    Examples:
        >>> import pandas as pd
        >>> from pyseroepi.formulation import Formulation
        >>> rankings = pd.DataFrame({'antigen': ['K1', 'K2'], 'estimate': [0.5, 0.3]})
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
            from pyseroepi.plotting._base import BasePlotter
        except ImportError as e:
            # The Sophisticated Trap: Catch the specific missing dependency and give a beautiful error
            raise ImportError(
                "Plotting features require the optional 'plot' dependencies.\n"
                "Please install them by running: pip install pyseroepi[plot]"
            ) from None  # 'from None' hides the ugly raw stack trace from the user

        plotter_map = BasePlotter._PLOT_REGISTRY.get(type(self), {})
        if (plotter := plotter_map.get(kind)) is not None:
            available = list(plotter_map.keys())
            raise ValueError(f"Plot type '{kind}' is not registered. Available: {available}")

        return plotter.render(self, **kwargs)

    def get_formulation(self) -> list[str]:
        """
        Returns the top N antigens for the proposed vaccine.

        Returns:
            A list of antigen names.
        """
        return self.rankings.head(self.max_valency)['antigen'].tolist()


class BaseFormulationDesigner(ABC):
    """
    Abstract base class for formulation designers.

    Designers are responsible for evaluating prevalence estimates and generating
    a vaccine formulation with stability metrics.

    Attributes:
        valency: The target valency (number of antigens) for the vaccine.
    """

    def __init__(self, valency: int = 6):
        """
        Initializes the designer.

        Args:
            valency: The target valency. Defaults to 6.
        """
        self.valency = valency


class PostHocFormulationDesigner(BaseFormulationDesigner):
    """
    Fast formulation design using post-hoc estimation.

    This method is valid ONLY for Frequentist estimates where retraining is not
    required for Leave-One-Out (LOO) analysis.
    """

    def evaluate(self, result: 'PrevalenceEstimates', loo_col: str) -> 'Formulation':
        """
        Evaluates prevalence results to design a formulation.

        Args:
            result: The prevalence estimates to evaluate.
            loo_col: The column name to use for Leave-One-Out cross-validation.

        Returns:
            A Formulation object.
        """
        raw_df = result.data
        target = result.target

        # 1. Baseline
        baseline = _extract_ranks(raw_df, target, 'baseline_rank')

        # 2. Permutations
        loo_records = []
        for group in raw_df[loo_col].unique():
            holdout_df = raw_df[raw_df[loo_col] != group]

            ranks = _extract_ranks(holdout_df, target, 'loo_rank')
            ranks['holdout_group'] = group
            loo_records.append(ranks)

        history = pd.concat(loo_records, ignore_index=True)

        # 3. Compile
        return _compile_stability_metrics(baseline, history, target, self.valency)


class CrossValidatedFormulationDesigner(BaseFormulationDesigner):
    """
    Rigorous formulation design using true Leave-One-Out (LOO) cross-validation.

    This method retrains the model for each LOO permutation, which is more
    computationally expensive but necessary for complex models.
    """
    def evaluate(self, estimator: BaseEstimator, agg_df: pd.DataFrame, loo_col: str) -> 'Formulation':
        """
        Evaluates an estimator using LOO cross-validation to design a formulation.

        Args:
            estimator: The estimator instance to use.
            agg_df: The aggregated data for the estimator.
            loo_col: The column name to use for Leave-One-Out cross-validation.

        Returns:
            A Formulation object.
        """
        # 1. Baseline
        baseline_result = copy.deepcopy(estimator).calculate(agg_df)
        target = baseline_result.target
        baseline = _extract_ranks(baseline_result.data, target, 'baseline_rank')

        # 2. Permutations
        loo_records = []
        for group in agg_df[loo_col].unique():
            holdout_df = agg_df[agg_df[loo_col] != group]

            # Retrain from scratch
            loo_result = copy.deepcopy(estimator).calculate(holdout_df)

            ranks = _extract_ranks(loo_result.data, target, 'loo_rank')
            ranks['holdout_group'] = group
            loo_records.append(ranks)

        history = pd.concat(loo_records, ignore_index=True)

        # 3. Compile
        return _compile_stability_metrics(baseline, history, target, self.valency)


# Functions ------------------------------------------------------------------------------------------------------------
def _extract_ranks(df: pd.DataFrame, target: str, rank_col_name: str) -> pd.DataFrame:
    """Consistently groups, sums, sorts, and ranks antigens."""
    ranks = df.groupby(target)['estimate'].sum().sort_values(ascending=False).reset_index()
    ranks[rank_col_name] = ranks.index + 1
    ranks.rename(columns={target: 'antigen'}, inplace=True)
    return ranks


def _compile_stability_metrics(
        baseline: pd.DataFrame,
        history: pd.DataFrame,
        target: str,
        valency: int
) -> 'Formulation':
    """Compiles the final variance and probability matrix for the formulation."""
    stability = []
    for antigen in baseline['antigen']:
        v_hist = history[history['antigen'] == antigen]
        stability.append({
            'antigen': antigen,
            'mean_loo_rank': v_hist['loo_rank'].mean(),
            'rank_variance': v_hist['loo_rank'].var(),
            'probability_in_top_n': (v_hist['loo_rank'] <= valency).mean()
        })

    return Formulation(
        target=target,
        max_valency=valency,
        rankings=baseline,
        stability_metrics=pd.DataFrame(stability).set_index('antigen'),
        permutation_history=history
    )
