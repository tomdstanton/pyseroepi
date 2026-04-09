from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic, Literal
from dataclasses import dataclass
from warnings import warn
import pandas as pd
import numpy as np
from scipy.stats import norm, beta



# Classes --------------------------------------------------------------------------------------------------------------
# Define a TypeVar that represents ANY dataclass you might invent in the future
T_Result = TypeVar('T_Result')

# The BaseEstimator inherits from Generic[T_Result]
class BaseEstimator(ABC, Generic[T_Result]):
    """
    The universal contract for all pyseroepi statistical models.
    """

    def _extract_strata(self, agg_df: pd.DataFrame, exclude_cols: list[str] = None) -> Tuple[list[str], dict]:
        if exclude_cols is None:
            exclude_cols = []

        meta = agg_df.attrs.get("prevalence_meta", {})
        inferred_strata = [col for col in agg_df.columns if col not in exclude_cols]
        stratified_by = meta.get("stratified_by", inferred_strata)
        self._validate_suitability(agg_df, stratified_by)
        return stratified_by, meta

    @staticmethod
    def _validate_suitability(df: pd.DataFrame, strata: list[str]):
        """
        Intelligently checks strata columns for statistical traps using
        Pandas dtypes and naming conventions.
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
        All child estimators MUST implement this.
        The exact return type is defined by the child class signature.
        """
        pass


@dataclass
class PrevalenceEstimates:
    data: pd.DataFrame
    stratified_by: list[str]
    method: str
    prevalence_type: str  # "trait" or "compositional"
    target: str           # e.g., "blaKPC" or "Serotype"


class FrequentistPrevalenceEstimator(BaseEstimator[PrevalenceEstimates]):

    Method = Literal['wilson', 'wald', 'agresti_coull', 'clopper_pearson', 'jeffreys']

    def __init__(self, method: Method = 'wilson', alpha: float = 0.05):
        self.method = method.lower()
        self._method_label = f"frequentist_{self.method}"
        self._method_func = self._METHODS.get(self.method, None)
        if self._method_func is None:
            raise ValueError(f"Unknown method: {self.method}. "
                             f"Choose from: {list(self._METHODS.keys())}")
        self.alpha = alpha

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """Expects the output of df.epi.aggregate_prevalence()"""

        stratified_by, meta = self._extract_strata(agg_df, exclude_cols=['event', 'n'])

        # Extract vectors for fast numpy math
        counts = agg_df['event'].values
        denominators = agg_df['n'].values

        # Route to the selected mathematical method
        prop, lower, upper = self._method_func(counts, denominators, self.alpha)

        new_cols = {
            'estimate': prop,
            'lower': lower,
            'upper': upper,
            'method': self._method_label
        }

        # 2. Fast horizontal concatenation (ignores the deep copy overhead)
        result_df = pd.concat([agg_df, pd.DataFrame(new_cols, index=agg_df.index)], axis=1)
        meta = agg_df.attrs.get("prevalence_meta", {})

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=stratified_by,
            method=self._method_label,
            prevalence_type=meta.get("type", "unknown"),
            target=meta.get("target", "unknown")
        )

    # --- Statistical Methods ---

    @staticmethod
    def _wald_interval(x: np.ndarray, n: np.ndarray, alpha: float):
        """The standard textbook method. Terrible for zero counts."""
        p = x / n
        z = norm.ppf(1 - alpha / 2)
        margin = z * np.sqrt(p * (1 - p) / n)
        return p, np.clip(p - margin, 0, 1), np.clip(p + margin, 0, 1)

    @staticmethod
    def _wilson_score_interval(x: np.ndarray, n: np.ndarray, alpha: float):
        """Robust for small sample sizes and zero events."""
        p = x / n
        z = norm.ppf(1 - alpha / 2)
        z2 = z ** 2
        center = (x + z2 / 2) / (n + z2)
        margin = (z / (n + z2)) * np.sqrt((p * (1 - p) * n) + (z2 / 4))
        return p, np.clip(center - margin, 0, 1), np.clip(center + margin, 0, 1)

    @staticmethod
    def _agresti_coull_interval(x: np.ndarray, n: np.ndarray, alpha: float):
        """Adds z^2/2 successes and z^2 failures. Great alternative to Wilson."""
        p_raw = x / n
        z = norm.ppf(1 - alpha / 2)
        z2 = z ** 2
        n_tilde = n + z2
        p_tilde = (x + z2 / 2) / n_tilde
        margin = z * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        # Note: the estimate returned is still the raw proportion
        return p_raw, np.clip(p_tilde - margin, 0, 1), np.clip(p_tilde + margin, 0, 1)

    @staticmethod
    def _clopper_pearson_interval(x: np.ndarray, n: np.ndarray, alpha: float):
        """The 'Exact' binomial interval using the Beta distribution."""
        p = x / n
        # Lower bound: 0 if x=0, else beta.ppf(alpha/2, x, n-x+1)
        lower = np.where(x == 0, 0.0, beta.ppf(alpha / 2, x, n - x + 1))
        # Upper bound: 1 if x=n, else beta.ppf(1-alpha/2, x+1, n-x)
        upper = np.where(x == n, 1.0, beta.ppf(1 - alpha / 2, x + 1, n - x))
        return p, lower, upper

    @staticmethod
    def _jeffreys_interval(x: np.ndarray, n: np.ndarray, alpha: float):
        """Bayesian-flavored interval using the Jeffreys prior Beta(0.5, 0.5)."""
        p = x / n
        # Lower bound: 0 if x=0
        lower = np.where(x == 0, 0.0, beta.ppf(alpha / 2, x + 0.5, n - x + 0.5))
        # Upper bound: 1 if x=n
        upper = np.where(x == n, 1.0, beta.ppf(1 - alpha / 2, x + 0.5, n - x + 0.5))
        return p, lower, upper

    _METHODS = {
        'wilson': _wilson_score_interval,
        'wald': _wald_interval,
        'agresti_coull': _agresti_coull_interval,
        'clopper_pearson': _clopper_pearson_interval,
        'jeffreys': _jeffreys_interval
    }