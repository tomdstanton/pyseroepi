from typing import Literal
import pandas as pd
import numpy as np
from scipy.stats import norm, beta, entropy
from scipy.spatial.distance import pdist, squareform
from pyseroepi.estimators._base import (BaseEstimator, PrevalenceEstimates, AlphaDiversityEstimates,
                                        BetaDiversityEstimates)


# Classes --------------------------------------------------------------------------------------------------------------
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
            adjusted_for=meta.get("adjusted_for", 'unknown'),
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


class AlphaDiversityEstimator(BaseEstimator[AlphaDiversityEstimates]):
    Metric = Literal['shannon', 'simpson', 'richness']
    _DEFAULT_METRICS = ['shannon', 'simpson', 'richness']
    def __init__(self, target: str = None, metrics: list[Metric] = None):
        self.target = target
        self.metrics = metrics or self._DEFAULT_METRICS
        self._method_label = "alpha_diversity"

    def calculate(self, div_df: pd.DataFrame) -> AlphaDiversityEstimates:
        # Extract the metadata attached by the accessor
        meta = div_df.attrs.get("diversity_meta", {})
        target_col = meta.get("target", self.target)
        strata = meta.get("stratified_by", [])

        if not target_col:
            raise ValueError("Target trait must be defined either in init or via accessor metadata.")

        results = []

        # If stratified, group by the strata. Otherwise, treat as one global group.
        groups = div_df.groupby(strata, observed=True) if strata else [('Global', div_df)]

        for name, group in groups:
            # We already have the counts! No need to run value_counts() again.
            counts = group['variant_count'].values

            # Filter out true zeroes (important for richness)
            counts = counts[counts > 0]
            if len(counts) == 0:
                continue

            p = counts / counts.sum()

            row = {k: v for k, v in zip(strata, name)} if strata and isinstance(name, tuple) else {}
            if strata and not isinstance(name, tuple):
                row = {strata[0]: name}

            if 'shannon' in self.metrics:
                row['shannon'] = entropy(p, base=np.e)
            if 'simpson' in self.metrics:
                row['simpson'] = 1.0 - np.sum(p ** 2)
            if 'richness' in self.metrics:
                row['richness'] = len(counts)

            row['n_samples'] = counts.sum()
            results.append(row)

        res_df = pd.DataFrame(results)

        return AlphaDiversityEstimates(
            data=res_df,
            stratified_by=strata,
            adjusted_for=meta.get("adjusted_for", 'unknown'),
            target=target_col,
            metrics=self.metrics
        )


class BetaDiversityEstimator(BaseEstimator[BetaDiversityEstimates]):
    def __init__(self, target: str = None, metric: str = 'braycurtis'):
        """
        Calculates between-group dissimilarity.
        Common metrics: 'braycurtis' (abundance-weighted), 'jaccard' (presence/absence).
        """
        self.target = target
        self.metric = metric
        self._method_label = f"beta_diversity_{self.metric}"

    def calculate(self, div_df: pd.DataFrame) -> BetaDiversityEstimates:
        # 1. Extract metadata from the accessor
        meta = div_df.attrs.get("diversity_meta", {})
        target_col = meta.get("target", self.target)
        strata = meta.get("stratified_by", [])

        if not target_col:
            raise ValueError("Target trait must be defined either in init or via accessor metadata.")
        if not strata:
            raise ValueError("Beta diversity requires at least one stratification level to compare groups.")

        # 2. Pivot the data into a Wide Matrix
        # Rows = Strata (e.g., Hospitals), Columns = Variants (e.g., K_loci), Values = Counts
        pivot_df = div_df.pivot_table(
            index=strata,
            columns=target_col,
            values='variant_count',
            fill_value=0,  # CRITICAL: Missing variants in a group must be explicitly 0
            aggfunc='sum'
        )

        # 3. Calculate Pairwise Distances
        # pdist calculates the condensed distance vector, squareform turns it into an NxN matrix
        distances = pdist(pivot_df.values, metric=self.metric)
        dist_matrix = squareform(distances)

        # 4. Format the row/column names for the UI
        # If stratified by multiple columns (e.g., ['Region', 'Year']), we flatten the tuple for display
        strata_names = [
            " | ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
            for idx in pivot_df.index
        ]

        # 5. Wrap back into an explicitly labeled Pandas DataFrame
        result_matrix = pd.DataFrame(
            dist_matrix,
            index=strata_names,
            columns=strata_names
        )

        return BetaDiversityEstimates(
            data=result_matrix,
            stratified_by=strata,
            adjusted_for=meta.get("adjusted_for", 'unknown'),
            target=target_col,
            metric=self.metric
        )