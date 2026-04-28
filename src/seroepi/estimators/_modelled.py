from typing import Literal, Union, TypeVar, Type
from abc import ABC, abstractmethod
from pathlib import Path
from joblib import dump as joblib_dump, load as joblib_load
from warnings import warn
from multiprocessing import cpu_count
import functools
import io
import contextlib
import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random, vmap, jit
from numpyro import optim, distributions as dist, diagnostics as diag, sample as samp, set_host_device_count
from numpyro.infer import MCMC, NUTS, Trace_ELBO, SVI, autoguide, Predictive
from seroepi.estimators._base import BaseEstimator, PrevalenceEstimates, IncidenceEstimates
from seroepi.constants import InferenceMethod


# Set-up ---------------------------------------------------------------------------------------------------------------
# Tell JAX to split the CPU into multiple virtual devices for parallel chains
set_host_device_count(min(cpu_count(), 4))


# TypeVars -------------------------------------------------------------------------------------------------------------
T_Modelled = TypeVar('T_Modelled', bound='ModelledMixin')


# TypeVars -------------------------------------------------------------------------------------------------------------
class ModelledMixin(ABC):
    """
    Contract for estimators with an internal fitted state.

    Enforces the scikit-learn fit/predict paradigm and provides universal
    serialization for fitted models.

    Attributes:
        is_fitted_: Boolean indicating if the model has been fitted.
    """

    # State tracking
    is_fitted_: bool = False

    def check_is_fitted(self):
        """
        Checks if the model is fitted.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This estimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'ModelledMixin':
        """Calculates the internal state (e.g., MCMC samples) and saves it to self."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        """Uses the fitted internal state to generate predictions on the dataframe."""
        pass

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Universally serializes the fitted estimator instance to disk.

        Args:
            filepath: Path where the model should be saved.
        """
        if not self.is_fitted_:
            warn(f"You are saving a {self.__class__.__name__} that hasn't been fitted yet.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(self, path)

    @classmethod
    def load_model(cls: Type[T_Modelled], filepath: Union[str, Path]) -> T_Modelled:
        """
        Loads a serialized estimator from disk.

        Args:
            filepath: Path to the serialized model file.

        Returns:
            The loaded estimator instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the loaded model is not of the expected type.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        estimator = joblib_load(path)

        # Strict Type Guard
        if not isinstance(estimator, cls):
            raise TypeError(
                f"Type mismatch: Attempted to load into {cls.__name__}, "
                f"but the file contains a {type(estimator).__name__}."
            )

        return estimator


class BayesianMixin:
    """
    Shared inference logic for NumPyro-based Bayesian estimators.
    """
    def _init_bayesian(self, method: InferenceMethod, num_samples: int, num_chains: int, 
                       num_warmup: int, svi_steps: int, seed: int):
        self.method = InferenceMethod(method) if isinstance(method, str) else method
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.svi_steps = svi_steps
        self.seed = seed
        self.samples_ = None

    def _run_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        """Routes to the correct inference engine based on self.method."""
        if self.method == InferenceMethod.MCMC:
            return self._mcmc_inference(jax_data, rng_key)
        elif self.method == InferenceMethod.SVI:
            return self._svi_inference(jax_data, rng_key)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from MCMC or SVI.")

    def _mcmc_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        """Runs MCMC inference using NUTS."""
        mcmc = MCMC(NUTS(self._model), num_warmup=self.num_warmup, num_samples=self.num_samples,
                    num_chains=self.num_chains, progress_bar=False)
        mcmc.run(rng_key, **jax_data)
        return mcmc.get_samples()

    def _svi_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        """Runs Stochastic Variational Inference."""
        opt_key, pred_key = random.split(rng_key)
        guide = autoguide.AutoNormal(self._model)
        optimizer = optim.Adam(step_size=0.01)
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(opt_key, num_steps=self.svi_steps, **jax_data)
        predictive = Predictive(self._model, guide=guide, params=svi_result.params, num_samples=self.num_samples)
        return predictive(pred_key, **jax_data)

    def diagnostics(self) -> pd.DataFrame:
        """Returns MCMC diagnostics (R-hat, ESS) as a formatted DataFrame."""
        if hasattr(self, 'check_is_fitted'):
            self.check_is_fitted()
        if self.method != InferenceMethod.MCMC:
            raise TypeError("Diagnostics are only available for MCMC inference.")
            
        summary_dict = diag.summary(self.samples_, prob=0.95, group_by_chain=False)
        
        # Flatten multi-dimensional parameters (like fixed/random effects) into individual rows
        rows = []
        for param, stats in summary_dict.items():
            param_shape = np.shape(stats['mean'])
            
            if len(param_shape) == 0:
                row = {'Parameter': param}
                row.update({k: float(v) for k, v in stats.items()})
                rows.append(row)
            else:
                it = np.nditer(np.empty(param_shape), flags=['multi_index'])
                for _ in it:
                    idx = it.multi_index
                    idx_str = ",".join(map(str, idx))
                    row = {'Parameter': f"{param}[{idx_str}]"}
                    row.update({k: float(np.asarray(v)[idx]) for k, v in stats.items()})
                    rows.append(row)
                    
        return pd.DataFrame(rows)


# Estimators -----------------------------------------------------------------------------------------------------------
class BayesianPrevalenceEstimator(BaseEstimator[PrevalenceEstimates], ModelledMixin, BayesianMixin):
    """
    Bayesian hierarchical model for prevalence estimation.

    This estimator uses MCMC or SVI to fit a binomial model with random effects
    for groups and fixed effects for targets. It handles overdispersion and
    provides credible intervals.

    Examples:
        >>> from seroepi.estimators import BayesianPrevalenceEstimator
        >>> estimator = BayesianPrevalenceEstimator(method='mcmc')
        >>> # result = estimator.calculate(agg_df)
    """
    def __init__(self, method: InferenceMethod = InferenceMethod.MCMC, num_samples: int = 1500, num_chains: int = 4,
                 num_warmup: int = 1000, svi_steps: int = 3000, target_event: str = 'event', target_n: str = 'n', seed: int = 42):
        """
        Initializes the BayesianPrevalenceEstimator.

        Args:
            method: Inference method ('mcmc' or 'svi'). Defaults to 'mcmc'.
            num_samples: Number of posterior samples to draw. Defaults to 1500.
            num_chains: Number of MCMC chains. Defaults to 4.
            num_warmup: Number of warmup steps for MCMC. Defaults to 1000.
            svi_steps: Number of optimization steps for SVI. Defaults to 3000.
            target_event: Column name for event counts. Defaults to 'event'.
            target_n: Column name for total counts (denominators). Defaults to 'n'.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self._method_label = f'bayesian_{self.method.value}'
        
        self.target_event = target_event
        self.target_n = target_n

        # Fitted attributes (trailing underscores)
        self.encoders_ = {}
        self.strata_ = []
        self.meta_ = {}

    def _model(self, target_idx, group_idx, n, n_targets, n_groups, event=None):
        """Internal NumPyro model definition."""
        # 1. Global Intercept (Regularized)
        alpha = samp("alpha", dist.Normal(0, 1.5))

        # 2. Fixed effect: Target deviation from baseline
        b_target = samp("b_target", dist.Normal(0, 1).expand([n_targets]))

        # 3. Random effect: Group variation (Non-centered parameterization)
        sd_group = samp("sd_group", dist.HalfNormal(1))
        z_group = samp("z_group", dist.Normal(0, 1).expand([n_groups]))
        r_group = z_group * sd_group

        # 4. Generalized Logit link
        logit_p = alpha + b_target[target_idx] + r_group[group_idx]

        # 5. Likelihood
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    def fit(self, agg_df: pd.DataFrame) -> 'BayesianPrevalenceEstimator':
        """
        Parses data, fits encoders, and runs inference to get posterior samples.

        Args:
            agg_df: The aggregated DataFrame.

        Returns:
            The fitted estimator instance.
        """
        self.strata_, self.meta_ = self._extract_strata(agg_df, exclude_cols=[self.target_event, self.target_n])

        # FIX: Gracefully handle 1-dimensional stratification by injecting a dummy hierarchy
        if len(self.strata_) == 1:
            agg_df = agg_df.copy()
            agg_df['_dummy_target'] = 'All Targets'
            self.strata_.append('_dummy_target')

        if len(self.strata_) < 2:
            raise ValueError(f"Requires at least two strata levels. Found: {self.strata_}")

        group_col = self.strata_[0]
        target_col = self.strata_[1]

        # Fit and store the encoders exactly once!
        for col in [group_col, target_col]:
            le = LabelEncoder()
            # We fit_transform here, establishing the strict mapping
            agg_df[f'{col}_idx'] = le.fit_transform(agg_df[col].astype(str))
            self.encoders_[col] = le

        jax_data = {
            "target_idx": jnp.array(agg_df[f'{target_col}_idx'].values),
            "group_idx": jnp.array(agg_df[f'{group_col}_idx'].values),
            "n": jnp.array(agg_df[self.target_n].values),
            "event": jnp.array(agg_df[self.target_event].values),
            "n_targets": len(self.encoders_[target_col].classes_),
            "n_groups": len(self.encoders_[group_col].classes_)
        }

        # Run inference
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """
        Uses the fitted samples and encoders to calculate prevalence bounds.

        Args:
            agg_df: The DataFrame to generate predictions for.

        Returns:
            A PrevalenceEstimates object.
        """
        self.check_is_fitted()

        # If a dummy target was added during fit, we need to inject it here too
        predict_df = agg_df.copy()
        if '_dummy_target' in self.strata_ and '_dummy_target' not in predict_df.columns:
            predict_df['_dummy_target'] = 'All Targets'

        group_col = self.strata_[0]
        target_col = self.strata_[1]

        target_idx = jnp.array(self.encoders_[target_col].transform(predict_df[target_col].astype(str)))
        group_idx = jnp.array(self.encoders_[group_col].transform(predict_df[group_col].astype(str)))

        # Matrix math against self.samples_ ...
        alpha_draws = self.samples_["alpha"][:, None]
        r_group_draws = self.samples_["z_group"] * self.samples_["sd_group"][:, None]
        logit_p_draws = alpha_draws + self.samples_["b_target"][:, target_idx] + r_group_draws[:, group_idx]
        p_draws = expit(logit_p_draws)

        estimate = jnp.mean(p_draws, axis=0)
        lower, upper = jnp.quantile(p_draws, jnp.array([0.025, 0.975]), axis=0)

        new_cols = {
            'estimate': np.array(estimate),
            'lower': np.array(lower),
            'upper': np.array(upper),
            'method': self._method_label
        }

        # 2. Fast horizontal concatenation (ignores the deep copy overhead)
        result_df = pd.concat([agg_df, pd.DataFrame(new_cols, index=agg_df.index)], axis=1)

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[s for s in self.strata_ if s != '_dummy_target'],
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            method=self._method_label,
            aggregation_type=self.meta_.get("type", "unknown"),
            target=self.meta_.get("target", "unknown")
        )

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """One-liner to fit and predict on the same data."""
        return self.fit(agg_df).predict(agg_df)


class RegressionPrevalenceEstimator(BaseEstimator[PrevalenceEstimates], ModelledMixin):
    """
    Frequentist binomial GLM for prevalence estimation.

    Uses statsmodels to fit a Generalized Linear Model with a binomial family
    and logit link.

    Examples:
        >>> from seroepi.estimators import RegressionPrevalenceEstimator
        >>> estimator = RegressionPrevalenceEstimator()
        >>> # result = estimator.calculate(agg_df)
    """
    def __init__(self, target_event: str = 'event', target_n: str = 'n'):
        """
        Initializes the RegressionPrevalenceEstimator.

        Args:
            target_event: Column name for event counts. Defaults to 'event'.
            target_n: Column name for total counts. Defaults to 'n'.
        """
        self.target_event = target_event
        self.target_n = target_n
        self._method_label = "binomial_glm"

    def fit(self, agg_df: pd.DataFrame) -> 'RegressionPrevalenceEstimator':
        """Fits the binomial GLM."""
        self.strata_, self.meta_ = self._extract_strata(agg_df, exclude_cols=[self.target_event, self.target_n])

        # 1. Fit the encoder (We still use sklearn here because statsmodels' categorical handling can be clunky)
        self.encoder_ = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_encoded = self.encoder_.fit_transform(agg_df[self.strata_])

        # Add the intercept
        X = sm.add_constant(X_encoded)

        # 2. The Statsmodels Magic: Just pass [Successes, Failures] directly!
        successes = agg_df[self.target_event].values
        failures = agg_df[self.target_n].values - successes
        Y = np.column_stack((successes, failures))

        # 3. Fit the Binomial GLM safely
        # It handles the Fisher Information / Hessian inversion automatically
        glm_model = sm.GLM(Y, X, family=sm.families.Binomial())
        self.fit_results_ = glm_model.fit()

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """Generates predictions and confidence intervals."""
        self.check_is_fitted()

        # Transform new data
        X_encoded = self.encoder_.transform(agg_df[self.strata_])
        X = sm.add_constant(X_encoded, has_constant='add')

        # statsmodels natively handles the delta method and inverse-link transformations
        predictions = self.fit_results_.get_prediction(X).summary_frame(alpha=0.05)

        new_cols = {
            'estimate': predictions['mean'].values,
            'lower': predictions['mean_ci_lower'].values,
            'upper': predictions['mean_ci_upper'].values,
            'method': self._method_label
        }

        result_df = pd.concat([agg_df.copy(), pd.DataFrame(new_cols, index=agg_df.index)], axis=1)

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=self.strata_,
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            method=self._method_label,
            aggregation_type=self.meta_.get("type", "unknown"),
            target=self.meta_.get("target", "unknown")
        )

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """One-liner to fit and predict."""
        return self.fit(agg_df).predict(agg_df)


class SpatialPrevalenceEstimator(BaseEstimator[PrevalenceEstimates], ModelledMixin, BayesianMixin):
    """
    Gaussian Process (GP) based spatial prevalence estimator.

    Fits a GP model to spatial binomial data, allowing for continuous mapping
    of prevalence across a geographic area.

    Examples:
        >>> from seroepi.estimators import SpatialPrevalenceEstimator
        >>> estimator = SpatialPrevalenceEstimator(lat_col='lat', lon_col='lon')
        >>> # result = estimator.calculate(agg_df)
    """
    def __init__(self, lat_col: str = 'lat', lon_col: str = 'lon',
                 method: InferenceMethod = InferenceMethod.MCMC, num_samples: int = 1500, num_chains: int = 4,
                 num_warmup: int = 1000, svi_steps: int = 3000,
                 target_event: str = 'event', target_n: str = 'n', seed: int = 42):
        """
        Initializes the SpatialPrevalenceEstimator.

        Args:
            lat_col: Column name for latitude. Defaults to 'lat'.
            lon_col: Column name for longitude. Defaults to 'lon'.
            method: Inference method. Defaults to 'mcmc'.
            num_samples: Number of samples. Defaults to 1500.
            num_chains: Number of chains. Defaults to 4.
            num_warmup: Number of warmup steps. Defaults to 1000.
            svi_steps: Number of SVI steps. Defaults to 3000.
            target_event: Column for events. Defaults to 'event'.
            target_n: Column for totals. Defaults to 'n'.
            seed: Random seed. Defaults to 42.
        """
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self.lat_col = lat_col
        self.lon_col = lon_col
        self._method_label = f'spatial_gp_{self.method.value}'
        
        self.target_event = target_event
        self.target_n = target_n

        # Fitted attributes
        self.X_train_ = None
        self.loc_mean_ = None
        self.loc_scale_ = None
        self.meta_ = {}

    def _model(self, X, n, event=None):
        """Internal NumPyro GP model."""
        # 1. Global Intercept
        alpha = samp("alpha", dist.Normal(0, 1.5))

        # 2. GP Kernel Parameters (Variance/Amplitude and Length-scale)
        var = samp("var", dist.HalfNormal(1.0))
        length = samp("length", dist.InverseGamma(2.0, 1.0))

        # 3. Spatial Covariance Matrix
        K = _rbf_kernel(X, X, var, length)

        # 4. Latent Spatial Field (f)
        f = samp("f", dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=K))

        # 5. Likelihood
        logit_p = alpha + f
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    def fit(self, agg_df: pd.DataFrame) -> 'SpatialPrevalenceEstimator':
        """Aggregates to unique locations, normalizes, and fits the GP."""
        if self.lat_col not in agg_df.columns or self.lon_col not in agg_df.columns:
            raise KeyError(
                f"Spatial estimator requires '{self.lat_col}' and '{self.lon_col}' "
                "in the aggregated data. Please ensure you included them in the 'Stratify By' dropdown."
            )

        # 1. Ensure we only have ONE row per unique lat/lon to prevent singular matrices
        spatial_df = agg_df.groupby([self.lat_col, self.lon_col], as_index=False).agg({
            self.target_event: 'sum',
            self.target_n: 'sum'
        })

        # 2. Extract and Standardize Coordinates
        raw_coords = spatial_df[[self.lat_col, self.lon_col]].values
        self.loc_mean_ = np.mean(raw_coords, axis=0)
        self.loc_scale_ = np.std(raw_coords, axis=0) + 1e-8  # Prevent div by zero

        self.X_train_ = (raw_coords - self.loc_mean_) / self.loc_scale_

        jax_data = {
            "X": jnp.array(self.X_train_),
            "n": jnp.array(spatial_df[self.target_n].values),
            "event": jnp.array(spatial_df[self.target_event].values)
        }

        # 3. Run Inference
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)

        self.meta_ = agg_df.attrs.get("metric_meta", {})
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> PrevalenceEstimates:
        """
        Calculates the conditional predictive posterior for any set of coordinates.

        Args:
            df: DataFrame containing latitude and longitude columns.

        Returns:
            A PrevalenceEstimates object with predicted values at the locations.
        """
        self.check_is_fitted()

        # 1. Standardize new coordinates using the FITTED scaler
        raw_X_test = df[[self.lat_col, self.lon_col]].values
        X_test = jnp.array((raw_X_test - self.loc_mean_) / self.loc_scale_)
        X_train = jnp.array(self.X_train_)

        # 2. Use JAX vmap to instantly vectorize this complex math across all 1500 samples!
        logit_p_draws = _vectorized_gp_predict(
            self.samples_['var'],
            self.samples_['length'],
            self.samples_['alpha'],
            self.samples_['f'],
            X_train,
            X_test
        )

        p_draws = expit(logit_p_draws)

        # 4. Extract Estimates
        estimate = jnp.mean(p_draws, axis=0)
        lower, upper = jnp.quantile(p_draws, jnp.array([0.025, 0.975]), axis=0)

        result_df = df.copy()
        result_df['estimate'] = np.array(estimate)
        result_df['lower'] = np.array(lower)
        result_df['upper'] = np.array(upper)
        result_df['method'] = self._method_label

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[self.lat_col, self.lon_col],
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            method=self._method_label,
            aggregation_type=self.meta_.get("type", "unknown"),
            target=self.meta_.get("target", "unknown")
        )

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """One-liner to fit and predict."""
        return self.fit(agg_df).predict(agg_df)


class RegressionIncidenceEstimator(BaseEstimator[IncidenceEstimates], ModelledMixin):
    """
    Negative Binomial GLM for time-series incidence estimation.

    Fits a Negative Binomial model to count data over time, optionally
    adjusting for sequencing volume (relative incidence).

    Examples:
        >>> from seroepi.estimators import RegressionIncidenceEstimator
        >>> estimator = RegressionIncidenceEstimator(use_relative_incidence=True)
        >>> # result = estimator.calculate(inc_df)
    """
    def __init__(self, use_relative_incidence: bool = True):
        """
        Initializes the RegressionIncidenceEstimator.

        Args:
            use_relative_incidence: If True, models cases adjusting for total
                sequencing volume (offset). If False, models absolute counts.
        """
        self.use_relative_incidence = use_relative_incidence
        self._method_label = "neg_binomial_glm"

        # Internal state tracking
        self.fit_results_ = {}
        self.strata_ = []
        self.meta_ = {}

    def fit(self, inc_df: pd.DataFrame) -> 'RegressionIncidenceEstimator':
        """Fits the Negative Binomial GLM to each stratum."""
        self.meta_ = inc_df.attrs.get("metric_meta", {})
        target_col = self.meta_.get("target")
        self.freq_ = self.meta_.get("freq")
        self.strata_ = self.meta_.get("stratified_by", [])

        if not target_col or not self.freq_:
            raise ValueError("Incidence metadata missing. Ensure data came from `epi.aggregate_incidence`.")

        # Sort entirely upstream to avoid O(N log N) operations inside the loop
        inc_df_sorted = inc_df.sort_values('date')
        groups = inc_df_sorted.groupby(self.strata_, observed=True) if self.strata_ else [('Global', inc_df_sorted)]

        for name, group in groups:
            # Drop the .sort_values() here. Just copy.
            df_group = group.copy()

            # Filter out periods with ZERO sequencing volume for the GLM fit
            df_model = df_group[df_group['total_sequenced'] > 0].copy()

            if len(df_model) < 3:
                self.fit_results_[name] = None  # Not enough data to model
                continue

            # Create a numeric Time Step for the slope
            time_deltas = df_model['date'] - df_model['date'].min()

            if self.freq_.startswith('M'):
                df_model['time_step'] = np.round(time_deltas / np.timedelta64(1, 'M'))
            elif self.freq_.startswith('W'):
                df_model['time_step'] = np.round(time_deltas / np.timedelta64(1, 'W'))
            elif self.freq_.startswith('Y'):
                df_model['time_step'] = np.round(time_deltas / np.timedelta64(1, 'Y'))
            else:
                df_model['time_step'] = np.round(time_deltas / np.timedelta64(1, 'D'))

            Y = df_model['variant_count']
            X = sm.add_constant(df_model['time_step'])

            offset = np.log(df_model['total_sequenced']) if self.use_relative_incidence else None

            try:
                # alpha=1.0 is a robust starting guess for overdispersion
                model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
                self.fit_results_[name] = model.fit()
            except Exception as e:
                warn(f"GLM failed to converge for stratum {name}: {e}")
                self.fit_results_[name] = None

        self.is_fitted_ = True
        return self

    def predict(self, inc_df: pd.DataFrame) -> IncidenceEstimates:
        """Extracts Incidence Rate Ratios (IRR) and intervals."""
        self.check_is_fitted()

        groups = inc_df.groupby(self.strata_, observed=True) if self.strata_ else [('Global', inc_df)]
        results = []

        for name, _ in groups:
            fit = self.fit_results_.get(name)

            row = {k: v for k, v in zip(self.strata_, name)} if self.strata_ and isinstance(name, tuple) else {}
            if self.strata_ and not isinstance(name, tuple):
                row = {self.strata_[0]: name}

            if fit is None:
                row.update({
                    'IRR': np.nan, 'IRR_lower': np.nan, 'IRR_upper': np.nan,
                    'p_value': np.nan, 'status': 'Failed/Insufficient Data'
                })
            else:
                coef = fit.params.get('time_step', 0)
                p_val = fit.pvalues.get('time_step', 1.0)
                ci_lower = fit.conf_int()[0].get('time_step', 0)
                ci_upper = fit.conf_int()[1].get('time_step', 0)

                row.update({
                    'IRR': np.exp(coef),
                    'IRR_lower': np.exp(ci_lower),
                    'IRR_upper': np.exp(ci_upper),
                    'p_value': p_val,
                    'status': 'Converged'
                })
            results.append(row)

        return IncidenceEstimates(
            data=inc_df.copy(),
            stratified_by=self.strata_,
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            target=self.meta_.get("target", "unknown"),
            freq=self.freq_,
            aggregation_type=self.meta_.get("type", "unknown"),
            model_results=pd.DataFrame(results)
        )

    def calculate(self, inc_df: pd.DataFrame) -> IncidenceEstimates:
        """One-liner to fit and predict."""
        return self.fit(inc_df).predict(inc_df)


# Kernels --------------------------------------------------------------------------------------------------------------
def _rbf_kernel(X, Z, var, length, jitter=1e-5):
    """Calculates the Exponentiated Quadratic (RBF) Kernel matrix."""
    # Compute squared distance between all pairs of points in X and Z
    dist_sq = jnp.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    K = var * jnp.exp(-0.5 * dist_sq / (length ** 2))

    # Add jitter to the diagonal for numerical stability (only if computing symmetric K_XX)
    if X.shape == Z.shape:
        K += jitter * jnp.eye(X.shape[0])
    return K


@functools.partial(jit)
@functools.partial(vmap, in_axes=(0, 0, 0, 0, None, None))
def _vectorized_gp_predict(var, length, alpha, f, X_train, X_test):
    """Pure function for exact GP conditional predictions. JIT compiled once."""
    K_XX = _rbf_kernel(X_train, X_train, var, length, jitter=1e-5)
    K_Xstar_X = _rbf_kernel(X_test, X_train, var, length, jitter=0.0)
    v = jsp.linalg.solve(K_XX, f, assume_a='pos')
    f_star = K_Xstar_X @ v
    return alpha + f_star