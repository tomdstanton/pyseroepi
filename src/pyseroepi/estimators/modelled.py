from typing import Literal, Union, TypeVar, Type
from abc import ABC, abstractmethod
from pathlib import Path
from multiprocessing import cpu_count

import pandas as pd
import numpy as np

from scipy.special import expit
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression

import jax.numpy as jnp
import jax.scipy as jsp
from jax import random, vmap

from numpyro import optim, distributions as dist, diagnostics as diag, sample as samp, set_host_device_count
from numpyro.infer import MCMC, NUTS, Trace_ELBO, SVI, autoguide, Predictive

from pyseroepi.estimators import PrevalenceEstimates, BaseEstimator


# Set-up ---------------------------------------------------------------------------------------------------------------
# Tell JAX to split the CPU into multiple virtual devices for parallel chains
set_host_device_count(min(cpu_count(), 4))


# TypeVars -------------------------------------------------------------------------------------------------------------
T_Modelled = TypeVar('T_Modelled', bound='ModelledMixin')


# ABCs -----------------------------------------------------------------------------------------------------------------
class ModelledMixin(ABC):
    """
    Contract for estimators with an internal fitted state.
    Enforces the scikit-learn fit/predict paradigm.
    """

    # State tracking
    is_fitted_: bool = False

    def check_is_fitted(self):
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

    @abstractmethod
    def save_model(self, filepath: Union[str, Path]) -> None:
        pass

    @classmethod
    @abstractmethod
    def load_model(cls: Type[T_Modelled], filepath: Union[str, Path]) -> T_Modelled:
        pass


# Estimators -----------------------------------------------------------------------------------------------------------
class BayesianPrevalenceEstimator(BaseEstimator[PrevalenceEstimates], ModelledMixin):
    Method = Literal['mcmc', 'svi']

    def __init__(self, method: Method = 'mcmc', num_samples: int = 1500, num_chains: int = 4,
                 num_warmup: int = 1000, svi_steps: int = 3000, target_event='event', target_n='n', seed: int = 42):
        self.method = method
        self._method_label = f'bayesian_{self.method}'
        self._method_func = self._METHODS.get(self.method, None)
        if self._method_func is None:
            raise ValueError(f"Unknown method: {self.method}. "
                             f"Choose from: {list(self._METHODS.keys())}")

        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.svi_steps = svi_steps
        self.num_chains = num_chains
        self.target_event = target_event
        self.target_n = target_n
        self.seed = seed

        # Fitted attributes (trailing underscores)
        self.samples_ = None
        self.encoders_ = {}
        self.strata_ = []
        self.meta_ = {}

    def _model(self, target_idx, group_idx, n, n_targets, n_groups, event=None):
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

    def _mcmc_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        mcmc = MCMC(NUTS(self._model), num_warmup=self.num_warmup, num_samples=self.num_samples, num_chains=self.num_chains)
        mcmc.run(rng_key, **jax_data)
        return mcmc.get_samples()

    def _svi_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        opt_key, pred_key = random.split(rng_key)
        guide = autoguide.AutoNormal(self._model)
        optimizer = optim.Adam(step_size=0.01)
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(opt_key, num_steps=self.svi_steps, **jax_data)
        predictive = Predictive(self._model, guide=guide, params=svi_result.params, num_samples=self.num_samples)
        return predictive(pred_key, **jax_data)

    def fit(self, agg_df: pd.DataFrame) -> 'BayesianPrevalenceEstimator':
        """Step 1: Parse data, fit encoders, and run inference to get samples."""
        self.strata_, self.meta_ = self._extract_strata(agg_df, exclude_cols=[self.target_event, self.target_n])

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

        # Run inference using getattr trick
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._method_func(self, jax_data, rng_key)

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """Step 2: Use the fitted samples and encoders to calculate bounds."""
        self.check_is_fitted()

        group_col = self.strata_[0]
        target_col = self.strata_[1]

        # Use the PRE-FITTED encoders to safely transform new or existing data
        # (If the user passes unseen data, .transform() will naturally throw a warning)
        target_idx = jnp.array(self.encoders_[target_col].transform(agg_df[target_col].astype(str)))
        group_idx = jnp.array(self.encoders_[group_col].transform(agg_df[group_col].astype(str)))

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
            stratified_by=self.strata_,
            method=self._method_label,
            prevalence_type=self.meta_.get("type", "unknown"),
            target=self.meta_.get("target", "unknown")
        )

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """The frictionless one-liner to fulfill the BaseEstimator contract."""
        return self.fit(agg_df).predict(agg_df)

    def diagnostics(self):
        self.check_is_fitted()
        if self.method != 'mcmc':
            print("Diagnostics are only available for MCMC inference.")
            return

        # NumPyro's built-in summary prints R-hat and ESS directly to the console
        print(diag.print_summary(self.samples_, prob=0.95, group_by_chain=False))

    def save_model(self, filepath: Union[str, Path]) -> None:
        self.check_is_fitted()
        config = {
            'method': self.method, 'num_samples': self.num_samples, 'num_chains': self.num_chains,
            'num_warmup': self.num_warmup, 'svi_steps': self.svi_steps, 'target_event': self.target_event,
            'target_n': self.target_n, 'seed': self.seed
        }

        # We package the config, the raw dictionary of LabelEncoders, strata, and samples
        np.savez(filepath, config=config, encoders=self.encoders_,
                 strata=self.strata_, meta=self.meta_, samples=self.samples_)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'BayesianPrevalenceEstimator':
        data = np.load(filepath, allow_pickle=True)

        # Instantiate fresh with old hyperparams
        estimator = cls(**data['config'].item())

        # Inject fitted state
        estimator.samples_ = data['samples'].item()
        estimator.encoders_ = data['encoders'].item()
        estimator.strata_ = data['strata'].tolist()
        estimator.meta_ = data['meta'].item()
        estimator.is_fitted_ = True

        return estimator

    _METHODS = {'mcmc': _mcmc_inference, 'svi': _svi_inference}


class RegressionPrevalenceEstimator(BaseEstimator[PrevalenceEstimates], ModelledMixin):
    def __init__(self, target_event='event', target_n='n'):
        self.target_event = target_event
        self.target_n = target_n
        self._method_label = 'regression_logistic'

        # Fitted attributes
        self.model_ = None
        self.encoder_ = None
        self.cov_matrix_ = None
        self.strata_ = []
        self.meta_ = {}

    def fit(self, agg_df: pd.DataFrame) -> 'RegressionPrevalenceEstimator':
        self.strata_, self.meta_ = self._extract_strata(agg_df, exclude_cols=[self.target_event, self.target_n])

        if not self.strata_:
            raise ValueError("Regression estimator requires at least one stratification level.")

        # 1. Fit the encoder ONCE on the raw, unduplicated data
        self.encoder_ = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_base = self.encoder_.fit_transform(agg_df[self.strata_])

        # 2. Duplicate the numerical matrix instantly using numpy
        X_train = np.vstack([X_base, X_base])

        # 3. Create the Targets (1s then 0s) and Weights directly from the arrays
        n_rows = len(agg_df)
        y_train = np.concatenate([np.ones(n_rows, dtype=int), np.zeros(n_rows, dtype=int)])

        success_weights = agg_df[self.target_event].values
        fail_weights = (agg_df[self.target_n] - agg_df[self.target_event]).values
        weights = np.concatenate([success_weights, fail_weights])

        # 4. Filter out any 0-weight rows to save regression compute time
        valid_mask = weights > 0
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
        weights = weights[valid_mask]

        # 5. Fit standard Logistic Regression
        self.model_ = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        self.model_.fit(X_train, y_train, sample_weight=weights)

        # 6. Calculate the Covariance Matrix for Confidence Intervals
        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        p = self.model_.predict_proba(X_train)[:, 1]

        W = p * (1 - p) * weights
        fisher_info = X_design.T @ (W[:, None] * X_design)
        self.cov_matrix_ = np.linalg.pinv(fisher_info)

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        self.check_is_fitted()

        # 1. Encode the incoming data
        X_test = self.encoder_.transform(agg_df[self.strata_])
        X_design_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

        # 2. Get Point Estimates (Adjusted Prevalence)
        estimates = self.model_.predict_proba(X_test)[:, 1]

        # 3. Calculate Confidence Intervals using the Covariance Matrix
        # Variance on the logit (link) scale
        var_link = np.sum((X_design_test @ self.cov_matrix_) * X_design_test, axis=1)
        se_link = np.sqrt(var_link)

        # Get the linear predictor (logit) values
        linear_predictor = self.model_.decision_function(X_test)

        # Calculate 95% Wald CI on the link scale, then expit back to probability scale
        z = stats.norm.ppf(0.975)
        lower = expit(linear_predictor - z * se_link)
        upper = expit(linear_predictor + z * se_link)

        new_cols = {
            'estimate': np.array(estimates),
            'lower': np.array(lower),
            'upper': np.array(upper),
            'method': self._method_label
        }

        # 2. Fast horizontal concatenation (ignores the deep copy overhead)
        result_df = pd.concat([agg_df, pd.DataFrame(new_cols, index=agg_df.index)], axis=1)

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=self.strata_,
            method=self._method_label,
            prevalence_type=self.meta_.get("type", "unknown"),
            target=self.meta_.get("target", "unknown")
        )

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        return self.fit(agg_df).predict(agg_df)

    def save_model(self, filepath: Union[str, Path]) -> None:
        self.check_is_fitted()
        config = {
            'target_event': self.target_event,
            'target_n': self.target_n
        }

        # np.savez securely packages everything, including the sklearn model (via pickle)
        np.savez(filepath, config=config,
                 model=self.model_, encoder=self.encoder_,
                 cov_matrix=self.cov_matrix_, strata=self.strata_, meta=self.meta_)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'RegressionPrevalenceEstimator':
        data = np.load(filepath, allow_pickle=True)

        estimator = cls(**data['config'].item())

        # Inject fitted state
        estimator.model_ = data['model'].item()
        estimator.encoder_ = data['encoder'].item()
        estimator.cov_matrix_ = data['cov_matrix']
        estimator.strata_ = data['strata'].tolist()
        estimator.meta_ = data['meta'].item()
        estimator.is_fitted_ = True

        return estimator


class SpatialPrevalenceEstimator(BaseEstimator[PrevalenceEstimates], ModelledMixin):
    Method = Literal['mcmc', 'svi']

    def __init__(self, lat_col: str = 'lat', lon_col: str = 'lon',
                 method: Method = 'mcmc', num_samples: int = 1500, num_chains: int = 4,
                 num_warmup: int = 1000, svi_steps: int = 3000,
                 target_event='event', target_n='n', seed: int = 42):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.method = method
        self._method_label = f'spatial_gp_{self.method}'

        self.num_samples = num_samples
        self.num_warmup = num_warmup
        self.svi_steps = svi_steps
        self.num_chains = num_chains
        self.target_event = target_event
        self.target_n = target_n
        self.seed = seed

        # Fitted attributes
        self.samples_ = None
        self.X_train_ = None
        self.loc_mean_ = None
        self.loc_scale_ = None
        self.meta_ = {}

    def _model(self, X, n, event=None):
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

        # 3. Run Inference (Dynamic routing via getattr)
        rng_key = random.PRNGKey(self.seed)
        inference_func = getattr(self, f"_{self.method}_inference")
        self.samples_ = inference_func(jax_data, rng_key)

        self.meta_ = agg_df.attrs.get("prevalence_meta", {})
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> PrevalenceEstimates:
        """
        Calculates the conditional predictive posterior for ANY set of coordinates.
        This allows you to pass a meshgrid of points to draw a continuous map!
        """
        self.check_is_fitted()

        # 1. Standardize new coordinates using the FITTED scaler
        raw_X_test = df[[self.lat_col, self.lon_col]].values
        X_test = jnp.array((raw_X_test - self.loc_mean_) / self.loc_scale_)
        X_train = jnp.array(self.X_train_)

        # 2. Define the exact GP conditional prediction for a single MCMC sample
        def predict_single_sample(var, length, alpha, f):
            # K_XX: Covariance of training data
            K_XX = _rbf_kernel(X_train, X_train, var, length, jitter=1e-5)
            # K_Xstar_X: Covariance between test data and training data
            K_Xstar_X = _rbf_kernel(X_test, X_train, var, length, jitter=0.0)

            # Mathematical exact conditional mean: f_* = K_{*X} * (K_{XX})^-1 * f
            # Solved efficiently without full matrix inversion
            v = jsp.linalg.solve(K_XX, f, assume_a='pos')
            f_star = K_Xstar_X @ v

            return alpha + f_star

        # 3. Use JAX vmap to instantly vectorize this complex math across all 1500 samples!
        vectorized_predict = vmap(predict_single_sample)

        logit_p_draws = vectorized_predict(
            self.samples_['var'],
            self.samples_['length'],
            self.samples_['alpha'],
            self.samples_['f']
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
            method=self._method_label,
            prevalence_type=self.meta_.get("type", "unknown"),
            target=self.meta_.get("target", "unknown")
        )

    # --- Standard Inference Wrappers (Inherited/Shared Logic) ---
    def _mcmc_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        mcmc = MCMC(NUTS(self._model), num_warmup=self.num_warmup, num_samples=self.num_samples,
                    num_chains=self.num_chains)
        mcmc.run(rng_key, **jax_data)
        return mcmc.get_samples()

    def _svi_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        opt_key, pred_key = random.split(rng_key)
        guide = autoguide.AutoNormal(self._model)
        optimizer = optim.Adam(step_size=0.01)
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(opt_key, num_steps=self.svi_steps, **jax_data)
        predictive = Predictive(self._model, guide=guide, params=svi_result.params, num_samples=self.num_samples)
        return predictive(pred_key, **jax_data)

    def calculate(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        return self.fit(agg_df).predict(agg_df)

    def save_model(self, filepath: Union[str, Path]) -> None:
        self.check_is_fitted()
        config = {
            'lat_col': self.lat_col, 'lon_col': self.lon_col, 'method': self.method,
            'num_samples': self.num_samples, 'num_chains': self.num_chains,
            'num_warmup': self.num_warmup, 'svi_steps': self.svi_steps,
            'target_event': self.target_event, 'target_n': self.target_n, 'seed': self.seed
        }
        np.savez(filepath, config=config, X_train=self.X_train_,
                 loc_mean=self.loc_mean_, loc_scale=self.loc_scale_,
                 meta=self.meta_, samples=self.samples_)

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'SpatialPrevalenceEstimator':
        data = np.load(filepath, allow_pickle=True)
        estimator = cls(**data['config'].item())
        estimator.X_train_ = data['X_train']
        estimator.loc_mean_ = data['loc_mean']
        estimator.loc_scale_ = data['loc_scale']
        estimator.samples_ = data['samples'].item()
        estimator.meta_ = data['meta'].item()
        estimator.is_fitted_ = True
        return estimator


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
