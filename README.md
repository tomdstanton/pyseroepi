# 🦠💉🌍 seroepi

`seroepi` is a comprehensive Python toolkit and interactive Shiny dashboard for the epidemiological, geospatial, and genotypic analysis of pathogen isolates.

Built seamlessly on top of `pandas`, it is specifically designed to ingest output from genomic pipelines (like **Kleborate** and **Pathogenwatch**), calculate statistical burdens, map spatial transmissions, and computationally design optimal vaccine formulations.

---

## ✨ What You Can Do

- **Robust Data Stewardship**: Validate inputs seamlessly against our `UnifiedIsolateSchema` (powered by Pandera) to ensure completely standardized genotypic, spatial, and temporal datasets.
- **Interactive Dashboarding**: Launch a beautifully designed, dark-mode Shiny app to explore your data, train models, and generate publication-ready plots instantly.
- **Smart Pandas Accessors**: Clean coordinates, query AMR genes, and generate epidemic curves natively using `df.geo`, `df.geno`, `df.epi`, and `df.qc`.
- **Robust Statistical Modeling**: Estimate global and regional prevalence using Frequentist, Bayesian MCMC/SVI, and Gaussian Process (GP) Spatial models.
- **Vaccine Formulation Engine**: Use rigorous Leave-One-Out (LOO) cross-validation to algorithmically identify the most stable and high-coverage target antigens (e.g., K-loci) for vaccine design.
- **Outbreak & Transmission Clustering**: Instantly generate transmission networks and spatial cliques using SNP distance matrices and spatiotemporal thresholds.

---

## 📦 Installation

You can install `seroepi` directly from PyPI. We highly recommend using uv for lightning-fast installations, but standard `pip` works perfectly too.

```bash
# Using uv (Recommended)
uv pip install seroepi

# Using standard pip
pip install seroepi
```

### Optional Dependencies
To unlock the advanced Bayesian models, Gaussian Processes, and Plotly visualizations, install the optional dependencies:

```bash
uv pip install seroepi[models,plot]
```

---

## 🚀 Running the Interactive Dashboard

`seroepi` comes with a world-class, fully featured Shiny web dashboard. You can upload data, configure models, chat
with a built-in AI assistant, and download your results without writing a single line of code.

To launch the app locally from your command line:

```bash
shiny run seroepi.app
```

---

## 💻 Python API Quickstart

If you prefer working in Jupyter Notebooks or Python scripts, `seroepi` extends standard Pandas DataFrames to make bioinformatics workflows effortless.

### 1. Data Ingestion, Validation & Spatial Cleaning
```python
import pandas as pd
from seroepi.io import PathogenwatchKleborateParser
import seroepi.accessors  # Magically registers .epi, .geo, .geno, .qc

# Load your Kleborate output and optional Metadata
geno_df = pd.read_csv("kleborate_results.csv")
meta_df = pd.read_csv("metadata.csv")

# Parse, merge, and strictly validate against the UnifiedIsolateSchema
df = PathogenwatchKleborateParser.parse(
    geno_df, 
    meta_df=meta_df, 
    meta_kwargs={"id_col": "sample_id", "country_col": "country"}
)

# Automatically impute missing coordinates based on Country names!
df = df.geo.standardize_and_impute()
```

### 2. Genomic Clustering
```python
from seroepi.dist import Distances

# Load a pairwise SNP matrix (e.g., from Pathogenwatch)
dist = Distances.from_pathogenwatch("distances.csv")

# Find isolates separated by ≤ 20 SNPs
clusters = dist.connected_components(threshold=20)

# Merge straight back into your dataframe
df = df.join(clusters, on='sample_id')
```

### 3. Prevalence Estimation & Plotting
```python
from seroepi.estimators import FrequentistPrevalenceEstimator

# Aggregate the data to find the prevalence of K-loci across different countries
agg_df = df.epi.aggregate_prevalence(stratify_by=['country'], target_col='K_locus')

# Fit the estimator
estimator = FrequentistPrevalenceEstimator(method='wilson')
results = estimator.calculate(agg_df)

# Generate a publication-ready Plotly Forest Plot
fig = results.plot('forest')
fig.show()
```

### 4. Algorithmic Vaccine Formulation
```python
from seroepi.formulation import CVFormulationDesigner

# Design a 6-valent vaccine, using 'country' as the cross-validation holdout
designer = CVFormulationDesigner(valency=6, n_jobs=-1)
designer.fit(estimator, agg_df, loo_col='country')

# View the most stable optimal targets
optimal_vaccine = designer.formulation_
print(optimal_vaccine.get_formulation())
```

---

## 📚 Documentation

For a complete deep-dive into the available methods, classes, and architectural concepts, please refer to the fully documented **API Reference** (Automatically generated).
