from typing import Union
import warnings
import pandas as pd
from pyseroepi.data.gazetteer_data import GAZETTEER_DICT
# Suppress the harmless Pandas accessor overwrite warnings during interactive development
warnings.filterwarnings("ignore", message="registration of accessor")


# Accessors ------------------------------------------------------------------------------------------------------------
@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._gazetteer = pd.DataFrame.from_dict(GAZETTEER_DICT, orient='index')

    @property
    def gazetteer(self):
        return self._gazetteer

    def standardize_and_impute(self) -> pd.DataFrame:
        df = self._obj.copy()
        ref_data = self.gazetteer

        # 1. Initialize tracking
        if 'spatial_resolution' not in df.columns:
            df['spatial_resolution'] = 'unknown'

        exact_mask = df['latitude'].notna() & df['longitude'].notna()
        df.loc[exact_mask, 'spatial_resolution'] = 'exact'

        # 2. Impute using the instant dictionary-backed dataframe
        if 'country' in df.columns:
            clean_countries = df['country'].str.strip()  # Natural Earth doesn't use title case
            needs_imputation = (~exact_mask) & clean_countries.notna()

            # The .map() function looks up the clean_countries strings against
            # the index of ref_data in a fraction of a millisecond
            df.loc[needs_imputation, 'latitude'] = clean_countries.map(ref_data['centroid_lat'])
            df.loc[needs_imputation, 'longitude'] = clean_countries.map(ref_data['centroid_lon'])

            imputed_mask = needs_imputation & df['latitude'].notna()
            df.loc[imputed_mask, 'spatial_resolution'] = 'country'

            if 'iso3' not in df.columns or df['iso3'].isna().all():
                df['iso3'] = clean_countries.map(ref_data['iso3'])
            if 'region' not in df.columns or df['region'].isna().all():
                df['region'] = clean_countries.map(ref_data['region'])

        return df

    def reverse_geocode(self, shapefile_path: str) -> pd.DataFrame:
        """
        Optional Geopandas integration.
        Takes exact Lat/Lon points and determines which Country/Region polygon they fall into.
        """
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for reverse geocoding. Install with pyseroepi[spatial]")

        df = self._obj.copy()
        exact_mask = df['latitude'].notna() & df['longitude'].notna()

        # Convert our standard Pandas dataframe to a GeoDataFrame (Points)
        gdf_points = gpd.GeoDataFrame(
            df[exact_mask],
            geometry=gpd.points_from_xy(df.loc[exact_mask, 'longitude'], df.loc[exact_mask, 'latitude']),
            crs="EPSG:4326"  # Standard GPS coordinates
        )

        # Load the boundary polygons (e.g., WHO regions or Country borders)
        gdf_polygons = gpd.read_file(shapefile_path)

        # The Spatial Join (Finds which polygon each point intersects)
        joined = gpd.sjoin(gdf_points, gdf_polygons, how="left", predicate="intersects")

        # Map the found regions back to the main dataframe
        df.loc[exact_mask, 'country'] = joined['country_name_from_shapefile']

        return df


@pd.api.extensions.register_dataframe_accessor("epi")
class EpiAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    # --- Shiny UI State Checkers ---

    @property
    def has_temporal(self) -> bool:
        """Perfect for Shiny conditional panels. Schema guarantees it is called 'date'."""
        # Checks if 'date' exists AND actually has non-null data
        return 'date' in self._obj.columns and self._obj['date'].notna().any()

    @property
    def has_spatial(self) -> bool:
        """Perfect for Shiny conditional panels."""
        return 'latitude' in self._obj.columns and 'longitude' in self._obj.columns

    # --- Spatiotemporal Helpers ---

    @property
    def temporal(self) -> pd.Series:
        """Returns the primary isolation date series."""
        if not self.has_temporal:
            raise ValueError("Temporal column 'date' is missing or completely empty.")
        return self._obj['date']

    @property
    def spatial(self) -> pd.DataFrame:
        """Returns the core spatial coordinates."""
        if not self.has_spatial:
            raise ValueError("Spatial columns ('latitude', 'longitude') are missing.")
        return self._obj[['latitude', 'longitude']]

    # --- Time Series / Epidemic Curve Methods ---

    def epidemic_curve(self, freq: str = 'W', stratify_by: str = None) -> pd.DataFrame:
        """
        Generates a clean timeseries dataframe for plotting epidemic curves.
        freq: 'D' (Day), 'W' (Week), 'M' (Month), 'Y' (Year)
        """
        if not self.has_temporal:
            raise ValueError("Cannot generate epi curve: No temporal data available.")

        df = self._obj.copy()

        # We know exactly what the column is called now!
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        # Set time as index for resampling
        df = df.set_index('date')

        if stratify_by:
            curve = df.groupby(stratify_by).resample(freq).size().unstack(level=0, fill_value=0)
        else:
            curve = df.resample(freq).size().to_frame(name='count')

        return curve.reset_index()

    @property
    def metadata_columns(self) -> list[str]:
        """
        Dynamically sweeps the DataFrame for any user-uploaded clinical data.
        Returns the RAW column names (e.g., 'meta_Patient_Age') for backend math.
        """
        return self._obj.filter(regex="^meta_").columns.tolist()

    @property
    def ui_metadata_columns(self) -> list[str]:
        """
        Returns CLEAN metadata names (e.g., 'Patient_Age') for populating
        Shiny UI dropdown menus.
        """
        return [c.replace('meta_', '', 1) for c in self.metadata_columns]

    @property
    def genotypes(self) -> list[str]:
        """
        Dynamically compiles the master list of all genetic variables in the dataset.
        Includes Core Genotypes (ST, K_locus) + Accessory Traits (amr_, vir_).
        """
        # 1. Grab the core grouping genotypes if they exist
        core_genos = [col for col in ['ST', 'K_locus', 'O_locus'] if col in self._obj.columns]

        # 2. Grab all the dynamically prefixed traits
        trait_genos = self._obj.filter(regex="^(amr|vir)_").columns.tolist()

        return core_genos + trait_genos

    def aggregate_prevalence(self, stratify_by: list[str], target_col: str = None,
                             adjust_for_clone: str = None,
                             negative_indicator: Union[str, list[str]] = '-') -> pd.DataFrame:
        """
        Aggregates 1-row-per-sample data into Binomial counts (event / n).

        - If target_col is None: Calculates compositional prevalence of the LAST
          item in stratify_by relative to the preceding items.
          (e.g., stratify_by=['Region', 'Serotype'] -> prevalence of Serotype per Region)

        - If target_col is provided: Calculates trait prevalence of target_col
          within the exact strata. (e.g., target_col='blaKPC', stratify_by=['Region', 'Serotype'])
        """
        df = self._obj.copy()

        # --- PARADIGM 1: Trait Prevalence (e.g., AMR gene from Kleborate) ---
        if target_col:
            denom_cols = stratify_by

            # 1. Drop True Unknowns: If the column is pd.NA, it was never evaluated.
            # We remove it from 'valid_df' so it doesn't artificially inflate the denominator (n).
            valid_df = df.dropna(subset=[target_col]).copy()

            # 2. Smart Boolean Mapping with Dynamic Indicators
            if pd.api.types.is_bool_dtype(valid_df[target_col]):
                valid_df['_target_bool'] = valid_df[target_col]
            else:
                # Standardize the input into a list for pandas .isin()
                if isinstance(negative_indicator, str):
                    neg_list = [negative_indicator]
                else:
                    neg_list = negative_indicator

                # True IF the value is NOT in the list of known negative strings
                valid_df['_target_bool'] = ~valid_df[target_col].isin(neg_list)

            # 3. Calculate Denoms and Events using the valid, evaluated genomes
            if adjust_for_clone:
                denoms = valid_df.groupby(denom_cols)[adjust_for_clone].nunique().rename('n')
                events = valid_df[valid_df['_target_bool']].groupby(denom_cols)[adjust_for_clone].nunique().rename(
                    'event')
            else:
                denoms = valid_df.groupby(denom_cols).size().rename('n')
                events = valid_df.groupby(denom_cols)['_target_bool'].sum().rename('event')

        # --- PARADIGM 2: Compositional Prevalence (e.g., Serotype from Kaptive) ---
        else:
            if len(stratify_by) < 2:
                raise ValueError(
                    "Compositional prevalence requires at least 2 stratify_by columns (e.g., ['Region', 'Serotype']).")

            # Denominator is everything EXCEPT the last stratification level
            denom_cols = stratify_by[:-1]

            if adjust_for_clone:
                denoms = df.groupby(denom_cols)[adjust_for_clone].nunique().rename('n')
                events = df.groupby(stratify_by)[adjust_for_clone].nunique().rename('event')
            else:
                denoms = df.groupby(denom_cols).size().rename('n')
                events = df.groupby(stratify_by).size().rename('event')

        # --- EXPAND GRID (The Bayesian zero-inflation savior) ---
        unique_levels = [df[col].dropna().unique() for col in stratify_by]
        expanded_index = pd.MultiIndex.from_product(unique_levels, names=stratify_by)

        agg_df = pd.DataFrame(index=expanded_index)
        agg_df = agg_df.join(events, how='left').fillna({'event': 0})
        agg_df = agg_df.reset_index()

        # Map denominators back safely
        if len(denom_cols) == 1:
            agg_df['n'] = agg_df[denom_cols[0]].map(denoms).fillna(0)
        else:
            # Multi-index mapping for denominator
            agg_df = agg_df.set_index(denom_cols)
            agg_df['n'] = denoms
            agg_df = agg_df.reset_index().fillna({'n': 0})

        # Drop strata where the denominator is 0 (prevents division by zero errors in estimators)
        agg_df = agg_df[agg_df['n'] > 0].copy()

        agg_df.attrs = self._obj.attrs.copy()
        agg_df.attrs['prevalence_meta'] = {
            "stratified_by": stratify_by,
            "target": target_col if target_col else stratify_by[-1],
            "type": "trait" if target_col else "compositional"
        }

        return agg_df


@pd.api.extensions.register_dataframe_accessor("geno")
class GenoAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def amr(self) -> pd.DataFrame:
        """Returns only the boolean AMR determinant matrix with CLEAN names."""
        return self._obj.filter(regex="^amr_").rename(columns=lambda c: c.replace('amr_', '', 1))

    @property
    def virulence(self) -> pd.DataFrame:
        """Returns only the Virulence marker matrix with CLEAN names."""
        return self._obj.filter(regex="^vir_").rename(columns=lambda c: c.replace('vir_', '', 1))

    def has_any(self, traits: list[str], domain: str = 'amr') -> pd.Series:
        """
        The vectorized replacement for TraitCollection.has_any().
        Instantly checks 10,000 isolates to see if they have ANY of the target traits.
        """
        # Safely prepend the prefix and check if the columns actually exist
        target_cols = [f"{domain}_{t}" for t in traits if f"{domain}_{t}" in self._obj.columns]

        if not target_cols:
            # If none of the genes exist in the dataset, no isolate has them
            return pd.Series(False, index=self._obj.index)

        # Pandas matrix math: Check across the columns (axis=1) for any True values
        return self._obj[target_cols].any(axis=1)

    def has_all(self, traits: list[str], domain: str = 'vir') -> pd.Series:
        """
        The vectorized replacement for TraitCollection.has_all().
        """
        target_cols = [f"{domain}_{t}" for t in traits]

        # If the user asks for a gene that isn't even in the dataset, they can't have 'all'
        missing_cols = set(target_cols) - set(self._obj.columns)
        if missing_cols:
            return pd.Series(False, index=self._obj.index)

        # Matrix math: Check if ALL target columns are True
        return self._obj[target_cols].all(axis=1)

    def has_gene(self, gene_col: str, target_gene: str) -> pd.Series:
        """
        Kleborate often outputs comma-separated genes (e.g., 'blaKPC-2,sul1').
        This cleanly returns a boolean mask for isolates possessing a specific gene.
        """
        # Fills NA with empty string, then does a substring search
        return self._obj[gene_col].fillna('').str.contains(target_gene, regex=False)

    def sort_loci(self, locus_col: str) -> pd.DataFrame:
        """
        Sorts the DataFrame numerically by locus rather than alphabetically
        so K2 comes before K10.
        """
        df = self._obj.copy()
        # Extract the integer part of the locus (e.g., "K10" -> 10, "O2v1" -> 2)
        sort_key = df[locus_col].str.extract(r'(\d+)').astype(float)
        return df.iloc[sort_key.sort_values().index]


@pd.api.extensions.register_dataframe_accessor("qc")
class QCAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def metrics(self) -> pd.DataFrame:
        """Returns just the quality control metrics matrix with CLEAN names."""
        return self._obj.filter(regex="^qc_").rename(columns=lambda c: c.replace('qc_', '', 1))

    def filter_assemblies(self, min_n50: int = 10000, max_contigs: int = 500,
                          require_species: str = None) -> pd.DataFrame:
        """
        Drops poor-quality genomes from the dataset.
        """
        df = self._obj.copy()

        # Start with a boolean mask where everything is True
        mask = pd.Series(True, index=df.index)

        if 'qc_N50' in df.columns:
            # Coerce to numeric in case Kleborate spat out weird strings like '-'
            n50 = pd.to_numeric(df['qc_N50'], errors='coerce')
            mask &= (n50 >= min_n50) | n50.isna()

        if 'qc_contig_count' in df.columns:
            contigs = pd.to_numeric(df['qc_contig_count'], errors='coerce')
            mask &= (contigs <= max_contigs) | contigs.isna()

        if require_species and 'qc_species' in df.columns:
            mask &= df['qc_species'].str.contains(require_species, case=False, na=False)

        return df[mask]

    def report(self) -> pd.DataFrame:
        """Generates a summary of the dataset quality for the Shiny UI."""
        metrics = self.metrics
        report = {}
        if 'qc_QC_warnings' in metrics.columns:
            report['Total Warnings'] = (metrics['qc_QC_warnings'] != '-').sum()
        if 'qc_N50' in metrics.columns:
            report['Median N50'] = pd.to_numeric(metrics['qc_N50'], errors='coerce').median()

        return pd.Series(report)


# Functions ------------------------------------------------------------------------------------------------------------
def test():
    from pathlib import Path
    from pyseroepi.dist import Distances
    from pyseroepi.io import PathogenwatchKleborateParser

    test_dir = Path('tests')
    dataset = 'pathogenwatch-klepn-klebnet-neonatal-sepsis'
    meta_df = pd.read_csv(test_dir / f'{dataset}-metadata.csv')
    dist = Distances.from_pathogenwatch(test_dir / f'{dataset}-difference-matrix.csv')

    df = PathogenwatchKleborateParser.parse(
        pd.read_csv(test_dir / f'{dataset}-kleborate.csv'), meta_df=meta_df,
        meta_kwargs={
            "id_col": 'NAME',
            "date_col": 'COLLECTION DATE',
            "country_col": 'COUNTRY',
            "lat_col": 'LATITUDE',
            "lon_col": 'LONGITUDE'
        }
    ).assign(
        Outbreak_Cluster = dist.connected_components(threshold=20)
    )


    from pyseroepi.estimators import FrequentistPrevalenceEstimator
    from pyseroepi.estimators.modelled import BayesianPrevalenceEstimator
    freq_estimator = FrequentistPrevalenceEstimator()
    bayes_estimator = BayesianPrevalenceEstimator()

    agg = df.epi.aggregate_prevalence(stratify_by=['country', 'K_locus'])
    r = freq_estimator.calculate(agg)
    # r1 = bayes_estimator.calculate(agg)

    from pyseroepi.plotting import PrevalenceEstimatesPlotter
    plt = PrevalenceEstimatesPlotter(r)
