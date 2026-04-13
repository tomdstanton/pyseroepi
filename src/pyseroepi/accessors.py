from typing import Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pyseroepi.data.gazetteer_data import GAZETTEER_DICT
# import warnings
# warnings.filterwarnings("ignore", message="registration of accessor")


# Accessors ------------------------------------------------------------------------------------------------------------
@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    """
    Pandas accessor for geographical operations on isolate datasets.

    Provides methods for standardizing country names, imputing missing
    coordinates using a gazetteer, and performing reverse geocoding.

    Attributes:
        gazetteer (pd.DataFrame): A DataFrame containing centroid coordinates and
            metadata for countries.

    Examples:
        >>> import pandas as pd
        >>> import pyseroepi.accessors
        >>> df = pd.DataFrame({'country': ['Australia', 'United Kingdom']})
        >>> df = df.geo.standardize_and_impute()
        >>> print(df[['latitude', 'longitude']])
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self._gazetteer = pd.DataFrame.from_dict(GAZETTEER_DICT, orient='index')

    @property
    def gazetteer(self):
        """Returns the internal gazetteer used for coordinate imputation."""
        return self._gazetteer

    def standardize_and_impute(self) -> pd.DataFrame:
        """
        Standardizes country names and imputes missing coordinates.

        Uses the internal gazetteer to find centroids for countries when exact
        latitude and longitude are missing.

        Returns:
            A new DataFrame with imputed coordinates and spatial resolution metadata.
        """
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
        Performs reverse geocoding to determine country/region from coordinates.

        Requires Geopandas to be installed (`pip install pyseroepi[spatial]`).

        Args:
            shapefile_path: Path to the shapefile containing boundary polygons.

        Returns:
            A new DataFrame with updated 'country' information.

        Raises:
            ImportError: If geopandas is not installed.
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
    """
    Pandas accessor for epidemiological analysis on isolate datasets.

    Provides methods for generating epidemic curves, calculating prevalence,
    diversity, and incidence, and identifying transmission clusters.

    Examples:
        >>> import pandas as pd
        >>> import pyseroepi.accessors
        >>> df = pd.DataFrame({
        ...     'date': pd.to_datetime(['2023-01-01', '2023-01-15']),
        ...     'K_locus': ['KL1', 'KL2']
        ... })
        >>> curve = df.epi.epidemic_curve(freq='ME')
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    # --- Shiny UI State Checkers ---

    @property
    def has_temporal(self) -> bool:
        """Checks if the dataset contains valid temporal data (the 'date' column)."""
        # Checks if 'date' exists AND actually has non-null data
        return 'date' in self._obj.columns and self._obj['date'].notna().any()

    @property
    def has_spatial(self) -> bool:
        """Checks if the dataset contains valid spatial coordinates (latitude and longitude)."""
        return 'latitude' in self._obj.columns and 'longitude' in self._obj.columns

    # --- Spatiotemporal Helpers ---

    @property
    def temporal(self) -> pd.Series:
        """
        Returns the primary isolation date series.

        Raises:
            ValueError: If the 'date' column is missing or empty.
        """
        if not self.has_temporal:
            raise ValueError("Temporal column 'date' is missing or completely empty.")
        return self._obj['date']

    @property
    def spatial(self) -> pd.DataFrame:
        """
        Returns the core spatial coordinates (latitude, longitude).

        Raises:
            ValueError: If spatial columns are missing.
        """
        if not self.has_spatial:
            raise ValueError("Spatial columns ('latitude', 'longitude') are missing.")
        return self._obj[['latitude', 'longitude']]

    # --- Time Series / Epidemic Curve Methods ---

    def epidemic_curve(self, freq: str = 'W', stratify_by: str = None) -> pd.DataFrame:
        """
        Generates a time-series DataFrame for plotting epidemic curves.

        Args:
            freq: Time frequency for resampling (e.g., 'D', 'W', 'ME', 'YE').
                Defaults to 'W'.
            stratify_by: Column name to group by before resampling.

        Returns:
            A DataFrame with counts of isolates per time interval.

        Raises:
            ValueError: If temporal data is not available.
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
        """Returns the raw names of user-uploaded clinical/metadata columns."""
        return self._obj.filter(regex="^meta_").columns.tolist()

    @property
    def ui_metadata_columns(self) -> list[str]:
        """Returns clean metadata names (without 'meta_' prefix) for UI display."""
        return [c.replace('meta_', '', 1) for c in self.metadata_columns]

    @property
    def genotypes(self) -> list[str]:
        """Compiles a list of all genetic variables (Core + Accessory traits)."""
        # 1. Grab the core grouping genotypes if they exist
        core_genos = [col for col in ['ST', 'K_locus', 'O_locus'] if col in self._obj.columns]

        # 2. Grab all the dynamically prefixed traits
        trait_genos = self._obj.filter(regex="^(amr|vir)_").columns.tolist()

        return core_genos + trait_genos

    def aggregate_prevalence(self, stratify_by: list[str], target_col: str = None,
                             cluster_col: str = None, negative_indicator: Union[str, list[str]] = '-') -> pd.DataFrame:
        """
        Aggregates data to calculate event counts and denominators for prevalence.

        Supports both trait prevalence (presence/absence of a marker) and
        compositional prevalence (distribution of variants within a locus).

        Args:
            stratify_by: Columns to group by (e.g., ['country']).
            target_col: The column containing the trait/marker to measure.
                If None, compositional prevalence is calculated for the last
                column in `stratify_by`.
            cluster_col: Column containing cluster IDs to adjust for (e.g., nosocomial outbreaks).
            negative_indicator: Value(s) representing the absence of a trait.
                Defaults to '-'.

        Returns:
            An aggregated DataFrame with 'event' and 'n' columns.

        Raises:
            ValueError: If compositional prevalence is requested without enough strata.
        """
        df = self._obj.copy()

        if target_col:
            denom_cols = stratify_by
            valid_df = df.dropna(subset=[target_col]).copy()

            if pd.api.types.is_bool_dtype(valid_df[target_col]):
                valid_df['_target_bool'] = valid_df[target_col]
            else:
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else negative_indicator
                valid_df['_target_bool'] = ~valid_df[target_col].isin(neg_list)

            if cluster_col:
                denoms = valid_df.groupby(denom_cols)[cluster_col].nunique().rename('n') if denom_cols else valid_df[
                    cluster_col].nunique()
                events = valid_df[valid_df['_target_bool']].groupby(denom_cols)[cluster_col].nunique().rename(
                    'event') if denom_cols else valid_df[valid_df['_target_bool']][cluster_col].nunique()
            else:
                denoms = valid_df.groupby(denom_cols).size().rename('n') if denom_cols else len(valid_df)
                events = valid_df.groupby(denom_cols)['_target_bool'].sum().rename('event') if denom_cols else valid_df[
                    '_target_bool'].sum()

        else:
            if len(stratify_by) < 1:
                raise ValueError("Compositional prevalence requires at least 1 stratify_by column.")
            denom_cols = stratify_by[:-1]

            # Drop true unknowns from the target variant column
            valid_df = df.dropna(subset=[stratify_by[-1]]).copy()

            if cluster_col:
                denoms = valid_df.groupby(denom_cols)[cluster_col].nunique().rename('n') if denom_cols else valid_df[
                    cluster_col].nunique()
                events = valid_df.groupby(stratify_by)[cluster_col].nunique().rename('event')
            else:
                denoms = valid_df.groupby(denom_cols).size().rename('n') if denom_cols else len(valid_df)
                events = valid_df.groupby(stratify_by).size().rename('event')

        # Expand Grid
        unique_levels = [df[col].dropna().unique() for col in stratify_by]
        expanded_index = pd.MultiIndex.from_product(unique_levels, names=stratify_by)
        agg_df = pd.DataFrame(index=expanded_index).join(events, how='left').fillna({'event': 0}).reset_index()

        # Safely map denominators (Fixed to handle empty denom_cols for global prevalence)
        if len(denom_cols) == 0:
            agg_df['n'] = denoms
        elif len(denom_cols) == 1:
            agg_df['n'] = agg_df[denom_cols[0]].map(denoms).fillna(0)
        else:
            agg_df = agg_df.set_index(denom_cols)
            agg_df['n'] = denoms
            agg_df = agg_df.reset_index().fillna({'n': 0})

        agg_df = agg_df[agg_df['n'] > 0].copy()

        agg_df.attrs = self._obj.attrs.copy()
        agg_df.attrs['prevalence_meta'] = {
            "stratified_by": stratify_by,
            "target": target_col if target_col else stratify_by[-1],
            "type": "trait" if target_col else "compositional",
            "adjusted_for": cluster_col
        }

        return agg_df

    def aggregate_diversity(self, stratify_by: list[str], target_col: str = None,
                            cluster_col: str = None, negative_indicator: Union[str, list[str]] = '-') -> pd.DataFrame:
        """
        Aggregates data to calculate counts for diversity analysis (e.g., Shannon index).

        Args:
            stratify_by: Columns to group by.
            target_col: The locus or trait to measure diversity for.
            cluster_col: Optional cluster column to adjust for.
            negative_indicator: Value(s) to exclude from diversity counts.

        Returns:
            A DataFrame with 'variant_count' and 'n_total'.

        Raises:
            ValueError: If compositional diversity is requested without enough strata.
        """
        df = self._obj.copy()

        if target_col:
            groupers = stratify_by
            target_strata = stratify_by + [target_col]
            valid_df = df.dropna(subset=[target_col]).copy()

            # For trait diversity, we often want to strip out the "absence" indicators
            # so they don't count as a diversity variant mathematically
            if not pd.api.types.is_bool_dtype(valid_df[target_col]):
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else negative_indicator
                valid_df = valid_df[~valid_df[target_col].isin(neg_list)]
        else:
            if not stratify_by:
                raise ValueError("Compositional diversity requires at least 1 stratify_by column.")
            groupers = stratify_by[:-1]
            target_col = stratify_by[-1]
            target_strata = stratify_by
            valid_df = df.dropna(subset=[target_col]).copy()

        if cluster_col:
            div_df = valid_df.groupby(target_strata, observed=True)[cluster_col].nunique().reset_index()
            div_df = div_df.rename(columns={cluster_col: 'variant_count'})
        else:
            div_df = valid_df.groupby(target_strata, observed=True).size().reset_index(name='variant_count')

        if groupers:
            div_df['n_total'] = div_df.groupby(groupers, observed=True)['variant_count'].transform('sum')
        else:
            div_df['n_total'] = div_df['variant_count'].sum()

        div_df.attrs = self._obj.attrs.copy()
        div_df.attrs["diversity_meta"] = {
            "stratified_by": groupers,
            "target": target_col,
            "type": "trait" if target_col else "compositional",
            "adjusted_for": cluster_col
        }

        return div_df

    def aggregate_incidence(self, stratify_by: list[str], target_col: str = None, freq: str = 'ME',
                            cluster_col: str = None, negative_indicator: Union[str, list[str]] = '-') -> pd.DataFrame:
        """
        Aggregates data for time-series incidence analysis.

        Args:
            stratify_by: Columns to group by.
            target_col: The marker to measure incidence for.
            freq: Time frequency for binning (e.g., 'ME', 'YE'). Defaults to 'ME'.
            cluster_col: Optional cluster column to adjust for.
            negative_indicator: Value(s) representing absence.

        Returns:
            A DataFrame with 'variant_count', 'total_sequenced', and binned dates.

        Raises:
            ValueError: If temporal data is missing or inappropriate strata provided.
        """
        df = self._obj.copy()

        if 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
            raise ValueError("Incidence aggregation requires a valid datetime64 'date' column.")

        # Translate Pandas 2.2+ point offsets ('ME') to period spans ('M')
        period_freq = freq.replace('ME', 'M').replace('YE', 'Y')

        # Snap dates to the requested frequency bin using the safe string
        df['date_bin'] = df['date'].dt.to_period(period_freq).dt.to_timestamp()

        if target_col:
            denom_cols = ['date_bin'] + stratify_by
            target_strata = ['date_bin'] + stratify_by

            valid_df = df.dropna(subset=[target_col]).copy()
            if pd.api.types.is_bool_dtype(valid_df[target_col]):
                valid_df['_target_bool'] = valid_df[target_col]
            else:
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else negative_indicator
                valid_df['_target_bool'] = ~valid_df[target_col].isin(neg_list)

            if cluster_col:
                denoms = valid_df.groupby(denom_cols)[cluster_col].nunique().rename('total_sequenced')
                events = valid_df[valid_df['_target_bool']].groupby(target_strata)[cluster_col].nunique().rename(
                    'variant_count')
            else:
                denoms = valid_df.groupby(denom_cols).size().rename('total_sequenced')
                events = valid_df.groupby(target_strata)['_target_bool'].sum().rename('variant_count')

        else:
            if not stratify_by:
                raise ValueError("Compositional incidence requires at least 1 stratify_by column.")

            denom_cols = ['date_bin'] + stratify_by[:-1]
            target_strata = ['date_bin'] + stratify_by
            valid_df = df.dropna(subset=[stratify_by[-1]]).copy()

            if cluster_col:
                denoms = valid_df.groupby(denom_cols)[cluster_col].nunique().rename('total_sequenced')
                events = valid_df.groupby(target_strata)[cluster_col].nunique().rename('variant_count')
            else:
                denoms = valid_df.groupby(denom_cols).size().rename('total_sequenced')
                events = valid_df.groupby(target_strata).size().rename('variant_count')

        # --- TIME GRID EXPANSION ---
        # Generate an unbroken sequence of dates from the earliest to the latest record
        min_date, max_date = df['date_bin'].min(), df['date_bin'].max()
        all_dates = pd.period_range(min_date, max_date, freq=period_freq).to_timestamp().tolist() if not pd.isna(
            min_date) else []

        unique_levels = [all_dates] + [df[col].dropna().unique() for col in target_strata[1:]]
        expanded_index = pd.MultiIndex.from_product(unique_levels, names=target_strata)

        inc_df = pd.DataFrame(index=expanded_index).join(events, how='left').fillna({'variant_count': 0}).reset_index()

        if len(denom_cols) == 1:  # Only date_bin exists in the strata
            inc_df['total_sequenced'] = inc_df['date_bin'].map(denoms).fillna(0)
        else:
            inc_df = inc_df.set_index(denom_cols)
            inc_df['total_sequenced'] = denoms
            inc_df = inc_df.reset_index().fillna({'total_sequenced': 0})

        # Unlike Prevalence, we do NOT drop rows where total_sequenced == 0.
        # A true 0 sequence volume is critical information for an epicurve gap.
        inc_df = inc_df.rename(columns={'date_bin': 'date'})

        inc_df.attrs = self._obj.attrs.copy()
        inc_df.attrs['incidence_meta'] = {
            "stratified_by": stratify_by,
            "target": target_col if target_col else stratify_by[-1],
            "type": "trait" if target_col else "compositional",
            "freq": freq,
            "adjusted_for": cluster_col
        }

        return inc_df

    def get_transmission_clusters(
            self,
            clone_col: str,
            spatial_threshold_km: float = 10.0,
            temporal_threshold_days: int = 20,
            col_name: str = 'transmission_cluster'
    ) -> pd.DataFrame:
        """
        Identifies transmission clusters based on spatial and temporal proximity.

        Uses a graph-based connected components approach where isolates of the
        same clone are linked if they fall within both distance and time thresholds.

        Args:
            clone_col: Column containing clone IDs (e.g., 'ST' or a custom cluster).
            spatial_threshold_km: Maximum distance in kilometers. Defaults to 10.0.
            temporal_threshold_days: Maximum time difference in days. Defaults to 20.
            col_name: Name for the resulting cluster column.

        Returns:
            A new DataFrame with the transmission cluster labels added.

        Raises:
            KeyError: If required columns ('lat', 'lon', 'date') are missing.
        """
        df = self._obj.copy()

        # 1. Validation Checks
        if clone_col not in df.columns:
            raise KeyError(f"Clone column '{clone_col}' not found in DataFrame.")

        # Intelligently check for your geo accessor/columns
        if 'lat' not in df.columns or 'lon' not in df.columns:
            raise KeyError(
                "Spatial clustering requires 'lat' and 'lon' columns. Ensure geo accessors have parsed coordinates.")

        if 'date' not in df.columns:
            raise KeyError("A 'date' column is required for temporal clustering.")

        # Coerce dates to ensure clean math (drops timezones if present to allow raw day subtraction)
        df['_temp_date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        # Keep track of the original index to map cluster labels back perfectly
        df['_temp_idx'] = np.arange(len(df))

        # Initialize an array to hold the global cluster IDs
        cluster_labels = np.full(len(df), -1)
        current_cluster_id = 0

        # 2. Iterate and Cluster by Clone
        for clone_id, group in df.groupby(clone_col, dropna=True):
            n_points = len(group)

            # If the clone only appears once, it is its own isolated cluster
            if n_points == 1:
                cluster_labels[group['_temp_idx'].values] = current_cluster_id
                current_cluster_id += 1
                continue

            # --- A. Temporal Distance Matrix ---
            # Convert to raw integer days for instantaneous vectorized subtraction
            dates = group['_temp_date'].values.astype('datetime64[D]').astype(int)
            time_dist = np.abs(dates[:, None] - dates[None, :])

            # --- B. Spatial Distance Matrix ---
            # Sklearn haversine requires coordinates in radians [lat, lon]
            coords = np.radians(group[['lat', 'lon']].values)
            # Multiply by Earth's radius (~6371 km) to convert to kilometers
            space_dist = haversine_distances(coords, coords) * 6371.0

            # --- C. Graph Adjacency ---
            # Two isolates are linked ONLY if they meet BOTH the temporal and spatial thresholds
            adjacency = (time_dist <= temporal_threshold_days) & (space_dist <= spatial_threshold_km)

            # --- D. Extract Transmission Chains ---
            graph = csr_matrix(adjacency)
            n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

            # Offset the local labels by the global counter to ensure unique IDs across all clones
            global_labels = labels + current_cluster_id
            cluster_labels[group['_temp_idx'].values] = global_labels

            current_cluster_id += n_components

        # 3. Cleanup and Formatting
        # Format beautifully as strings (e.g., "TC_0001", "TC_0002")
        df[col_name] = [f"TC_{i:04d}" if i != -1 else np.nan for i in cluster_labels]

        return df.drop(columns=['_temp_idx', '_temp_date'])


@pd.api.extensions.register_dataframe_accessor("geno")
class GenoAccessor:
    """
    Pandas accessor for genetic and trait-based operations.

    Provides methods for filtering determinants, checking for trait patterns,
    and sorting loci.

    Examples:
        >>> import pandas as pd
        >>> import pyseroepi.accessors
        >>> df = pd.DataFrame({'amr_blaKPC': [True, False], 'vir_ybt': [True, True]})
        >>> amr_matrix = df.geno.amr
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def amr(self) -> pd.DataFrame:
        """Returns the AMR determinant matrix with 'amr_' prefix removed from names."""
        return self._obj.filter(regex="^amr_").rename(columns=lambda c: c.replace('amr_', '', 1))

    @property
    def virulence(self) -> pd.DataFrame:
        """Returns the Virulence marker matrix with 'vir_' prefix removed from names."""
        return self._obj.filter(regex="^vir_").rename(columns=lambda c: c.replace('vir_', '', 1))

    def has_any(self, traits: list[str], domain: str = 'amr') -> pd.Series:
        """
        Checks if isolates possess ANY of the target traits.

        Args:
            traits: List of trait names (without prefix).
            domain: The domain prefix ('amr', 'vir', etc.). Defaults to 'amr'.

        Returns:
            A boolean Series indicating presence of any target trait.
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
        Checks if isolates possess ALL of the target traits.

        Args:
            traits: List of trait names.
            domain: Domain prefix. Defaults to 'vir'.

        Returns:
            A boolean Series.
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
        Searches for a specific gene within a comma-separated column.

        Args:
            gene_col: Column name containing gene lists.
            target_gene: The specific gene to find.

        Returns:
            A boolean Series.
        """
        # Fills NA with empty string, then does a substring search
        return self._obj[gene_col].fillna('').str.contains(target_gene, regex=False)

    def sort_loci(self, locus_col: str) -> pd.DataFrame:
        """
        Sorts the DataFrame numerically by locus (e.g., K2 before K10).

        Args:
            locus_col: Column containing locus names.

        Returns:
            A sorted copy of the DataFrame.
        """
        df = self._obj.copy()
        # Extract the integer part of the locus (e.g., "K10" -> 10, "O2v1" -> 2)
        sort_key = df[locus_col].str.extract(r'(\d+)').astype(float)
        return df.iloc[sort_key.sort_values().index]


@pd.api.extensions.register_dataframe_accessor("qc")
class QCAccessor:
    """
    Pandas accessor for quality control operations.

    Provides methods for filtering assemblies based on metrics like N50 and
    contig count.

    Examples:
        >>> import pandas as pd
        >>> import pyseroepi.accessors
        >>> df = pd.DataFrame({'qc_N50': [50000, 5000]})
        >>> clean_df = df.qc.filter_assemblies(min_n50=10000)
    """
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    @property
    def metrics(self) -> pd.DataFrame:
        """Returns the QC metrics matrix with 'qc_' prefix removed."""
        return self._obj.filter(regex="^qc_").rename(columns=lambda c: c.replace('qc_', '', 1))

    def filter_assemblies(self, min_n50: int = 10000, max_contigs: int = 500,
                          require_species: str = None) -> pd.DataFrame:
        """
        Filters genomes based on quality thresholds.

        Args:
            min_n50: Minimum N50 value. Defaults to 10000.
            max_contigs: Maximum number of contigs. Defaults to 500.
            require_species: Optional species name to filter for.

        Returns:
            A filtered copy of the DataFrame.
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
        """
        Generates a summary report of dataset quality.

        Returns:
            A Series containing summary metrics (e.g., Total Warnings, Median N50).
        """
        metrics = self.metrics
        report = {}
        if 'qc_QC_warnings' in metrics.columns:
            report['Total Warnings'] = (metrics['qc_QC_warnings'] != '-').sum()
        if 'qc_N50' in metrics.columns:
            report['Median N50'] = pd.to_numeric(metrics['qc_N50'], errors='coerce').median()

        return pd.Series(report)
