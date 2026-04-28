"""
Module for genotype file I/O and parsing.
"""

from typing import Optional, Any
import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series


# Schema & Data Stewardship --------------------------------------------------------------------------------------------
class UnifiedIsolateSchema(pa.DataFrameModel):
    """
    Pandera schema for validating and standardizing isolate datasets.

    This schema ensures that all input data, whether from Pathogenwatch or user
    uploads, conforms to a unified structure for downstream analysis.

    Attributes:
        sample_id: Unique identifier for each isolate.
        date: Isolation date (datetime64[ns]).
        date_resolution: Precision of the isolation date ('year', 'month', 'day', 'unknown').
        latitude: Latitude coordinate (-90 to 90).
        longitude: Longitude coordinate (-180 to 180).
        country: Country name.
        region: Geographical region.
        iso3: ISO 3166-1 alpha-3 country code.
        spatial_resolution: Precision of spatial data.
        ST: Multi-locus sequence type.
        K_locus: Klebsiella K-locus.
        O_locus: Klebsiella O-locus.
        K_type: Predicted K-serotype.
        O_type: Predicted O-serotype.
        qc_metrics: Dynamic columns for quality control (prefixed with 'qc_').
        amr_traits: Dynamic columns for AMR markers (prefixed with 'amr_').
        vir_traits: Dynamic columns for virulence markers (prefixed with 'vir_').
        user_metadata: Dynamic columns for user metadata (prefixed with 'meta_').

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'sample_id': ['S1'], 'K_locus': ['KL1']})
        >>> validated_df = UnifiedIsolateSchema.validate(df)
    """
    # Core
    sample_id: Series["string"] = pa.Field(unique=True, coerce=True)

    # ADDED: Temporal Fields (Required so 'strict=filter' doesn't delete them!)
    date: Optional[Series["datetime64[ns]"]] = pa.Field(nullable=True, coerce=True)
    date_resolution: Optional[Series["category"]] = pa.Field(
        isin=['year', 'month', 'day', 'unknown'], nullable=True, coerce=True
    )

    # Spatial Fields
    latitude: Optional[Series["Float64"]] = pa.Field(ge=-90, le=90, nullable=True, coerce=True)
    longitude: Optional[Series["Float64"]] = pa.Field(ge=-180, le=180, nullable=True, coerce=True)
    country: Optional[Series["string"]] = pa.Field(nullable=True, coerce=True)
    region: Optional[Series["string"]] = pa.Field(nullable=True, coerce=True)
    iso3: Optional[Series["string"]] = pa.Field(nullable=True, coerce=True)
    spatial_resolution: Optional[Series["category"]] = pa.Field(
        isin=['exact', 'hospital', 'city', 'country', 'region', 'unknown'], nullable=True, coerce=True
    )

    # Genotypes
    ST: Optional[Series["category"]] = pa.Field(nullable=True, coerce=True)
    K_locus: Optional[Series["category"]] = pa.Field(nullable=True, coerce=True)
    O_locus: Optional[Series["category"]] = pa.Field(nullable=True, coerce=True)
    K_type: Optional[Series["category"]] = pa.Field(nullable=True, coerce=True)
    O_type: Optional[Series["category"]] = pa.Field(nullable=True, coerce=True)

    # Domains (Using Any allows regex matching on mixed dtypes)
    qc_metrics: Optional[Series[Any]] = pa.Field(alias="^qc_.*$", regex=True, nullable=True)
    amr_traits: Optional[Series[Any]] = pa.Field(alias="^amr_.*$", regex=True, nullable=True)
    vir_traits: Optional[Series[Any]] = pa.Field(alias="^vir_.*$", regex=True, nullable=True)
    user_metadata: Optional[Series[Any]] = pa.Field(alias="^meta_.*$", regex=True, nullable=True)

    class Config:
        strict = "filter"


# Parsers --------------------------------------------------------------------------------------------------------------
class BaseParser:
    """
    Base class for standardizing external datasets.

    Subclasses must define column mappings and category definitions for specific
    input formats (e.g., Kleborate output).
    """

    # Subclasses define how to map raw columns to UnifiedIsolateSchema columns
    column_map: dict[str, str] = {}
    qc_cols: list[str] = []
    vir_cols: list[str] = []
    amr_cols: list[str] = []

    @staticmethod
    def _clean_mixed_dates(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
        """
        Standardizes mixed-format dates (YYYY, YYYY-MM, YYYY-MM-DD).

        Args:
            df: The DataFrame to clean.
            date_col: The column containing date information. Defaults to 'date'.

        Returns:
            The DataFrame with standardized 'date' and 'date_resolution' columns.
        """
        if date_col not in df.columns:
            return df
        s = df[date_col].copy()
        s = s.astype(str).str.strip()
        s = s.replace(['nan', '<NA>', 'None', ''], pd.NA)
        lengths = s.str.len()

        df['date_resolution'] = 'unknown'
        df.loc[lengths == 4, 'date_resolution'] = 'year'
        df.loc[(lengths >= 6) & (lengths <= 7) & s.str.contains(r'-|/', na=False), 'date_resolution'] = 'month'
        df.loc[lengths >= 8, 'date_resolution'] = 'day'

        is_year = df['date_resolution'] == 'year'
        is_month = df['date_resolution'] == 'month'

        s.loc[is_year] = s.loc[is_year] + '-07-02'
        s.loc[is_month] = s.loc[is_month] + '-15'

        df[date_col] = pd.to_datetime(s, errors='coerce')
        df.loc[df[date_col].isna(), 'date_resolution'] = 'unknown'
        return df

    @classmethod
    def _ingest_user_metadata(cls, meta_df: pd.DataFrame, id_col: str,
                              date_col: str = None, region_col: str = None,
                              country_col: str = None, lat_col: str = None, lon_col: str = None) -> pd.DataFrame:
        """
        Standardizes user-uploaded metadata.

        Args:
            meta_df: The metadata DataFrame.
            id_col: Column name for sample IDs.
            date_col: Column name for isolation dates.
            region_col: Column name for regions.
            country_col: Column name for countries.
            lat_col: Column name for latitudes.
            lon_col: Column name for longitudes.

        Returns:
            A cleaned metadata DataFrame with prefixed user columns.
        """
        df = meta_df.copy()
        rename_map = {id_col: 'sample_id'}
        if date_col: rename_map[date_col] = 'date'
        if region_col: rename_map[region_col] = 'region'
        if country_col: rename_map[country_col] = 'country'
        if lat_col: rename_map[lat_col] = 'latitude'
        if lon_col: rename_map[lon_col] = 'longitude'

        df = df.rename(columns=rename_map)

        if 'date' in df.columns:
            df = cls._clean_mixed_dates(df, date_col='date')

        core_cols = ['sample_id', 'date', 'region', 'country', 'latitude', 'longitude', 'date_resolution']
        new_names = {col: f"meta_{col}" for col in df.columns if col not in core_cols}
        return df.rename(columns=new_names)

    @staticmethod
    def _optimize_categorical_dtypes(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Converts string columns to categorical dtypes if cardinality is low.

        Args:
            df: The DataFrame to optimize.
            threshold: The ratio of unique values to total rows below which
                a column is converted to categorical. Defaults to 0.5.

        Returns:
            The optimized DataFrame.
        """
        df_opt = df.copy()
        total_rows = len(df_opt)
        if total_rows == 0:
            return df_opt

        string_cols = df_opt.select_dtypes(include=['object', 'string', 'category']).columns
        for col in string_cols:
            if isinstance(df_opt[col].dtype, pd.CategoricalDtype):
                continue
            num_unique = df_opt[col].nunique(dropna=False)
            if (num_unique / total_rows) < threshold and num_unique < total_rows:
                df_opt[col] = df_opt[col].astype('category')
        return df_opt

    @staticmethod
    def _optimize_binary_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts numeric columns containing only 0 and 1 to Int8.

        Args:
            df: The DataFrame to optimize.

        Returns:
            The optimized DataFrame.
        """
        df_opt = df.copy()
        for col in df_opt.columns:
            if pd.api.types.is_numeric_dtype(df_opt[col]):
                unique_vals = df_opt[col].dropna().unique()
                if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    df_opt[col] = df_opt[col].astype('Int8')
        return df_opt

    @classmethod
    def parse(
            cls,
            genotype_df: pd.DataFrame,
            meta_df: Optional[pd.DataFrame] = None,
            meta_kwargs: dict = None
    ) -> pd.DataFrame:
        """
        Parses and validates a genotype dataset, optionally merging with metadata.

        Args:
            genotype_df: The raw genotype DataFrame.
            meta_df: Optional metadata DataFrame to merge.
            meta_kwargs: Arguments for metadata ingestion (e.g., column names).

        Returns:
            A validated DataFrame conforming to UnifiedIsolateSchema.
        """
        attrs = genotype_df.attrs
        df = genotype_df.copy()
        df = df.rename(columns=cls.column_map)
        new_names = {}
        for col in df.columns:
            if col in cls.qc_cols:
                new_names[col] = f"qc_{col}"
            elif col in cls.vir_cols:
                new_names[col] = f"vir_{col}"
            elif col in cls.amr_cols:
                new_names[col] = f"amr_{col}"

        df = df.rename(columns=new_names)

        for col in UnifiedIsolateSchema.to_schema().columns.keys():
            if col not in df.columns and not col.startswith(('qc_', 'amr_', 'vir_')):
                df[col] = pd.NA


        if meta_df is not None:
            kwargs = meta_kwargs or {}
            # Use the namespaced schema method
            clean_meta = cls._ingest_user_metadata(meta_df, **kwargs)
            overlap_cols = [col for col in clean_meta.columns
                            if col in df.columns and col != 'sample_id']
            df = pd.merge(df.drop(columns=overlap_cols), clean_meta,
                          on='sample_id', how='left')

        # Use the namespaced optimization methods
        df = cls._optimize_binary_dtypes(df)
        df = cls._optimize_categorical_dtypes(df, threshold=0.4)

        # Pandera often strips attrs on validation, so attach them AFTER returning the validated df
        valid_df = UnifiedIsolateSchema.validate(df)
        valid_df.attrs = attrs

        return valid_df


class PathogenwatchKleborateParser(BaseParser):
    """
    Adapter for Kleborate files downloaded from Pathogenwatch.

    This parser maps Kleborate's specific column names and categories to the
    UnifiedIsolateSchema.
    """
    column_map = {
        'Genome Name': 'sample_id',
        'ST': 'ST',
        'K_locus': 'K_locus',
        'O_locus': 'O_locus',
        'K_type': 'K_type',
        'O_type': 'O_type'
    }
    qc_cols = ['species', 'species_match', 'contig_count', 'N50', 'largest_contig', 'total_size', 'ambiguous_bases',
               'QC_warnings', 'K_locus_confidence', 'O_locus_confidence']
    vir_cols = ['YbST', 'Yersiniabactin', 'CbST', 'Colibactin', 'AbST', 'Aerobactin', 'SmST', 'Salmochelin', 'RmST',
                'RmpADC', 'virulence_score', 'rmpA2']
    amr_cols = ['AGly_acquired', 'Col_acquired', 'Fcyn_acquired', 'Flq_acquired', 'Gly_acquired', 'MLS_acquired',
                'Phe_acquired', 'Rif_acquired', 'Sul_acquired', 'Tet_acquired', 'Tgc_acquired', 'Tmt_acquired',
                'Bla_acquired', 'Bla_inhR_acquired', 'Bla_ESBL_acquired', 'Bla_ESBL_inhR_acquired', 'Bla_Carb_acquired',
                'Bla_chr', 'SHV_mutations', 'Omp_mutations', 'Col_mutations', 'Flq_mutations', 'resistance_score',
                'num_resistance_classes', 'num_resistance_genes', 'Ciprofloxacin_prediction', 'Ciprofloxacin_profile',
                'Ciprofloxacin_MIC_prediction']
