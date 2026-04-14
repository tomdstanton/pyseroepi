

# Functions ------------------------------------------------------------------------------------------------------------
def test():
    from pathlib import Path
    import pandas as pd
    from pyseroepi.dist import Distances
    from pyseroepi.io import PathogenwatchKleborateParser
    from pyseroepi.estimators import FrequentistPrevalenceEstimator, BayesianPrevalenceEstimator

    test_dir = Path('tests')
    dataset = 'pathogenwatch-klepn-klebnet-neonatal-sepsis'
    meta_df = pd.read_csv(test_dir / f'{dataset}-metadata.csv')
    dist = Distances.from_pathogenwatch(test_dir / f'{dataset}-difference-matrix.csv')
    clusters = dist.connected_components(threshold=20)

    df = PathogenwatchKleborateParser.parse(
        pd.read_csv(test_dir / f'{dataset}-kleborate.csv'),
        meta_df=meta_df,
        meta_kwargs={
            "id_col": 'NAME',
            "date_col": 'COLLECTION DATE',
            "country_col": 'COUNTRY',
            "lat_col": 'LATITUDE',
            "lon_col": 'LONGITUDE'
        }
    ).join(clusters, on='sample_id')

    t_clusters = df.epi.transmission_clusters(clusters.name)

    from pyseroepi.estimators import RegressionPrevalenceEstimator
    est = RegressionPrevalenceEstimator()
    agg = df.epi.aggregate_prevalence(stratify_by=['country', 'K_locus'], cluster_col=clusters.name)
    res = est.calculate(agg)
