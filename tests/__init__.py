from pyseroepi.constants import PlotType


# Functions ------------------------------------------------------------------------------------------------------------
def test():
    from pathlib import Path
    import pandas as pd
    from pyseroepi.dist import Distances
    from pyseroepi.io import PathogenwatchKleborateParser

    test_dir = Path('tests')
    dataset = 'pathogenwatch-klepn-klebnet-neonatal-sepsis'
    meta_df = pd.read_csv(test_dir / f'{dataset}-metadata.csv', engine="pyarrow")
    dist = Distances.from_pathogenwatch(test_dir / f'{dataset}-difference-matrix.csv')
    clusters = dist.connected_components(threshold=20)

    df = PathogenwatchKleborateParser.parse(
        pd.read_csv(test_dir / f'{dataset}-kleborate.csv', engine="pyarrow"),
        meta_df=meta_df,
        meta_kwargs={
            "id_col": 'NAME',
            "date_col": 'COLLECTION DATE',
            "country_col": 'COUNTRY',
            "lat_col": 'LATITUDE',
            "lon_col": 'LONGITUDE'
        }
    ).join(clusters, on='sample_id')

    df = df.geo.standardize_and_impute()
    t_clusters = df.epi.transmission_clusters(clusters.name)
    df[t_clusters.name] = t_clusters

    from pyseroepi.estimators import FrequentistPrevalenceEstimator as Estimator
    from pyseroepi.formulation import PostHocFormulationDesigner as Designer

    est = Estimator()
    agg = df.epi.aggregate_prevalence(stratify_by=['country', 'K_locus'], cluster_col=t_clusters.name)
    res = est.calculate(agg)
    res.plot(PlotType.COMPOSITION_BAR)
    vax = Designer(valency=20)
    formulation = vax.evaluate(res, 'country')
    formulation.plot('rank_stability_bump')


    from pyseroepi.estimators import BayesianPrevalenceEstimator as Estimator
    from pyseroepi.formulation import CVFormulationDesigner
    est = Estimator()
    agg = df.epi.aggregate_prevalence(stratify_by=['country', 'K_locus'])
    res = est.calculate(agg)
    est.diagnostics()
    vax = Designer()
    formulation = vax.evaluate(est, agg, 'country')
