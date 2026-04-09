"""
Module for dealing with the web-app server logic
"""
import pandas as pd
from scipy.sparse import coo_matrix
from pathlib import Path
from shiny import Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import render_plotly
from pathogenx.io import GenotypeFile, MetaFile, DistFile
from pathogenx.dataset import Dataset
from pathogenx.calculators import PrevalenceCalculator, PrevalenceResult
from pathogenx.app.plotters import (PrevalencePlotter, StrataPlotter, SummaryBarPlotter, CoveragePlotter, MapPlotter,
                                 merge_prevalence_figs)
from .ui import prevalence_panel, coverage_panel, dataframe_panel

# Constants ------------------------------------------------------------------------------------------------------------
_VAR_CATEGORIES = ('genotype', 'adjustment', 'spatial', 'temporal', 'custom')


# Main server ----------------------------------------------------------------------------------------------------------
def main_server(input: Inputs, output: Outputs, session: Session):
    # Reactive container for user uploaded files to be loaded ----------------------
    reactive_dataset = reactive.Value[Dataset | None](None)

    def _load_genotypes() -> pd.DataFrame | None:
        file_infos = input.genotype_upload()
        if not file_infos:
            ui.notification_show("Genotype file is required.", type="error")
            return None
        f = file_infos[0]
        genotype_file = GenotypeFile.from_flavour(Path(f["datapath"]), input.genotype_flavour())
        try:
            data = genotype_file.load()
            ui.notification_show('Successfully loaded genotype file', type='message')
            return data
        except Exception as e:
            ui.notification_show(f"Error loading genotype file: {e}", type="error", duration=None)
            return None

    def _load_metadata() -> pd.DataFrame | None:
        file_infos = input.metadata_upload()
        if not file_infos:
            return None
        f = file_infos[0]
        metadata_file = MetaFile.from_flavour(Path(f["datapath"]), input.metadata_flavour())
        try:
            data = metadata_file.load()
            ui.notification_show('Successfully loaded metadata file', type='message')
            return data
        except Exception as e:
            ui.notification_show(f"Error loading metadata file: {e}", type="error", duration=None)
            return None

    def _load_distances() -> tuple[coo_matrix, list[str]] | None:
        file_infos = input.distance_upload()
        if not file_infos:
            return None
        f = file_infos[0]
        distance_file = DistFile.from_flavour(Path(f["datapath"]), input.distance_flavour())
        try:
            data = distance_file.load()
            ui.notification_show('Successfully loaded distance file', type='message')
            return data
        except Exception as e:
            ui.notification_show(f"Error loading distance file: {e}", type="error", duration=None)
            return None

    @reactive.effect
    @reactive.event(input.load_data, ignore_none=False)
    def _load_data_and_create_dataset():
        """
        This event runs when the user clicks the 'Load data' button.
        It loads all files, creates a Dataset object, calculates clusters,
        and sets the reactive value, triggering downstream updates.
        """
        genotypes = _load_genotypes()
        if genotypes is None or genotypes.empty:
            return
        dataset = Dataset(genotypes, _load_metadata(), _load_distances())
        if dataset.distances is not None:
            ui.notification_show('Calculating clusters...')
            clusters = dataset.calculate_clusters(method=input.cluster_method(), distance=input.snp_distance())
            ui.notification_show(f'{clusters.nunique()} unique clusters', type="message")
        reactive_dataset.set(dataset)

    # User data to be filtered and used for prevalences ----------------------------
    @reactive.calc
    def reactive_data() -> pd.DataFrame | None:
        if (d := reactive_dataset.get()) is None or len(d) == 0:
            return None

        filtered_data = d.data  # This creates a copy via the Dataset.data attribute method

        for var in _VAR_CATEGORIES:
            if (variable_col := input[f"{var}_variable"]()) and (filter_values := input[f"{var}_filter"]()):
                if var == 'temporal':
                    min_val, max_val = filter_values
                    filtered_data = filtered_data[filtered_data[variable_col].between(min_val, max_val)]
                else:
                    filtered_data = filtered_data[filtered_data[variable_col].isin(filter_values)]

        if filtered_data.empty:
            ui.notification_show("No data matches the current filter selection.", type="warning")
            return None

        return filtered_data

    @reactive.effect
    @reactive.event(reactive_dataset)
    def _toggle_panels_on_load():
        d: Dataset | None = reactive_dataset.get()
        if d is None or len(d) == 0:
            ui.update_sidebar("sidebar", show=False)  # Start with sidebar hidden
            ui.update_accordion_panel('accordion', "upload_panel", show=True)  # Show the upload panel
            for i in (prevalence_panel, coverage_panel, dataframe_panel):
                ui.remove_accordion_panel('accordion', i)
        else:
            ui.update_sidebar("sidebar", show=True)  # Show the sidebar
            ui.update_accordion_panel('accordion', "upload_panel", show=False)  # Hide the upload panel
            metadata_cols = list(d.metadata_columns) if d.metadata_columns is not None else []
            genotype_cols = list(d.genotype_columns)
            all_cols = sorted(genotype_cols + metadata_cols)
            adjust_cols = ['Cluster'] if d.distances is not None else []
            for var, cols in zip(_VAR_CATEGORIES, (genotype_cols, adjust_cols, metadata_cols, metadata_cols, all_cols)):
                # Add a blank choice to allow the input to be unselected
                ui.update_selectize(f"{var}_variable", choices=[''] + cols)
            ui.update_selectize("heatmap_x", choices=[''] + all_cols)
            ui.update_selectize("bars_x", choices=[''] + all_cols)
            for i in (prevalence_panel, coverage_panel, dataframe_panel):
                ui.insert_accordion_panel('accordion', i)

    def _create_event_lambda(var_name: str):
        """Function factory to correctly capture the loop variable for the lambda."""
        return lambda: input[f"{var_name}_variable"]()

    @reactive.effect
    # This effect is explicitly triggered when any of the variable selection dropdowns change.
    # We use a function factory (_create_event_lambda) to avoid the classic Python closure-in-a-loop issue.
    @reactive.event(*(_create_event_lambda(var) for var in _VAR_CATEGORIES), ignore_init=True)
    def _update_filter_selectors():
        """
        Populates filter controls based on the unique values in the columns
        the user has chosen in the variable selectors.
        """
        if (d := reactive_dataset.get()) is None or len(d) == 0:
            return
        for var in _VAR_CATEGORIES:
            # Only proceed if a column has been selected from the dropdown.
            # This check handles both None and empty string ""
            if selected_col := input[f"{var}_variable"]():
                col_data = d.data[selected_col]
                if var == 'temporal':
                    min_, max_ = int(col_data.min()), int(col_data.max())
                    ui.update_slider(f"{var}_filter", min=min_, max=max_, value=(min_, max_))
                else:
                    choices = sorted(col_data.dropna().unique().tolist())
                    ui.update_selectize(f"{var}_filter", choices=choices, selected=[])

    # Output dataframe -------------------------------------------------------------
    @render.data_frame
    def dataframe():
        if (df := reactive_data()) is None or df.empty:
            return None
        return df

    # Output summary ---------------------------------------------------------------
    @render.text
    def summary():
        if (dataset := reactive_dataset.get()) is None or len(dataset) == 0:
            return ''
        d = dataset.data
        if (f := reactive_data()) is None or f.empty:
            return f"Samples: 0/{len(d)}"
        out = [f"Samples: {len(f)}/{len(d)}"]
        out += [f'{f[i].nunique()}/{d[i].nunique()} unique {i} {v} variables' for v in _VAR_CATEGORIES if (i := input[v]())]
        return '; '.join(out)

    @reactive.calc
    def prevalence() -> PrevalenceResult | None:
        """Calculates overall prevalence for the selected genotype."""
        if (dataset := reactive_dataset()) is None or len(dataset) == 0:
            return None
        if (data := reactive_data()) is None or len(data) == 0:
            return None
        if not (genotype := input.genotype_variable()):
            return None
        adjust_for = input.adjustment_variable()
        n_distinct = input.bars_x()
        calc = PrevalenceCalculator(stratify_by=[genotype], adjust_for=[adjust_for] if adjust_for else None,
                                    n_distinct=[n_distinct] if n_distinct else None)
        result = calc.calculate(data)
        return result

    @reactive.calc
    def prevalence_stratified() -> PrevalenceResult | None:
        """Calculates prevalence stratified by a second variable for the heatmap."""
        if (data := reactive_data()) is None or len(data) == 0:
            return None
        if (genotype := input.genotype_variable()) is None:
            return None
        if not (heatmap_x := input.heatmap_x()):
            return None
        adjust_by = input.adjustment_variable()
        calc = PrevalenceCalculator(stratify_by=[genotype, heatmap_x], adjust_for=[adjust_by] if adjust_by else None,
                                    denominator=(genotype if input.heatmap_swap_denominator() else heatmap_x))
        result = calc.calculate(data)
        return result

    @reactive.calc
    def prevalence_coverage() -> PrevalenceResult | None:
        """Calculates prevalence stratified by a second variable for the heatmap."""
        if (data := reactive_data()) is None or len(data) == 0:
            return None
        if not (genotype := input.genotype_variable()):
            return None
        if not (spatial := input.spatial_variable()):
            return None
        adjust_by = input.adjustment_variable()
        calc = PrevalenceCalculator(stratify_by=[spatial, genotype],adjust_for=[adjust_by] if adjust_by else None)
        result = calc.calculate(data)
        return result

    @output
    @render_plotly
    def merged_plot():
        """Renders the main combined plot (pyramid, heatmap, bars)."""
        # Get reactive variables
        if (r1 := prevalence()) is None or len(r1) == 0:
            return None
        p1 = PrevalencePlotter().plot(r1)
        p2 = StrataPlotter().plot(r2) if (r2 := prevalence_stratified()) else None
        p3 = SummaryBarPlotter(fill_by=bars_x).plot(r1) if (bars_x := input.bars_x()) else None
        return merge_prevalence_figs(p1, p2, p3)

    @output
    @render_plotly
    def coverage_plot():
        if (r := prevalence_coverage()) is None or len(r) == 0:
            return None
        return CoveragePlotter().plot(r)

    @output
    @render_plotly
    def map_plot():
        """Renders the main combined plot (pyramid, heatmap, bars)."""
        if (r := prevalence_coverage()) is None or len(r) == 0:
            return None
        return  MapPlotter().plot(r)

