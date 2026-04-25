from shiny import module, reactive, render, ui
from shinywidgets import render_widget, output_widget

import pandas as pd

from pyseroepi.formulation import CVFormulationDesigner, Formulation
from pyseroepi.estimators import FrequentistPrevalenceEstimator
from pyseroepi.io import PathogenwatchKleborateParser
from pyseroepi.dist import Distances


# =====================================================================================
# MODULE 1: GLOBAL BURDEN (Tab 1)
# =====================================================================================
@module.server
def burden_server(input, output, session, shared_df: reactive.Value):

    # --- STAGE 1: DYNAMIC COLUMN MAPPING ---
    @render.ui
    def dynamic_meta_mapping():
        meta_info = input.metadata_file()
        if not meta_info:
            return ui.div()  # Return nothing if no file

        try:
            # PEEK AT THE HEADERS: nrows=0 makes this parse instantly
            cols = pd.read_csv(meta_info[0]["datapath"], nrows=0, engine="pyarrow").columns.tolist()

            # Inject a beautiful accordion with dropdowns
            return ui.accordion(
                ui.accordion_panel(
                    "Map Metadata Columns",
                    ui.p("Please match your columns to the required fields:", class_="text-muted small"),
                    ui.input_selectize("map_id", "Sample ID", choices=[""] + cols, selected=""),
                    ui.input_selectize("map_date", "Collection Date", choices=[""] + cols, selected=""),
                    ui.input_selectize("map_country", "Country", choices=[""] + cols, selected=""),
                    ui.input_selectize("map_lat", "Latitude", choices=[""] + cols, selected=""),
                    ui.input_selectize("map_lon", "Longitude", choices=[""] + cols, selected=""),
                ),
                id="meta_accordion",
                open="Map Metadata Columns"
            )
        except Exception as e:
            return ui.div(f"Could not read metadata columns: {e}", class_="text-danger small")

    # --- STAGE 2: THE EXECUTION PIPELINE ---
    @reactive.Effect
    @reactive.event(input.btn_process)
    def load_data():
        genotype_info = input.genotype_file()
        if not genotype_info:
            ui.notification_show("Kleborate output is required.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Parsing Kleborate Output...", value=10)
                genotype_df = pd.read_csv(genotype_info[0]["datapath"], engine="pyarrow")
                metadata_df = None
                metadata_kwargs = {}

                # Check if metadata was uploaded AND if they mapped the columns
                if metadata_info := input.metadata_file():
                    p.set(message="Parsing Metadata...", value=20)
                    metadata_df = pd.read_csv(metadata_info[0]["datapath"], engine="pyarrow")

                    # Dynamically build the kwargs from the UI dropdowns!
                    # The `or None` ensures we don't pass empty strings if they left a dropdown blank
                    metadata_kwargs = {
                        "id_col": input.map_id() or None,
                        "date_col": input.map_date() or None,
                        "country_col": input.map_country() or None,
                        "lat_col": input.map_lat() or None,
                        "lon_col": input.map_lon() or None
                    }

                    # Optional: Add a quick guard clause to ensure they actually mapped the ID
                    if not metadata_kwargs["id_col"]:
                        ui.notification_show("You must select a Sample ID column to merge metadata.", type="error")
                        return

                p.set(message="Merging Datasets...", value=30)
                df = PathogenwatchKleborateParser.parse(
                    genotype_df,
                    meta_df=metadata_df,
                    meta_kwargs=metadata_kwargs
                )

                if distance_info := input.distance_file():
                    p.set(message="Parsing Distances...", value=40)
                    dist = Distances.from_pathogenwatch(distance_info[0]["datapath"])

                    p.set(message="Calculating clusters...", value=50)
                    clusters = dist.connected_components(threshold=20)

                    p.set(message="Merging Datasets...", value=60)
                    df = df.join(clusters, on='sample_id')

                # df = df.geo.standardize_and_impute()
                # t_clusters = df.epi.transmission_clusters(clusters.name)
                # df[t_clusters.name] = t_clusters

                p.set(message="Pipeline Complete!", value=100)
                shared_df.set(df)
                ui.notification_show("Data successfully parsed and networked.", type="message", duration=4)

            except Exception as e:
                ui.notification_show(f"Pipeline Error: {str(e)}", type="error", duration=15)


# =====================================================================================
# MODULE 2: FORMULATION ENGINE (Tab 2)
# =====================================================================================
@module.server
def formulation_server(input, output, session, shared_df: reactive.Value):
    baseline_res = reactive.Value(None)  # The PrevalenceEstimates object
    current_vaccine = reactive.Value(None)  # The active Formulation object

    @reactive.Effect
    @reactive.event(input.btn_run_designer)
    def generate_optimal():
        df = shared_df.get()
        if df is None:
            ui.notification_show("Please upload data in Tab 1 first.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Aggregating target prevalences...", value=20)

                # 1. Baseline Math (We store this so the Custom Override can use it later)
                est = FrequentistPrevalenceEstimator()  # Or Bayesian!
                agg_df = df.epi.aggregate_prevalence(stratify_by=[input.holdout_col(), 'K_locus'])
                base_result = est.calculate(agg_df)
                baseline_res.set(base_result)

                # 2. Run the Designer!
                p.set(message=f"Running LOO Cross-Validation on {input.holdout_col()}...", value=50)
                designer = CVFormulationDesigner(valency=input.max_valency())
                optimal_formulation = designer.evaluate(est, agg_df, loo_col=input.holdout_col())

                # Update the active vaccine state
                current_vaccine.set(optimal_formulation)

                # Populate the choices for the custom drag-and-drop box!
                all_loci = base_result.data[base_result.target].unique().tolist()
                ui.update_selectize("custom_targets", choices=all_loci)

                p.set(message="Rendering dashboards...", value=95)
                ui.notification_show("Optimal Formulation Designed!", type="message")

            except Exception as e:
                ui.notification_show(f"Designer Error: {str(e)}", type="error", duration=10)

    # --- THE CUSTOM OVERRIDE LOGIC ---
    @reactive.Effect
    @reactive.event(input.btn_copy_optimal)
    def copy_to_custom():
        vaccine = current_vaccine.get()
        if vaccine:
            ui.update_selectize("custom_targets", selected=vaccine.get_targets())
            ui.notification_show("Copied to manual editor.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_eval_custom)
    def evaluate_custom():
        targets = input.custom_targets()
        res = baseline_res.get()
        if not targets or res is None: return

        # Call our brilliant alternative constructor!
        custom_vaccine = Formulation.from_custom(list(targets), res)
        current_vaccine.set(custom_vaccine)
        ui.notification_show("Custom vaccine evaluated.", type="message")

    # --- TAB 2: RENDERING THE DASHBOARD ---

    @render.ui
    def formulation_main_panel():
        vaccine = current_vaccine.get()
        if vaccine is None:
            return ui.div("Click 'Generate Optimal Vaccine' to begin.", class_="text-center mt-5 text-muted fs-4")

        # The dynamic target string: e.g., "[K2] - [K1] - [K10]"
        target_str = " - ".join(f"[{t}]" for t in vaccine.get_targets())

        return ui.div(
            ui.layout_column_wrap(
                ui.value_box("Target Assembly", ui.h4(target_str), theme="bg-primary"),
                ui.value_box("Valency", f"{vaccine.max_valency}-valent", theme="bg-dark"),
                width=1 / 2, class_="mb-3"
            ),
            ui.card(
                ui.card_header("Geographical Coverage (Stacked Composition)"),
                output_widget("coverage_plot")
            ),
            ui.card(
                ui.card_header("Cross-Validation Stability Matrix"),
                output_widget("stability_plot")
            )
        )

    @render_widget
    def coverage_plot():
        vaccine = current_vaccine.get()
        # Dispatches to your Plotly Registry!
        return vaccine.plot('composition_bar') if vaccine else None

    @render_widget
    def stability_plot():
        vaccine = current_vaccine.get()
        # If it's a custom formulation, the stability matrix is empty, so we handle that gracefully in the plotter
        return vaccine.plot('rank_stability_bump') if vaccine else None


def server(input, output, session):
    global_df = reactive.Value(None)
    burden_server("tab1_namespace", shared_df=global_df)
    formulation_server("tab1_namespace", shared_df=global_df)
