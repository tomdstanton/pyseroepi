from pathlib import Path
from typing import Optional, Union, Callable, get_origin, get_args, Literal
import asyncio
import dataclasses
import inspect
import tempfile
import re
import os

from shiny import module, reactive, render, ui
from shinywidgets import render_widget

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import MDS
import pandera.errors
from joblib import dump as joblib_dump, load as joblib_load
from openai import AsyncOpenAI

from seroepi.app.ui import dt_download_ui
from seroepi.formulation import CVFormulationDesigner, PostHocFormulationDesigner, Formulation
from seroepi import estimators
from seroepi.io import PathogenwatchKleborateParser
from seroepi.dist import Distances
from seroepi.constants import EstimatorType, AggregationType, PlotType, InferenceMethod
from seroepi.client import PathogenwatchClient


# =====================================================================================
# HELPERS
# =====================================================================================
def _format_metadata_ui(meta_dict: dict) -> list:
    """Helper to cleanly format a dictionary into bolded UI paragraphs."""
    formatted_ui = []
    for k, v in meta_dict.items():
        if isinstance(v, list):
            val_str = ", ".join(map(str, v)) if v else "None"
        elif hasattr(v, 'value'):
            val_str = str(v.value).title()
        else:
            val_str = str(v)
        formatted_ui.append(ui.p(ui.tags.strong(f"{k.replace('_', ' ').title()}: "), val_str, class_="mb-1"))
    return formatted_ui

# =====================================================================================
# REUSABLE MODULES
# =====================================================================================
@module.server
def safe_plot_server(input, output, session, data_reactive: reactive.Value, plot_type: Union[PlotType, str, Callable]):
    """A universal plotting server module that handles null-checks and error boundaries."""

    @render_widget
    def plot():
        if (data := data_reactive.get()) is None:
            return None

        # Resolve the plot type if it's passed as a reactive function (e.g. dropdowns)
        p_type = plot_type() if callable(plot_type) else plot_type

        try:
            return data.plot(p_type)
        except Exception as e:
            ui.notification_show(f"Plotting Error: {str(e)}", type="error", duration=10)
            return None


@module.server
def dt_download_server(input, output, session, data_callable: Callable, filename: str, height: str = "600px"):
    """A universal module for rendering interactive DataTables with an attached CSV export."""
    @render.data_frame
    def table():
        if (df := data_callable()) is None:
            return pd.DataFrame()
        return render.DataTable(df, selection_mode="none", height=height, filters=True)

    @render.download(filename=filename)
    async def btn_download():
        if (df := data_callable()) is not None:
            yield await asyncio.to_thread(df.to_csv, index=False)

# =====================================================================================
# MODULE 1: GLOBAL BURDEN (Tab 1)
# =====================================================================================
@module.server
def _burden_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    prev_results = app_state["prev_results"]
    fitted_estimator = app_state["fitted_estimator"]
    pw_collections_cache = app_state["pw_collections_cache"]
    shared_dist = app_state["shared_dist"]
    shared_agg_df = app_state["shared_agg_df"]

    # --- STAGE 0: DYNAMIC TAB VISIBILITY ---
    @reactive.Effect
    async def manage_tab_visibility():
        est = fitted_estimator.get()
        tabs = [
            ("tab_aggregated_data", shared_agg_df.get() is not None),
            ("tab_prevalence_data", prev_results.get() is not None),
            ("tab_prevalence_plot", prev_results.get() is not None),
            ("tab_model_diagnostics", est is not None and hasattr(est, 'diagnostics')),
            ("tab_cluster_network", shared_dist.get() is not None)
        ]
        for tab, show in tabs:
            await session.send_custom_message("toggle_tab", {"tab": tab, "show": show})

    @reactive.Effect
    def manage_accordion_state():
        if shared_df.get() is None:
            ui.update_accordion("burden_accordion", show="Load Data 💽")

    # --- STAGE 1: DYNAMIC COLUMN MAPPING ---
    @render.ui
    def dynamic_meta_mapping():
        if not (meta_info := input.metadata_file()):
            return ui.div()  # Return nothing if no file

        try:
            # PEEK AT THE HEADERS: nrows=0 makes this parse instantly
            cols = pd.read_csv(meta_info[0]["datapath"], nrows=0).columns.tolist()

            # Define aliases for intelligent column guessing
            aliases = {
                "map_id": ["sample", "id", "isolate", "name", "run", "barcode"],
                "map_date": ["date", "year", "collection", "time"],
                "map_country": ["country", "nation"],
                "map_lat": ["lat", "latitude"],
                "map_lon": ["lon", "long", "longitude"]
            }

            def guess_match(field_id: str, available_cols: list[str]) -> str:
                targets = aliases.get(field_id, [])
                lower_cols = [c.lower().strip() for c in available_cols]
                
                # 1. Exact match first (e.g. 'sample_id' == 'sample_id')
                for t in targets:
                    if t in lower_cols: return available_cols[lower_cols.index(t)]
                        
                # 2. Substring match (e.g. 'collection' in 'collection_date')
                for t in targets:
                    for i, c in enumerate(lower_cols):
                        if t == 'id' and not re.search(r'\bid\b', c.replace('_', ' ')): continue # Prevent 'id' matching 'width'
                        if t in c: return available_cols[i]
                return ""

            # Clean setup for UI mapping fields to avoid repetitive code
            mapping_fields = [
                ("map_id", "Sample ID"),
                ("map_date", "Collection Date"),
                ("map_country", "Country"),
                ("map_lat", "Latitude"),
                ("map_lon", "Longitude"),
            ]

            # Inject a beautiful accordion with dropdowns
            return ui.accordion(
                ui.accordion_panel(
                    "Map Metadata Columns",
                    ui.p("Please match your columns to the required fields:", class_="text-muted small"),
                    *[ui.input_selectize(idx, label, choices=[""] + cols, selected=guess_match(idx, cols)) for idx, label in mapping_fields]
                ),
                id="meta_accordion",
                open="Map Metadata Columns"
            )
        except Exception as e:
            return ui.div(f"Could not read metadata columns: {e}", class_="text-danger small")

    # --- STAGE 2: THE EXECUTION PIPELINE ---
    @reactive.Effect
    @reactive.event(input.btn_process)
    async def load_data():
        if not (genotype_info := input.genotype_file()):
            ui.notification_show("Kleborate output is required.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Parsing Kleborate Output...", value=10)
                await asyncio.sleep(0)  # Yield to event loop to update UI
                genotype_df = await asyncio.to_thread(pd.read_csv, genotype_info[0]["datapath"], engine="pyarrow")
                metadata_df = None
                metadata_kwargs = {}

                # Check if metadata was uploaded AND if they mapped the columns
                if metadata_info := input.metadata_file():
                    p.set(message="Parsing Metadata...", value=20)
                    await asyncio.sleep(0)
                    metadata_df = await asyncio.to_thread(pd.read_csv, metadata_info[0]["datapath"], engine="pyarrow")

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
                await asyncio.sleep(0)
                
                def parse_and_impute():
                    d = PathogenwatchKleborateParser.parse(
                        genotype_df,
                        meta_df=metadata_df,
                        meta_kwargs=metadata_kwargs
                    )
                    return d.geo.standardize_and_impute()
                    
                df = await asyncio.to_thread(parse_and_impute)

                if distance_info := input.distance_file():
                    p.set(message="Parsing Distances...", value=40)
                    await asyncio.sleep(0)
                    shared_dist.set(await asyncio.to_thread(Distances.from_pathogenwatch, distance_info[0]["datapath"]))

                p.set(message="Pipeline Complete!", value=100)
                await asyncio.sleep(0)
                shared_df.set(df)
                ui.notification_show("Data successfully loaded.", type="message", duration=4)

                # Automatically advance the accordion to the prevalence panel
                ui.update_accordion("burden_accordion", show="Cluster Generation 🕸️")

            except pandera.errors.SchemaError as e:
                ui.notification_show("Validation Error: Uploaded data does not match the required Kleborate schema. Please check your file.", type="error", duration=15)
                shared_df.set(None)
                # Reset the UI input process by forcing the accordion to stay open
                ui.update_accordion("burden_accordion", show="Data Ingestion")
            except pandera.errors.SchemaErrors as e:
                ui.notification_show("Validation Error: Multiple schema violations detected in the uploaded file.", type="error", duration=15)
                shared_df.set(None)
                ui.update_accordion("burden_accordion", show="Data Ingestion")
            except Exception as e:
                ui.notification_show(f"Pipeline Error: {str(e)}", type="error", duration=15)

    # --- STAGE 2B: PATHOGENWATCH API PROTOTYPE ---
    @reactive.Effect
    @reactive.event(input.btn_fetch_pw)
    async def fetch_pw_collections():
        api_key = input.pw_api_key()
        if not api_key:
            ui.notification_show("Please enter an API Key.", type="warning")
            return
            
        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Connecting to Pathogenwatch...", value=30)
                await asyncio.sleep(0)
                
                def fetch_collections():
                    with PathogenwatchClient(api_key) as client:
                        return list(client.get_collections())
                collections = await asyncio.to_thread(fetch_collections)
                    
                if not collections:
                    ui.notification_show("No collections found.", type="warning")
                    return
                    
                collections.sort(key=lambda x: x.name.lower() if x.name else "")
                
                cache_map = {c.uuid: c for c in collections}
                choices_map = {"": "Select..."}
                choices_map.update({c.uuid: c.name or "Unnamed Collection" for c in collections})
                
                pw_collections_cache.set(cache_map)
                ui.update_selectize("pw_collection", choices=choices_map)
                
                p.set(message="Collections fetched!", value=100)
                ui.notification_show(f"Found {len(collections)} collections.", type="message")
            except Exception as e:
                ui.notification_show(f"API Error: {str(e)}", type="error", duration=10)

    @reactive.Effect
    @reactive.event(input.btn_load_pw)
    async def load_pw_collection():
        api_key = input.pw_api_key()
        selected_uuid = input.pw_collection()
        cache = pw_collections_cache.get()
        
        if not api_key or not selected_uuid or selected_uuid not in cache:
            ui.notification_show("Please fetch and select a collection.", type="warning")
            return
            
        collection = cache[selected_uuid]
        
        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message=f"Fetching genomes for {collection.name}...", value=30)
                await asyncio.sleep(0)
                
                def fetch_genomes():
                    with PathogenwatchClient(api_key) as client:
                        return collection.get_genomes(client)
                genomes = await asyncio.to_thread(fetch_genomes)
                    
                if not genomes:
                    ui.notification_show("Collection is empty.", type="warning")
                    return
                    
                p.set(message="Parsing to DataFrame...", value=80)
                await asyncio.sleep(0)
                
                df = await asyncio.to_thread(pd.DataFrame, genomes)
                
                p.set(message="Pipeline Complete!", value=100)
                await asyncio.sleep(0)
                
                shared_df.set(df)
                ui.notification_show(f"Loaded {len(df)} genomes from Pathogenwatch.", type="message", duration=4)
                
                # Automatically advance the accordion
                ui.update_accordion("burden_accordion", show="Cluster Generation 🕸️")
                
            except Exception as e:
                ui.notification_show(f"Load Error: {str(e)}", type="error", duration=15)

    @render.download(filename="seroepi_workspace.sero")
    async def btn_save_workspace():
        export_state = {k: v.get() for k, v in app_state.items()}
        
        fd, temp_path_str = tempfile.mkstemp(suffix=".sero")
        os.close(fd)
        path = Path(temp_path_str)
        
        await asyncio.to_thread(joblib_dump, export_state, path)
        data = await asyncio.to_thread(path.read_bytes)
        path.unlink()
        yield data

    @reactive.Effect
    @reactive.event(input.workspace_file)
    async def load_workspace():
        if not (file_info := input.workspace_file()):
            return
            
        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Restoring Workspace...", value=50)
                await asyncio.sleep(0)
                
                imported_state = await asyncio.to_thread(joblib_load, file_info[0]["datapath"])
                
                # Safely restore all reactive values that match our state dictionary
                for key, r_val in app_state.items():
                    if key in imported_state:
                        r_val.set(imported_state[key])
                        
                p.set(message="Workspace Restored!", value=100)
                ui.notification_show("Session successfully restored.", type="message")
                
                # Jump to preview to confirm data loaded
                ui.update_accordion("burden_accordion", show="Cluster Generation 🕸️")
                
            except Exception as e:
                ui.notification_show(f"Failed to restore workspace: {str(e)}", type="error", duration=15)

    @reactive.Effect
    @reactive.event(input.btn_clear_data)
    def clear_data():
        for r_val in app_state.values():
            r_val.set(None)
        app_state["pw_collections_cache"].set({})
        ui.notification_show("All data and results cleared.", type="message")
        ui.update_accordion("burden_accordion", show="Load Data 💽")

    # --- STAGE 2C: CLUSTER GENERATION ---
    @reactive.Effect
    @reactive.event(input.btn_calc_snp_clusters)
    async def calc_snp_clusters():
        df = shared_df.get()
        dist = shared_dist.get()

        if df is None:
            ui.notification_show("Please load data first.", type="warning")
            return
        if dist is None:
            ui.notification_show("Please upload a SNP distance matrix in the Data Ingestion tab.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Calculating SNP clusters...", value=30)
                await asyncio.sleep(0)
                clusters = await asyncio.to_thread(dist.connected_components, threshold=input.snp_threshold())
                clusters.name = f"snp_cluster_{input.snp_threshold()}"

                p.set(message="Merging to dataset...", value=70)
                await asyncio.sleep(0)

                # Safely drop existing column if running the identical threshold twice to prevent Pandas overlap suffixes
                if clusters.name in df.columns:
                    df = df.drop(columns=[clusters.name])

                df = df.join(clusters, on='sample_id')
                shared_df.set(df)

                p.set(message="Done!", value=100)
                ui.notification_show(f"Successfully generated {clusters.name}", type="message")
            except Exception as e:
                ui.notification_show(f"SNP Clustering Error: {str(e)}", type="error", duration=15)

    @reactive.Effect
    @reactive.event(input.btn_calc_trans_clusters)
    async def calc_trans_clusters():
        df = shared_df.get()
        if df is None:
            ui.notification_show("Please load data first.", type="warning")
            return
        if not input.trans_clone_col():
            ui.notification_show("Please select a Clone Column.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Calculating transmission clusters...", value=40)
                await asyncio.sleep(0)

                if not (df.epi.has_spatial and df.epi.has_temporal):
                    ui.notification_show("Transmission clustering requires both spatial and temporal metadata.",
                                         type="error")
                    return

                def run_trans_clusters():
                    return df.epi.transmission_clusters(
                        clone_col=input.trans_clone_col(),
                        spatial_threshold_km=input.trans_spatial_thr(),
                        temporal_threshold_days=input.trans_temporal_thr()
                    )
                t_clusters = await asyncio.to_thread(run_trans_clusters)

                if t_clusters.name in df.columns:
                    df = df.drop(columns=[t_clusters.name])

                # Create a copy so Shiny recognizes it as a new object and triggers UI updates
                df = df.copy()
                df[t_clusters.name] = t_clusters
                shared_df.set(df)

                p.set(message="Done!", value=100)
                ui.notification_show(f"Successfully generated {t_clusters.name}", type="message")
            except Exception as e:
                ui.notification_show(f"Transmission Clustering Error: {str(e)}", type="error", duration=15)

    # --- STAGE 3: PREVALENCE ANALYSIS WORKFLOW ---
    @reactive.Effect
    def update_prev_dropdowns():
        if (df := shared_df.get()) is not None:
            # Utilize the core library's accessor properties to determine suitable columns dynamically
            ui.update_selectize("prev_target", choices=[""] + df.epi.genotypes, selected="")
            ui.update_selectize("prev_stratify", choices=df.epi.stratify_cols)
            ui.update_selectize("prev_cluster", choices=[""] + df.epi.cluster_cols, selected="")
            # Populate the transmission clone dropdown perfectly using the genotypes accessor!
            st_select = "ST" if "ST" in df.epi.genotypes else ""
            ui.update_selectize("trans_clone_col", choices=[""] + df.epi.genotypes, selected=st_select)

    @reactive.Effect
    def update_network_dropdowns():
        if (df := shared_df.get()) is not None:
            # Combine categories and remove duplicates while preserving order
            color_choices = list(dict.fromkeys(df.epi.stratify_cols + df.epi.cluster_cols + df.epi.genotypes))
            ui.update_selectize("network_color_col", choices=[""] + color_choices, selected="")
            
            t_clusters = [c for c in df.epi.cluster_cols if 'transmission' in c.lower()]
            ui.update_selectize("network_trans_col", choices=[""] + t_clusters, selected=t_clusters[-1] if t_clusters else "")

    @render.ui
    def estimator_params_ui():
        est_type = input.prev_estimator()
        
        # Hide if a model has been uploaded since we won't be compiling it
        if "model_upload" in input and input.model_upload():
            return ui.div()
            
        try:
            estimator_class_name = EstimatorType(est_type).class_name
            EstimatorClass = getattr(estimators, estimator_class_name, None)
        except Exception:
            return ui.div()
            
        if not EstimatorClass:
            return ui.div()

        # Check for in-memory model
        est = fitted_estimator.get()
        if est and type(est).__name__ == estimator_class_name and getattr(est, 'is_fitted_', False):
            return ui.div(
                ui.hr(),
                ui.p(
                    "A fitted model of this type is currently in memory. "
                    "Prevalence will be predicted using this model's learned weights. "
                    "To train a new model, click 'Clear Fitted Model' below.",
                    class_="text-info small mb-1"
                )
            )
            
        # Safely inspect the signature
        sig = inspect.signature(EstimatorClass.__init__)
        elements = []
        
        # Dynamically build standard Shiny inputs depending on the annotated python Type
        for name, param in sig.parameters.items():
            # Exclude internal/inappropriate variables from the user UI
            if name in ['self', 'target_event', 'target_n', 'lat_col', 'lon_col']:
                continue
                
            input_id = f"est_param_{name}"
            label = name.replace("_", " ").title()
            default = param.default if param.default != inspect.Parameter.empty else None
            
            origin = get_origin(param.annotation)
            
            if param.annotation is int or param.annotation is float:
                elements.append(ui.input_numeric(input_id, label, value=default))
            elif param.annotation is bool:
                elements.append(ui.input_checkbox(input_id, label, value=default))
            elif origin is Literal:
                choices = get_args(param.annotation)
                choices_dict = {c: str(c).replace('_', ' ').title() for c in choices}
                elements.append(ui.input_select(input_id, label, choices=choices_dict, selected=default))
            elif 'InferenceMethod' in str(param.annotation):
                default_val = default.value if hasattr(default, 'value') else default
                elements.append(ui.input_select(input_id, label, choices=InferenceMethod.ui_labels(), selected=default_val))
        
        if elements:
            return ui.div(ui.hr(), ui.p("Hyperparameters", class_="text-muted small mb-1"), *elements)
        return ui.div()

    @render.ui
    def model_io_ui():
        est_type = input.prev_estimator()
        try:
            estimator_class_name = EstimatorType(est_type).class_name
            EstimatorClass = getattr(estimators, estimator_class_name, None)
        except Exception:
            return ui.div()
            
        # Safely hide the UI if the estimator (like Frequentist) doesn't support model loading
        if not EstimatorClass or not hasattr(EstimatorClass, 'load_model'):
            return ui.div()
            
        elements = [
            ui.hr(),
            ui.p("Model Weights (Optional)", class_="text-muted small mb-1"),
            ui.input_file("model_upload", "Load Fitted Model (.pkl)", accept=[".pkl"])
        ]
        
        # If a model of the CURRENT type is fitted in memory, provide the download button
        est = fitted_estimator.get()
        if est and type(est).__name__ == EstimatorClass.__name__ and getattr(est, 'is_fitted_', False):
            elements.append(
                ui.download_button("model_download", "Download Fitted Model", class_="btn-outline-primary w-100 mb-2")
            )
            elements.append(
                ui.input_action_button("btn_clear_model", "Clear Fitted Model", class_="btn-outline-danger w-100 mb-3")
            )
            
        return ui.div(*elements)

    @reactive.Effect
    @reactive.event(input.btn_clear_model)
    def clear_model():
        fitted_estimator.set(None)
        ui.notification_show("Fitted model cleared from memory.", type="message")

    @render.download(filename=lambda: f"fitted_{EstimatorType(input.prev_estimator()).name.lower()}_model.pkl")
    async def model_download():
        est = fitted_estimator.get()
        if est and getattr(est, 'is_fitted_', False):
            fd, temp_path_str = tempfile.mkstemp(suffix=".pkl")
            os.close(fd)
            path = Path(temp_path_str)
            await asyncio.to_thread(est.save_model, path)
            data = await asyncio.to_thread(path.read_bytes)
            path.unlink()
            yield data

    @reactive.Effect
    @reactive.event(input.btn_aggregate_prev)
    async def aggregate_prevalence():
        if (df := shared_df.get()) is None:
            ui.notification_show("Please upload and process data first.", type="warning")
            return

        target = input.prev_target() or None
        cluster = input.prev_cluster() or None
        agg_mode = input.prev_agg_type()
        
        stratify = list(input.prev_stratify())

        if agg_mode == AggregationType.TRAIT and not stratify:
            ui.notification_show("Trait prevalence requires at least one stratification column.", type="error")
            return
            
        if not target and not stratify:
            ui.notification_show("Please select a target or stratification column.", type="warning")
            return

        # INTENT ROUTING: 
        # If the user wants a compositional breakdown, the target column acts as the primary grouping variable!
        if agg_mode == AggregationType.COMPOSITIONAL and target:
            if target not in stratify:
                stratify.append(target)
            target_col = None  # Leave None so the accessor groups compositionally by the last strata
        else:
            target_col = target

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Aggregating data...", value=20)
                await asyncio.sleep(0)
                def run_aggregation():
                    return df.epi.aggregate_prevalence(
                        stratify_by=stratify,
                        target_col=target_col,
                        cluster_col=cluster,
                        negative_indicator=input.prev_negative()
                    )
                agg_df = await asyncio.to_thread(run_aggregation)
                
                shared_agg_df.set(agg_df)
                p.set(message="Done!", value=100)
                ui.notification_show("Data aggregated successfully!", type="message")
                ui.update_accordion("burden_accordion", show="Prevalence Estimation 📈")
                ui.update_navset("main_dashboard_tabs", selected="tab_aggregated_data")
                
            except Exception as e:
                ui.notification_show(f"Aggregation Error: {str(e)}", type="error", duration=15)

    @reactive.Effect
    @reactive.event(input.btn_estimate_prev)
    async def estimate_prevalence():
        if (agg_df := shared_agg_df.get()) is None:
            ui.notification_show("Please aggregate data first.", type="warning")
            return
            
        est_type = input.prev_estimator()
        
        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Instantiating estimator...", value=20)
                await asyncio.sleep(0)

                estimator_class_name = EstimatorType(est_type).class_name
                
                if not hasattr(estimators, estimator_class_name):
                    ui.notification_show(f"{estimator_class_name} is not available. Did you install seroepi[models]?", type="error")
                    return
                
                EstimatorClass = getattr(estimators, estimator_class_name)
                
                # Check if we have a model already compiled and fitted in memory
                in_memory_est = fitted_estimator.get()
                can_reuse_memory = (
                    in_memory_est is not None 
                    and type(in_memory_est).__name__ == EstimatorClass.__name__ 
                    and getattr(in_memory_est, 'is_fitted_', False)
                )
                
                # Intercept the model upload input if it exists
                file_info = input.model_upload() if "model_upload" in input else None
                
                if hasattr(EstimatorClass, 'load_model') and file_info:
                    p.set(message="Loading model...", value=40)
                    await asyncio.sleep(0)
                    try:
                        est = EstimatorClass.load_model(file_info[0]["datapath"])
                    except Exception as e:
                        ui.notification_show(f"Failed to load model: {str(e)}", type="error", duration=10)
                        return
                        
                    p.set(message="Predicting prevalence...", value=60)
                    await asyncio.sleep(0)
                    res = await asyncio.to_thread(est.predict, agg_df)
                elif can_reuse_memory:
                    p.set(message="Using in-memory fitted model...", value=60)
                    await asyncio.sleep(0)
                    est = in_memory_est
                    res = await asyncio.to_thread(est.predict, agg_df)
                else:
                    # Dynamically extract and assign kwargs for the selected estimator class
                    kwargs = {}
                    sig = inspect.signature(EstimatorClass.__init__)
                    for name, param in sig.parameters.items():
                        if name in ['self', 'target_event', 'target_n', 'lat_col', 'lon_col']: 
                            continue
                        
                        input_id = f"est_param_{name}"
                        if input_id in input:
                            val = input[input_id]()
                            if val == "":  # Skip empty numeric inputs to fallback on defaults
                                continue
                                
                            origin = get_origin(param.annotation)
                            
                            if param.annotation is int: kwargs[name] = int(val)
                            elif param.annotation is float: kwargs[name] = float(val)
                            elif param.annotation is bool: kwargs[name] = bool(val)
                            elif origin is Literal: kwargs[name] = val
                            elif 'InferenceMethod' in str(param.annotation): kwargs[name] = InferenceMethod(val)
                            else: kwargs[name] = val
                            
                    # Ensure SpatialPrevalenceEstimator maps correctly to Pandas generated columns
                    if estimator_class_name == "SpatialPrevalenceEstimator":
                        kwargs['lat_col'] = 'latitude'
                        kwargs['lon_col'] = 'longitude'
                                
                    est = EstimatorClass(**kwargs)
                    
                    p.set(message="Fitting model and calculating...", value=60)
                    await asyncio.sleep(0)
                    res = await asyncio.to_thread(est.calculate, agg_df)
                
                fitted_estimator.set(est)
                prev_results.set(res)
                
                ui.notification_show("Prevalence calculated successfully! Check the plot tab.", type="message")
                p.set(message="Done", value=100)
                ui.update_navset("main_dashboard_tabs", selected="tab_prevalence_plot")
                await asyncio.sleep(0)
            except Exception as e:
                ui.notification_show(f"Calculation Error: {str(e)}", type="error", duration=15)

    # --- STAGE 4: RENDERING THE DASHBOARDS ---
    @render.ui
    def dashboard_content():
        if (df := shared_df.get()) is None:
            return ui.div("Please upload and process files to view the global dataset.", class_="text-center mt-5 text-muted fs-4")

        return dt_download_ui("dashboard", "Global Dataset Preview")
        
    dt_download_server("dashboard", data_callable=lambda: shared_df.get(), filename="global_dataset.csv")

    @render.ui
    def agg_data_content():
        if (agg_df := shared_agg_df.get()) is None:
            return ui.div("Please aggregate data in the sidebar to view the aggregated dataset.", class_="text-center mt-5 text-muted fs-4")

        meta_dict = agg_df.attrs.get('metric_meta', {})
        meta_ui = _format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Aggregates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("agg_data", "Aggregates Data")
        )
        
    dt_download_server("agg_data", data_callable=lambda: shared_agg_df.get(), filename="aggregated_data.csv")

    @render.ui
    def prev_summary_content():
        if (res := prev_results.get()) is None:
            return ui.div("Calculate prevalence to view the results data.", class_="text-center mt-5 text-muted fs-4")

        # Dynamically extract instance attributes into a dictionary
        meta_dict = {f.name: getattr(res, f.name) for f in dataclasses.fields(res) if f.name not in ['data', 'model_results']}
        meta_ui = _format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Estimates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("prev_summary", "Estimates Data")
        )
        
    dt_download_server("prev_summary", data_callable=lambda: prev_results.get().data if prev_results.get() else None, filename="estimates_data.csv", height="400px")

    @render.ui
    def model_diagnostics_content():
        if (est := fitted_estimator.get()) is None:
            return ui.div("Estimate prevalence to view model diagnostics.", class_="text-center mt-5 text-muted fs-4")
            
        if not hasattr(est, 'diagnostics'):
            return ui.div(
                ui.p(f"Diagnostics are not applicable for {type(est).__name__}."), 
                class_="text-center mt-5 text-muted fs-4"
            )
            
        try:            
            # Execute a dummy call to gracefully catch TypeErrors (like attempting diagnostics on SVI)
            _ = est.diagnostics()
            return dt_download_ui("model_diagnostics", "NumPyro Posterior Diagnostics")
        except Exception as e:
            return ui.div(f"Diagnostics Error: {str(e)}", class_="text-danger text-center mt-5 fs-5")

    def get_model_diagnostics():
        if (est := fitted_estimator.get()) is not None and hasattr(est, 'diagnostics'):
            try:
                return est.diagnostics()
            except Exception:
                pass
        return None

    dt_download_server("model_diagnostics", data_callable=get_model_diagnostics, filename="model_diagnostics.csv", height="500px")

    safe_plot_server("prev_plot", data_reactive=prev_results, plot_type=input.prev_plot_type)

    @render.download(filename=lambda: f"prevalence_plot.{input.plot_format()}")
    async def btn_download_plot():
        res = prev_results.get()
        if res is None:
            ui.notification_show("No plot data available to export.", type="warning")
            return
            
        try:
            # Generate the raw Plotly Figure (go.Figure) from the registry
            fig = res.plot(input.prev_plot_type())
            
            fd, temp_path_str = tempfile.mkstemp(suffix=f".{input.plot_format()}")
            os.close(fd)
            path = Path(temp_path_str)
            
            # Kaleido naturally intercepts write_image locally
            def save_fig():
                fig.write_image(str(path), format=input.plot_format(), width=input.plot_width(), height=input.plot_height())
                return path.read_bytes()
                
            data = await asyncio.to_thread(save_fig)
            path.unlink()
            yield data
        except Exception as e:
            ui.notification_show(f"Plot Export Error: {str(e)}", type="error", duration=10)
            return

    @reactive.Calc
    async def network_layout():
        dist = shared_dist.get()
        if dist is None:
            return None, None
            
        with ui.Progress(min=0, max=100) as p:
            p.set(message="Preparing distance matrix...", value=20)
            await asyncio.sleep(0)  # Flush UI
            
            # 1. Prepare the full dense distance matrix
            dense_dist = dist.matrix.toarray().astype(float)
            
            # If the matrix was sparse, 0s off the diagonal represent missing data. Fill with max distance.
            mask = (dense_dist == 0) & (~np.eye(dense_dist.shape[0], dtype=bool))
            if mask.any():
                dense_dist[mask] = dense_dist.max() * 2

            p.set(message="Calculating network layout (MDS)...", value=50)
            await asyncio.sleep(0)
            
            # 2. Generate 2D Layout using Multi-Dimensional Scaling
            # n_init=1 and max_iter=100 keeps it lightning fast for Shiny UI!
            def run_mds():
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=100)
                return mds.fit_transform(dense_dist)
            pos = await asyncio.to_thread(run_mds)
            
            p.set(message="Layout complete!", value=100)
            await asyncio.sleep(0)
            
            return dense_dist, pos

    @render_widget
    async def cluster_network_plot():
        df = shared_df.get()
        dist = shared_dist.get()
        
        if df is None or dist is None:
            return go.Figure().update_layout(title="Upload a SNP distance matrix in Tab 1 to view the network.", template="plotly_white")
            
        dense_dist, pos = await network_layout()
        if pos is None:
            return go.Figure().update_layout(title="Processing distance matrix...", template="plotly_white")
            
        # Call the new routing method on the Distances core object!
        return dist.plot(
            PlotType.NETWORK,
            df=df,
            pos=pos,
            edge_type=input.network_edge_type(),
            threshold=input.network_snp_threshold(),
            trans_col=input.network_trans_col(),
            color_col=input.network_color_col()
        )


# =====================================================================================
# MODULE 2: FORMULATION ENGINE (Tab 2)
# =====================================================================================
@module.server
def _formulation_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    baseline_res = app_state["baseline_res"]
    current_vaccine = app_state["current_vaccine"]
    shared_agg_df = app_state["shared_agg_df"]
    prev_results = app_state["prev_results"]
    fitted_estimator = app_state["fitted_estimator"]

    @reactive.Effect
    async def manage_form_tabs():
        has_vac = current_vaccine.get() is not None
        for tab in ["tab_form_plots", "tab_form_rankings", "tab_form_stability", "tab_form_permutations"]:
            await session.send_custom_message("toggle_tab", {"tab": tab, "show": has_vac})

    @reactive.Effect
    def update_form_inputs():
        res = prev_results.get()
        est = fitted_estimator.get()
        
        if res is None or est is None:
            ui.update_selectize("form_holdout", choices=[])
            ui.update_select("form_designer", choices={"cv": "Cross-Validated (Rigorous)"})
            return

        # 1. Populate the Stratum dropdown exactly from the prior calculation metadata
        holdouts = [c for c in res.stratified_by if c != res.target]
        if holdouts:
            ui.update_selectize("form_holdout", choices=[""] + holdouts, selected=holdouts[0])
        else:
            ui.update_selectize("form_holdout", choices=[])
            
        # 2. Restrict Designer Type based on the exact estimator instance
        if est and hasattr(est, 'is_fitted_'):
            # Modelled estimators (Bayesian, Spatial, Regression) MUST be cross-validated
            ui.update_select("form_designer", choices={"cv": "Cross-Validated (Rigorous)"}, selected="cv")
        else:
            # Stateless estimators (Frequentist) unlock the O(N) Post-Hoc math
            ui.update_select("form_designer", choices={"posthoc": "Post-Hoc (Fast)", "cv": "Cross-Validated (Rigorous)"}, selected="posthoc")

    @reactive.Effect
    @reactive.event(input.btn_run_designer)
    async def generate_optimal():
        res = prev_results.get()
        est = fitted_estimator.get()
        agg_df = shared_agg_df.get()

        if res is None or est is None or agg_df is None:
            ui.notification_show("Please calculate prevalence in the Global Burden tab first.", type="warning")
            return

        # Designers physically cannot run on Binary Trait presence/absence math!
        if res.aggregation_type != AggregationType.COMPOSITIONAL:
            ui.notification_show("Formulation Designer requires Compositional Prevalence. Please recalculate in Tab 1.", type="error")
            return

        holdout = input.form_holdout()

        if not holdout:
            ui.notification_show("Cross-Validation Stratum must be selected. Ensure your prevalence data is stratified.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                # We can instantly use the already-calculated results from Tab 1 as the Baseline!
                baseline_res.set(res)

                designer_type = input.form_designer()
                p.set(message=f"Running {designer_type.upper()} Designer on {holdout}...", value=50)
                await asyncio.sleep(0)
                
                if designer_type == 'posthoc':
                    designer = PostHocFormulationDesigner(valency=input.max_valency(), n_jobs=-1)
                    await asyncio.to_thread(designer.fit, res, loo_col=holdout)
                else:
                    designer = CVFormulationDesigner(valency=input.max_valency(), n_jobs=-1)
                    await asyncio.to_thread(designer.fit, est, agg_df, loo_col=holdout)

                optimal_formulation = designer.formulation_

                # Update the active vaccine state
                current_vaccine.set(optimal_formulation)

                # Populate the choices for the custom drag-and-drop box!
                all_loci = res.data[res.target].unique().tolist()
                ui.update_selectize("custom_targets", choices=all_loci)

                p.set(message="Rendering dashboards...", value=95)
                await asyncio.sleep(0)
                ui.notification_show("Optimal Formulation Designed!", type="message")
                ui.update_navset("formulation_tabs", selected="tab_form_plots")

            except Exception as e:
                ui.notification_show(f"Designer Error: {str(e)}", type="error", duration=10)

    # --- THE CUSTOM OVERRIDE LOGIC ---
    @reactive.Effect
    @reactive.event(input.btn_copy_optimal)
    def copy_to_custom():
        if vaccine := current_vaccine.get():  # type: Formulation
            ui.update_selectize("custom_targets", selected=vaccine.get_formulation())
            ui.notification_show("Copied to manual editor.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_eval_custom)
    def evaluate_custom():
        if not (targets := input.custom_targets()):
            ui.notification_show("Please select at least one custom target.", type="warning")
            return
            
        if (res := baseline_res.get()) is None:
            ui.notification_show("Please run the 'Generate Optimal Vaccine' at least once to establish baseline metrics.", type="warning")
            return

        custom_vaccine = Formulation.from_custom(list(targets), res)
        current_vaccine.set(custom_vaccine)
        ui.notification_show("Custom vaccine evaluated.", type="message")

    # --- TAB 2: RENDERING THE DASHBOARD ---

    @render.ui
    def formulation_summary():
        if (vaccine := current_vaccine.get()) is None:  # type: Optional[Formulation]
            return ui.div("Configure and generate an optimal formulation in the sidebar.", class_="text-center mt-5 text-muted fs-4")

        # The dynamic target string: e.g., "[K2] - [K1] - [K10]"
        target_str = " - ".join(f"[{t}]" for t in vaccine.get_formulation())

        return ui.layout_column_wrap(
            ui.value_box("Target Assembly", ui.h4(target_str), theme="bg-primary"),
            ui.value_box("Valency", f"{vaccine.max_valency}-valent", theme="bg-dark"),
            width=1 / 2, class_="mb-3"
        )

    dt_download_server("form_rankings", data_callable=lambda: current_vaccine.get().rankings if current_vaccine.get() else None, filename="vaccine_rankings.csv")
    dt_download_server("form_stability", data_callable=lambda: current_vaccine.get().stability_metrics.reset_index() if current_vaccine.get() else None, filename="vaccine_stability_metrics.csv")
    dt_download_server("form_history", data_callable=lambda: current_vaccine.get().permutation_history if current_vaccine.get() else None, filename="vaccine_permutation_history.csv")

    safe_plot_server("coverage_plot", data_reactive=current_vaccine, plot_type=PlotType.COMPOSITION_BAR.value)
    safe_plot_server("stability_plot", data_reactive=current_vaccine, plot_type=PlotType.STABILITY_BUMP.value)


# =====================================================================================
# MODULE 3: TARGET LOGISTICS (Tab 3)
# =====================================================================================
@module.server
def _logistics_server(input, output, session, app_state: dict):
    """Server logic for clinical trial site selection and spatial density analysis."""
    shared_df = app_state["shared_df"]

    @reactive.Effect
    def update_target_choices():
        if (df := shared_df.get()) is not None:
            # Populate based on available genotypes
            choices = df.epi.genotypes
            ui.update_selectize("target_select", choices=[""] + choices)

    @render.ui
    def logistics_content():
        if (df := shared_df.get()) is None:
            return ui.div("Please upload data in Tab 1 first.", class_="text-center mt-5 text-muted fs-4")

        if not input.target_select():
            return ui.div("Select a K-Locus to visualize spatial density.", class_="text-center mt-5 text-muted")

        return ui.card(
            ui.card_header(f"Spatial Density: {input.target_select()}"),
            ui.p("Feature coming soon: Gaussian Process surface mapping.")
        )


def main_server(input, output, session):
    app_state = {
        "shared_df": reactive.Value(None),
        "shared_dist": reactive.Value(None),
        "shared_agg_df": reactive.Value(None),
        "prev_results": reactive.Value(None),
        "fitted_estimator": reactive.Value(None),
        "pw_collections_cache": reactive.Value({}),
        "baseline_res": reactive.Value(None),
        "current_vaccine": reactive.Value(None)
    }
    _burden_server("tab_burden", app_state=app_state)
    _formulation_server("tab_formulation", app_state=app_state)
    _logistics_server("tab_logistics", app_state=app_state)

    # =====================================================================================
    # AI ASSISTANT (Context-Aware Chatbot)
    # =====================================================================================
    chat = ui.Chat(id="ai_assistant")

    @chat.on_user_submit
    async def handle_chat_message():
        # 1. Build the dynamic context from the current application state!
        context_lines = []
        if (df := app_state["shared_df"].get()) is not None:
            context_lines.append(f"- Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")
        if (res := app_state["prev_results"].get()) is not None:
            # Pass a small preview of the results to the LLM so it can answer specific questions
            context_lines.append(f"- Calculated Prevalence ({res.method}, Target: {res.target}):\n{res.data.head(5).to_string()}")
        if (vac := app_state["current_vaccine"].get()) is not None:
            context_lines.append(f"- Optimal Vaccine Formulation: {', '.join(vac.get_formulation())} (Valency {vac.max_valency})")

        sys_prompt = (
            "You are a world-class bioinformatics and epidemiology AI assistant. "
            "You are embedded directly within the 'seroepi' Shiny dashboard. "
            "Answer the user's questions based on the current state of their workspace data:\n\n"
            + "\n".join(context_lines)
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            await chat.append_message("⚠️ `OPENAI_API_KEY` environment variable is missing. Please set it to use the AI assistant.")
            return

        # 2. Query the LLM and stream the response back to the UI smoothly
        client = AsyncOpenAI(api_key=api_key)
        messages = [{"role": "system", "content": sys_prompt}] + list(chat.messages())

        try:
            # Stream=True creates that satisfying "typing" effect in the UI
            response = await client.chat.completions.create(
                model="gpt-4o",  # Standard, highly capable model
                messages=messages,
                stream=True
            )
            await chat.append_message_stream(response)
        except Exception as e:
            await chat.append_message(f"Error communicating with AI: {str(e)}")
