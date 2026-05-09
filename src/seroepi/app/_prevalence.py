import inspect
from pathlib import Path
from typing import get_origin, Literal, get_args
from dataclasses import fields
from shiny import ui, module, reactive, render
from asyncio import to_thread, sleep

from seroepi import estimators
from seroepi.app._utils import dt_download_server, dt_download_ui, format_metadata_ui, safe_plot_server, safe_plot_ui, \
    ui_task, generate_temp_download, _clean_ui_label, build_grouped_choices
from seroepi.constants import PlotType, EstimatorType, AggregationType, BayesianInferenceMethod
from seroepi.plotting import render_plot


@module.ui
def prevalence_ui():
    """UI layout for the global epidemiology dashboard."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Prevalence Aggregation 🧮",
                    ui.tooltip(ui.input_selectize("prev_trait", "Trait Column (Required)", choices=[]),
                               "The specific genetic trait to measure. This field is required to aggregate your data."),
                    ui.tooltip(
                        ui.input_radio_buttons(
                            "prev_agg_type",
                            "Aggregation Mode",
                            choices={
                                AggregationType.COMPOSITIONAL.value: "Compositional 🎵🎶",
                                AggregationType.TRAIT.value: "Trait ➕➖"
                            },
                            selected=AggregationType.COMPOSITIONAL.value
                        ),
                        "Compositional calculates the proportion of variants within a group. "
                        "Trait calculates the simple presence/absence of a specific marker."
                    ),
                    ui.tooltip(ui.input_selectize("prev_stratify", "Stratify By", choices=[], multiple=True,
                                                  options={"placeholder": "Optional, none selected..."}),
                               "Variables to group the data by (e.g., Spatial, Date) before calculating prevalence."),
                    ui.tooltip(ui.input_selectize("prev_cluster", "Cluster Column (Optional)", choices=[],
                                                  options={"placeholder": "Optional, none selected..."}),
                               "A cluster variable (e.g., Transmission Cluster, Hospital) to adjust the "
                               "prevalence estimates for sampling bias/outbreaks."),
                    ui.tooltip(ui.input_text("prev_negative", "Negative Indicator", value="-"),
                               "The string or character in your data that indicates a trait is absent (commonly '-' or '0')."),
                    ui.tooltip(
                        ui.input_checkbox("prev_pad_zeros", "Pad Zeroes (Zero-fill missing)", value=False),
                        "Pads missing combinations of strata with zero counts. Essential for Spatial and Hierarchical Bayesian models to map empty regions correctly."
                    ),
                    ui.input_action_button("btn_aggregate_prev", "Aggregate Data", class_="btn-primary w-100 mt-3")
                ),
                ui.accordion_panel(
                    "Prevalence Estimation 📈",
                    ui.tooltip(ui.input_select("prev_estimator", "Estimator", choices=EstimatorType.ui_labels()),
                               "The statistical model used to calculate prevalence and confidence intervals."),
                    ui.output_ui("estimator_params_ui"),
                    ui.output_ui("model_io_ui"),
                    ui.input_action_button("btn_estimate_prev", "Estimate Prevalence 🚀", class_="btn-success w-100")
                ),
                id="prevalence_accordion",
                open=["Prevalence Aggregation 🧮"], multiple=True
            ),
            width=350
        ),
        ui.navset_card_tab(
            ui.nav_panel("Aggregates 🧮", ui.output_ui("agg_data_content"), value="tab_aggregated_data"),
            ui.nav_panel("Estimates 📈", ui.output_ui("prev_summary_content"), value="tab_prevalence_data"),
            ui.nav_panel("Diagnostics 🩺", ui.output_ui("model_diagnostics_content"), value="tab_model_diagnostics"),
            ui.nav_panel(
                "Prevalence Plot 📊",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.tooltip(
                            ui.input_select(
                                "prev_plot_type",
                                "Plot Type",
                                choices={
                                    PlotType.FOREST.value: "Forest Plot",
                                    PlotType.COMPOSITION_BAR.value: "Composition Bar",
                                    PlotType.COMPOSITION_HEATMAP.value: "Composition Heatmap",
                                    PlotType.LONGITUDINAL_PREVALENCE.value: "Longitudinal Prevalence"
                                }
                            ),
                            "The visualization style for the estimated prevalence data."
                        ),
                        ui.hr(),
                        ui.h6("Export Settings", class_="mb-2"),
                        ui.tooltip(ui.input_select("plot_format", "Format", choices=["png", "pdf", "svg", "jpeg"]),
                                   "The image format for the exported plot."),
                        ui.tooltip(ui.input_numeric("plot_width", "Width (px)", value=1200),
                                   "Exported image width in pixels."),
                        ui.tooltip(ui.input_numeric("plot_height", "Height (px)", value=800),
                                   "Exported image height in pixels."),
                        ui.download_button("btn_download_plot", "Download Plot", class_="btn-outline-primary w-100"),
                        width=280
                    ),
                    safe_plot_ui("prev_plot")
                ),
                value="tab_prevalence_plot"
            ),
            id="main_dashboard_tabs"
        )
    )


@module.server
def prevalence_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    prev_results = app_state["prev_results"]
    fitted_estimator = app_state["fitted_estimator"]
    shared_agg_df = app_state.setdefault("shared_agg_df", reactive.Value(None))

    # --- STAGE 0: DYNAMIC TAB VISIBILITY ---
    @reactive.Effect
    async def manage_tab_visibility():
        try:
            est = fitted_estimator.get()
            tabs = [
                ("tab_aggregated_data", shared_agg_df.get() is not None),
                ("tab_prevalence_data", prev_results.get() is not None),
                ("tab_prevalence_plot", prev_results.get() is not None),
                ("tab_model_diagnostics", est is not None and hasattr(est, 'diagnostics'))
            ]
            for tab, show in tabs:
                await session.send_custom_message("toggle_tab", {"tab": tab, "show": show})
        except Exception as e:
            print(f"Error in manage_tab_visibility: {e}")

    @reactive.Effect
    def manage_accordion_state():
        if shared_df.get() is None:
            ui.update_accordion("prevalence_accordion", show="Prevalence Aggregation 🧮")

    # --- STAGE 3: PREVALENCE ANALYSIS WORKFLOW ---
    @reactive.Effect
    def update_prev_dropdowns():
        try:
            if (df := shared_df.get()) is not None:
                trait_choices = {"": "Select..."}
                trait_choices.update(build_grouped_choices(df.epi.genotypes, "Other Traits"))

                stratify_choices = {}
                if df.epi.has_spatial:
                    stratify_choices["Spatial Coordinates 📍"] = {"latitude": "Latitude", "longitude": "Longitude"}
                stratify_choices.update(build_grouped_choices(df.epi.stratify_cols, "Other Variables"))

                cluster_choices = {"": "Select..."}
                all_cluster_cols = list(dict.fromkeys(df.epi.cluster_cols + df.epi.genotypes))
                cluster_choices.update(build_grouped_choices(all_cluster_cols, "Clusters 🕸️"))

                ui.update_selectize("prev_trait", choices=trait_choices, selected="")
                ui.update_selectize("prev_stratify", choices=stratify_choices)
                ui.update_selectize("prev_cluster", choices=cluster_choices, selected="")
        except Exception as e:
            print(f"Error updating prev dropdowns: {e}")

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
                elements.append(
                    ui.input_select(input_id, label, choices=BayesianInferenceMethod.ui_labels(), selected=default_val))

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
    def model_download():
        est = fitted_estimator.get()
        if est and getattr(est, 'is_fitted_', False):
            return generate_temp_download(est.save_model, ".pkl", "Model Export Error")

    @reactive.Effect
    @reactive.event(input.btn_aggregate_prev)
    async def aggregate_prevalence():
        if (df := shared_df.get()) is None:
            ui.notification_show("Please upload and process data first.", type="warning")
            return

        trait = input.prev_trait() or None
        cluster = input.prev_cluster() or None
        agg_mode = input.prev_agg_type()

        if not trait:
            ui.notification_show("Please select a Trait Column before aggregating.", type="warning")
            return

        stratify = list(input.prev_stratify())

        if agg_mode == AggregationType.TRAIT and not stratify:
            ui.notification_show("Trait prevalence requires at least one stratification column.", type="error")
            return

        # INTENT ROUTING:
        # If the user wants a compositional breakdown, the trait column acts as the primary grouping variable!
        if agg_mode == AggregationType.COMPOSITIONAL and trait:
            if trait not in stratify:
                stratify.append(trait)
            trait_col = None  # Leave None so the accessor groups compositionally by the last strata
        else:
            trait_col = trait

        async with ui_task("Aggregation Error") as p:
            p.set(message="Aggregating data...", value=20)
            await sleep(0)

            def run_aggregation():
                return df.epi.aggregate_prevalence(
                    stratify_by=stratify,
                    trait_col=trait_col,
                    cluster_col=cluster,
                    negative_indicator=input.prev_negative(),
                    pad_zeros=input.prev_pad_zeros()
                )

            agg_df = await to_thread(run_aggregation)

            shared_agg_df.set(agg_df)
            p.set(message="Done!", value=100)
            ui.notification_show("Data aggregated successfully!", type="message")
            ui.update_accordion("prevalence_accordion", show="Prevalence Estimation 📈")
            ui.update_navset("main_dashboard_tabs", selected="tab_aggregated_data")

    @reactive.Effect
    @reactive.event(input.btn_estimate_prev)
    async def estimate_prevalence():
        if (agg_df := shared_agg_df.get()) is None:
            ui.notification_show("Please aggregate data first.", type="warning")
            return

        est_type = input.prev_estimator()

        async with ui_task("Calculation Error") as p:
            p.set(message="Instantiating estimator...", value=20)
            await sleep(0)

            estimator_class_name = EstimatorType(est_type).class_name

            if not hasattr(estimators, estimator_class_name):
                ui.notification_show(f"{estimator_class_name} is not available. Did you install seroepi[models]?",
                                     type="error")
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
                await sleep(0)
                try:
                    est = EstimatorClass.load_model(Path(file_info[0]["datapath"]))
                except Exception as e:
                    ui.notification_show(f"Failed to load model: {str(e)}", type="error", duration=10)
                    return

                p.set(message="Predicting prevalence...", value=60)
                await sleep(0)
                res = await to_thread(est.predict, agg_df)
            elif can_reuse_memory:
                p.set(message="Using in-memory fitted model...", value=60)
                await sleep(0)
                est = in_memory_est
                res = await to_thread(est.predict, agg_df)
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

                        if param.annotation is int:
                            kwargs[name] = int(val)
                        elif param.annotation is float:
                            kwargs[name] = float(val)
                        elif param.annotation is bool:
                            kwargs[name] = bool(val)
                        elif origin is Literal:
                            kwargs[name] = val
                        elif 'InferenceMethod' in str(param.annotation):
                            kwargs[name] = BayesianInferenceMethod(val)
                        else:
                            kwargs[name] = val

                # Ensure SpatialPrevalenceEstimator maps correctly to Pandas generated columns
                if estimator_class_name == "SpatialPrevalenceEstimator":
                    kwargs['lat_col'] = 'latitude'
                    kwargs['lon_col'] = 'longitude'

                est = EstimatorClass(**kwargs)

                p.set(message="Fitting model and calculating...", value=60)
                await sleep(0)
                res = await to_thread(est.calculate, agg_df)

            fitted_estimator.set(est)
            prev_results.set(res)

            # Cache the run dynamically so it can be picked up by the Formulation tab!
            registry = app_state.setdefault("results_registry", reactive.Value({})).get().copy()
            trait_clean = _clean_ui_label(res.trait)
            strata_str = ", ".join([_clean_ui_label(s) for s in res.stratified_by]) if res.stratified_by else "Global"
            current_df = shared_df.get()
            ds_name = current_df.attrs.get("dataset_name",
                                           "Unknown Dataset") if current_df is not None else "Unknown Dataset"
            run_name = f"[{ds_name}] {trait_clean} by {strata_str} ({EstimatorType(est_type).name.title()})"
            registry[run_name] = {"res": res, "est": est, "agg_df": agg_df}
            app_state["results_registry"].set(registry)
            app_state["active_run_name"].set(run_name)

            ui.notification_show("Prevalence calculated successfully! Check the plot tab.", type="message")
            p.set(message="Done", value=100)
            ui.update_navset("main_dashboard_tabs", selected="tab_prevalence_plot")
            await sleep(0)

    # --- STAGE 4: RENDERING THE DASHBOARDS ---
    @render.ui
    def agg_data_content():
        if (agg_df := shared_agg_df.get()) is None:
            return ui.div("Please aggregate data in the sidebar to view the aggregated dataset.",
                          class_="text-center mt-5 text-muted fs-4")

        meta_dict = agg_df.attrs.get('metric_meta', {})
        meta_ui = format_metadata_ui(meta_dict)

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
        meta_dict = {f.name: getattr(res, f.name) for f in fields(res) if
                     f.name not in ['data', 'model_results']}
        meta_ui = format_metadata_ui(meta_dict)

        return ui.div(
            ui.card(ui.card_header("Estimates Metadata"), ui.div(*meta_ui, class_="p-2")),
            dt_download_ui("prev_summary", "Estimates Data")
        )

    dt_download_server("prev_summary", data_callable=lambda: prev_results.get().data if prev_results.get() else None,
                       filename="estimates_data.csv", height="400px")

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

    dt_download_server("model_diagnostics", data_callable=get_model_diagnostics, filename="model_diagnostics.csv",
                       height="500px")

    safe_plot_server("prev_plot", data_reactive=prev_results, plot_type=input.prev_plot_type)

    @render.download(filename=lambda: f"prevalence_plot.{input.plot_format()}")
    def btn_download_plot():
        res = prev_results.get()
        if res is None:
            ui.notification_show("No plot data available to export.", type="warning")
            return

        # Generate the raw Plotly Figure (go.Figure) from the router
        fig = render_plot(res, input.prev_plot_type())

        # Kaleido naturally intercepts write_image locally
        def save_fig(p: Path):
            fig.write_image(p, format=input.plot_format(), width=input.plot_width(), height=input.plot_height())

        return generate_temp_download(save_fig, f".{input.plot_format()}", "Plot Export Error")