import inspect
from pathlib import Path
from typing import get_origin, Literal
from shiny import ui, module, reactive, render
from shinywidgets import output_widget, render_widget
from plotly.graph_objs import Figure
from asyncio import to_thread, sleep

from seroepi.constants import PlotType, Domain, BayesianInferenceMethod
from seroepi import estimators
from seroepi.app._utils import safe_plot_ui, safe_plot_server, ui_task, build_grouped_choices, generate_temp_download, \
    build_estimator_params_ui
from seroepi.plotting import render_plot


@module.ui
def logistics_ui():
    """UI layout for deep-diving into specific traits for clinical trials."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Coverage Settings 📊",
                    ui.tooltip(ui.input_selectize("logistics_stratify", "Stratify General Coverage By:", choices=[""]),
                               "Select a variable to group the population coverage by.")
                ),
                ui.accordion_panel(
                    "Longevity Forecasting 🔮",
                    ui.p("Forecast the evolutionary trajectory of the targeted traits to predict vaccine lifespan.", class_="small text-muted mb-3"),
                    ui.input_select("longevity_estimator", "Estimator Model", choices={"bayesian": "Bayesian (BSTS)", "glm": "Frequentist (GLM)"}),
                    ui.output_ui("longevity_estimator_params_ui"),
                    ui.output_ui("longevity_model_io_ui"),
                    ui.input_action_button("btn_run_longevity", "Forecast Longevity", class_="btn-success w-100 mt-2")
                ),
                id="logistics_accordion",
                open=["Coverage Settings 📊"],
                multiple=True
            ),
            width=350
        ),
        # Main Dashboard Array
        ui.navset_card_tab(
            ui.nav_panel("General Coverage 📊", ui.output_ui("logistics_general_content"), value="tab_logistics_general"),
            ui.nav_panel("Spatial Coverage 🌍", ui.output_ui("logistics_spatial_content"), value="tab_logistics_spatial"),
            ui.nav_panel("Historical Coverage 📈", ui.output_ui("logistics_temporal_content"), value="tab_logistics_temporal"),
            ui.nav_panel("Longevity Forecast 🔮", ui.output_ui("logistics_longevity_content"), value="tab_logistics_longevity"),
            id="logistics_tabs"
        )
    )


@module.server
def logistics_server(input, output, session, app_state: dict):
    """Server logic for clinical trial site selection and spatial density analysis."""
    shared_df = app_state["shared_df"]
    shared_forecast = reactive.Value(None)
    current_formulation = app_state["current_formulation"]
    fitted_longevity_estimator = app_state.setdefault("fitted_longevity_estimator", reactive.Value(None))
    ESTIMATOR_MAP = {
        "bayesian": "BayesianIncidenceEstimator",
        "glm": "GLMIncidenceEstimator"
    }

    @reactive.Effect
    def update_stratify_dropdown():
        df = shared_df.get()
        if df is not None:
            strat_choices = {"": "Global (No Stratification)"}
            strat_choices.update(build_grouped_choices(df.epi.stratify_cols, "Other Variables"))
            ui.update_selectize("logistics_stratify", choices=strat_choices)

    @reactive.Effect
    async def manage_tabs():
        df = shared_df.get()
        vac = current_formulation.get()
        has_vac = bool(vac is not None)
        has_spatial = bool(df is not None and df.epi.has_spatial)
        has_temporal = bool(df is not None and df.epi.has_temporal)
        has_forecast = bool(shared_forecast.get() is not None)

        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_spatial", "show": has_vac and has_spatial})
        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_temporal", "show": has_vac and has_temporal})
        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_longevity", "show": has_vac and has_temporal and has_forecast})

    @reactive.Calc
    def covered_df():
        df = shared_df.get()
        vac = current_formulation.get()
        if df is None or vac is None: return None
        
        df = df.copy()
        targets = vac.get_formulation()
        
        # Assign binary coverage trait
        df['Vaccine_Coverage'] = False
        valid_mask = df[vac.trait].notna()
        df.loc[valid_mask, 'Vaccine_Coverage'] = df.loc[valid_mask, vac.trait].isin(targets)
        return df

    @render.ui
    def logistics_general_content():
        if current_formulation.get() is None:
            return ui.div("Please design a vaccine formulation in Tab 2 first.", class_="text-center mt-5 text-muted fs-4")
        return safe_plot_ui("logistics_general_plot")

    @render.ui
    def logistics_spatial_content():
        if current_formulation.get() is None:
            return ui.div("Please design a vaccine formulation in Tab 2 first.", class_="text-center mt-5 text-muted fs-4")
        df = shared_df.get()
        if df is None or not df.epi.has_spatial:
            return ui.div("Spatial metadata is required to view spatial coverage.", class_="text-center mt-5 text-muted fs-4")
        return output_widget("logistics_spatial_plot")

    @render.ui
    def logistics_temporal_content():
        if current_formulation.get() is None:
            return ui.div("Please design a vaccine formulation in Tab 2 first.", class_="text-center mt-5 text-muted fs-4")
        df = shared_df.get()
        if df is None or not df.epi.has_temporal:
            return ui.div("Temporal metadata is required to view historical coverage.", class_="text-center mt-5 text-muted fs-4")
        return safe_plot_ui("logistics_temporal_plot")

    @render.ui
    def logistics_longevity_content():
        if current_formulation.get() is None:
            return ui.div("Please design a vaccine formulation in Tab 2 first.", class_="text-center mt-5 text-muted fs-4")
        if shared_forecast.get() is None:
            return ui.div("Run longevity forecasting in the sidebar to view this plot.", class_="text-center mt-5 text-muted fs-4")
        return output_widget("logistics_longevity_plot")

    @reactive.Calc
    def general_coverage_res():
        df = covered_df()
        strat_col = input.logistics_stratify()
        if df is None: return None
        
        strat_list = [strat_col] if strat_col else []
        agg_df = df.epi.aggregate_prevalence(stratify_by=strat_list, trait_col='Vaccine_Coverage')
        est = estimators.UnpooledPrevalenceEstimator()
        return est.calculate(agg_df)

    @reactive.Calc
    def spatial_coverage_res():
        df = covered_df()
        if df is None or not df.epi.has_spatial: return None
        
        spatial_cols = df.filter(regex=f"^{Domain.SPATIAL.value}_(?!res_)").columns
        if not len(spatial_cols): return None
        spatial_col = spatial_cols[0]
        
        agg_df = df.epi.aggregate_prevalence(stratify_by=[spatial_col], trait_col='Vaccine_Coverage')
        est = estimators.UnpooledPrevalenceEstimator()
        return {"res": est.calculate(agg_df), "geo_col": spatial_col}

    @reactive.Calc
    def temporal_coverage_res():
        df = covered_df()
        if df is None or not df.epi.has_temporal:
            return None
        
        inc_df = df.epi.aggregate_incidence(stratify_by=[], trait_col='Vaccine_Coverage')

        # Standardize the dynamic temporal column to 'date' for the estimator backend
        temporal_cols = [c for c in inc_df.columns if str(c).startswith(f"{Domain.TEMPORAL.value}_")]
        if temporal_cols:
            inc_df = inc_df.rename(columns={temporal_cols[0]: 'date'})

        est = estimators.GLMIncidenceEstimator(use_relative_incidence=False)
        return est.calculate(inc_df)

    @render.ui
    def longevity_estimator_params_ui():
        est_key = input.longevity_estimator()
        
        if "longevity_model_upload" in input and input.longevity_model_upload():
            return ui.div()
            
        estimator_class_name = ESTIMATOR_MAP.get(est_key)
        if not estimator_class_name:
            return ui.div()
            
        EstimatorClass = getattr(estimators, estimator_class_name, None)
        if not EstimatorClass:
            return ui.div()
            
        est = fitted_longevity_estimator.get()
        if est and type(est).__name__ == estimator_class_name and getattr(est, 'is_fitted_', False):
            return ui.div(
                ui.hr(),
                ui.p(
                    "A fitted model of this type is currently in memory. "
                    "Forecasting will use this model's learned weights. "
                    "To train a new model, click 'Clear Fitted Model' below.",
                    class_="text-info small mb-1"
                )
            )
            
        return build_estimator_params_ui(
            EstimatorClass, 
            prefix="longevity_param_", 
            exclude=['self'], 
            default_overrides={'use_relative_incidence': False}
        )

    @render.ui
    def longevity_model_io_ui():
        est_key = input.longevity_estimator()
        estimator_class_name = ESTIMATOR_MAP.get(est_key)
        EstimatorClass = getattr(estimators, estimator_class_name, None)
        
        if not EstimatorClass or not hasattr(EstimatorClass, 'load_model'):
            return ui.div()
            
        elements = [
            ui.hr(),
            ui.p("Model Weights (Optional)", class_="text-muted small mb-1"),
            ui.input_file("longevity_model_upload", "Load Fitted Model (.pkl)", accept=[".pkl"])
        ]
        
        est = fitted_longevity_estimator.get()
        if est and type(est).__name__ == EstimatorClass.__name__ and getattr(est, 'is_fitted_', False):
            elements.append(ui.download_button("longevity_model_download", "Download Fitted Model", class_="btn-outline-primary w-100 mb-2"))
            elements.append(ui.input_action_button("btn_clear_longevity_model", "Clear Fitted Model", class_="btn-outline-danger w-100 mb-3"))
            
        return ui.div(*elements)

    @reactive.Effect
    @reactive.event(input.btn_clear_longevity_model)
    def clear_longevity_model():
        fitted_longevity_estimator.set(None)
        ui.notification_show("Fitted longevity model cleared from memory.", type="message")
        
    @render.download(filename=lambda: f"fitted_{input.longevity_estimator()}_incidence_model.pkl")
    def longevity_model_download():
        est = fitted_longevity_estimator.get()
        if est and getattr(est, 'is_fitted_', False):
            return generate_temp_download(est.save_model, ".pkl", "Model Export Error")

    @reactive.Effect
    @reactive.event(input.btn_run_longevity)
    async def run_longevity():
        df = shared_df.get()
        vac = current_formulation.get()
        if df is None or vac is None:
            ui.notification_show("Please select a vaccine formulation.", type="warning")
            return

        async with ui_task("Longevity Forecasting Error") as p:
            p.set(message="Aggregating incidence data...", value=20)
            await sleep(0)
            def agg_incidence():
                df_inc = df.epi.aggregate_incidence(stratify_by=[vac.trait], trait_col=None, pad_zeros=True)
                # Standardize the dynamic temporal column to 'date' for the estimator backend
                temporal_cols = [c for c in df_inc.columns if str(c).startswith(f"{Domain.TEMPORAL.value}_")]
                if temporal_cols:
                    df_inc = df_inc.rename(columns={temporal_cols[0]: 'date'})
                return df_inc
            inc_df = await to_thread(agg_incidence)

            p.set(message="Instantiating estimator...", value=40)
            await sleep(0)
            
            est_key = input.longevity_estimator()
            estimator_class_name = ESTIMATOR_MAP.get(est_key)
            EstimatorClass = getattr(estimators, estimator_class_name)
            
            in_memory_est = fitted_longevity_estimator.get()
            can_reuse_memory = (
                in_memory_est is not None
                and type(in_memory_est).__name__ == EstimatorClass.__name__
                and getattr(in_memory_est, 'is_fitted_', False)
            )
            
            file_info = input.longevity_model_upload() if "longevity_model_upload" in input else None
            
            if hasattr(EstimatorClass, 'load_model') and file_info:
                p.set(message="Loading model...", value=50)
                await sleep(0)
                try:
                    est = EstimatorClass.load_model(Path(file_info[0]["datapath"]))
                except Exception as e:
                    ui.notification_show(f"Failed to load model: {str(e)}", type="error", duration=10)
                    return
                    
                p.set(message="Forecasting longevity...", value=70)
                await sleep(0)
                forecast_res = await to_thread(est.predict, inc_df)
                
            elif can_reuse_memory:
                p.set(message="Using in-memory fitted model...", value=50)
                await sleep(0)
                est = in_memory_est
                forecast_res = await to_thread(est.predict, inc_df)
                
            else:
                kwargs = {}
                sig = inspect.signature(EstimatorClass.__init__)
                for name, param in sig.parameters.items():
                    if name in ['self']: continue
                        
                    input_id = f"longevity_param_{name}"
                    if input_id in input:
                        val = input[input_id]()
                        if val == "": continue
                            
                        origin = get_origin(param.annotation)
                        if param.annotation is int: kwargs[name] = int(val)
                        elif param.annotation is float: kwargs[name] = float(val)
                        elif param.annotation is bool: kwargs[name] = bool(val)
                        elif origin is Literal: kwargs[name] = val
                        elif 'InferenceMethod' in str(param.annotation): kwargs[name] = BayesianInferenceMethod(val)
                        else: kwargs[name] = val
                            
                est = EstimatorClass(**kwargs)
                p.set(message="Fitting model and forecasting...", value=60)
                await sleep(0)
                forecast_res = await to_thread(est.calculate, inc_df)
                
            fitted_longevity_estimator.set(est)
            shared_forecast.set(forecast_res)
            
            p.set(message="Done!", value=100)
            ui.notification_show("Longevity forecast complete!", type="message")
            ui.update_navset("logistics_tabs", selected="tab_logistics_longevity")
            await sleep(0)

    # --- Plot Rendering ---
    safe_plot_server("logistics_general_plot", data_reactive=general_coverage_res, plot_type=PlotType.FOREST)
    safe_plot_server("logistics_temporal_plot", data_reactive=temporal_coverage_res, plot_type=PlotType.EPICURVE)

    @render_widget
    def logistics_spatial_plot():
        res_dict = spatial_coverage_res()
        empty_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
        if res_dict is None: 
            return Figure().update_layout(title="No spatial data available.", **empty_theme)
        return render_plot(res_dict['res'], PlotType.CHOROPLETH, geo_col=res_dict['geo_col'])
        
    @render_widget
    def logistics_longevity_plot():
        vac = current_formulation.get()
        forecast = shared_forecast.get()
        empty_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
        if vac is None or forecast is None: 
            return Figure().update_layout(title="Run forecasting first.", **empty_theme)
        return render_plot(vac, PlotType.LONGEVITY, forecast=forecast)
