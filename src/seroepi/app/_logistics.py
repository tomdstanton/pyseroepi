from shiny import ui, module, reactive, render
from shinywidgets import output_widget, render_widget
from plotly.graph_objs import Figure
from asyncio import to_thread, sleep

from seroepi.constants import PlotType, Domain
from seroepi import estimators
from seroepi.app._utils import safe_plot_ui, safe_plot_server, ui_task, build_grouped_choices
from seroepi.plotting import render_plot


@module.ui
def logistics_ui():
    """UI layout for deep-diving into specific traits for clinical trials."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Vaccine Registry 💉",
                    ui.tooltip(ui.input_select("logistics_active_vac", "Target Vaccine Formulation", choices=["Awaiting Data..."]),
                               "Select a previously designed vaccine formulation from the Formulation Engine tab.")
                ),
                ui.accordion_panel(
                    "Coverage Settings 📊",
                    ui.tooltip(ui.input_selectize("logistics_stratify", "Stratify General Coverage By:", choices=[""]),
                               "Select a variable to group the population coverage by.")
                ),
                ui.accordion_panel(
                    "Longevity Forecasting 🔮",
                    ui.p("Forecast the evolutionary trajectory of the targeted traits to predict vaccine lifespan.", class_="small text-muted mb-3"),
                    ui.input_numeric("longevity_horizon", "Forecast Horizon (Months)", value=12, min=1, max=60),
                    ui.input_action_button("btn_run_longevity", "Forecast Longevity", class_="btn-success w-100 mt-2")
                ),
                id="logistics_accordion",
                open=["Vaccine Registry 💉", "Coverage Settings 📊"],
                multiple=True
            ),
            width=350
        ),
        # Main Dashboard Array
        ui.navset_card_tab(
            ui.nav_panel("General Coverage 📊", safe_plot_ui("logistics_general_plot"), value="tab_logistics_general"),
            ui.nav_panel("Spatial Coverage 🌍", output_widget("logistics_spatial_plot"), value="tab_logistics_spatial"),
            ui.nav_panel("Historical Coverage 📈", safe_plot_ui("logistics_temporal_plot"), value="tab_logistics_temporal"),
            ui.nav_panel("Longevity Forecast 🔮", output_widget("logistics_longevity_plot"), value="tab_logistics_longevity"),
            id="logistics_tabs"
        )
    )


@module.server
def logistics_server(input, output, session, app_state: dict):
    """Server logic for clinical trial site selection and spatial density analysis."""
    shared_df = app_state["shared_df"]
    vaccine_registry = app_state.setdefault("vaccine_registry", reactive.Value({}))
    shared_forecast = reactive.Value(None)

    @reactive.Calc
    def active_vac():
        reg = vaccine_registry.get()
        vac_name = input.logistics_active_vac()
        if reg is not None and vac_name and vac_name in reg:
            return reg[vac_name]
        return None

    @reactive.Effect
    def update_vac_dropdown():
        reg = vaccine_registry.get()
        if not reg:
            ui.update_select("logistics_active_vac", choices=["Awaiting Data..."])
            return
            
        choices = list(reg.keys())
        current = input.logistics_active_vac()
        selected = current if current in choices else choices[-1]
        ui.update_select("logistics_active_vac", choices=choices, selected=selected)

        df = shared_df.get()
        if df is not None:
            strat_choices = {"": "Global (No Stratification)"}
            strat_choices.update(build_grouped_choices(df.epi.stratify_cols, "Other Variables"))
            ui.update_selectize("logistics_stratify", choices=strat_choices)

    @reactive.Effect
    async def manage_tabs():
        df = shared_df.get()
        vac = active_vac()
        has_vac = bool(vac is not None)
        has_spatial = bool(df is not None and df.epi.has_spatial)
        has_temporal = bool(df is not None and df.epi.has_temporal)
        has_forecast = bool(shared_forecast.get() is not None)

        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_general", "show": has_vac})
        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_spatial", "show": has_vac and has_spatial})
        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_temporal", "show": has_vac and has_temporal})
        await session.send_custom_message("toggle_tab", {"tab": "tab_logistics_longevity", "show": has_vac and has_temporal and has_forecast})

    @reactive.Calc
    def covered_df():
        df = shared_df.get()
        vac = active_vac()
        if df is None or vac is None: return None
        
        df = df.copy()
        targets = vac.get_formulation()
        
        # Assign binary coverage trait
        df['Vaccine_Coverage'] = False
        valid_mask = df[vac.trait].notna()
        df.loc[valid_mask, 'Vaccine_Coverage'] = df.loc[valid_mask, vac.trait].isin(targets)
        return df

    @reactive.Calc
    def general_coverage_res():
        df = covered_df()
        strat_col = input.logistics_stratify()
        if df is None: return None
        
        strat_list = [strat_col] if strat_col else []
        agg_df = df.epi.aggregate_prevalence(stratify_by=strat_list, trait_col='Vaccine_Coverage')
        est = estimators.FrequentistPrevalenceEstimator()
        return est.calculate(agg_df)

    @reactive.Calc
    def spatial_coverage_res():
        df = covered_df()
        if df is None or not df.epi.has_spatial: return None
        
        spatial_cols = df.filter(regex=f"^{Domain.SPATIAL.value}_(?!res_)").columns
        if not len(spatial_cols): return None
        spatial_col = spatial_cols[0]
        
        agg_df = df.epi.aggregate_prevalence(stratify_by=[spatial_col], trait_col='Vaccine_Coverage')
        est = estimators.FrequentistPrevalenceEstimator()
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

        est = estimators.RegressionIncidenceEstimator(use_relative_incidence=False)
        return est.calculate(inc_df)

    @reactive.Effect
    @reactive.event(input.btn_run_longevity)
    async def run_longevity():
        df = shared_df.get()
        vac = active_vac()
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

            p.set(message="Running Bayesian BSTS Forecast...", value=40)
            await sleep(0)
            def fit_predict():
                est = estimators.BayesianIncidenceEstimator(forecast_horizon=input.longevity_horizon())
                return est.calculate(inc_df)
            
            forecast_res = await to_thread(fit_predict)
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
        vac = active_vac()
        forecast = shared_forecast.get()
        empty_theme = dict(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#94A3B8'))
        if vac is None or forecast is None: 
            return Figure().update_layout(title="Run forecasting first.", **empty_theme)
        return render_plot(vac, PlotType.LONGEVITY, forecast=forecast)
