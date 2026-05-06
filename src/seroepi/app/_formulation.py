from asyncio import sleep, get_running_loop, to_thread
from typing import Optional
from pathlib import Path

from shiny import ui, module, reactive, render

from seroepi.app._utils import safe_plot_ui, dt_download_ui, dt_download_server, safe_plot_server, ui_task, generate_temp_download
from seroepi.constants import PlotType, AggregationType
from seroepi.formulation import Formulation, CVFormulationDesigner, PostHocFormulationDesigner
from seroepi.plotting import render_plot


@module.ui
def formulation_ui():
    """UI layout for the algorithmic and manual vaccine designer."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Prevalence Registry 🎯",
                    ui.tooltip(ui.input_select("form_active_run", "Prevalence Run", choices=["Awaiting Data..."]),
                               "Select a previously calculated prevalence model from the Global Burden tab.")
                ),
                ui.accordion_panel(
                    "Algorithmic Design 💊",
                    ui.tooltip(ui.input_slider("max_valency", "Trait Valency",
                                               min=2, max=30, value=10, step=1),
                               "The maximum number of distinct variants to include in the optimal vaccine formulation."),
                    ui.tooltip(ui.input_selectize("form_holdout", "Cross-Validation Stratum", choices=["Awaiting Data..."]),
                               "The spatial or demographic level used to evaluate stability via Leave-One-Out Cross-Validation."),
                    ui.tooltip(ui.input_select("form_designer", "Designer Type",
                                               choices={"posthoc": "Post-Hoc (Fast)", "cv": "Cross-Validated (Rigorous)"}),
                               "Post-Hoc is faster but assumes independence. Cross-Validated re-fits the model iteratively for rigorous stability testing."),
                    ui.input_action_button("btn_run_designer", "Generate Optimal Vaccine", class_="btn-success w-100 mt-3")
                ),
                ui.accordion_panel(
                    "Manual Override 🛠️",
                    ui.p("Drag to reorder, click 'x' to remove.", class_="text-muted small"),
                    ui.tooltip(
                        ui.input_selectize(
                            "custom_traits",
                            "Custom Formulation Pipeline:",
                            choices=[],
                            multiple=True,
                            options={"plugins": ["remove_button", "drag_drop"], "placeholder": "e.g. K1, K2..."}
                        ),
                        "Drag to reorder, or manually type and add variants to evaluate a custom vaccine formulation."
                    ),
                    ui.div(
                        ui.input_action_button("btn_eval_custom", "Evaluate Custom", class_="btn-warning w-50"),
                        ui.input_action_button("btn_copy_optimal", "Copy Optimal", class_="btn-outline-info w-50"),
                        class_="d-flex gap-2 mt-3"
                    )
                ),
                id="formulation_accordion",
                open=["Prevalence Registry 🎯", "Algorithmic Design 💊"], multiple=True
            ),
            width=350
        ),
        ui.div(
            ui.output_ui("formulation_summary"),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Plots 📊",
                    ui.layout_sidebar(
                        ui.sidebar(
                            ui.h6("Export Settings", class_="mb-2"),
                            ui.tooltip(ui.input_select("form_plot_format", "Format", choices=["png", "pdf", "svg", "jpeg"]),
                                       "The image format for the exported plot."),
                            ui.tooltip(ui.input_numeric("form_plot_width", "Width (px)", value=1200), "Exported image width in pixels."),
                            ui.tooltip(ui.input_numeric("form_plot_height", "Height (px)", value=800), "Exported image height in pixels."),
                            ui.download_button("btn_download_coverage", "Download Coverage Plot", class_="btn-outline-primary w-100 mb-2"),
                            ui.output_ui("dl_stability_btn_ui"),
                            width=280
                        ),
                        ui.div(
                            ui.card(ui.card_header("Vaccine Coverage (Cumulative)"), safe_plot_ui("coverage_plot")),
                            ui.output_ui("stability_plot_card")
                        )
                    ),
                    value="tab_form_plots"
                ),
                ui.nav_panel(
                    "Rankings 🏆",
                    dt_download_ui("form_rankings", "Formulation Rankings"),
                    value="tab_form_rankings"
                ),
                ui.nav_panel(
                    "Stability Metrics ⚖️",
                    dt_download_ui("form_stability", "LOO Stability Metrics"),
                    value="tab_form_stability"
                ),
                ui.nav_panel(
                    "Permutations 🔄",
                    dt_download_ui("form_history", "Permutation History"),
                    value="tab_form_permutations"
                ),
                id="formulation_tabs"
            )
        )
    )


@module.server
def formulation_server(input, output, session, app_state: dict):
    shared_df = app_state["shared_df"]
    baseline_res = app_state["baseline_res"]
    current_vaccine = app_state["current_vaccine"]
    shared_agg_df = app_state["shared_agg_df"]
    prev_results = app_state["prev_results"]
    fitted_estimator = app_state["fitted_estimator"]
    results_registry = app_state.setdefault("results_registry", reactive.Value({}))
    vaccine_registry = app_state.setdefault("vaccine_registry", reactive.Value({}))

    @reactive.Calc
    def active_run():
        reg = results_registry.get()
        run_name = input.form_active_run()
        if reg is not None and run_name and run_name in reg:
            return reg[run_name]
        return None

    @reactive.Effect
    def update_run_dropdown():
        reg = results_registry.get()
        if not reg:
            ui.update_select("form_active_run", choices=["Awaiting Data..."])
            return
            
        choices = list(reg.keys())
        # Safely default to the most recently added run
        current = input.form_active_run()
        selected = current if current in choices else choices[-1]
        ui.update_select("form_active_run", choices=choices, selected=selected)

    @reactive.Effect
    async def manage_form_tabs():
        vac = current_vaccine.get()
        has_vac = vac is not None
        has_stability = has_vac and not vac.stability_metrics.empty
        
        for tab, show in [
            ("tab_form_plots", has_vac),
            ("tab_form_rankings", has_vac),
            ("tab_form_stability", has_stability),
            ("tab_form_permutations", has_stability)
        ]:
            await session.send_custom_message("toggle_tab", {"tab": tab, "show": show})

    @reactive.Effect
    def update_form_inputs():
        run = active_run()
        res = run["res"] if run else None
        est = run["est"] if run else None

        if res is None or est is None:
            ui.update_selectize("form_holdout", choices=["Awaiting Data..."])
            ui.update_select("form_designer", choices={"cv": "Cross-Validated (Rigorous)"})
            return

        # 1. Populate the Stratum dropdown exactly from the prior calculation metadata
        holdouts = res.stratified_by
        if holdouts:
            ui.update_selectize("form_holdout", choices=holdouts, selected=holdouts[0])
        else:
            ui.update_selectize("form_holdout", choices=["Global (CV Not Possible)"],
                                selected="Global (CV Not Possible)")

        # 2. Restrict Designer Type based on the exact estimator instance
        est_name = type(est).__name__.replace('PrevalenceEstimator', '')

        if est and hasattr(est, 'is_fitted_'):
            # Modelled estimators (Bayesian, Spatial, Regression) MUST be cross-validated
            ui.update_select("form_designer", choices={"cv": f"Cross-Validated ({est_name})"}, selected="cv")
        else:
            # Stateless estimators (Frequentist) unlock the O(N) Post-Hoc math
            ui.update_select(
                "form_designer",
                choices={"posthoc": f"Post-Hoc ({est_name})", "cv": f"Cross-Validated ({est_name})"},
                selected="posthoc"
            )

    @reactive.Effect
    def update_custom_traits_choices():
        run = active_run()
        if run is not None:
            res = run["res"]
            all_traits = res.data['trait'].unique().tolist()
            ui.update_selectize("custom_traits", choices=all_traits)

    @reactive.Effect
    @reactive.event(input.btn_run_designer)
    async def generate_optimal():
        run = active_run()
        if run is None:
            ui.notification_show("Please calculate prevalence in the Global Burden tab first.", type="warning")
            return
            
        res, est, agg_df = run["res"], run["est"], run["agg_df"]

        if res.aggregation_type != AggregationType.COMPOSITIONAL:
            ui.notification_show(
                "Note: Evaluating a single binary trait. For multi-antigen vaccines, use Compositional mode in Tab 1.",
                type="message", duration=8)

        holdout = input.form_holdout()

        if not holdout or "Not Possible" in holdout or "Awaiting Data" in holdout:
            ui.notification_show(
                "Cross-Validation Stratum must be selected. Ensure your prevalence data is stratified in Tab 1.",
                type="warning")
            return

        async with ui_task("Designer Error") as p:
                # We can instantly use the already-calculated results from Tab 1 as the Baseline!
                baseline_res.set(res)

                designer_type = input.form_designer()
                p.set(message=f"Running {designer_type.upper()} Designer on {holdout}...", value=50)
                await sleep(0)

                loop = get_running_loop()

                def ui_progress_callback(current, total):
                    # Safely update the Shiny UI from the background thread
                    percentage = 50 + int(45 * (current / total))
                    message = f"Running CV Fold {current}/{total}..." if designer_type != 'posthoc' else f"Running Permutation {current}/{total}..."
                    loop.call_soon_threadsafe(p.set, percentage, message)

                if designer_type == 'posthoc':
                    designer = PostHocFormulationDesigner(valency=input.max_valency(), n_jobs=-1)
                    await to_thread(designer.fit, res, loo_col=holdout, progress_callback=ui_progress_callback)
                else:
                    designer = CVFormulationDesigner(valency=input.max_valency(), n_jobs=-1)
                    await to_thread(designer.fit, est, agg_df, loo_col=holdout,
                                            progress_callback=ui_progress_callback)

                optimal_formulation = designer.formulation_

                # Update the active vaccine state
                current_vaccine.set(optimal_formulation)

                # Cache the formulation dynamically
                registry = vaccine_registry.get().copy()
                run_name = input.form_active_run()
                vac_name = f"Optimal {optimal_formulation.max_valency}-valent ({designer_type.upper()}) | {run_name}"
                registry[vac_name] = optimal_formulation
                vaccine_registry.set(registry)

                p.set(message="Rendering dashboards...", value=95)
                await sleep(0)
                ui.notification_show("Optimal Formulation Designed!", type="message")
                ui.update_navset("formulation_tabs", selected="tab_form_plots")


    # --- THE CUSTOM OVERRIDE LOGIC ---
    @reactive.Effect
    @reactive.event(input.btn_copy_optimal)
    def copy_to_custom():
        if vaccine := current_vaccine.get():  # type: Formulation
            ui.update_selectize("custom_traits", selected=vaccine.get_formulation())
            ui.notification_show("Copied to manual editor.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_eval_custom)
    def evaluate_custom():
        if not (traits := input.custom_traits()):
            ui.notification_show("Please select at least one custom trait.", type="warning")
            return

        run = active_run()
        if run is None:
            ui.notification_show("Please select a valid prevalence run.", type="warning")
            return

        custom_vaccine = Formulation.from_custom(list(traits), run["res"])
        current_vaccine.set(custom_vaccine)
        
        # Cache the custom formulation
        registry = vaccine_registry.get().copy()
        run_name = input.form_active_run()
        vac_name = f"Custom {len(traits)}-valent | {run_name}"
        registry[vac_name] = custom_vaccine
        vaccine_registry.set(registry)

        ui.notification_show("Custom vaccine evaluated.", type="message")
        ui.update_navset("formulation_tabs", selected="tab_form_plots")

    # --- TAB 2: RENDERING THE DASHBOARD ---

    @render.ui
    def stability_plot_card():
        vac = current_vaccine.get()
        if vac is not None and not vac.stability_metrics.empty:
            return ui.card(
                ui.card_header("Cross-Validation Stability Matrix"),
                safe_plot_ui("stability_plot")
            )
        return ui.div()

    @render.ui
    def dl_stability_btn_ui():
        vac = current_vaccine.get()
        if vac is not None and not vac.stability_metrics.empty:
            return ui.download_button("btn_download_stability", "Download Stability Plot", class_="btn-outline-primary w-100")
        return ui.div()

    @reactive.Calc
    def coverage_plot_data():
        run = active_run()
        vac = current_vaccine.get()
        if run is not None and vac is not None:
            return {"res": run["res"], "formulation": vac}
        return None

    @render.download(filename=lambda: f"vaccine_coverage.{input.form_plot_format()}")
    def btn_download_coverage():
        plot_data = coverage_plot_data()
        if plot_data is None:
            return
        fig = render_plot(plot_data, PlotType.VACCINE_COVERAGE)
        def save_fig(p: Path):
            fig.write_image(p, format=input.form_plot_format(), width=input.form_plot_width(), height=input.form_plot_height())
        return generate_temp_download(save_fig, f".{input.form_plot_format()}", "Plot Export Error")

    @render.download(filename=lambda: f"vaccine_stability.{input.form_plot_format()}")
    def btn_download_stability():
        vac = current_vaccine.get()
        if vac is None or vac.stability_metrics.empty:
            return
        fig = render_plot(vac, PlotType.STABILITY_BUMP)
        def save_fig(p: Path):
            fig.write_image(p, format=input.form_plot_format(), width=input.form_plot_width(), height=input.form_plot_height())
        return generate_temp_download(save_fig, f".{input.form_plot_format()}", "Plot Export Error")

    dt_download_server("form_rankings",
                       data_callable=lambda: current_vaccine.get().rankings if current_vaccine.get() else None,
                       filename="vaccine_rankings.csv")
    dt_download_server("form_stability",
                       data_callable=lambda: current_vaccine.get().stability_metrics.reset_index() if current_vaccine.get() else None,
                       filename="vaccine_stability_metrics.csv")
    dt_download_server("form_history",
                       data_callable=lambda: current_vaccine.get().permutation_history if current_vaccine.get() else None,
                       filename="vaccine_permutation_history.csv")

    safe_plot_server("coverage_plot", data_reactive=coverage_plot_data, plot_type=PlotType.VACCINE_COVERAGE)
    safe_plot_server("stability_plot", data_reactive=current_vaccine, plot_type=PlotType.STABILITY_BUMP)
