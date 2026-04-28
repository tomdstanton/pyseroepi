from pathlib import Path
import importlib.metadata

from shiny import ui, module
import shinyswatch
from shinywidgets import output_widget

from seroepi.constants import EstimatorType, PlotType, AggregationType


# =====================================================================================
# REUSABLE MODULES
# =====================================================================================
@module.ui
def safe_plot_ui():
    """Reusable UI module for a Plotly widget with standard error handling."""
    return output_widget("plot")

@module.ui
def dt_download_ui(title: str):
    """Reusable UI module for a DataTable with a CSV download button."""
    return ui.card(
        ui.card_header(
            ui.div(
                title,
                ui.download_button("btn_download", "Download CSV", class_="btn-sm btn-outline-primary"),
                class_="d-flex justify-content-between align-items-center w-100"
            )
        ),
        ui.output_data_frame("table")
    )

# =====================================================================================
# HOMEPAGE
# =====================================================================================
@module.ui
def _home_ui():
    readme = Path(__file__).parent.parent.parent.parent / "README.md"
    return ui.div(
        ui.card(
            ui.markdown(readme.read_text()),
            class_="p-4 shadow-sm border-0"
        ),
        class_="container mt-4",
        style="max-width: 900px;"
    )

# =====================================================================================
# MODULE 1: GLOBAL BURDEN (Tab 1)
# =====================================================================================
@module.ui
def _burden_ui():
    """UI layout for the data ingestion and global epidemiology dashboard."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Load Data 💽",
                    ui.navset_pill(
                        ui.nav_panel(
                            "Local Files 📁",
                            ui.div(
                                ui.input_file("genotype_file", "Kleborate Output (Required)", accept=[".csv"]),
                                ui.input_file("metadata_file", "Metadata (Optional)", accept=[".csv"]),
                                ui.output_ui("dynamic_meta_mapping"),
                                ui.input_file("distance_file", "Distance Matrix (Optional)", accept=[".csv"]),
                                ui.hr(),
                                ui.input_action_button("btn_process", "Load Files", class_="btn-primary w-100"),
                                class_="mt-3"
                            )
                        ),
                        ui.nav_panel(
                            "Pathogenwatch 🔭",
                            ui.div(
                                ui.p("Fetch data directly from Pathogenwatch Next.", class_="text-muted small"),
                                ui.input_password("pw_api_key", "API Key"),
                                ui.input_action_button("btn_fetch_pw", "Fetch Collections", class_="btn-secondary w-100 mb-3"),
                                ui.input_selectize("pw_collection", "Select Collection", choices=[]),
                                ui.hr(),
                                ui.input_action_button("btn_load_pw", "Load Collection", class_="btn-primary w-100"),
                                class_="mt-3"
                            )
                        ),
                        ui.nav_panel(
                            "Workspace 💾",
                            ui.div(
                                ui.p("Restore a completely processed session.", class_="text-muted small"),
                                ui.input_file("workspace_file", "Upload Workspace (.sero)", accept=[".sero"]),
                                class_="mt-3"
                            )
                        ),
                        ui.nav_panel(
                            "Kaptive-Web 🕷️",
                            ui.div(
                                ui.p("Fetch data directly from Kaptive-Web.", class_="text-muted small"),
                                ui.input_password("kw_api_key", "API Key"),
                                ui.input_action_button("btn_fetch_kw", "Fetch Results",
                                                       class_="btn-secondary w-100 mb-3"),
                                ui.input_selectize("kw_collection", "Select Results", choices=[]),
                                ui.hr(),
                                ui.input_action_button("btn_load_kw", "Load Results", class_="btn-primary w-100"),
                                class_="mt-3"
                            )
                        ),
                    ),
                    ui.hr(),
                    ui.download_button("btn_save_workspace", "Save Workspace (.sero)", class_="btn-outline-primary w-100 mb-2"),
                    ui.input_action_button("btn_clear_data", "Clear All Data", class_="btn-outline-danger w-100")
                ),
                ui.accordion_panel(
                    "Cluster Generation 🕸️",
                    ui.navset_pill(
                        ui.nav_panel(
                            "Genomic 🧬",
                            ui.div(
                                ui.p("Identify genomic clusters using a SNP distance matrix.", class_="text-muted small mb-3"),
                                ui.input_numeric("snp_threshold", "SNP Threshold", value=20),
                                ui.input_action_button("btn_calc_snp_clusters", "Calculate Clusters", class_="btn-primary w-100 mt-2"),
                                class_="mt-3"
                            )
                        ),
                        ui.nav_panel(
                            "Transmission 🏥",
                            ui.div(
                                ui.p("Identify outbreaks based on spatial and temporal proximity.", class_="text-muted small mb-3"),
                                ui.input_selectize("trans_clone_col", "Clone Column", choices=[], selected=""),
                                ui.input_numeric("trans_spatial_thr", "Spatial Dist (km)", value=10.0),
                                ui.input_numeric("trans_temporal_thr", "Temporal Dist (days)", value=20),
                                ui.input_action_button("btn_calc_trans_clusters", "Calculate Clusters", class_="btn-primary w-100 mt-2"),
                                class_="mt-3"
                            )
                        )
                    )
                ),
                ui.accordion_panel(
                    "Prevalence Aggregation 🧮",
                    ui.input_selectize("prev_target", "Target Column (Trait/Genotype)", choices=[]),
                    ui.input_radio_buttons(
                        "prev_agg_type",
                        "Aggregation Mode",
                        choices={
                            AggregationType.COMPOSITIONAL.value: "Compositional (Variant Breakdown)", 
                            AggregationType.TRAIT.value: "Trait (Binary Presence)"
                        },
                        selected=AggregationType.COMPOSITIONAL.value
                    ),
                    ui.input_selectize("prev_stratify", "Stratify By (Optional for Comp)", choices=[], multiple=True),
                    ui.input_selectize("prev_cluster", "Cluster Column (Optional)", choices=[]),
                    ui.input_text("prev_negative", "Negative Indicator", value="-"),
                    ui.input_action_button("btn_aggregate_prev", "Aggregate Data", class_="btn-primary w-100 mt-3")
                ),
                ui.accordion_panel(
                    "Prevalence Estimation 🤖",
                    ui.input_select("prev_estimator", "Estimator", choices=EstimatorType.ui_labels()),
                    ui.output_ui("estimator_params_ui"),
                    ui.output_ui("model_io_ui"),
                    ui.input_action_button("btn_estimate_prev", "Estimate Prevalence 🚀", class_="btn-success w-100")
                ),
                id="burden_accordion",
                open="Load Data 💽"
            ),
            width=350
        ),
        ui.navset_card_tab(
            ui.nav_panel("Dataset 🔍", ui.output_ui("dashboard_content"), value="tab_dataset_preview"),
            ui.nav_panel("Aggregates 🧮", ui.output_ui("agg_data_content"), value="tab_aggregated_data"),
            ui.nav_panel("Estimates 📈", ui.output_ui("prev_summary_content"), value="tab_prevalence_data"),
            ui.nav_panel("Diagnostics 🩺", ui.output_ui("model_diagnostics_content"), value="tab_model_diagnostics"),
            ui.nav_panel(
                "Prevalence Plot 📊",
                ui.layout_sidebar(
                    ui.sidebar(
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
                        ui.hr(),
                        ui.h6("Export Settings", class_="mb-2"),
                        ui.input_select("plot_format", "Format", choices=["png", "pdf", "svg", "jpeg"]),
                        ui.input_numeric("plot_width", "Width (px)", value=1200),
                        ui.input_numeric("plot_height", "Height (px)", value=800),
                        ui.download_button("btn_download_plot", "Download Plot", class_="btn-outline-primary w-100"),
                        width=280
                    ),
                    safe_plot_ui("prev_plot")
                ),
                value="tab_prevalence_plot"
            ),
            ui.nav_panel(
                "Network 🕸️",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_selectize("network_color_col", "Color Nodes By", choices=[]),
                        ui.hr(),
                        ui.input_radio_buttons(
                            "network_edge_type", 
                            "Edges to Plot", 
                            choices={"snp": "Genomic (SNP)", "trans": "Transmission", "none": "None"},
                            selected="snp"
                        ),
                        ui.panel_conditional(
                            "input.network_edge_type === 'snp'",
                            ui.input_numeric("network_snp_threshold", "SNP Edge Threshold", value=20)
                        ),
                        ui.panel_conditional(
                            "input.network_edge_type === 'trans'",
                            ui.input_selectize("network_trans_col", "Transmission Cluster", choices=[])
                        ),
                        width=280
                    ),
                    output_widget("cluster_network_plot")
                ),
                value="tab_cluster_network"
            ),
            id="main_dashboard_tabs"
        )
    )


# =====================================================================================
# MODULE 2: FORMULATION ENGINE (Tab 2)
# =====================================================================================
@module.ui
def _formulation_ui():
    """UI layout for the algorithmic and manual vaccine designer."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.accordion(
                ui.accordion_panel(
                    "Algorithmic Design 💊",
                    ui.input_slider("max_valency", "Target Valency (# antigens)", min=2, max=30, value=10, step=1),
                    ui.input_selectize("form_holdout", "Cross-Validation Stratum", choices=[]),
                    ui.input_select("form_designer", "Designer Type", choices={"posthoc": "Post-Hoc (Fast)", "cv": "Cross-Validated (Rigorous)"}),
                    ui.input_action_button("btn_run_designer", "Generate Optimal Vaccine", class_="btn-success w-100 mt-3")
                ),
                ui.accordion_panel(
                    "Manual Override 🛠️",
                    ui.p("Drag to reorder, click 'x' to remove.", class_="text-muted small"),
                    ui.input_selectize(
                        "custom_targets",
                        "Custom Formulation Pipeline:",
                        choices=[],
                        multiple=True,
                        options={"plugins": ["remove_button", "drag_drop"], "placeholder": "e.g. K1, K2..."}
                    ),
                    ui.div(
                        ui.input_action_button("btn_eval_custom", "Evaluate Custom", class_="btn-warning w-50"),
                        ui.input_action_button("btn_copy_optimal", "Copy Optimal", class_="btn-outline-info w-50"),
                        class_="d-flex gap-2 mt-3"
                    )
                ),
                id="formulation_accordion",
                open="Algorithmic Design 💊"
            ),
            width=350
        ),
        ui.div(
            ui.output_ui("formulation_summary"),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Plots 📊",
                    ui.card(
                        ui.card_header("Geographical Coverage (Stacked Composition)"),
                        safe_plot_ui("coverage_plot")
                    ),
                    ui.card(
                        ui.card_header("Cross-Validation Stability Matrix"),
                        safe_plot_ui("stability_plot")
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


# =====================================================================================
# MODULE 3: TARGET LOGISTICS (Tab 3)
# =====================================================================================
@module.ui
def _logistics_ui():
    """UI layout for deep-diving into specific targets for clinical trials."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Target Selection"),
            ui.input_selectize(
                "target_select",
                "Select a target:",
                choices=[],  # Populated on the server side
            ),
            ui.hr(),
            ui.p("Evaluate spatial density and evolutionary trajectory for clinical trial site selection.",
                 class_="text-muted"),
            width=300
        ),
        # Main Dashboard Array
        ui.output_ui("logistics_content")
    )


# =====================================================================================
# APP FOOTER (Package Metadata)
# =====================================================================================
_meta = importlib.metadata.metadata("seroepi")
__version__ = _meta.get("Version", "dev")

# Dynamically extract author info mapped from pyproject.toml
_author_raw = _meta.get("Author-email", "Tom Stanton <tomdstanton@gmail.com>")
__author_name = _author_raw.split("<")[0].strip() if "<" in _author_raw else _author_raw
__author_email = _author_raw.split("<")[1].replace(">", "").strip() if "<" in _author_raw else ""
# Extract GitHub/Repository URL
__github_url = "https://github.com/tsta0015/seroepi"
if _meta.get_all("Project-URL"):
    for url_str in _meta.get_all("Project-URL"):
        if any(kw in url_str for kw in ["Repository", "Source", "GitHub"]):
            __github_url = url_str.split(",")[1].strip()
            break

app_footer = ui.div(
    ui.HTML(f"""
        <div style="text-align: center; font-size: 0.85rem; color: #64748B; border-top: 1px solid #1E293B; padding-top: 15px; padding-bottom: 15px; margin-top: auto;">
            <strong>seroepi</strong> v{__version__} &bull; 
            Built by <a href="mailto:{__author_email}" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-envelope-fill"></i> {__author_name}</a> &bull; 
            <a href="{__github_url}" target="_blank" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-github"></i> GitHub</a> &bull; 
            <a href="https://pypi.org/project/seroepi/" target="_blank" style="color: #0EA5E9; text-decoration: none;"><i class="bi bi-box-seam-fill"></i> PyPI</a>
        </div>
    """),
    class_="container-fluid d-flex flex-column justify-content-end"
)

# =====================================================================================
# THE MASTER ROUTER (Main App Assembly)
# =====================================================================================
main_ui = ui.page_navbar(
    # Notice we pass a unique ID string to each module function!
    ui.nav_panel("Home 🏠", _home_ui("tab_home")),
    ui.nav_panel("Global Burden 🦠", _burden_ui("tab_burden")),
    ui.nav_panel("Formulation Engine 💉", _formulation_ui("tab_formulation")),
    ui.nav_panel("Target Logistics 🌍", _logistics_ui("tab_logistics")),

    ui.nav_spacer(),  # Pushes everything after this to the right side of the navbar
    ui.nav_control(
        ui.tags.button(
            "Ask AI 🤖", 
            class_="btn btn-sm btn-outline-info mt-1", 
            type="button", 
            **{"data-bs-toggle": "offcanvas", "data-bs-target": "#chat_offcanvas"}
        )
    ),
    ui.nav_control(ui.input_dark_mode(id="dark_mode")),

    theme=shinyswatch.theme.pulse(),
    title="seroepi",
    id="main_nav",
    window_title="seroepi",
    footer=ui.TagList(
        app_footer,
        ui.div(
            ui.div(
                ui.h5("SeroEpi Assistant 🤖", class_="offcanvas-title"),
                ui.tags.button(type="button", class_="btn-close text-reset", **{"data-bs-dismiss": "offcanvas"}),
                class_="offcanvas-header"
            ),
            ui.div(
                ui.chat_ui("ai_assistant"),
                class_="offcanvas-body pb-3"
            ),
            class_="offcanvas offcanvas-end",
            tabindex="-1",
            id="chat_offcanvas",
            style="width: 500px;",
            **{"data-bs-scroll": "true", "data-bs-backdrop": "false"}  # Allows interacting with dashboard while open!
        )
    ),
    header=ui.tags.head(
        # 1. This unlocks the drag-and-drop Selectize plugin
        ui.tags.script(src="https://code.jquery.com/ui/1.13.2/jquery-ui.min.js"),

        # 1b. Load Bootstrap Icons for the footer
        ui.tags.link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"),

    # 2. Custom JS handler to dynamically show/hide navigation tabs
    ui.tags.script("""
        $(function() {
            if (window.Shiny) {
                Shiny.addCustomMessageHandler("toggle_tab", function(msg) {
                    var link = document.querySelector('a[data-value="' + msg.tab + '"]');
                    if (link) {
                        link.parentElement.style.display = msg.show ? '' : 'none';
                    }
                });
            }
        });
    """),

    # 3. Your existing custom CSS
        ui.tags.style("""
            /* Premium spacing for inputs */
            .shiny-input-container { margin-bottom: 15px; }

            /* Clean typography for the navigation bar */
            .nav-link { font-weight: 500; font-family: 'Inter', sans-serif; letter-spacing: 0.5px; }

            /* Subtle glow effect on value boxes */
            .bslib-value-box { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); border: 1px solid #334155; }
        """)
    )
)
