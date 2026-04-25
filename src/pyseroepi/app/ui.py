from shiny import ui, module
import shinyswatch

from pyseroepi.constants import HoldoutStrategy


# =====================================================================================
# MODULE 1: GLOBAL BURDEN (Tab 1)
# =====================================================================================
@module.ui
def burden_ui():
    """UI layout for the data ingestion and global epidemiology dashboard."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Upload Files"),

            # 1. Required Main File
            ui.input_file("genotype_file", "1. Kleborate Output (Required)", accept=[".csv"]),

            # 2. Optional Metadata & Dynamic Mapping
            ui.input_file("metadata_file", "2. Metadata (Optional)", accept=[".csv"]),
            ui.output_ui("dynamic_meta_mapping"),  # Ghost UI expands when file drops

            # 3. Optional Genomic Distances
            ui.input_file("distance_file", "3. SNP Distance Matrix (Optional)", accept=[".csv"]),

            ui.hr(),
            ui.input_action_button("btn_process", "Process Files", class_="btn-primary w-100"),
            width=350
        ),
        # Main Dashboard Array
        ui.output_ui("dashboard_content")
    )


# =====================================================================================
# MODULE 2: FORMULATION ENGINE (Tab 2)
# =====================================================================================
@module.ui
def formulation_ui():
    """UI layout for the algorithmic and manual vaccine designer."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Algorithmic Design"),
            ui.input_slider("max_valency", "Target Valency (N)", min=2, max=15, value=6, step=1),

            ui.input_select(
                "holdout_col",
                "Cross-Validation Stratum",
                choices=HoldoutStrategy.ui_labels(),
                selected=HoldoutStrategy.COUNTRY
            ),

            ui.input_action_button("btn_run_designer", "Generate Optimal Vaccine", class_="btn-success w-100"),

            ui.hr(),

            ui.h4("Manual Override"),
            ui.p("Drag to reorder, click 'x' to remove.", class_="text-muted small"),
            ui.input_selectize(
                "custom_targets",
                "Custom Formulation Pipeline:",
                choices=[],
                multiple=True,
                options={"plugins": ["remove_button", "drag_drop"], "placeholder": "e.g. K1, K2..."}
            ),

            ui.div(
                ui.input_action_button("btn_eval_custom", "Evaluate Custom", class_="btn-warning"),
                ui.input_action_button("btn_copy_optimal", "Copy Optimal", class_="btn-outline-info"),
                class_="d-flex gap-2 mt-3"  # Flexbox for side-by-side buttons
            ),
            width=350
        ),
        # Main Dashboard Array
        ui.output_ui("designer_content")
    )


# =====================================================================================
# MODULE 3: TARGET LOGISTICS (Tab 3)
# =====================================================================================
@module.ui
def logistics_ui():
    """UI layout for deep-diving into specific targets for clinical trials."""
    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Target Selection"),
            ui.input_selectize(
                "target_select",
                "Select a K-Locus:",
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
# THE MASTER ROUTER (Main App Assembly)
# =====================================================================================
app_ui = ui.page_navbar(
    # Notice we pass a unique ID string to each module function!
    ui.nav_panel("1. Global Burden", burden_ui("tab_burden")),
    ui.nav_panel("2. Formulation Engine", formulation_ui("tab_formulation")),
    ui.nav_panel("3. Target Logistics", logistics_ui("tab_logistics")),

    theme=shinyswatch.theme.slate(),
    title="pyseroepi App",
    id="main_nav",
    window_title="pyseroepi App",
    header=ui.tags.style("""
        /* Premium spacing for inputs */
        .shiny-input-container { margin-bottom: 15px; }

        /* Clean typography for the navigation bar */
        .nav-link { font-weight: 500; font-family: 'Inter', sans-serif; letter-spacing: 0.5px; }

        /* Subtle glow effect on value boxes */
        .bslib-value-box { box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); border: 1px solid #334155; }
    """)
)

# --- TEMPORARY PREVIEW CODE ---
# Delete this before you publish the package!
from shiny import App
app = App(app_ui, lambda input, output, session: None)