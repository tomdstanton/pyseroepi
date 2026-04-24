from shiny import ui
import shinyswatch

app_ui = ui.page_navbar(
    shinyswatch.theme.slate(),

    ui.nav_panel(
        "1. Global Burden",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Pipeline Inputs"),

                ui.input_file("kleb_file", "1. Kleborate Output (Required)", accept=[".csv"]),

                ui.input_file("meta_file", "2. Metadata (Optional)", accept=[".csv"]),

                # --- THE GHOST UI ---
                # This populates instantly when meta_file is uploaded
                ui.output_ui("dynamic_meta_mapping"),

                ui.input_file("snp_file", "3. SNP Distance Matrix (Optional)", accept=[".csv"]),

                ui.hr(),
                ui.input_action_button("btn_process", "Run Pipeline", class_="btn-primary w-100"),
                width=350
            ),
            ui.output_ui("burden_dashboard")
        )
    ),
    title="Pyseroepi | Vaccine Designer",
    id="main_nav",
)