from shiny import reactive, render, ui
import pandas as pd

# Import your custom modules
from pyseroepi.parsers import PathogenwatchKleborateParser
from pyseroepi.distances import Distances


def server(input, output, session):
    global_df = reactive.Value(None)

    # --- STAGE 1: DYNAMIC COLUMN MAPPING ---
    @render.ui
    def dynamic_meta_mapping():
        meta_info = input.meta_file()
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
        kleb_info = input.kleb_file()
        if not kleb_info:
            ui.notification_show("Kleborate output is required.", type="warning")
            return

        with ui.Progress(min=0, max=100) as p:
            try:
                p.set(message="Parsing Kleborate Output...", value=20)
                kleb_df = pd.read_csv(kleb_info[0]["datapath"], engine="pyarrow")
                meta_df = None
                meta_kwargs = {}

                # Check if metadata was uploaded AND if they mapped the columns
                meta_info = input.meta_file()
                if meta_info:
                    p.set(message="Parsing Metadata...", value=40)
                    meta_df = pd.read_csv(meta_info[0]["datapath"], engine="pyarrow")

                    # Dynamically build the kwargs from the UI dropdowns!
                    # The `or None` ensures we don't pass empty strings if they left a dropdown blank
                    meta_kwargs = {
                        "id_col": input.map_id() or None,
                        "date_col": input.map_date() or None,
                        "country_col": input.map_country() or None,
                        "lat_col": input.map_lat() or None,
                        "lon_col": input.map_lon() or None
                    }

                    # Optional: Add a quick guard clause to ensure they actually mapped the ID
                    if not meta_kwargs["id_col"]:
                        ui.notification_show("You must select a Sample ID column to merge metadata.", type="error")
                        return

                p.set(message="Merging Datasets...", value=60)
                df = PathogenwatchKleborateParser.parse(
                    kleb_df,
                    meta_df=meta_df,
                    meta_kwargs=meta_kwargs
                )

                # ... (Keep your SNP Distance and Clustering logic here exactly as we wrote it previously) ...

                p.set(message="Pipeline Complete!", value=100)
                global_df.set(df)
                ui.notification_show("Data successfully parsed and networked.", type="message", duration=4)

            except Exception as e:
                ui.notification_show(f"Pipeline Error: {str(e)}", type="error", duration=15)