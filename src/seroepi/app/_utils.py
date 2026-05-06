from typing import Callable, Union, AsyncGenerator, Any
from shiny import ui, module, reactive, render
import pandas as pd
from asyncio import to_thread
from re import compile as re_compile
from contextlib import asynccontextmanager
import tempfile
from pathlib import Path

from shinywidgets import render_widget, output_widget

from seroepi.constants import PlotType, Domain
from seroepi.plotting import render_plot


# Functions ------------------------------------------------------------------------------------------------------------
def _clean_ui_label(text: Any) -> str:
    """Strips domain prefixes and title-cases strings for clean UI rendering."""
    if not isinstance(text, str):
        return str(text)
    
    # Sort prefixes by length descending to catch 'spatial_res_' before 'spatial_'
    prefixes = sorted([f"{d.value}_" for d in Domain], key=len, reverse=True)
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.replace(prefix, "", 1)
            break
            
    return text.replace('_', ' ').title()

def build_grouped_choices(columns: list[str], fallback_group: str = "Other") -> dict:
    domain_map = {
        Domain.GENOTYPE.value: "Genotypes 🧬",
        Domain.PHENOTYPE.value: "Phenotypes 🔬",
        Domain.AMR.value: "AMR Determinants 💊",
        Domain.VIRULENCE.value: "Virulence Markers ⚔️",
        Domain.SPATIAL.value: "Spatial 🌍",
        Domain.TEMPORAL.value: "Temporal 📅",
        Domain.CLUSTER.value: "Clusters 🕸️"
    }
    
    choices = {}
    for col in columns:
        prefix = col.split('_')[0]
        
        group_name = domain_map.get(prefix, fallback_group)
        if prefix in domain_map:
            clean_name = col.split('_', 1)[-1].replace('_', ' ').title()
        else:
            clean_name = col.replace('_', ' ').title()
            
        if group_name not in choices:
            choices[group_name] = {}
        choices[group_name][col] = clean_name
        
    return choices

def format_metadata_ui(meta_dict: dict) -> list:
    """Helper to cleanly format a dictionary into bolded UI paragraphs."""
    formatted_ui = []
    for k, v in meta_dict.items():
        if isinstance(v, list):
            val_str = ", ".join(map(_clean_ui_label, v)) if v else "None"
        elif hasattr(v, 'value'):
            val_str = _clean_ui_label(v.value)
        else:
            val_str = _clean_ui_label(v)
        formatted_ui.append(ui.p(ui.tags.strong(f"{k.replace('_', ' ').title()}: "), val_str, class_="mb-1"))
    return formatted_ui


class ColMapper:
    _aliases = {  # Define aliases for intelligent column guessing
        "map_id": ["sample", "id", "isolate", "name", "run", "barcode"],
        "map_date": ["date", "year", "collection", "time"],
        "map_spatial": ["spatial", "country", "nation", "region", "location", "site"],
        "map_lat": ["lat", "latitude"],
        "map_lon": ["lon", "long", "longitude"]
    }
    _id_regex = re_compile(r'\bid\b')  # ID regex
    __slots__ = ('_col_map',)
    def __init__(self, available_cols: list[str]):
        self._col_map = {c.lower().strip(): c for c in available_cols}

    def guess(self, field_id: str) -> str:
        terms = self._aliases.get(field_id, [])

        # 1. Exact match first (e.g. 'sample_id' == 'sample_id')
        for t in terms:
            if col := self._col_map.get(t, ""):
                return col

        # 2. Substring match (e.g. 'collection' in 'collection_date')
        for t in terms:
            for i, (c, col) in enumerate(self._col_map.items()):
                if t == 'id' and not self._id_regex.search(c.replace('_', ' ')):
                    continue  # Prevent 'id' matching 'width'
                if t in c:
                    return col
        return ""


@asynccontextmanager
async def ui_task(error_prefix: str = "Task Error"):
    """
    Standardizes background task execution in the Shiny app by wrapping it
    in a Progress bar and safely catching/notifying unhandled exceptions.
    """
    with ui.Progress(min=0, max=100) as p:
        try:
            yield p
        except Exception as e:
            ui.notification_show(f"{error_prefix}: {str(e)}", type="error", duration=15)


async def generate_temp_download(save_func: Callable[[Path], Any], suffix: str, error_prefix: str = "Export Error") -> AsyncGenerator[bytes, None]:
    """
    Standardizes the creation, writing, and byte-yielding of temporary files for Shiny downloads.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        path = Path(tmp.name)
    try:
        await to_thread(save_func, path)
        data = await to_thread(path.read_bytes)
        yield data
    except Exception as e:
        ui.notification_show(f"{error_prefix}: {str(e)}", type="error", duration=15)
    finally:
        path.unlink(missing_ok=True)


# UIs ------------------------------------------------------------------------------------------------------------------
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


# Servers --------------------------------------------------------------------------------------------------------------
@module.server
def safe_plot_server(input, output, session, data_reactive: Union[reactive.Value, Callable], plot_type: Union[PlotType, str, Callable]):
    """A universal plotting server module that handles null-checks and error boundaries."""

    @render_widget
    def plot():
        data = data_reactive() if callable(data_reactive) else data_reactive.get()
        if data is None:
            return None

        # Resolve the plot type if it's passed as a reactive function (e.g. dropdowns)
        p_type = plot_type() if callable(plot_type) else plot_type

        try:
            return render_plot(data, p_type)
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
            yield await to_thread(df.to_csv, index=False)
