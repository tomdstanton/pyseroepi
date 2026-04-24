from pathlib import Path
from shiny import App
from pyseroepi.app.ui import main_ui
from pyseroepi.app.server import main_server

def app():
    return App(main_ui, main_server, static_assets=Path(__file__).parent / "www")
