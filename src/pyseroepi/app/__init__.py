from pathlib import Path
from shiny import App
from pathogenx.app.ui import main_ui
from pathogenx.app.server import main_server

def app():
    return App(main_ui, main_server, static_assets=Path(__file__).parent / "www")
