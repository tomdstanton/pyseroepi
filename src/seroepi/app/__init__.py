from shiny import App
from seroepi.app._app import main_ui, main_server

app = App(main_ui, main_server)
