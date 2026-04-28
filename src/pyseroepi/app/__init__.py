from shiny import App
from seroepi.app.ui import main_ui
from seroepi.app.server import main_server

app = App(main_ui, main_server)
