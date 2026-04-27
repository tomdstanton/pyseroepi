from shiny import App
from pyseroepi.app.ui import main_ui
from pyseroepi.app.server import main_server

app = App(main_ui, main_server)
