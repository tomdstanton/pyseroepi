# PathoGenX Web App
###### A web-based GUI for pathogen genotype exploration

> [!WARNING]
> 🚧 This package is currently under construction, proceed with caution 🚧

## Introduction
This app provides a web-based GUI for the exploration of pathogen genotyping data.
This begun as a Python port of the R-Shiny app and backend R code for the Klebsiella Neonatal Sepsis Sero-epi app,
but we generalised it for the exploration of different genotypes from multiple sources.
You can read more about the `pathogenx` code [here](../../../README.md).

## Installation

The app is an optional module under the `pathogenx` library and can therefore be installed with pip:
```shell
pip install pathogenx[app]
```

## Usage
We've exposed the app module via the PathoGenX CLI, and can be run in your browser with the following command:

```shell
pathogenx app
```

### Arguments
```shell
pathogenx app -h
usage: pathogenx app [options]

========================|> PathoGenX |>========================
      A Python library for Pathogen Genotype eXploration       

App options:
  
  Arguments to be passed to `shiny.run_app()`

  --host              The address that the app should listen on (default: 127.0.0.1)
  --port              The port that the app should listen on.
                      Set to 0 to use a random port (default: 8000)
  --autoreload-port   The port that should be used for an additional websocket that is used to
                      support hot-reload. Set to 0 to use a random port (default: 0)
  --reload            Enable auto-reload
  --ws-max-size       WebSocket max size message in bytes (default: 16777216)
  --launch-browser    Launch app browser after app starts, using the Python webbrowser module
  --dev-mode          Run in development mode

Other options:

  -v, --version       Show version number and exit
  -h, --help          Show this help message and exit
```