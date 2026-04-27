#!/usr/bin/env -S just --justfile

alias t := test
log := "warn"
export JUST_LOG := log
set script-interpreter := ['uv', 'run', '--script']

app:
    uv run shiny run src/pyseroepi/app/ui.py --reload

docs:
    cp README.md docs/index.md
    uv run --group docs zensical build --clean
    rm docs/index.md