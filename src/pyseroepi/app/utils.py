"""
Module containing generic app utility functions
"""
from shiny import ui
from faicons import icon_svg as icon
from pathlib import Path
from pathogenx.utils import bold


# Functions ------------------------------------------------------------------------------------------------------------
def dropdown_function(id_, *args) -> ui.Tag:
    """Creates a dropdown menu with a gear icon.

    Args:
        id_ (str): The ID for the action button.
        *args: Arguments to be passed to the popover.

    Returns:
        A Shiny UI popover element.
    """
    return ui.popover(
        ui.input_action_button(id_, "Configure plot", icon=icon("gear")),
        *args,
        title="Options",
        placement="bottom"
    )

def nice_name(name: str, replace_chars: tuple[str] = ('_', '.'), replace_with: str = ' ') -> str:
    """Formats a string into a human-readable 'nice' name.

    This is done by replacing specified characters and converting to title case.

    Args:
        name (str): The string to format.
        replace_chars (tuple[str], optional): A tuple of characters to replace.
            Defaults to ('_', '.').
        replace_with (str, optional): The character to replace with.
            Defaults to ' '.

    Returns:
        str: The formatted string.
    """
    for old in replace_chars:
        name = name.replace(old, replace_with)
    return name.title()


def plural_name(name: str, default: str = 's', niceify: bool=False, *args, **kwargs) -> str:
    """Converts a singular noun to its plural form.

    Handles common English pluralization rules.

    Args:
        name (str): The noun to pluralize.
        default (str, optional): The default plural suffix. Defaults to 's'.
        niceify (bool, optional): Whether to format the name using nice_name()
            before pluralizing. Defaults to False.
        *args: Additional arguments to pass to nice_name().
        **kwargs: Additional keyword arguments to pass to nice_name().

    Returns:
        str: The pluralized noun.
    """
    if niceify:
        name = nice_name(name, *args, **kwargs)
    if name.endswith('us'):
        return name[:-2] + 'i'
    elif name.endswith('y'):
        return name[:-1] + 'ies'
    elif name.endswith('s'):
        return name
    return name + default


def create_logo_link(src: str, url: str, width: str, tooltip_text: str | None = None):
    """
    Convenience function for creating a clickable image link that opens in a new tab.
    """
    link_tag = ui.a(ui.img(src=src, width=width, style="vertical-align: middle;"),
                    href=f'"https://{url}', target="_blank", id=f'{Path(src).stem}_logo')
    if tooltip_text:
        return ui.tooltip(link_tag, tooltip_text, placement='bottom')
    return link_tag


def app_cli_parser(subparsers, package: str, description: str, formatter_class, version):
    name, desc = 'app', 'Run the Shiny app'
    parser = subparsers.add_parser(
        name, description=description, prog=f'{package} {name}',
        formatter_class=formatter_class, help=desc, usage="%(prog)s [options]", add_help=False
    )
    app = parser.add_argument_group(bold('App options'), '\nArguments to be passed to `shiny.run_app()`')
    app.add_argument('--host', default='127.0.0.1', metavar='',
                     help='The address that the app should listen on (default: %(default)s)')
    app.add_argument('--port', default=8000, type=int, metavar='',
                     help='The port that the app should listen on.\n'
                          'Set to 0 to use a random port (default: %(default)s)')
    app.add_argument('--autoreload-port', default=0, type=int, metavar='',
                     help='The port that should be used for an additional websocket that is used to\n'
                          'support hot-reload. Set to 0 to use a random port (default: %(default)s)')
    app.add_argument('--reload', action='store_true', help='Enable auto-reload')
    app.add_argument('--ws-max-size', default=16777216, type=int, metavar='',
                     help='WebSocket max size message in bytes (default: %(default)s)')
    app.add_argument('--launch-browser', action='store_true',
                     help='Launch app browser after app starts, using the Python webbrowser module')
    app.add_argument('--dev-mode', action='store_true', help='Run in development mode')
    app.add_argument('--factory', action='store_true', help='Treat app as an application factory\n'
                                                            'i.e. a () -> <ASGI app> callable.')

    opts = parser.add_argument_group(bold('Other options'), '')
    opts.add_argument('-v', '--version', help='Show version number and exit', action='version', version=version)
    opts.add_argument('-h', '--help', help='Show this help message and exit', action='help')
