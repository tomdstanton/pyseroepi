from typing import Any, Callable, Union
from abc import ABC, abstractmethod
from json import load as json_load
from warnings import warn
from importlib.resources import files

from plotly.graph_objects import Figure

from seroepi.constants import PlotType


# Classes --------------------------------------------------------------------------------------------------------------
class BasePlotter(ABC):
    """
    Stateless base class for all plotting engines in seroepi.

    This class houses global configurations, themes, and shared utilities for
    rendering interactive Plotly figures. It uses a dark-slate aesthetic
    optimized for modern web dashboards.

    Attributes:
        _MAIN_COLOUR (str): Primary color for data (Electric Cyan).
        _CI_COLOUR (str): Color for confidence interval ribbons.
        _ACCENT_COLOUR (str): Secondary highlight color (Neon Pink).
    """
    # The Hero Palette: Electric Cyan and Neon Pink
    # Cyan is scientifically colorblind-safe while looking stunning on dark backgrounds.
    _MAIN_COLOUR = '#0EA5E9'
    # Translucent Cyan for Confidence Interval Ribbons (20% Opacity)
    _CI_COLOUR = 'rgba(14, 165, 233, 0.2)'
    # A secondary highlight color (Optional, but great for distinguishing target groups)
    _ACCENT_COLOUR = '#EC4899'  # Vibrant Neon Pink
    # Global cache to prevent reading the file from disk multiple times
    _WORLD_GEOJSON = None

    @classmethod
    def _get_world_geojson(cls) -> dict:
        """
        Lazily loads and caches the internal world boundaries GeoJSON.

        Returns:
            A dictionary containing the GeoJSON data.
        """
        if cls._WORLD_GEOJSON is None:
            try:
                # Safely navigates the package structure regardless of where it's installed
                geojson_path = files('seroepi.data').joinpath('world_boundaries.geojson')
                with geojson_path.open(mode='r', encoding='utf-8') as f:
                    cls._WORLD_GEOJSON = json_load(f)
            except Exception as e:
                warn(f"Could not load internal world boundaries. Ensure the file exists: {e}")
                cls._WORLD_GEOJSON = {}
        return cls._WORLD_GEOJSON

    _PLOT_REGISTRY = {}

    @classmethod
    def register_plotter(cls, result_class: Union[type, tuple], plot_type: PlotType) -> Callable:
        """
        Decorator to register a plotter class for a specific result type.

        Args:
            result_class: The result class (or tuple of classes) to register for.
            plot_type: The name/kind of the plot (e.g., 'bar').

        Returns:
            A decorator function.
        """
        def decorator(plotter_cls):
            classes = result_class if isinstance(result_class, tuple) else (result_class,)
            for rc in classes:
                if rc not in cls._PLOT_REGISTRY:
                    cls._PLOT_REGISTRY[rc] = {}
                cls._PLOT_REGISTRY[rc][plot_type] = plotter_cls
            return plotter_cls

        return decorator

    @classmethod
    @abstractmethod
    def render(cls, result_obj: Any, **kwargs) -> 'Figure':
        """
        Renders the result object into a Plotly figure.

        Args:
            result_obj: The result object to visualize.
            **kwargs: Additional plotting arguments.

        Returns:
            A plotly Figure object.
        """
        pass
