"""
Module for generating interactive plots for `pathogenx.calculators.PrevalenceResult`s
"""
from typing import Literal, List
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathogenx.calculators import PrevalenceResult

# Constants ------------------------------------------------------------------------------------------------------------
_NE_COUNTRIES = Path(__file__).parent / "data" / "world-administrative-boundaries"
_WORLD = geopandas.read_file(_NE_COUNTRIES)
_THEME = {
    'template': 'ggplot2',
    'legend': dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    'font': dict(family="sans-serif", size=12, color="black"),
    'margin': dict(l=20, r=20, t=40, b=20),
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
}
_LOW_COLOUR: str = '#2E91E5'
_HIGH_COLOUR: str = '#FD3216'
_FILL_COLOUR: str = '#D1C1E1'
_MAX_X = 20


# Classes --------------------------------------------------------------------------------------------------------------
class PrevalencePlotter:
    """Creates an interactive bar plot for `PrevalenceResult` objects."""
    def __init__(self, y_order: list[str] | pd.Series = None, fill_by: Literal['prop', 'rank', 'count'] = 'count',
                 low_col: str = _LOW_COLOUR, high_col: str = _HIGH_COLOUR, theme: dict = _THEME):
        """Initializes the PrevalencePlotter.

        Args:
            y_order (list[str] | pd.Series, optional): The order of categories on the y-axis.
                Defaults to None, which uses the order from the data.
            fill_by (Literal['prop', 'rank', 'count'], optional): The variable to use for color-filling the bars.
                Defaults to 'count'.
            low_col (str, optional): The color for the low end of the scale. Defaults to _LOW_COLOUR.
            high_col (str, optional): The color for the high end of the scale. Defaults to _HIGH_COLOUR.
            theme (dict, optional): A dictionary of Plotly layout settings. Defaults to _THEME.
        """
        self.y_order = y_order
        self.fill_by = fill_by
        self.low_col = low_col
        self.high_col = high_col
        self.theme = theme

    def _create_go(self, data: pd.DataFrame, y_col_name: str, col_type: str) -> go.Bar:
        """Creates a single bar trace for a subplot."""
        prop_col = f'prop.{col_type}'
        upper_col = f'upper.{col_type}'
        lower_col = f'lower.{col_type}'
        fill_col = f'{self.fill_by}.{col_type}'

        # Vectorized hover text generation (highly efficient)
        hover_text = (
                f"<b>{y_col_name}</b>: " + data[y_col_name].astype(str) +
                "<br><b>Proportion</b>: " + data[prop_col].map('{:.3f}'.format) +
                f"<br><b>{self.fill_by.capitalize()}</b>: " + data[fill_col].astype(str) +
                "<br><b>Denominator</b>: " + data[f'denominator.{col_type}'].astype(str)
        )
        return go.Bar(
            y=data[y_col_name],
            x=data[prop_col],
            orientation='h',
            name=prop_col,
            hoverinfo='text',
            hovertext=hover_text,
            marker=go.bar.Marker(color=data[fill_col], colorscale=[self.low_col, self.high_col]),
            error_x=go.bar.ErrorX(type='data', symmetric=False, array=data[upper_col], arrayminus=data[lower_col])
        )

    def plot(self, result: PrevalenceResult) -> go.Figure:
        """Generates a bar plot from a PrevalenceResult.

        The plot will show raw prevalence and adjusted prevalence if available.

        Args:
            result (PrevalenceResult): The prevalence calculation result to plot.

        Returns:
            go.Figure: A Plotly figure object.
        """
        data = result.data
        y_col_name = result.stratified_by[-1]  # Plot by the finest stratification level
        # Set a sensible default sort order if none is provided
        y_order = self.y_order or data[y_col_name].unique().tolist()
        # Filter data to only include categories in y_order
        data = data[data[y_col_name].isin(y_order)]
        col_types = ['raw', 'adj'] if result.adjusted_for else ['raw']

        fig = make_subplots(
            rows=1, cols=len(col_types), shared_yaxes=True,
            subplot_titles=[f'{i.capitalize()} Prevalence' for i in col_types],
            horizontal_spacing=0.01
        )

        for col, col_type in enumerate(col_types, start=1):  # Add subplots
            fig.add_trace(self._create_go(data, y_col_name, col_type), row=1, col=col)

        fig.update_layout(
            **self.theme,
            showlegend=False,
            barmode='relative',
            yaxis=dict(title=y_col_name, categoryorder='array', categoryarray=y_order, autorange="reversed"),
            xaxis1=dict(title='Raw Proportion', autorange="reversed", tickformat='.1f'),
        )
        if len(col_types) > 1:
            return fig.update_layout(xaxis2=dict(title='Adjusted Proportion', tickformat='.1f'))
        return fig


class StrataPlotter:
    """
    Creates an interactive heatmap plot for looking at prevalence of one stratum within another.
    """

    def __init__(
            self, y_order: list[str] | pd.Series = None,
            # fill_by: Literal['prop.raw', 'rank.raw', 'count.raw', 'prop.adj', 'rank.adj', 'count.adj'] = 'prop.raw',
            max_x: int = _MAX_X, low_col: str = _LOW_COLOUR, high_col: str = _HIGH_COLOUR, theme: dict = _THEME
    ):
        """Initializes the StrataPlotter.

        Args:
            y_order (list[str] | pd.Series, optional): The order of categories on the y-axis.
                Defaults to None, which uses the order from the data.
            max_x (int, optional): The maximum number of categories to show on the x-axis
                before grouping the rest into "Other". Defaults to _MAX_X.
            low_col (str, optional): The color for the low end of the heatmap scale.
                Defaults to _LOW_COLOUR.
            high_col (str, optional): The color for the high end of the heatmap scale.
                Defaults to _HIGH_COLOUR.
            theme (dict, optional): A dictionary of Plotly layout settings. Defaults to _THEME.
        """
        self.y_order = y_order
        # self.fill_by = fill_by
        self.max_x = max_x
        self.low_col = low_col
        self.high_col = high_col
        self.theme = theme

    def plot(self, result: PrevalenceResult) -> go.Figure:
        """Generates a heatmap from a PrevalenceResult stratified by two columns.

        Args:
            result (PrevalenceResult): The prevalence result, which must be stratified
                by exactly two columns.

        Returns:
            go.Figure: A Plotly figure object representing the heatmap.

        Raises:
            ValueError: If the result is not stratified by exactly two columns.
        """
        if not result.denominator or len(result.stratified_by) != 2:
            raise ValueError("StrataPlotter requires results stratified by exactly two columns with a denominator.")

        data = result.data
        # The first stratified column is always Y, the second is always X.
        y_col_name, x_col_name = result.stratified_by[0], result.stratified_by[1]
        if y_col_name == result.denominator:  # denominator has not been swapped
            numerator = x_col_name
            denominator_swapped = False
        else:
            numerator = y_col_name
            denominator_swapped = True
        # Determine Y-axis order and filter data
        y_order = self.y_order or data[y_col_name].unique().tolist()
        data = data[data[y_col_name].isin(y_order)]

        # Determine X-axis order and group less frequent categories into "Other"
        x_order = data[x_col_name].value_counts().index.tolist()
        if len(x_order) > self.max_x:
            top_x = x_order[:self.max_x]
            if not denominator_swapped:
                other_name = f"Other ({len(x_order) - self.max_x})"
                # Use the more efficient, vectorized .where() method instead of .apply()
                data[x_col_name] = data[x_col_name].where(data[x_col_name].isin(top_x), other_name)
                x_order = top_x + [other_name]

        # Summarize data after potential "Other" grouping.
        plot_data = data.groupby([y_col_name, x_col_name])['prop.raw'].sum().reset_index()
        # Create hover text
        hover_text = (
                f"<b>{y_col_name}</b>: " + plot_data[y_col_name].astype(str) +
                f"<br><b>{x_col_name}</b>: " + plot_data[x_col_name].astype(str) +
                f"<br><b>Prop</b>: " +
                plot_data['prop.raw'].map('{:.3f}'.format)
        )
        # Create the heatmap trace
        return go.Figure(
            go.Heatmap(
                x=plot_data[x_col_name],
                y=plot_data[y_col_name],
                z=plot_data['prop.raw'],
                text=hover_text,
                hoverinfo='text',
                colorscale=[self.low_col, self.high_col],
                colorbar={'title': 'Proportion'},
            )
        ).update_layout(
            **self.theme,
            title=f'Proportion of<br>{numerator} within {result.denominator}',
            xaxis=dict(title=x_col_name, categoryorder='array', categoryarray=x_order),
            yaxis=dict(title=y_col_name, categoryorder='array', categoryarray=y_order, autorange="reversed"),
            # Set hover label background to match tile color
            hoverlabel=dict(
                bgcolor="rgba(0,0,0,0)",  # Make the hover box background transparent
                font=dict(color="white")  # Set font color to white for visibility
            )
        ).update_xaxes(tickangle=45)


class SummaryBarPlotter:
    """
    Creates a horizontal bar plot to summarize a numeric variable against a
    categorical variable from a PrevalenceResult.
    """

    def __init__(self, fill_by: str, y_order: list[str] | pd.Series = None, fill_color: str = _FILL_COLOUR,
                 theme: dict = _THEME):
        """Initializes the SummaryBarPlotter.

        Args:
            fill_by (str): The numeric variable to plot on the x-axis (e.g., a distinct count).
            y_order (list[str] | pd.Series, optional): The order of categories on the y-axis.
                Defaults to None, which uses the order from the data.
            fill_color (str, optional): The color for the bars. Defaults to _FILL_COLOUR.
            theme (dict, optional): A dictionary of Plotly layout settings. Defaults to _THEME.
        """
        self.fill_by = fill_by
        self.y_order = y_order
        self.fill_color = fill_color
        self.theme = theme

    def plot(self, result: PrevalenceResult) -> go.Figure:
        """Generates a summary bar plot from a PrevalenceResult.

        Args:
            result (PrevalenceResult): The prevalence calculation result to plot.

        Returns:
            go.Figure: A Plotly figure object.

        Raises:
            ValueError: If the `fill_by` variable is not found in the result data.
        """
        data = result.data
        y_col_name = result.stratified_by[0]
        x_col_name = f"# {self.fill_by}"

        # Set a sensible default sort order if none is provided, and filter data
        y_order = self.y_order or data[y_col_name].unique().tolist()
        data = data[data[y_col_name].isin(y_order)]
        if self.fill_by not in result.n_distinct or x_col_name not in data.columns:
            raise ValueError(f"Variable '{x_col_name}' not found in the result data. "
                             f"Ensure '{self.fill_by}' was in `n_distinct` during calculation.")

        hover_text = (
                f"<b>{y_col_name}</b>: " + data[y_col_name].astype(str) +
                f"<br><b>{x_col_name}</b>: " + data[x_col_name].astype(str)
        )

        return go.Figure(go.Bar(
            x=data[x_col_name],
            y=data[y_col_name],
            orientation='h',
            text=hover_text,
            hoverinfo='text',
            marker=dict(color=self.fill_color, line=dict(color='black', width=0.5))
        )).update_layout(
            **self.theme,
            showlegend=False,
            xaxis_title=x_col_name,
            yaxis=dict(title=y_col_name, categoryorder='array', categoryarray=y_order, autorange="reversed"),
            title=f'Distinct {x_col_name}<br>per {y_col_name}'
        )


class CoveragePlotter:
    """
    Creates an interactive plot to visualize cumulative coverage.

    This plot shows the cumulative proportion (coverage) of a target as more
    categories of a stratifying variable are included. It's useful for
    understanding how many unique genotypes, for example, are needed to
    cover a certain percentage of the population in different countries.
    """

    def __init__(self, x_order: List[str] = None, max_x: int = 15, theme: dict = _THEME):
        """Initializes the CoveragePlotter.

        Args:
            x_order (List[str], optional): The order of categories on the x-axis.
                Defaults to None, which orders by descending total count.
            max_x (int, optional): The maximum number of x-axis categories to display.
                Defaults to 15.
            theme (dict, optional): A dictionary of Plotly layout settings. Defaults to _THEME.
        """
        self.x_order = x_order
        self.max_x = max_x
        self.theme = theme

    def plot(self, result: PrevalenceResult) -> go.Figure:
        """Generates a cumulative coverage plot from a PrevalenceResult.

        Args:
            result (PrevalenceResult): The prevalence result, which must be stratified
                by at least two columns and have a denominator.

        Returns:
            go.Figure: A Plotly figure object.

        Raises:
            ValueError: If the result is not stratified by at least two columns
                or does not have a denominator.
        """
        if not result.denominator or len(result.stratified_by) < 2:
            raise ValueError("CoveragePlotter requires results stratified by at least two columns with a denominator.")

        data = result.data
        denominator_col = result.denominator
        x_col = next(c for c in result.stratified_by if c != denominator_col)

        # Determine X-axis order
        if self.x_order:
            x_order = self.x_order
        else:
            # Default to ordering by total count across all denominators
            x_order = data.groupby(x_col)['count.raw'].sum().sort_values(ascending=False).index.tolist()

        if self.max_x and len(x_order) > self.max_x:
            x_order = x_order[:self.max_x]

        # Prepare data for plotting by creating all combinations
        plot_data = (
            data.set_index([denominator_col, x_col])
            .reindex(
                pd.MultiIndex.from_product([data[denominator_col].unique(), x_order], names=[denominator_col, x_col]))
            .fillna(0)
            .reset_index()
        )

        # Set the custom sort order for the x-axis column and then sort
        plot_data[x_col] = pd.Categorical(plot_data[x_col], categories=x_order, ordered=True)
        plot_data = plot_data.sort_values(by=[denominator_col, x_col])

        # Calculate cumulative values within each denominator group
        grouped = plot_data.groupby(denominator_col, sort=False)
        plot_data['cum_prop'] = grouped['prop.raw'].cumsum()
        plot_data['se_squared'] = plot_data['se.raw'] ** 2
        # The error of a sum of proportions is complex. We approximate the
        # cumulative error by summing the individual standard errors in quadrature.
        plot_data['cum_se'] = np.sqrt(grouped['se_squared'].cumsum())

        z = norm.ppf(1 - (0.05 / 2))
        plot_data['cum_lower'] = (plot_data['cum_prop'] - z * plot_data['cum_se']).clip(0, 1)
        plot_data['cum_upper'] = (plot_data['cum_prop'] + z * plot_data['cum_se']).clip(0, 1)

        # Create hover text
        hover_text = (
                f"<b>{x_col}</b>: " + plot_data[x_col].astype(str) +
                f"<br><b>{denominator_col}</b>: " + plot_data[denominator_col].astype(str) +
                "<br>Individual Prop: " + plot_data['prop.raw'].map('{:.3f}'.format) +
                "<br>Cumulative Prop: " + plot_data['cum_prop'].map('{:.3f}'.format)
        )

        fig = go.Figure()
        for name, group in plot_data.groupby(denominator_col):
            fig.add_trace(go.Scatter(
                x=group[x_col], y=group['cum_prop'], name=name,
                mode='lines+markers', text=hover_text[group.index], hoverinfo='text'
            ))
            fig.add_trace(go.Scatter(
                x=group[x_col], y=group['cum_upper'], name=f'{name}_upper',
                fill=None, mode='lines', line=dict(width=0), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=group[x_col], y=group['cum_lower'], name=f'{name}_lower',
                fill='tonexty', mode='lines', line=dict(width=0), showlegend=False,
                hoverinfo='none'
            ))

        return fig.update_layout(
            **self.theme,
            title=f'Cumulative Coverage of {x_col} within {denominator_col}',
            xaxis=dict(title=x_col, categoryorder='array', categoryarray=x_order),
            yaxis=dict(title='Cumulative Proportion (Coverage)', tickformat='.2f'),
        ).update_xaxes(tickangle=45)


class MapPlotter:
    """
    Creates an interactive choropleth map to visualize geographical coverage.

    This plotter takes a prevalence result, summarizes the cumulative coverage
    for a selected list of antigens within each geographical area, and
    displays it on a world map.
    """

    def __init__(self, low_col: str = _LOW_COLOUR, high_col: str = _HIGH_COLOUR):
        """Initializes the MapPlotter.

        Args:
            low_col (str, optional): The color for low coverage areas on the map.
                Defaults to _LOW_COLOUR.
            high_col (str, optional): The color for high coverage areas on the map.
                Defaults to _HIGH_COLOUR.
        """
        # shape_file: Path = _NE_COUNTRIES
        # self.world: geopandas.GeoDataFrame = geopandas.read_file(shape_file)
        self.world: geopandas.GeoDataFrame = _WORLD
        self.low_col = low_col
        self.high_col = high_col

    def plot(self, result: PrevalenceResult, world_col: str = 'name') -> go.Figure:
        """Generates a choropleth map from a PrevalenceResult.

        The plot visualizes geographical data by merging prevalence results with a
        world map shapefile.

        Args:
            result (PrevalenceResult): The prevalence result containing geographical data.
            world_col (str, optional): The column name in the world shapefile to join on.
                Defaults to 'name'.

        Returns:
            go.Figure: A Plotly figure object representing the map.

        Raises:
            ValueError: If `world_col` is not in the world dataframe, or if the
                `PrevalenceResult` is not in a format that can be mapped.
        """
        data = result.data
        if world_col not in self.world.columns:
            raise ValueError(f'{world_col} not in world dataframe')

        # Define color interpolation logic here for the callback
        low_rgb = tuple(int(self.low_col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))
        high_rgb = tuple(int(self.high_col.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

        def color_map(coverage: float) -> str:
            rgb = [int(low + (high - low) * coverage) for low, high in zip(low_rgb, high_rgb)]
            return f'rgb({rgb[0]},{rgb[1]},{rgb[2]})'

        # Scenario 1: Data is already aggregated by a single geo-stratum
        if not result.denominator and len(result.stratified_by) == 1:
            geo_col = result.stratified_by[0]
            map_data = data.rename(columns={'prop.raw': 'Coverage', 'count.raw': 'Infections'})
        # Scenario 2: Data is stratified by geo (denominator) and something else
        elif result.denominator and len(result.stratified_by) > 1:
            geo_col = result.denominator
            map_data = (
                data.groupby(geo_col)
                .agg(
                    Infections=('count.raw', 'sum'),
                    Coverage=('prop.raw', 'sum')
                )
                .assign(Coverage=lambda df: df['Coverage'].clip(upper=1.0))
                .reset_index()
            )
        else:
            raise ValueError(
                "MapPlotter requires a PrevalenceResult with either a single geographical stratum "
                "or multiple strata with the geographical column as the denominator."
            )
        # Merge summarized data with geospatial data
        # Note: This assumes the geo_col in your data ('Country') matches the 'name' column in the world dataset.
        # You may need to adjust this mapping if your names differ (e.g., 'United States' vs 'United States of America').
        merged_data = self.world.merge(map_data, how="inner", left_on=world_col, right_on=geo_col)
        merged_data['Coverage'] = merged_data['Coverage'].fillna(0)  # Fill non-matching countries with 0

        fig = go.Figure(go.Choroplethmapbox(
            geojson=merged_data.__geo_interface__,
            locations=merged_data.index,  # Use the GeoDataFrame index
            z=merged_data['Coverage'],
            colorscale=[self.low_col, self.high_col],
            zmin=0,
            zmax=1,
            marker_opacity=0.6,
            marker_line_width=0.5,
            colorbar_title='Coverage',
            customdata=merged_data[['name', 'Infections', 'Coverage']],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Infections: %{customdata[1]}<br>"
                "Coverage: %{customdata[2]:.3f}<extra></extra>"
            )
        ))
        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=1,
            mapbox_center={"lat": 25, "lon": 0},
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return fig


# Functions ------------------------------------------------------------------------------------------------------------
def merge_prevalence_figs(prevalence: go.Figure, strata: go.Figure = None, summary_bar: go.Figure = None) -> go.Figure | None:
    """Merges traces from up to three figures into a single figure."""
    plots = [prevalence]
    if strata:
        plots.append(strata)
    if summary_bar:
        plots.append(summary_bar)

    traces = [trace for p in plots for trace in p.data]
    if not traces:
        return None

    num_prevalence_plots = len(prevalence.data)

    column_widths = []
    if num_prevalence_plots == 1:
        column_widths.extend([0.4])
    elif num_prevalence_plots == 2:
        column_widths.extend([0.2, 0.2])

    if strata:
        column_widths.append(0.4)
    if summary_bar:
        column_widths.append(0.2)

    # Subplot titles for prevalence are in annotations. For others, in title.
    subplot_titles = []
    for ann in prevalence.layout.annotations:
        subplot_titles.append(ann.text)
    if strata:
        subplot_titles.append(strata.layout.title.text)
    if summary_bar:
        subplot_titles.append(summary_bar.layout.title.text)

    fig = make_subplots(
        rows=1,
        cols=len(traces),
        shared_yaxes=True,
        column_widths=column_widths if len(column_widths) == len(traces) else None,
        horizontal_spacing=0.01,
        subplot_titles=subplot_titles
    )

    for n, trace in enumerate(traces, start=1):
        fig.add_trace(trace, row=1, col=n)

    fig.update_layout(yaxis=prevalence.layout.yaxis)

    col_idx = 1
    if num_prevalence_plots == 2:  # Raw + Adjusted
        fig.update_xaxes(title=prevalence.layout.xaxis1.title, tickformat=prevalence.layout.xaxis1.tickformat,
                         autorange=prevalence.layout.xaxis1.autorange, row=1, col=col_idx)
        col_idx += 1
        fig.update_xaxes(title=prevalence.layout.xaxis2.title, tickformat=prevalence.layout.xaxis2.tickformat,
                         autorange=prevalence.layout.xaxis2.autorange, row=1, col=col_idx)
        col_idx += 1
    elif num_prevalence_plots == 1:  # Raw only
        # The prevalence plotter has a bug where the title is on xaxis1 even for a single plot.
        fig.update_xaxes(title=prevalence.layout.xaxis1.title, tickformat=prevalence.layout.xaxis1.tickformat,
                         autorange=prevalence.layout.xaxis1.autorange, row=1, col=col_idx)
        col_idx += 1

    if strata:
        fig.update_xaxes(patch=strata.layout.xaxis, row=1, col=col_idx)
        col_idx += len(strata.data)

    if summary_bar:
        fig.update_xaxes(patch=summary_bar.layout.xaxis, row=1, col=col_idx)

    return fig.update_layout(**_THEME, showlegend=False)
