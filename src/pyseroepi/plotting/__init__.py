from typing import Union
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyseroepi.plotting._base import BasePlotter
from pyseroepi.constants import PlotType, AggregationType
from pyseroepi import estimators
from pyseroepi.formulation import Formulation
from pyseroepi.dist import Distances


# Classes --------------------------------------------------------------------------------------------------------------
@BasePlotter.register_plotter((estimators.PrevalenceEstimates, Formulation), PlotType.COMPOSITION_BAR)
class CompositionBarPlotter(BasePlotter):

    @classmethod
    def render(cls, result: Union['estimators.PrevalenceEstimates', Formulation], **kwargs) -> go.Figure:
        is_formulation = isinstance(result, Formulation)
        
        if is_formulation:
            df = result.rankings.copy()
            target = result.target
            # The formulation dataframe strictly renames the target column to 'target'. Rename it back for plotting.
            df.rename(columns={'target': target}, inplace=True)
            
            # Filter to only show the antigens selected in the vaccine formulation
            df = df[df[target].isin(result.get_formulation())]
            
            group_cols = []
            strata_label = "Global Baseline"
            title_prefix = "Vaccine Formulation Composition"
        else:
            is_comp = getattr(result, 'aggregation_type', 'unknown') == AggregationType.COMPOSITIONAL
            if not is_comp:
                raise ValueError("Composition Bar Plot strictly requires Compositional aggregation mode.")
    
            df = result.data.copy()
            target = result.target
            group_cols = [c for c in result.stratified_by if c != target]
            strata_label = ', '.join(result.stratified_by)
            title_prefix = "Sample Composition"

        fig = go.Figure()

        # We calculate the global rank of the targets to ensure the colors and stack
        # order remain perfectly consistent across all bars
        target_ranks = df.groupby(target)['estimate'].sum().sort_values(ascending=False).index

        if not group_cols:
            # --- 1 VARIABLE: Global Composition ---
            for t in target_ranks:
                t_df = df[df[target] == t]
                fig.add_trace(go.Bar(
                    x=["Global Formulation"],
                    y=t_df['estimate'],
                    name=str(t),
                    hovertemplate=f"<b>{t}</b><br>Prevalence: %{{y:.1%}}<extra></extra>"
                ))
        else:
            # --- 2 VARIABLES: Grouped Composition ---
            group_col = group_cols[0]
            for t in target_ranks:
                t_df = df[df[target] == t]
                fig.add_trace(go.Bar(
                    x=t_df[group_col],
                    y=t_df['estimate'],
                    name=str(t),
                    hovertemplate=f"<b>{t}</b><br>{group_col}: %{{x}}<br>Prevalence: %{{y:.1%}}<extra></extra>"
                ))

        # Apply the global theme and force the stacking mode
        return fig.update_layout(
            barmode='stack',
            title=f"<b>{title_prefix}</b><br><sup>Stratified by {strata_label}</sup>",
            yaxis_title="Cumulative Prevalence",
            yaxis=dict(tickformat='.0%'),  # Renders 0.8 as 80%
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.PrevalenceEstimates,  PlotType.COMPOSITION_HEATMAP)
class CompositionHeatmapPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', **kwargs) -> go.Figure:
        df = result.data.copy()
        target = result.target
        is_comp = getattr(result, 'aggregation_type', 'unknown') == AggregationType.COMPOSITIONAL

        if not is_comp:
            # Trait Heatmap
            if len(result.stratified_by) != 2:
                raise ValueError(
                    f"Trait Heatmap strictly requires exactly 2 stratification variables. "
                    f"Found: {result.stratified_by}"
                )
            y_col = result.stratified_by[0]
            x_col = result.stratified_by[1]
            title_prefix = "Prevalence Matrix"
        else:
            y_col = target
            group_cols = [c for c in result.stratified_by if c != target]
            if len(group_cols) != 1:
                raise ValueError(
                    f"Composition Heatmap strictly requires exactly 1 grouping variable alongside the target. "
                    f"Found target '{target}' and groups: {group_cols}"
                )
            x_col = group_cols[0]
            title_prefix = "Density Matrix"

        # 2. Reshape from Long to Wide (The Z-Matrix)
        # fill_value=0 ensures that if a K-locus isn't found in a country, it shows as empty, not NaN
        pivot_df = df.pivot_table(
            index=y_col,
            columns=x_col,
            values='estimate',
            fill_value=0
        )

        # 3. Sort the Y-Axis so the most globally prevalent targets sit at the top of the heatmap
        pivot_df['total_burden'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('total_burden', ascending=True).drop(columns=['total_burden'])

        # 4. Transparent Color Scale
        cyan_accent = cls._MAIN_COLOUR
        custom_colorscale = [
            [0.0, 'rgba(0,0,0,0)'],  # 0% prevalence fades perfectly into the background
            [0.3, '#38BDF8'],  # Soft mid-tone for lower prevalences
            [1.0, cyan_accent]  # Maximum intensity for the dominant clones
        ]

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=custom_colorscale,
            showscale=True,  # Shows the colorbar legend on the right
            colorbar=dict(tickformat='.0%'),
            hovertemplate=f"<b>%{{y}}</b><br>{x_col}: %{{x}}<br>Prevalence: %{{z:.1%}}<extra></extra>"
        ))

        return fig.update_layout(
            title=f"<b>{title_prefix}</b><br><sup>{y_col} vs {x_col}</sup>",
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )



@BasePlotter.register_plotter(estimators.PrevalenceEstimates, PlotType.FOREST)
class ForestPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', sort_by: str = 'estimate', **kwargs) -> go.Figure:
        df = result.data.copy()
        target = result.target
        is_comp = getattr(result, 'aggregation_type', 'unknown') == AggregationType.COMPOSITIONAL

        if is_comp:
            y_col = target
            group_cols = [c for c in result.stratified_by if c != target]
            color_col = group_cols[0] if group_cols else None
        else:
            y_col = result.stratified_by[0] if result.stratified_by else target
            color_col = result.stratified_by[1] if len(result.stratified_by) > 1 else None

        # Sort the dataframe so the highest prevalences appear at the top
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=True)

        fig = go.Figure()

        if color_col:
            # --- MULTI-VARIABLE: Grouped Forest Plot ---
            for group_name, group_df in df.groupby(color_col):
                fig.add_trace(go.Scatter(
                    x=group_df['estimate'],
                    y=group_df[y_col],
                    name=str(group_name),
                    mode='markers',
                    marker=dict(size=10, symbol='square'),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=group_df['upper'] - group_df['estimate'],
                        arrayminus=group_df['estimate'] - group_df['lower'],
                        width=0,  # Removes the vertical caps for a cleaner "Tufte" look
                        thickness=2
                    ),
                    customdata=group_df[[color_col, 'lower', 'upper']].values,
                    hovertemplate=(
                        f"<b>{y_col}:</b> %{{y}}<br>"
                        f"<b>{color_col}:</b> %{{customdata[0]}}<br>"
                        "<b>Prevalence:</b> %{x:.1%}<br>"
                        "<b>95% CI:</b> %{customdata[1]:.1%} - %{customdata[2]:.1%}<extra></extra>"
                    )
                ))

            # THE MAGIC FIX: This prevents categorical points from overlapping!
            fig.update_layout(scattermode='group')

        else:
            # --- SINGLE-VARIABLE: Standard Forest Plot ---
            fig.add_trace(go.Scatter(
                x=df['estimate'],
                y=df[y_col],
                mode='markers',
                marker=dict(size=10, color=cls._MAIN_COLOUR, symbol='square'),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=df['upper'] - df['estimate'],
                    arrayminus=df['estimate'] - df['lower'],
                    color=cls._CI_COLOUR,
                    width=0, thickness=2
                ),
                customdata=df[['lower', 'upper']].values,
                hovertemplate=(
                    f"<b>{y_col}:</b> %{{y}}<br>"
                    "<b>Prevalence:</b> %{x:.1%}<br>"
                    "<b>95% CI:</b> %{customdata[0]:.1%} - %{customdata[1]:.1%}<extra></extra>"
                )
            ))

        return fig.update_layout(
            title=f"<b>Prevalence Estimates</b><br><sup>Target: {target} | Stratified by {', '.join(result.stratified_by)}</sup>",
            xaxis_title="Prevalence (%)",
            yaxis_title=y_col.replace('_', ' ').title(),
            xaxis=dict(tickformat='.0%'),  # Clean percentage formatting on the X-axis
            yaxis=dict(autorange='reversed'),  # Ensures highest rank is at the top
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.IncidenceEstimates, PlotType.EPICURVE)
class EpicurvePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.IncidenceEstimates', **kwargs):
        """
        Plots a classic Epidemic Curve (Cases over Time).
        Displays total sequencing volume in the background to visually address sampling bias.
        """
        data = result.data.copy().sort_values(by='date')

        fig = go.Figure()

        # Background Area: Total Sequenced Volume
        # Plotted first so it sits behind the bars
        fig.add_trace(go.Scatter(
            x=data['date'], y=data['total_sequenced'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(51, 65, 85, 0.3)',  # Translucent Slate
            line=dict(color='#475569', width=1, dash='dot'),
            name='Total Sequenced Volume',
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Sequenced</b>: %{y}<extra></extra>"
        ))

        # Foreground Bar: Variant Counts
        fig.add_trace(go.Bar(
            x=data['date'], y=data['variant_count'],
            marker_color=cls._MAIN_COLOUR,
            name=f"{result.target} Cases",
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Cases</b>: %{y}<extra></extra>"
        ))

        # Intelligently extract the Incidence Rate Ratio (IRR) to display in the subtitle
        subtitle = ""
        if not result.model_results.empty and 'IRR' in result.model_results.columns:
            irr = result.model_results['IRR'].iloc[0]
            if pd.notna(irr):
                direction = "Increasing" if irr > 1 else "Decreasing"
                subtitle = f"<br><sup>Trend: {direction} (IRR: {irr:.2f} per time step)</sup>"

        return fig.update_layout(
            title=f"<b>Epidemic Curve: {result.target}</b>{subtitle}",
            xaxis=dict(title="Date", type='date'),
            yaxis=dict(title='Count'),
            barmode='overlay',  # Allows the bars and the area chart to overlap safely
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.PrevalenceEstimates, PlotType.LONGITUDINAL_PREVALENCE)
class LongitudinalPrevalencePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', **kwargs):
        """
        Plots a time-series line chart for longitudinal prevalence data.
        """
        # Intelligently check if a date/time column exists in the strata
        time_cols = [col for col in result.stratified_by if
                     'date' in col.lower() or 'year' in col.lower() or 'month' in col.lower()]

        if not time_cols:
            raise ValueError(
                "No temporal column detected in the strata. Ensure you stratified by a Date, Year, or Month column.")

        time_col = time_cols[0]
        data = result.data.sort_values(by=time_col)

        fig = go.Figure()

        # Add the translucent confidence band
        fig.add_trace(go.Scattergl(
            x=pd.concat([data[time_col], data[time_col][::-1]]),
            y=pd.concat([data['upper'], data['lower'][::-1]]),
            fill='toself',
            fillcolor=cls._CI_COLOUR,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Add the main trend line with neon ring markers
        fig.add_trace(go.Scatter(
            x=data[time_col], y=data['estimate'],
            mode='lines+markers',
            line=dict(color=cls._MAIN_COLOUR, width=3),
            marker=dict(size=8, color=cls._MAIN_COLOUR),
            name="Prevalence",
            hovertemplate="<b>Date</b>: %{x}<br><b>Prevalence</b>: %{y:.2%}<extra></extra>"
        ))

        return fig.update_layout(
            title=f"<b>Longitudinal Prevalence of {result.target}</b>",
            xaxis=dict(title=time_col.title()),
            yaxis=dict(title='Prevalence', tickformat='.0%', range=[0, 1.05]),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.PrevalenceEstimates, PlotType.VACCINE_COVERAGE)
class VaccineCoveragePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates',  max_valencies: int = None):
        """
        Calculates cumulative population coverage.
        Crucial for designing multivalent vaccines (e.g., K-locus targeting).
        """
        data = result.data.copy()
        target_col = result.stratified_by[0]

        # Sort strictly by raw count to simulate prioritizing the most common variants
        data = data.sort_values(by='event', ascending=False).reset_index(drop=True)

        if max_valencies:
            data = data.head(max_valencies)

        # Calculate cumulative metrics
        data['cum_prop'] = data['estimate'].cumsum().clip(0, 1)

        # Approximate cumulative Standard Error (summing variance)
        # SE = (upper - lower) / (2 * 1.96)
        data['se'] = (data['upper'] - data['lower']) / 3.92
        data['cum_se'] = np.sqrt((data['se'] ** 2).cumsum())

        data['cum_lower'] = (data['cum_prop'] - 1.96 * data['cum_se']).clip(0, 1)
        data['cum_upper'] = (data['cum_prop'] + 1.96 * data['cum_se']).clip(0, 1)

        hover_text = (
            f"<b>{target_col}</b>: " + data[target_col].astype(str) +
            "<br><b>Valency Added</b>: +" + data['estimate'].map('{:.2%}'.format) +
            "<br><b>Total Coverage</b>: " + data['cum_prop'].map('{:.2%}'.format)
        )

        fig = go.Figure()

        # Upper bound ribbon
        fig.add_trace(go.Scatter(
            x=data[target_col], y=data['cum_upper'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))

        # Lower bound ribbon (fills up to the upper bound)
        fig.add_trace(go.Scatter(
            x=data[target_col], y=data['cum_lower'],
            mode='lines', fill='tonexty', fillcolor=cls._CI_COLOUR,
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))

        # The actual cumulative line and markers
        fig.add_trace(go.Scatter(
            x=data[target_col], y=data['cum_prop'],
            mode='lines+markers',
            name='Cumulative Coverage',
            line=dict(color=cls._MAIN_COLOUR, width=3),
            marker=dict(size=8, color=cls._MAIN_COLOUR),
            text=hover_text, hoverinfo='text'
        ))

        return fig.update_layout(
            title=f"<b>Cumulative Vaccine Coverage</b><br><sup>Targeting top {len(data)} {target_col} variants</sup>",
            xaxis=dict(title=f"Variant ({target_col}) added to formulation", tickangle=45),
            yaxis=dict(title='Cumulative Population Coverage', tickformat='.0%', range=[0, 1.05]),
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.PrevalenceEstimates, PlotType.CHOROPLETH)
class ChoroplethPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', geo_col: str = 'country', target_variant: str = None,
               feature_id_key: str = "properties.NAME"):
        """
        Plots discrete regional prevalence on a map.
        """
        data = result.data.copy()

        if geo_col not in result.stratified_by:
            raise ValueError(f"Cannot plotting choropleth: '{geo_col}' was not used as a stratification variable.")

        # --- THE FIX: Isolate the specific variant for compositional data ---
        target_name = result.target
        if result.aggregation_type == AggregationType.COMPOSITIONAL:
            if target_variant is None:
                # Safely default to the most frequent variant if the user forgot to specify one
                target_variant = data.groupby(target_name)['event'].sum().idxmax()

            data = data[data[target_name] == target_variant]
            target_name = f"{result.target}: {target_variant}"

        # Lazy-load the default map if none supplied
        geojson_data = cls._get_world_geojson()

        hover_text = (
                f"<b>{geo_col}</b>: " + data[geo_col].astype(str) +
                "<br><b>Prevalence</b>: " + data['estimate'].map('{:.2%}'.format) +
                "<br><b>95% CI</b>: [" + data['lower'].map('{:.2%}'.format) + ", " + data['upper'].map(
            '{:.2%}'.format) + "]"
        )

        custom_colorscale = [
            [0.0, 'rgba(0,0,0,0)'],  # Transparent
            [1.0, cls._MAIN_COLOUR]  # Electric Cyan
        ]

        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=data[geo_col],
            featureidkey=feature_id_key,
            z=data['estimate'],
            colorscale=custom_colorscale,
            zmin=0, zmax=1,
            marker_opacity=0.85,
            marker_line_width=0.5,
            marker_line_color='#0F172A',
            colorbar_title='Prevalence',
            text=hover_text,
            hoverinfo='text'
        ))

        title_prefix = "Regional Composition of" if result.aggregation_type == AggregationType.COMPOSITIONAL else "Regional Prevalence of"

        return fig.update_layout(
            # Dynamically updates the title so the user knows exactly what is mapped!
            title=f"<b>{title_prefix} {target_name}</b>",
            mapbox_style="carto-positron",
            mapbox_zoom=1,
            mapbox_center={"lat": 0, "lon": 0},
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.PrevalenceEstimates, PlotType.SPATIAL_SURFACE)
class SpatialSurfacePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', lat_col: str = 'lat', lon_col: str = 'lon'):
        """
        Plots a continuous heatmap surface.
        Designed specifically for the dense grid output of the SpatialPrevalenceEstimator.
        """
        data = result.data.copy()

        if lat_col not in data.columns or lon_col not in data.columns:
            raise ValueError(f"Spatial surface requires '{lat_col}' and '{lon_col}' columns in the data.")

        hover_text = (
            "<b>Lat</b>: " + data[lat_col].round(3).astype(str) +
            " | <b>Lon</b>: " + data[lon_col].round(3).astype(str) +
            "<br><b>Predicted Prevalence</b>: " + data['estimate'].map('{:.2%}'.format)
        )

        # Smooth gradient from Transparent -> Cyan -> Neon Pink
        custom_colorscale = [
            [0.0, 'rgba(0,0,0,0)'],     # Transparent
            [0.5, cls._MAIN_COLOUR],    # Electric Cyan (Mid-prevalence)
            [1.0, cls._ACCENT_COLOUR]   # Neon Pink (High-prevalence hotspots)
        ]

        fig = go.Figure(go.Densitymapbox(
            lat=data[lat_col],
            lon=data[lon_col],
            z=data['estimate'],
            radius=25,  # Size of the spatial smoothing kernel
            colorscale=custom_colorscale,
            zmin=0, zmax=1,
            opacity=0.85,
            text=hover_text,
            hoverinfo='text',
            colorbar_title='Predicted<br>Prevalence'
        ))

        title_prefix = "Spatial Composition of" if result.aggregation_type == AggregationType.COMPOSITIONAL else "Spatial Surface for"

        return fig.update_layout(
            title=f"<b>{title_prefix} {result.target}</b><br><sup>Gaussian Process Prediction Surface</sup>",
            mapbox_style="carto-positron", # Clean neutral map
            mapbox_zoom=3,
            mapbox_center={"lat": data[lat_col].mean(), "lon": data[lon_col].mean()},
            margin={"r": 0, "t": 60, "l": 0, "b": 0},
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(estimators.AlphaDiversityEstimates, PlotType.ALPHA_DIVERSITY)
class AlphaDiversityPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.AlphaDiversityEstimates',  metric: str = 'shannon',
                    sort_ascending: bool = False) -> go.Figure:
        """
        Plots a glowing lollipop chart for within-group diversity.
        """
        if metric not in result.metrics:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {result.metrics}")

        data = result.data.copy()

        # Identify the primary stratification column
        group_col = result.stratified_by[0] if result.stratified_by else 'Global'

        if sort_ascending is not None:
            data = data.sort_values(by=metric, ascending=sort_ascending)

        fig = go.Figure()

        # 1. The Stems (Thin glowing lines)
        # We use go.Bar with an ultra-thin width to create perfect vertical lines down to 0
        fig.add_trace(go.Bar(
            x=data[group_col],
            y=data[metric],
            width=0.02,  # Ultra-thin stem
            marker_color='rgba(14, 165, 233, 0.4)',  # Translucent Cyan
            hoverinfo='skip',
            showlegend=False
        ))

        # 2. The Lollipops (Neon markers)
        fig.add_trace(go.Scatter(
            x=data[group_col],
            y=data[metric],
            mode='markers',
            marker=dict(
                size=12,
                color=cls._MAIN_COLOUR
            ),
            name=metric.title(),
            hovertemplate=f"<b>%{{x}}</b><br>{metric.title()}: %{{y:.3f}}<br>Sequenced (n): %{{customdata}}<extra></extra>",
            customdata=data['n_samples']
        ))

        return fig.update_layout(
            title=f"<b>Alpha Diversity ({metric.title()})</b><br><sup>Target: {result.target}</sup>",
            xaxis=dict(title=group_col.title() if group_col != 'Global' else ''),
            yaxis=dict(title=f'{metric.title()} Index', rangemode='tozero'),  # Stems must anchor to 0
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )



@BasePlotter.register_plotter(estimators.BetaDiversityEstimates, PlotType.BETA_HEATMAP)
class BetaHeatmapPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'estimators.BetaDiversityEstimates', mask_upper: bool = True) -> go.Figure:
        """
        Plots an N x N dissimilarity matrix.
        If mask_upper is True, perfectly formats it as an academic lower-triangle heatmap.
        """
        dist_matrix = result.data.copy()

        # The Pro-Move: Mask the upper triangle and the diagonal
        if mask_upper:
            # np.triu returns the upper triangle. We set those coordinates to NaN.
            mask = np.triu(np.ones(dist_matrix.shape, dtype=bool))
            dist_matrix = dist_matrix.mask(mask)

        # Hover text configuration
        labels = dist_matrix.columns.tolist()

        # Distances Colorscale
        distance_colorscale = [
            [0.0, 'rgba(0,0,0,0)'],  # Transparent
            [0.5, '#8B5CF6'],  # Deep Purple transition
            [1.0, cls._ACCENT_COLOUR]  # Neon Pink
        ]

        fig = go.Figure(data=go.Heatmap(
            z=dist_matrix.values,
            x=labels,
            y=labels,
            colorscale=distance_colorscale,
            zmin=0,
            zmax=1,  # Bray-Curtis and Jaccard are naturally bounded 0-1
            hoverongaps=False,  # Prevents hovering over the NaN upper triangle
            hovertemplate="<b>Group 1:</b> %{y}<br><b>Group 2:</b> %{x}<br><b>Dissimilarity:</b> %{z:.3f}<extra></extra>",
            colorbar=dict(title=f"{result.metric.title()}<br>Distance")
        ))

        return fig.update_layout(
            title=f"<b>Beta Diversity ({result.metric.title()})</b><br><sup>Target: {result.target_trait}</sup>",
            xaxis=dict(title='', tickangle=45),
            yaxis=dict(title='', autorange='reversed'),  # Reversing Y puts the triangle in the correct academic orientation
            margin=dict(l=20, r=20, t=60, b=80),  # Extra bottom margin for angled text
            width=700,
            height=700,  # Force a square aspect ratio so the matrix cells are perfect squares
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )


@BasePlotter.register_plotter(Formulation, PlotType.STABILITY_BUMP)
class StabilityBumpPlotter(BasePlotter):

    @classmethod
    def render(cls, formulation: 'Formulation', **kwargs) -> go.Figure:
        """
        Plots a Slopegraph (Bump Chart) showing how target ranks shift
        when different data groups are held out.
        """
        history = formulation.permutation_history.copy()
        baseline = formulation.rankings.copy()
        valency = formulation.max_valency

        fig = go.Figure()

        # We only want to plot the lines for targets that actually made it into the baseline formulation
        top_targets = baseline.head(valency)['target'].tolist()

        # X-axis categories: Baseline first, then all the holdout permutations
        x_categories = ['Baseline'] + history['holdout_group'].unique().tolist()

        for target in top_targets:
            v_history = history[history['target'] == target]

            # Build the Y-coordinates (Ranks) mapping to the X-coordinates
            y_ranks = [baseline[baseline['target'] == target]['baseline_rank'].values[0]]
            for group in x_categories[1:]:
                rank = v_history[v_history['holdout_group'] == group]['loo_rank'].values
                y_ranks.append(rank[0] if len(rank) > 0 else np.nan)

            fig.add_trace(go.Scatter(
                x=x_categories,
                y=y_ranks,
                mode='lines+markers',
                name=str(target),
                line=dict(width=4),
                marker=dict(size=10, color=cls._MAIN_COLOUR),
                hovertemplate="<b>Holdout: %{x}</b><br>Rank: %{y}<extra></extra>"
            ))

        # Draw a subtle horizontal line representing the "Valency Cutoff"
        fig.add_hline(
            y=valency + 0.5,
            line_dash="dot",
            line_color="#EF4444",  # Red warning line
            annotation_text="Valency Cutoff",
            annotation_position="bottom right"
        )

        return fig.update_layout(
            title=f"<b>Target Priority Stability (LOO)</b><br><sup>Targeting Top {valency} {formulation.target} Variants</sup>",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        ).update_xaxes(
            title="Excluded Group (Leave-One-Out)"
        ).update_yaxes(
            title="Priority Rank",
            autorange='reversed',  # Crucial: Rank 1 must be at the TOP of the Y-axis!
            dtick=1
        )


@BasePlotter.register_plotter(Distances, PlotType.NETWORK)
class NetworkPlotter(BasePlotter):

    @classmethod
    def render(cls, result: Distances, df: pd.DataFrame = None, pos: np.ndarray = None,
               edge_type: str = 'snp', threshold: int = 20, trans_col: str = None,
               color_col: str = None, **kwargs) -> go.Figure:
        """
        Plots an interactive force-directed network using MDS coordinates.
        """
        dense_dist = result.matrix.toarray().astype(float)

        # Fallback to calculate MDS if it wasn't passed in via a cache
        if pos is None:
            from sklearn.manifold import MDS
            mask = (dense_dist == 0) & (~np.eye(dense_dist.shape[0], dtype=bool))
            if mask.any():
                dense_dist[mask] = dense_dist.max() * 2
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=1, max_iter=100)
            pos = mds.fit_transform(dense_dist)

        # Align the dataframe rows to perfectly match the distance matrix index
        if df is not None and 'sample_id' in df.columns:
            df_aligned = df.set_index('sample_id').reindex(result.index)
        else:
            df_aligned = pd.DataFrame(index=result.index)

        edge_x, edge_y = [], []
        title = "<b>Isolate Network</b><br><sup>Nodes positioned by SNP distance (MDS)</sup>"

        if edge_type == "snp":
            adj = (dense_dist <= threshold).astype(int)
            np.fill_diagonal(adj, 0)
            rows, cols = np.where(adj == 1)
            for r, c in zip(rows, cols):
                if r < c:
                    edge_x.extend([pos[r, 0], pos[c, 0], None])
                    edge_y.extend([pos[r, 1], pos[c, 1], None])
            title = f"<b>Genomic SNP Network</b><br><sup>Edges connect isolates ≤ {threshold} SNPs apart</sup>"

        elif edge_type == "trans" and trans_col and trans_col in df_aligned.columns:
            cluster_assignments = df_aligned[trans_col].values
            unique_clusters = df_aligned[trans_col].dropna().unique()
            for c in unique_clusters:
                idx = np.where(cluster_assignments == c)[0]
                for i in range(len(idx)):
                    for j in range(i + 1, len(idx)):
                        edge_x.extend([pos[idx[i], 0], pos[idx[j], 0], None])
                        edge_y.extend([pos[idx[i], 1], pos[idx[j], 1], None])
            title = f"<b>Transmission Network</b><br><sup>Edges connect isolates in {trans_col.replace('_', ' ').title()}</sup>"

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.4, color='#888'), hoverinfo='none', mode='lines')

        if color_col and color_col in df_aligned.columns:
            color_vals = df_aligned[color_col].fillna("Unknown").astype(str)
            color_map = {c: i for i, c in enumerate(color_vals.unique())}
            colors = [color_map[c] for c in color_vals]
            hover_text = [f"ID: {idx}<br>{color_col}: {c}" for idx, c in zip(result.index, color_vals)]
            marker_dict = dict(showscale=False, color=colors, colorscale='Turbo', size=10, line=dict(width=1, color='white'))
        else:
            hover_text = [f"ID: {idx}" for idx in result.index]
            marker_dict = dict(color=cls._MAIN_COLOUR, size=10, line=dict(width=1, color='white'))

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode='markers',
            hovertext=hover_text, hoverinfo="text", marker=marker_dict
        )

        return go.Figure(data=[edge_trace, node_trace]).update_layout(
            title=title,
            showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(b=20, l=5, r=5, t=60),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
