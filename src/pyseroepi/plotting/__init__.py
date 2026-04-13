import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyseroepi.plotting._base import BasePlotter
from typing import TYPE_CHECKING
if TYPE_CHECKING:  # Delay import to avoid circular dependencies
    from pyseroepi.estimators._base import (PrevalenceEstimates, AlphaDiversityEstimates, BetaDiversityEstimates,
                                            IncidenceEstimates)
    from pyseroepi.formulation import Formulation

# Classes --------------------------------------------------------------------------------------------------------------
@BasePlotter.register_plotter(PrevalenceEstimates, 'forest')
class ForestPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates', top_n: int = None, sort_ascending: bool = False):
        """Generates a dot-and-whisker forest plotting for prevalence data."""
        data = result.data.copy()
        y_col_name = result.stratified_by[0]

        data = data.sort_values(by='estimate', ascending=sort_ascending)
        if top_n:
            data = data.head(top_n)

        hover_text = (
                f"<b>{y_col_name}</b>: " + data[y_col_name].astype(str) +
                "<br><b>Estimate</b>: " + data['estimate'].map('{:.2%}'.format) +
                "<br><b>95% CI</b>: [" + data['lower'].map('{:.2%}'.format) + ", " + data['upper'].map(
            '{:.2%}'.format) + "]" +
                f"<br><b>Events (n)</b>: " + data['event'].astype(str) + " (" + data['n'].astype(str) + ")"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['estimate'], y=data[y_col_name], mode='markers', name="Prevalence",
            hoverinfo='text', hovertext=hover_text,
            marker=dict(color=cls._MAIN_COLOUR, size=10, symbol='circle'),
            error_x=dict(
                type='data', symmetric=False,
                array=data['upper'] - data['estimate'],
                arrayminus=data['estimate'] - data['lower'],
                color=cls._MAIN_COLOUR, thickness=2, width=4
            )
        ))

        title_prefix = "Composition of" if result.prevalence_type == "compositional" else "Prevalence of"

        return fig.update_layout(
            dict1=cls._THEME,
            title=f"<b>{title_prefix} {result.target}</b><br><sup>Method: {result.method}</sup>",
            yaxis=dict(title=y_col_name, categoryorder='array'),
            xaxis=dict(title='Prevalence (95% CI)', tickformat='.0%', range=[0, 1]),
            showlegend=False
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'epicurve')
class EpicurvePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'IncidenceEstimates', **kwargs):
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
            dict1=cls._THEME,
            title=f"<b>Epidemic Curve: {result.target}</b>{subtitle}",
            xaxis=dict(title="Date", type='date'),
            yaxis=dict(title='Count'),
            barmode='overlay',  # Allows the bars and the area chart to overlap safely
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(color="#94A3B8"), bgcolor="rgba(0,0,0,0)"
            )
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'longitudinal_prevalence')
class LongitudinalPrevalencePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates', **kwargs):
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
            marker=dict(size=8, color='#0F172A', line=dict(width=2, color=cls._MAIN_COLOUR)),
            name="Prevalence",
            hovertemplate="<b>Date</b>: %{x}<br><b>Prevalence</b>: %{y:.2%}<extra></extra>"
        ))

        return fig.update_layout(
            dict1=cls._THEME,
            title=f"<b>Longitudinal Prevalence of {result.target}</b>",
            xaxis=dict(title=time_col.title()),
            yaxis=dict(title='Prevalence', tickformat='.0%', range=[0, 1.05])
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'vaccine_coverage')
class VaccineCoveragePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates',  max_valencies: int = None):
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
            # Using the dark slate background color for the inside of the marker to create a neon ring effect
            marker=dict(size=8, color='#0F172A', line=dict(width=2, color=cls._MAIN_COLOUR)),
            text=hover_text, hoverinfo='text'
        ))

        return fig.update_layout(
            dict1=cls._THEME,
            title=f"<b>Cumulative Vaccine Coverage</b><br><sup>Targeting top {len(data)} {target_col} variants</sup>",
            xaxis=dict(title=f"Variant ({target_col}) added to formulation", tickangle=45),
            yaxis=dict(title='Cumulative Population Coverage', tickformat='.0%', range=[0, 1.05]),
            hovermode="x unified"
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'choropleth')
class ChoroplethPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates', geo_col: str, target_variant: str = None,
               feature_id_key: str = "properties.NAME"):
        """
        Plots discrete regional prevalence on a map.
        """
        data = result.data.copy()

        if geo_col not in result.stratified_by:
            raise ValueError(f"Cannot plotting choropleth: '{geo_col}' was not used as a stratification variable.")

        # --- THE FIX: Isolate the specific variant for compositional data ---
        target_name = result.target
        if result.prevalence_type == "compositional":
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
            [0.0, '#1E293B'],  # Slate-800
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

        title_prefix = "Regional Composition of" if result.prevalence_type == "compositional" else "Regional Prevalence of"

        return fig.update_layout(
            dict1=cls._THEME,
            # Dynamically updates the title so the user knows exactly what is mapped!
            title=f"<b>{title_prefix} {target_name}</b>",
            mapbox_style="carto-darkmatter",
            mapbox_zoom=1,
            mapbox_center={"lat": 0, "lon": 0},
            margin={"r": 0, "t": 60, "l": 0, "b": 0}
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'spatial_surface')
class SpatialSurfacePlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates', lat_col: str = 'lat', lon_col: str = 'lon'):
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

        # The Bioluminescence Colorscale
        # Smooth gradient from the dark map background -> Cyan -> Neon Pink
        custom_colorscale = [
            [0.0, '#1E293B'],      # Slate-800 (Invisible against the dark UI)
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

        title_prefix = "Spatial Composition of" if result.prevalence_type == "compositional" else "Spatial Surface for"

        return fig.update_layout(
            dict1=cls._THEME,
            title=f"<b>{title_prefix} {result.target}</b><br><sup>Gaussian Process Prediction Surface</sup>",
            mapbox_style="carto-darkmatter", # Blends seamlessly with the dark theme
            mapbox_zoom=3,
            mapbox_center={"lat": data[lat_col].mean(), "lon": data[lon_col].mean()},
            margin={"r": 0, "t": 60, "l": 0, "b": 0}
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'alpha_diversity')
class AlphaDiversityPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates',  metric: str = 'shannon',
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
                color='#0F172A',  # Hollow center matching background
                line=dict(width=3, color=cls._MAIN_COLOUR)  # Glowing cyan border
            ),
            name=metric.title(),
            hovertemplate=f"<b>%{{x}}</b><br>{metric.title()}: %{{y:.3f}}<br>Sequenced (n): %{{customdata}}<extra></extra>",
            customdata=data['n_samples']
        ))

        return fig.update_layout(
            dict1=cls._THEME,
            title=f"<b>Alpha Diversity ({metric.title()})</b><br><sup>Target: {result.target_trait}</sup>",
            xaxis=dict(title=group_col.title() if group_col != 'Global' else ''),
            yaxis=dict(title=f'{metric.title()} Index', rangemode='tozero'),  # Stems must anchor to 0
            showlegend=False
        )



@BasePlotter.register_plotter(PrevalenceEstimates, 'beta_heatmap')
class BetaHeatmapPlotter(BasePlotter):

    @classmethod
    def render(cls, result: 'PrevalenceEstimates', mask_upper: bool = True) -> go.Figure:
        """
        Plots an N x N dissimilarity matrix.
        If mask_upper is True, perfectly formats it as an academic lower-triangle heatmap.
        """
        dist_matrix = result.distance_matrix.copy()

        # The Pro-Move: Mask the upper triangle and the diagonal
        if mask_upper:
            # np.triu returns the upper triangle. We set those coordinates to NaN.
            mask = np.triu(np.ones(dist_matrix.shape, dtype=bool))
            dist_matrix = dist_matrix.mask(mask)

        # Hover text configuration
        labels = dist_matrix.columns.tolist()

        # The Bioluminescence Colorscale for Distances
        # 0 = Identical (Dark Slate, fades into background)
        # 1 = Completely Dissimilar (Neon Pink Hotspot)
        distance_colorscale = [
            [0.0, '#1E293B'],  # Slate-800
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
            dict1=cls._THEME,
            title=f"<b>Beta Diversity ({result.metric.title()})</b><br><sup>Target: {result.target_trait}</sup>",
            xaxis=dict(title='', tickangle=45),
            yaxis=dict(title='', autorange='reversed'),  # Reversing Y puts the triangle in the correct academic orientation
            margin=dict(l=20, r=20, t=60, b=80),  # Extra bottom margin for angled text
            width=700,
            height=700  # Force a square aspect ratio so the matrix cells are perfect squares
        )


@BasePlotter.register_plotter(PrevalenceEstimates, 'rank_stability_bump')
class RankStabilityBumpPlotter(BasePlotter):

    @classmethod
    def render(cls, formulation: 'Formulation') -> go.Figure:
        """
        Plots a Slopegraph (Bump Chart) showing how antigen ranks shift
        when different data groups are held out.
        """
        history = formulation.permutation_history.copy()
        baseline = formulation.rankings.copy()
        valency = formulation.max_valency

        fig = go.Figure()

        # We only want to plotting the lines for antigens that actually made it into the baseline formulation
        top_antigens = baseline.head(valency)['antigen'].tolist()

        # X-axis categories: Baseline first, then all the holdout permutations
        x_categories = ['Baseline'] + history['holdout_group'].unique().tolist()

        for antigen in top_antigens:
            v_history = history[history['antigen'] == antigen]

            # Build the Y-coordinates (Ranks) mapping to the X-coordinates
            y_ranks = [baseline[baseline['antigen'] == antigen]['baseline_rank'].values[0]]
            for group in x_categories[1:]:
                rank = v_history[v_history['holdout_group'] == group]['loo_rank'].values
                y_ranks.append(rank[0] if len(rank) > 0 else np.nan)

            fig.add_trace(go.Scatter(
                x=x_categories,
                y=y_ranks,
                mode='lines+markers',
                name=str(antigen),
                line=dict(width=4),
                marker=dict(size=10, color='#0F172A', line=dict(width=2)),  # Midnight hollow ring effect
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
            **cls._THEME,
            title=f"<b>Antigen Priority Stability (LOO)</b><br><sup>Targeting Top {valency} {formulation.target} Variants</sup>",
            hovermode="x unified",
        ).update_xaxes(
            title="Excluded Group (Leave-One-Out)"
        ).update_yaxes(
            title="Priority Rank",
            autorange='reversed',  # Crucial: Rank 1 must be at the TOP of the Y-axis!
            dtick=1
        )

