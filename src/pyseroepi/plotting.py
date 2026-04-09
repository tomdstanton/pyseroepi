import numpy as np
import pandas as pd
import plotly.graph_objects as go


class PrevalenceEstimatesPlotter:
    """Namespace for generating interactive and static plots from PrevalenceEstimates."""

    # A cleaner, more academic theme suitable for both Shiny and static PDF export
    _DEFAULT_THEME = {
        'template': 'simple_white',
        'font': dict(family="Helvetica, Arial, sans-serif", size=13, color="#2c3e50"),
        'margin': dict(l=20, r=20, t=50, b=20),
        'hoverlabel': dict(bgcolor="white", font_size=13, font_family="Helvetica")
    }

    _MAIN_COLOUR = '#1E88E5'  # A nice, colorblind-friendly academic blue
    _CI_COLOUR = 'rgba(30, 136, 229, 0.2)'  # Transparent blue for confidence ribbons
    _MAX_X = 20

    def __init__(self, result: 'PrevalenceEstimates'):
        self.estimates = result
        self._theme = self._DEFAULT_THEME
        self._top_n = None
        self._sort_ascending = False

    def with_theme(self, theme_dict: dict) -> 'PrevalenceEstimatesPlotter':
        """Updates the theme state and returns self for chaining."""
        self._theme = theme_dict
        return self

    def top(self, n: int) -> 'PrevalenceEstimatesPlotter':
        """Limits the data to the top N categories."""
        self._top_n = n
        return self

    def sort(self, ascending: bool = True) -> 'PrevalenceEstimatesPlotter':
        """Sets the sorting direction."""
        self._sort_ascending = ascending
        return self

    def forest(self) -> go.Figure:
        """
        The epidemiologically rigorous replacement for the bar/pyramid plot.
        Uses dot-and-whisker to represent point estimates and 95% intervals.
        """
        data = self.estimates.data.copy()
        y_col_name = self.estimates.stratified_by[0]

        # 1. Apply internal state filters before plotting
        data = data.sort_values(by='estimate', ascending=self._sort_ascending)
        if self._top_n:
            data = data.head(self._top_n)

        hover_text = (
                f"<b>{y_col_name}</b>: " + data[y_col_name].astype(str) +
                "<br><b>Estimate</b>: " + data['estimate'].map('{:.2%}'.format) +
                "<br><b>95% CI</b>: [" + data['lower'].map('{:.2%}'.format) + ", " + data['upper'].map(
            '{:.2%}'.format) + "]" +
                f"<br><b>Events (n)</b>: " + data['event'].astype(str) + " (" + data['n'].astype(str) + ")"
        )

        fig = go.Figure()

        # Add the scatter points with error bars (The "Whiskers")
        fig.add_trace(go.Scatter(
            x=data['estimate'],
            y=data[y_col_name],
            mode='markers',
            name="Prevalence",
            hoverinfo='text',
            hovertext=hover_text,
            marker=dict(color=self._MAIN_COLOUR, size=10, symbol='circle'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=data['upper'] - data['estimate'],
                arrayminus=data['estimate'] - data['lower'],
                color=self._MAIN_COLOUR,
                thickness=2,
                width=4
            )
        ))

        title = "Composition of" if self.estimates.prevalence_type == "compositional" else "Prevalence of"

        return fig.update_layout(
            **self._theme,
            title=f"<b>{title} {self.estimates.target}</b><br><sup>Method: {self.estimates.method}</sup>",
            yaxis=dict(title=y_col_name, categoryorder='array'),
            xaxis=dict(title='Prevalence (95% CI)', tickformat='.0%', range=[0, 1]),
            showlegend=False
        )

    def vaccine_coverage(self, max_valencies: int = None) -> go.Figure:
        """
        Calculates cumulative population coverage.
        Crucial for designing multivalent vaccines (e.g., K-locus targeting).
        """
        data = self.estimates.data.copy()
        target_col = self.estimates.stratified_by[0]

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
            mode='lines', fill='tonexty', fillcolor=self._CI_COLOUR,
            line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))

        # The actual cumulative line and markers
        fig.add_trace(go.Scatter(
            x=data[target_col], y=data['cum_prop'],
            mode='lines+markers',
            name='Cumulative Coverage',
            line=dict(color=self._MAIN_COLOUR, width=3),
            marker=dict(size=8, color='white', line=dict(width=2, color=self._MAIN_COLOUR)),
            text=hover_text, hoverinfo='text'
        ))

        return fig.update_layout(
            **self._theme,
            title=f"<b>Cumulative Vaccine Coverage</b><br><sup>Targeting top {len(data)} {target_col} variants</sup>",
            xaxis=dict(title=f"Variant ({target_col}) added to formulation", tickangle=45),
            yaxis=dict(title='Cumulative Population Coverage', tickformat='.0%', range=[0, 1.05]),
            hovermode="x unified"
        )

    def epicurve(self) -> go.Figure:
        """
        Plots a time-series line chart if temporal data was used as a stratification.
        """
        # Intelligently check if a date/time column exists in the strata
        time_cols = [col for col in self.estimates.stratified_by if
                     'date' in col.lower() or 'year' in col.lower() or 'month' in col.lower()]

        if not time_cols:
            raise ValueError(
                "No temporal column detected in the strata. Ensure you stratified by a Date, Year, or Month column.")

        time_col = time_cols[0]
        data = self.estimates.data.sort_values(by=time_col)

        fig = go.Figure()

        # Add the confidence band
        fig.add_trace(go.Scattergl(
            x=pd.concat([data[time_col], data[time_col][::-1]]),
            y=pd.concat([data['upper'], data['lower'][::-1]]),
            fill='toself',
            fillcolor=self._CI_COLOUR,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))

        # Add the main trend line
        fig.add_trace(go.Scatter(
            x=data[time_col], y=data['estimate'],
            mode='lines+markers',
            line=dict(color=self._MAIN_COLOUR, width=3),
            marker=dict(size=6),
            name="Prevalence",
            hovertemplate="<b>Date</b>: %{x}<br><b>Prevalence</b>: %{y:.2%}<extra></extra>"
        ))

        return fig.update_layout(
            **self._theme,
            title=f"<b>Longitudinal Prevalence of {self.estimates.target}</b>",
            xaxis=dict(title="Time"),
            yaxis=dict(title='Prevalence', tickformat='.0%', range=[0, 1.05])
        )

    def choropleth(self, geojson_data, geo_col: str, feature_id_key: str = "properties.name") -> go.Figure:
        """
        Plots discrete regions. Best for Frequentist/Bayesian results stratified by region.

        geojson_data: A parsed GeoJSON dictionary or GeoPandas __geo_interface__
        geo_col: The column in your estimates.data that matches the GeoJSON features
        feature_id_key: The path in the GeoJSON to match geo_col against
        """
        data = self.estimates.data.copy()

        # Verify the requested column exists in the strata
        if geo_col not in self.estimates.stratified_by:
            raise ValueError(f"Cannot plot choropleth: '{geo_col}' was not used as a stratification variable.")

        hover_text = (
                f"<b>{geo_col}</b>: " + data[geo_col].astype(str) +
                "<br><b>Prevalence</b>: " + data['estimate'].map('{:.2%}'.format) +
                "<br><b>95% CI</b>: [" + data['lower'].map('{:.2%}'.format) + ", " + data['upper'].map(
            '{:.2%}'.format) + "]"
        )

        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=data[geo_col],
            featureidkey=feature_id_key,
            z=data['estimate'],
            colorscale=[self._MAIN_COLOUR, '#0D47A1'],  # Light blue to deep navy
            zmin=0, zmax=1,
            marker_opacity=0.75,
            marker_line_width=0.5,
            marker_line_color='white',
            colorbar_title='Prevalence',
            text=hover_text,
            hoverinfo='text'
        ))

        title = "Regional Composition of" if self.estimates.prevalence_type == "compositional" else "Regional Prevalence of"

        return fig.update_layout(
            **self._theme,
            title=f"<b>{title} {self.estimates.target}</b>",
            mapbox_style="carto-positron",  # Clean, white/grey academic map background
            mapbox_zoom=1,
            mapbox_center={"lat": 0, "lon": 0},
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )

    def spatial_surface(self, lat_col: str = 'lat', lon_col: str = 'lon') -> go.Figure:
        """
        Plots a continuous heatmap surface.
        Designed specifically for the dense grid output of the SpatialPrevalenceEstimator.
        """
        if lat_col not in self.estimates.data.columns or lon_col not in self.estimates.data.columns:
            raise ValueError("Spatial surface requires latitude and longitude columns in the data.")

        data = self.estimates.data.copy()

        hover_text = (
                "<b>Lat</b>: " + data[lat_col].round(3).astype(str) +
                " | <b>Lon</b>: " + data[lon_col].round(3).astype(str) +
                "<br><b>Predicted Prevalence</b>: " + data['estimate'].map('{:.2%}'.format)
        )

        # go.Densitymapbox creates a smooth, radiating heatmap over the map
        # It is perfect for visualizing Gaussian Process predictions
        fig = go.Figure(go.Densitymapbox(
            lat=data[lat_col],
            lon=data[lon_col],
            z=data['estimate'],
            radius=25,  # Size of the smoothing kernel
            colorscale='Viridis',  # A scientifically perceptually uniform colormap
            zmin=0, zmax=1,
            opacity=0.8,
            text=hover_text,
            hoverinfo='text',
            colorbar_title='Predicted<br>Prevalence'
        ))

        return fig.update_layout(
            **self._theme,
            title=f"<b>Continuous Spatial Map: {self.estimates.target}</b><br><sup>Gaussian Process Prediction Surface</sup>",
            mapbox_style="carto-positron",
            mapbox_zoom=3,
            # Dynamically center the map on the data
            mapbox_center={"lat": data[lat_col].mean(), "lon": data[lon_col].mean()},
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )