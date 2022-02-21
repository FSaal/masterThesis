#!usr/bin/env python

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class PlotLib(object):
    def __init__(self, df, df_stats, df_eval, x_range, y_range, all_points, ramp_indices):
        """Visualize different stuff from lidar algo

        Args:
            df (Dataframe): Location and bool if inside or outside of ramp region for every lidar point
            df_stats (Dataframe): Angle, width, dist to ramp every time a ramp has been detected by algo
            df_eval (Dataframe): How many times ramp has been detected vs how many samples in certain ranges
            x_range (list): Lower and upper x-limit of ramp (from pcd map)
            y_range (list): Lower and upper y_limit of ramp (from pcd map)
        """
        self.df = df
        self.df_stats = df_stats
        self.df_eval = df_eval
        self.x_range = x_range
        self.y_range = y_range
        self.all_points = all_points
        self.true_inliers = df_stats["TrueInliers"]
        self.ramp_indices = ramp_indices

    def plot_for_latex(self, x=None, y=None, mode="lines+markers"):
        """Layout template for exporting plots to latex"""
        fig = go.Figure()
        if x is not None:
            fig.add_trace(go.Scatter(x=x, y=y, mode=mode))
        fig.update_layout(title="Difference between estimated and actual angle of ramp")
        fig.update_xaxes(
            title_text="Distance to ramp [m]",
            autorange="reversed",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )
        fig.update_yaxes(
            title_text="?",
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
        )
        fig.update_layout(
            width=600,
            height=300,
            font_family="Serif",
            font_size=14,
            font_color="black",
            margin_l=5,
            margin_t=5,
            margin_b=5,
            margin_r=5,
            title="",
            legend=dict(x=0.01, y=0.01, traceorder="normal", bordercolor="Gray", borderwidth=1),
        )
        # fig.show()
        return fig

    def distance_estimation(self, show_statistics=True):
        """Plot estimated distance to ramp from algo, compared to ground truth (hdl_slam)"""
        fig = go.Figure()
        # Distance from HDL slam against distance from HDL slam
        fig.add_trace(
            go.Scatter(
                x=self.df_stats["TrueDist"], y=self.df_stats["TrueDist"], name="Measured (hdl)"
            )
        )
        # Distance from lidar ramp detection algorithm against distance from HDL slam
        fig.add_trace(
            go.Scatter(x=self.df_stats["TrueDist"], y=self.df_stats["Dist"], name="Estimated")
        )
        fig.update_traces(mode="lines+markers")
        fig.update_layout(title="Difference between estimated and actual distance to ramp")
        fig.update_xaxes(title_text="(actual) Distance [m]", autorange="reversed")
        fig.update_yaxes(title_text="Distance [m]")
        if show_statistics:
            # Difference between estimation and true value
            diff = self.df_stats["Dist"] - self.df_stats["TrueDist"]
            fig.add_annotation(
                text="Standard deviation: {:.2f}m<br>Average error: {:.2f}m<br>Median error: {:.2f}m".format(
                    np.std(diff), np.mean(diff), np.median(diff)
                ),
                xref="paper",
                yref="paper",
                x=1,
                y=1,
                showarrow=False,
            )
        return fig

    def angle_estimation(self, true_angle=6, show_statistics=True):
        """Plot estimated angle of ramp from algo, compared to ground truth"""
        fig = go.Figure()
        # Angle from measuring by hand against distance from HDL slam
        fig.add_trace(
            go.Scatter(
                x=self.df_stats["TrueDist"],
                y=true_angle * np.ones(len(self.df_stats)),
                name="Measured",
                mode="lines",
            )
        )
        # Angle from lidar ramp detection algorithm against distance from HDL slam
        fig.add_trace(
            go.Scatter(
                x=self.df_stats["TrueDist"],
                y=self.df_stats["Angle"],
                name="Estimated",
                mode="lines+markers",
            )
        )
        fig.update_layout(title="Difference between estimated and actual angle of ramp")
        fig.update_xaxes(title_text="Distance to ramp [m]", autorange="reversed")
        fig.update_yaxes(title_text="Ramp angle [deg]")
        if show_statistics:
            # Difference between estimation and true value
            diff = self.df_stats["Angle"] - true_angle
            fig.add_annotation(
                text="Average: {:.2f}<br>Standard deviation: {:.2f}deg<br>Average error: {:.2f}deg<br>Median error: {:.2f}deg".format(
                    np.mean(self.df_stats["Angle"]), np.std(diff), np.mean(diff), np.median(diff)
                ),
                xref="paper",
                yref="paper",
                x=1,
                y=1,
                showarrow=False,
            )
        return fig

    def width_estimation(self, drivable_width=2.9, show_stats=True):
        """Plot estimated angle of ramp from algo, compared to ground truth"""
        # Whole width (including curb)
        true_width = self.y_range[-1] - self.y_range[0]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.df_stats["TrueDist"],
                y=true_width * np.ones(len(self.df_stats)),
                name="Measured (whole)",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.df_stats["TrueDist"],
                y=drivable_width * np.ones(len(self.df_stats)),
                name="Measured (drivable)",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.df_stats["TrueDist"],
                y=self.df_stats["Width"],
                name="Estimated",
                mode="lines+markers",
            )
        )
        fig.update_layout(title="Difference between estimated and actual width of ramp")
        fig.update_xaxes(title_text="Distance to ramp [m]", autorange="reversed")
        fig.update_yaxes(title_text="Ramp width [m]")
        if show_stats:
            # Difference between estimation and true value (whole ramp)
            diff = self.df_stats["Width"] - true_width
            # Difference between estimation and true value (only drivable part of ramp)
            diff_drive = self.df_stats["Width"] - drivable_width
            fig.add_annotation(
                text="Average: {:.2f}<br>Standard deviation: {:.2f}m<br>Average error: {:.2f}m<br>Median error: {:.2f}m".format(
                    np.mean(self.df_stats["Width"]), np.std(diff), np.mean(diff), np.median(diff)
                ),
                xref="paper",
                yref="paper",
                x=1,
                y=1,
                showarrow=False,
            )
            fig.add_annotation(
                text="Standard deviation: {:.2f}m<br>Average error: {:.2f}m<br>Median error: {:.2f}m".format(
                    np.std(diff_drive), np.mean(diff_drive), np.median(diff_drive)
                ),
                xref="paper",
                yref="paper",
                x=1,
                y=0,
                showarrow=False,
            )
        return fig

    @staticmethod
    def add_buffer(range, margin=3):
        for i, v in enumerate(range):
            if i == 0:
                low = int(round(v)) - margin
            else:
                high = int(round(v)) + margin
        return [low, high]

    def animation_only_detections(self):
        """Every"""
        # Add some buffer to the true ramp region (for tidy plots)
        x_fixed = self.add_buffer(self.x_range)
        y_fixed = self.add_buffer(self.y_range)

        fig = go.Figure()
        trace_list1 = []
        trace_list2 = []
        # Add traces, one for each slider step
        for step in range(self.df["sampleIdx"].max() + 1):
            trace_list1.append(
                go.Scatter(
                    visible=False,
                    mode="markers",
                    x=self.df[self.df["sampleIdx"] == step]["x"],
                    y=self.df[self.df["sampleIdx"] == step]["y"],
                    name="Detected by algorithm as ramp",
                )
            )
            trace_list2.append(
                go.Scatter(
                    visible=False,
                    mode="markers",
                    marker=dict(opacity=0.5),
                    x=self.all_points[self.ramp_indices[step]][:, 0],
                    y=self.all_points[self.ramp_indices[step]][:, 1],
                    name="All lidar points",
                )
            )
        fig = go.Figure(data=trace_list2 + trace_list1)
        fig.data[0].visible = True
        fig.add_shape(
            type="rect",
            x0=self.x_range[0],
            x1=self.x_range[1],
            y0=self.y_range[0],
            y1=self.y_range[1],
            fillcolor="red",
            opacity=0.2,
        )
        # Set static axes limits
        fig.update_xaxes(range=x_fixed)
        fig.update_yaxes(range=y_fixed)

        # Create and add slider
        steps = []
        for i in range(len(fig.data) / 2):
            step = dict(
                method="update",
                args=[
                    {"visible": [False] * len(fig.data)},
                    {
                        "title": "{:.2f}% of detected points lie in the ramp region. Was detected {:.2f}m infront of ramp".format(
                            self.true_inliers[i] * 100, self.df_stats.iloc[i, -1]
                        )
                    },
                ],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            step["args"][0]["visible"][
                i + len(fig.data) / 2
            ] = True  # Toggle i'th trace to "visible"
            steps.append(step)
        sliders = [dict(active=0, currentvalue={"prefix": "Deteced plane: "}, steps=steps)]
        fig.update_layout(
            sliders=sliders,
            xaxis_title="Global x coor [m]",
            yaxis_title="Global y coor [x]",
            showlegend=True,
        )
        return fig

    def all_detected_points(self, show_as_heatmap=False):
        """Show all points in global map, which have been detected as part of ramp"""
        if show_as_heatmap:
            fig = px.density_heatmap(
                self.df,
                x="x",
                y="y",
                nbinsx=200,
                nbinsy=50,
                title="Heatmap, where do the most points get detected?",
            )
            # Add ramp region as rectangle
            fig.add_shape(
                type="rect",
                x0=self.x_range[0],
                x1=self.x_range[1],
                y0=self.y_range[0],
                y1=self.y_range[1],
                line=dict(color="White", width=3),
                opacity=0.9,
            )
            return fig
        fig = px.scatter(x=self.df["x"], y=self.df["y"], title="Coordinates of all points")
        # Add ramp region as rectangle
        fig.add_shape(
            type="rect",
            x0=self.x_range[0],
            x1=self.x_range[1],
            y0=self.y_range[0],
            y1=self.y_range[1],
            fillcolor="red",
            opacity=0.2,
        )
        return fig

    def bar_plot(self):
        """Bar plot to show ratio of how often ramp has been detected in a certain range before ramp"""
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=self.df_eval["expectedDetections"],
                x=self.df_eval["distToRamp"],
                name="False negatives (Not detected)",
                offsetgroup=0,
            )
        )
        fig.add_trace(
            go.Bar(
                y=self.df_eval["actualDetections"],
                x=self.df_eval["distToRamp"],
                name="True positives (Detected)",
                offsetgroup=0,
            )
        )
        fig.update_xaxes(title_text="Distance to ramp [m]", autorange="reversed")
        fig.update_yaxes(title_text="Number of ramps detected")
        fig.update_layout(
            title_text="How many samples were collected in 1m intervals and how many of them have been identified as ramp"
        )
        return fig
