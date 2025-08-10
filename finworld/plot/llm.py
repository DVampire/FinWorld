import os
from pyecharts.charts import Bar
from pyecharts import options as opts
from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot

from finworld.registry import PLOT

@PLOT.register_module(force=True)
class PlotLLMScore():
    def __init__(self):
        super(PlotLLMScore, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def _create_hierarchical_color_scheme(self):
        """
        Returns a hierarchical color scheme for bar series.
        """
        return {
            # Blue shades
            "blue1": "#EEF4FA",
            "blue2": "#E3EDF7",
            "blue3": "#D7E5F4",
            "blue4": "#D0E1EE",
            "blue5": "#C9DDF0",

            # Green shades
            "green1": "#E4F7F9",
            "green2": "#D8F3F5",
            "green3": "#D1EFF2",
            "green4": "#E2F3E7",
            "green5": "#D6EADB",

            # Yellow shades
            "yellow1": "#FFFDF8",
            "yellow2": "#FFF9E3",
            "yellow3": "#FFF6D8",
            "yellow4": "#FFF2C8",
            "yellow5": "#FFEEC0",

            # Red shades
            "red1": "#FBEAEA",
            "red2": "#F8D8D8",
            "red3": "#F7CFCF",
            "red4": "#F7BABA",
            "red5": "#F29B9B",

            # Purple shades
            "purple1": "#F3EDFA",
            "purple2": "#EAE3F7",
            "purple3": "#E5D7F4",
            "purple4": "#D6C8E7",
            "purple5": "#C9B8DF",
        }

    def plot(self,
             data_dict: dict,
             title: str = "LLM Score Overview",
             main_model: str = "FinReasoner",
             savefig: str = "llm_score_overview.png"):
        """
        Draw a bar chart with pyecharts.
        Highlight the main_model with a hatch pattern (decal).
        Each legend text is separated and clear.

        Args:
            data_dict (dict): Should contain "datasets", "models", and "values".
            title (str): Title of the chart.
            main_model (str): The model name to highlight with hatch.
            savefig (str): Output image file name.

        Returns:
            None
        """
        colors = self._create_hierarchical_color_scheme()
        # Assign colors to each model (can be changed)
        color_order = [
            colors["blue3"],
            colors["green3"],
            colors["yellow3"],
            colors["red3"],
            colors["purple3"],
        ]

        bar = Bar().add_xaxis(data_dict["datasets"])

        # Add y-axis for each model with a clean legend entry
        for i, (model, vals) in enumerate(zip(data_dict["models"], data_dict["values"])):
            if model.lower() == main_model.lower():
                bar.add_yaxis(
                    series_name=model,
                    y_axis=vals,
                    gap="0%",
                    itemstyle_opts=opts.ItemStyleOpts(
                        color=color_order[i % len(color_order)],
                        border_color=colors["blue5"],  # border color for hatch
                        border_width=2  # border thickness (adjust as you like)
                    ),
                    label_opts=opts.LabelOpts(
                        is_show=True,
                        position="top",
                        formatter="{c}",
                        color="#000",
                        font_size=12,
                    ),
                )
            else:
                bar.add_yaxis(
                    series_name=model,
                    y_axis=vals,
                    gap="0%",
                    itemstyle_opts=opts.ItemStyleOpts(
                        color=color_order[i % len(color_order)]
                    ),
                    label_opts=opts.LabelOpts(
                        is_show=True,
                        position="top",
                        formatter="{c}",
                        color="#000",
                        font_size=12,
                    ),
                )

        bar.set_global_opts(
            yaxis_opts=opts.AxisOpts(
                name="Score",
                axislabel_opts=opts.LabelOpts(color="#000"),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#000")
                ),
                axistick_opts=opts.AxisTickOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(color="#000")
                ),
            ),
            xaxis_opts=opts.AxisOpts(
                name="Dataset",
                axislabel_opts=opts.LabelOpts(color="#000"),
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#000")
                ),
                axistick_opts=opts.AxisTickOpts(
                    is_show=True,
                    linestyle_opts=opts.LineStyleOpts(color="#000")
                ),
            ),
            legend_opts=opts.LegendOpts(
                pos_top="5%",
                item_gap=40,
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
        )

        # Render and save as html and image
        html_name = os.path.splitext(os.path.basename(savefig))[0] + ".html"
        html_path = os.path.join(os.path.dirname(savefig), html_name)
        os.makedirs(os.path.dirname(savefig), exist_ok=True) if os.path.dirname(savefig) else None
        make_snapshot(driver, bar.render(html_path), savefig)

# ====== Example usage ======
if __name__ == "__main__":
    data_dict = {
        "datasets": ["FinQA", "ConvFinQA", "FinEval", "CFLUE"],
        "models": [
            "FinReasoner",  # This bar will have hatch
            "Qwen2.5-7B-Instruct",
            "Qwen3-8B",
            "Fin-R1-7B",
            "DeepSeek-R1",
        ],
        "values": [
            [80.0, 89.8, 94.0, 85.9],
            [60.0, 66.0, 85.0, 78.4],
            [68.3, 70.9, 84.7, 80.1],
            [76.0, 85.0, 81.0, 73.7],
            [71.0, 82.0, 90.0, 83.3],
        ]
    }
    plotter = PlotLLMScore()
    plotter(data_dict, title="LLM Score Overview", savefig="llm_score.png")
