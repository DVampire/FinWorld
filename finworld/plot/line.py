import os
import numpy as np
from pyecharts.charts import Line, Grid
from pyecharts import options as opts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot as driver
import json

from finworld.utils import assemble_project_path

class PlotLongTail:
    def __init__(self):
        super(PlotLongTail, self).__init__()

    def _create_hierarchical_color_scheme(self):
        """
        Returns a hierarchical color scheme: 5 shades each for blue, green, yellow, red, purple, orange.
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

            # Orange shades
            "orange1": "#FFF4E6",
            "orange2": "#FFE8CC",
            "orange3": "#FFD8A8",
            "orange4": "#FFA94D",
            "orange5": "#FF922B",
        }

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self, data_dict: dict, savefig: str = "LongTail.html", color_keys=None, **kwargs):
        """
        Plot multi-stock long-tail distributions with custom color scheme.
        Args:
            data_dict: dict, stock symbol to value list.
            savefig: output file name (png/svg/pdf/html).
            color_keys: list of 6 color keys (eg. ["blue3","green3","yellow3","red3","purple3","orange3"])
        """
        color_scheme = self._create_hierarchical_color_scheme()
        # Default: one from each color, medium shade
        if color_keys is None:
            color_keys = ["blue3", "green3", "yellow3", "red3", "purple3", "orange3"]
        color_list = [color_scheme[k] for k in color_keys]
        stock_names = list(data_dict.keys())
        x_data = list(range(max(len(v) for v in data_dict.values())))

        c = Line(
            init_opts=opts.InitOpts(
                width="1000px",
                height="600px",
                bg_color="#fff",  # White background
            )
        )
        c.add_xaxis(x_data)

        for idx, stock in enumerate(stock_names):
            y = np.array(data_dict[stock])
            y = np.sort(y)[::-1]  # Sort descending for long-tail
            color = color_list[idx % len(color_list)]
            c.add_yaxis(
                series_name=stock,
                y_axis=y.tolist(),
                is_smooth=True,
                is_symbol_show=True,
                symbol="circle",
                symbol_size=7,
                linestyle_opts=opts.LineStyleOpts(color=color, width=2),
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(color="#fff", border_color=color, border_width=2),
                areastyle_opts=None,
                z=5,
            )

        c.set_global_opts(
            # title_opts=opts.TitleOpts(
            #     title="6 Stocks Long-tail Distribution",
            #     pos_top="2%",
            #     pos_left="center",
            #     title_textstyle_opts=opts.TextStyleOpts(color="#000", font_size=20),  # Black title
            # ),
            legend_opts=opts.LegendOpts(
                pos_top="7%",
                orient="horizontal",
                textstyle_opts=opts.TextStyleOpts(color="#000", font_size=14),
                item_width=30,
                item_height=15,
            ),
            xaxis_opts=opts.AxisOpts(
                name="Items",
                name_textstyle_opts=opts.TextStyleOpts(color="#000", font_size=16),
                type_="category",
                boundary_gap=False,
                axislabel_opts=opts.LabelOpts(margin=20, color="#000", font_size=13),
                axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#000")),
                axistick_opts=opts.AxisTickOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#000")),
                splitline_opts=opts.SplitLineOpts(is_show=False),  # No grid
            ),
            yaxis_opts=opts.AxisOpts(
                name="Tokens",
                name_textstyle_opts=opts.TextStyleOpts(color="#000", font_size=16),
                type_="value",
                axislabel_opts=opts.LabelOpts(margin=20, color="#000", font_size=13),
                axisline_opts=opts.AxisLineOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#000")),
                axistick_opts=opts.AxisTickOpts(is_show=True, linestyle_opts=opts.LineStyleOpts(color="#000")),
                splitline_opts=opts.SplitLineOpts(is_show=False),  # No grid
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                background_color="rgba(50,50,50,0.9)",
                border_color="#777",
                textstyle_opts=opts.TextStyleOpts(color="#000"),
            ),
        )

        grid = Grid()
        grid.add(
            c,
            grid_opts=opts.GridOpts(
                pos_top="15%",
                pos_left="10%",
                pos_right="10%",
                pos_bottom="10%",
                is_contain_label=True,
            ),
        )

        html_name = os.path.splitext(os.path.basename(savefig))[0] + ".html"
        html_path = os.path.join(os.path.dirname(savefig), html_name)
        os.makedirs(os.path.dirname(savefig), exist_ok=True) if os.path.dirname(savefig) else None
        make_snapshot(driver, grid.render(html_path), savefig)

if __name__ == '__main__':
    data_path = assemble_project_path('res/stats/tokens.json')
    with open(data_path, 'r') as f:
        data_dict = json.load(f)['origin']

    plotter = PlotLongTail()
    # Example: use default 6-color medium set
    plotter(
        data_dict,
        savefig='original_tokens_distribution.png',
    )

    data_path = assemble_project_path('res/stats/tokens.json')
    with open(data_path, 'r') as f:
        data_dict = json.load(f)['summarized']

    plotter = PlotLongTail()
    # Example: use default 6-color medium set
    plotter(
        data_dict,
        savefig='summarized_tokens_distribution.png',
    )
