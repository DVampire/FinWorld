import os.path
from typing import List, Sequence, Union
import pandas as pd
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Kline, Line, Bar, Grid
from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot

from finworld.utils import assemble_project_path
from finworld.registry import PLOT

@PLOT.register_module(force=True)
class PlotProKline():
    def __init__(self,
                 mas = [5, 10, 20]
                 ):
        self.mas = mas

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the data for plotting.
        """
        data['volume'] = data['volume'].astype(float)/ 1e6  # Convert volume to millions
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['difs'] = ema12 - ema26
        data['deas'] = data['difs'].ewm(span=9).mean()
        data['macds'] = 2 * (data['difs'] - data['deas'])

        for window in self.mas:
            data[f'ma{window}'] = data['close'].rolling(window=window).mean()

        data = data.iloc[max(self.mas):]  # Remove initial rows with NaN values due to rolling mean

        golden_cross = (data['difs'] > data['deas']) & (data['difs'].shift(1) <= data['deas'].shift(1))
        # death_cross = (data['difs'] < data['deas']) & (data['dif'].shift(1) >= data['deas'].shift(1))
        data['macd_flag'] = 0
        data.loc[golden_cross, 'macd_flag'] = 1
        # data.loc[death_cross, 'macd_flag'] = -1

        columns = ['timestamp', 'open', 'close', 'low', 'high', 'volume', 'macd_flag', 'macds', 'difs', 'deas'] + [f'ma{window}' for window in self.mas]
        data = data[columns]

        return data

    def _mark_line_data(self, data: pd.DataFrame):

        mark_line_data = []
        idx = 0
        tag = 0
        vols = 0

        timestamps = data.index.tolist()
        open = data['open'].tolist()
        close = data['close'].tolist()
        high = data['high'].tolist()
        low = data['low'].tolist()
        volume = data['volume'].tolist()
        macd_flag = data['macd_flag'].tolist()

        # 0 open, 1 close, 2 low, 3 high
        for i in range(len(timestamps)):
            if macd_flag[i] != 0 and tag == 0:
                idx = i
                vols = volume[i]
                tag = 1
            if tag == 1:
                vols += volume[i]
            if macd_flag[i] != 0 or tag == 1:
                mark_line_data.append(
                    [
                        {
                            "xAxis": idx,
                            "yAxis": float("%.2f" % high[idx])
                            if close[idx] > open[idx]
                            else float("%.2f" % low[idx]),
                            "value": vols,
                        },
                        {
                            "xAxis": i,
                            "yAxis": float("%.2f" % high[i])
                            if close[i] > open[i]
                            else float("%.2f" % low[i]),
                        },
                    ]
                )
                idx = i
                vols = volume[i]
                tag = 2
            if tag == 2:
                vols += volume[i]
            if macd_flag[i] != 0 and tag == 2:
                mark_line_data.append(
                    [
                        {
                            "xAxis": idx,
                            "yAxis": float("%.2f" % high[idx])
                            if close[i] > open[i]
                            else float("%.2f" % low[i]),
                            "value": str(float("%.2f" % (vols / (i - idx + 1)))) + " M",
                        },
                        {
                            "xAxis": i,
                            "yAxis": float("%.2f" % high[i])
                            if close[i] > open[i]
                            else float("%.2f" % low[i]),
                        },
                    ]
                )
                idx = i
                vols = volume[i]
        return mark_line_data


    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self,
             data: pd.DataFrame,
             title: str = "Kline Chart",
             savefig: str = "kline_plot.pdf",
             **kwargs):

        data = self._prepare_data(data)

        xaxis_data = data["timestamp"].dt.strftime('%Y-%m-%d').tolist()
        yaxis_data = data[[col for col in data.columns if col != "timestamp"]].values.tolist()
        volume = data['volume'].values.tolist()
        macds = data['macds'].values.tolist()
        difs = data['difs'].values.tolist()
        deas = data['deas'].values.tolist()
        ma5 = data['ma5'].values.tolist()
        ma10 = data['ma10'].values.tolist()
        ma20 = data['ma20'].values.tolist()
        markline_data = self._mark_line_data(data)

        kline = (
            Kline()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="",
                y_axis=yaxis_data,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#ef232a",
                    color0="#14b143",
                    border_color="#ef232a",
                    border_color0="#14b143",
                ),
                markpoint_opts=opts.MarkPointOpts(
                    data=[
                        opts.MarkPointItem(type_="max", name="MaxValue"),
                        opts.MarkPointItem(type_="min", name="MinValue"),
                    ]
                ),
                markline_opts=opts.MarkLineOpts(
                    label_opts=opts.LabelOpts(
                        position="middle", color="blue", font_size=15
                    ),
                    data=markline_data,
                    symbol=["circle", "none"],
                ),
            )
            .set_series_opts(
                markarea_opts=opts.MarkAreaOpts(is_silent=True, data=markline_data)
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title, pos_left="0"),
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    is_scale=True,
                    boundary_gap=False,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    split_number=20,
                    min_="dataMin",
                    max_="dataMax",
                ),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True, splitline_opts=opts.SplitLineOpts(is_show=True)
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="line"),
                # datazoom_opts=[
                #     opts.DataZoomOpts(type_="inside", xaxis_index=[0, 0], range_end=100
                #     ),
                #     opts.DataZoomOpts(
                #         is_show=False, xaxis_index=[0, 1], pos_top="100%", range_end=100
                #     ),
                #     opts.DataZoomOpts(is_show=False, xaxis_index=[0, 2], range_end=100),
                # ],
            )
        )

        kline_line = (
            Line()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="MA5",
                y_axis=ma5,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA10",
                y_axis=ma10,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="MA20",
                y_axis=ma20,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=1,
                    split_number=3,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
            )
        )

        # Overlap Kline + Line
        overlap_kline_line = kline.overlap(kline_line)

        # Bar-1
        bar_1 = (
            Bar()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="Volume",
                y_axis=volume,
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                    function(params) {
                        var colorList;
                        if (barData[params.dataIndex][1] > barData[params.dataIndex][0]) {
                            colorList = '#ef232a';
                        } else {
                            colorList = '#14b143';
                        }
                        return colorList;
                    }
                    """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=1,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        # Bar-2 (Overlap Bar + Line)
        bar_2 = (
            Bar()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="MACD",
                y_axis=macds,
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
                itemstyle_opts=opts.ItemStyleOpts(
                    color=JsCode(
                        """
                            function(params) {
                                var colorList;
                                if (params.data >= 0) {
                                  colorList = '#ef232a';
                                } else {
                                  colorList = '#14b143';
                                }
                                return colorList;
                            }
                            """
                    )
                ),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                    type_="category",
                    grid_index=2,
                    axislabel_opts=opts.LabelOpts(is_show=False),
                ),
                yaxis_opts=opts.AxisOpts(
                    grid_index=2,
                    split_number=4,
                    axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                    axistick_opts=opts.AxisTickOpts(is_show=False),
                    splitline_opts=opts.SplitLineOpts(is_show=False),
                    axislabel_opts=opts.LabelOpts(is_show=True),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        line_2 = (
            Line()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="DIF",
                y_axis=difs,
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .add_yaxis(
                series_name="DIF",
                y_axis=deas,
                xaxis_index=2,
                yaxis_index=2,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(legend_opts=opts.LegendOpts(is_show=False))
        )

        # Overlap Bar + Line
        overlap_bar_line = bar_2.overlap(line_2)

        grid_chart = Grid(init_opts=opts.InitOpts(width="1000px", height="700px"))

        # Add js_funcs to handle dynamic color assignment in Bar chart
        grid_chart.add_js_funcs("var barData = {}".format(yaxis_data))

        # Kline + MA Lines
        grid_chart.add(
            overlap_kline_line,
            grid_opts=opts.GridOpts(pos_left="3%", pos_right="1%", height="60%"),
        )

        # Volumn Bar
        grid_chart.add(
            bar_1,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", pos_top="71%", height="10%"
            ),
        )

        # MACD DIFS DEAS
        grid_chart.add(
            overlap_bar_line,
            grid_opts=opts.GridOpts(
                pos_left="3%", pos_right="1%", pos_top="82%", height="14%"
            ),
        )

        # Save the chart to a file
        html_name = os.path.basename(savefig).split(".")[0] + ".html"
        html_path = os.path.join(os.path.dirname(savefig), html_name)
        make_snapshot(driver, grid_chart.render(html_path), savefig)


@PLOT.register_module(force=True)
@PLOT.register_module(force=True)
class PlotSimpleKline:
    def __init__(self, mas: List[int] = [5, 10, 20]):
        self.mas = mas

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare input DataFrame.
        Expect columns: ['timestamp', 'open', 'close', 'low', 'high', 'volume']
        """
        data = data.copy()
        data = data.sort_values(by="timestamp").reset_index(drop=True)

        data["volume"] = data["volume"].astype(float) / 1e6  # million units
        data["timestamp"] = pd.to_datetime(data["timestamp"])

        for ma in self.mas:
            data[f"ma{ma}"] = data["close"].rolling(window=ma).mean().fillna(method="bfill")

        data = data.iloc[max(self.mas):]  # Remove initial rows with NaN values due to rolling mean

        return data

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def plot(self,
             data: pd.DataFrame,
             title: str = "Simple Kline Chart",
             savefig: str = "simple_kline_plot.pdf",
             **kwargs):

        data = self._prepare_data(data)

        xaxis_data = data["timestamp"].dt.strftime("%Y-%m-%d").tolist()
        yaxis_data = data[["open", "close", "low", "high"]].values.tolist()
        volume_data = data["volume"].tolist()

        kline = (
            Kline()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="Kline",
                y_axis=yaxis_data,
                itemstyle_opts=opts.ItemStyleOpts(color="#ec0000", color0="#00da3c"),
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(title=title),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(type_="category"),
                yaxis_opts=opts.AxisOpts(is_scale=True),
                # datazoom_opts=[
                #     opts.DataZoomOpts(type_="inside"),
                # ]
            )
        )

        # MA lines
        line = Line().add_xaxis(xaxis_data=xaxis_data)
        for ma in self.mas:
            line.add_yaxis(
                series_name=f"MA{ma}",
                y_axis=data[f"ma{ma}"].tolist(),
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(opacity=0.5),
                label_opts=opts.LabelOpts(is_show=False),
            )

        overlap_kline_ma = kline.overlap(line)

        # Volume bar
        bar = (
            Bar()
            .add_xaxis(xaxis_data=xaxis_data)
            .add_yaxis(
                series_name="Volume",
                y_axis=volume_data,
                xaxis_index=1,
                yaxis_index=1,
                label_opts=opts.LabelOpts(is_show=False),
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(type_="category", grid_index=1, axislabel_opts=opts.LabelOpts(is_show=False)),
                yaxis_opts=opts.AxisOpts(grid_index=1, axislabel_opts=opts.LabelOpts(is_show=True)),
                legend_opts=opts.LegendOpts(is_show=False),
            )
        )

        # Grid chart
        grid = Grid(init_opts=opts.InitOpts(width="1000px", height="700px"))
        grid.add(
            overlap_kline_ma,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="60%"),
        )
        grid.add(
            bar,
            grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", pos_top="70%", height="15%"),
        )

        # Render to HTML & save as image
        html_name = os.path.basename(savefig).split(".")[0] + ".html"
        html_path = os.path.join(os.path.dirname(savefig), html_name)
        make_snapshot(driver, grid.render(html_path), savefig)


if __name__ == '__main__':
    data_path = assemble_project_path(os.path.join("datasets/exp/exp_fmp_price_1day/AAPL.jsonl"))

    df = pd.read_json(data_path, lines=True)
    df = df.iloc[-100:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    plotpro = PlotProKline()
    plotpro(df,
         title="AAPL Pro Kline Chart",
         savefig="aapl_pro_kline.png")

    data_path = assemble_project_path(os.path.join("datasets/exp/exp_fmp_price_1day/TSLA.jsonl"))

    df = pd.read_json(data_path, lines=True)
    df = df.iloc[-1000:]
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    plotsimple= PlotSimpleKline()
    plotsimple(df,
        title="TSLA Simple Kline Chart",
        savefig="tsla_simple_kline.png")