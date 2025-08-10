import os.path
from typing import List, Sequence, Union
import pandas as pd
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode
from pyecharts.charts import Tree
from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot

from finworld.utils import assemble_project_path
from finworld.registry import PLOT


@PLOT.register_module(force=True)
class PlotProject():
    def __init__(self):
        super(PlotProject, self).__init__()

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def _create_hierarchical_color_scheme(self):
        return {
            # 蓝色系
            "blue1": "#EEF4FA",
            "blue2": "#E3EDF7",
            "blue3": "#D7E5F4",
            "blue4": "#D0E1EE",
            "blue5": "#C9DDF0",

            # 绿色系
            "green1": "#E4F7F9",
            "green2": "#D8F3F5",
            "green3": "#D1EFF2",
            "green4": "#E2F3E7",
            "green5": "#D6EADB",

            # 黄色系
            "yellow1": "#FFFDF8",
            "yellow2": "#FFF9E3",
            "yellow3": "#FFF6D8",
            "yellow4": "#FFF2C8",
            "yellow5": "#FFEEC0",
        }

    def plot(
            self,
            data_dict: dict,
            title: str = "Project Overview",
            savefig: str = "project_overview.png",
    ):
        """
        Draw a radial tree with pyecharts based on `data_dict.

        Args:
            data_dict (dict): Nested dictionary following the D3/flare style
                with fields `name, description and children.
            title (str): Chart title shown at the top.
            save_path (str, optional):
                * None               -> save to `<project_root>/html/{title}.html
                * "*.html"           -> render HTML to the given path
                * "*.png" / "*.svg"  -> export a snapshot using snapshot‑selenium
        Returns:
            pyecharts.charts.Tree: The generated Tree object so you can
            `render_notebook() it in Jupyter if needed.
        """
        # 获取颜色方案
        colors = self._create_hierarchical_color_scheme()

        # 为节点添加颜色信息
        def add_colors_to_data(data, level=0):
            if isinstance(data, dict):
                # 根据层级选择颜色系
                if level == 0:  # 根节点
                    data['itemStyle'] = {'color': colors['blue1'], 'borderColor': colors['blue4']}
                elif level == 1:  # 第一层子节点
                    data['itemStyle'] = {'color': colors['green2'], 'borderColor': colors['green4']}
                else:  # 更深层级
                    data['itemStyle'] = {'color': colors['yellow2'], 'borderColor': colors['yellow4']}

                if 'children' in data:
                    for child in data['children']:
                        add_colors_to_data(child, level + 1)
            return data

        # 深拷贝数据并添加颜色
        import copy
        colored_data = copy.deepcopy(data_dict)
        add_colors_to_data(colored_data)

        # ---- build the Tree chart ----
        tree = (
            Tree()
            .add(
                series_name="",
                data=[colored_data],  # 使用带颜色的数据
                layout="radial",  # radial = root at the center
                symbol="circle",
                symbol_size=12,  # 稍微增大节点以更好展示颜色
                edge_shape="curve",
                initial_tree_depth=-1,  # expand all layers
                label_opts=opts.LabelOpts(
                    position="rotate",  # rotate text along the tangent
                    font_size=10,
                    color="#333",
                    formatter=JsCode(
                        # combine name and description into two lines with bold name
                        """
                        function(params){
                            var n = params.data.name || '';
                            var d = params.data.description || '';
                            return '{bold|' + n + '}\\n' + d;
                        }
                        """
                    ),
                    # 添加富文本样式配置
                    rich={
                        "bold": {
                            "fontWeight": "bold",
                            "color": "#000"  # 可以设置更深的颜色来突出显示
                        }
                    }
                ),
                # # 设置线条颜色
                # linestyle_opts=opts.LineStyleOpts(color=colors['blue3'], width=2),
            )
            .set_global_opts(
                # title_opts=opts.TitleOpts(title=title),
                # 设置背景颜色
                graphic_opts=[
                    opts.GraphicGroup(
                        graphic_item=opts.GraphicItem(
                            left="center",
                            top="center",
                        ),
                        children=[
                            opts.GraphicRect(
                                graphic_basicstyle_opts=opts.GraphicBasicStyleOpts(
                                    fill="rgba(255,255,255,0.8)"
                                )
                            )
                        ]
                    )
                ]
            )
        )

        html_name = os.path.basename(savefig).split(".")[0] + ".html"
        html_path = os.path.join(os.path.dirname(savefig), html_name)
        make_snapshot(driver, tree.render(html_path), savefig)

        return tree


if __name__ == '__main__':
    data = {
        "name": "Config",
        "description": "Root Configuration",
        "children": [
            {"name": "General", "description": "Basic Settings & Parameters"},
            {"name": "Dataset", "description": "Data Sources & Processing"},
            {"name": "Dataloader", "description": "Data Loader & Batching"},
            {"name": "Model", "description": "Architecture & Layers"},
            {"name": "Optimizer", "description": "Optimization & LR"},
            {"name": "Scheduler", "description": "Learning Rate Scheduling"},
            {"name": "Loss", "description": "Loss Functions"},
            {"name": "Metric", "description": "Evaluation & Performance"},
            {"name": "Trainer", "description": "Training Control & Loops"},
            {"name": "Task", "description": "Task Wrapper & Interface"},
            {"name": "Tracker", "description": "Logging & Monitoring"},
            {"name": "More", "description": "Others ..."},
        ],
    }

    plot = PlotProject()
    plot(data, title="FinWorld Configuration", savefig=assemble_project_path("finworld_config_tree_radial.png"))