import json
import os.path
from typing import List
import math
import shutil
from dataclasses import dataclass
import tempfile
import subprocess

from finworld.utils import assemble_project_path
from finworld.registry import PLOT


@dataclass
class InnerLevel:
    """Inner CLEVA-Compass level with method attributes."""

    ARR: int
    SR: int
    MDD: int
    CR: int
    SOR: int
    VOL: int

    def __init__(self, data: dict):
        self.ARR = data["ARR"]
        self.SR = data["SR"]
        self.MDD = data["MDD"]
        self.CR = data["CR"]
        self.SOR = data["SOR"]
        self.VOL = data["VOL"]

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        blank.tex file.
        """
        for item in [
            self.ARR,
            self.SR,
            self.MDD,
            self.CR,
            self.SOR,
            self.VOL,
        ]:
            yield item


@dataclass
class OuterLevel:
    """Outer CLEVA-Compass level with measurement attributes."""
    open_source: bool
    reasoning: bool
    financial_tuning: bool
    multi_lingual: bool
    api: bool
    explainability: bool
    community: bool

    def __init__(self, data: dict):
        self.open_source = data["open_source"]
        self.reasoning = data["reasoning"]
        self.financial_tuning = data["financial_tuning"]
        self.multi_lingual = data["multi_lingual"]
        self.api = data["api"]
        self.explainability = data["explainability"]
        self.community = data["community"]

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        cleva_template.tex file.
        """
        for item in [
            self.open_source,
            self.reasoning,
            self.financial_tuning,
            self.multi_lingual,
            self.api,
            self.explainability,
            self.community,
        ]:
            yield item


@dataclass
class CompassEntry:
    """Compass entry containing color, label, and attributes."""

    color: str  # Color, can be one of [magenta, green, blue, orange, cyan, brown]
    label: str  # Legend label
    inner_level: InnerLevel
    outer_level: OuterLevel


@PLOT.register_module(force=True)
class PlotCompass():
    def __init__(self, template_path: str = None):
        super(PlotCompass, self).__init__()
        self.template_path = template_path

        # Constants
        self.D = 6  # Number of protocol dimensions
        self.EV = 7  # Number of evaluation measures
        self.A = 360 / self.D  # Angle between method axes
        self.B = 360 / self.EV  # Angle between evaluation measure axes
        
        self.legend_nums = 7 # Number of legend items

        self.color_scheme = self._create_color_scheme()

    def __call__(self, *args, **kwargs):
        return self.plot(*args, **kwargs)

    def _create_color_scheme(self):
        color_scheme = {
            "magenta": "magenta",
            "green": "green!50!black",
            "blue": "blue!70!black",
            "orange": "orange!90!black",
            "cyan": "cyan!90!black",
            "brown": "brown!90!black",
            "grey": "grey!90!black",
            "purple": "purple!90!black",
            "red": "red!90!black",
            "yellow": "yellow!90!black",
            "pink": "pink!90!black",
            "teal": "teal!90!black",
            "lime": "lime!90!black",
            "maroon": "maroon!90!black",
            "navy": "navy!90!black",
        }
        return color_scheme

    def _insert_legend(self, template, entries):
        """Insert the CLEVA-Compass legend below the compass."""

        # Skip if no entries are given (else the empty tabular will produce compile errors)
        if len(entries) == 0:
            return template

        # Compute number of rows/columns with max. three elements per row
        n_rows = math.ceil(len(entries) / self.legend_nums)
        n_cols = self.legend_nums if len(entries) >= self.legend_nums else len(entries)

        # Begin legend tabular
        legend_str = ""
        legend_str += r"\begin{tabular}{" + " ".join(["l"] * n_cols) + "} \n"

        for i, e in enumerate(entries):
            # x/y coordinates of the entry
            x = i % self.legend_nums
            y = round(i // self.legend_nums)

            # Actual entry which uses \lentry defined in the tikz template
            legend_str += r"\lentry{" + self.color_scheme[e.color] + "}{" + e.label + "}"

            # Depending on last column/row
            is_last_column = x == n_cols - 1
            is_last_row = y == n_rows - 1
            if not is_last_column:
                # Add & for next element in row
                legend_str += r" & "
            else:
                if not is_last_row:
                    # Add horizontal space if there is another row
                    legend_str += " \\\\[0.15cm] \n"
                else:
                    # Add no horizontal space if this is the last row
                    legend_str += " \\\\ \n"

        # End legend tabular
        legend_str += "\end{tabular} \n"

        # Replace the generated string in template
        template = template.replace("%-$LEGEND$", legend_str)
        return template

    def _insert_outer_level(self, template, entries: List[CompassEntry]):
        """Insert outer level attributes."""
        oc_str = ""
        M = len(entries)
        for e_idx, e in enumerate(entries):
            # Add comment for readability
            s = "% Entry for: " + e.label + "\n"

            # For each outer level attribute
            for ol_idx, has_attribute in enumerate(e.outer_level):
                # If attribute is not present, skip and leave white
                if not has_attribute:
                    continue
                angle_start = str(ol_idx * self.B + e_idx * self.B / M)
                angle_end = str(ol_idx * self.B + (e_idx + 1) * self.B / M)

                # Invert stripe direction when in the lower half (index larger than 7)
                if ol_idx > 7:
                    angle_start, angle_end = angle_end, angle_start

                shell = e.color + "shell"
                s += (
                        "\pic at (0,0){strip={\Instrip,"
                        + str(angle_start)
                        + ","
                        + str(angle_end)
                        + ","
                        + shell
                        + ", black, {}}};\n"
                )
            oc_str += s + "\n"

        template = template.replace("%-$OUTER-CIRCLE$", oc_str)
        return template

    def _insert_inner_level(self, template, entries: List[CompassEntry]):
        """Insert inner level path connections."""
        ir_str = ""
        for e in entries:
            path = " -- ".join(f"(D{i + 1}-{irv})" for i, irv in enumerate(e.inner_level))
            ir_str += f"\draw [color={self.color_scheme[e.color]},line width=1.5pt,opacity=0.6, fill={self.color_scheme[e.color]}!10, fill opacity=0.4] {path} -- cycle;\n"

        template = template.replace("%-$INNER-CIRCLE$", ir_str)
        return template

    def _insert_number_of_methods(self, template, entries: List[CompassEntry]):
        """Insert number of methods as newcommand \M."""
        n_methods_str = r"\newcommand{\M}{" + str(len(entries)) + "}"
        template = template.replace("%-$NUMBER-OF-METHODS$", n_methods_str)
        return template

    def _insert_document(self, output):
        """
        Insert the document environment around the output.
        Args:
            output:

        Returns:
        """
        latex_docx = r"""\documentclass[tikz, border=0pt]{standalone}
\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage[colorlinks=true,citecolor=blue]{hyperref}
% \usepackage[colorlinks=true,linkcolor=blue]{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{tikz}
\usepackage{wrapfig}

\usepackage{subcaption}
\usepackage{makecell}
\usepackage{multirow}
\usepackage{ragged2e}
\usetikzlibrary{external}
\title{test_compass}
\date{June 2022}

\begin{document}
""" + output + r"""
\end{document}"""

        return latex_docx

    def _read_json_entries(self, entries_dict: dict):
        """Read the compass entries from a json file."""
        entries = []
        for d in entries_dict:
            dil = d["inner_level"]
            dol = d["outer_level"]
            entry = CompassEntry(
                color=d["color"],
                label=d["label"],
                inner_level=InnerLevel(
                    data=dil
                ),
                outer_level=OuterLevel(
                    data=dol
                ),
            )
            entries.append(entry)
        return entries

    def plot(self,
             data_dict: dict,
             savefig: str = 'compass.pdf',
             **kwargs):

        entries = self._read_json_entries(data_dict)

        with open(self.template_path) as f:
            template = "".join(f.readlines())

        # Replace respective parts in template
        output = template
        output = self._insert_legend(output, entries)
        output = self._insert_outer_level(output, entries)
        output = self._insert_inner_level(output, entries)
        output = self._insert_number_of_methods(output, entries)
        output = self._insert_document(output)

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "document.tex")
            pdf_path = os.path.join(tmpdir, "document.pdf")

            # Write LaTeX source to temp .tex file
            with open(tex_path, "w") as f:
                f.write(output)

            # Compile using pdflatex
            try:
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-shell-escape", "-output-directory", tmpdir, tex_path],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                pass

            # Move final PDF to a non-temporary location if needed
            shutil.copyfile(pdf_path, savefig)
            shutil.copy(tex_path, savefig.replace('.pdf', '.tex'))

        return savefig


if __name__ == "__main__":

    symbols = ["AAPL", "AMZN", "GOOGL", "MSFT", "TSLA", "META"]

    for symbol in symbols:

        template_path = assemble_project_path(os.path.join("others/res/latex/compass_trading_template.tex"))
        data_path = assemble_project_path(os.path.join(f"others/res/latex/compass_trading_data_{symbol}.json"))
        data_dict = json.load(open(data_path))

        # ========== Normalization and special field handling ==========
        # Adapt to your own normalization rules as needed below!
        for entry in data_dict["entries"]:
            inner = entry["inner_level"]

            # Example: ARR is already a percentage [0, 100], keep as integer
            inner["ARR"] = int(round(inner["ARR"]))
            # Example: MDD is percentage, but reverse the scale (smaller is better)
            inner["MDD"] = 100 - int(round(inner["MDD"]))

            # VOL (volatility): convert from [0, 1] to [0, 100], then reverse (smaller is better)
            inner["VOL"] = int(round((1 - inner["VOL"]) * 100))

            # SR (Sharpe Ratio): linear normalization to [0, 100], clip to [0, 2]
            sr = max(min(inner["SR"], 4), 0)
            inner["SR"] = int(round((sr - 0) / (4 - 0) * 100))

            # CR (Calmar Ratio): linear normalization to [0, 100], clip to [0, 3]
            cr = max(min(inner["CR"], 4), 0)
            inner["CR"] = int(round((cr - 0) / (4 - 0) * 100))

            # SOR (Sortino Ratio): linear normalization to [0, 100], clip to [0, 3]
            sor = max(min(inner["SOR"], 4), 0)
            inner["SOR"] = int(round((sor - 0) / (4 - 0) * 100))
        # ========== END ==========

        entiries = data_dict["entries"]
        for item in entiries:
            print(item)

        plot = PlotCompass(
            template_path=template_path
        )
        savefig = f"{symbol}_compass.pdf"

        plot(entiries, savefig=savefig)
