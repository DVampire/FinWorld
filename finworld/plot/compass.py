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

    Risk_Control: int
    Proftability: int
    Explainability: int
    Reliability: int
    Diversity: int
    University: int

    def __init__(self, data: dict):
        self.Risk_Control = data["Risk_Control"]
        self.Proftability = data["Proftability"]
        self.Explainability = data["Explainability"]
        self.Reliability = data["Reliability"]
        self.Diversity = data["Diversity"]
        self.University = data["University"]

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        blank.tex file.
        """
        for item in [
            self.Risk_Control,
            self.Proftability,
            self.Explainability,
            self.Reliability,
            self.Diversity,
            self.University,
        ]:
            yield item


@dataclass
class OuterLevel:
    """Outer CLEVA-Compass level with measurement attributes."""
    country: bool
    assert_type: bool
    time_scale: bool
    risk: bool
    risk_adjusted: bool
    extreme_market: bool
    profit: bool
    alpha_decay: bool

    equity_curve: bool
    profile: bool
    variability: bool
    rank_order: bool
    t_SNE: bool
    entropy: bool
    correlation: bool
    rolling_window: bool

    def __init__(self, data: dict):
        self.country = data["country"]
        self.assert_type = data["assert_type"]
        self.time_scale = data["time_scale"]
        self.risk = data["risk"]
        self.risk_adjusted = data["risk_adjusted"]
        self.extreme_market = data["extreme_market"]
        self.profit = data["profit"]
        self.alpha_decay = data["alpha_decay"]

        self.equity_curve = data["equity_curve"]
        self.profile = data["profile"]
        self.variability = data["variability"]
        self.rank_order = data["rank_order"]
        self.t_SNE = data["t_SNE"]
        self.entropy = data["entropy"]
        self.correlation = data["correlation"]
        self.rolling_window = data["rolling_window"]

    def __iter__(self):
        """
        Defines the iteration order. This needs to be the same order as defined in the
        cleva_template.tex file.
        """
        for item in [

            self.country,
            self.assert_type,
            self.time_scale,
            self.risk,
            self.risk_adjusted,
            self.extreme_market,
            self.profit,
            self.alpha_decay,

            self.equity_curve,
            self.profile,
            self.variability,
            self.rank_order,
            self.t_SNE,
            self.entropy,
            self.correlation,
            self.rolling_window,

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
        self.EV = 16  # Number of evaluation measures
        self.A = 360 / self.D  # Angle between method axes
        self.B = 360 / self.EV  # Angle between evaluation measure axes

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
        }
        return color_scheme

    def _insert_legend(self, template, entries):
        """Insert the CLEVA-Compass legend below the compass."""

        # Skip if no entries are given (else the empty tabular will produce compile errors)
        if len(entries) == 0:
            return template

        # Compute number of rows/columns with max. three elements per row
        n_rows = math.ceil(len(entries) / 6)
        n_cols = 6 if len(entries) >= 6 else len(entries)

        # Begin legend tabular
        legend_str = ""
        legend_str += r"\begin{tabular}{" + " ".join(["l"] * n_cols) + "} \n"

        for i, e in enumerate(entries):
            # x/y coordinates of the entry
            x = i % 6
            y = round(i // 6)

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
\usepackage[utf8]{inputenc}
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

        return savefig


if __name__ == "__main__":
    template_path = assemble_project_path(os.path.join("res/latex/compass_template.tex"))
    data_path = assemble_project_path(os.path.join("others/res/latex/compass_sample_data.json"))
    data_dict = json.load(open(data_path))

    entiries = data_dict["entries"]

    plot = PlotCompass(
        template_path=template_path
    )
    savefig = "compass.pdf"

    plot(entiries, savefig=savefig)
