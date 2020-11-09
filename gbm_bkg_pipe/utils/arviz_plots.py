import re
from datetime import datetime
import os
import arviz
import arviz as az
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt


class ArvizPlotter(object):
    def __init__(self, date, path_to_netcdf):

        self._date = date

        self._arviz_result = az.InferenceData.from_netcdf(path_to_netcdf)

        self._global_names = self._arviz_result.constant_data[
            "global_param_names"
        ].values
        self._cont_names = self._arviz_result.constant_data["cont_param_names"].values

    def plot_posterior(self, outdir):
        ax = az.plot_posterior(
            self._arviz_result,
            var_names=["norm_fixed"],
        )
        for i in range(len(ax)):
            title = ax[i].title.get_text()
            idx = title.split("\n")[1].replace(" ", "")
            new_title = self._global_names[int(idx)]
            ax[i].set_title(new_title)

        plt.savefig(
            os.path.join(outdir, f"{self._date}_global_posterior.png"),
            transparent=True,
            dpi=100,
        )

        ax = az.plot_posterior(
            self._arviz_result,
            var_names=["norm_cont"],
        )
        for i in range(len(ax)):
            title = ax[i].title.get_text()
            idx = title.split("\n")[1].replace(" ", "")
            new_title = self._cont_names[int(idx[0]), int(idx[2]), int(idx[4])]
            ax[i].set_title(new_title)

        plt.savefig(
            os.path.join(outdir, f"{self._date}_cont_posterior.png"),
            transparent=True,
            dpi=100,
        )

    def plot_traces(self, outdir):

        ax = az.plot_trace(
            self._arviz_result,
            var_names=["norm_fixed"],
        )

        for i in range(len(ax)):
            for j in range(len(ax[i])):
                title = ax[i, j].title.get_text()
                idx = int(title.split("\n")[1].replace(" ", ""))

                ax[i, j].set_title(self._global_names[idx])

        plt.savefig(
            os.path.join(outdir, f"{self._date}_global_traces.png"),
            transparent=True,
            dpi=100,
        )

        ax = az.plot_trace(
            self._arviz_result,
            var_names=["norm_cont"],
        )

        for i in range(len(ax)):
            for j in range(len(ax[i])):

                title = ax[i, j].title.get_text()
                idx = title.split("\n")[1].replace(" ", "")
                new_title = self._cont_names[int(idx[0]), int(idx[2]), int(idx[4])]
                ax[i, j].set_title(new_title)

        plt.savefig(
            os.path.join(outdir, f"{self._date}_cont_traces.png"),
            transparent=True,
            dpi=100,
        )

    def plot_pairs(self, outdir):

        with az.rc_context({"plot.max_subplots": 100}):
            ax = az.plot_pair(
                self._arviz_result, var_names=["norm_fixed"], textsize=25, kind="hexbin"
            )

            for i in range(len(ax)):
                for j in range(len(ax[i])):
                    try:
                        xlabel = ax[i, j].get_xlabel()
                        if xlabel != "":
                            idx = int(xlabel.split("\n")[1].replace(" ", ""))

                            label = "\n".join(self._global_names[idx].split("_"))

                            ax[i, j].set_xlabel(label)

                        ylabel = ax[i, j].get_ylabel()
                        if ylabel != "":
                            idx = int(ylabel.split("\n")[1].replace(" ", ""))

                            label = "\n".join(self._global_names[idx].split("_"))

                            ax[i, j].set_ylabel(label)
                    except Exception as e:
                        pass

            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, f"{self._date}_global_pairs.png"),
                transparent=True,
                dpi=100,
            )

        with az.rc_context({"plot.max_subplots": 100}):
            ax = az.plot_pair(
                self._arviz_result, var_names=["norm_cont"], textsize=25, kind="hexbin"
            )

            for i in range(len(ax)):
                for j in range(len(ax[i])):
                    try:
                        xlabel = ax[i, j].get_xlabel()
                        if xlabel != "":
                            idx = xlabel.split("\n")[1].replace(" ", "")
                            new_label = self._cont_names[
                                int(idx[0]), int(idx[2]), int(idx[4])
                            ]
                            new_label = "\n".join(new_label.split("_"))
                            ax[i, j].set_xlabel(label)

                        ylabel = ax[i, j].get_ylabel()
                        if ylabel != "":
                            idx = ylabel.split("\n")[1].replace(" ", "")
                            new_label = self._cont_names[
                                int(idx[0]), int(idx[2]), int(idx[4])
                            ]
                            new_label = "\n".join(new_label.split("_"))
                            ax[i, j].set_ylabel(new_label)
                    except Exception as e:
                        pass

            plt.tight_layout()
            plt.savefig(
                os.path.join(outdir, f"{self._date}_global_pairs.png"),
                transparent=True,
                dpi=100,
            )
