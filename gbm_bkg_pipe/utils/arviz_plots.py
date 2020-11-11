import re
from datetime import datetime
import os
import arviz
import arviz as az
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
import logging


class ArvizPlotter(object):
    def __init__(self, date, path_to_netcdf):

        self._date = date

        self._arviz_result = az.InferenceData.from_netcdf(path_to_netcdf)

        self._global_names = self._arviz_result.constant_data[
            "global_param_names"
        ].values
        self._cont_names = self._arviz_result.constant_data["cont_param_names"].values

        self._saa_decay_names = np.array([])
        self._saa_norm_names = np.array([])

    def get_param_name(self, stan_name):
        source_name = stan_name.split("\n")[0].replace(" ", "")
        idx = stan_name.split("\n")[1].replace(" ", "")

        if source_name == "norm_fixed":
            param_name = self._global_names[int(idx)]

        elif source_name == "norm_cont":
            param_name = self._cont_names[int(idx[0]), int(idx[2]), int(idx[4])]

        elif source_name == "norm_saa":
            param_name = self._saa_norm_names[int(idx[0]), int(idx[2]), int(idx[4])]

        elif source_name == "decay_saa":
            param_name = self._saa_decay_names[int(idx[0]), int(idx[2]), int(idx[4])]

        else:
            raise Exception(f"Unkown source {source_name} in {stan_name}")

        return param_name

    def plot_posterior(self, var_names, plot_path=None, dpi=100):

        nr_subplots = 10

        if "norm_fixed" in var_names:
            nr_subplots += len(self._global_names.flatten())
        if "norm_cont" in var_names:
            nr_subplots += len(self._cont_names.flatten())
        if "norm_saa" in var_names:
            nr_subplots += len(self._saa_norm_names.flatten())
        if "decay_saa" in var_names:
            nr_subplots += len(self._saa_decay_names.flatten())

        with az.rc_context({"plot.max_subplots": nr_subplots}):

            ax = az.plot_posterior(
                self._arviz_result,
                var_names=var_names,
            )
            ax = np.array(ax)

            for i in range(len(ax)):
                title = ax[i].title.get_text()

                new_title = self.get_param_name(title).replace("_", " ")

                ax[i].set_title(new_title)

            if plot_path is not None:
                plt.savefig(
                    plot_path,
                    transparent=True,
                    dpi=dpi,
                )

    def plot_traces(self, var_names, plot_path=None, dpi=100):
        nr_subplots = 0

        if "norm_fixed" in var_names:
            nr_subplots += len(self._global_names.flatten())
        if "norm_cont" in var_names:
            nr_subplots += len(self._cont_names.flatten())
        if "norm_saa" in var_names:
            nr_subplots += len(self._saa_norm_names.flatten())
        if "decay_saa" in var_names:
            nr_subplots += len(self._saa_decay_names.flatten())

        nr_subplots = 2 * nr_subplots + 10

        with az.rc_context({"plot.max_subplots": nr_subplots}):

            ax = az.plot_trace(
                self._arviz_result,
                var_names=var_names,
            )

            ax = np.array(ax)

            for i in range(len(ax)):

                for j in range(len(ax[i])):

                    title = ax[i, j].title.get_text()

                    new_title = self.get_param_name(title).replace("_", " ")

                    ax[i, j].set_title(new_title)

            if plot_path is not None:
                plt.savefig(
                    plot_path,
                    transparent=True,
                    dpi=dpi,
                )

    def plot_pairs(self, var_names, plot_path, dpi=100):
        nr_subplots = 0

        if "norm_fixed" in var_names:
            nr_subplots += len(self._global_names.flatten())
        if "norm_cont" in var_names:
            nr_subplots += len(self._cont_names.flatten())
        if "norm_saa" in var_names:
            nr_subplots += len(self._saa_norm_names.flatten())
        if "decay_saa" in var_names:
            nr_subplots += len(self._saa_decay_names.flatten())

        nr_subplots = nr_subplots ** 2 + 10

        with az.rc_context({"plot.max_subplots": nr_subplots}):

            ax = az.plot_pair(
                self._arviz_result, var_names=var_names, textsize=25, kind="hexbin"
            )

            if type(ax) != np.ndarray:
                xlabel = ax.get_xlabel()
                if xlabel != "":
                    new_label = self.get_param_name(xlabel).replace("_", " \n")
                    ax.set_xlabel(new_label)

                ylabel = ax.get_ylabel()
                if ylabel != "":
                    new_label = self.get_param_name(ylabel).replace("_", " \n")
                    ax.set_ylabel(new_label)
            else:
                for i in range(len(ax)):

                    for j in range(len(ax[i])):

                        xlabel = ax[i, j].get_xlabel()
                        if xlabel != "":
                            new_label = self.get_param_name(xlabel).replace("_", " \n")
                            ax[i, j].set_xlabel(new_label)

                        ylabel = ax[i, j].get_ylabel()
                        if ylabel != "":
                            new_label = self.get_param_name(ylabel).replace("_", " \n")
                            ax[i, j].set_ylabel(new_label)

            if plot_path is not None:
                plt.savefig(
                    plot_path,
                    transparent=True,
                    dpi=dpi,
                )
