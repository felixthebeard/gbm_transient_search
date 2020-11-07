import os
import re
from datetime import datetime
import arviz
import arviz as az
import numpy as np


class BkgArvizReader(object):
    def __init__(self, path_to_netcdf):

        self._create_result_dict(path_to_netcdf)
        self._create_summaries()

    def _create_result_dict(self, path_to_netcdf):

        arviz_result = az.InferenceData.from_netcdf(path_to_netcdf)

        time_bins = arviz_result.constant_data["time_bins"].values

        ndets = len(arviz_result.constant_data["dets"].values)
        nechans = len(arviz_result.constant_data["echans"].values)
        ntime_bins = len(arviz_result.constant_data["time_bins"].values)
        nsamples = len(arviz_result.predictions["chain"]) * len(
            arviz_result.predictions["draw"]
        )

        ppc_counts = arviz_result.posterior_predictive.stack(sample=("chain", "draw"))[
            "ppc"
        ].values.T.reshape((nsamples, ntime_bins, ndets, nechans))

        total_counts = arviz_result.predictions.stack(sample=("chain", "draw"))[
            "tot"
        ].values.T.reshape((nsamples, ntime_bins, ndets, nechans))

        result_dict = dict()
        result_dict["dates"] = arviz_result.constant_data["dates"].values
        result_dict["detectors"] = arviz_result.constant_data["dets"].values
        result_dict["echans"] = arviz_result.constant_data["echans"].values

        result_dict["day_start_times"] = [time_bins[0, 0]]
        result_dict["day_stop_times"] = [time_bins[-1, 1]]
        result_dict["time_bins_start"] = time_bins[:, 0]
        result_dict["time_bins_stop"] = time_bins[:, 1]
        result_dict["total_time_bins"] = time_bins

        result_dict["saa_mask"] = np.ones(ntime_bins, dtype=bool)
        jump_large = time_bins[1:, 0] - time_bins[0:-1, 1] > 10
        idx = jump_large.nonzero()[0] + 1
        result_dict["saa_mask"][idx - 1] = False
        result_dict["saa_mask"][idx] = False

        result_dict["model_counts"] = np.mean(total_counts, axis=0)
        result_dict["observed_counts"] = arviz_result.observed_data[
            "counts"
        ].values.reshape((ntime_bins, ndets, nechans))

        result_dict["sources"] = {}

        # Get the individual sources
        model_parts = arviz_result.predictions.keys()
        predictions = arviz_result.predictions.stack(sample=("chain", "draw"))
        for key in model_parts:
            if key == "tot":
                continue

            model_group = predictions[key].values

            if len(model_group.shape) == 3:
                for k in range(len(model_group)):
                    if key == "f_fixed_global":
                        source_name = (
                            self._arviz_result.constant_data["global_param_names"]
                            .values[k]
                            .replace("norm_", "")
                        )

                    elif key == "f_cont":
                        source_name = ["Constant", "CR_approx"][k]
                    else:
                        source_name = f"{key}_{k}"
                    result_dict["sources"][source_name] = np.mean(
                        model_group[k].T.reshape(
                            (nsamples, ntime_bins, ndets, nechans)
                        ),
                        axis=0,
                    )
            else:
                result_dict["sources"][key] = model_group.T.reshape(
                    (nsamples, ntime_bins, ndets, nechans)
                )

        # Set ppcs in SAA region to zero
        ppc_counts[:, ~result_dict["saa_mask"], :, :] = 0.0
        result_dict["ppc_counts"] = ppc_counts
        result_dict["ppc_time_bins"] = arviz_result.constant_data["time_bins"].values

        result_dict["time_stamp"] = datetime.now().strftime("%y%m%d_%H%M")

        self._arviz_result = arviz_result
        self._result_dict = result_dict

    def _create_summaries(self):
        fixed_summary = az.summary(self._arviz_result, var_names=["norm_fixed"])
        fixed_summary["param_name"] = self._arviz_result.constant_data[
            "global_param_names"
        ].values
        fixed_summary["stan_name"] = fixed_summary.index
        fixed_summary = fixed_summary.set_index("param_name")

        cont_summary = az.summary(self._arviz_result, var_names=["norm_cont"])
        cont_summary["param_name"] = self._arviz_result.constant_data[
            "cont_param_names"
        ].values.flatten()
        cont_summary["stan_name"] = cont_summary.index
        cont_summary = cont_summary.set_index("param_name")

        self._global_summary = fixed_summary
        self._cont_summary = cont_summary

    def hide_point_sources(self, norm_threshold=1.0):
        hide_sources = []

        for param_name, summary in self._global_summary.iterrows():

            if re.search("norm_(.*?)_pl", param_name):

                if summary["mean"] <= norm_threshold:
                    hide_sources.append(param_name.replace("norm_", ""))

        self._sources_to_hide = hide_sources

    @property
    def arviz_result(self):
        return self._arviz_result

    @property
    def result_dict(self):
        return self._result_dict

    @property
    def cont_summary(self):
        return self._cont_summary

    @property
    def global_summary(self):
        return self._global_summary

    @property
    def source_to_hide(self):
        return self._sources_to_hide
