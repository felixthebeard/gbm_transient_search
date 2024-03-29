import os
import time
import json
import numpy as np
import h5py
import yaml

from gbm_transient_search.utils.file_utils import (
    if_dir_containing_file_not_existing_then_make,
)
from gbmgeometry import GBMTime
from gbmbkgpy.io.export import PHAWriter


class LocalizationSetup(object):
    def __init__(self, transient_search_result, bkg_fit_result):

        self._read_search_result(transient_search_result)

        self._read_bkg_result(bkg_fit_result)

    def _read_search_result(self, result_file):

        with open(result_file, "r") as f:
            trigger_search_result = yaml.safe_load(f)

        self._search_result = trigger_search_result

    def _read_bkg_result(self, result_file):

        self._pha_writer = PHAWriter.from_combined_hdf5(result_file)

    def create_trigger_information(self, output_dir):

        trigger_information = {}

        for trigger in self._search_result["triggers"].values():

            use_dets = self._choose_dets(trigger["most_significant_detector"])
            peak_time = trigger["peak_time"]

            active_time_start = peak_time - 10
            active_time_end = peak_time + 10

            trigger["good_bkg_fit_mask"] = self._search_result["good_bkg_fit_mask"]
            trigger["data_type"] = self._search_result["data_type"]
            trigger["data_timestamp"] = self._search_result.get("data_timestamp", None)
            trigger["active_time_start"] = active_time_start
            trigger["active_time_end"] = active_time_end
            trigger["use_dets"] = use_dets
            trigger["spectral_model"] = "blackbody"

            output_file = os.path.join(
                output_dir, trigger["trigger_name"], "trigger_info.yml"
            )

            if_dir_containing_file_not_existing_then_make(output_file)

            with open(output_file, "w") as f:
                yaml.dump(trigger, f, default_flow_style=False)

            trigger_information[trigger["trigger_name"]] = trigger

        self.trigger_information = trigger_information

        output_file = os.path.join(output_dir, "trigger_information.yml")
        with open(output_file, "w") as f:
            yaml.dump(trigger_information, f, default_flow_style=False)

    def _choose_dets(self, max_det):
        """
        Function to automatically choose the detectors which should be used in the fit
        :return:
        """
        side_1_dets = ["n0", "n1", "n2", "n3", "n4", "n5"]  # , "b0"]
        side_2_dets = ["n6", "n7", "n8", "n9", "na", "nb"]  # , "b1"]

        # only use the detectors on the same side as the detector with the most significance
        if max_det in side_1_dets:

            use_dets = side_1_dets

        else:
            use_dets = side_2_dets

        return use_dets

    def write_pha(self, output_dir):

        for t_info in self.trigger_information.values():

            output_path = os.path.join(output_dir, t_info["trigger_name"], "pha")

            if_dir_containing_file_not_existing_then_make(output_path)

            self._pha_writer.write_pha(
                output_dir=output_path,
                trigger_time=t_info["trigger_time"],
                active_time_start=t_info["active_time_start"],
                active_time_end=t_info["active_time_end"],
                file_name=t_info["trigger_name"],
                overwrite=True,
            )
