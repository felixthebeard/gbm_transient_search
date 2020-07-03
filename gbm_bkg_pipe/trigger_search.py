import os
import time
import json
import numpy as np
import luigi
import yaml
from matplotlib import pyplot as plt

from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.bkg_fit_handler import GBMBackgroundModelFit
from gbm_bkg_pipe.utils.search import Search

base_dir = os.environ.get("GBMDATA")

_gbm_detectors = (
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
)


class TriggerSearch(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")

    resources = {
        "cpu": 1
    }

    def requires(self):

        return GBMBackgroundModelFit(
            date=self.date, data_type=self.data_type
        )

    def output(self):

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                "bkg_pipe",
                self.data_type,
                f"{self.date:%y%m%d}",
                "triggers.txt",
            )
        )

    def run(self):
        search = Search(
            result_file=self.input().path,
            min_bin_width=15,
        )

        methods = {
            "binseg": dict(model="l2"),
            "pelt": dict(min_size=3, jump=5, model="l1"),
            "dynp": dict(min_size=3, jump=5, model="l1"),
            "bottomup": dict(model="l2"),
            "window": dict(width=40, model="l2")
        }

        for method, kwargs in methods.items():
            print(method)
            search.find_change_points_raptures(method=method, **kwargs)

            search.find_change_points_raptures()

            search.get_significant_regions(snr=False, required_significance=3)

            search.correlate_detectors()

            plt.plot(
                search._rebinned_mean_time[search._rebinned_saa_mask],
                search._significant_mask_dets
            )

            plt.xlabel("Time [MET]");
            plt.ylabel("# of detectors with ROI");

            plot_path = os.path.join(
                base_dir,
                "bkg_pipe",
                self.data_type,
                f"{self.date:%y%m%d}",
                f"{method}.png",
            )
            plt.savefig(plot_path)

        os.system(f"touch {self.output().path}")
