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
                f"{self.date:%y%m%d}",
                self.data_type,
                "trigger_result.yml",
            )
        )

    def run(self):
        search = Search(
            result_file=self.input().path,
            min_bin_width=5,
        )

        search.find_changepoints_angles(min_size=3, jump=5, model="l2")

        search.get_significance(required_significance=3)

        search.get_trigger_information()

        search.save_result(self.output().path)
