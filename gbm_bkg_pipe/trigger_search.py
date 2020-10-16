import datetime as dt
import os
from datetime import datetime, timedelta

import luigi

from gbm_bkg_pipe.bkg_fit_remote_handler import GBMBackgroundModelFit
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
    remote_host = luigi.Parameter()

    resources = {"cpu": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def requires(self):

        return GBMBackgroundModelFit(
            date=self.date, data_type=self.data_type, remote_host=self.remote_host
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

        search.calc_significances(required_significance=3)

        search.build_trigger_information()

        search.create_result_dict()

        plot_dir = os.path.join(os.path.dirname(self.output().path))

        search.plot_results(plot_dir)

        search.save_result(self.output().path)
