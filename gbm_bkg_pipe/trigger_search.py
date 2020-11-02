import datetime as dt
import os
from datetime import datetime, timedelta

import luigi
import numpy as np

from gbm_bkg_pipe.bkg_fit_remote_handler import DownloadData, GBMBackgroundModelFit
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.utils.search import Search

_valid_gbm_detectors = np.array(gbm_bkg_pipe_config["data"]["detectors"]).flatten()
base_dir = os.environ.get("GBMDATA")


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
        det = _valid_gbm_detectors[0]
        return dict(
            bkg_fit=GBMBackgroundModelFit(
                date=self.date, data_type=self.data_type, remote_host=self.remote_host
            ),
            gbm_data_file=DownloadData(
                date=self.date,
                data_type=self.data_type,
                detector=det,
                remote_host=self.remote_host,
            ),
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
            result_file=self.input()["bkg_fit"].path,
            min_bin_width=5,
            bad_fit_threshold=60,
        )

        search.find_changepoints_angles(min_size=3, jump=5, model="l2")

        search.calc_significances(required_significance=3, max_interval_time=1000)

        search.build_trigger_information()

        search.create_result_dict()

        plot_dir = os.path.join(os.path.dirname(self.output().path))

        search.plot_results(plot_dir)

        search.set_data_timestamp(self.input()["gbm_data_file"].path)

        search.save_result(self.output().path)
