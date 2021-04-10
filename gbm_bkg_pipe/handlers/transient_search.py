import datetime as dt
import os
from datetime import datetime, timedelta

import luigi
import numpy as np

from gbm_bkg_pipe.handlers.background import GBMBackgroundModelFit
from gbm_bkg_pipe.handlers.download import DownloadData
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.processors.transient_detector import TransientDetector
from gbm_bkg_pipe.utils.env import get_bool_env_value, get_env_value

_valid_gbm_detectors = np.array(gbm_bkg_pipe_config["data"]["detectors"]).flatten()
base_dir = get_env_value("GBMDATA")

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")


class TransientSearch(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

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
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
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
                self.step,
                "trigger_result.yml",
            )
        )

    def run(self):
        plot_dir = os.path.join(os.path.dirname(self.output().path))

        transient_detector = TransientDetector(
            result_file=self.input()["bkg_fit"].path,
            min_bin_width=5,
            bad_fit_threshold=100,
        )

        transient_detector.run()

        transient_detector.plot_results(plot_dir)

        transient_detector.set_data_timestamp(
            self.input()["gbm_data_file"]["local_file"].path
        )

        transient_detector.save_result(self.output().path)
