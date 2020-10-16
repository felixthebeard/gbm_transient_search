import datetime as dt
import os
from datetime import datetime, timedelta

import luigi

from gbm_bkg_pipe.balrog_handler import LocalizeTriggers
from gbm_bkg_pipe.bkg_fit_remote_handler import BkgModelPlots
from gbm_bkg_pipe.plots import PlotTriggers

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")


class CreateReportDate(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def requires(self):
        return {
            "loc_triggers": LocalizeTriggers(date=self.date, data_type=self.data_type),
            "plot_triggers": PlotTriggers(date=self.date, data_type=self.data_type),
            "bkg_model_plots": BkgModelPlots(date=self.date, data_type=self.data_type),
        }

    def output(self):
        filename = f"{self.date}_{self.data_type}_report_done.txt"
        return luigi.LocalTarget(
            os.path.join(base_dir, f"{self.date:%y%m%d}", self.data_type, filename)
        )

    def run(self):
        os.system(f"touch {self.output().path}")
