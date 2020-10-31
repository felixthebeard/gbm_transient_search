import datetime as dt
import os
from datetime import datetime, timedelta

import luigi
from luigi.contrib.ssh import RemoteContext, RemoteTarget

from gbm_bkg_pipe.balrog_handler import LocalizeTriggers
from gbm_bkg_pipe.bkg_fit_remote_handler import BkgModelPlots
from gbm_bkg_pipe.upload import UploadTriggers, UploadBkgResultPlots
from gbm_bkg_pipe.plots import PlotTriggers
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
import logging

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")

remote_hosts_config = gbm_bkg_pipe_config["remote_hosts_config"]


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

    def output(self):
        filename = f"{self.date}_{self.data_type}_report_done.txt"
        return luigi.LocalTarget(
            os.path.join(base_dir, f"{self.date:%y%m%d}", self.data_type, filename)
        )

    def run(self):

        remote_hosts = list(remote_hosts_config["hosts"].values())

        available_host_names = []
        available_hosts = []

        for remote_config in remote_hosts:
            try:
                remote = RemoteContext(
                    host=remote_config["hostname"],
                    username=remote_config["username"],
                    sshpass=True,
                )

                check_status_cmd = ["squeue", "-u", remote_config["username"]]

                status = remote.check_output(check_status_cmd)
                nr_queued_jobs = len(status.decode().split("\n")) - 1

                remote_config["nr_queued_jobs"] = nr_queued_jobs

                remote_config["free_capacity"] = (
                    remote_config["job_limit"] - nr_queued_jobs
                )

                available_hosts.append(remote_config)
                available_host_names.append(remote_config["hostname"])

            except Exception as e:

                logging.exception(f"Check remote capacity resulted in: {e}")

        if len(available_hosts) == 0:
            raise Exception("No remote host available to run the heavy tasks")

        # use the host that has the most free capacity
        run_host = sorted(
            available_hosts, key=lambda k: k["free_capacity"], reverse=True
        )[0]["hostname"]

        # if high priority use the priority host
        if self.priority > 1:

            if remote_hosts_config["priority_host"] in available_host_names:

                run_host = remote_hosts_config["priority_host"]

        required_tasks = {
            "upload_bkg_plots": UploadBkgResultPlots(
                date=self.date, data_type=self.data_type, remote_host=run_host
            ),
            "upload_triggers": UploadTriggers(
                date=self.date, data_type=self.data_type, remote_host=run_host
            ),
            "loc_triggers": LocalizeTriggers(
                date=self.date, data_type=self.data_type, remote_host=run_host
            ),
            "plot_triggers": PlotTriggers(
                date=self.date, data_type=self.data_type, remote_host=run_host
            ),
            "bkg_model_plots": BkgModelPlots(
                date=self.date, data_type=self.data_type, remote_host=run_host
            ),
        }

        yield required_tasks

        os.system(f"touch {self.output().path}")
