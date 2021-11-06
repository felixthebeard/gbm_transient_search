import datetime as dt
import logging
import os
from datetime import datetime, timedelta

import luigi
from gbm_transient_search.utils.luigi_ssh import RemoteContext

from gbm_transient_search.handlers.localization import LocalizeTriggers
from gbm_transient_search.handlers.plotting import BkgModelPlots
from gbm_transient_search.utils.configuration import gbm_transient_search_config
from gbm_transient_search.handlers.plotting import PlotTriggers
from gbm_transient_search.handlers.transient_search import TransientSearch
from gbm_transient_search.handlers.upload import (
    UploadBkgResultPlots,
    UploadTriggers,
    UploadBkgPerformancePlots,
    UploadBkgFitResult,
)
from gbm_transient_search.utils.env import get_bool_env_value, get_env_value

base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")

remote_hosts_config = gbm_transient_search_config["remote_hosts_config"]


class CreateReportDate(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter(default="default")

    resources = {"ssh_connections": 1}

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
        if self.remote_host == "default":
            remote_host_names = remote_hosts_config["hosts"].get_child_names()
            remote_hosts = [remote_hosts_config["hosts"][rh_name] for rh_name in remote_host_names]

            available_host_names = []
            available_hosts = []

            for remote_config in remote_hosts:
                try:
                    remote = RemoteContext(
                        host=remote_config["hostname"],
                        username=remote_config["username"],
                        # sshpass=True,
                    )

                    check_status_cmd = ["squeue", "-u", remote_config["username"]]

                    status = remote.check_output(check_status_cmd)
                    nr_queued_jobs = len(status.decode().strip().split("\n")) - 1

                    remote_config["nr_queued_jobs"] = nr_queued_jobs

                    remote_config["free_capacity"] = (
                        remote_config["job_limit"] - nr_queued_jobs
                    )

                    available_hosts.append(remote_config)
                    available_host_names.append(remote_config["hostname"])

                except Exception as e:

                    logging.exception(
                        f"Check remote capacity for {remote_config['hostname']} resulted in: {e}"
                    )

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
        else:
            run_host = self.remote_host

        required_tasks = {
            "upload_bkg_plots": UploadBkgResultPlots(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
            "upload_bkg_performance_plots": UploadBkgPerformancePlots(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
            "upload_triggers": UploadTriggers(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
            "loc_triggers": LocalizeTriggers(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
            "plot_triggers": PlotTriggers(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
            "bkg_model_plots_base": BkgModelPlots(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
            "upload_bkg_fit_result": UploadBkgFitResult(
                date=self.date,
                data_type=self.data_type,
                remote_host=run_host,
                step="base",
            ),
        }

        yield required_tasks

        os.system(f"touch {self.output().path}")


class CreateTriggerSearchReport(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()
    step = luigi.Parameter(default="all")

    resources = {"ssh_connections": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def output(self):
        filename = f"{self.date}_{self.data_type}_trigger_search_report_done.txt"
        return luigi.LocalTarget(
            os.path.join(base_dir, f"{self.date:%y%m%d}", self.data_type, filename)
        )

    def requires(self):
        if self.step != "all":
            requires = {
                "bkg_model_plots": BkgModelPlots(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
                "search_triggers": TransientSearch(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
                "plot_triggers": PlotTriggers(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step=self.step,
                    loc_plots=False,
                ),
            }
        else:
            requires = {
                "bkg_model_plots_base": BkgModelPlots(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step="base",
                ),
                "bkg_model_plots_final": BkgModelPlots(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step="final",
                ),
                "search_triggers_base": TransientSearch(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step="base",
                ),
                "search_triggers_final": TransientSearch(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step="final",
                ),
                "plot_triggers_base": PlotTriggers(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step="base",
                    loc_plots=False,
                ),
                "plot_triggers_final": PlotTriggers(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step="final",
                    loc_plots=False,
                ),
            }
        return requires

    def run(self):
        os.system(f"touch {self.output().path}")
