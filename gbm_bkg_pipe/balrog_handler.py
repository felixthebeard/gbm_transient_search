import datetime as dt
import os
from datetime import datetime, timedelta

import luigi
import yaml
from luigi.contrib.external_program import ExternalProgramTask

from gbm_bkg_pipe.bkg_fit_remote_handler import GBMBackgroundModelFit
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.trigger_search import TriggerSearch
from gbm_bkg_pipe.utils.localization_handler import LocalizationHandler
from gbm_bkg_pipe.utils.result_reader import ResultReader

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")

balrog_n_cores_multinest = gbm_bkg_pipe_config["balrog"]["multinest"]["n_cores"]
balrog_path_to_python = gbm_bkg_pipe_config["balrog"]["multinest"]["path_to_python"]
balrog_timeout = gbm_bkg_pipe_config["balrog"]["timeout"]

run_detectors = gbm_bkg_pipe_config["data"]["detectors"]
run_echans = gbm_bkg_pipe_config["data"]["echans"]


class LocalizeTriggers(luigi.Task):
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

        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date, data_type=self.data_type, remote_host=self.remote_host
            ),
            "trigger_search": TriggerSearch(
                date=self.date, data_type=self.data_type, remote_host=self.remote_host
            ),
        }

        return requirements

    def output(self):
        done_filename = f"localize_triggers_done.txt"
        result_filename = f"localization_result.yml"

        return dict(
            done=luigi.LocalTarget(
                os.path.join(
                    base_dir, f"{self.date:%y%m%d}", self.data_type, done_filename
                )
            ),
            result_file=luigi.LocalTarget(
                os.path.join(
                    base_dir, f"{self.date:%y%m%d}", self.data_type, result_filename
                )
            ),
        )

    def run(self):
        output_dir = os.path.join(
            base_dir, f"{self.date:%y%m%d}", self.data_type, "trigger"
        )

        loc_handler = LocalizationHandler(
            trigger_search_result=self.input()["trigger_search"].path,
            bkg_fit_result=self.input()["bkg_fit"].path,
        )

        loc_handler.create_trigger_information(output_dir)

        loc_handler.write_pha(output_dir)

        balrog_tasks = []
        for t_info in loc_handler.trigger_information.values():

            balrog_tasks.append(
                ProcessLocalizationResult(
                    date=datetime.strptime(t_info["date"], "%y%m%d"),
                    data_type=t_info["data_type"],
                    trigger_name=t_info["trigger_name"],
                    remote_host=self.remote_host,
                )
            )
        yield balrog_tasks

        with self.input()["trigger_search"].open("r") as f:

            trigger_result = yaml.safe_load(f)

        for task in balrog_tasks:

            with task.output()["result_file"].open("r") as f:

                result_yaml = yaml.safe_load(f)

                # Write localization result in original yaml

                trigger_result["triggers"][task.trigger_name].update(
                    dict(
                        balrog_one_sig_err_circle=result_yaml["fit_result"][
                            "balrog_one_sig_err_circle"
                        ],
                        balrog_two_sig_err_circle=result_yaml["fit_result"][
                            "balrog_two_sig_err_circle"
                        ],
                        dec=result_yaml["fit_result"]["dec"],
                        dec_err=result_yaml["fit_result"]["dec_err"],
                        ra=result_yaml["fit_result"]["ra"],
                        ra_err=result_yaml["fit_result"]["ra_err"],
                    )
                )

        with self.output()["result_file"].open("w") as f:

            yaml.dump(trigger_result, f, default_flow_style=False)

        os.system(f"touch {self.output()['done'].path}")


class ProcessLocalizationResult(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
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
        return dict(
            balrog=RunBalrog(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
        )

    def output(self):
        base_job = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "trigger",
            self.trigger_name,
        )
        return {
            "result_file": luigi.LocalTarget(
                os.path.join(base_job, "localization_result.yml")
            ),
            "post_equal_weights": self.input()["balrog"]["post_equal_weights"],
        }

    def run(self):
        trigger_file = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "trigger",
            self.trigger_name,
            "trigger_info.yml",
        )

        result_reader = ResultReader(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            trigger_file=trigger_file,
            post_equal_weights_file=self.input()["balrog"]["post_equal_weights"].path,
            result_file=self.input()["balrog"]["fit_result"].path,
        )

        result_reader.save_result_yml(self.output()["result_file"].path)


class RunBalrog(ExternalProgramTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": balrog_n_cores_multinest}
    worker_timeout = balrog_timeout
    always_log_stderr = True

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def requires(self):
        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date, data_type=self.data_type, remote_host=self.remote_host
            ),
            "trigger_search": TriggerSearch(
                date=self.date, data_type=self.data_type, remote_host=self.remote_host
            ),
        }

        return requirements

    def output(self):
        base_job = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "trigger",
            self.trigger_name,
        )
        fit_result_name = f"{self.trigger_name}_loc_results.fits"
        spectral_plot_name = f"{self.trigger_name}_spectrum_plot.png"

        return {
            "fit_result": luigi.LocalTarget(os.path.join(base_job, fit_result_name)),
            "post_equal_weights": luigi.LocalTarget(
                os.path.join(
                    base_job, "chains", f"{self.trigger_name}_post_equal_weights.dat"
                )
            ),
            #'spectral_plot': luigi.LocalTarget(os.path.join(base_job, 'plots', spectral_plot_name))
        }

    def program_args(self):
        trigger_file = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "trigger",
            self.trigger_name,
            "trigger_info.yml",
        )

        fit_script_path = (
            f"{os.path.dirname(os.path.abspath(__file__))}/balrog/fit_script.py"
        )

        command = []

        # Run with mpi in parallel
        if balrog_n_cores_multinest > 1:

            command.extend(
                [
                    "mpiexec",
                    "-n",
                    f"{balrog_n_cores_multinest}",
                    "--timeout",
                    f"{balrog_timeout - 10}",  # Use timeout - 10s to quite mpi before the task gets killed
                ]
            )

        command.extend(
            [
                f"{balrog_path_to_python}",
                f"{fit_script_path}",
                f"{self.trigger_name}",
                f"{trigger_file}",
            ]
        )

        return command
