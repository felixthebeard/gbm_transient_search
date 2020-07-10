import os
import luigi
from datetime import datetime
from luigi.contrib.external_program import ExternalProgramTask

from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.bkg_fit_handler import GBMBackgroundModelFit
from gbm_bkg_pipe.trigger_search import TriggerSearch
from gbm_bkg_pipe.utils.localization_handler import LocalizationHandler


base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")
balrog_n_cores_multinest = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["n_cores"]
balrog_path_to_python = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["path_to_python"]
bkg_n_cores_multinest = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["n_cores"]
bkg_path_to_python = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["path_to_python"]

run_detectors = gbm_bkg_pipe_config["data"]["detectors"]
run_echans = gbm_bkg_pipe_config["data"]["echans"]


class LocalizeTriggers(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")

    resources = {
        "cpu": 1
    }
    def requires(self):

        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date, data_type=self.data_type
            ),
            "trigger_search": TriggerSearch(
                date=self.date, data_type=self.data_type
            ),
        }

        return requirements

    def output(self):
        filename = f"localize_triggers_done.txt"

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                filename
            )
        )

    def run(self):
        output_dir = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "trigger"
        )

        loc_handler = LocalizationHandler(
            trigger_search_result=self.input()["trigger_search"].path,
            bkg_fit_result=self.input()["bkg_fit"].path
        )

        loc_handler.create_trigger_information(output_dir)

        loc_handler.write_pha(output_dir)

        for t_info in loc_handler.trigger_information:

            yield RunBalrog(
                date=datetime.strptime(t_info["date"], "%y%m%d"),
                data_type=t_info["data_type"],
                trigger_name=t_info["trigger_name"],
            )


class RunBalrog(ExternalProgramTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()

    resources = {
        "cpu": balrog_n_cores_multinest
    }
    worker_timeout = 60 * 60  # 1 hour
    always_log_stderr = True

    def requires(self):

        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date, data_type=self.data_type
            ),
            "trigger_search": TriggerSearch(
                date=self.date, data_type=self.data_type
            ),
        }

        return requirements

    def output(self):
        base_job = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.trigger_name
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

        trigger_info = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            'trigger',
            self.trigger_name,
            "trigger_info.yml"
        )

        fit_script_path = (
            f"{os.path.dirname(os.path.abspath(__file__))}/balrog/fit_script.py"
        )

        command = [
            "mpiexec",
            f"-n",
            f"{balrog_n_cores_multinest}",
            f"{balrog_path_to_python}",
            f"{fit_script_path}",
            f"{self.trigger_name}",
            f"{trigger_info}",
        ]

        return command
