import datetime as dt
import logging
import os
import random
import subprocess
import time
from datetime import datetime, timedelta

import luigi
import numpy as np
import yaml
from luigi.contrib.external_program import ExternalProgramTask

from gbm_bkg_pipe.utils.luigi_ssh import (
    RemoteCalledProcessError,
    RemoteContext,
    RemoteTarget,
)
from gbm_bkg_pipe.bkg_fit_remote_handler import (
    DownloadData,
    DownloadPoshistData,
    GBMBackgroundModelFit,
)
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.trigger_search import TriggerSearch
from gbm_bkg_pipe.utils.env import get_bool_env_value, get_env_value
from gbm_bkg_pipe.utils.localization_handler import LocalizationHandler
from gbm_bkg_pipe.utils.result_reader import ResultReader

base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")

balrog_run_destination = gbm_bkg_pipe_config["balrog"]["run_destination"]
balrog_n_cores_multinest = gbm_bkg_pipe_config["balrog"]["multinest"]["n_cores"]
balrog_path_to_python = gbm_bkg_pipe_config["balrog"]["multinest"]["path_to_python"]
balrog_timeout = gbm_bkg_pipe_config["balrog"]["timeout"]

_valid_gbm_detectors = np.array(gbm_bkg_pipe_config["data"]["detectors"]).flatten()
run_detectors = gbm_bkg_pipe_config["data"]["detectors"]
run_echans = gbm_bkg_pipe_config["data"]["echans"]

remote_hosts_config = gbm_bkg_pipe_config["remote_hosts_config"]


class LocalizeTriggers(luigi.Task):
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

        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "trigger_search": TriggerSearch(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "setup_loc": SetupTriggerLocalization(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
        }

        return requirements

    def output(self):
        done_filename = f"localize_triggers_done.txt"
        result_filename = f"localization_result.yml"

        return dict(
            done=luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    self.data_type,
                    self.step,
                    done_filename,
                )
            ),
            result_file=luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    self.data_type,
                    self.step,
                    result_filename,
                )
            ),
        )

    def run(self):
        with self.input()["setup_loc"]["trigger_information"].open("r") as f:

            trigger_information = yaml.safe_load(f)

        balrog_tasks = []
        for t_info in trigger_information.values():

            balrog_tasks.append(
                ProcessLocalizationResult(
                    date=datetime.strptime(t_info["date"], "%y%m%d"),
                    data_type=t_info["data_type"],
                    trigger_name=t_info["trigger_name"],
                    remote_host=self.remote_host,
                    step=self.step,
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


class SetupTriggerLocalization(luigi.Task):
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

        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "trigger_search": TriggerSearch(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
        }

        return requirements

    def output(self):
        done_filename = f"setup_localization_done.txt"

        return dict(
            done=luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    self.data_type,
                    self.step,
                    done_filename,
                )
            ),
            trigger_information=luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    self.data_type,
                    self.step,
                    "trigger",
                    "trigger_information.yml",
                )
            ),
        )

    def run(self):
        output_dir = os.path.join(
            base_dir, f"{self.date:%y%m%d}", self.data_type, self.step, "trigger"
        )

        loc_handler = LocalizationHandler(
            trigger_search_result=self.input()["trigger_search"].path,
            bkg_fit_result=self.input()["bkg_fit"].path,
        )

        loc_handler.create_trigger_information(output_dir)

        loc_handler.write_pha(output_dir)

        os.system(f"touch {self.output()['done'].path}")


class ProcessLocalizationResult(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
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

    @property
    def job_dir(self):
        return os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    def requires(self):
        if balrog_run_destination == "local":
            return dict(
                balrog=RunBalrog(
                    date=self.date,
                    data_type=self.data_type,
                    trigger_name=self.trigger_name,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
            )
        elif balrog_run_destination == "remote":
            # Add all intermediate tasks to the requeriements here
            # to speed up building the luigi DAG
            return dict(
                balrog_remote_run=RunBalrogTasksRemote(
                    date=self.date,
                    data_type=self.data_type,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
                run_balrog=RunBalrogRemote(
                    date=self.date,
                    data_type=self.data_type,
                    trigger_name=self.trigger_name,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
                balrog=CopyRemoteBalrogResult(
                    date=self.date,
                    data_type=self.data_type,
                    trigger_name=self.trigger_name,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
                copy_trigger_files=CopyTriggerFilesToRemote(
                    date=self.date,
                    data_type=self.data_type,
                    trigger_name=self.trigger_name,
                    remote_host=self.remote_host,
                    step=self.step,
                ),
            )
        else:
            raise Exception("Unkown balrog run destination")

    def output(self):
        return {
            "result_file": luigi.LocalTarget(
                os.path.join(self.job_dir, "localization_result.yml")
            ),
            "post_equal_weights": self.input()["balrog"]["post_equal_weights"],
        }

    def run(self):
        trigger_file = os.path.join(self.job_dir, "trigger_info.yml")

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
    step = luigi.Parameter()

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

    @property
    def job_dir(self):
        return os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    def requires(self):
        requirements = {
            "bkg_fit": GBMBackgroundModelFit(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "trigger_search": TriggerSearch(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
        }

        return requirements

    def output(self):
        fit_result_name = f"{self.trigger_name}_loc_results.fits"
        spectral_plot_name = f"{self.trigger_name}_spectrum_plot.png"

        return {
            "fit_result": luigi.LocalTarget(
                os.path.join(self.job_dir, fit_result_name)
            ),
            "post_equal_weights": luigi.LocalTarget(
                os.path.join(
                    self.job_dir,
                    "chains",
                    f"{self.trigger_name}_post_equal_weights.dat",
                )
            ),
            #'spectral_plot': luigi.LocalTarget(os.path.join(base_job, 'plots', spectral_plot_name))
        }

    def program_args(self):
        trigger_file = os.path.join(self.job_dir, "trigger_info.yml")

        fit_script_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/scripts/run_balrog.py"

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
                "--trigger_name",
                f"{self.trigger_name}",
                "--trigger_info",
                f"{trigger_file}",
            ]
        )

        return command


class CopyTriggerFilesToRemote(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    resources = {"cpu": 1, "ssh_connections": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    @property
    def job_dir(self):
        return os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    @property
    def remote_job_dir(self):
        return os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["base_dir"],
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    def requires(self):
        return SetupTriggerLocalization(
            date=self.date,
            data_type=self.data_type,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        return dict(
            success=luigi.LocalTarget(
                os.path.join(self.job_dir, "copied_trigger_files.done")
            ),
        )

    def remote_output(self):
        return dict(
            pha_dir=RemoteTarget(
                os.path.join(
                    self.remote_job_dir,
                    "pha",
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            trigger_info=RemoteTarget(
                os.path.join(
                    self.remote_job_dir,
                    f"trigger_info.yml",
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
        )

    def run(self):
        local_pha_dir = luigi.LocalTarget(os.path.join(self.job_dir, "pha"))
        if local_pha_dir.exists():
            self.remote_output()["pha_dir"].put(local_pha_dir.path)
        else:
            raise Exception(f"Local pha dir {local_pha_dir.path} is missing.")

        local_trigger_info = luigi.LocalTarget(
            os.path.join(self.job_dir, "trigger_info.yml")
        )

        if local_trigger_info.exists():
            self.remote_output()["trigger_info"].put(local_trigger_info.path)
        else:
            raise Exception(f"Local pha dir {local_trigger_info.path} is missing.")

        os.system(f"touch {self.output()['success'].path}")


class CopyRemoteBalrogResult(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    resources = {"cpu": 1, "ssh_connections": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    @property
    def job_dir(self):
        return os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    def requires(self):
        return dict(
            balrog_remote=RunBalrogRemote(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
        )

    def output(self):
        fit_result_name = f"{self.trigger_name}_loc_results.fits"
        spectral_plot_name = f"{self.trigger_name}_spectrum_plot.png"

        return {
            "fit_result": luigi.LocalTarget(
                os.path.join(self.job_dir, fit_result_name)
            ),
            "post_equal_weights": luigi.LocalTarget(
                os.path.join(
                    self.job_dir,
                    "chains",
                    f"{self.trigger_name}_post_equal_weights.dat",
                )
            ),
        }

    def optional_output(self):
        # Optional outputs that should be accessible for upstream tasks but should not
        # fail a task if not existing.
        return {
            "spectrum_plot": luigi.LocalTarget(
                os.path.join(
                    self.job_dir,
                    "plots",
                    f"{self.trigger_name}_spectrum_plot_{self.data_type}.png",
                )
            ),
        }

    def run(self):
        # Copy result file to local folder
        self.requires()["balrog_remote"].remote_output()["fit_result"].get(
            self.output()["fit_result"].path
        )

        # Copy arviz file to local folder
        self.requires()["balrog_remote"].remote_output()["post_equal_weights"].get(
            self.output()["post_equal_weights"].path
        )

        if self.requires()["balrog_remote"].optional_output()["spectrum_plot"].exists():
            self.requires()["balrog_remote"].optional_output()["spectrum_plot"].get(
                self.optional_output()["spectrum_plot"].path
            )


class RunBalrogRemote(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    resources = {"cpu": 1, "ssh_connections": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    @property
    def job_dir(self):
        return os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    @property
    def job_dir_remote(self):
        return os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["base_dir"],
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
            self.trigger_name,
        )

    def requires(self):
        return dict(
            balrog_remote_tasks=RunBalrogTasksRemote(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            )
        )

    def output(self):
        return dict(
            success=luigi.LocalTarget(
                os.path.join(self.job_dir, "run_balrog_remote.done")
            )
        )

    def remote_output(self):
        fit_result_name = f"{self.trigger_name}_loc_results.fits"

        return {
            "fit_result": RemoteTarget(
                os.path.join(self.job_dir_remote, fit_result_name),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            "post_equal_weights": RemoteTarget(
                os.path.join(
                    self.job_dir_remote,
                    "chains",
                    f"{self.trigger_name}_post_equal_weights.dat",
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            "success": RemoteTarget(
                os.path.join(
                    self.job_dir_remote,
                    f"{self.trigger_name}_balrog.success",
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
        }

    def optional_output(self):
        # Optional outputs that should be accessible for upstream tasks but should not
        # fail a task if not existing.
        return {
            "spectrum_plot": RemoteTarget(
                os.path.join(
                    self.job_dir_remote,
                    "plots",
                    f"{self.trigger_name}_spectrum_plot_{self.data_type}.png",
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
        }

    def run_remote_command(self, cmd):
        remote = RemoteContext(
            host=self.remote_host,
            username=remote_hosts_config["hosts"][self.remote_host]["username"],
            # sshpass=True,
        )
        output = remote.check_output(cmd)
        del remote
        return output

    def run(self):

        check_status_cmd = [
            "squeue",
            "-u",
            remote_hosts_config["hosts"][self.remote_host]["username"],
        ]

        with self.input()["balrog_remote_tasks"]["job_id"].open("r") as f:
            job_id = f.readlines()[0]

        if isinstance(job_id, bytes):
            job_id = job_id.decode()

        # the time spent waiting so far
        time_spent = 0  # seconds
        wait_time = 5 * 60
        max_time = 2 * 60 * 60

        # Sleep for 5m initially and add reandom sleep to avoid querying at the same time
        time.sleep(5 * 60 + random.randint(30, 100))

        while True:

            if self.remote_output()["success"].exists():

                os.system("touch {self.output()['success'].path}")

                return True

            else:

                if time_spent >= max_time:

                    return False

                else:

                    status = self.run_remote_command(check_status_cmd)

                    status = status.decode()

                    if not str(job_id) in status:
                        # Remove the job_id file to allow for rerun.
                        self.input()["balrog_remote_tasks"]["job_id"].remove()
                        raise Exception(f"The job {job_id} did fail, kill task.")

                    for line in status.split("\n"):

                        if str(job_id) in line:

                            logging.info(f"The squeue status: {line}")

                    time.sleep(wait_time)

                    time_spent += wait_time


class RunBalrogTasksRemote(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    result_timeout = 2 * 60 * 60

    resources = {"ssh_connections": 1}

    @property
    def retry_count(self):
        return 0

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    @property
    def job_dir(self):
        return os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
        )

    @property
    def job_dir_remote(self):
        return os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["base_dir"],
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "trigger",
        )

    def requires(self):
        requires = dict(
            poshist_file=DownloadPoshistData(
                date=self.date, remote_host=self.remote_host
            ),
            setup_loc=SetupTriggerLocalization(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
        )
        for det in _valid_gbm_detectors:
            requires[f"data_{det}"] = DownloadData(
                date=self.date,
                data_type=self.data_type,
                detector=det,
                remote_host=self.remote_host,
            )

        return requires

    def output(self):
        return dict(
            job_id=luigi.LocalTarget(
                os.path.join(
                    self.job_dir,
                    "job_id.txt",
                ),
            ),
        )

    def run_remote_command(self, cmd):
        remote = RemoteContext(
            host=self.remote_host,
            username=remote_hosts_config["hosts"][self.remote_host]["username"],
            # sshpass=True,
        )

        p = remote.Popen(cmd, stdout=subprocess.PIPE)
        output, _ = p.communicate()
        if p.returncode != 0:
            raise RemoteCalledProcessError(
                p.returncode, cmd, self.remote_host, output=output
            )
        try:
            p.terminate()
        except Exception as e:
            print(e)
        del remote

        return output

    def run(self):

        with self.input()["setup_loc"]["trigger_information"].open("r") as f:

            trigger_information = yaml.safe_load(f)

        trigger_files = {}

        for t_info in trigger_information.values():
            trigger_files[t_info["trigger_name"]] = CopyTriggerFilesToRemote(
                date=self.date,
                data_type=self.data_type,
                trigger_name=t_info["trigger_name"],
                remote_host=self.remote_host,
                step=self.step,
            )

        yield trigger_files

        remote_trigger_information = RemoteTarget(
            os.path.join(
                self.job_dir_remote,
                f"trigger_information.yml",
            ),
            host=self.remote_host,
            username=remote_hosts_config["hosts"][self.remote_host]["username"],
            # sshpass=True,
        )

        remote_trigger_information.put(
            self.input()["setup_loc"]["trigger_information"].path
        )

        job_script_path = os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["script_dir"],
            "balrog_tasks.job",
        )

        if self.priority > 1:
            nice = 0
        else:
            nice = 100

        run_cmd = [
            "sbatch",
            "--parsable",
            f"--nice={nice}",
            "-D",
            f"{self.job_dir_remote}",
            f"{job_script_path}",
            f"{remote_trigger_information.path}",
        ]

        logging.info(" ".join(run_cmd))

        job_output = self.run_remote_command(run_cmd)

        job_id = job_output.decode().strip().replace("\n", "")

        with self.output()["job_id"].open("w") as outfile:
            outfile.write(job_id)
