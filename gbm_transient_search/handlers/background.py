import datetime as dt
import logging
import os
import random
import time
from datetime import datetime, timedelta

import h5py
import luigi
import yaml
from gbm_transient_search.utils.configuration import gbm_transient_search_config
from gbm_transient_search.handlers.download import (
    DownloadData,
    DownloadLATData,
    DownloadPoshistData,
    UpdatePointsourceDB,
)
from gbm_transient_search.processors.bkg_config_writer import BkgConfigWriter
from gbm_transient_search.utils.env import get_bool_env_value, get_env_value
from gbm_transient_search.utils.luigi_ssh import (
    RemoteCalledProcessError,
    RemoteContext,
    RemoteTarget,
)
from gbmbkgpy.io.export import PHAWriter

base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
data_dir = os.environ.get("GBMDATA")

run_detectors = gbm_transient_search_config["data"]["detectors"]
run_echans = gbm_transient_search_config["data"]["echans"]

remote_hosts_config = gbm_transient_search_config["remote_hosts_config"]


class GBMBackgroundModelFit(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def requires(self):

        bkg_fit_tasks = {}

        for dets in run_detectors:

            for echans in run_echans:

                bkg_fit_tasks[
                    f"bkg_d{'_'.join(dets)}_e{'_'.join(echans)}"
                ] = CopyResults(
                    date=self.date,
                    echans=echans,
                    detectors=dets,
                    remote_host=self.remote_host,
                    step=self.step,
                )

        return bkg_fit_tasks

    def output(self):

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "phys_bkg",
                "phys_bkg_combined.hdf5",
            )
        )

    def run(self):

        bkg_fit_results = []

        for dets in run_detectors:

            for echans in run_echans:

                bkg_fit_results.append(
                    self.input()[f"bkg_d{'_'.join(dets)}_e{'_'.join(echans)}"][
                        "result_file"
                    ].path
                )

        # PHACombiner and save combined file
        pha_writer = PHAWriter.from_result_files(bkg_fit_results)

        pha_writer.save_combined_hdf5(self.output().path)


class BkgModelTask(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

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
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )

    @property
    def job_dir_remote(self):
        return os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["base_dir"],
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )


class CreateBkgConfig(BkgModelTask):

    resources = {"cpu": 1, "ssh_connections": 1}

    def requires(self):
        requires = {
            "pointsource_db": UpdatePointsourceDB(
                date=self.date, remote_host=self.remote_host
            )
        }
        if self.step == "final":
            from gbm_transient_search.handlers.transient_search import TransientSearch

            requires["transient_search"] = TransientSearch(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step="base",
            )
        return requires

    def output(self):
        return dict(
            config=luigi.LocalTarget(
                os.path.join(self.job_dir, "config_fit.yml")
            ),
            ps_file=luigi.LocalTarget(
                os.path.join(
                    data_dir,
                    "point_sources",
                    "ps_all_swift.dat"
                )
            )
        )

    def remote_output(self):
        return dict(
            config=RemoteTarget(
                os.path.join(self.job_dir_remote, "config_fit.yml"),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            ps_file=RemoteTarget(
                os.path.join(
                    remote_hosts_config["hosts"][self.remote_host]["data_dir"],
                    "point_sources",
                    "ps_all_swift.dat"
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
            )
        )

    def run(self):

        config_writer = BkgConfigWriter(
            self.date, self.data_type, self.echans, self.detectors, step=self.step
        )

        config_writer.build_config()

        if self.step == "final":
            config_writer.mask_triggers(self.input()["transient_search"].path)

        with self.remote_output()["config"].open("w") as outfile:

            yaml.dump(config_writer._config, outfile, default_flow_style=False)

        # Copy config file to local folder
        self.remote_output()["config"].get(self.output()["config"].path)

        # Copy ps file
        self.remote_output()["ps_file"].put(self.output()["ps_file"].path)


class CopyResults(BkgModelTask):
    resources = {"cpu": 1, "ssh_connections": 1}

    def requires(self):
        return dict(
            bkg_fit=RunPhysBkgModel(
                date=self.date,
                echans=self.echans,
                detectors=self.detectors,
                remote_host=self.remote_host,
                step=self.step,
            ),
            config_file=CreateBkgConfig(
                date=self.date,
                data_type=self.data_type,
                echans=self.echans,
                detectors=self.detectors,
                remote_host=self.remote_host,
                step=self.step,
            ),
        )

    def output(self):
        result_file_name = "fit_result_{}_{}_e{}.hdf5".format(
            f"{self.date:%y%m%d}",
            "-".join(self.detectors),
            "-".join(self.echans),
        )
        arviz_file_name = "fit_result_{}_{}_e{}.nc".format(
            f"{self.date:%y%m%d}",
            "-".join(self.detectors),
            "-".join(self.echans),
        )
        return {
            "result_file": luigi.LocalTarget(
                os.path.join(self.job_dir, result_file_name)
            ),
            "arviz_file": luigi.LocalTarget(
                os.path.join(self.job_dir, arviz_file_name)
            ),
            "best_fit_file": luigi.LocalTarget(
                os.path.join(self.job_dir, "best_fit_params.yml")
            ),
        }

    def run(self):
        # Copy result file to local folder
        if not self.output()["result_file"].exists():
            self.requires()["bkg_fit"].remote_output()["result_file"].get(
                self.output()["result_file"].path
            )

        # Copy arviz file to local folder
        if not self.output()["arviz_file"].exists():
            self.requires()["bkg_fit"].remote_output()["arviz_file"].get(
                self.output()["arviz_file"].path
            )

        with h5py.File(self.output()["result_file"].path, "r") as f:
            best_fit_values = f.attrs["best_fit_values"].tolist()
            param_names = f.attrs["param_names"].tolist()

        with self.output()["best_fit_file"].open("w") as f:
            yaml.dump(
                dict(zip(param_names, best_fit_values)), f, default_flow_style=False
            )


class RunPhysBkgModel(BkgModelTask):
    result_timeout = gbm_transient_search_config["phys_bkg"]["timeout"]

    resources = {"ssh_connections": 1}

    @property
    def retry_count(self):
        return 4

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 2

    def requires(self):
        requires = {
            "config": CreateBkgConfig(
                date=self.date,
                data_type=self.data_type,
                echans=self.echans,
                detectors=self.detectors,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "poshist_file": DownloadPoshistData(
                date=self.date, remote_host=self.remote_host
            ),
        }

        for det in self.detectors:
            requires[f"data_{det}"] = DownloadData(
                date=self.date,
                data_type=self.data_type,
                detector=det,
                remote_host=self.remote_host,
            )

        # For the simulation we are using the MCL CR approximation
        if not simulate:
            # Download bgo cspec data for CR approximation
            bgos = ["b0", "b1"]
            for det in bgos:
                requires[f"data_{det}"] = DownloadData(
                    date=self.date,
                    data_type="cspec",
                    detector=det,
                    remote_host=self.remote_host,
                )
        else:

            requires["lat_files"] = DownloadLATData(
                date=self.date, remote_host=self.remote_host
            )

        return requires

    def output(self):
        return {
            "job_id": luigi.LocalTarget(os.path.join(self.job_dir, "job_id.txt")),
            "success": luigi.LocalTarget(os.path.join(self.job_dir, "success.txt")),
        }

    def remote_output(self):
        result_file_name = "fit_result_{}_{}_e{}.hdf5".format(
            f"{self.date:%y%m%d}",
            "-".join(self.detectors),
            "-".join(self.echans),
        )
        arviz_file_name = "fit_result_{}_{}_e{}.nc".format(
            f"{self.date:%y%m%d}",
            "-".join(self.detectors),
            "-".join(self.echans),
        )
        return {
            "chains_dir": RemoteTarget(
                os.path.join(self.job_dir_remote, "stan_chains"),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            "result_file": RemoteTarget(
                os.path.join(self.job_dir_remote, result_file_name),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            "arviz_file": RemoteTarget(
                os.path.join(self.job_dir_remote, arviz_file_name),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
            "success": RemoteTarget(
                os.path.join(self.job_dir_remote, "success.txt"),
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

        script_path = os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["script_dir"],
            "stan_fit_pipe.job",
        )

        if gbm_transient_search_config["balrog"]["run_destination"] == "local":
            balrog_offset = 0
        else:
            balrog_offset = 1

        if self.priority > 1:
            nice = 0 + balrog_offset
        else:
            nice = 100 + balrog_offset

        run_cmd = [
            "sbatch",
            "--parsable",
            f"--nice={nice}",
            "-D",
            f"{self.job_dir_remote}",
            f"{script_path}",
            f"{self.date:%y%m%d}",
            f"{self.requires()['config'].remote_output()['config'].path}",
            f"{self.job_dir_remote}",
        ]

        check_status_cmd = [
            "squeue",
            "-u",
            remote_hosts_config["hosts"][self.remote_host]["username"],
        ]

        logging.info(" ".join(run_cmd))

        if self.remote_output()["success"].exists():
            # Job has already successfully completed, if job_id file is missing
            # just create a dummy version
            if not self.output()["job_id"].exists():
                os.system(f"touch {self.output()['job_id'].path}")
            if not self.output()["success"].exists():
                os.system(f"touch {self.output()['success'].path}")

            return True

        else:
            run_fit = True

            # Check if job already has been created and is still running
            if self.output()["job_id"].exists():
                with self.output()["job_id"].open("r") as f:
                    job_id = f.readlines()[0]

                if isinstance(job_id, bytes):
                    job_id = job_id.decode()

                status = self.run_remote_command(check_status_cmd).decode()

                if str(job_id) in status:

                    run_fit = False

            if run_fit:
                job_output = self.run_remote_command(run_cmd)

                job_id = job_output.decode().strip().replace("\n", "")

                with self.output()["job_id"].open("w") as outfile:
                    outfile.write(job_id)

        # the time spent waiting so far
        time_spent = 0  # seconds
        wait_time = 5 * 60
        max_time = gbm_transient_search_config["phys_bkg"]["timeout"]

        # Sleep for 20 mins initially and add random sleep to avoid multiple bkg fits
        # querying at the same time
        time.sleep(20 * 60 + random.randint(30, 100))

        while True:

            if self.remote_output()["success"].exists():

                os.system(f"touch {self.output()['success'].path}")

                return True

            else:

                if time_spent >= max_time:

                    return False

                else:

                    status = self.run_remote_command(check_status_cmd)

                    status = status.decode()

                    if not str(job_id) in status:

                        self.output()["job_id"].remove()
                        raise Exception(f"The job {job_id} did fail, kill task.")

                    for line in status.split("\n"):

                        if str(job_id) in line:

                            logging.info(f"The squeue status: {line}")

                    time.sleep(wait_time)

                    time_spent += wait_time
