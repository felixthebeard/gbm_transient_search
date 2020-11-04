import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
import datetime as dt

import luigi
import numpy as np
import yaml
from chainconsumer import ChainConsumer
from gbmbkgpy.io.export import PHAWriter
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.utils.select_pointsources import build_swift_pointsource_database
from luigi.contrib.ssh import RemoteContext, RemoteTarget
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.utils.bkg_helper import BkgConfigWriter
from gbm_bkg_pipe.utils.download_file import BackgroundDataDownload

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")

data_dir = os.environ.get("GBMDATA")

run_detectors = gbm_bkg_pipe_config["data"]["detectors"]
run_echans = gbm_bkg_pipe_config["data"]["echans"]

remote_hosts_config = gbm_bkg_pipe_config["remote_hosts_config"]


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


class BkgModelPlots(luigi.WrapperTask):
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

        bkg_fit_tasks = {}

        for dets in run_detectors:

            for echans in run_echans:
                bkg_fit_tasks[
                    f"result_plot_d{'_'.join(dets)}_e{'_'.join(echans)}"
                ] = BkgModelResultPlot(
                    date=self.date,
                    echans=echans,
                    detectors=dets,
                    remote_host=self.remote_host,
                    step=self.step,
                )

                # bkg_fit_tasks[
                #     f"corner_plot_d{'_'.join(dets)}_e{'_'.join(echans)}"
                # ] = BkgModelCornerPlot(date=self.date, echans=echans, detectors=dets)

        return bkg_fit_tasks


class CreateBkgConfig(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
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
        requires = {
            "pointsource_db": UpdatePointsourceDB(
                date=self.date, remote_host=self.remote_host
            )
        }
        if self.step == "final":
            from gbm_bkg_pipe.trigger_search import TriggerSearch

            requires["trigger_search"] = TriggerSearch(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step="base",
            )

        return requires

    def output(self):
        job_dir_remote = os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["base_dir"],
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )
        return RemoteTarget(
            os.path.join(job_dir_remote, "config_fit.yml"),
            host=self.remote_host,
            username=remote_hosts_config["hosts"][self.remote_host]["username"],
            sshpass=True,
        )

    def run(self):

        config_writer = BkgConfigWriter(
            self.date, self.data_type, self.echans, self.detectors
        )

        config_writer.build_config()

        if self.step == "final":
            config_writer.mask_triggers(self.input()["trigger_search"].path)

        with self.output().open("w") as outfile:

            yaml.dump(config_writer._config, outfile, default_flow_style=False)


class CopyResults(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
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
        job_dir = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )
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
            "result_file": luigi.LocalTarget(os.path.join(job_dir, result_file_name)),
            "arviz_file": luigi.LocalTarget(os.path.join(job_dir, arviz_file_name)),
            "config_file": luigi.LocalTarget(os.path.join(job_dir, "config_fit.yml")),
        }

    def run(self):
        # Copy result file to local folder
        self.input()["bkg_fit"]["result_file"].get(self.output()["result_file"].path)

        # Copy arviz file to local folder
        self.input()["bkg_fit"]["arviz_file"].get(self.output()["arviz_file"].path)

        # Copy config file to local folder
        self.input()["config_file"].get(self.output()["config_file"].path)


class RunPhysBkgModel(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    result_timeout = 2 * 60 * 60

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
        # Download bgo cspec data for CR approximation
        bgos = ["b0", "b1"]
        for det in bgos:
            requires[f"data_{det}"] = DownloadData(
                date=self.date,
                data_type="cspec",
                detector=det,
                remote_host=self.remote_host,
            )

        return requires

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
            "job_id": luigi.LocalTarget(os.path.join(self.job_dir, "job_id.txt")),
            "chains_dir": RemoteTarget(
                os.path.join(self.job_dir_remote, "stan_chains"),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                sshpass=True,
            ),
            "result_file": RemoteTarget(
                os.path.join(self.job_dir_remote, result_file_name),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                sshpass=True,
            ),
            "arviz_file": RemoteTarget(
                os.path.join(self.job_dir_remote, arviz_file_name),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                sshpass=True,
            ),
            "success": RemoteTarget(
                os.path.join(self.job_dir_remote, "success.txt"),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                sshpass=True,
            ),
        }

    def run(self):

        script_path = os.path.join(
            remote_hosts_config["hosts"][self.remote_host]["script_dir"],
            "stan_fit_pipe.job",
        )

        remote = RemoteContext(
            host=self.remote_host,
            username=remote_hosts_config["hosts"][self.remote_host]["username"],
            sshpass=True,
        )

        if gbm_bkg_pipe_config["balrog"]["run_destination"] == "local":
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
            f"{self.input()['config'].path}",
            f"{self.job_dir_remote}",
        ]

        check_status_cmd = [
            "squeue",
            "-u",
            remote_hosts_config["hosts"][self.remote_host]["username"],
        ]

        logging.info(" ".join(run_cmd))

        if self.output()["success"].exists():
            # Job has already successfully completed, if job_id file is missing
            # just create a dummy version
            if not self.output()["job_id"].exists():
                with self.output()["job_id"].open("w") as outfile:
                    outfile.write("dummy")

            return True

        else:
            run_fit = True

            # Check if job already has been created and is still running
            if self.output()["job_id"].exists():
                with self.output()["job_id"].open("r") as f:
                    job_id = f.readlines()[0]

                if isinstance(job_id, bytes):
                    job_id = job_id.decode()

                status = remote.check_output(check_status_cmd).decode()

                if str(job_id) in status:

                    run_fit = False

            if run_fit:
                job_output = remote.check_output(run_cmd)

                job_id = job_output.decode().strip().replace("\n", "")

                with self.output()["job_id"].open("w") as outfile:
                    outfile.write(job_id)

        # the time spent waiting so far
        time_spent = 0  # seconds
        wait_time = 20
        max_time = 2 * 60 * 60

        # Sleep for 10s initially
        time.sleep(10)

        while True:

            if self.output()["success"].exists():

                return True

            else:

                if time_spent >= max_time:

                    return False

                else:

                    status = remote.check_output(check_status_cmd)

                    status = status.decode()

                    if (
                        not str(job_id) in status
                        and not self.output()["success"].exists()
                    ):

                        print(f"The job {job_id} did fail, kill task.")
                        return False

                    for line in status.split("\n"):

                        if str(job_id) in line:

                            logging.info(f"The squeue status: {line}")

                    time.sleep(wait_time)

                    time_spent += wait_time


class BkgModelResultPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    resources = {"cpu": 1}

    def requires(self):
        return CopyResults(
            date=self.date,
            echans=self.echans,
            detectors=self.detectors,
            remote_host=self.remote_host,
            step=self.step,
        )

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
            "plots",
        )

    def output(self):
        plot_files = {}

        for detector in self.detectors:
            for echan in self.echans:

                filename = (
                    f"bkg_model_{self.date:%y%m%d}_det_{detector}_echan_{echan}.png"
                )

                plot_files[f"{detector}_{echan}"] = luigi.LocalTarget(
                    os.path.join(self.job_dir, filename)
                )

        return plot_files

    def run(self):
        self.output()[f"{self.detectors[0]}_{self.echans[0]}"].makedirs()

        config_plot_path = f"{os.path.dirname(os.path.abspath(__file__))}/phys_bkg_model/config_result_plot.yml"

        plot_generator = ResultPlotGenerator.from_result_file(
            config_file=config_plot_path,
            result_data_file=self.input()["result_file"].path,
        )

        plot_generator.create_plots(
            output_dir=self.job_dir,
            plot_name="bkg_model_",
            time_stamp="",
        )


class BkgModelCornerPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
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
        return CopyResults(
            date=self.date,
            echans=self.echans,
            detectors=self.detectors,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        job_dir = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            self.step,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )

        return luigi.LocalTarget(
            os.path.join(
                job_dir,
                "corner_plot.pdf",
            )
        )

    def run(self):

        with self.input()["params_json"].open() as f:

            param_names = json.load(f)

        safe_param_names = [name.replace("_", " ") for name in param_names]

        if len(safe_param_names) > 1:

            chain = np.loadtxt(self.input()["posteriour"].path, ndmin=2)

            c2 = ChainConsumer()

            c2.add_chain(chain[:, :-1], parameters=safe_param_names).configure(
                plot_hists=False,
                contour_labels="sigma",
                colors="#cd5c5c",
                flip=False,
                max_ticks=3,
            )

            c2.plotter.plot(filename=self.output().path)

        else:
            print(
                "Your model only has one paramter, we cannot make a cornerplot for this."
            )

        # with self.output().temporary_path() as self.temp_output_path:
        #     run_some_external_command(output_path=self.temp_output_path)


class DownloadData(luigi.Task):
    """
    Downloads a DataFile
    """

    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    detector = luigi.ListParameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def output(self):
        datafile_name = (
            f"glg_{self.data_type}_{self.detector}_{self.date:%y%m%d}_v00.pha"
        )
        return dict(
            local_file=luigi.LocalTarget(
                os.path.join(
                    data_dir, self.data_type, f"{self.date:%y%m%d}", datafile_name
                )
            ),
            remote_file=RemoteTarget(
                os.path.join(
                    remote_hosts_config["hosts"][self.remote_host]["data_dir"],
                    self.data_type,
                    f"{self.date:%y%m%d}",
                    datafile_name,
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                sshpass=True,
            ),
        )

    def run(self):

        if not self.output()["local_file"].exists():

            dl = BackgroundDataDownload(
                f"{self.date:%y%m%d}",
                self.data_type,
                self.detector,
                wait_time=float(gbm_bkg_pipe_config["download"]["interval"]),
                max_time=float(gbm_bkg_pipe_config["download"]["max_time"]),
            )
            file_readable = dl.run()

        else:
            file_readable = True

        if file_readable:

            self.output()["remote_file"].put(self.output()["local_file"].path)


class DownloadPoshistData(luigi.Task):
    """
    Downloads a DataFile
    """

    date = luigi.DateParameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def output(self):
        datafile_name = f"glg_poshist_all_{self.date:%y%m%d}_v00.fit"

        return dict(
            local_file=luigi.LocalTarget(
                os.path.join(data_dir, "poshist", datafile_name)
            ),
            remote_file=RemoteTarget(
                os.path.join(
                    remote_hosts_config["hosts"][self.remote_host]["data_dir"],
                    "poshist",
                    datafile_name,
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                sshpass=True,
            ),
        )

    def run(self):
        local_path = os.path.join(
            get_path_of_external_data_dir(),
            "poshist",
            f"glg_poshist_all_{self.date:%y%m%d}_v00.fit",
        )

        if not self.output()["local_file"].exists():

            dl = BackgroundDataDownload(
                f"{self.date:%y%m%d}",
                "poshist",
                "all",
                wait_time=float(gbm_bkg_pipe_config["download"]["interval"]),
                max_time=float(gbm_bkg_pipe_config["download"]["max_time"]),
            )
            file_readable = dl.run()

        else:
            file_readable = True

        if file_readable:

            self.output()["remote_file"].put(self.output()["local_file"].path)


class UpdatePointsourceDB(luigi.Task):
    """
    Downloads a DataFile
    """

    date = luigi.DateParameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": 1}

    @property
    def priority(self):
        yesterday = dt.date.today() - timedelta(days=1)
        if self.date >= yesterday:
            return 10
        else:
            return 1

    def output(self):
        return dict(
            local_ps_db_file=luigi.LocalTarget(
                os.path.join(
                    data_dir, "background_point_sources", "pointsources_swift.h5"
                )
            ),
            db_updated=luigi.LocalTarget(
                os.path.join(base_dir, f"{self.date:%y%m%d}", "ps_db_updated.txt")
            ),
            # remote_ps_db_file=RemoteTarget(
            #     os.path.join(
            #         remote_hosts_config["hosts"][self.remote_host]["data_dir"],
            #         "background_point_sources",
            #         "pointsources_swift.h5",
            #     ),
            #     host=self.remote_host,
            #     username=remote_hosts_config["hosts"][self.remote_host]["username"],
            #     sshpass=True,
            # ),
        )

    def run(self):

        update_running = os.path.join(
            os.path.dirname(self.output()["local_ps_db_file"].path), "updating.txt"
        )

        if self.output()["local_ps_db_file"].exists():
            local_db_creation = datetime.fromtimestamp(
                os.path.getmtime(self.output()["local_ps_db_file"].path)
            )
        else:
            # use old dummy date in this case
            local_db_creation = datetime(year=2000, month=1, day=1)

        # the time spent waiting so far
        time_spent = 0  # seconds
        wait_time = 20
        max_time = 1 * 60 * 60

        # Check if local db is older than one day
        if (datetime.now() - local_db_creation) > timedelta(days=1):

            # Check if there is already an update running
            # this could be from running the pipeline on a different day of data
            if os.path.exists(update_running):

                while True:

                    if not os.path.exists(update_running):

                        if self.output()["local_ps_db_file"].exists():
                            local_db_creation = datetime.fromtimestamp(
                                os.path.getmtime(self.output()["local_ps_db_file"].path)
                            )

                        if (datetime.now() - local_db_creation) < timedelta(days=1):

                            break

                    else:

                        if time_spent >= max_time:

                            break

                        else:

                            time.sleep(wait_time)

            if self.output()["local_ps_db_file"].exists():
                # Check again the creation time in case we exited from the loop
                local_db_creation = datetime.fromtimestamp(
                    os.path.getmtime(self.output()["local_ps_db_file"].path)
                )

            # If the db file is older then start building it
            if (datetime.now() - local_db_creation) > timedelta(days=1):

                os.system(f"touch {update_running}")

                with tempfile.TemporaryDirectory() as tmpdirname:
                    build_swift_pointsource_database(
                        tmpdirname, multiprocessing=True, force=True
                    )

                # delete the update running file once we are done
                os.remove(update_running)

        # NOTE: This is not necessary as we write the PS to the config file
        # Now copy the new db file over to the remote machine
        # self.output()["remote_ps_db_file"].put(
        #     self.output()["local_ps_db_file"].path
        # )

        self.output()["db_updated"].makedirs()
        os.system(f"touch {self.output()['db_updated'].path}")
