import datetime as dt
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta

import h5py
import luigi
import numpy as np
import yaml
from chainconsumer import ChainConsumer
from gbmbkgpy.io.export import PHAWriter
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.utils.select_pointsources import build_swift_pointsource_database
from luigi.contrib.ssh import RemoteContext, RemoteTarget

from gbm_bkg_pipe.utils.file_utils import if_directory_not_existing_then_make
from gbm_bkg_pipe.utils.arviz_plots import ArvizPlotter
from gbm_bkg_pipe.utils.bkg_result_reader import BkgArvizReader
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.utils.bkg_helper import BkgConfigWriter
from gbm_bkg_pipe.utils.download_file import (
    BackgroundDataDownload,
    BackgroundLATDownload,
)
from gbm_bkg_pipe.utils.env import get_bool_env_value, get_env_value

base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
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

        bkg_plot_tasks = {}

        for dets in run_detectors:

            for echans in run_echans:
                bkg_plot_tasks[
                    f"result_plot_d{'_'.join(dets)}_e{'_'.join(echans)}"
                ] = BkgModelResultPlot(
                    date=self.date,
                    echans=echans,
                    detectors=dets,
                    remote_host=self.remote_host,
                    step=self.step,
                )

                bkg_plot_tasks[
                    f"performance_plots_d{'_'.join(dets)}_e{'_'.join(echans)}"
                ] = BkgModelPerformancePlot(
                    date=self.date,
                    echans=echans,
                    detectors=dets,
                    remote_host=self.remote_host,
                    step=self.step,
                )

                # bkg_fit_tasks[
                #     f"corner_plot_d{'_'.join(dets)}_e{'_'.join(echans)}"
                # ] = BkgModelCornerPlot(date=self.date, echans=echans, detectors=dets)

        return bkg_plot_tasks


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
            self.date, self.data_type, self.echans, self.detectors, step=self.step
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
            "best_fit_file": luigi.LocalTarget(
                os.path.join(job_dir, "best_fit_params.yml")
            ),
        }

    def run(self):
        # Copy result file to local folder
        self.input()["bkg_fit"]["result_file"].get(self.output()["result_file"].path)

        # Copy arviz file to local folder
        self.input()["bkg_fit"]["arviz_file"].get(self.output()["arviz_file"].path)

        # Copy config file to local folder
        self.input()["config_file"].get(self.output()["config_file"].path)

        with h5py.File(self.output()["result_file"].path, "r") as f:
            best_fit_values = f.attrs["best_fit_values"].tolist()
            param_names = f.attrs["param_names"].tolist()

        with self.output()["best_fit_file"].open("w") as f:
            yaml.dump(
                dict(zip(param_names, best_fit_values)), f, default_flow_style=False
            )


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
        return 4

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


class BkgModelPerformancePlot(luigi.Task):
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
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
            "plots",
        )

    def output(self):
        plot_files = {
            "posterior_global": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_global_posterior.png")
            ),
            "posterior_cont": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_cont_posterior.png")
            ),
            "pairs_global": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_global_pairs.png")
            ),
            "pairs_cont": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_cont_pairs.png")
            ),
            "traces_global": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_global_traces.png")
            ),
            "traces_cont": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_cont_traces.png")
            ),
            "posterior_all": luigi.LocalTarget(
                os.path.join(
                    self.job_dir, f"{self.date:%y%m%d}_all_global_posterior.png"
                )
            ),
            "pairs_all": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_all_global_pairs.png")
            ),
            "traces_all": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_all_traces.png")
            ),
        }

        return plot_files

    def run(self):
        if_directory_not_existing_then_make(self.job_dir)

        arviz_plotter = ArvizPlotter(
            date=f"{self.date:%y%m%d}", path_to_netcdf=self.input()["arviz_file"].path
        )

        # Plot global sources
        arviz_plotter.plot_posterior(
            var_names=["norm_fixed"], plot_path=self.output()["posterior_global"].path
        )
        arviz_plotter.plot_pairs(
            var_names=["norm_fixed"], plot_path=self.output()["pairs_global"].path
        )
        arviz_plotter.plot_traces(
            var_names=["norm_fixed"], plot_path=self.output()["traces_global"].path
        )

        # Plot contiuum sources
        arviz_plotter.plot_posterior(
            var_names=["norm_cont"], plot_path=self.output()["posterior_conr"].path
        )
        arviz_plotter.plot_traces(
            var_names=["norm_cont"], plot_path=self.output()["traces_cont"].path
        )
        arviz_plotter.plot_pairs(
            var_names=["norm_cont"], plot_path=self.output()["pairs_cont"].path
        )

        # Joint plots
        arviz_plotter.plot_posterior(
            var_names=["norm_fixed", "norm_cont"],
            plot_path=self.output()["posterior_global"].path,
        )
        arviz_plotter.plot_pairs(
            var_names=["norm_fixed", "norm_cont"],
            plot_path=self.output()["pairs_global"].path,
        )
        arviz_plotter.plot_traces(
            var_names=["norm_fixed", "norm_cont"],
            plot_path=self.output()["traces_global"].path,
        )


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
        )

    def output(self):
        plot_files = {
            "summary": luigi.LocalTarget(
                os.path.join(
                    self.job_dir, f"bkg_model_{self.date:%y%m%d}_fit_summary.yaml"
                )
            )
        }

        for detector in self.detectors:
            for echan in self.echans:

                filename = (
                    f"bkg_model_{self.date:%y%m%d}_det_{detector}_echan_{echan}.png"
                )

                plot_files[f"{detector}_{echan}"] = luigi.LocalTarget(
                    os.path.join(self.job_dir, "plots", filename)
                )

        return plot_files

    def run(self):
        self.output()[f"{self.detectors[0]}_{self.echans[0]}"].makedirs()

        config_plot_path = f"{os.path.dirname(os.path.abspath(__file__))}/data/bkg_model/config_result_plot.yml"

        arviz_reader = BkgArvizReader(self.input()["arviz_file"].path)

        plot_generator = ResultPlotGenerator(
            config_file=config_plot_path,
            result_dict=arviz_reader.result_dict,
        )
        arviz_reader.hide_point_sources(norm_threshold=0.001, max_ps=6)
        plot_generator._hide_sources = arviz_reader.source_to_hide

        plot_generator.create_plots(
            output_dir=os.path.join(self.job_dir, "plots"),
            plot_name="bkg_model_",
            time_stamp="",
        )

        arviz_reader.save_summary(self.output()["summary"].path)


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

    @property
    def local_data_dir(self):
        if simulate:
            return os.path.join(
                data_dir, "simulation", self.data_type, f"{self.date:%y%m%d}"
            )
        else:
            return os.path.join(data_dir, self.data_type, f"{self.date:%y%m%d}")

    def output(self):
        datafile_name = (
            f"glg_{self.data_type}_{self.detector}_{self.date:%y%m%d}_v00.pha"
        )
        return dict(
            local_file=luigi.LocalTarget(
                os.path.join(self.local_data_dir, datafile_name)
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

            if simulate:
                raise Exception(
                    "Running in simulation mode, but simulation data file not existing"
                )

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


class DownloadLATData(luigi.Task):
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
        return luigi.LocalTarget(
            os.path.join(base_dir, f"{self.date:%y%m%d}" "download_lat_file.done")
        )

    def run(self):
        dl = BackgroundLATDownload(
            f"{self.date:%y%m%d}",
            wait_time=float(gbm_bkg_pipe_config["download"]["interval"]),
            max_time=float(gbm_bkg_pipe_config["download"]["max_time"]),
        )

        files_readable, file_names = dl.run()

        if files_readable:

            for file_name in file_names:

                local_file = luigi.LocalTarget(os.path.join(data_dir, "lat", file_name))
                remote_file = RemoteTarget(
                    os.path.join(
                        remote_hosts_config["hosts"][self.remote_host]["data_dir"],
                        "lat",
                        file_name,
                    ),
                    host=self.remote_host,
                    username=remote_hosts_config["hosts"][self.remote_host]["username"],
                    sshpass=True,
                )

                remote_file.put(local_file.path)

            os.system(f"touch {self.output().path}")

        else:
            return False


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

        if local_db_creation.date() > self.date:
            # DB is newer than the date to be processed (backprocessing)
            pass

        # Check if local db is older than one day
        elif (datetime.now() - local_db_creation) > timedelta(days=1):

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

                try:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        build_swift_pointsource_database(
                            tmpdirname, multiprocessing=True, force=True
                        )
                except Exception as e:
                    # In case this throws an exception remote the update running file
                    # to permit the task to be re-run
                    os.remove(update_running)
                    raise e

                # delete the update running file once we are done
                os.remove(update_running)

        # NOTE: This is not necessary as we write the PS to the config file
        # Now copy the new db file over to the remote machine
        # self.output()["remote_ps_db_file"].put(
        #     self.output()["local_ps_db_file"].path
        # )

        self.output()["db_updated"].makedirs()
        os.system(f"touch {self.output()['db_updated'].path}")
