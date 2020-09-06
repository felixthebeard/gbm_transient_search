import os
import time
import json
import numpy as np
import luigi
import yaml
import arviz
from chainconsumer import ChainConsumer

from luigi.contrib.external_program import ExternalProgramTask
from luigi.contrib.ssh import RemoteContext, RemoteTarget

from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.utils.bkg_helper import BkgConfigWriter

from gbmbkgpy.io.export import PHAWriter, StanDataExporter
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator
from gbmbkgpy.utils.model_generator import BackgroundModelGenerator
from gbmbkgpy.utils.stan import StanDataConstructor, StanModelConstructor

from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.io.downloading import download_data_file
from gbmbkgpy.io.file_utils import file_existing_and_readable

from cmdstanpy import cmdstan_path, CmdStanModel

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")

data_dir_remote = gbm_bkg_pipe_config["remote"]["gbm_data"]
base_dir_remote = gbm_bkg_pipe_config["remote"]["base_dir"]

run_detectors = gbm_bkg_pipe_config["data"]["detectors"]
run_echans = gbm_bkg_pipe_config["data"]["echans"]

remote_host = gbm_bkg_pipe_config["remote"]["host"]
remote_username = gbm_bkg_pipe_config["remote"]["username"]

class GBMBackgroundModelFit(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")

    resources = {"cpu": 1}

    def requires(self):

        bkg_fit_tasks = {}

        for dets in run_detectors:

            for echans in run_echans:

                bkg_fit_tasks[
                    f"bkg_d{'_'.join(dets)}_e{'_'.join(echans)}"
                ] = RunPhysBkgStanModel(date=self.date, echans=echans, detectors=dets)

                bkg_fit_tasks[
                    f"result_plot_d{'_'.join(dets)}_e{'_'.join(echans)}"
                ] = BkgModelResultPlot(date=self.date, echans=echans, detectors=dets)

                # bkg_fit_tasks[
                #     f"corner_plot_d{'_'.join(dets)}_e{'_'.join(echans)}"
                # ] = BkgModelCornerPlot(date=self.date, echans=echans, detectors=dets)

        return bkg_fit_tasks

    def output(self):

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                "phys_bkg",
                "phys_bkg_combined.hdf5",
            )
        )

    def run(self):

        bkg_fit_results = []

        for dets in run_detectors:

            for echans in run_echans:

                bkg_fit_results.append(
                    self.input()[f"bkg_d{'_'.join(dets)}_e{'_'.join(echans)}"]["result_file"].path
                )

        # PHACombiner and save combined file
        pha_writer = PHAWriter.from_result_files(bkg_fit_results)

        pha_writer.save_combined_hdf5(self.output().path)


class CreateBkgConfig(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()

    resources = {"cpu": 1}

    def requires(self):
        return None

    def output(self):
        job_dir = os.path.join(
            base_dir_remote,
            f"{self.date:%y%m%d}",
            self.data_type,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )
        return RemoteTarget(
            os.path.join(job_dir, "config_fit.yml"),
            host=remote_host,
            username=remote_username,
            sshpass=True
        )

    def run(self):

        config_writer = BkgConfigWriter(self.date, self.data_type, self.echans, self.detectors)

        with self.output().open('w') as outfile:

            yaml.dump(
                config_writer._config,
                outfile,
                default_flow_style=False
            )

class RunPhysBkgStanModel(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()

    def requires(self):
        requires = {
            "config": CreateBkgConfig(
                date=self.date,
                data_type=self.data_type,
                echans=self.echans,
                detectors=self.detectors,
            ),
            "poshist_file": DownloadPoshistData(date=self.date),
        }

        for det in self.detectors:
            requires[f"data_{det}"] = DownloadData(
                date=self.date,
                data_type=self.data_type,
                detector=det
            )
        # Download bgo cspec data for CR approximation
        bgos = ["b0", "b1"]
        for det in bgos:
            requires[f"data_{det}"] = DownloadData(
                date=self.date,
                data_type="cspec",
                detector=det
            )

        return requires

    def output(self):
        job_dir = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )
        return {
            "result_file": luigi.LocalTarget(os.path.join(job_dir, "fit_result.hdf5")),
        }

    def run(self):

        script_path = os.path.join(
            gbm_bkg_pipe_config['remote']['script_dir'],
            "stan_fit_pipe.job"
        )

        with RemoteContext(
            host=remote_host,
            username=remote_username,
            sshpass=True
        ) as remote:
            output = remote.check_output[
                f"sbatch {script_path} --parsable {self.input()['config'].path} {self.date:%y%m%d}"
            ]

        print(output)





class BkgModelResultPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()

    resources = {"cpu": 1}

    def requires(self):
        return RunPhysBkgStanModel(
            date=self.date, echans=self.echans, detectors=self.detectors
        )

    def output(self):
        job_dir = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )

        plot_files = []

        for detector in self.detectors:
            for echan in self.echans:

                filename = f"plot_date_{self.date:%y%m%d}_det_{detector}_echan_{echan}__{''}.pdf"

                plot_files.append(luigi.LocalTarget(os.path.join(job_dir, filename)))
        return plot_files

    def run(self):

        config_plot_path = f"{os.path.dirname(os.path.abspath(__file__))}/phys_bkg_model/config_result_plot.yml"

        plot_generator = ResultPlotGenerator.from_result_file(
            config_file=config_plot_path,
            result_data_file=self.input()["result_file"].path,
        )

        plot_generator.create_plots(
            output_dir=os.path.dirname(self.output()[0].path), time_stamp=""
        )


class BkgModelCornerPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()

    resources = {"cpu": 1}

    def requires(self):
        return RunPhysBkgStanModel(
            date=self.date, echans=self.echans, detectors=self.detectors
        )

    def output(self):
        job_dir = os.path.join(
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )

        return luigi.LocalTarget(os.path.join(job_dir, "corner_plot.pdf",))

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

    resources = {"cpu": 1}

    def output(self):
        datafile_name = f"glg_{self.data_type}_{self.detector}_{self.date:%y%m%d}_v00.pha"

        return RemoteTarget(
            os.path.join(
                data_dir_remote,
                self.data_type,
                f"{self.date:%y%m%d}",
                datafile_name
            ),
            host=remote_host,
            username=remote_username,
            sshpass=True
        )

    def run(self):
        local_path = os.path.join(
            get_path_of_external_data_dir(),
            self.data_type,
            f"{self.date:%y%m%d}",
            f"glg_{self.data_type}_{self.detector}_{self.date:%y%m%d}_v00.pha"
        )

        if not os.path.exists(local_path):

            dl = BackgroundDownload(
                f"{self.date:%y%m%d}",
                self.data_type,
                self.detector,
                wait_time=float(
                    gbm_bkg_pipe_config["download"]["interval"]
                ),
                max_time=float(
                    gbm_bkg_pipe_config["download"]["max_time"]
                ),
            )
            file_readable = dl.run()

        else:
            file_readable = True


        if file_readable:

            self.output().put(local_path)

           
class DownloadPoshistData(luigi.Task):
    """
    Downloads a DataFile
    """
    date = luigi.DateParameter()

    resources = {"cpu": 1}

    def output(self):
        datafile_name = f"glg_poshist_all_{self.date:%y%m%d}_v00.fit"

        return RemoteTarget(
            os.path.join(
                data_dir_remote,
                "poshist",
                datafile_name
            ),
            host=remote_host,
            username=remote_username,
            sshpass=True
        )

    def run(self):
        local_path = os.path.join(
            get_path_of_external_data_dir(),
            "poshist",
            f"glg_poshist_all_{self.date:%y%m%d}_v00.fit"
        )

        if not os.path.exists(local_path):

            dl = BackgroundDownload(
                f"{self.date:%y%m%d}",
                "poshist",
                wait_time=float(
                    gbm_bkg_pipe_config["download"]["interval"]
                ),
                max_time=float(
                    gbm_bkg_pipe_config["download"]["max_time"]
                ),
            )
            file_readable = dl.run()

        else:
            file_readable = True


        if file_readable:

            self.output().put(local_path)
