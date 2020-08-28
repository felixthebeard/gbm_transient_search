import os
import time
import json
import numpy as np
import luigi
import yaml
import arviz
from chainconsumer import ChainConsumer

from luigi.contrib.external_program import ExternalProgramTask

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
bkg_n_cores_multinest = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["n_cores"]
bkg_n_cores_stan = gbm_bkg_pipe_config["phys_bkg"]["stan"]["n_cores"]
bkg_path_to_python = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["path_to_python"]
bkg_timeout = gbm_bkg_pipe_config["phys_bkg"]["timeout"]

run_detectors = gbm_bkg_pipe_config["data"]["detectors"]
run_echans = gbm_bkg_pipe_config["data"]["echans"]


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
            base_dir,
            f"{self.date:%y%m%d}",
            self.data_type,
            "phys_bkg",
            f"det_{'_'.join(self.detectors)}",
            f"e{'_'.join(self.echans)}",
        )
        return luigi.LocalTarget(os.path.join(job_dir, "config_fit.yml"))

    def run(self):

        config_writer = BkgConfigWriter(self.date, self.data_type, self.echans, self.detectors)

        config_writer.write_config_file(self.output)


class RunPhysBkgModel(ExternalProgramTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()
    always_log_stderr = True

    # block twice the amount of cores in order to not rely on hyperthreadding
    resources = {"cpu": 2 * bkg_n_cores_multinest}

    worker_timeout = bkg_timeout

    def requires(self):
        return CreateBkgConfig(
            date=self.date,
            data_type=self.data_type,
            echans=self.echans,
            detectors=self.detectors,
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
        return {
            "result_file": luigi.LocalTarget(os.path.join(job_dir, "fit_result.hdf5")),
            "posteriour": luigi.LocalTarget(
                os.path.join(job_dir, "post_equal_weights.dat")
            ),
            "params_json": luigi.LocalTarget(os.path.join(job_dir, "params.json")),
        }

    def program_args(self):

        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/phys_bkg_model/fit_phys_bkg.py"

        command = []

        # Run with mpi in parallel
        if bkg_n_cores_multinest > 1:

            command.extend(["mpiexec", f"-n", f"{bkg_n_cores_multinest}"])

        command.extend(
            [
                bkg_path_to_python,
                fit_script_path,
                "--config_file",
                self.input().path,
                "--output_dir",
                os.path.dirname(self.output()["result_file"].path),
            ]
        )

        return command


class RunPhysBkgStanModel(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echans = luigi.ListParameter()
    detectors = luigi.ListParameter()

    # block twice the amount of cores in order to not rely on hyperthreadding
    resources = {"cpu": bkg_n_cores_stan}

    worker_timeout = bkg_timeout

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
            "arviz_file": luigi.LocalTarget(os.path.join(job_dir, "fit_result.nc")),
            # "posteriour": luigi.LocalTarget(
            #     os.path.join(job_dir, "post_equal_weights.dat")
            # ),
            # "params_json": luigi.LocalTarget(os.path.join(job_dir, "params.json")),
        }

    def run(self):
        os.environ["gbm_bkg_multiprocessing_n_cores"] = str(bkg_n_cores_stan)

        output_dir = os.path.dirname(self.output()["arviz_file"].path)

        model_generator = BackgroundModelGenerator()
        model_generator.from_config_file(self.input()["config"].path)

        stan_model_const = StanModelConstructor(model_generator=model_generator)
        stan_model_file = os.path.join(output_dir, "background_model.stan")
        stan_model_const.create_stan_file(stan_model_file)

        # Create Stan Model
        model = CmdStanModel(
            stan_file=stan_model_file,
            cpp_options={'STAN_THREADS': 'TRUE'}
        )

        # StanDataConstructor
        stan_data = StanDataConstructor(
            model_generator=model_generator,
            threads_per_chain=bkg_n_cores_stan
        )

        data_dict = stan_data.construct_data_dict()

        # Sample
        stan_fit = model.sample(
            data=data_dict,
            output_dir=os.path.join(output_dir, "stan_chains"),
            chains=1,
            seed=int(np.random.rand()*10000),
            parallel_chains=1,
            threads_per_chain=bkg_n_cores_stan,
            iter_warmup=300,
            iter_sampling=300,
            show_progress=True
        )

        # Export fine binned data
        config = model_generator.config
        # Create a copy of the response precalculation
        response_precalculation = model_generator._resp

        # Create a copy of the geomtry precalculation
        geometry_precalculation = model_generator._geom

        # Create copy of config dictionary
        config_export = config

        config_export["general"]["min_bin_width"] = 5

        # Create a new model generator instance of the same type
        model_generator_export = type(model_generator)()

        model_generator_export.from_config_dict(
            config=config_export,
            response=response_precalculation,
            geometry=geometry_precalculation,
        )
        # StanDataConstructor
        stan_data_export = StanDataConstructor(
            model_generator=model_generator_export,
            threads_per_chain=bkg_n_cores_stan
        )

        data_dict_export = stan_data_export.construct_data_dict()

        stan_model_file_export = os.path.join(output_dir, "background_model_export.stan")
        stan_model_const.create_stan_file(stan_model_file_export, total_only=True)

        # Create Stan Model
        model_export = CmdStanModel(
            stan_file=stan_model_file_export,
            cpp_options={'STAN_THREADS': 'TRUE'}
        )

        model_export.compile()

        export_quantities = model_export.generate_quantities(
            data=data_dict_export,
            mcmc_sample=stan_fit,
            gq_output_dir=os.path.join(output_dir, "stan_chains")
        )

        # Decrease CPU resource to 1
        self.decrease_running_resources({"cpu": bkg_n_cores_stan - 1})

        stan_data_export = StanDataExporter.from_generated_quantities(
            model_generator,
            export_quantities
        )

        stan_data_export.save_data(file_path=self.output()["result_file"].path)

        # Build arviz object
        arviz_result = arviz.from_cmdstanpy(
            posterior=stan_fit,
            posterior_predictive="ppc",
            observed_data={"counts": data_dict["counts"]},
            constant_data={
                "time_bins": data_dict["time_bins"],
                "dets": model_generator.data.detectors,
                "echans": model_generator.data.echans
            },
            predictions=stan_model_const.generated_quantities()
        )
        # Save this object
        arviz_result.to_netcdf(self.output()["arviz_file"].path)

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

        return luigi.LocalTarget(os.path.join(
                get_path_of_external_data_dir(),
                self.data_type,
                f"{self.date:%y%m%d}",
                datafile_name
            )
        )

    def run(self):

        if not file_existing_and_readable(self.output().path):
            download_data_file(
                f"{self.date:%y%m%d}",
                self.data_type,
                self.detector
            )

           
class DownloadPoshistData(luigi.Task):
    """
    Downloads a DataFile
    """
    date = luigi.DateParameter()

    resources = {"cpu": 1}

    def output(self):
        datafile_name = f"glg_poshist_all_{self.date:%y%m%d}_v00.fit"

        return luigi.LocalTarget(os.path.join(
                get_path_of_external_data_dir(),
                "poshist",
                datafile_name
            )
        )

    def run(self):

        if not file_existing_and_readable(self.output().path):
            download_data_file(
                f"{self.date:%y%m%d}",
                "poshist"
            )
