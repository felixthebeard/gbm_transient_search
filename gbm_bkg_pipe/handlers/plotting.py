import datetime as dt
import json
import os
from datetime import datetime, timedelta

import luigi
import numpy as np
import yaml
from chainconsumer import ChainConsumer
from gbm_bkg_pipe.utils.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.handlers.background import BkgModelTask
from gbm_bkg_pipe.handlers.download import (
    DownloadPoshistData,
)
from gbm_bkg_pipe.handlers.localization import ProcessLocalizationResult
from gbm_bkg_pipe.handlers.transient_search import TransientSearch
from gbm_bkg_pipe.processors.bkg_result_reader import BkgArvizReader
from gbm_bkg_pipe.utils.env import get_bool_env_value, get_env_value
from gbm_bkg_pipe.utils.file_utils import (
    if_directory_not_existing_then_make,
)
from gbm_bkg_pipe.utils.plotting.arviz_plots import ArvizPlotter
from gbm_bkg_pipe.utils.plotting.plot_utils import (
    azimuthal_plot_sat_frame,
    create_corner_all_plot,
    create_corner_loc_plot,
    interactive_3D_plot,
    mollweide_plot,
    swift_gbm_plot,
)
from gbm_bkg_pipe.utils.plotting.trigger_plot import TriggerPlot
from gbmbkgpy.io.plotting.plot_result import ResultPlotGenerator

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")
_valid_gbm_detectors = np.array(gbm_bkg_pipe_config["data"]["detectors"]).flatten()
_valid_echans = np.array(gbm_bkg_pipe_config["data"]["echans"]).flatten()


class PlotTriggers(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()
    step = luigi.Parameter()
    loc_plots = luigi.Parameter(default=True)

    resources = {"cpu": 1}

    def requires(self):
        return TransientSearch(
            date=self.date,
            data_type=self.data_type,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        filename = f"plot_triggers_done.txt"

        return luigi.LocalTarget(
            os.path.join(base_dir, f"{self.date:%y%m%d}", self.data_type, filename)
        )

    def run(self):

        with self.input().open("r") as f:
            trigger_information = yaml.safe_load(f)

        plot_tasks = []
        for t_info in trigger_information["triggers"].values():

            plot_tasks.append(
                CreateAllPlots(
                    date=datetime.strptime(t_info["date"], "%y%m%d"),
                    data_type=trigger_information["data_type"],
                    trigger_name=t_info["trigger_name"],
                    remote_host=self.remote_host,
                    step=self.step,
                    loc_plots=self.loc_plots,
                )
            )
        yield plot_tasks

        os.system(f"touch {self.output().path}")


class CreateAllPlots(luigi.WrapperTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()
    loc_plots = luigi.Parameter(default=True)

    def requires(self):
        requires = {
            "lightcurves": CreateAllLightcurves(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            )
        }

        if self.loc_plots:
            requires.update(
                {
                    "location": CreateLocationPlot(
                        date=self.date,
                        data_type=self.data_type,
                        trigger_name=self.trigger_name,
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                    "corner": CreateCornerPlot(
                        date=self.date,
                        data_type=self.data_type,
                        trigger_name=self.trigger_name,
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                    "satellite": CreateSatellitePlot(
                        date=self.date,
                        data_type=self.data_type,
                        trigger_name=self.trigger_name,
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                    "molllocation": CreateMollLocationPlot(
                        date=self.date,
                        data_type=self.data_type,
                        trigger_name=self.trigger_name,
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                    "spectrum": CreateSpectrumPlot(
                        date=self.date,
                        data_type=self.data_type,
                        trigger_name=self.trigger_name,
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                }
            )


class CreateAllLightcurves(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return dict(
            transient_search=TransientSearch(
                date=self.date,
                data_type=self.data_type,
                remote_host=self.remote_host,
                step=self.step,
            ),
        )

    def output(self):
        lightcurves = dict()

        for det in _valid_gbm_detectors:
            lightcurves[det] = luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    self.data_type,
                    self.step,
                    "trigger",
                    self.trigger_name,
                    "plots",
                    "lightcurves",
                    f"{self.trigger_name}_lightcurve_detector_{det}_plot.png",
                )
            )
        return lightcurves

    def run(self):
        plotter = TriggerPlot.from_hdf5(
            trigger_yaml=self.input()["transient_search"].path,
            data_path=os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                "plot_data.hdf5",
            ),
        )

        plotter.create_trigger_plots(
            trigger_name=self.trigger_name,
            outdir=os.path.join(
                base_dir, f"{self.date:%y%m%d}", self.data_type, self.step
            ),
        )


class CreateLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "plots",
                f"{self.trigger_name}_location_plot_{self.data_type}.png",
            )
        )

    def run(self):
        with self.input()["result_file"].open() as f:
            result = yaml.safe_load(f)

        create_corner_loc_plot(
            post_equal_weights_file=self.input()["post_equal_weights"].path,
            model=result["fit_result"]["model"],
            save_path=self.output().path,
        )


class CreateCornerPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "plots",
                f"{self.trigger_name}_allcorner_plot_{self.data_type}.png",
            )
        )

    def run(self):
        with self.input()["result_file"].open() as f:
            result = yaml.safe_load(f)

        create_corner_all_plot(
            post_equal_weights_file=self.input()["post_equal_weights"].path,
            model=result["fit_result"]["model"],
            save_path=self.output().path,
        )


class CreateMollLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            poshist_file=DownloadPoshistData(
                date=self.date, remote_host=self.remote_host
            ),
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "plots",
                f"{self.trigger_name}_molllocation_plot_{self.data_type}.png",
            )
        )

    def run(self):
        with self.input()["fit_result"]["result_file"].open() as f:
            result = yaml.safe_load(f)

        mollweide_plot(
            trigger_name=self.trigger_name,
            poshist_file=self.input()["poshist_file"]["local_file"].path,
            post_equal_weights_file=self.input()["fit_result"][
                "post_equal_weights"
            ].path,
            trigger_time=result["trigger"]["trigger_time"],
            used_dets=result["trigger"]["use_dets"],
            model=result["fit_result"]["model"],
            ra=result["fit_result"]["ra"],
            dec=result["fit_result"]["dec"],
            swift=None,
            save_path=self.output().path,
        )


class CreateSatellitePlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            poshist_file=DownloadPoshistData(
                date=self.date, remote_host=self.remote_host
            ),
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "plots",
                f"{self.trigger_name}_satellite_plot_{self.data_type}.png",
            )
        )

    def run(self):
        with self.input()["fit_result"]["result_file"].open() as f:
            result = yaml.safe_load(f)

        azimuthal_plot_sat_frame(
            trigger_name=self.trigger_name,
            poshist_file=self.input()["poshist_file"]["local_file"].path,
            trigger_time=result["trigger"]["trigger_time"],
            ra=result["fit_result"]["ra"],
            dec=result["fit_result"]["dec"],
            save_path=self.output().path,
        )


class CreateSpectrumPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "plots",
                f"{self.trigger_name}_spectrum_plot_{self.data_type}.png",
            )
        )

    def run(self):
        # The spectrum plot is created in the balrog fit Task, this task will check if the creation was successful
        pass


class Create3DLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            poshist_file=DownloadPoshistData(
                date=self.date, remote_host=self.remote_host
            ),
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "plots",
                f"{self.trigger_name}_3dlocation_plot_{self.data_type}.html",
            )
        )

    def run(self):
        with self.input()["fit_result"]["result_file"].open() as f:
            result = yaml.safe_load(f)

        interactive_3D_plot(
            poshist_file=self.input()["poshist_file"]["local_file"].path,
            post_equal_weights_file=self.input()["fit_result"][
                "post_equal_weights"
            ].path,
            trigger_time=result["trigger"]["trigger_time"],
            used_dets=result["trigger"]["use_dets"],
            model=result["fit_result"]["model"],
            save_path=self.output().path,
        )


class CreateBkgModelPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    remote_host = luigi.Parameter()
    detector = luigi.Parameter()
    echan = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return BkgModelPlots(
            date=self.date,
            data_type=self.data_type,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "phys_bkg",
                "plots",
                f"bkg_model_{self.date:%y%m%d}_det_{self.detector}_echan_{self.echan}.png",
            )
        )

    def run(self):
        # The lightcurve is created plot triggers task,
        # This task will check if the creation was successful
        pass


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


class BkgModelPerformancePlot(BkgModelTask):

    resources = {"cpu": 1}

    def requires(self):
        return CopyResults(
            date=self.date,
            echans=self.echans,
            detectors=self.detectors,
            remote_host=self.remote_host,
            step=self.step,
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
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_all_posterior.png")
            ),
            "pairs_all": luigi.LocalTarget(
                os.path.join(self.job_dir, f"{self.date:%y%m%d}_all_pairs.png")
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
        arviz_plotter.plot_traces(
            var_names=["norm_fixed"],
            plot_path=self.output()["traces_global"].path,
            dpi=80,
        )
        arviz_plotter.plot_pairs(
            var_names=["norm_fixed"],
            plot_path=self.output()["pairs_global"].path,
            dpi=30,
        )

        # Plot contiuum sources
        arviz_plotter.plot_posterior(
            var_names=["norm_cont"], plot_path=self.output()["posterior_cont"].path
        )
        arviz_plotter.plot_traces(
            var_names=["norm_cont"], plot_path=self.output()["traces_cont"].path, dpi=80
        )
        arviz_plotter.plot_pairs(
            var_names=["norm_cont"], plot_path=self.output()["pairs_cont"].path, dpi=30
        )

        # Joint plots
        arviz_plotter.plot_posterior(
            var_names=["norm_fixed", "norm_cont"],
            plot_path=self.output()["posterior_all"].path,
        )
        arviz_plotter.plot_traces(
            var_names=["norm_fixed", "norm_cont"],
            plot_path=self.output()["traces_all"].path,
            dpi=80,
        )
        arviz_plotter.plot_pairs(
            var_names=["norm_fixed", "norm_cont"],
            plot_path=self.output()["pairs_all"].path,
            dpi=30,
        )


class BkgModelResultPlot(BkgModelTask):
    resources = {"cpu": 1}

    def requires(self):
        return CopyResults(
            date=self.date,
            echans=self.echans,
            detectors=self.detectors,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        plot_files = {
            "summary": luigi.LocalTarget(
                os.path.join(
                    self.job_dir,
                    f"bkg_model_{self.date:%y%m%d}_fit_summary.yaml",
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


class BkgModelCornerPlot(BkgModelTask):
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
        return luigi.LocalTarget(
            os.path.join(
                self.job_dir,
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
