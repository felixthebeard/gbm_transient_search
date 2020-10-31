import os
import numpy as np
import luigi
import yaml
from datetime import datetime

from gbm_bkg_pipe.bkg_fit_remote_handler import DownloadPoshistData, BkgModelPlots
from gbm_bkg_pipe.balrog_handler import ProcessLocalizationResult
from gbm_bkg_pipe.trigger_search import TriggerSearch
from gbm_bkg_pipe.utils.env import get_env_value
from gbm_bkg_pipe.utils.plotting import TriggerPlot

from gbm_bkg_pipe.utils.plot_utils import (
    azimuthal_plot_sat_frame,
    create_corner_all_plot,
    create_corner_loc_plot,
    interactive_3D_plot,
    mollweide_plot,
    swift_gbm_plot,
)
from gbm_bkg_pipe.utils.file_utils import if_dir_containing_file_not_existing_then_make
from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")
_valid_gbm_detectors = np.array(gbm_bkg_pipe_config["data"]["detectors"]).flatten()
_valid_echans = np.array(gbm_bkg_pipe_config["data"]["echans"]).flatten()


class PlotTriggers(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()

    resources = {"cpu": 1}

    def requires(self):
        return TriggerSearch(
            date=self.date, data_type=self.data_type, remote_host=self.remote_host
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
                )
            )
        yield plot_tasks

        os.system(f"touch {self.output().path}")


class CreateAllPlots(luigi.WrapperTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()

    def requires(self):
        return {
            "lightcurves": CreateAllLightcurves(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
            "location": CreateLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
            "corner": CreateCornerPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
            "satellite": CreateSatellitePlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
            "molllocation": CreateMollLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
            "spectrum": CreateSpectrumPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
        }


class CreateAllLightcurves(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()

    def requires(self):
        return dict(
            loc_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
            ),
            trigger_search=TriggerSearch(
                date=self.date, data_type=self.data_type, remote_host=self.remote_host
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
            trigger_yaml=self.input()["trigger_search"].path,
            data_path=os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                "trigger",
                "plot_data.hdf5",
            ),
        )

        plotter.create_trigger_plots(
            trigger_name=self.trigger_name,
            outdir=os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
            ),
        )


class CreateLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
            remote_host=self.remote_host,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
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

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
            remote_host=self.remote_host,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
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

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
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

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
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

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
            remote_host=self.remote_host,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
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

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
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

    def requires(self):
        return BkgModelPlots(
            date=self.date,
            data_type=self.data_type,
            remote_host=self.remote_host,
        )

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                "phys_bkg",
                "plots",
                f"bkg_model_{self.date:%y%m%d}_det_{self.detector}_echan_{self.echan}.pdf",
            )
        )

    def run(self):
        # The lightcurve is created plot triggers task,
        # This task will check if the creation was successful
        pass
