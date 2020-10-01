import os

import luigi
import yaml
from datetime import datetime

from gbm_bkg_pipe.balrog_handler import ProcessLocalizationResult
from gbm_bkg_pipe.trigger_search import TriggerSearch
from gbm_bkg_pipe.utils.env import get_env_value
from gbm_bkg_pipe.utils.plot_utils import (
    azimuthal_plot_sat_frame,
    create_corner_all_plot,
    create_corner_loc_plot,
    interactive_3D_plot,
    mollweide_plot,
    swift_gbm_plot,
)

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")


class PlotTriggers(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")

    resources = {"cpu": 1}

    def requires(self):
        return TriggerSearch(date=self.date, data_type=self.data_type)

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
                )
            )
        yield plot_tasks

        os.system(f"touch {self.output().path}")


class CreateAllPlots(luigi.WrapperTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()

    def requires(self):
        return {
            "location": CreateLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
            ),
            "corner": CreateCornerPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
            ),
            # "satellite": CreateSatellitePlot(
            #     grb_name=self.grb_name,
            #     report_type=self.report_type,
            #     version=self.version,
            #     phys_bkg=self.phys_bkg
            # ),
            # "molllocation": CreateMollLocationPlot(
            #     grb_name=self.grb_name,
            #     report_type=self.report_type,
            #     version=self.version,
            #     phys_bkg=self.phys_bkg
            # ),
            "spectrum": CreateSpectrumPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
            ),
        }


class CreateLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
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

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
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

    def requires(self):
        return dict(
            fit_result=ProcessLocalizationResult(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
            ),
            poshist_file=DownloadPoshistData(date=self.date),
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
            poshist_file=self.input()["poshist_file"].path,
            post_equal_weights_file=self.input()["fit_result"][
                "post_equal_weights"
            ].path,
            used_dets=result["time_selection"]["used_detectors"],
            model=result["fit_result"]["model"],
            ra=result["fit_result"]["ra"],
            dec=result["fit_result"]["dec"],
            swift=result["general"]["swift"],
            save_path=self.output().path,
        )


class CreateSatellitePlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
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

        if self.report_type.lower() == "tte":
            with self.input()["trigdat_version"].open() as f:
                trigdat_version = yaml.safe_load(f)["trigdat_version"]

            trigdat_file = DownloadTrigdat(
                grb_name=self.grb_name, version=trigdat_version
            ).output()

        elif self.report_type.lower() == "trigdat":
            trigdat_file = DownloadTrigdat(
                grb_name=self.grb_name, version=self.version
            ).output()

        else:
            raise UnkownReportType(
                f"The report_type '{self.report_type}' is not valid!"
            )

        azimuthal_plot_sat_frame(
            grb_name=self.grb_name,
            trigdat_file=trigdat_file.path,
            ra=result["fit_result"]["ra"],
            dec=result["fit_result"]["dec"],
            save_path=self.output().path,
        )


class CreateSpectrumPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    trigger_name = luigi.Parameter()

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
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

    def requires(self):
        return ProcessLocalizationResult(
            date=self.date,
            data_type=self.data_type,
            trigger_name=self.trigger_name,
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

        if self.report_type.lower() == "tte":
            with self.input()["trigdat_version"].open() as f:
                trigdat_version = yaml.safe_load(f)["trigdat_version"]

            trigdat_file = DownloadTrigdat(
                grb_name=self.grb_name, version=trigdat_version
            ).output()

        elif self.report_type.lower() == "trigdat":
            trigdat_file = DownloadTrigdat(
                grb_name=self.grb_name, version=self.version
            ).output()

        else:
            raise UnkownReportType(
                f"The report_type '{self.report_type}' is not valid!"
            )

        interactive_3D_plot(
            trigdat_file=trigdat_file.path,
            post_equal_weights_file=self.input()["fit_result"][
                "post_equal_weights"
            ].path,
            used_dets=result["time_selection"]["used_detectors"],
            model=result["fit_result"]["model"],
            save_path=self.output().path,
        )
