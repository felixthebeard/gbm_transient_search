import os
from datetime import datetime

import luigi
import numpy as np
import yaml
from gbm_transient_search.utils.configuration import gbm_transient_search_config
from gbm_transient_search.handlers.localization import ProcessLocalizationResult
from gbm_transient_search.handlers.transient_search import TransientSearch
from gbm_transient_search.handlers.plotting import (
    BkgModelPlots,
    Create3DLocationPlot,
    CreateAllLightcurves,
    CreateBkgModelPlot,
    CreateCornerPlot,
    CreateLocationPlot,
    CreateMollLocationPlot,
    CreateSatellitePlot,
    CreateSpectrumPlot,
)
from gbm_transient_search.utils.env import get_bool_env_value, get_env_value
from gbm_transient_search.utils.file_utils import if_dir_containing_file_not_existing_then_make
from gbm_transient_search.utils.upload_utils import (
    upload_date_plot,
    upload_plot,
    upload_transient_report,
)

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")

_valid_gbm_detectors = np.array(gbm_transient_search_config["data"]["detectors"]).flatten()
_valid_echans = np.array(gbm_transient_search_config["data"]["echans"]).flatten()


class UploadTriggers(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    resources = {"cpu": 1}

    def requires(self):
        return TransientSearch(
            date=self.date,
            data_type=self.data_type,
            remote_host=self.remote_host,
            step=self.step,
        )

    def output(self):
        filename = f"upload_triggers_done.txt"

        return luigi.LocalTarget(
            os.path.join(
                base_dir, f"{self.date:%y%m%d}", self.data_type, self.step, filename
            )
        )

    def run(self):
        with self.input().open("r") as f:
            trigger_information = yaml.safe_load(f)

        upload_tasks = []

        for t_info in trigger_information["triggers"].values():

            upload_tasks.extend(
                [
                    UploadReport(
                        date=datetime.strptime(t_info["date"], "%y%m%d"),
                        data_type=trigger_information["data_type"],
                        trigger_name=t_info["trigger_name"],
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                    UploadAllPlots(
                        date=datetime.strptime(t_info["date"], "%y%m%d"),
                        data_type=trigger_information["data_type"],
                        trigger_name=t_info["trigger_name"],
                        remote_host=self.remote_host,
                        step=self.step,
                    ),
                ]
            )
        yield upload_tasks

        os.system(f"touch {self.output().path}")


class UploadReport(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
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
                f"{self.trigger_name}_report.yml",
            )
        )

    def run(self):
        with self.input()["result_file"].open() as f:
            result = yaml.safe_load(f)

        report = upload_transient_report(
            trigger_name=self.trigger_name,
            result=result,
            wait_time=float(gbm_transient_search_config["upload"]["report"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["report"]["max_time"]),
        )

        with open(self.output().path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)


class UploadAllPlots(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "lightcurves": UploadAllLightcurves(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "location": UploadLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "corner": UploadCornerPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "molllocation": UploadMollLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "satellite": UploadSatellitePlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "spectrum": UploadSpectrumPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "3d_location": Upload3DLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            # "balrogswift": UploadBalrogSwiftPlot(
            #     date=self.date,
            #     data_type=self.data_type,
            #     trigger_name=self.trigger_name,
            #     remote_host=self.remote_host,
            #     step=self.step
            # ),
        }

    def output(self):

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_all.done",
            )
        )

    def run(self):
        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadAllLightcurves(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        upload_lightcurves = {}

        for det in _valid_gbm_detectors:
            upload_lightcurves[det] = UploadLightcurve(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                detector=det,
                step=self.step,
            )
        return upload_lightcurves

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_all_lightcurves.done",
            )
        )

    def run(self):
        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadLightcurve(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    detector = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "plot_file": CreateAllLightcurves(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_{self.detector}_upload_plot_lightcurve.done",
            )
        )

    def run(self):

        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"][self.detector].path,
            plot_type="lightcurve",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
            det_name=self.detector,
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
            "plot_file": CreateLocationPlot(
                date=self.date,
                data_type=self.data_type,
                trigger_name=self.trigger_name,
                remote_host=self.remote_host,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_location.done",
            )
        )

    def run(self):

        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="location",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadCornerPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
            "plot_file": CreateCornerPlot(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_corner.done",
            )
        )

    def run(self):

        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="allcorner",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadMollLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
            "plot_file": CreateMollLocationPlot(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_molllocation.done",
            )
        )

    def run(self):
        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="molllocation",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadSatellitePlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
            "plot_file": CreateSatellitePlot(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_satellite.done",
            )
        )

    def run(self):

        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="satellite",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadSpectrumPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
            "plot_file": CreateSpectrumPlot(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_spectrum.done",
            )
        )

    def run(self):

        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="spectrum",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class Upload3DLocationPlot(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    trigger_name = luigi.Parameter()
    remote_host = luigi.Parameter()
    step = luigi.Parameter()

    def requires(self):
        return {
            "create_report": UploadReport(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
            "plot_file": Create3DLocationPlot(
                date=self.date,
                remote_host=self.remote_host,
                trigger_name=self.trigger_name,
                data_type=self.data_type,
                step=self.step,
            ),
        }

    def output(self):
        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                f"{self.date:%y%m%d}",
                self.data_type,
                self.step,
                "trigger",
                self.trigger_name,
                "upload",
                f"{self.trigger_name}_upload_plot_3dlocation.done",
            )
        )

    def run(self):

        upload_plot(
            trigger_name=self.trigger_name,
            data_type=self.data_type,
            plot_file=self.input()["plot_file"].path,
            plot_type="3dlocation",
            wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
            max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


# class UploadBalrogSwiftPlot(luigi.Task):
#     date = luigi.DateParameter()
#     data_type = luigi.Parameter()
#     trigger_name = luigi.Parameter()
#     remote_host = luigi.Parameter()

#     def requires(self):
#         return {
#             "create_report": UploadReport(
#                 date=self.date,
#                 remote_host=self.remote_host,
#                 trigger_name=self.trigger_name,
#                 data_type=self.data_type,
#                        step=self.step
#             ),
#             "plot_file": CreateBalrogSwiftPlot(
#                 date=self.date,
#                 remote_host=self.remote_host,
#                 trigger_name=self.trigger_name,
#                 data_type=self.data_type,
#                        step=self.step
#             ),
#         }

#     def output(self):
#         return luigi.LocalTarget(
#             os.path.join(
#                 base_dir,
#                 f"{self.date:%y%m%d}",
#                 self.data_type, self.step,
#                 "trigger",
#                 self.trigger_name,
#                 "upload",
#                 f"{self.trigger_name}_upload_plot_balrogswift.done",
#             )
#         )

#     def run(self):

#         upload_plot(
#             trigger_name=self.trigger_name,
#             data_type=self.data_type,
#             plot_file=self.input()["plot_file"].path,
#             plot_type="balrogswift",
#             wait_time=float(gbm_transient_search_config["upload"]["plot"]["interval"]),
#             max_time=float(gbm_transient_search_config["upload"]["plot"]["max_time"]),
#         )

#         if_dir_containing_file_not_existing_then_make(self.output().path)

#         os.system(f"touch {self.output().path}")


class UploadBkgResultPlots(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    remote_host = luigi.Parameter()
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
                "upload",
                "upload_plot_all_bkg_results.done",
            )
        )

    def run(self):

        for task_name, task_outputs in self.requires().input().items():

            if "result_plot" in task_name:

                for det_echan, plot_file in task_outputs.items():
                    if "summary" in det_echan:
                        continue

                    det, echan = det_echan.split("_")

                    upload_date_plot(
                        date=self.date,
                        plot_name="",
                        data_type=self.data_type,
                        plot_file=plot_file.path,
                        plot_type="bkg_result",
                        wait_time=float(
                            gbm_transient_search_config["upload"]["plot"]["interval"]
                        ),
                        max_time=float(
                            gbm_transient_search_config["upload"]["plot"]["max_time"]
                        ),
                        det_name=det,
                        echan=echan,
                    )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")


class UploadBkgPerformancePlots(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter()
    remote_host = luigi.Parameter()
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
                "upload",
                "upload_plot_all_performance_plots.done",
            )
        )

    def run(self):

        for task_name, task in self.requires().requires().items():

            if "performance_plots" in task_name:

                for plot_type, plot_file in task.output().items():
                    if "all" in plot_type:

                        plot_name = f"dets: {', '.join(task.detectors)} echans: {', '.join(task.echans)}"

                        upload_date_plot(
                            date=self.date,
                            plot_name=plot_name,
                            data_type=self.data_type,
                            plot_file=plot_file.path,
                            plot_type=f"{plot_type.split('_')[0]}",
                            wait_time=float(
                                gbm_transient_search_config["upload"]["plot"]["interval"]
                            ),
                            max_time=float(
                                gbm_transient_search_config["upload"]["plot"]["max_time"]
                            ),
                            det_name=f"{', '.join(task.detectors)}",
                            echan=f"{', '.join(task.echans)}",
                        )

        if_dir_containing_file_not_existing_then_make(self.output().path)

        os.system(f"touch {self.output().path}")
