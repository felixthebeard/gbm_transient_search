import os
import time

import numpy as np
import luigi
import yaml

from luigi.contrib.external_program import ExternalProgramTask

from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config
from gbm_bkg_pipe.utils.iteration import chunked_iterable

from gbmbkgpy.io.export import PHAWriter

base_dir = os.environ.get("GBMDATA")
bkg_n_parallel_fits = gbm_bkg_pipe_config["phys_bkg"]["n_parallel_fits"]
bkg_n_cores_multinest = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["n_cores"]
bkg_path_to_python = gbm_bkg_pipe_config["phys_bkg"]["multinest"]["path_to_python"]

_gbm_detectors = (
    "n0",
    "n1",
    "n2",
    "n3",
    "n4",
    "n5",
    "n6",
    "n7",
    "n8",
    "n9",
    "na",
    "nb",
    "b0",
    "b1",
)


class GBMBackgroundModelFit(luigi.Task):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")

    def requires(self):
        echans = [str(i) for i in range(0, 2)]

        bkg_fit_tasks = {}

        for det in _gbm_detectors[:1]:

            for echan in echans:

                bkg_fit_tasks[f"{det}_e{echan}"] = RunPhysBkgModel(
                    date=self.date, echan=echan, detector=det
                )

        return bkg_fit_tasks

    def output(self):

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                "bkg_pipe",
                self.data_type,
                f"{self.date:%y%m%d}",
                "phys_bkg",
                "phys_bkg_combined.hdf5",
            )
        )

    def run(self):

        bkg_fit_results = [bkg_fit.path for bkg_fit in self.input().values()]

        # PHACombiner and save combined file
        pha_writer = PHAWriter.from_result_files(bkg_fit_results)

        pha_writer.save_combined_hdf5(self.output().path)


class RunPhysBkgModel(ExternalProgramTask):
    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    echan = luigi.Parameter()
    detector = luigi.Parameter()
    always_log_stderr = True

    def requires(self):
        return None

    def output(self):

        return luigi.LocalTarget(
            os.path.join(
                base_dir,
                "bkg_pipe",
                self.data_type,
                f"{self.date:%y%m%d}",
                "phys_bkg",
                f"{self.detector}",
                f"e{self.echan}",
                "fit_result.hdf5",
            )
        )

    def program_args(self):

        fit_script_path = f"{os.path.dirname(os.path.abspath(__file__))}/phys_bkg_model/fit_phys_bkg.py"

        command = []

        # Run with mpi in parallel
        if bkg_n_cores_multinest > 1:

            command.extend(
                ["mpiexec", f"-n", f"{bkg_n_cores_multinest}",]
            )

        command.extend(
            [
                f"{bkg_path_to_python}",
                f"{fit_script_path}",
                f"-dates",
                f"{self.date:%y%m%d}",
                f"-dtype",
                f"{self.data_type}",
                f"-dets",
                f"{self.detector}",
                f"-e",
                f"{self.echan}",
                f"-out",
                f"{os.path.dirname(self.output().path)}",
            ]
        )
        return command
