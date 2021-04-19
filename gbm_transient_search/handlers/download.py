import datetime as dt
import os
import tempfile
import time
from datetime import datetime, timedelta

import luigi
from gbm_transient_search.utils.configuration import gbm_transient_search_config
from gbm_transient_search.utils.download_file import (
    BackgroundDataDownload,
    BackgroundLATDownload,
)
from gbm_transient_search.utils.env import get_bool_env_value, get_env_value
from gbm_transient_search.utils.file_utils import (
    if_dir_containing_file_not_existing_then_make,
    if_directory_not_existing_then_make,
)
from gbm_transient_search.utils.luigi_ssh import (
    RemoteCalledProcessError,
    RemoteContext,
    RemoteTarget,
)
from gbmbkgpy.utils.select_pointsources import build_swift_pointsource_database

base_dir = os.path.join(get_env_value("GBMDATA"), "bkg_pipe")

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
data_dir = os.environ.get("GBMDATA")

run_detectors = gbm_transient_search_config["data"]["detectors"]
run_echans = gbm_transient_search_config["data"]["echans"]

remote_hosts_config = gbm_transient_search_config["remote_hosts_config"]


class DownloadData(luigi.Task):
    """
    Downloads a DataFile
    """

    date = luigi.DateParameter()
    data_type = luigi.Parameter(default="ctime")
    detector = luigi.ListParameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": 1, "ssh_connections": 1}

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
            remote_success=luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    "data_remote",
                    f"copied_{self.data_type}_{self.detector}_{self.date:%y%m%d}_to_{self.remote_host}",
                )
            ),
        )

    def remote_output(self):
        datafile_name = (
            f"glg_{self.data_type}_{self.detector}_{self.date:%y%m%d}_v00.pha"
        )
        return dict(
            remote_file=RemoteTarget(
                os.path.join(
                    remote_hosts_config["hosts"][self.remote_host]["data_dir"],
                    self.data_type,
                    f"{self.date:%y%m%d}",
                    datafile_name,
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
        )

    def run(self):
        if not self.output()["remote_success"].exists():
            if not self.output()["local_file"].exists():

                if simulate:
                    raise Exception(
                        "Running in simulation mode, but simulation data file not existing"
                    )

                dl = BackgroundDataDownload(
                    f"{self.date:%y%m%d}",
                    self.data_type,
                    self.detector,
                    wait_time=float(
                        gbm_transient_search_config["download"]["interval"]
                    ),
                    max_time=float(gbm_transient_search_config["download"]["max_time"]),
                )
                file_readable = dl.run()

            else:
                file_readable = True

            if file_readable:

                self.remote_output()["remote_file"].put(
                    self.output()["local_file"].path
                )

                self.output()["remote_success"].makedirs()
                os.system(f"touch {self.output()['remote_success'].path}")

            else:

                raise Exception(
                    f"Download of data for {self.detector} on {self.date:%y%m%d} failed"
                )


class DownloadPoshistData(luigi.Task):
    """
    Downloads a DataFile
    """

    date = luigi.DateParameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": 1, "ssh_connections": 1}

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
            remote_success=luigi.LocalTarget(
                os.path.join(
                    base_dir,
                    f"{self.date:%y%m%d}",
                    "data_remote",
                    f"copied_poshist_{self.date:%y%m%d}_to_{self.remote_host}",
                )
            ),
        )

    def remote_output(self):
        datafile_name = f"glg_poshist_all_{self.date:%y%m%d}_v00.fit"

        return dict(
            remote_file=RemoteTarget(
                os.path.join(
                    remote_hosts_config["hosts"][self.remote_host]["data_dir"],
                    "poshist",
                    datafile_name,
                ),
                host=self.remote_host,
                username=remote_hosts_config["hosts"][self.remote_host]["username"],
                # sshpass=True,
            ),
        )

    def run(self):
        if not self.output()["remote_success"].exists():
            if not self.output()["local_file"].exists():

                dl = BackgroundDataDownload(
                    f"{self.date:%y%m%d}",
                    "poshist",
                    "all",
                    wait_time=float(
                        gbm_transient_search_config["download"]["interval"]
                    ),
                    max_time=float(gbm_transient_search_config["download"]["max_time"]),
                )
                file_readable = dl.run()

            else:
                file_readable = True

            if file_readable:

                self.remote_output()["remote_file"].put(
                    self.output()["local_file"].path
                )

                self.output()["remote_success"].makedirs()
                os.system(f"touch {self.output()['remote_success'].path}")

            else:

                raise Exception(
                    f"Download of poshist data for {self.date:%y%m%d} failed"
                )


class DownloadLATData(luigi.Task):
    """
    Downloads a DataFile
    """

    date = luigi.DateParameter()
    remote_host = luigi.Parameter()

    resources = {"cpu": 1, "ssh_connections": 1}

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
            wait_time=float(gbm_transient_search_config["download"]["interval"]),
            max_time=float(gbm_transient_search_config["download"]["max_time"]),
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
                    # sshpass=True,
                )

                remote_file.put(local_file.path)

            os.system(f"touch {self.output().path}")

        else:

            raise Exception(f"Download of LAT data for {self.date:%y%m%d} failed")


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
