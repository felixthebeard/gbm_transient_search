import os
import time

import astropy.io.fits as fits
import astropy.time as astro_time
import astropy.units as u
import numpy as np
from gbmbkgpy.io.downloading import (
    download_data_file,
    download_flares,
    download_lat_spacecraft,
)
from gbmbkgpy.io.file_utils import (
    file_existing_and_readable,
    if_dir_containing_file_not_existing_then_make,
)
from gbmbkgpy.io.package_data import (
    get_path_of_data_file,
    get_path_of_external_data_dir,
    get_path_of_external_data_file,
)
from gbmgeometry import GBMTime


class BackgroundDataDownload(object):
    def __init__(
        self,
        date,
        data_type,
        detector,
        wait_time=60,
        max_time=60 * 60,
    ):
        """
        :param wait_time: the wait time interval for checking files
        :param max_time: the max time to wait for files
        :returns:
        :rtype:

        """
        self._date = date
        self._data_type = data_type
        self._detector = detector

        self._wait_time = wait_time
        self._max_time = max_time

    def run(self):

        # set a flag to kill the job

        flag = True

        sucess = False

        # the time spent waiting so far
        time_spent = 0  # seconds

        while flag:

            # try to download the file
            try:

                download_data_file(self._date, self._data_type, self._detector)

                # kill the loop

                flag = False

                sucess = True

            except:

                # ok, we have not found a file yet

                # see if we should still wait for the file

                if time_spent >= self._max_time:

                    # we are out of time so give up

                    flag = False

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(self._wait_time)

                    # up date the time we have left

                    time_spent += self._wait_time

        return sucess


class BackgroundLATDownload(object):
    def __init__(
        self,
        date,
        wait_time=60,
        max_time=60 * 60,
    ):
        """
        :param wait_time: the wait time interval for checking files
        :param max_time: the max time to wait for files
        :returns:
        :rtype:

        """
        self._date = date

        self._wait_time = wait_time
        self._max_time = max_time

    def run(self):

        # set a flag to kill the job

        flag = True

        sucess = False

        file_names = []

        # the time spent waiting so far
        time_spent = 0  # seconds

        while flag:

            # try to download the file
            try:

                file_names = download_lat_check_week(self._date)

                # kill the loop

                flag = False

                sucess = True

            except:

                # ok, we have not found a file yet

                # see if we should still wait for the file

                if time_spent >= self._max_time:

                    # we are out of time so give up

                    flag = False

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(self._wait_time)

                    # up date the time we have left

                    time_spent += self._wait_time

        return sucess, file_names


def download_lat_check_week(date):
    # read the file
    day = astro_time.Time(f"20{date[:2]}-{date[2:-2]}-{date[-2:]}")

    min_met = GBMTime(day).met

    max_met = GBMTime(day + u.Quantity(1, u.day)).met

    gbm_time = GBMTime(day)

    mission_week = np.floor(gbm_time.mission_week.value)

    filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % mission_week
    filepath = get_path_of_external_data_file("lat", filename)

    file_names = [filename]

    if not file_existing_and_readable(filepath):
        download_lat_spacecraft(mission_week)

    # lets check that this file has the right information
    week_before = False
    week_after = False

    with fits.open(filepath) as f:

        if f["PRIMARY"].header["TSTART"] >= min_met:

            # we need to get week before

            week_before = True

            before_filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % (
                mission_week - 1
            )
            before_filepath = get_path_of_external_data_file("lat", before_filename)
            if not file_existing_and_readable(before_filepath):
                download_lat_spacecraft(mission_week - 1)

            file_names.append(before_filename)

        if f["PRIMARY"].header["TSTOP"] <= max_met:

            # we need to get week after

            week_after = True

            after_filename = "lat_spacecraft_weekly_w%d_p202_v001.fits" % (
                mission_week + 1
            )
            after_filepath = get_path_of_external_data_file("lat", after_filename)
            if not file_existing_and_readable(after_filepath):
                download_lat_spacecraft(mission_week + 1)

            file_names.append(after_filename)

    return file_names
