import os
import shutil
import time

import astropy.utils.data as astro_data

import gbm_bkg_pipe.utils.file_utils as file_utils

from gbmbkgpy.io.downloading import download_data_file


class BackgroundDataDownload(object):
    def __init__(
        self, date, data_type, detector, wait_time=60, max_time=60 * 60,
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

                download_data_file(
                    self._date,
                    self._data_type,
                    self._detector
                )

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
