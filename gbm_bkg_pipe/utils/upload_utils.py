import json
import time

import requests
from datetime import datetime

from gbm_bkg_pipe.exceptions.custom_exceptions import (
    EmptyFileError,
    TransientNotFound,
    UnauthorizedRequest,
    UnexpectedStatusCode,
    UploadFailed,
)
from gbm_bkg_pipe.utils.env import get_env_value
from gbm_bkg_pipe.utils.env import get_bool_env_value

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
base_url = get_env_value("GBM_BKG_PIPE_BASE_URL")
auth_token = get_env_value("GBM_BKG_PIPE_AUTH_TOKEN")

model_lookup = {
    "pl": "powerlaw",
    "cpl": "cutoff_powerlaw",
    "band": "band_function",
    "blackbody": "blackbody",
}


def check_transient_on_website(trigger_name):
    headers = {"Authorization": f"Token {auth_token}"}

    check_existing_url = f"{base_url}/api/check_transient_name/{trigger_name}/"

    response = requests.get(url=check_existing_url, headers=headers, verify=True)

    if simulate:
        return False

    # TRANSIENT not in DB
    if response.status_code == 204:
        return False

    # TRANSIENT already in DB
    elif response.status_code == 200:
        return True

    elif response.status_code == 401:
        raise UnauthorizedRequest("The authentication token is not valid")

    # This should not happen, but we will try to upload anyway
    else:
        return False


def create_report_from_result(result):
    web_version = "v01"

    report = {
        "name": result["trigger"]["trigger_name"],
        "name_gcn": result["trigger"].get("trigger_name_gcn", None),
        "hide_burst": False,
        "trigger_number": None,
        "trigger_timestamp": result["trigger"]["trigger_time_utc"],
        "transient_params": [
            {
                "version": web_version,
                "model_type": model_lookup[result["fit_result"]["model"]],
                "trigger_number": None,
                "trigger_timestamp": result["trigger"]["trigger_time_utc"],
                "data_timestamp": result["trigger"]["data_timestamp"],
                "localization_timestamp": datetime.utcnow().strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                ),
                "balrog_ra": result["fit_result"]["ra"],
                "balrog_ra_err": result["fit_result"]["ra_err"],
                "balrog_dec": result["fit_result"]["dec"],
                "balrog_dec_err": result["fit_result"]["dec_err"],
                "swift_ra": result["trigger"]["swift"].get("ra", None)
                if result["trigger"].get("swift", None) is not None
                else None,
                "swift_dec": result["trigger"]["swift"].get("dec", None)
                if result["trigger"].get("swift", None) is not None
                else None,
                "spec_K": result["fit_result"]["spec_K"],
                "spec_K_err": result["fit_result"]["spec_K_err"],
                "spec_index": result["fit_result"]["spec_index"],
                "spec_index_err": result["fit_result"]["spec_index_err"],
                "spec_xc": result["fit_result"]["spec_xc"],
                "spec_xc_err": result["fit_result"]["spec_xc_err"],
                "spec_kT": result["fit_result"]["spec_kT"],
                "spec_kT_err": result["fit_result"]["spec_kT_err"],
                "sat_phi": result["fit_result"]["sat_phi"],
                "sat_theta": result["fit_result"]["sat_theta"],
                "spec_alpha": result["fit_result"]["spec_alpha"],
                "spec_alpha_err": result["fit_result"]["spec_alpha_err"],
                "spec_xp": result["fit_result"]["spec_xp"],
                "spec_xp_err": result["fit_result"]["spec_xp_err"],
                "spec_beta": result["fit_result"]["spec_beta"],
                "spec_beta_err": result["fit_result"]["spec_beta_err"],
                "trigger_interval_start": result["trigger"]["interval"]["start"],
                "trigger_interval_stop": result["trigger"]["interval"]["stop"],
                "active_time_start": result["trigger"]["active_time_start"],
                "active_time_stop": result["trigger"]["active_time_end"],
                "used_detectors": ", ".join(
                    str(det_nr) for det_nr in result["trigger"]["use_dets"]
                ),
                "balrog_one_sig_err_circle": result["fit_result"][
                    "balrog_one_sig_err_circle"
                ],
                "balrog_two_sig_err_circle": result["fit_result"][
                    "balrog_two_sig_err_circle"
                ],
            }
        ],
    }
    return report


def upload_transient_report(trigger_name, result, wait_time, max_time):
    headers = {
        "Authorization": "Token {}".format(auth_token),
        "Content-Type": "application/json",
    }
    upload_new_version = False
    transient_on_website = check_transient_on_website(trigger_name)

    # Upload new version of report
    if transient_on_website:
        upload_new_version = True
        url = f"{base_url}/api/transients/{trigger_name}/params/"

    # Create TRANSIENT entry on website and upload report
    else:
        url = f"{base_url}/api/transients/"

    report = create_report_from_result(result)

    if simulate:
        return report

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    while flag:

        # try to download the file
        try:

            response = requests.post(
                url=url, data=json.dumps(report), headers=headers, verify=True
            )
            if response.status_code == 201:
                print("Uploaded new TRANSIENT")
                # kill the loop

                flag = False

            elif response.status_code == 206:
                print("Uploaded new TRANSIENT but notification went wrong")
                # kill the loop

                flag = False

            elif response.status_code == 401:
                raise UnauthorizedRequest("The authentication token is not valid")

            elif response.status_code == 409 and not upload_new_version:
                print("TRANSIENT already existing, try uploading new version")
                url = f"{base_url}/api/transients/{trigger_name}/params/"

            elif response.status_code == 409 and upload_new_version:
                print("################################################")
                print("The report for this version is already in the DB")
                print("################################################")
                flag = False

            else:
                raise UnexpectedStatusCode(
                    f"Request returned unexpected status {response.status_code} with message {response.text}"
                )

        except UnauthorizedRequest as e:
            raise e

        except Exception as e:

            print(e)

            # ok, we have not uploaded the transient yet

            # see if we should still wait for the upload

            if time_spent >= max_time:

                # we are out of time so give up

                raise UploadFailed("The max time has been exceeded!")

            else:

                # ok, let's sleep for a bit and then check again

                time.sleep(wait_time)

                # up date the time we have left

                time_spent += wait_time
        else:

            print(f"Response {response.status_code} at {url}: {response.text}")
    return report


def update_transient_report(trigger_name, result, wait_time, max_time):
    headers = {
        "Authorization": "Token {}".format(auth_token),
        "Content-Type": "application/json",
    }

    # Update the TRANSIENT report on the website
    if check_transient_on_website(trigger_name):
        url = f"{base_url}/'api/transients/{trigger_name}/params/"

    # Update of not possible if TRANSIENT is not already there
    else:
        raise TransientNotFound(
            f"Update of {trigger_name} not possible, because it is not on Website"
        )

    report = create_report_from_result(result)

    if simulate:
        return True

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    while flag:

        # try to download the file
        try:

            response = requests.put(
                url=url, data=json.dumps(report), headers=headers, verify=True
            )
            if response.status_code == 201:
                print("Updated TRANSIENT")
                # kill the loop

                flag = False

            elif response.status_code == 401:
                raise UnauthorizedRequest("The authentication token is not valid")

            else:
                raise UnexpectedStatusCode(
                    f"Request returned unexpected status {response.status_code} with message {response.text}"
                )

        except UnauthorizedRequest as e:
            raise e

        except Exception as e:

            print(e)

            # ok, we have not uploaded the transient yet

            # see if we should still wait for the upload

            if time_spent >= max_time:

                # we are out of time so give up

                raise UploadFailed("The max time has been exceeded!")

            else:

                # ok, let's sleep for a bit and then check again

                time.sleep(wait_time)

                # up date the time we have left

                time_spent += wait_time
        else:

            print(f"Response {response.status_code} at {url}: {response.text}")


def upload_plot(
    trigger_name,
    data_type,
    plot_file,
    plot_type,
    wait_time,
    max_time,
    det_name="",
):
    headers = {
        "Authorization": "Token {}".format(auth_token),
    }

    web_version = "v01"

    payload = {"plot_type": plot_type, "version": web_version, "det_name": det_name}

    # Update the TRANSIENT report on the website
    if check_transient_on_website(trigger_name):
        url = f"{base_url}/api/transients/{trigger_name}/plot/"

    # Update of not possible if TRANSIENT is not already there
    else:
        raise TransientNotFound(
            f"Upload of plot for {trigger_name} not possible, because TRANSIENT is missing"
        )

    if simulate:
        return True

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    with open(plot_file, "rb") as file_:

        while flag:

            # try to download the file
            try:
                response = requests.post(
                    url=url,
                    data=payload,
                    headers=headers,
                    files={"file": file_},
                    verify=True,
                )
                if response.status_code == 201:
                    print("Uploaded new plot")
                    flag = False

                elif response.status_code == 401:
                    raise UnauthorizedRequest("The authentication token is not valid")

                elif response.status_code == 204:
                    raise EmptyFileError(f"The plot file is empty {plot_file}")

                elif response.status_code == 409:
                    print("####################################################")
                    print("The plot for this version is already in the Database")
                    print("####################################################")
                    flag = False

                else:
                    raise UnexpectedStatusCode(
                        f"Request returned unexpected status {response.status_code} with message {response.text}"
                    )

            except UnauthorizedRequest as e:
                raise e

            except EmptyFileError as e:
                print(plot_file)
                raise e

            except Exception as e:

                print(e)

                # ok, we have not uploaded the transient yet

                # see if we should still wait for the upload

                if time_spent >= max_time:

                    # we are out of time so give up

                    raise UploadFailed("The max time has been exceeded!")

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(wait_time)

                    # up date the time we have left

                    time_spent += wait_time
            else:

                print(f"Response {response.status_code} at {url}: {response.text}")


def upload_datafile(
    trigger_name,
    data_type,
    data_file,
    file_type,
    version,
    wait_time,
    max_time,
    phys_bkg=False,
):
    headers = {
        "Authorization": "Token {}".format(auth_token),
    }

    web_version = "v01"

    payload = {"file_type": file_type, "version": web_version}

    # Update the TRANSIENT report on the website
    if check_transient_on_website(trigger_name):
        url = f"{base_url}/api/transients/{trigger_name}/datafile/"

    # Update of not possible if TRANSIENT is not already there
    else:
        raise TransientNotFound(
            f"Upload of datafile for {trigger_name} not possible, because TRANSIENT is missing"
        )

    if simulate:
        return True

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    with open(data_file, "rb") as file_:

        while flag:

            # try to download the file
            try:
                response = requests.post(
                    url=url,
                    data=payload,
                    headers=headers,
                    files={"file": file_},
                    verify=True,
                )
                if response.status_code == 201:
                    print("Uploaded new plot")
                    flag = False

                elif response.status_code == 401:
                    raise UnauthorizedRequest("The authentication token is not valid")

                elif response.status_code == 204:
                    raise EmptyFileError("The datafile was empty")

                elif response.status_code == 409:
                    print("#########################################################")
                    print("The data file for this version is already in the Database")
                    print("#########################################################")
                    flag = False

                else:
                    raise UnexpectedStatusCode(
                        f"Request returned unexpected status {response.status_code} with message {response.text}"
                    )

            except UnauthorizedRequest as e:
                raise e

            except EmptyFileError as e:
                raise e

            except Exception as e:

                print(e)

                # ok, we have not uploaded the transient yet

                # see if we should still wait for the upload

                if time_spent >= max_time:

                    # we are out of time so give up

                    raise UploadFailed("The max time has been exceeded!")

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(wait_time)

                    # up date the time we have left

                    time_spent += wait_time
            else:

                print(f"Response {response.status_code} at {url}: {response.text}")


def upload_date_plot(
    date, data_type, plot_file, plot_type, wait_time, max_time, det_name="", echan=""
):
    headers = {
        "Authorization": "Token {}".format(auth_token),
    }

    payload = {
        "plot_type": plot_type,
        "data_type": data_type,
        "det_name": det_name,
        "echan": echan,
    }

    url = f"{base_url}/api/transients/date/{date:%y%m%d}/plot/"

    if simulate:
        return True

    # set a flag to kill the job
    flag = True

    # the time spent waiting so far
    time_spent = 0  # seconds

    with open(plot_file, "rb") as file_:

        while flag:

            # try to download the file
            try:
                response = requests.post(
                    url=url,
                    data=payload,
                    headers=headers,
                    files={"file": file_},
                    verify=True,
                )
                if response.status_code == 201:
                    print("Uploaded new plot")
                    flag = False

                elif response.status_code == 401:
                    raise UnauthorizedRequest("The authentication token is not valid")

                elif response.status_code == 204:
                    raise EmptyFileError(f"The plot file is empty {plot_file}")

                elif response.status_code == 409:
                    print("####################################################")
                    print("The plot for this version is already in the Database")
                    print("####################################################")
                    flag = False

                else:
                    raise UnexpectedStatusCode(
                        f"Request returned unexpected status {response.status_code} with message {response.text}"
                    )

            except UnauthorizedRequest as e:
                raise e

            except EmptyFileError as e:
                print(plot_file)
                raise e

            except Exception as e:

                print(e)

                # ok, we have not uploaded the transient yet

                # see if we should still wait for the upload

                if time_spent >= max_time:

                    # we are out of time so give up

                    raise UploadFailed("The max time has been exceeded!")

                else:

                    # ok, let's sleep for a bit and then check again

                    time.sleep(wait_time)

                    # up date the time we have left

                    time_spent += wait_time
            else:

                print(f"Response {response.status_code} at {url}: {response.text}")
