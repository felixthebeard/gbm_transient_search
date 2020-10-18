import shlex
import subprocess

from gbm_bkg_pipe import gbm_bkg_pipe_config

n_workers = int(gbm_bkg_pipe_config["luigi"]["n_workers"])


def form_cmd_string(date):
    """
    makes the command string for luigi

    :param grb:
    :returns:
    :rtype:

    """

    base_cmd = "luigi --module gbm_bkg_pipe "

    cmd = f"{base_cmd} CreateReportDate --date {date} --data-type ctime "

    cmd += f"--workers {n_workers} --scheduler-host localhost"

    cmd = shlex.split(cmd)

    return cmd
