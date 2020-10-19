from configya import YAMLConfig
import copy


run_detectors = [
    ["n0", "n1", "n2", "n3", "n4", "n5"],  # , "b0"]
    ["n6", "n7", "n8", "n9", "na", "nb"],  # , "b1"]
]

run_echans = [["0", "1", "2"], ["3", "4", "5"]]

structure = {}

structure["luigi"] = dict(n_workers=32)
structure["phys_bkg"] = dict(
    stan=dict(n_cores=16),
    multinest=dict(
        n_cores=4, path_to_python="/home/fkunzwei/data1/envs/bkg_pipe/bin/python"
    ),
    timeout=2 * 60 * 60,  # 1 hour
)

structure["remote_hosts_config"] = dict(
    priority_host="raven",
    hosts=dict(
        raven=dict(
            hostname="raven",
            username="fkunzwei",
            script_dir="/u/fkunzwei/scripts/bkg_pipe/",
            base_dir="/ptmp/fkunzwei/gbm_data/bkg_pipe/",
            data_dir="/ptmp/fkunzwei/gbm_data/",
            job_limit=8,
        ),
        cobra=dict(
            hostname="cobra",
            username="fkunzwei",
            script_dir="/u/fkunzwei/scripts/bkg_pipe/",
            base_dir="/ptmp/fkunzwei/gbm_data/bkg_pipe/",
            data_dir="/ptmp/fkunzwei/gbm_data/",
            job_limit=8,
        ),
    ),
)


structure["download"] = dict(
    interval=5 * 60, max_time=6 * 60 * 60  # run every 5 min  # run for 6h
)

structure["balrog"] = dict(
    multinest=dict(n_cores=4, path_to_python="python"), timeout=2 * 60 * 60  # 1 hour
)

structure["data"] = dict(data_type="ctime", detectors=run_detectors, echans=run_echans)


class GBMBkgPipeConfig(YAMLConfig):
    def __init__(self):

        super(GBMBkgPipeConfig, self).__init__(
            structure=structure,
            config_path="~/.gbm_bkg_pipe",
            config_name="gbm_bkg_pipe_config.yml",
        )


gbm_bkg_pipe_config = GBMBkgPipeConfig()
