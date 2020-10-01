from configya import YAMLConfig
import copy

# # run_detectors = [
# #     ["n0"],
# #     ["n1"],
# #     ["n2"],
# #     ["n3"],
# #     ["n4"],
# #     ["n5"],
# #     ["n6"],
# #     ["n7"],
# #     ["n8"],
# #     ["n9"],
# #     ["na"],
# #     ["nb"],
# #     ["b0"],
# #     ["b1"],
# # ]

# # run_echans = [
# #     ["0"],
# #     ["1"],
# #     ["2"],
# #     ["3"],
# #     ["4"],
# #     ["5"],
# #     ["6"],
# #     ["7"],
# # ]

# run_detectors = [
#     ["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "na", "nb"],
# ]
# run_echans = [["0", "1", "2", "3"], ["4", "5", "6", "7"]]

run_detectors = [
    ["n0", "n1", "n2", "n3", "n4", "n5"],  # , "b0"]
    ["n6", "n7", "n8", "n9", "na", "nb"],  # , "b1"]
]

run_echans = [["0", "1", "2"], ["3", "4", "5"]]

structure = {}

structure["luigi"] = dict(n_workers=4)
structure["phys_bkg"] = dict(
    stan=dict(n_cores=16),
    multinest=dict(n_cores=4, path_to_python="python"),
    timeout=2 * 60 * 60,  # 1 hour
    # bkg_source_setup=bkg_source_setup,
)

structure["remote"] = dict(
    host="cobra",
    username="fkunzwei",
    script_dir="/u/fkunzwei/scripts/bkg_pipe/",
    base_dir="/u/fkunzwei/gbm_data/bkg_pipe/",
    gbm_data="/u/fkunzwei/gbm_data/",
)

structure["download"] = dict(
    interval=5 * 60, max_time=6 * 60 * 60  # run every 5 min  # run for 6h
)

structure["balrog"] = dict(
    multinest=dict(n_cores=4, path_to_python="python"), timeout=1 * 60 * 60  # 1 hour
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
