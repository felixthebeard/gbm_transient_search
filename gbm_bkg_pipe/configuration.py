from configya import YAMLConfig

run_detectors = [
    ["n0"],
    ["n1"],
    ["n2"],
    ["n3"],
    ["n4"],
    ["n5"],
    ["n6"],
    ["n7"],
    ["n8"],
    ["n9"],
    ["na"],
    ["nb"],
    ["b0"],
    ["b1"],
]

run_echans = [
    ["0"],
    ["1"],
    ["2"],
    ["3"],
    ["4"],
    ["5"],
    ["6"],
    ["7"],
]

structure = {}

structure["luigi"] = dict(n_workers=4)
structure["phys_bkg"] = dict(
    multinest=dict(n_cores=4, path_to_python="python")
)

structure["data"] = dict(
    data_type="ctime",
    detectors=run_detectors,
    echans=run_echans
)


class GBMBkgPipeConfig(YAMLConfig):
    def __init__(self):

        super(GBMBkgPipeConfig, self).__init__(
            structure=structure,
            config_path="~/.gbm_bkg_pipe",
            config_name="gbm_bkg_pipe_config.yml",
        )


gbm_bkg_pipe_config = GBMBkgPipeConfig()
