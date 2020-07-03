from configya import YAMLConfig

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

structure = {}

structure["luigi"] = dict(n_workers=4)
structure["phys_bkg"] = dict(
    multinest=dict(n_cores=4, path_to_python="python")
)
structure["data"] = dict(
    data_type="ctime",
    detectors=_gbm_detectors,
    echans=list(range(0, 7))
)

class GBMBkgPipeConfig(YAMLConfig):
    def __init__(self):

        super(GBMBkgPipeConfig, self).__init__(
            structure=structure,
            config_path="~/.gbm_bkg_pipe",
            config_name="gbm_bkg_pipe_config.yml",
        )


gbm_bkg_pipe_config = GBMBkgPipeConfig()
