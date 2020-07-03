from configya import YAMLConfig


structure = {}

structure["luigi"] = dict(n_workers=4)
structure["multinest"] = dict(n_cores=4, path_to_python="python")
structure["phys_bkg"] = dict(
    n_parallel_fits=8, multinest=dict(n_cores=4, path_to_python="python")
)


class GBMBkgPipeConfig(YAMLConfig):
    def __init__(self):

        super(GBMBkgPipeConfig, self).__init__(
            structure=structure,
            config_path="~/.gbm_bkg_pipe",
            config_name="gbm_bkg_pipe_config.yml",
        )


gbm_bkg_pipe_config = GBMBkgPipeConfig()
