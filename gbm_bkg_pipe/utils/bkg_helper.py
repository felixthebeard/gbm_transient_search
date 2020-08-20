import os
import yaml
import h5py
from datetime import timedelta

from gbm_bkg_pipe.configuration import gbm_bkg_pipe_config

base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")
bkg_source_setup = gbm_bkg_pipe_config["phys_bkg"]["bkg_source_setup"]


class BkgConfigWriter(object):

    def __init__(self, date, data_type, echans, detectors):
        self._date = date
        self._data_type = data_type
        self._echans = echans
        self._detectors = detectors

        self._load_default_config()

        self._update_general()

        self._update_source_setup()

    def _load_default_config(self):
        config_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/phys_bkg_model/config_fit.yml"

        # Load the default config file
        with open(config_path) as f:
            self._config = yaml.load(f)

    def _update_general(self):

        general_config = dict(
            general=dict(
                dates=[f"{self._date:%y%m%d}"],
                data_type=self._data_type,
                echans=[int(echan) for echan in self._echans],
                detectors=list(self._detectors),
                min_bin_width=40
            ),
        )

        # Update the config parameters with fit specific values
        self._config.update(general_config)

    def _update_source_setup(self):

        source_config = dict(setup=bkg_source_setup["_".join(self._echans)])

        # Update the config parameters with fit specific values
        self._config.update(source_config)

    def _update_bounds(self):

        day_before = self._date - timedelta(days=1)

        job_dir_day_before = os.path.join(
            base_dir,
            f"{day_before:%y%m%d}",
            self._data_type,
            "phys_bkg",
            f"det_{'_'.join(self._detectors)}",
            f"e{'_'.join(self._echans)}",
        )

        if os.path.exists(job_dir_day_before):

            print("daybefore found")

        result_file = os.path.join(day_before, "fit_result.hdf5")

        if os.path.exists(result_file):

            with h5py.File(result_file, "r") as f:

                param_names = f.attrs['param_names']

                best_fit_values = f.attrs["best_fit_values"]

            params = dict(zip(param_names, best_fit_values))

        # TODO: Use the best fit values from the previous day to construct tight priors
        # This needs some updates in the background model...

    def write_config_file(self, output):

        output().makedirs()

        with output().open(mode="w") as f:
            yaml.dump(self._config, f, default_flow_style=False)
