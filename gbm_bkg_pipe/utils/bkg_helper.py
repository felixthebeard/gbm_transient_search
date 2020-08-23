import os
import re
import yaml
import h5py
from datetime import timedelta

from gbmbkgpy.utils.select_pointsources import SelectPointsources
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

        self._update_ps_setup()

        self._update_priors()

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
                echans=[echan for echan in self._echans],
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

    def _update_ps_setup(self):
        # Only inlcude point sources for echans 0-3
        if int(max(self._echans)) < 4:
            ps_select = SelectPointsources(
                limit1550Crab=0.1,
                time_string=f"{self._date:%y%m%d}",
                update=False
            )

            ps_setup = {}

            for ps_name in ps_select.ps_dict.keys():

                ps_setup[ps_name.upper()] = dict(
                    fixed=True,
                    spectrum=dict(
                        pl=dict(
                            spectrum_type="pl",
                            powerlaw_index="swift"
                        )
                    )
                )

            self._ps_dict = ps_select.ps_dict

        else:
            ps_setup = []

        self._config["setup"].update(ps_list=ps_setup)

    def _update_priors(self):

        for delta_days in range(1, 5):

            day_before = self._date - timedelta(days=delta_days)

            job_dir_day_before = os.path.join(
                base_dir,
                f"{day_before:%y%m%d}",
                self._data_type,
                "phys_bkg",
                f"det_{'_'.join(self._detectors)}",
                f"e{'_'.join(self._echans)}",
            )

            result_file = os.path.join(job_dir_day_before, "fit_result.hdf5")

            if os.path.exists(result_file):

                print("daybefore found")

                with h5py.File(result_file, "r") as f:

                    param_names = f.attrs['param_names']

                    best_fit_values = f.attrs["best_fit_values"]

                params = dict(zip(param_names, best_fit_values))

                for param_name, best_fit_value in params.items():

                    param_mean = float('%.3g' % best_fit_value)

                    if param_name == "norm_earth_albedo":
                        self._config["priors"]["earth"] = dict(fixed=dict())
                        self._config["priors"]["earth"]["fixed"]["norm"] = dict(
                            prior="truncated_gaussian",
                            bounds=[0.5E-2, 5.0E-2],
                            gaussian=[param_mean, 0.1]
                        )

                    elif param_name == "norm_cgb":
                        self._config["priors"]["cgb"] = dict(fixed=dict())
                        self._config["priors"]["cgb"]["fixed"]["norm"] = dict(
                            prior="truncated_gaussian",
                            bounds=[4.0E-2, 3.0E-1],
                            gaussian=[param_mean, 0.1]
                        )

                    elif "constant_echan-" in param_name:
                        echan = param_name[-1]
                        self._config["priors"][f"cr_echan-{echan}"] = {}
                        self._config["priors"][f"cr_echan-{echan}"]["const"] = dict(
                            prior="truncated_gaussian",
                            bounds=[1.0E-1, 1.0E+2],
                            gaussian=[param_mean, 0.1]
                        )
                    elif "norm_magnetic_echan-" in param_name:
                        echan = param_name[-1]
                        self._config["priors"][f"cr_echan-{echan}"] = {}
                        self._config["priors"][f"cr_echan-{echan}"]["norm"] = dict(
                            prior="truncated_gaussian",
                            bounds=[1.0E-1, 1.0E+2],
                            gaussian=[param_mean, 0.1]
                        )

                    elif "norm_point_source" in param_name:
                        ps_name = re.search('norm_point_source-(.*?)_pl', param_name).groups()[0]
                        self._config["priors"][f"ps"][ps_name.upper()] = dict(pl=dict())
                        self._config["priors"][f"ps"][ps_name.upper()]["pl"]["norm"] = dict(
                            prior="truncated_gaussian",
                            bounds=[1.0E-4, 1.0E+9],
                            gaussian=[param_mean, 0.1]
                        )

                    elif "eff_area_corr_" in param_name:
                        det_name = re.search('eff_area_corr_(.*?)\b', param_name).groups()[0]
                        self._config["priors"][f"eff_area_correction_{det_name}"] = dict(
                            prior="truncated_gaussian",
                            bounds=[0.8, 1.2],
                            gaussian=[param_mean, 0.01]
                        )

                    elif "norm_saa-" in param_name:
                        pass

                    else:
                        raise Exception(f"Unknown param_name '{param_name}' provided")
                break

    def write_config_file(self, output):

        output().makedirs()

        with output().open(mode="w") as f:
            yaml.dump(self._config, f, default_flow_style=False)
