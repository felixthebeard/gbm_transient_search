import os
import re
import yaml
import h5py
import numpy as np
from datetime import timedelta

from gbmbkgpy.utils.select_pointsources import SelectPointsources

from gbm_transient_search.utils.configuration import gbm_transient_search_config
from gbm_transient_search.utils.env import get_bool_env_value, get_env_value

simulate = get_bool_env_value("BKG_PIPE_SIMULATE")
data_dir = os.environ.get("GBMDATA")
base_dir = os.path.join(data_dir, "bkg_pipe")
# bkg_source_setup = gbm_transient_search_config["phys_bkg"]["bkg_source_setup"]


class BkgConfigWriter(object):
    def __init__(self, date, data_type, echans, detectors, step="final"):
        self._date = date
        self._data_type = data_type
        self._echans = echans
        self._detectors = detectors
        self._step = step

        self._load_default_config()

    def build_config(self):
        self._update_general()

        self._update_saa_setup()

        self._update_source_setup()

        self._update_ps_setup()

        # self._update_priors()

        self._update_export()

    def mask_triggers(self, trigger_result):

        with open(trigger_result, "r") as f:

            trigger_info = yaml.safe_load(f)

        self._config.update(
            dict(
                mask_intervals=[
                    trigger["interval"] for trigger in trigger_info["triggers"].values()
                ]
            )
        )

    def _load_default_config(self):
        config_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/bkg_model/config_fit.yml"

        # Load the default config file
        with open(config_path) as f:
            self._config = yaml.safe_load(f)

    def _update_general(self):

        general_config = dict(
            general=dict(
                dates=[f"{self._date:%y%m%d}"],
                data_type=self._data_type,
                echans=[echan for echan in self._echans],
                detectors=list(self._detectors),
                min_bin_width=40,
            ),
        )

        # Update the config parameters with fit specific values
        self._config.update(general_config)

    def _update_export(self):

        export_config = dict(
            export=dict(save_unbinned=True, save_whole_day=False),
        )

        # Update the config parameters with fit specific values
        self._config.update(export_config)

    def _update_saa_setup(self):

        saa_config = dict(
            saa=dict(
                time_after_saa=5000,  # 100,
                time_before_saa=50,
                short_time_intervals=False,  # False,  # True,
                nr_decays_per_exit=1,
                decay_at_day_start=True,
                decay_per_detector=False,
                decay_model="exponential",
            )
        )

        self._config.update(saa_config)

    def _update_source_setup(self):
        setup_sources = dict(
            setup=dict(
                use_saa=False,
                use_constant=True,
                use_cr=True,
                use_earth=True,
                use_cgb=True,
                fix_earth=True,
                fix_cgb=True,
                use_sun=False,
                ps_list=[],
                cr_approximation="BGO",
                use_eff_area_correction=False,
            )
        )

        self._config.update(setup_sources)

        if "b0" in self._detectors or "b1" in self._detectors:
            self._config["setup"]["cr_approximation"] = "MCL"

        if simulate:
            self._config["setup"]["cr_approximation"] = "MCL"

    def _update_ps_setup(self):
        # Only inlcude point sources for echans 0-3
        if int(max(self._echans)) < 3:
            ps_select = SelectPointsources(
                limit1550Crab=0.10,
                time_string=f"{self._date:%y%m%d}",
                update=False,
                min_separation_angle=2.0,
            )
            # write the new ps file in the data folder
            filepath_all = os.path.join(
                data_dir,
                "point_sources",
                "ps_all_swift.dat"
            )
            ps_select.write_all_psfile(filepath_all)

            ps_setup = {}

            for ps_name in ps_select.ps_dict.keys():

                ps_setup[ps_name.upper()] = dict(
                    fixed=True,
                    spectrum=dict(pl=dict(spectrum_type="pl", powerlaw_index="swift")),
                )

            self._ps_dict = ps_select.ps_dict

            # self._config["setup"]["use_cr"] = False
            # self._config["setup"]["use_constant"] = False

        else:
            ps_setup = []

        self._config["setup"].update(ps_list=ps_setup)

    def _update_priors(self):

        if self._step == "final":

            job_dir = os.path.join(
                base_dir,
                f"{self._date:%y%m%d}",
                self._data_type,
                "base",
                "phys_bkg",
                f"det_{'_'.join(self._detectors)}",
                f"e{'_'.join(self._echans)}",
            )

            file_name = f"fit_result_{self._date:%y%m%d}_{'-'.join(self._detectors)}_e{'-'.join(self._echans)}.hdf5"

            result_file = os.path.join(job_dir, file_name)

            if os.path.exists(result_file):

                self.build_priors_from_result(result_file)

                return True
        else:

            for delta_days in range(0, 5):

                day_before = self._date - timedelta(days=delta_days)

                job_dir_day_before = os.path.join(
                    base_dir,
                    f"{day_before:%y%m%d}",
                    self._data_type,
                    "final",
                    "phys_bkg",
                    f"det_{'_'.join(self._detectors)}",
                    f"e{'_'.join(self._echans)}",
                )

                file_name = f"fit_result_{day_before:%y%m%d}_{'-'.join(self._detectors)}_e{'-'.join(self._echans)}.hdf5"

                result_file = os.path.join(job_dir_day_before, file_name)

                if os.path.exists(result_file):

                    self.build_priors_from_result(result_file)

                    return True

    def build_priors_from_result(self, result_file, fit_method="stan"):

        with h5py.File(result_file, "r") as f:

            param_names = f.attrs["param_names"]

            best_fit_values = f.attrs["best_fit_values"]

        params = dict(zip(param_names, best_fit_values))

        for param_name, best_fit_value in params.items():

            param_mean = float("%.3g" % best_fit_value)
            log_param_mean = float("%.3g" % np.log(best_fit_value))

            if param_name == "norm_earth_albedo":
                self._config["priors"]["earth"] = dict(fixed=dict())

                if fit_method == "stan":
                    self._config["priors"]["earth"]["fixed"]["norm"] = dict(
                        prior="normal_on_log",
                        gaussian=[log_param_mean, 1],
                        bounds=[0.5e-2, 5.0e-2],
                    )
                elif fit_method == "multinest":
                    self._config["priors"]["earth"]["fixed"]["norm"] = dict(
                        prior="truncated_gaussian",
                        bounds=[0.5e-2, 5.0e-2],
                        gaussian=[param_mean, 1],
                    )
                else:
                    raise Exception("Unknown fit method")

            elif param_name == "norm_cgb":
                self._config["priors"]["cgb"] = dict(fixed=dict())
                if fit_method == "stan":
                    self._config["priors"]["cgb"]["fixed"]["norm"] = dict(
                        prior="normal_on_log",
                        gaussian=[log_param_mean, 1],
                        bounds=[4.0e-2, 3.0e-1],
                    )
                elif fit_method == "multinest":
                    self._config["priors"]["cgb"]["fixed"]["norm"] = dict(
                        prior="truncated_gaussian",
                        bounds=[4.0e-2, 3.0e-1],
                        gaussian=[param_mean, 1],
                    )
                else:
                    raise Exception("Unknown fit method")

            elif re.search("norm_(.*?)_pl", param_name):
                ps_name = re.search("norm_(.*?)_pl", param_name).groups()[0]

                self._config["priors"][f"ps"][ps_name.upper()] = dict(pl=dict())

                if fit_method == "stan":
                    self._config["priors"][f"ps"][ps_name.upper()]["pl"]["norm"] = dict(
                        prior="normal_on_log",
                        gaussian=[log_param_mean, 1],
                        bounds=[1.0e-4, 1.0e9],
                    )
                elif fit_method == "multinest":
                    self._config["priors"][f"ps"][ps_name.upper()]["pl"]["norm"] = dict(
                        prior="truncated_gaussian",
                        bounds=[1.0e-4, 1.0e9],
                        gaussian=[param_mean, 1],
                    )
                else:
                    raise Exception("Unknown fit method")

            elif "eff_area_corr_" in param_name:
                det_name = re.search("eff_area_corr_(.*?)\b", param_name).groups()[0]
                if fit_method == "stan":
                    self._config["priors"][f"eff_area_correction_{det_name}"] = dict(
                        prior="normal_on_log",
                        gaussian=[log_param_mean, 0.01],
                        bounds=[0.8, 1.2],
                    )
                elif fit_method == "multinest":
                    self._config["priors"][f"eff_area_correction_{det_name}"] = dict(
                        prior="truncated_gaussian",
                        bounds=[0.8, 1.2],
                        gaussian=[param_mean, 0.01],
                    )
                else:
                    raise Exception("Unknown fit method")

            elif "norm_saa-" in param_name:
                pass

        # TODO: Use detector specific prior instead of mean over priors
        # this requires adjustment in the background model
        for echan in self._echans:

            detector_mean = np.mean(
                [v for k, v in params.items() if f"norm_constant_echan-{echan}" in k]
            )

            param_mean = float("%.3g" % detector_mean)
            log_param_mean = float("%.3g" % np.log(detector_mean))

            if f"cr_echan-{echan}" not in self._config["priors"].keys():
                self._config["priors"][f"cr_echan-{echan}"] = {}

            if fit_method == "stan":
                self._config["priors"][f"cr_echan-{echan}"]["const"] = dict(
                    prior="normal_on_log",
                    gaussian=[log_param_mean, 1],
                    bounds=[1.0e-1, 1.0e2],
                )
            elif fit_method == "multinest":
                self._config["priors"][f"cr_echan-{echan}"]["const"] = dict(
                    prior="truncated_gaussian",
                    bounds=[1.0e-1, 1.0e2],
                    gaussian=[param_mean, 1],
                )
            else:
                raise Exception("Unknown fit method")

        for echan in self._echans:

            detector_mean = np.mean(
                [v for k, v in params.items() if f"norm_magnetic_echan-{echan}" in k]
            )

            param_mean = float("%.3g" % detector_mean)
            log_param_mean = float("%.3g" % np.log(detector_mean))

            if f"cr_echan-{echan}" not in self._config["priors"].keys():
                self._config["priors"][f"cr_echan-{echan}"] = {}

            if fit_method == "stan":
                self._config["priors"][f"cr_echan-{echan}"]["norm"] = dict(
                    prior="normal_on_log",
                    gaussian=[log_param_mean, 1],
                    bounds=[1.0e-1, 1.0e2],
                )
            elif fit_method == "multinest":
                self._config["priors"][f"cr_echan-{echan}"]["norm"] = dict(
                    prior="truncated_gaussian",
                    bounds=[1.0e-1, 1.0e2],
                    gaussian=[param_mean, 1],
                )
            else:
                raise Exception("Unknown fit method")

    def write_config_file(self, output):

        output().makedirs()

        with output().open(mode="w") as f:
            yaml.dump(self._config, f, default_flow_style=False)


class TableWrapper(object):
    def __init__(self, df):
        self.df = df
        self.generated_quantities = df.to_numpy()
        self.column_names = list(df.columns)
