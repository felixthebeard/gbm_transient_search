from mpi4py import MPI

using_mpi = True

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import yaml
from gbm_drm_gen import BALROG_DRM, BALROGLike, DRMGenCTIME

warnings.simplefilter("ignore")
from threeML import Band  # Thermal_bremsstrahlung_optical_thin,
from threeML import (
    BayesianAnalysis,
    Blackbody,
    Broken_powerlaw,
    Cutoff_powerlaw,
    DataList,
    Gaussian,
    Log_normal,
    Log_uniform_prior,
    Model,
    OGIPLike,
    PointSource,
    Powerlaw,
    SmoothlyBrokenPowerLaw,
    Uniform_prior,
    display_spectrum_model_counts,
)


def sanitize_filename(filename, abspath=False):
    sanitized = os.path.expandvars(os.path.expanduser(filename))

    if abspath:

        return os.path.abspath(sanitized)

    else:

        return sanitized


def if_dir_containing_file_not_existing_then_make(filename):
    """
    If the given directory does not exists, then make it
    If basename of path contains a '.' we assume it is a file and check the parent dir
    :param filename: directory to check or make
    :return: None
    """

    sanitized_directory = sanitize_filename(filename)

    if "." in os.path.basename(sanitized_directory):
        sanitized_directory = os.path.dirname(sanitized_directory)

    if not os.path.exists(sanitized_directory):
        os.makedirs(sanitized_directory)


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

gbm_data = os.environ.get("GBMDATA")
base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")


class BalrogFit(object):
    def __init__(
        self,
        trigger_name,
        trigger_info_file,
        trigger_dir,
    ):
        """
        Initalize MultinestFit for Balrog
        :param grb_name: Name of GRB
        :param version: Version of data
        :param bkg_fit_yaml_file: Path to bkg fit yaml file
        """
        # Basic input
        self._trigger_name = trigger_name
        self._trigger_dir = trigger_dir

        # Load yaml information
        with open(trigger_info_file, "r") as f:
            trigger_info = yaml.safe_load(f)

        self._trigger_info = trigger_info

        self._good_bkg_fit_mask = self._trigger_info["good_bkg_fit_mask"]

        self._set_plugins()
        self._define_model(spectrum=self._trigger_info.get("spectral_model", "cpl"))

    def _set_plugins(self):
        """
        Set the plugins
        :return:
        """
        fluence_plugins = []

        for det in self._trigger_info["use_dets"]:

            ctime_file = os.path.join(
                gbm_data,
                "ctime",
                self._trigger_info["date"],
                f"glg_ctime_{det}_{self._trigger_info['date']}_v00.pha",
            )

            poshist_file = os.path.join(
                gbm_data,
                "poshist",
                f"glg_poshist_all_{self._trigger_info['date']}_v00.fit",
            )

            drm_gen = DRMGenCTIME(
                ctime_file=ctime_file,
                time=self._trigger_info["active_time_start"],
                poshist=poshist_file,
                T0=self._trigger_info["trigger_time"],
                mat_type=2,
                occult=True,
            )

            rsp = BALROG_DRM(drm_gen, 0, 0)

            ogip_like = OGIPLike(
                f"grb{det}",
                observation=os.path.join(
                    self._trigger_dir, "pha", f"{self._trigger_name}_{det}.pha"
                ),
                background=os.path.join(
                    self._trigger_dir, "pha", f"{self._trigger_name}_{det}_bak.pha"
                ),
                response=rsp,
                spectrum_number=1,
            )

            balrog_like = BALROGLike.from_spectrumlike(
                spectrum_like=ogip_like,
                time=self._trigger_info["active_time_start"],
                drm_generator=rsp,
            )

            fit_echans = self._get_valid_echans(det)

            balrog_like.set_active_measurements(*fit_echans)

            fluence_plugins.append(balrog_like)

        self._data_list = DataList(*fluence_plugins)

    def _get_valid_echans(self, det):

        fit_echans = []

        start_echan = None

        for e, fit_good in enumerate(self._good_bkg_fit_mask[det]):

            if fit_good:

                if start_echan is None:

                    start_echan = e

                elif e == 7:
                    stop_echan = e
                    echan_str = f"c{start_echan}-c{stop_echan}"
                    fit_echans.append(echan_str)
                    start_echan = None

            else:

                if start_echan is not None:
                    stop_echan = e - 1

                    if start_echan < stop_echan:
                        echan_str = f"c{start_echan}-c{stop_echan}"

                        fit_echans.append(echan_str)

                    start_echan = None
        print(fit_echans)
        return fit_echans

    def _define_model(self, spectrum="cpl"):
        """
        Define a Model for the fit
        :param spectrum: Which spectrum type should be used (cpl, band, pl, sbpl or solar_flare)
        """
        # data_list=comm.bcast(data_list, root=0)
        if spectrum == "cpl":
            # we define the spectral model
            cpl = Cutoff_powerlaw()
            cpl.K.max_value = 10 ** 4
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)
            cpl.xc.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            cpl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("GRB_cpl_", 0.0, 0.0, spectral_shape=cpl))

        elif spectrum == "band":

            band = Band()
            band.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1200)
            band.alpha.set_uninformative_prior(Uniform_prior)
            band.xp.prior = Log_uniform_prior(lower_bound=10, upper_bound=1e4)
            band.beta.set_uninformative_prior(Uniform_prior)

            self._model = Model(PointSource("GRB_band", 0.0, 0.0, spectral_shape=band))

        elif spectrum == "pl":

            pl = Powerlaw()
            pl.K.max_value = 10 ** 4
            pl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10 ** 4)
            pl.index.set_uninformative_prior(Uniform_prior)
            # we define a point source model using the spectrum we just specified
            self._model = Model(PointSource("GRB_pl", 0.0, 0.0, spectral_shape=pl))

        elif spectrum == "sbpl":

            sbpl = SmoothlyBrokenPowerLaw()
            sbpl.K.min_value = 1e-5
            sbpl.K.max_value = 1e4
            sbpl.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e4)
            sbpl.alpha.set_uninformative_prior(Uniform_prior)
            sbpl.beta.set_uninformative_prior(Uniform_prior)
            sbpl.break_energy.min_value = 1
            sbpl.break_energy.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
            self._model = Model(PointSource("GRB_sbpl", 0.0, 0.0, spectral_shape=sbpl))

        elif spectrum == "blackbody":
            blackbody = Blackbody()
            blackbody.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e6)
            blackbody.kT.min_value = 1e-5
            blackbody.kT.max_value = 5e2
            blackbody.kT.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=5e2)
            # blackbody.K.prior = Log_normal(mu=-15, sigma=1)
            # blackbody.kT.prior = Log_normal(mu=-15, sigma=1)
            # blackbody.kT.prior = Gaussian(mu=3, sigma=5)

            self._model = Model(
                PointSource("GRB_blackbody", 0.0, 0.0, spectral_shape=blackbody)
            )

        # elif spectrum == "solar_flare":

        #     # broken powerlaw
        #     bpl = Broken_powerlaw()
        #     bpl.K.max_value = 10 ** 5
        #     bpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10 ** 5)
        #     bpl.xb.prior = Log_uniform_prior(lower_bound=1, upper_bound=1e4)
        #     bpl.alpha.set_uninformative_prior(Uniform_prior)
        #     bpl.beta.set_uninformative_prior(Uniform_prior)

        #     # thermal brems
        #     tb = Thermal_bremsstrahlung_optical_thin()
        #     tb.K.max_value = 1e5
        #     tb.K.min_value = 1e-5
        #     tb.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=10 ** 5)
        #     tb.kT.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)
        #     tb.Epiv.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=1e4)

        #     # combined
        #     total = bpl + tb

        #     self._model = Model(
        #         PointSource("Solar_flare", 0.0, 0.0, spectral_shape=total)
        #     )
        else:
            raise Exception("Use valid model type: cpl, pl, sbpl, band or solar_flare")

    def fit(self):
        """
        Fit the model to data using multinest
        :return:
        """
        print(self._model.parameters)
        print(self._model.free_parameters)
        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)

        # Create the chains directory
        chains_dir, self._temp_chains_dir = self._create_chains_dir()

        chains_path = os.path.join(self._temp_chains_dir, f"{self._trigger_name}_")

        self._bayes.set_sampler("multinest", share_spectrum=True)

        self._bayes.sampler.setup(n_live_points=800, chain_name=chains_path)

        _ = self._bayes.sample()

    def save_fit_result(self):
        """
        :return:
        """
        fit_result_name = f"{self._trigger_name}_loc_results.fits"
        fit_result_path = os.path.join(
            self._trigger_dir,
            fit_result_name,
        )

        if using_mpi:
            if rank == 0:
                self._bayes.restore_median_fit()
                self._bayes.results.write_to(fit_result_path, overwrite=True)

        else:
            self._bayes.restore_median_fit()
            self._bayes.results.write_to(fit_result_path, overwrite=True)

    def _create_chains_dir(self):
        """
        Create chains directory and symbolic link for MultiNest
        :return:
        """
        chains_dir = os.path.join(
            self._trigger_dir,
            "chains",
        )

        temp_chains_dir = os.path.join(base_dir, "tmp", f"c_{self._trigger_name}")

        if using_mpi:
            if rank == 0:

                # Make chains folder if it does not exists already
                if not os.path.exists(chains_dir):
                    os.makedirs(chains_dir)

                if not os.path.exists(os.path.dirname(temp_chains_dir)):
                    os.makedirs(os.path.dirname(temp_chains_dir))

                # Create a temp symbolic link with shorter path for MultiNest
                if not os.path.exists(temp_chains_dir):
                    os.symlink(chains_dir, temp_chains_dir)

        else:
            # Make chains folder if it does not exists already
            if not os.path.exists(chains_dir):
                os.makedirs(chains_dir)

            # Create a temp symbolic link with shorter path for MultiNest
            if not os.path.exists(temp_chains_dir):
                os.symlink(chains_dir, temp_chains_dir)

        return chains_dir, temp_chains_dir

    def unlink_temp_chains_dir(self):
        # Remove the symbolic link
        if using_mpi:
            if rank == 0:
                if os.path.exists(self._temp_chains_dir):
                    os.unlink(self._temp_chains_dir)
        else:
            if os.path.exists(self._temp_chains_dir):
                os.unlink(self._temp_chains_dir)

    def create_spectrum_plot(self):
        """
        Create the spectral plot to show the fit results for all used dets
        :return:
        """
        plot_name = (
            f"{self._trigger_name}_spectrum_plot_{self._trigger_info['data_type']}.png"
        )
        plot_path = os.path.join(
            self._trigger_dir,
            "plots",
            plot_name,
        )

        color_dict = {
            "n0": "#FF9AA2",
            "n1": "#FFB7B2",
            "n2": "#FFDAC1",
            "n3": "#E2F0CB",
            "n4": "#B5EAD7",
            "n5": "#C7CEEA",
            "n6": "#DF9881",
            "n7": "#FCE2C2",
            "n8": "#B3C8C8",
            "n9": "#DFD8DC",
            "na": "#D2C1CE",
            "nb": "#6CB2D1",
            "b0": "#58949C",
            "b1": "#4F9EC4",
        }

        color_list = []
        for d in self._trigger_info["use_dets"]:
            color_list.append(color_dict[d])

        set = plt.get_cmap("Set1")
        color_list = set.colors

        if using_mpi:
            if rank == 0:

                if_dir_containing_file_not_existing_then_make(plot_path)

                try:
                    spectrum_plot = display_spectrum_model_counts(
                        self._bayes, data_colors=color_list, model_colors=color_list
                    )

                    spectrum_plot.savefig(plot_path, bbox_inches="tight")

                except:

                    try:
                        spectrum_plot = display_spectrum_model_counts(
                            self._bayes,
                            data_colors=color_list,
                            model_colors=color_list,
                            min_rate=-99,
                        )

                        spectrum_plot.savefig(plot_path, bbox_inches="tight")

                    except Exception as e:
                        print("No spectral plot possible...")
                        print(e)

        else:

            if_dir_containing_file_not_existing_then_make(plot_path)

            try:
                spectrum_plot = display_spectrum_model_counts(
                    self._bayes, data_colors=color_list, model_colors=color_list
                )

                spectrum_plot.savefig(plot_path, bbox_inches="tight")

            except:

                try:
                    spectrum_plot = display_spectrum_model_counts(
                        self._bayes,
                        data_colors=color_list,
                        model_colors=color_list,
                        min_rate=-99,
                    )

                    spectrum_plot.savefig(plot_path, bbox_inches="tight")

                except Exception as e:
                    print("No spectral plot possible...")
                    print(e)


if __name__ == "__main__":
    import argparse

    ############## Argparse for parsing bash arguments ################
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-name",
        "--trigger_name",
        type=str,
        help="Name of one trigger to localize",
        required=False,
    )

    parser.add_argument(
        "-tinfo",
        "--trigger_info",
        type=str,
        help="Path to the information file of one trigger",
        required=False,
    )

    parser.add_argument(
        "-tinfos",
        "--multi_trigger_info",
        type=str,
        help="Path to the file containing multiple trigger informations",
        required=False,
    )

    parser.add_argument(
        "-subtasks",
        "--subtasks",
        type=int,
        help="Number of subtasks",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        help="Index of the slurm task to split the workload.",
        required=False,
    )

    args = parser.parse_args()

    # Run balrog for a single trigger
    if args.trigger_name is not None:

        assert args.trigger_info is not None

        trigger_dir = os.path.dirname(args.trigger_info)

        # get fit object
        multinest_fit = BalrogFit(args.trigger_name, args.trigger_info, trigger_dir)

        multinest_fit.fit()
        multinest_fit.save_fit_result()
        multinest_fit.create_spectrum_plot()
        multinest_fit.unlink_temp_chains_dir()

        success_file = os.path.join(
            os.path.dirname(args.trigger_info), f"{args.trigger_name}_balrog.success"
        )
        if rank == 0:
            os.system(f"touch {success_file}")

    # Run balrog in a loop for all triggers in the trigger information file
    else:

        assert args.multi_trigger_info is not None

        with open(args.multi_trigger_info, "r") as f:

            trigger_information = yaml.safe_load(f)

        trigger_names = trigger_information.keys()

        if args.subtasks is not None:
            assert args.index is not None

            # Split the list of trigger_names in subtasks and run the subset.
            trigger_names = np.array_split(trigger_names, args.subtasks)[
                args.index
            ].tolist()

        # Now run balrog for each trigger
        for trigger_name in trigger_names:

            t_info_file = os.path.join(
                os.path.dirname(args.multi_trigger_info),
                trigger_name,
                "trigger_info.yml",
            )

            trigger_dir = os.path.join(
                os.path.dirname(args.multi_trigger_info), trigger_name
            )

            # get fit object
            multinest_fit = BalrogFit(trigger_name, t_info_file, trigger_dir)

            multinest_fit.fit()
            multinest_fit.save_fit_result()
            multinest_fit.create_spectrum_plot()
            multinest_fit.unlink_temp_chains_dir()

            success_file = os.path.join(
                os.path.dirname(args.multi_trigger_info),
                trigger_name,
                f"{trigger_name}_balrog.success",
            )
            if rank == 0:
                os.system(f"touch {success_file}")
