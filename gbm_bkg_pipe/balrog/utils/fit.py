import os
import shutil
import time

import gbm_drm_gen as drm
import matplotlib.pyplot as plt
import numpy as np
import yaml
from gbm_drm_gen import BALROG_DRM, BALROGLike, DRMGenCTIME

from threeML import (
    OGIPLike,
    DataList,
    Powerlaw,
    Cutoff_powerlaw,
    Broken_powerlaw,
    SmoothlyBrokenPowerLaw,
    Band,
    # Thermal_bremsstrahlung_optical_thin,
    Uniform_prior,
    Log_uniform_prior,
    Model,
    PointSource,
    BayesianAnalysis,
    display_spectrum_model_counts,
)

from gbm_bkg_pipe.utils.file_utils import if_dir_containing_file_not_existing_then_make

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

try:
    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:
        using_mpi = True

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        time.sleep(rank * 0.5)
    else:
        using_mpi = False
except:
    using_mpi = False

gbm_data = os.environ.get("GBMDATA")
base_dir = os.path.join(os.environ.get("GBMDATA"), "bkg_pipe")


class BalrogFit(object):
    def __init__(
        self,
        trigger_name,
        trigger_info_file,
    ):
        """
        Initalize MultinestFit for Balrog
        :param grb_name: Name of GRB
        :param version: Version of data
        :param bkg_fit_yaml_file: Path to bkg fit yaml file
        """
        # Basic input
        self._trigger_name = trigger_name

        # Load yaml information
        with open(trigger_info_file, "r") as f:
            trigger_info = yaml.safe_load(f)

        self._trigger_info = trigger_info

        self._set_plugins()
        self._define_model()

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

            pha_output_dir = os.path.join(
                base_dir,
                self._trigger_info["date"],
                self._trigger_info["data_type"],
                "trigger",
                self._trigger_name,
                "pha",
            )

            ogip_like = OGIPLike(
                f"grb{det}",
                observation=os.path.join(
                    pha_output_dir, f"{self._trigger_name}_{det}.pha"
                ),
                background=os.path.join(
                    pha_output_dir, f"{self._trigger_name}_{det}_bak.pha"
                ),
                response=rsp,
                spectrum_number=1,
            )

            balrog_like = BALROGLike.from_spectrumlike(
                spectrum_like=ogip_like,
                time=self._trigger_info["active_time_start"],
                drm_generator=rsp,
            )

            balrog_like.set_active_measurements("c0-c3")

            fluence_plugins.append(balrog_like)

        self._data_list = DataList(*fluence_plugins)

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
            cpl.K.prior = Log_uniform_prior(lower_bound=1e-3, upper_bound=10 ** 4)
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
        # define bayes object with model and data_list
        self._bayes = BayesianAnalysis(self._model, self._data_list)

        # Create the chains directory
        chains_dir, self._temp_chains_dir = self._create_chains_dir()

        chains_path = os.path.join(self._temp_chains_dir, f"{self._trigger_name}_")

        self._bayes.set_sampler("multinest")

        self._bayes.sampler.setup(n_live_points=800, chain_name=chains_path)

        _ = self._bayes.sample()

    def save_fit_result(self):
        """
        :return:
        """
        fit_result_name = f"{self._trigger_name}_loc_results.fits"
        fit_result_path = os.path.join(
            base_dir,
            self._trigger_info["date"],
            self._trigger_info["data_type"],
            self._trigger_name,
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
            base_dir,
            self._trigger_info["date"],
            self._trigger_info["data_type"],
            self._trigger_name,
            "chains",
        )

        temp_chains_dir = os.path.join(base_dir, "tmp", f"c_{self._trigger_name}")

        if using_mpi:
            if rank == 0:

                # Make chains folder if it does not exists already
                if not os.path.exists(chains_dir):
                    os.makedirs(chains_dir)

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
        os.unlink(self._temp_chains_dir)

    def create_spectrum_plot(self):
        """
        Create the spectral plot to show the fit results for all used dets
        :return:
        """
        plot_name = f"{self._trigger_name}_spectrum_plot.png"
        plot_path = os.path.join(
            base_dir,
            self._trigger_info["date"],
            self._trigger_info["data_type"],
            self._trigger_name,
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
        for d in self._use_dets:
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

                print("No spectral plot possible...")
