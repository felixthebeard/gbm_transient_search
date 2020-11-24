#!/usr/bin/env python3

import h5py
import numpy as np
from gbmbkgpy.simulation.simulator import BackgroundSimulator
from gbmbkgpy.utils.progress_bar import progress_bar


class TransientSimulator(BackgroundSimulator):
    """
    A Transient Simulator that simulates the background of GBM and adds transient sources.
    """

    def __init__(self, *args, **kwargs):
        self._observed_counts = None
        self._observed_counts_raw = None
        self._bkg_counts = None

        super(TransientSimulator, self).__init__(*args, **kwargs)

    def run(self):

        self._simulate_background()

        self._simulate_transients()

        counts_detectors = {}

        for det_idx, det in enumerate(self._valid_det_names):

            counts_detectors[det] = (
                self._counts_background[det] + self._counts_transients[det]
            )

        self._counts_detectors = counts_detectors

    def _simulate_transients(self):

        counts_transients = {}

        with progress_bar(
            len(self._valid_det_names),
            title="Simulate transient sources for all 12 NaI detectors:",
        ) as p:
            for det_idx, det in enumerate(self._valid_det_names):

                counts_sum = np.zeros((len(self._time_bins), len(self._echans)))

                # Simulate Transients
                if self._config.get("use_transients", False):

                    for transient in self._config["sources"]["transient_sources"]:

                        counts_sum += self._simulate_transient(
                            det_idx=det_idx,
                            ra=transient["ra"],
                            dec=transient["dec"],
                            spectrum=transient["spectrum"],
                            time_evolution=transient["time_evolution"],
                        )

                counts_transients[det] = counts_sum

                p.increase()

        self._counts_transients = counts_transients

    def _simulate_transient(self, det_idx, ra, dec, spectrum, time_evolution):

        transient_const_counts = self._simulate_pointsource(det_idx, ra, dec, spectrum)

        if time_evolution["model"] == "norris":

            time_evol = self._get_norris_pulse(
                time_bins=self._time_bins,
                t_start=time_evolution["t_start"],
                t_rise=time_evolution["t_rise"],
                t_decay=time_evolution["t_decay"],
            )

        elif time_evolution["model"] == "step_function":

            time_evol = self._get_step_function(
                time_bins=self._time_bins,
                t_start=time_evolution["t_start"],
                duration=time_evolution["duration"],
            )

        else:
            raise NotImplemented("The selected shape model is not implemented!")

        transient_counts = transient_const_counts * time_evol

        return transient_counts

    def _get_norris_pulse(self, time_bins, t_start, t_rise, t_decay, norm=1.0):

        time_bin_means = np.mean(time_bins, axis=1)

        t_start += time_bin_means[0]

        idx_start = time_bin_means > t_start

        t = time_bin_means[idx_start] - t_start

        out = np.zeros((len(time_bin_means), 1))

        out[idx_start, 0] = (
            norm
            * np.exp(2 * np.sqrt(t_rise / t_decay))
            * np.exp(-t_rise / t - t / t_decay)
        )

        return out

    def _get_step_function(self, time_bins, t_start, duration):

        time_bin_means = np.mean(time_bins, axis=1)

        t_start += time_bin_means[0]

        idx_source = np.logical_and(
            time_bin_means > t_start, time_bin_means < t_start + duration
        )

        out = np.zeros((len(time_bin_means), 1))

        out[idx_source, 0] = 1

        return out

    @property
    def observed_counts(self):
        if self._observed_counts is None:

            self._observed_counts = np.zeros((len(self._time_bins), 14, 8))

            for det_idx, det in enumerate(self._valid_det_names):

                self._observed_counts[:, det_idx, :] = np.random.poisson(
                    self._counts_detectors[det]
                )

        return self._observed_counts

    @property
    def observed_counts_raw(self):
        if self._observed_counts_raw is None:

            self._observed_counts_raw = np.zeros((len(self._time_bins), 14, 8))

            for det_idx, det in enumerate(self._valid_det_names):

                self._observed_counts_raw[:, det_idx, :] = self._counts_detectors[det]

        return self._observed_counts_raw

    @property
    def bkg_counts(self):
        if self._bkg_counts is None:

            self._bkg_counts = np.zeros((len(self._time_bins), 14, 8))

            for det_idx, det in enumerate(self._valid_det_names):

                self._bkg_counts[:, det_idx, :] = self._counts_background[det]

        return self._bkg_counts

    @property
    def stat_err(self):
        return self._stat_err

    def save_combined_hdf5(self, output_path):

        with h5py.File(output_path, "w") as f:

            f.attrs["dates"] = [self._day]

            f.attrs["trigger"] = "None"

            f.attrs["trigger_time"] = 0.0

            f.attrs["data_type"] = self._data_type

            f.attrs["echans"] = list(range(8))

            f.attrs["detectors"] = self._valid_det_names

            f.create_dataset(
                "time_bins",
                data=self._time_bins,
                compression="lzf",
            )

            f.create_dataset(
                "saa_mask",
                data=np.ones(len(self._time_bins)),
                compression="lzf",
            )

            f.create_dataset(
                "observed_counts",
                data=self.observed_counts,
                compression="lzf",
            )

            f.create_dataset(
                "model_counts",
                data=self.bkg_counts,
                compression="lzf",
            )

            f.create_dataset("stat_err", data=self.stat_err, compression="lzf")
