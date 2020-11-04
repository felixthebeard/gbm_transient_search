#!/usr/bin/env python3

import numpy as np
from gbmbkgpy.simulation.simulator import BackgroundSimulator
from gbmbkgpy.utils.progress_bar import progress_bar


class TransientSimulator(BackgroundSimulator):
    """
    A Transient Simulator that simulates the background of GBM and adds transient sources.
    """

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
