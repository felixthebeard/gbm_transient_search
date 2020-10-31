#!/usr/bin/env python3
import os

import h5py
import numpy as np
import yaml
from gbmgeometry import GBMTime
from matplotlib import cm
from matplotlib import pyplot as plt

valid_det_names = [
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
]


class TriggerPlot(object):
    def __init__(
        self,
        triggers,
        time,
        counts,
        bkg_counts,
        counts_cleaned,
        saa_mask,
        good_bkg_fit_mask,
        detectors,
        echans,
        angles=None,
        show_counts=True,
        show_counts_cleaned=True,
        show_all_echans=True,
        show_angles=True,
    ):
        self._triggers = triggers
        self._time = time
        self._saa_mask = saa_mask
        self._counts = counts
        self._echans = echans
        self._detectors = detectors
        self._bkg_counts = bkg_counts
        self._counts_cleaned = counts_cleaned
        self._angles = angles

        self._good_bkg_fit_mask = good_bkg_fit_mask

        self._show_counts = show_counts
        self._show_counts_cleaned = show_counts_cleaned
        self._show_all_echans = show_all_echans
        self._show_angles = show_angles

        self._nr_subplots = 0

        if show_counts:
            self._nr_subplots += len(self._echans)

        if show_counts_cleaned:
            self._nr_subplots += 1

        if show_all_echans:
            self._nr_subplots += 1

        if show_angles:
            self._nr_subplots += 1

    @classmethod
    def from_hdf5(cls, trigger_yaml, data_path):

        with open(trigger_yaml, "r") as f:

            triggers = yaml.safe_load(f)["triggers"]

        with h5py.File(data_path, "r") as f:

            echans = f.attrs["echans"]

            detectors = f.attrs["detectors"]

            time = f["time"][()]

            saa_mask = f["saa_mask"][()]

            counts = f["counts"][()]

            bkg_counts = f["bkg_counts"][()]

            angles = f["angles"][()]

            good_bkg_fit_mask = f["good_bkg_fit_mask"][()]

        counts_cleaned = counts - bkg_counts

        return cls(
            triggers,
            time,
            counts,
            bkg_counts,
            counts_cleaned,
            saa_mask,
            good_bkg_fit_mask,
            detectors,
            echans,
            angles,
        )

    def save_plot_data(self, outdir):
        output_path = os.path.join(outdir, "trigger", "plot_data.hdf5")

        with h5py.File(output_path, "w") as f:
            f.attrs["echans"] = self._echans
            f.attrs["detectors"] = self._detectors

            f.create_dataset(
                "time",
                data=self._time,
                compression="lzf",
            )

            f.create_dataset(
                "saa_mask",
                data=self._saa_mask,
                compression="lzf",
            )

            f.create_dataset(
                "counts",
                data=self._counts,
                compression="lzf",
            )

            f.create_dataset(
                "bkg_counts",
                data=self._bkg_counts,
                compression="lzf",
            )

            f.create_dataset(
                "angles",
                data=self._angles,
                compression="lzf",
            )

            f.create_dataset(
                "good_bkg_fit_mask",
                data=self._good_bkg_fit_mask,
                compression="lzf",
            )

    def create_overview_plots(self, outdir=None):

        self.create_day_overview(outdir)

        self.create_day_overview_cleaned(outdir)

    def create_trigger_plots(self, trigger_name, outdir=None):

        trigger = self._triggers[trigger_name]

        self.create_individual_plots(trigger, outdir)

        self.create_individual_overview_plots(trigger, outdir)

        self.create_lightcurves(trigger, outdir)

    def create_day_overview(self, outdir=None):
        echans = [0, 1, 2]
        ndets = 12

        nechans = len(echans)
        n_subplots = ndets * nechans

        date_utc_str = GBMTime.from_MET(self._time[100]).utc[:10]

        day_start_met = GBMTime.from_UTC_fits(f"{date_utc_str}T00:00:00").met

        fig, ax = plt.subplots(n_subplots, 1, sharex=True, figsize=[10, 30])

        cm_subsection = np.linspace(0.0, 1.0, len(self._triggers.values()))
        colors = [cm.jet(x) for x in cm_subsection]

        for i in range(ndets):
            if i < ndets:
                ax[i * nechans + 0].spines["bottom"].set_color("#dddddd")
                ax[i * nechans + 1].spines["top"].set_color("#dddddd")
                ax[i * nechans + 1].spines["bottom"].set_color("#dddddd")
                ax[i * nechans + 2].spines["top"].set_color("#dddddd")

            for e in echans:

                good_fit = self._good_bkg_fit_mask[i, e]

                if good_fit:
                    data_color = "black"
                else:
                    data_color = "lightcoral"

                ax[i * nechans + e].scatter(
                    (self._time - day_start_met) / (60 * 60),
                    self._counts[:, i, e],
                    alpha=0.9,
                    linewidth=0.5,
                    s=2,
                    facecolors="none",
                    edgecolors=data_color,
                )

                ax[i * nechans + e].plot(
                    (self._time - day_start_met) / (60 * 60),
                    self._bkg_counts[:, i, e],
                    label="bkg model",
                    color="red",
                    linewidth=1,
                )

                if e == 1:
                    ax[i * nechans + e].set_ylabel(f"Det {valid_det_names[i]} \n e{e}")
                else:
                    ax[i * nechans + e].set_ylabel(f"e{e}")

                ymin = np.percentile(self._counts[:, i, e], 0, axis=0)
                ymax = np.percentile(self._counts[:, i, e], 99.9, axis=0)
                ymin_bkg = np.percentile(self._bkg_counts[:, i, e], 0, axis=0)
                ymax_bkg = np.percentile(self._bkg_counts[:, i, e], 99.9, axis=0)

                ymin = min(ymin, ymin_bkg)
                ymax = max(ymax, ymax_bkg)

                ymin *= 0.9
                ymax *= 1.1

                ax[i * nechans + e].set_ylim(ymin, ymax)

                ax[i * nechans + e].margins(x=0)

                for color_idx, trigger in enumerate(self._triggers.values()):

                    ax[i * nechans + e].axvline(
                        (trigger["trigger_time"] - day_start_met) / (60 * 60),
                        label=f"{trigger['trigger_name']} | {trigger['trigger_time_utc']}",
                        ymin=0,
                        ymax=1,
                        c=colors[color_idx],
                        linewidth=1,
                        zorder=0,
                        clip_on=False,
                    )

        # ax[0].legend()
        lgd = ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        ax[-1].set_xlabel(f"{date_utc_str} | Time(UTC)")
        fig.subplots_adjust(hspace=0)

        if outdir is not None:

            plot_dir = os.path.join(outdir, "trigger")

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            savepath = os.path.join(plot_dir, f"{date_utc_str}_triggers.png")

            fig.savefig(
                savepath, dpi=300, bbox_extra_artists=(lgd,), bbox_inches="tight"
            )

    def create_day_overview_cleaned(self, outdir=None):
        echans = [0, 1, 2]
        ndets = 12

        nechans = len(echans)
        n_subplots = ndets * nechans

        date_utc_str = GBMTime.from_MET(self._time[100]).utc[:10]

        day_start_met = GBMTime.from_UTC_fits(f"{date_utc_str}T00:00:00").met

        fig, ax = plt.subplots(n_subplots, 1, sharex=True, figsize=[10, 30])

        cm_subsection = np.linspace(0.0, 1.0, len(self._triggers.values()))
        colors = [cm.jet(x) for x in cm_subsection]

        for i in range(ndets):
            if i < ndets:
                ax[i * nechans + 0].spines["bottom"].set_color("#dddddd")
                ax[i * nechans + 1].spines["top"].set_color("#dddddd")
                ax[i * nechans + 1].spines["bottom"].set_color("#dddddd")
                ax[i * nechans + 2].spines["top"].set_color("#dddddd")

            for e in echans:

                good_fit = self._good_bkg_fit_mask[i, e]

                if good_fit:
                    data_color = "black"
                else:
                    data_color = "lightcoral"

                ax[i * nechans + e].scatter(
                    (self._time[self._saa_mask] - day_start_met) / (60 * 60),
                    self._counts_cleaned[:, i, e],
                    alpha=0.9,
                    linewidth=0.5,
                    s=2,
                    facecolors="none",
                    edgecolors=data_color,
                )

                if e == 1:
                    ax[i * nechans + e].set_ylabel(f"Det {valid_det_names[i]} \n e{e}")
                else:
                    ax[i * nechans + e].set_ylabel(f"e{e}")

                ymin = np.percentile(self._counts_cleaned[:, i, e], 0, axis=0)
                ymax = np.percentile(self._counts_cleaned[:, i, e], 99.9, axis=0)

                ymin *= 1.1
                ymax *= 1.1

                ax[i * nechans + e].set_ylim(ymin, ymax)

                ax[i * nechans + e].margins(x=0)

                for color_idx, trigger in enumerate(self._triggers.values()):

                    ax[i * nechans + e].axvline(
                        (trigger["trigger_time"] - day_start_met) / (60 * 60),
                        label=f"{trigger['trigger_name']} | {trigger['trigger_time_utc']}",
                        ymin=0,
                        ymax=1,
                        c=colors[color_idx],
                        linewidth=1,
                        zorder=0,
                        clip_on=False,
                    )

        # ax[0].legend()
        lgd = ax[0].legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        ax[-1].set_xlabel(f"{date_utc_str} | Time(UTC)")
        fig.subplots_adjust(hspace=0)

        if outdir is not None:

            plot_dir = os.path.join(outdir, "trigger")

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            savepath = os.path.join(plot_dir, f"{date_utc_str}_triggers_cleaned.png")

            fig.savefig(
                savepath, dpi=300, bbox_extra_artists=(lgd,), bbox_inches="tight"
            )

    def create_individual_plots(self, trigger, outdir=None):
        fontsize = 8

        det_idx = valid_det_names.index(trigger["most_significant_detector"])

        fig, ax = plt.subplots(self._nr_subplots, 1, sharex=True, figsize=[6.4, 10])

        time_mask = np.logical_and(
            self._time[self._saa_mask] > trigger["interval"]["start"] - 1000,
            self._time[self._saa_mask] < trigger["interval"]["stop"] + 1000,
        )

        i = -1

        if self._show_counts:

            for e in self._echans:
                i += 1

                good_fit = self._good_bkg_fit_mask[det_idx, e]

                if good_fit:
                    data_color = "black"
                else:
                    data_color = "lightcoral"

                ax[i].scatter(
                    self._time[self._saa_mask][time_mask],
                    self._counts[:, det_idx, e][self._saa_mask][time_mask],
                    alpha=0.9,
                    linewidth=0.5,
                    s=2,
                    facecolors="none",
                    edgecolors=data_color,
                )

                ax[i].plot(
                    self._time[self._saa_mask][time_mask],
                    self._bkg_counts[:, det_idx, e][self._saa_mask][time_mask],
                    label="Bkg model",
                    color="red",
                    linewidth=1,
                )

                ax[i].axvspan(
                    trigger["interval"]["start"],
                    trigger["interval"]["stop"],
                    alpha=0.1,
                    color="blue",
                    label="Trigger region",
                )

                ax[i].axvspan(
                    trigger["trigger_time"] - 10,
                    trigger["trigger_time"] + 10,
                    alpha=0.4,
                    color="orange",
                    label="Selection",
                )

                ax[i].axvline(
                    x=trigger["trigger_time"],
                    ymin=-1.2,
                    ymax=1,
                    c="green",
                    linewidth=1,
                    zorder=0,
                    clip_on=False,
                    label="Peak counts",
                )

                ax[i].set_ylabel(f"Counts e{e}", fontsize=fontsize)

        ax[0].legend()

        if self._show_counts_cleaned:
            i += 1
            ax[i].plot(
                self._time[self._saa_mask][time_mask],
                self._counts_cleaned[:, det_idx, :][time_mask],
            )

            ax[i].axvspan(
                trigger["interval"]["start"],
                trigger["interval"]["stop"],
                alpha=0.1,
                color="blue",
                label="Trigger region",
            )
            ax[i].axvspan(
                trigger["trigger_time"] - 10,
                trigger["trigger_time"] + 10,
                alpha=0.4,
                color="orange",
                label="Selection",
            )

            ax[i].set_ylabel("Cleaned", fontsize=fontsize)

        if self._show_all_echans:
            i += 1
            ax[i].plot(
                self._time[self._saa_mask][time_mask],
                np.sum(
                    self._counts_cleaned[:, det_idx, :][
                        :, self._good_bkg_fit_mask[det_idx, :]
                    ][time_mask],
                    axis=1,
                ),
            )
            ax[i].axvline(
                x=trigger["trigger_time"],
                ymin=0,
                ymax=1.2,
                c="green",
                linewidth=1,
                zorder=0,
                clip_on=False,
            )

            ax[i].axvspan(
                trigger["interval"]["start"],
                trigger["interval"]["stop"],
                alpha=0.1,
                color="blue",
            )

            ax[i].axvspan(
                trigger["trigger_time"] - 10,
                trigger["trigger_time"] + 10,
                alpha=0.4,
                color="orange",
                label="Selection",
            )

            ax[i].set_ylabel("Cleaned combined", fontsize=fontsize)

        if self._show_angles:
            i += 1

            ax[i].plot(
                self._time[self._saa_mask][time_mask],
                self._angles[time_mask],
            )

            ax[i].axvspan(
                trigger["interval"]["start"],
                trigger["interval"]["stop"],
                alpha=0.1,
                color="blue",
            )

            ax[i].axvspan(
                trigger["trigger_time"] - 10,
                trigger["trigger_time"] + 10,
                alpha=0.4,
                color="orange",
                label="Selection",
            )

            ax[i].axvline(
                x=trigger["trigger_time"],
                ymin=0,
                ymax=1.2,
                c="green",
                linewidth=1,
                zorder=0,
                clip_on=False,
            )

            ax[i].set_ylabel("Angles", fontsize=fontsize)

        # fig.tight_layout()

        ax[0].set_title(
            f"Trigger {trigger['trigger_name']} | Det {trigger['most_significant_detector']}"
        )

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective
        fig.subplots_adjust(hspace=0)

        if outdir is not None:

            plot_dir = os.path.join(
                outdir,
                "trigger",
                trigger["trigger_name"],
                "plots",
            )

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            savepath = os.path.join(plot_dir, f"{trigger['trigger_name']}.png")

            fig.savefig(savepath, dpi=300)

    def _choose_dets(self, max_det):
        """
        Function to automatically choose the detectors which should be used in the fit
        :return:
        """
        side_1_dets = ["n0", "n1", "n2", "n3", "n4", "n5"]  # , "b0"]
        side_2_dets = ["n6", "n7", "n8", "n9", "na", "nb"]  # , "b1"]

        # only use the detectors on the same side as the detector with the most significance
        if max_det in side_1_dets:

            use_dets = side_1_dets

        else:
            use_dets = side_2_dets

        return use_dets

    def create_individual_overview_plots(self, trigger, outdir=None):
        fontsize = 8

        use_dets = self._choose_dets(trigger["most_significant_detector"])

        fig, ax = plt.subplots(
            self._nr_subplots - 1,
            len(use_dets),
            sharex=True,
            figsize=[6.4 * len(use_dets), 10],
        )

        time_mask = np.logical_and(
            self._time[self._saa_mask] > trigger["interval"]["start"] - 1000,
            self._time[self._saa_mask] < trigger["interval"]["stop"] + 1000,
        )

        for d, det in enumerate(use_dets):
            det_idx = valid_det_names.index(det)

            i = -1

            if self._show_counts:

                for e in self._echans:
                    i += 1

                    good_fit = self._good_bkg_fit_mask[det_idx, e]

                    if good_fit:
                        data_color = "black"
                    else:
                        data_color = "lightcoral"

                    ax[i, d].scatter(
                        self._time[self._saa_mask][time_mask],
                        self._counts[:, det_idx, e][self._saa_mask][time_mask],
                        alpha=0.9,
                        linewidth=0.5,
                        s=2,
                        facecolors="none",
                        edgecolors=data_color,
                    )

                    ax[i, d].plot(
                        self._time[self._saa_mask][time_mask],
                        self._bkg_counts[:, det_idx, e][self._saa_mask][time_mask],
                        label="Bkg model",
                        color="red",
                        linewidth=1,
                    )

                    ax[i, d].axvspan(
                        trigger["interval"]["start"],
                        trigger["interval"]["stop"],
                        alpha=0.1,
                        color="blue",
                        label="Trigger region",
                    )

                    ax[i, d].axvspan(
                        trigger["trigger_time"] - 10,
                        trigger["trigger_time"] + 10,
                        alpha=0.4,
                        color="orange",
                        label="Selection",
                    )

                    ax[i, d].axvline(
                        x=trigger["trigger_time"],
                        ymin=-1.2,
                        ymax=1,
                        c="green",
                        linewidth=1,
                        zorder=0,
                        clip_on=False,
                        label="Peak counts",
                    )

                    ax[i, 0].set_ylabel(f"Counts e{e}", fontsize=fontsize)

            if self._show_counts_cleaned:
                i += 1
                ax[i, d].plot(
                    self._time[self._saa_mask][time_mask],
                    self._counts_cleaned[:, det_idx, :][time_mask],
                )

                ax[i, d].axvspan(
                    trigger["interval"]["start"],
                    trigger["interval"]["stop"],
                    alpha=0.1,
                    color="blue",
                    label="Trigger region",
                )
                ax[i, d].axvspan(
                    trigger["trigger_time"] - 10,
                    trigger["trigger_time"] + 10,
                    alpha=0.4,
                    color="orange",
                    label="Selection",
                )

                ax[i, d].set_ylabel("Cleaned", fontsize=fontsize)

            if self._show_all_echans:
                i += 1
                ax[i, d].plot(
                    self._time[self._saa_mask][time_mask],
                    np.sum(
                        self._counts_cleaned[:, det_idx, :][
                            :, self._good_bkg_fit_mask[det_idx, :]
                        ][time_mask],
                        axis=1,
                    ),
                )
                ax[i, d].axvline(
                    x=trigger["trigger_time"],
                    ymin=0,
                    ymax=1.2,
                    c="green",
                    linewidth=1,
                    zorder=0,
                    clip_on=False,
                )

                ax[i, d].axvspan(
                    trigger["interval"]["start"],
                    trigger["interval"]["stop"],
                    alpha=0.1,
                    color="blue",
                )

                ax[i, d].axvspan(
                    trigger["trigger_time"] - 10,
                    trigger["trigger_time"] + 10,
                    alpha=0.4,
                    color="orange",
                    label="Selection",
                )

                ax[i, d].set_ylabel("Cleaned combined", fontsize=fontsize)

            ax[0, d].set_title(f"Trigger {trigger['trigger_name']} | Det {det}")
        # fig.tight_layout()

        ax[0, 0].legend()

        # Now remove the space between the two subplots
        # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective
        fig.subplots_adjust(hspace=0)

        if outdir is not None:

            plot_dir = os.path.join(
                outdir,
                "trigger",
                trigger["trigger_name"],
                "plots",
            )

            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)

            savepath = os.path.join(plot_dir, f"{trigger['trigger_name']}_overview.png")

            fig.savefig(savepath, dpi=300)

    def create_lightcurves(self, trigger, outdir=None):
        fontsize = 8

        for det_idx, det in enumerate(valid_det_names):

            fig, ax = plt.subplots(len(self._echans), 1, sharex=True, figsize=[6.4, 10])

            time_mask = np.logical_and(
                self._time[self._saa_mask] > trigger["interval"]["start"] - 500,
                self._time[self._saa_mask] < trigger["interval"]["stop"] + 500,
            )

            i = -1

            for e in self._echans:
                i += 1

                good_fit = self._good_bkg_fit_mask[det_idx, e]

                if good_fit:
                    data_color = "black"
                else:
                    data_color = "darkgray"

                ax[i].scatter(
                    self._time[self._saa_mask][time_mask] - trigger["trigger_time"],
                    self._counts[:, det_idx, e][self._saa_mask][time_mask],
                    alpha=0.9,
                    linewidth=0.5,
                    s=2,
                    facecolors="none",
                    edgecolors=data_color,
                )

                ax[i].plot(
                    self._time[self._saa_mask][time_mask] - trigger["trigger_time"],
                    self._bkg_counts[:, det_idx, e][self._saa_mask][time_mask],
                    label="Bkg model",
                    color="red",
                    linewidth=1,
                )

                ax[i].axvspan(
                    -10,
                    10,
                    alpha=0.4,
                    color="orange",
                    label="Selection",
                )

                ax[i].axvline(
                    x=0,
                    ymin=-1.2,
                    ymax=1,
                    c="green",
                    linewidth=1,
                    zorder=0,
                    clip_on=False,
                    label="Peak counts",
                )

                ax[i].set_ylabel(f"Counts e{e}", fontsize=fontsize)

            ax[0].legend()

            ax[0].set_title(
                f"Trigger {trigger['trigger_name']} | T0={trigger['trigger_time']} ({trigger['trigger_time_utc']}) | Det {det}"
            )

            # Now remove the space between the two subplots
            # NOTE: this must be placed *after* tight_layout, otherwise it will be ineffective
            fig.subplots_adjust(hspace=0)

            if outdir is not None:

                plot_dir = os.path.join(
                    outdir,
                    "trigger",
                    trigger["trigger_name"],
                    "plots",
                    "lightcurves",
                )

                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                savepath = os.path.join(
                    plot_dir,
                    f"{trigger['trigger_name']}_lightcurve_detector_{det}_plot.png",
                )

                fig.savefig(savepath, dpi=300)
