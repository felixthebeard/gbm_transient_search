#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

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
        self._bkg_counts = bkg_counts
        self._counts_cleaned = counts_cleaned
        self._angles = angles

        self._show_counts = show_counts
        self._show_counts_cleaned = show_counts_cleaned
        self._show_all_echans = show_all_echans
        self._show_angles = show_angles

        self._nr_subplots = 0

        if show_counts:
            self._nr_subplots += 8

        if show_counts_cleaned:
            self._nr_subplots += 1

        if show_all_echans:
            self._nr_subplots += 1

        if show_angles:
            self._nr_subplots += 1

    def create_plots(self, outdir):
        fontsize = 8

        for trigger in self._triggers:

            fig, ax = plt.subplots(self._nr_subplots, 1, sharex=True, figsize=[6.4, 10])

            det_idx = valid_det_names.index(trigger["most_significant_detector"])

            time_mask = np.logical_and(
                self._time[self._saa_mask] > trigger["interval"]["start"] - 1000,
                self._time[self._saa_mask] < trigger["interval"]["stop"] + 1000,
            )

            i = -1

            if self._show_counts:

                for e in range(8):
                    i += 1

                    ax[i].scatter(
                        self._time[self._saa_mask][time_mask],
                        self._counts[:, det_idx, e][self._saa_mask][time_mask],
                        alpha=0.9,
                        linewidth=0.5,
                        s=2,
                        facecolors="none",
                        edgecolors="black",
                    )

                    ax[i].plot(
                        self._time[self._saa_mask][time_mask],
                        self._bkg_counts[:, det_idx, e][self._saa_mask][time_mask],
                        label="bkg model",
                        color="red",
                        linewidth=1,
                    )

                    ax[i].axvspan(
                        trigger["interval"]["start"],
                        trigger["interval"]["stop"],
                        alpha=0.1,
                        color="blue",
                        label="trigger region",
                    )

                    ax[i].axvline(
                        x=trigger["trigger_time"],
                        ymin=-1.2,
                        ymax=1,
                        c="green",
                        linewidth=1,
                        zorder=0,
                        clip_on=False,
                        label="peak counts",
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
                    label="trigger region",
                )

                ax[i].set_ylabel("Cleaned", fontsize=fontsize)

            if self._show_all_echans:
                i += 1
                ax[i].plot(
                    self._time[self._saa_mask][time_mask],
                    np.sum(self._counts_cleaned[:, det_idx, :][time_mask], axis=1),
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
