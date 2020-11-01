import os
import copy
import yaml
import h5py
import numpy as np
from scipy import stats

from gbmbkgpy.utils.saa_calc import SAA_calc
from gbmbkgpy.utils.binner import Rebinner

from gbmgeometry import GBMTime
from gbmgeometry.position_interpolator import slice_disjoint
from trigger_hunter.wbs import WBS
from trigger_hunter.wbs2 import wbs_sdll
from trigger_hunter.utils.stats import poisson_gaussian
from astropy.stats import bayesian_blocks
import ruptures as rpt

from gbm_bkg_pipe.utils.saa_calc import SaaCalc
from gbm_bkg_pipe.utils.plotting import TriggerPlot

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


def distance_mapping(x, ref_vector=None):
    """
    Maps a multi dimensional vector to the length of the vector
    """

    if ref_vector is None:
        ref_vector = np.repeat(1, x.shape[1])

    distance = np.sqrt(np.sum((x - ref_vector) ** 2, axis=1))

    return distance


def angle(x, ref_vector):
    """
    Calculate the separation angle between a vector and a reference vector
    """
    dot_prod = np.dot(x, ref_vector)

    norm1 = np.sqrt(np.sum(x ** 2))
    norm2 = np.sqrt(np.sum(ref_vector ** 2))

    if abs(dot_prod / (norm1 * norm2)) > 1:
        return 0

    else:
        return np.arccos(dot_prod / (norm1 * norm2))


def angle_mapping(x, ref_vector=None):
    """
    Map a multi demensional vector to the separation angle between the vector
    and a reference vector.
    If no reference vector is passed use the unity vector (1, 1, 1  ...)
    """
    if ref_vector is None:
        ref_vector = np.repeat(1, x.shape[1])

    x_ang = np.apply_along_axis(angle, axis=1, arr=x, ref_vector=ref_vector)

    return x_ang


def calc_snr(data, background):
    """
    Calculate the naive signale to noise ratio, without taking the errors
    of the background estimation into account.
    """
    snr = np.sum(data - background) / np.sqrt(np.sum(background))

    return snr


def calc_significance(data, background, bkg_stat_err):
    """
    Calculate the significance of a signal with poisson noise
    over background with gaussian erros.
    """
    significance = poisson_gaussian.significance(
        n=np.sum(data), b=np.sum(background), sigma=np.sqrt(np.sum(bkg_stat_err ** 2))
    )

    return significance


class Search(object):
    """
    Transient search class.
    This finds change points in the time series,
    calculates the significances of source intervals,
    and filters them to construct triggers.
    """

    def __init__(
        self, result_file, min_bin_width, mad=False, sub_min=False, bad_fit_threshold=50
    ):
        """
        Instantiate the search class and prepare the data for processing.
        """

        self._load_result_file(result_file)

        self._dets_idx = []

        for det in self._detectors:
            self._dets_idx.append(valid_det_names.index(det))

        # Calculate new saa mask to fix stan fit
        saa_calc = SaaCalc(self._time_bins)
        self._saa_mask = saa_calc.saa_mask

        self._rebinn_data(min_bin_width)

        self._mask_bad_bkg_fits(bad_fit_threshold)

        # Combine all energy bins for significance calculation
        self._combine_energy_bins()

        # Clean data
        self._counts_cleaned = (
            self._rebinned_observed_counts[self._rebinned_saa_mask]
            - self._rebinned_bkg_counts[self._rebinned_saa_mask]
        )

        self._data_cleaned = (
            self._counts_cleaned.T
            / self._rebinned_time_bin_width[self._rebinned_saa_mask]
        ).T

        self._transform_data(mad, sub_min)

    def _rebinn_data(self, min_bin_width):
        """
        Rebinn the observed data and background
        """
        self._data_rebinner = Rebinner(
            self._time_bins, min_bin_width, mask=self._saa_mask
        )

        self._rebinned_time_bins = self._data_rebinner.time_rebinned

        self._rebinned_saa_mask = self._data_rebinner.rebinned_saa_mask

        self._rebinned_observed_counts = self._data_rebinner.rebin(
            self._observed_counts
        )[0]

        self._rebinned_bkg_counts = self._data_rebinner.rebin(self._bkg_counts)[0]

        self._rebinned_bkg_stat_err = self._data_rebinner.rebin_errors(
            self._bkg_stat_err
        )[0]

        self._valid_slices = slice_disjoint(self._rebinned_saa_mask)

        self._rebinned_time_bin_width = np.diff(self._rebinned_time_bins, axis=1)[:, 0]
        self._rebinned_mean_time = np.mean(self._rebinned_time_bins, axis=1)

    def _mask_bad_bkg_fits(
        self, max_med_sig=50, max_neg_med_sig=-30, max_neg_sig=-100, n_parts=10
    ):
        """
        Mask energy channels that have a very high significance for the
        interval covering the whole day.
        This usually happens when the background underfits strongly,
        and reduces the false triggers on bad background fits.
        """

        good_bkg_fit_mask = np.zeros((14, 8), dtype=bool)

        part_idx = np.linspace(0, self._observed_counts.shape[0], n_parts, dtype=int)

        for det in self._detectors:

            det_idx = valid_det_names.index(det)

            for e in self._echans:

                sigs = []

                for i in range(n_parts - 1):

                    sig = calc_significance(
                        self._observed_counts[
                            part_idx[i] : part_idx[i + 1] + 1, det_idx, e
                        ],
                        self._bkg_counts[part_idx[i] : part_idx[i + 1] + 1, det_idx, e],
                        self._bkg_stat_err[
                            part_idx[i] : part_idx[i + 1] + 1, det_idx, e
                        ],
                    )
                    sigs.append(sig)

                sig = calc_significance(
                    self._observed_counts[:, det_idx, e],
                    self._bkg_counts[:, det_idx, e],
                    self._bkg_stat_err[:, det_idx, e],
                )

                median_sig = np.median(sigs)

                good = True
                if sig <= max_neg_sig or median_sig <= max_neg_med_sig:
                    good = False
                if median_sig >= max_med_sig:
                    good = False

                if good:
                    good_bkg_fit_mask[det_idx, e] = True

        self._good_bkg_fit_mask = good_bkg_fit_mask

    def _combine_energy_bins(self, echans=[0, 1, 2, 3]):
        """
        Combine the energy bins that are used for the calculation of the significance
        """
        data = {}
        background = {}
        bkg_stat_err = {}

        if echans is None:
            echans = self._echans
            echan_mask = copy.deepcopy(self._good_bkg_fit_mask)

        else:
            # Build the mask of the echans to combine
            e_mask = np.zeros(8, dtype=bool)
            e_mask[echans] = True

            # Get the echans with a good bkg fit and combin with e_mask
            det_echan_mask = copy.deepcopy(self._good_bkg_fit_mask)
            det_echan_mask[:, ~e_mask] = False

        for det in self._detectors:
            det_idx = valid_det_names.index(det)

            echan_mask = det_echan_mask[det_idx, :]

            # Combine all energy channels for the calculation of the significance
            data[det] = self._rebinned_observed_counts[
                self._rebinned_saa_mask, det_idx, :
            ][:, echan_mask].sum(axis=1)
            background[det] = self._rebinned_bkg_counts[
                self._rebinned_saa_mask, det_idx, :
            ][:, echan_mask].sum(axis=1)
            bkg_stat_err[det] = np.sqrt(
                np.sum(
                    self._rebinned_bkg_stat_err[self._rebinned_saa_mask, det_idx, :][
                        :, echan_mask
                    ]
                    ** 2,
                    axis=1,
                )
            )

        self._observed_counts_total = data
        self._bkg_counts_total = background
        self._bkg_stat_err_total = bkg_stat_err

    def _transform_data(self, mad, sub_min):
        """
        Transform the data to and apply mapping
        """
        self._data_flattened = self._data_cleaned[:, self._good_bkg_fit_mask].reshape(
            (self._data_cleaned.shape[0], -1)
        )

        if mad:

            med_abs_div = stats.median_abs_deviation(self._data_flattened, axis=1)
            median = np.median(self._data_cleaned[:, :, 0], axis=1)

            med_abs_div = med_abs_div.reshape(
                (-1,) + (1,) * (self._data_flattened.ndim - 1)
            )
            median = median.reshape((-1,) + (1,) * (self._data_flattened.ndim - 1))

            self._data_trans = (self._data_flattened - median) / med_abs_div

        else:
            self._data_trans = self._data_flattened

        if sub_min:
            min_x = np.min(self._data_trans, axis=1).reshape(
                (-1,) + (1,) * (self._data_trans.ndim - 1)
            )

            self._data_trans = self._data_trans - min_x

        self._distances = distance_mapping(self._data_trans)
        self._angles = angle_mapping(self._data_trans)

    def _load_result_file(self, result_file):
        """
        Load result file from background fit
        """
        with h5py.File(result_file, "r") as f:

            dates = f.attrs["dates"]

            trigger = f.attrs["trigger"]

            trigger_time = f.attrs["trigger_time"]

            data_type = f.attrs["data_type"]

            echans = f.attrs["echans"]

            detectors = f.attrs["detectors"]

            time_bins = f["time_bins"][()]

            saa_mask = f["saa_mask"][()]

            observed_counts = f["observed_counts"][()]

            model_counts = f["model_counts"][()]

            stat_err = f["stat_err"][()]

        self._dates = dates
        self._detectors = detectors
        self._echans = np.array([int(echan) for echan in echans])
        self._data_type = data_type
        self._time_bins = time_bins
        self._saa_mask = saa_mask
        self._observed_counts = observed_counts
        self._bkg_counts = model_counts
        self._bkg_stat_err = stat_err

    def find_changepoints_angles(self, indiv_echans=True, **kwargs):
        """
        Find changepoints applying the ruptures PELT method
        in the angles time series
        """
        change_points_sections = []
        change_points = []

        if indiv_echans:

            for section, valid_slice in enumerate(self._valid_slices):

                cpts_seg = []

                for echan in self._echans:
                    data_flattend = self._data_cleaned[:, self._dets_idx][
                        :, :, echan
                    ].reshape((self._data_cleaned.shape[0], -1))

                    angles = angle_mapping(data_flattend)

                    angle_slice = angles[valid_slice[0] : valid_slice[1]]

                    penalty = 2 * np.log(len(angle_slice))

                    algo_ang = rpt.Pelt(**kwargs).fit(angle_slice)
                    cpts_seg.extend(algo_ang.predict(pen=penalty))

                change_points_sections.append(cpts_seg)
                change_points.append(cpts_seg + valid_slice[0])

        else:

            for section, valid_slice in enumerate(self._valid_slices):

                cpts_seg = []

                angle_slice = self._angles[valid_slice[0] : valid_slice[1]]

                penalty = 2 * np.log(len(angle_slice))

                algo_ang = rpt.Pelt(**kwargs).fit(angle_slice)
                cpts_seg.extend(algo_ang.predict(pen=penalty))

                change_points_sections.append(cpts_seg)
                change_points.append(cpts_seg + valid_slice[0])

        change_points_sections = np.array(change_points_sections)
        change_points = np.array(change_points)

        self._cpts_sections_all = change_points_sections
        self._change_points_all = change_points

    def calc_significances(self, required_significance=5, max_interval_time=1000):
        """
        Combine two arbitrary changepoints to a source interval and calculate the significance.
        If the segnificance is higher than the threshold, then add this source interval to the intervals list.
        This is treating each seaction (between SAA passages) individually.
        """
        intervals = {}
        significances = {}

        for det in self._detectors:
            det_idx = valid_det_names.index(det)

            intervals[det] = []
            significances[det] = []

            for segment in range(len(self._change_points_all)):

                for idx_low in self._change_points_all[segment]:

                    for idx_high in self._change_points_all[segment]:

                        if idx_low != idx_high and idx_low < idx_high:

                            # If the interval is longer than the max interval, use the max interval length
                            # this the does not use a change point as interval stop, but avoids discarding
                            # very smooth signals that only have a change point at the beginnen and the end
                            # of the long interval
                            if (
                                self._rebinned_mean_time[idx_high]
                                - self._rebinned_mean_time[idx_low]
                            ) > max_interval_time:

                                idx_high = np.argmax(
                                    self._rebinned_mean_time
                                    > (
                                        self._rebinned_mean_time[idx_low]
                                        + max_interval_time
                                    )
                                )

                            significance = calc_significance(
                                data=self._observed_counts_total[det][idx_low:idx_high],
                                background=self._bkg_counts_total[det][
                                    idx_low:idx_high
                                ],
                                bkg_stat_err=self._bkg_stat_err_total[det][
                                    idx_low:idx_high
                                ],
                            )

                            if (
                                not np.isnan(significance)
                                and significance > required_significance
                            ):

                                intervals[det].append([idx_low, idx_high])

                                significances[det].append(significance)

            intervals[det] = np.array(intervals[det])
            significances[det] = np.array(significances[det])

            sort_idx = significances[det].argsort()[::-1]

            intervals[det] = intervals[det][sort_idx]
            significances[det] = significances[det][sort_idx]

        self._intervals_all = intervals
        self._intervals_significance_all = significances

    def build_trigger_information(self):
        """
        Build the trigger information by filtering the found source intervals
        for overlap and same trigger times.
        At the end check if the selected active time is significant.
        """

        # Filter out overlapping intervals and select the one with highest significance
        # This is done twice to get rid of some leftovers
        intervals_selected, significances_selected = self._filter_overlap(
            self._intervals_all, self._intervals_significance_all
        )
        intervals_selected, significances_selected = self._filter_overlap(
            intervals_selected, significances_selected
        )

        # intervals_selected, significances_selected = self._filter_overlap_dets(
        #     intervals_selected, significances_selected
        # )

        # Get all intervals that are significant in at least one detector
        trigger_intervals = []
        for i, det in enumerate(self._detectors):
            trigger_intervals.extend(intervals_selected[det])

        trigger_intervals = np.unique(trigger_intervals, axis=0)

        # Get the significance of all detectors for the trigger intervals
        trigger_significance = []
        for interval in trigger_intervals:

            sig_dict = {}

            for i, det in enumerate(self._detectors):
                sig_dict[det] = float(
                    calc_significance(
                        data=self._observed_counts_total[det][
                            interval[0] : interval[1]
                        ],
                        background=self._bkg_counts_total[det][
                            interval[0] : interval[1]
                        ],
                        bkg_stat_err=self._bkg_stat_err_total[det][
                            interval[0] : interval[1]
                        ],
                    )
                )

            trigger_significance.append(sig_dict)

        trigger_times = self._rebinned_time_bins[self._rebinned_saa_mask][
            trigger_intervals[:, 0], 0
        ]

        most_significant_detectors = []
        for sig in trigger_significance:
            most_significant_detectors.append(max(sig, key=sig.get))

        trigger_peak_times = []

        for i, inter in enumerate(trigger_intervals):
            det = most_significant_detectors[i]
            counts = (
                self._observed_counts_total[det][inter[0] : inter[1]]
                - self._bkg_counts_total[det][inter[0] : inter[1]]
            )

            max_index = np.argmax(counts) + inter[0]

            trigger_peak_times.append(
                self._rebinned_time_bins[self._rebinned_saa_mask][max_index, 0]
            )

        # Filter out duplicate peak times and keep the shorter intervals
        unique_peak_ids = self._filter_duplicate_peaks(
            trigger_peak_times, trigger_intervals
        )

        significant_ids = self._filter_active_time_significance(
            valid_ids=unique_peak_ids,
            most_significant_detectors=np.array(most_significant_detectors)[
                unique_peak_ids
            ],
            peak_times=np.array(trigger_peak_times)[unique_peak_ids],
            required_significance=3,
        )

        self._trigger_intervals = np.array(trigger_intervals)[significant_ids]
        self._trigger_significance = np.array(trigger_significance)[significant_ids]
        self._trigger_most_significant_detector = np.array(most_significant_detectors)[
            significant_ids
        ]
        self._trigger_times = trigger_times[significant_ids]
        self._trigger_peak_times = np.array(trigger_peak_times)[significant_ids]

        self._tr_p = trigger_peak_times

        self._intervals = intervals_selected
        self._intervals_significance = significances_selected

    def _filter_overlap(self, intervals, significances):
        """
        Filter overlapping source intervals and select the one with
        the highest significance.
        """
        intervals_selected = {}
        significances_selected = {}

        for det in self._detectors:
            max_ids = []

            intervals_selected[det] = []
            significances_selected[det] = []

            # Iterate over all intervals of this detector
            for id1, interval1 in enumerate(intervals[det]):

                overlapping_idx = np.where(
                    np.logical_and(
                        interval1[0] <= intervals[det][:, 1],
                        intervals[det][:, 0] <= interval1[1],
                    )
                )[0]

                max_id = significances[det][overlapping_idx].argmax()

                # Get max id in original array
                max_id = max_id + overlapping_idx[max_id]

                max_ids.append(max_id)

            if len(max_ids) > 0:

                max_ids = np.unique(max_ids)

                intervals_selected[det] = intervals[det][max_ids]
                significances_selected[det] = significances[det][max_ids]

            else:
                intervals_selected[det] = []
                significances_selected[det] = []

        return intervals_selected, significances_selected

    def _filter_overlap_dets(self, intervals, significances):
        intervals_selected = {}
        significances_selected = {}

        # Filter out overlapping intervals and select the one with highest significance
        for det in self._detectors:
            max_ids = []

            intervals_selected[det] = []
            significances_selected[det] = []

            # Iterate over all intervals of this detector
            for id1, interval1 in enumerate(intervals[det]):

                is_max = True

                for det2 in self._detectors:

                    # Iterate over all intervals of this detector
                    for id2, interval2 in enumerate(intervals[det2]):

                        if (
                            interval1[0] <= interval2[1]
                            and interval2[0] <= interval1[1]
                        ):

                            if significances[det2][id2] > significances[det][id1]:

                                is_max = False
                if is_max:
                    max_ids.append(id1)

            if len(max_ids) > 0:

                max_ids = np.unique(max_ids)

                intervals_selected[det] = intervals[det][max_ids]
                significances_selected[det] = significances[det][max_ids]

            else:
                intervals_selected[det] = []
                significances_selected[det] = []

        return intervals_selected, significances_selected

    def _filter_duplicate_peaks(self, peak_times, trigger_intervals):
        """
        Filter intervals that lead to the same peak times and keep the shorter one.
        This gets rid of significant intervals that span very large times and not just the source.
        """

        interval_lengths = trigger_intervals[:, 1] - trigger_intervals[:, 0]

        min_length_ids = []

        for id1, peak_time in enumerate(peak_times):

            same_peak_ids = np.where(peak_times == peak_time)[0]

            min_length_id = interval_lengths[same_peak_ids].argmin()

            min_length_ids.append(same_peak_ids[min_length_id])

        min_length_ids = np.unique(min_length_ids)

        return min_length_ids

    def _filter_active_time_significance(
        self, valid_ids, most_significant_detectors, peak_times, required_significance=5
    ):
        """
        Filter the intervals by the significance of the active time
        selected arround the trigger time.
        """

        significant_ids = []

        for i, val_id in enumerate(valid_ids):

            start_time = peak_times[i] - 10
            stop_time = peak_times[i] + 10

            idx_low = np.where(self._rebinned_time_bins[:, 0] >= start_time)[0][0]
            idx_high = np.where(self._rebinned_time_bins[:, 0] <= stop_time)[0][-1]

            det = most_significant_detectors[i]
            print(det)

            significance = calc_significance(
                data=self._observed_counts_total[det][idx_low:idx_high],
                background=self._bkg_counts_total[det][idx_low:idx_high],
                bkg_stat_err=self._bkg_stat_err_total[det][idx_low:idx_high],
            )

            if significance >= required_significance:
                significant_ids.append(val_id)

        return significant_ids

    def create_result_dict(self):
        """
        Create the trigger result dictionary.
        """

        good_bkg_mask = dict()

        for det_idx, det in enumerate(valid_det_names):
            good_bkg_mask[det] = self._good_bkg_fit_mask[det_idx, :].tolist()

        trigger_information = {
            "dates": self._dates.tolist(),
            "data_type": self._data_type,
            "echans": self._echans.tolist(),
            "detectors": self._detectors.tolist(),
            "good_bkg_fit_mask": good_bkg_mask,
            "triggers": {},
        }

        for i, t0 in enumerate(self._trigger_times):
            t0 = self._trigger_peak_times[i]

            gbm_time = GBMTime.from_MET(t0)
            date_str = gbm_time.time.datetime.strftime("%y%m%d")
            day_fraction = f"{round(gbm_time.time.mjd % 1, 3):.3f}"[2:]

            trigger_name = f"TRG{date_str}{day_fraction}"
            peak_time = self._trigger_peak_times[i] - t0

            t_info = {
                "date": date_str,
                "trigger_name": trigger_name,
                "trigger_time": t0.tolist(),
                "trigger_time_utc": gbm_time.utc,
                "peak_time": peak_time.tolist(),
                "significances": self._trigger_significance[i],
                "interval": {
                    "start": self._rebinned_mean_time[self._trigger_intervals][i][
                        0
                    ].tolist(),
                    "stop": self._rebinned_mean_time[self._trigger_intervals][i][
                        1
                    ].tolist(),
                },
                "most_significant_detector": self._trigger_most_significant_detector.tolist()[
                    i
                ],
            }

            trigger_information["triggers"][trigger_name] = t_info

        self._trigger_information = trigger_information

    def plot_results(self, output_dir):

        plotter = TriggerPlot(
            triggers=self._trigger_information["triggers"],
            time=self._rebinned_mean_time,
            counts=self._rebinned_observed_counts,
            detectors=self._detectors,
            echans=self._echans,
            bkg_counts=self._rebinned_bkg_counts,
            counts_cleaned=self._counts_cleaned,
            saa_mask=self._rebinned_saa_mask,
            good_bkg_fit_mask=self._good_bkg_fit_mask,
            angles=self._angles,
            show_all_echans=True,
            show_angles=True,
        )

        plotter.create_overview_plots(output_dir)

        plotter.save_plot_data(output_dir)

    def save_result(self, output_path):
        # output_file = os.path.join(os.path.dirname(output_path), "trigger_information.yml")
        with open(output_path, "w") as f:
            yaml.dump(self._trigger_information, f, default_flow_style=False)

    def run_bayesian_blocks(self, **kwargs):

        echan_idx = 0

        change_points = {}

        self._block_edges = {}

        for det in self._detectors:

            det_idx = valid_det_names.index(det)

            self._block_edges[det] = []

            print(f"Calculate change points for {det}")

            change_points[det] = []

            for section, valid_slice in enumerate(self._valid_slices):

                data_slice = self._data_cleaned[
                    valid_slice[0] : valid_slice[1], det_idx, echan_idx
                ]

                error_slice = self._bkg_stat_err_rebinned[self._saa_mask][
                    valid_slice[0] : valid_slice[1], det_idx, echan_idx
                ]

                time_slice = self._rebinned_mean_time[self._saa_mask][
                    valid_slice[0] : valid_slice[1]
                ]

                block_edges = bayesian_blocks(
                    t=time_slice,
                    x=data_slice,
                    sigma=error_slice,
                    fitness="measures",
                    **kwargs,
                )

                cpts_seg = []

                for edge in block_edges:
                    cpt = np.where(
                        np.logical_and(
                            self._data.time_bins[:, 0] < edge,
                            self._data.time_bins[:, 1] > edge,
                        )
                    )

                    cpts_seg.extend(cpt[0])

                change_points[det].append(cpts_seg)

                self._block_edges[det].extend(block_edges)

            change_points[det] = np.array(change_points[det])

        self._change_points = change_points

    def find_change_points_raptures(
        self, max_nr_cpts=5, method="binseg", model="l2", **kwargs
    ):

        change_points_sections = {}
        change_points = {}

        echan_idx = 0

        for det in self._detectors:

            det_idx = valid_det_names.index(det)

            print(f"Calculate change points for {det}")

            change_points_sections[det] = []
            change_points[det] = []

            for section, valid_slice in enumerate(self._valid_slices):

                data_slice = self._data_cleaned[
                    valid_slice[0] : valid_slice[1], det_idx, echan_idx
                ]

                penalty = 2 * np.log(len(data_slice))

                if method == "binseg":
                    algo = rpt.Binseg(model=model, **kwargs).fit(data_slice)
                    cpts_seg = algo.predict(pen=penalty)

                elif method == "pelt":
                    algo = rpt.Pelt(model=model, **kwargs).fit(data_slice)
                    cpts_seg = algo.predict(pen=penalty)

                elif method == "dynp":
                    algo = rpt.Dynp(model=model, **kwargs).fit(data_slice)
                    cpts_seg = algo.predict(pen=penalty)

                elif method == "bottomup":
                    algo = rpt.BottomUp(model=model, **kwargs).fit(data_slice)
                    cpts_seg = algo.predict(pen=penalty)

                elif method == "window":
                    algo = rpt.Window(model=model, **kwargs).fit(data_slice)
                    cpts_seg = algo.predict(pen=penalty)

                else:
                    raise KeyError("Invalid method selected")

                change_points_sections[det].append(cpts_seg)
                change_points[det].append(cpts_seg + valid_slice[0])

            change_points_sections[det] = np.array(change_points_sections[det])
            change_points[det] = np.array(change_points[det])

        self._cpts_sections = change_points_sections
        self._change_points = change_points

    def find_change_points_wbs(self, wbs2=False, max_nr_cpts=5, nr_segments=5000):

        change_points_sections = {}
        change_points = {}

        random_cumsums = {}

        echan_idx = 0

        for det in self._detectors:

            det_idx = valid_det_names.index(det)

            print(f"Calculate change points for {det}")

            change_points_sections[det] = []
            change_points[det] = []
            random_cumsums[det] = []

            for section, valid_slice in enumerate(self._valid_slices):

                data_slice = self._data_cleaned[
                    valid_slice[0] : valid_slice[1], det_idx, echan_idx
                ]

                if wbs2:

                    est_seg, no_of_cpts_seg, cpts_seg, rand_cumsums_seg = wbs_sdll(
                        x=data_slice
                    )

                else:

                    wbs = WBS()

                    est_seg, no_of_cpts_seg, cpts_seg, rand_cumsums_seg = wbs.run(
                        x=data_slice, nr_segments=nr_segments, max_nr_cpts=max_nr_cpts
                    )

                change_points_sections[det].append(cpts_seg)

                change_points[det].append(cpts_seg + valid_slice[0])

                rand_cumsums_seg[:, 2] += valid_slice[0]

                random_cumsums[det].append(rand_cumsums_seg)

            change_points_sections[det] = np.array(change_points_sections[det])
            change_points[det] = np.array(change_points[det])

        self._cpts_sections = change_points_sections
        self._change_points = change_points
        self._random_cumsums = random_cumsums

    def get_significant_regions(self, snr=False, required_significance=5):

        wbs_intervals = {}
        wbs_significance = {}
        wbs_intervals_sorted = {}
        wbs_significance_sorted = {}

        echan_idx = 0
        echan = self._echans[0]

        for det in self._detectors:

            det_idx = valid_det_names.index(det)

            print(f"Calculate significance of intervals for {det}")

            wbs_intervals[det] = []
            wbs_significance[det] = []

            for segment in range(len(self._change_points[det])):

                for idx_low in self._change_points[det][segment]:

                    for idx_high in self._change_points[det][segment]:

                        if idx_low != idx_high and idx_low < idx_high:

                            if snr:
                                significance = calc_snr(
                                    data=self._rebinned_observed_counts[
                                        self._rebinned_saa_mask, det_idx, echan_idx
                                    ][idx_low:idx_high],
                                    background=self._rebinned_bkg_counts[
                                        self._rebinned_saa_mask, det_idx, echan
                                    ][idx_low:idx_high],
                                )

                            else:
                                significance = calc_significance(
                                    data=self._rebinned_observed_counts[
                                        self._rebinned_saa_mask, det_idx, echan_idx
                                    ][idx_low:idx_high],
                                    background=self._rebinned_bkg_counts[
                                        self._rebinned_saa_mask, det_idx, echan
                                    ][idx_low:idx_high],
                                    bkg_stat_err=self._rebinned_bkg_stat_err[
                                        self._rebinned_saa_mask, det_idx, echan
                                    ][idx_low:idx_high],
                                )

                            if (
                                not np.isnan(significance)
                                and significance > required_significance
                            ):

                                wbs_intervals[det].append([idx_low, idx_high])

                                wbs_significance[det].append(significance)

            wbs_intervals[det] = np.array(wbs_intervals[det])
            wbs_significance[det] = np.array(wbs_significance[det])

            sort_idx = wbs_significance[det].argsort()[::-1]

            wbs_intervals_sorted[det] = wbs_intervals[det][sort_idx]
            wbs_significance_sorted[det] = wbs_significance[det][sort_idx]

        self._intervals_sorted = wbs_intervals_sorted
        self._intervals_significance_sorted = wbs_significance_sorted

    def correlate_detectors(self):

        snr_mask = np.zeros(len(self._rebinned_mean_time[self._rebinned_saa_mask]))

        snr_mask_dets = np.zeros(len(self._rebinned_mean_time[self._rebinned_saa_mask]))

        for det in self._detectors:

            det_idx = valid_det_names.index(det)

            snr_mask_det = np.zeros(
                len(self._rebinned_mean_time[self._rebinned_saa_mask])
            )

            for interval in self._intervals_sorted[det]:
                snr_mask[interval[0] : interval[1]] += 1

                snr_mask_det[interval[0] : interval[1]] += 1

            # Avoid counting intervals multiple times if they overlap
            snr_mask_det = np.clip(snr_mask_det, 0, 1)

            snr_mask_dets += snr_mask_det

        self._significant_mask_dets = snr_mask_dets
