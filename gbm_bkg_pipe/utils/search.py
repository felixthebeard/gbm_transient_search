import h5py
import numpy as np
from scipy import stats

from gbmbkgpy.utils.saa_calc import SAA_calc
from gbmbkgpy.utils.binner import Rebinner

from gbmgeometry.position_interpolator import slice_disjoint
from trigger_hunter.wbs import WBS
from trigger_hunter.wbs2 import wbs_sdll
from trigger_hunter.utils.stats import poisson_gaussian
from astropy.stats import bayesian_blocks
import ruptures as rpt

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

    if ref_vector is None:
        ref_vector = np.repeat(1, x.shape[1])

    distance = np.sqrt(np.sum((x - ref_vector) ** 2, axis=1))

    return distance


def angle(x, ref_vector):

    dot_prod = np.dot(x, ref_vector)

    norm1 = np.sqrt(np.sum(x ** 2))
    norm2 = np.sqrt(np.sum(ref_vector ** 2))

    if abs(dot_prod / (norm1 * norm2)) > 1:
        return 0

    else:
        return np.arccos(dot_prod / (norm1 * norm2))


def angle_mapping(x, ref_vector=None):

    if ref_vector is None:
        ref_vector = np.repeat(1, x.shape[1])

    x_ang = np.apply_along_axis(angle, axis=1, arr=x, ref_vector=ref_vector)

    return x_ang


def calc_snr(data, background):

    snr = np.sum(data - background) / np.sqrt(np.sum(background))

    return snr


def calc_significance(data, background, bkg_stat_err):

    significance = poisson_gaussian.significance(
        n=np.sum(data), b=np.sum(background), sigma=np.sqrt(np.sum(bkg_stat_err ** 2))
    )

    return significance


class Search(object):
    def __init__(self, result_file, min_bin_width, mad=False, sub_min=False):

        self._load_result_file(result_file)

        self._dets_idx = []

        for det in self._detectors:
            self._dets_idx.append(valid_det_names.index(det))

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

        # Clean data
        self._counts_cleaned = (
            self._rebinned_observed_counts[self._rebinned_saa_mask]
            - self._rebinned_bkg_counts[self._rebinned_saa_mask]
        )

        self._data_cleaned = (
            self._counts_cleaned.T
            / self._rebinned_time_bin_width[self._rebinned_saa_mask]
        ).T

        self._data_flattened = self._data_cleaned[:, self._dets_idx, :][
            :, :, self._echans
        ].reshape((self._data_cleaned.shape[0], -1))

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

        self._penalty = 2 * np.log(len(self._data_cleaned))

    def _load_result_file(self, result_file):
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
        self._echans = echans
        self._time_bins = time_bins
        self._saa_mask = saa_mask
        self._observed_counts = observed_counts
        self._bkg_counts = model_counts
        self._bkg_stat_err = stat_err

    def find_changepoints_angles(self, **kwargs):

        change_points_sections = []
        change_points = []

        for section, valid_slice in enumerate(self._valid_slices):

            angle_slice = self._angles[valid_slice[0] : valid_slice[1]]

            penalty = 2 * np.log(len(angle_slice))

            algo_ang = rpt.Pelt(**kwargs).fit(angle_slice)
            cpts_seg = algo_ang.predict(pen=penalty)

            change_points_sections.append(cpts_seg)
            change_points.append(cpts_seg + valid_slice[0])

        change_points_sections = np.array(change_points_sections)
        change_points = np.array(change_points)

        self._cpts_sections_all = change_points_sections
        self._change_points_all = change_points

    def get_significance(self, required_significance=5):

        intervals = {}
        significances = {}

        echan_idx = 0
        echan = self._echans[0]

        for det in self._detectors:

            intervals[det] = []
            significances[det] = []

            det_idx = valid_det_names.index(det)

            for segment in range(len(self._change_points_all)):

                for idx_low in self._change_points_all[segment]:

                    for idx_high in self._change_points_all[segment]:

                        if idx_low != idx_high and idx_low < idx_high:

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

                                intervals[det].append([idx_low, idx_high])

                                significances[det].append(significance)

            intervals[det] = np.array(intervals[det])
            significances[det] = np.array(significances[det])

            sort_idx = significances[det].argsort()[::-1]

            intervals[det] = intervals[det][sort_idx]
            significances[det] = significances[det][sort_idx]

        # Filter out overlapping intervals and select the one with highest significance
        # This is done twice to get rid of some leftovers
        intervals_selected, significances_selected = self.filter_overlap(
            intervals, significances
        )
        intervals_selected, significances_selected = self.filter_overlap(
            intervals_selected, significances_selected
        )

        self._intervals_all = intervals
        self._intervals_significance_all = significances
        self._intervals = intervals_selected
        self._intervals_significance = significances_selected

    def filter_overlap(self, intervals, significances):

        intervals_selected = {}
        significances_selected = {}

        # Filter out overlapping intervals and select the one with highest significance
        for det in self._detectors:

            max_ids = []

            for id1, interval1 in enumerate(intervals[det]):

                tmp_ids = []

                tmp_ids.append(id1)

                for id2, interval2 in enumerate(intervals[det]):

                    if interval1[0] <= interval2[1] and interval2[0] <= interval1[1]:
                        tmp_ids.append(id2)

                max_id = significances[det][tmp_ids].argmax()

                max_ids.append(tmp_ids[max_id])

            max_ids = np.unique(max_ids)

            intervals_selected[det] = intervals[det][max_ids]
            significances_selected[det] = significances[det][max_ids]

        return intervals_selected, significances_selected

    def save_result(self, output_path):

        with h5py.File(output_path, "w") as f:

            f.attrs["dates"] = self._dates

            f.attrs["data_type"] = self._data_type

            f.attrs["echans"] = self._echans

            f.attrs["detectors"] = self._detectors

            for det in self._detectors:

                f.create_dataset(
                    f"{det}_intervals",
                    data=self._rebinned_mean_time(self._intervals_selected[det]),
                    compression="lzf",
                )

                f.create_dataset(
                    "{det}_significance",
                    data=self._intervals_significance[det],
                    compression="lzf",
                )

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
