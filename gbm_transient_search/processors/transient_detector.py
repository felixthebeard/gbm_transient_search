import copy
from copy import deepcopy
from datetime import datetime

import h5py
import numpy as np
import pytz
import ruptures as rpt
import yaml
from astropy.io import fits
from gbm_transient_search.processors.saa_calc import SaaCalc
from gbm_transient_search.utils.plotting.trigger_plot import TriggerPlot
from gbmbkgpy.utils.binner import Rebinner
from gbmgeometry import GBMTime
from loguru import logger
from pathos.multiprocessing import cpu_count
from pathos.pools import ProcessPool as Pool
from scipy import stats
from threeML.utils.statistics.stats_tools import Significance

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


class TransientDetector(object):
    """
    Transient search class.
    This finds change points in the time series,
    calculates the significances of source intervals,
    and filters them to construct triggers.
    """

    def __init__(
        self, result_file=None, min_bin_width=1e-99, mad=False, bad_fit_threshold=60
    ):
        """
        Instantiate the search class and prepare the data for processing.
        """

        self._min_bin_width = min_bin_width
        self._mad = mad
        self._bad_fit_threshold = bad_fit_threshold

        if result_file is not None:
            self._load_result_file(result_file)
            self._setup()

    def run(
        self,
        min_separation=5,
        min_size=1,
        jump=1,
        model="l2",
        min_significance_brightest=5,
        min_significance_others=5,
        min_significant_dets=2,
        max_significant_dets=2,
    ):
        """
        min_separation: Minimal separation (in bins) between the change points in angles and distnaces.
        min_size: Minimal separation (in bins) between changepoints.
        jump: Subsampling of time series.
        model: Model for the cost function.
        min_significance_brightest: Required significance for the brightest detector.
        min_significance_others: Required significance for other detectors,
        min_significant_dets: Min number of detectors required to be significant
        max_significant_dets: Max number of detectors allowed to be significant
        """
        self._detect_changepoints(
            min_separation=min_separation, min_size=min_size, jump=jump, model=model
        )
        self._calc_significances()
        self._apply_threshold_significance(
            significance_brightest=min_significance_brightest,
            significance_others=min_significance_others,
            min_dets=min_significant_dets,
            max_dets=max_significant_dets,
        )
        self._select_intervals()
        self._find_peak_times()
        self._create_result_dict()

    def _setup(self):
        self._dets_idx = []

        for det in self._detectors:
            self._dets_idx.append(valid_det_names.index(det))

        # Calculate new saa mask to fix stan fit
        saa_calc = SaaCalc(self._time_bins)
        self._saa_mask = saa_calc.saa_mask

        self._rebinn_data(self._min_bin_width)

        self._mask_bad_bkg_fits(self._bad_fit_threshold)

        # Combine all energy bins for significance calculation
        self._combine_energy_bins()

        # Clean data
        self._counts_cleaned = (
            self._rebinned_observed_counts[self._rebinned_saa_mask]
            - self._rebinned_bkg_counts[self._rebinned_saa_mask]
        )

        self._counts_cleaned_total = {
            det: (self._observed_counts_total[det] - self._bkg_counts_total[det])
            for det in self._detectors
        }

        self._rates_cleaned = (
            self._counts_cleaned.T
            / self._rebinned_time_bin_width[self._rebinned_saa_mask]
        ).T

        self._transform_data(self._mad)

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

    def _transform_data(self, mad):
        """
        Transform the data to and apply mapping
        """
        self._data_flattened = self._counts_cleaned[:, self._good_bkg_fit_mask]

        if mad:

            med_abs_div = stats.median_abs_deviation(self._data_flattened, axis=0)
            median = np.median(self._data_flattened, axis=0)

            self._data_trans = (self._data_flattened - median) / med_abs_div

        else:
            self._data_trans = self._data_flattened

        self._data_trans = (
            self._data_trans - np.min(self._data_trans, axis=0).reshape((1, -1)) + 1
        )

        self._distances = distance_mapping(self._data_trans)
        self._angles = angle_mapping(self._data_trans)

    def _mask_bad_bkg_fits(
        self, max_med_sig=60, max_neg_med_sig=-30, max_neg_sig=-100, n_parts=20
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

                sig = Significance(
                    self._observed_counts[:, det_idx, e].sum(),
                    self._bkg_counts[:, det_idx, e].sum(),
                )
                sig_total = sig.li_and_ma_equivalent_for_gaussian_background(
                    np.sum(self._bkg_stat_err[:, det_idx, e] ** 2)
                )

                sigs = []
                counts = np.empty(n_parts - 1)
                bkg_counts = np.empty(n_parts - 1)
                bkg_errs = np.empty(n_parts - 1)

                for i, (a, b) in enumerate(zip(part_idx[:-1], part_idx[1:])):

                    counts[i] = self._observed_counts[a:b, det_idx, e].sum()
                    bkg_counts[i] = self._bkg_counts[a:b, det_idx, e].sum()
                    bkg_errs[i] = np.sqrt(
                        np.sum(self._bkg_stat_err[a:b, det_idx, e] ** 2)
                    )

                sig = Significance(counts, bkg_counts)
                sigs.extend(sig.li_and_ma_equivalent_for_gaussian_background(bkg_errs))

                median_sig = np.median(sigs)

                good = True
                if sig_total <= max_neg_sig or median_sig <= max_neg_med_sig:
                    good = False
                if median_sig >= max_med_sig:
                    good = False

                if good:
                    good_bkg_fit_mask[det_idx, e] = True

        self._good_bkg_fit_mask = good_bkg_fit_mask

    def _combine_energy_bins(self, echans=[0, 1, 2, 3, 4, 5, 6, 7]):
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

    def _detect_changepoints(self, min_separation=0, **kwargs):
        """
        Find changepoints applying the ruptures PELT method
        in the angles time series
        """

        def detect_cpts(arg):

            array, mapping, slice_idx, valid_slice, kwargs = arg

            array_slice = array[valid_slice[0] : valid_slice[1]]

            penalty = 2 * np.log(len(array_slice))

            algo_dist = rpt.Pelt(**kwargs).fit(array_slice)

            cpts_seg = algo_dist.predict(pen=penalty)

            return (mapping, slice_idx, cpts_seg + valid_slice[0])

        def find_min_distance(array, value):
            array = np.asarray(array)
            return (np.abs(array - value)).min()

        jobs = []
        pool = Pool(cpu_count())

        for i, valid_slice in enumerate(self._valid_slices):
            jobs.append((self._angles, "angle", i, valid_slice, kwargs))

        for i, valid_slice in enumerate(self._valid_slices):
            jobs.append((self._distances, "distance", i, valid_slice, kwargs))

        cpts_output = pool.map(detect_cpts, jobs)

        change_points_angles = [None] * len(self._valid_slices)

        for mapping, slice_idx, cpts in cpts_output:
            if mapping == "angle":
                change_points_angles[slice_idx] = cpts

        change_points = deepcopy(change_points_angles)

        for mapping, slice_idx, cpts in cpts_output:
            if mapping == "distances":

                for cpt in cpts:

                    if (
                        find_min_distance(change_points_angles[slice_idx], cpt)
                        > min_separation
                    ):
                        change_points[slice_idx].append(cpt)

        self._change_points_all = change_points

    def _calc_significances(self):
        """
        Calculate the significance of the interval between two subsequent change points.
        Treat the sections between SAA passages individually to not have intervals spanning
        over long dead times.
        """

        intervals = []

        for cpts_segment in self._change_points_all:
            intervals_segment = list(zip(cpts_segment[:-1], cpts_segment[1:]))
            intervals.extend(intervals_segment)

        n = len(intervals)
        counts = np.empty(n)
        bkg_counts = np.empty(n)
        bkg_errs = np.empty(n)

        significances = np.zeros((len(intervals), len(self._detectors)))

        for det_idx, det in enumerate(self._detectors):

            for i, (a, b) in enumerate(intervals):

                counts[i] = self._observed_counts_total[det][a:b].sum()
                bkg_counts[i] = self._bkg_counts_total[det][a:b].sum()
                bkg_errs[i] = np.sqrt(np.sum(self._bkg_stat_err_total[det][a:b] ** 2))

            sig = Significance(counts, bkg_counts)
            significances[
                :, det_idx
            ] = sig.li_and_ma_equivalent_for_gaussian_background(bkg_errs)

        self._intervals_all = np.array(intervals)
        self._significances_all = significances

    def _apply_threshold_significance(
        self, significance_brightest=5, significance_others=2, min_dets=2, max_dets=10
    ):
        """
        Apply threshold to the significance of an interval.
        """
        nr_dets_brightest = np.sum(
            self._significances_all > significance_brightest, axis=1
        )
        nr_dets_others = np.sum(self._significances_all > significance_others, axis=1)

        valid_brightest = nr_dets_brightest >= 1

        valid_others = np.logical_and(
            nr_dets_others >= min_dets, nr_dets_others <= max_dets
        )

        valid_idx = np.logical_and(valid_brightest, valid_others)

        self._intervals = self._intervals_all[valid_idx]
        self._significances = self._significances_all[valid_idx]

    def _select_intervals(self):
        # Get non-overlapping segments
        trigger_intervals = segment_disjoint(self._intervals)
        disjoint_segments_idx = segment_disjoint_idx(self._intervals)

        max_dets = []
        max_intervals = []
        max_significances = []

        # For each trigger interval find the detector with the brightest (sub)-interval
        for seg_idx in disjoint_segments_idx:
            sigs = self._significances[seg_idx[0] : seg_idx[1]]
            ints = self._intervals[seg_idx[0] : seg_idx[1]]

            max_sig = np.max(sigs)
            int_idx, det_idx = np.where(sigs == max_sig)

            if int_idx.shape[0] != 1:
                logger.error(
                    "Found multiple intervals with the exact same significance"
                )
                int_idx = int_idx[0]
            if det_idx.shape[0] != 1:
                logger.error(
                    "Found multiple detectors with the exact same significance"
                )
                det_idx = det_idx[0]

            max_dets.append(self._detectors[det_idx[0]])
            max_intervals.append(ints[int_idx[0]])
            max_significances.append(max_sig)

        self._trigger_intervals = np.array(trigger_intervals)
        self._max_dets = np.array(max_dets)
        self._max_intervals = np.array(max_intervals)
        self._max_significances = np.array(max_significances)

    def _find_peak_times(self):

        # Get the peak time of the
        trigger_peak_times = []

        for i, (a, b) in enumerate(self._max_intervals):

            max_index = (
                np.argmax(self._counts_cleaned_total[self._max_dets[i]][a:b]) + a
            )

            trigger_peak_times.append(
                self._rebinned_time_bins[self._rebinned_saa_mask][max_index, 0]
            )

        self._trigger_times = self._rebinned_time_bins[self._rebinned_saa_mask][
            self._max_intervals[:, 0], 0
        ]
        self._trigger_peak_times = np.array(trigger_peak_times)

    @property
    def trigger_peak_times(self):
        return self._trigger_peak_times

    @property
    def trigger_times(self):
        return self._trigger_times

    @property
    def trigger_significances(self):
        return self._max_significances

    @property
    def trigger_intervals(self):
        return self._max_intervals

    @property
    def trigger_most_sig_det(self):
        return self._max_dets

    def _create_result_dict(self):
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

        for i, t0 in enumerate(self.trigger_times):
            t0 = self.trigger_times[i]

            gbm_time = GBMTime.from_MET(t0)
            date_str = gbm_time.time.datetime.strftime("%y%m%d")
            day_fraction = f"{round(gbm_time.time.mjd % 1, 3):.3f}"[2:]

            trigger_name = f"GRT{date_str}{day_fraction}"
            sig = self.trigger_significances.tolist()[i]
            max_det = self.trigger_most_sig_det.tolist()[i]

            peak_time = self.trigger_peak_times[i] - t0
            tstart = self._rebinned_mean_time[self.trigger_intervals[i]][0].tolist()
            tstop = self._rebinned_mean_time[self.trigger_intervals[i]][1].tolist()

            t_info = {
                "date": date_str,
                "trigger_name": trigger_name,
                "trigger_time": t0.tolist(),
                "trigger_time_utc": gbm_time.utc,
                "peak_time": peak_time.tolist(),
                "significances": sig,
                "interval": {
                    "start": tstart,
                    "stop": tstop,
                },
                "most_significant_detector": max_det,
            }

            trigger_information["triggers"][trigger_name] = t_info

        self._trigger_information = trigger_information

    def plot_results(self, output_dir):

        plotter = TriggerPlot(
            triggers=self._trigger_information["triggers"],
            time_bins=self._rebinned_time_bins,
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

    def set_data_timestamp(self, data_file_path):
        # Wrap in try except for the simulation to work
        try:
            with fits.open(data_file_path) as f:
                data_timestamp_goddard = f["PRIMARY"].header["DATE"] + ".000Z"

            datetime_ob_goddard = pytz.timezone("US/Eastern").localize(
                datetime.strptime(data_timestamp_goddard, "%Y-%m-%dT%H:%M:%S.%fZ")
            )

            data_timestamp = datetime_ob_goddard.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except:
            data_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        self._trigger_information["data_timestamp"] = data_timestamp

    def save_result(self, output_path):
        # output_file = os.path.join(os.path.dirname(output_path), "trigger_information.yml")
        with open(output_path, "w") as f:
            yaml.dump(self._trigger_information, f, default_flow_style=False)

    def load_simulation(self, simulation):
        """
        Load simulation
        """
        self._dates = simulation.dates
        self._detectors = simulation.detectors
        self._echans = np.array([int(echan) for echan in simulation.echans])
        self._data_type = simulation.data_type
        self._time_bins = simulation.time_bins
        self._saa_mask = simulation.saa_mask
        self._observed_counts = simulation.observed_counts
        self._bkg_counts = simulation.bkg_counts
        self._bkg_stat_err = simulation.bkg_stat_err

        self._setup()


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

    return (x_ang / np.pi) * 360


def slice_disjoint(arr):
    """
    Returns an array of disjoint indices from a bool array
    :param arr: and array of bools
    """

    arr = (arr).nonzero()[0]

    slices = []
    start_slice = arr[0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1] > arr[i] + 1:
            end_slice = arr[i]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1]
            counter += 1
    if counter == 0:
        return [[arr[0], arr[-1]]]
    if end_slice != arr[-1]:
        slices.append([start_slice, arr[-1]])
    return slices


def segment_disjoint(arr):
    """
    Returns an array of disjoint segments from an array of (overlapping) segments
    :param arr: and array of segments
    """
    arr = np.sort(arr)
    slices = []
    start_slice = arr[0][0]
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1][0] > arr[i][1]:
            end_slice = arr[i][1]
            slices.append([start_slice, end_slice])
            start_slice = arr[i + 1][0]
            counter += 1
    if counter == 0:
        return arr
    if end_slice != arr[-1][1]:
        slices.append([start_slice, arr[-1][1]])
    return slices


def segment_disjoint_idx(arr):
    """
    Returns an array of start and end indexes from an array of (overlapping) segments
    :param arr: and array of segments
    """
    arr = np.sort(arr)
    slices = []
    start_idx = 0
    counter = 0
    for i in range(len(arr) - 1):
        if arr[i + 1][0] > arr[i][1]:
            end_idx = i + 1
            slices.append([start_idx, end_idx])
            start_idx = i + 1
            counter += 1
    if counter == 0:
        return [[0, len(arr) - 1]]
    if end_idx != len(arr):
        slices.append([start_idx, len(arr)])
    return slices
