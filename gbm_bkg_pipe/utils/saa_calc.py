#!/usr/bin/env python3
import numpy as np
from gbmgeometry.position_interpolator import slice_disjoint


class SaaCalc(object):


    def __init__(self, time_bins):

        self._time_bins = time_bins

        self._min_length = np.ceil(time_bins[0][1] - time_bins[0][0]) + 1

        self._build_masks()

    def _build_masks(self):
        """
        Calculates masks that cover the SAAs and some time bins before and after the SAAs
        :params bins_to_add: number of bins to add to mask before and after time bin
        """

        # Calculate the time jump between two successive time bins. During the SAAs no data is recorded.
        # This leads to a time jump between two successive time bins before and after the SAA.
        jump = self._time_bins[1:, 0] - self._time_bins[:-1, 1]

        # Get mask for which the jump is > 10 second
        jump_large = jump > 10

        # Get the indices of the time bins for which the jump to the previous time bin is >1 second
        # +1 is needed because we started with second time bin (we can not calculate the time jump
        # between the first time bin and the time bin before that one)
        idx = jump_large.nonzero()[0] + 1

        if idx.shape[0] > 0:
            # Build slices, that have as first entry start of SAA and as second end of SAA
            slice_idx = np.array(self.slice_disjoint_idx(idx))


            slice_idx[:, 1][
                np.where(slice_idx[:, 1] <= len(self._time_bins) - 1)
            ] = (
                slice_idx[:, 1][
                    np.where(slice_idx[:, 1] <= len(self._time_bins) - 1)
                ]
            )

            # make a saa mask from the slices:
            self._saa_mask = np.ones(len(self._time_bins), bool)

            for sl in slice_idx:
                self._saa_mask[sl[0] : sl[1] + 1] = False

        self._valid_slices = slice_disjoint(self._saa_mask)

    def slice_disjoint_idx(self, arr):
        """
        Returns an array of disjoint indices from a bool array
        :param arr: and array of bools
        """

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

    @property
    def saa_mask(self):
        return self._saa_mask

    @property
    def valid_slices(self):
        return self._valid_slices
