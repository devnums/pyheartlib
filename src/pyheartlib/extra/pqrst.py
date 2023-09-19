#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import numpy as np
from scipy.signal import find_peaks


class PQRST:
    """
    Computes PQRST features for a given beat waveform
    """

    def __init__(self, fs=360):
        self.fs = fs

    def __call__(self, beatwave):
        self.beatwave = beatwave
        # self.grd = np.gradient(self.beatwave)
        self.rwave = self._comp_r_wave()
        self.qwave = self._comp_q_wave()
        self.pwave = self._comp_p_wave()
        self.swave = self._comp_s_wave()
        self.pr_interval = self._comp_pr_interval()
        self.qs_interval = self._comp_qs_interval()

    def _comp_r_wave(self):
        # R Wave Peak
        r_x = np.argmax(self.beatwave)
        r_y = self.beatwave[r_x]
        return [r_x, r_y]

    def _comp_q_wave(self):
        # Q wave Peak
        try:
            seg = self.beatwave[: self.rwave[0]]
            # if len(seg)<0.05*self.fs:
            #    return [np.nan,np.nan] #segment too small
            segrng = max(seg) - min(seg)
            peaks, prop = find_peaks(-seg, height=0.001 * segrng)  # TODO
            if len(peaks) != 0:
                q_x = peaks[-1]  # the peak on the most right
            else:
                idx = np.where(np.asarray(seg) == 0)[0]
                if len(idx) != 0:
                    q_x = idx[-1]  # first zero from right
                else:
                    q_x = 2  # extreme case scenario
            q_y = self.beatwave[q_x]
            return [q_x, q_y]
        except Exception:
            return [np.nan, np.nan]
            # import matplotlib.pyplot as plt

            # plt.plot(self.beatwave)
            # plt.plot(seg)

    def _comp_p_wave(self):
        # P Wave Peak (+,-)
        try:
            seg = self.beatwave[: self.qwave[0]]
            # if len(seg)<0.05*self.fs:
            #    return [np.nan,np.nan] #segment too small
            segrng = max(seg) - min(seg)
            peaks, prop = find_peaks(seg, height=0.05 * segrng)
            # if positive p wave available
            if len(peaks) != 0:
                p_x = peaks[np.argmax(prop["peak_heights"])]
                p_y = self.beatwave[p_x]
            # try negative p waves
            else:
                peaks, prop = find_peaks(-seg, height=0.05 * segrng)
                if len(peaks) != 0:
                    p_x = peaks[np.argmax(prop["peak_heights"])]
                    p_y = seg[p_x]
                else:
                    p_x = -200  # p wave not availble
                    p_y = -200
            return [p_x, p_y]
        except Exception:
            return [np.nan, np.nan]

    def _comp_s_wave(self):
        # S wave Peak
        try:
            seg = self.beatwave[self.rwave[0] :]
            # if len(seg)<0.05*self.fs:
            #    return [np.nan,np.nan] #segment too small
            segrng = max(seg) - min(seg)
            peaks, prop = find_peaks(-seg, height=0.001 * segrng)  # TODO
            if len(peaks) != 0:
                s_x = peaks[0]  # the peak on the most left
            else:
                idx = np.where(np.asarray(seg) == 0)[0]
                if len(idx) != 0:
                    s_x = idx[0]  # first zero from left
                else:
                    s_x = len(seg) - 2  # extreme case scenario
            s_x = s_x + self.rwave[0]
            s_y = self.beatwave[s_x]
            return [s_x, s_y]
        except Exception:
            return [np.nan, np.nan]
            # import matplotlib.pyplot as plt

            # plt.plot(self.beatwave)
            # plt.plot(seg)

    def _comp_pr_interval(self):
        # PR Interval in ms
        pr_inerval = self.rwave[0] - self.pwave[0]  # TODO pwave onset
        return pr_inerval / self.fs * 1000

    def _comp_qs_interval(self):
        # PR Interval in ms
        qs_inerval = self.swave[0] - self.qwave[0]
        return qs_inerval / self.fs * 1000
