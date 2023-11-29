#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import types

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.fft import rfft, rfftfreq  # noqa: F401


class BeatInfo:
    """
    Provides information and features for a single beat.

    Parameters
    ----------
    beat_loc : int
        Index of beat in the R-peaks locations list, considering preceding
        R-peak locations and subsequent R-peak locations.
    fs : int, optional
        Sampling rate, by default 360
    in_ms : bool, optional
        Whether to calculate RR-intervals in time (miliseconds) or samples,
        by default True

    Attributes
    ----------
    beat_loc : int
        Index of beat in the R-peaks locations list, considering preceding
        R-peak locations and subsequent R-peak locations.
    fs : int, optional
        Sampling rate, by default 360
    in_ms : bool, optional
        Whether to calculate RR-intervals in time (miliseconds) or samples,
        by default True
    input_waveform : numpy.ndarray
        Input waveform, that is not trimmed (len_waveform, #channels).
    bwaveform : numpy.ndarray
        Segmented beat waveform. Selected as a segment of
        the input waveform (len_waveform, #channels).
    start_idx: int
        The index of the first sample (onset) of the waveform
        on the original signal.
    rri : list
        List of RR-intervals, in miliseconds if the in_ms parameter
        has been set to True.
    rri_smpl : list
        RR-intervals in samples.
    sdrri : list
        Successive RR interval (RRI) differences.
    avail_features : list
        List of features names (strings) that already have a
        definition (predefined features).
    features : dict
        Dictionary containing computed features. Keys are feature names
        and values are features values.

    Notes
    -----
    `BeatInfo` includes some predefined features, however custom features
    can also be defined. Each custom feature definition must adhere to this
    syntax and its name must start with *F_*:

    >>> def F_new_feature(self):
    >>>    return return_result

    The `return_result` must be one of the following:

    1) A real number.

    2) A dictionary such as {"Feature_1": value_1, "Feature_2": value_2}.
    Each value can be a real number. Alternatively, each value can be a
    one-dimensional array, list, or tuple. In this case, their elements
    must correspond to the channels.

    3) Tuple or one-dimensional NumPy array. Their elements must correspond
    to the channels. For example, the return value of `F_feat_new()` as a
    tuple *(element1, element2)* will produce `F_feat_new(CH1)`
    for *element1* and `F_feat_new(CH2)` for *element2*. The order of
    channels is determined by the *CHANNEL* field in the *config.yaml* file.

    4) A list. An output as a list will be a list.

    The custom features can be added to the beatinfo object by using the
    `add_features()` method, the list of all available features can be
    obtained using the `available_features()` method, and to select desired
    features for computation the `select_features()` method can be used.
    """

    def __init__(self, beat_loc=10, fs=360, in_ms=True):
        self.fs = fs
        self.beat_loc = beat_loc
        self.in_ms = in_ms

    def __call__(self, data):
        """
        Parameters
        ----------
        data : dict
            A dict containing data about the beat with keys:

            'waveform' : numpy.ndarray
                Waveform (len_waveform, #channels).
            'rpeak_locs' : list
                A list containing locations of rpeaks.
            'rec_id' : int
                Record id of the original signal which the excerpt is
                extracted from.
            'start_idx' : int
                Index of beginnig of the segmented waveform in
                the original signal.
            'label' : str
                Type of the beat.
        """

        self.data = data
        self.label = data["label"]
        self.rpeaks = data["rpeak_locs"]
        self.input_waveform = data["waveform"]  # Input waveform
        self.rri = self.get_rris(in_ms=self.in_ms)
        self.rri_smpl = self.get_rris(in_ms=False)
        self.sdrri = self.get_sdrri()
        (
            self.bwaveform,
            self.wfpts,
        ) = self.get_beat_waveform()  # Trimmed waveform
        self.avail_features = self.available_features()
        self.features = self.compute_features()

    def available_features(self):
        """Returns available features that can be computed.

        Returns
        -------
        list
            List containing features names.
        """
        avail_feats = [f for f in dir(self) if f.startswith("F_")]
        return avail_feats

    def add_features(self, new_features):
        """Adds new features.

        Parameters
        ----------
        new_features : list
            Names of new features in a list.
            Such as: [new_feature_1, new_feature_2]
        """
        for new_feature in new_features:
            setattr(
                self, new_feature.__name__, types.MethodType(new_feature, self)
            )

    def select_features(self, features):
        """Select desired features for computation.

        Parameters
        ----------
        features : list
            List of features names (strings).
        """
        self.selected_features_names = features

    def compute_features(self):
        """Computes features.

        Returns
        -------
        dict
            Dictionary with keys(strings) equal to feature names and
            dict values equal to features values.
        """
        if (
            hasattr(self, "selected_features_names")
            and len(self.selected_features_names) > 0
        ):
            features_names = self.selected_features_names
        else:
            features_names = self.avail_features
        feature_dict = {}
        for f in features_names:
            ret = getattr(self, f)()
            # if the feature function returns one value
            if isinstance(ret, (int, float)):
                feature_dict[f] = ret
            # if the feature function returns multiple values as a dict
            elif isinstance(ret, dict):
                try:
                    keys = list(ret.keys())
                    float(ret[keys[0]])
                    for key, val in ret.items():
                        feature_dict[key] = val
                except TypeError:
                    for key, val in ret.items():
                        feature_dict[key] = tuple(val)
            # For channels
            elif isinstance(ret, tuple) or isinstance(ret, np.ndarray):
                ret = tuple(ret)
                for chnl, val in enumerate(ret):
                    key = f"{f}(CH{chnl+1})"
                    feature_dict[key] = val
            elif isinstance(ret, list):
                feature_dict[f] = ret
        return feature_dict

    def get_beat_waveform(self, win=[-0.35, 0.65]):
        """Segment beat waveform according to the
        pre and post RR-intervals."""

        try:
            beat_rpeak_idx = self.rpeaks[self.beat_loc]
            prerri = self.rri_smpl[self.beat_loc - 1]
            postrri = self.rri_smpl[self.beat_loc]
            beat_onset_org = int(beat_rpeak_idx + win[0] * prerri)
            beat_offset_org = int(beat_rpeak_idx + win[1] * postrri)

            beat_onset = beat_onset_org - self.data["start_idx"]
            beat_offset = beat_offset_org - self.data["start_idx"]
            if beat_onset < 0:
                beat_onset = 0
            if beat_offset > len(self.input_waveform):
                beat_offset = len(self.input_waveform) - 1
            rpk = beat_rpeak_idx - self.data["start_idx"]
            bwaveform = self.input_waveform[beat_onset:beat_offset]
            bwaveform = np.asarray(bwaveform)
            # if len(bwaveform)==0 or beat_onset*beat_offset<0:
            # print(len(self.input_waveform))
            # print([beat_onset_org,beat_offset_org])
            # print(beat_rpeak_idx)
            # print(self.start_idx)
            # print([beat_onset,beat_offset])
            assert len(bwaveform) > 0, "beat waveform length cannot be zero!"

            wfpts = {
                "beat_onset_org": beat_onset_org,
                "beat_offset_org": beat_offset_org,
                "beat_onset": beat_onset,
                "beat_offset": beat_offset,
                "rpk": rpk,
            }
            return bwaveform, wfpts
        except Exception:
            import traceback

            traceback.print_exc()
            return None, None
            print("waveform error!")
            print(
                "rec_id={}, sindex={}".format(
                    self.data["rec_id"], self.data["start_idx"]
                )
            )
            plt.plot(self.input_waveform)
            plt.plot(bwaveform)
            plt.scatter(
                beat_onset, self.input_waveform[beat_onset], color="yellow"
            )
            plt.scatter(
                beat_offset, self.input_waveform[beat_offset], color="orange"
            )
            plt.scatter(rpk, self.input_waveform[rpk], color="r")

    def get_rris(self, in_ms):
        """Compute RR-intervals.

        Parameters
        ----------
        in_ms : bool
            If True result will be in time(miliseconds),
            otherwise in samples.

        Returns
        -------
        list
            Contains RR-intervals.
        """
        if in_ms:
            rpeaks = [1000 * item / self.fs for item in self.rpeaks]
        else:
            rpeaks = self.rpeaks
        rri = [rpeaks[i] - rpeaks[i - 1] for i in range(1, len(rpeaks))]
        return rri

    def get_sdrri(self):
        """Computes successive differences of rri.

        Returns
        -------
        list
            Contains successive differences of rri.
        """
        sdrri = list(np.diff(np.asarray(self.rri)))
        return sdrri

    # ============================================================
    # RRI features
    # ============================================================
    def F_post_rri(self):
        """Post RR interval."""
        return self.rri[self.beat_loc]

    def F_pre_rri(self):
        """Pre RR interval."""
        return self.rri[self.beat_loc - 1]

    def F_ratio_post_pre(self):
        """Ratio of post and pre RR-intervals."""
        post = self.F_post_rri()
        pre = self.F_pre_rri()
        return post / pre

    def F_diff_post_pre(self):
        """Difference of post and pre RR-intervals."""
        post = self.F_post_rri()
        pre = self.F_pre_rri()
        return post - pre

    def F_diff_post_pre_nr(self):
        """Normalized difference of post and pre RR-intervals."""
        post = self.F_post_rri()
        pre = self.F_pre_rri()
        return (post - pre) / self.F_rms_rri()

    def F_median_rri(self):
        """Median of RR-intervals."""
        return np.median(self.rri)

    def F_mean_pre_rri(self):
        """Mean of pre RR-intervals."""
        pre_rri = self.rri[: self.beat_loc]
        return np.mean(pre_rri)

    def F_rms_rri(self):
        """RMS of RR-intervals."""
        rri = np.asarray(self.rri)
        rms = np.sqrt(np.mean(rri**2))
        return rms

    def F_std_rri(self):
        """STD of RR-intervals."""
        return np.std(self.rri)

    def F_ratio_pre_rms(self):
        """Ratio of pre RR interval to RMS of RR-intervals."""
        return self.F_pre_rri() / self.F_rms_rri()

    def F_ratio_post_rms(self):
        """Ratio of post RR interval to RMS of RR-intervals."""
        return self.F_post_rri() / self.F_rms_rri()

    def F_diff_pre_avg_nr(self):
        """Difference of pre RR interval and median of RR-intervals,
        normalized by their RMS."""

        return (self.F_pre_rri() - self.F_median_rri()) / self.F_rms_rri()

    def F_diff_post_avg_nr(self):
        """Difference of post RR interval and median of RR-intervals,
        normalized by their RMS."""

        return (self.F_post_rri() - self.F_median_rri()) / self.F_rms_rri()

    def F_compensate_ratio(self):
        """Compensation ratio."""
        postpre = self.F_post_rri() + self.F_pre_rri()
        _2t2ndpre = 2 * self.rri[self.beat_loc - 2]  # 2 times second pre rri
        return postpre / _2t2ndpre

    def F_compensate_diff_nr(self):
        """Compensation diff normalized by RMS."""
        postpre = self.F_post_rri() + self.F_pre_rri()
        _2t2ndpre = 2 * self.rri[self.beat_loc - 2]  # 2 times second pre rri
        return (postpre - _2t2ndpre) / self.F_rms_rri()

    def F_heart_rate(self):
        """Heart rate."""
        pre_rri = (
            self.rpeaks[self.beat_loc] - self.rpeaks[self.beat_loc - 1]
        ) / self.fs
        hr = 60 / pre_rri
        return hr

    # ============================================================
    # SDRRI features
    # ============================================================
    def F_mean_sdrri(self):
        """Mean of successive RRI differences."""
        return np.mean(self.sdrri)

    def F_absmean_sdrri(self):
        """Mean of absolute successive RRI differences."""
        return np.mean(np.abs(self.sdrri))

    def F_mean_pre_sdrri(self):
        """Mean of pre successive RRI differences."""
        pre_sdrri = self.sdrri[: self.beat_loc - 1]
        return np.mean(pre_sdrri)

    def F_rms_sdrri(self):
        """RMS of successive RRI differences."""
        sdrri = np.asarray(self.sdrri)
        return np.sqrt(np.mean(sdrri**2))

    def F_std_sdrri(self):
        """STD of successive RRI differences."""
        return np.std(self.sdrri)

    # ============================================================
    # Waveform features
    # ============================================================
    def reported_rpeak(self):
        return self.rpeaks[self.beat_loc] - self.wfpts["beat_onset_org"]

    def F_beat_max(self):
        """Max value of heartbeat waveform."""
        res = np.max(self.bwaveform, axis=0)
        return res

    def F_beat_min(self):
        """Min value of heartbeat waveform."""
        res = np.min(self.bwaveform, axis=0)
        return res

    def F_maxmin_diff(self):
        """Difference of max and min values of heartbeat waveform."""
        res = self.F_beat_max() - self.F_beat_min()
        return res

    def F_maxmin_diff_norm(self):
        """Difference of max and min values of heartbeat waveform,
        normalized by waveform RMS.

        """
        res = self.F_maxmin_diff() / self.F_beat_rms()
        return res

    def F_beat_mean(self):
        """Mean of heartbeat waveform."""
        res = np.mean(self.bwaveform, axis=0)
        return res

    def F_beat_std(self):
        """STD of heartbeat waveform."""
        res = np.std(self.bwaveform, axis=0)
        return res

    def F_beat_skewness(self):
        """Skewness of heartbeat waveform."""
        res = stats.skew(self.bwaveform)
        return res

    def F_beat_kurtosis(self):
        """Kurtosis of heartbeat waveform."""
        res = stats.kurtosis(self.bwaveform)
        return res

    def F_beat_rms(self):
        """RMS of heartbeat waveform."""
        res = np.sqrt(np.mean(self.bwaveform**2, axis=0))
        return res

    def F_nsampels(self, n_samples=10):
        """Gives equally spaced samples of the trimmed heartbeat waveform.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples, by default 10

        Returns
        -------
        dict
            Waveform sample values.

        """

        dt = int(len(self.bwaveform) / n_samples)
        samples = []
        for i in range(n_samples):
            samples.append(self.bwaveform[int(i * dt)])
        samples_dict = {}
        for i, sampl in enumerate(samples):
            key = "sample_" + str(i + 1)
            samples_dict[key] = sampl
        return samples_dict

    # def F_subsegs(self, n_subsegs=3, ftr="max"):
    #     """Computes a feature for equal-length subsegments of
    #     the beat waveform."""

    #     def func(arg, ftr):
    #         if ftr == "max":
    #             f = max(arg)
    #         elif ftr == "min":
    #             f = min(arg)
    #         elif ftr == "mean":
    #             f = np.mean(arg)
    #         elif ftr == "std":
    #             f = np.std(arg)
    #         elif ftr == "median":
    #             f = np.median(arg)
    #         elif ftr == "skewness":
    #             f = stats.skew(arg)
    #         elif ftr == "kurtosis":
    #             f = stats.kurtosis(arg)
    #         elif ftr == "rms":
    #             f = np.sqrt(np.mean(arg**2))
    #         return f

    #     ret_dict = {}
    #     for i in range(n_subsegs):
    #         s_ix = int(i / n_subsegs * len(self.bwaveform))
    #         e_ix = int((i + 1) / n_subsegs * len(self.bwaveform))
    #         subseg = self.bwaveform[s_ix:e_ix]
    #         key = f"seg[{str(i+1)}]_"
    #         ret_dict[key + ftr] = func(subseg, ftr)
    #     # agg results
    #     ls = [v for k, v in ret_dict.items()]
    #     arr = np.asarray(ls)
    #     ret_dict["agg_mean_" + ftr] = np.mean(arr)
    #     ret_dict["agg_std_" + ftr] = np.std(arr)
    #     ret_dict["agg_rms_" + ftr] = np.sqrt(np.mean(arr**2))
    #     return ret_dict

    # ============================================================
    # Spectral features of the beat waveform
    # ============================================================
    def F_fft_features(self):
        """Returns FFT of each heartbeat trimmed waveform.

        Returns
        -------
        dict

        """

        sig = self.bwaveform
        # num_samples = sig.size
        # xf = rfftfreq(num_samples, 1 / self.fs)
        yf = np.abs(rfft(sig))
        ret_dict = {}
        for i, ret in enumerate(list(yf)):
            key = "fft_" + str(i + 1)
            ret_dict[key] = ret
        return ret_dict
