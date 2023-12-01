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
from scipy import stats


def get_stat_features(data, features="all"):
    """
    Computes statistical features for the input samples.

    Parameters
    ----------
    data : numpy.array
        A 2D numpy array with shape (#samples,len_series).
    features : list
        A list of features to be computed.

    Returns
    -------
    features_arr : numpy.array
        A 2D numpy array with the shape (#samples, #features).
    """
    if features == "all":
        flist = [
            "max",
            "min",
            "mean",
            "std",
            "median",
            "skew",
            "kurtosis",
            "range",
        ]
    else:
        flist = features

    num_samples = data.shape[0]
    num_features = len(flist)
    features_arr = np.ndarray((num_samples, num_features), dtype=np.float32)

    if "max" in flist:
        ix = flist.index("max")
        features_arr[:, ix] = np.amax(data, axis=1)

    if "min" in flist:
        ix = flist.index("min")
        features_arr[:, ix] = np.amin(data, axis=1)

    if "mean" in flist:
        ix = flist.index("mean")
        features_arr[:, ix] = np.mean(data, axis=1)

    if "std" in flist:
        ix = flist.index("std")
        features_arr[:, ix] = np.std(data, axis=1)

    if "median" in flist:
        ix = flist.index("median")
        features_arr[:, ix] = np.median(data, axis=1)

    if "skew" in flist:
        ix = flist.index("skew")
        features_arr[:, ix] = stats.skew(data, axis=1)

    if "kurtosis" in flist:
        ix = flist.index("kurtosis")
        features_arr[:, ix] = stats.kurtosis(data, axis=1)

    if "range" in flist:
        ix = flist.index("range")
        features_arr[:, ix] = np.amax(data, axis=1) - np.amin(data, axis=1)

    return features_arr


def get_hrv_features(rri=None, features="all", only_names=False):
    """
    Computes hrv features for the input samples.

    Parameters
    ----------
    rri : numpy.array
        A 2D numpy array with shape (#samples, len_series).
        Series are rr intervals in milliseconds(ms).
    features : list
        A list of features to be computed.

    Returns
    -------
    features_arr : numpy.array
        A 2D numpy array with the shape (#samples, #features).
    """
    if features == "all":
        flist = [
            "meanrr",
            "sdrr",
            "medianrr",
            "rangerr",
            "nsdrr",
            "sdsd",
            "rmssd",
            "nrmssd",
        ]
        flist += ["prr50"]
        # flist += ['meanhr','maxhr','minhr','medianhr','sdhr']

    else:
        flist = features  # features list

    if not only_names:
        num_samples = rri.shape[0]
        num_features = len(flist)
        features_arr = np.ndarray(
            (num_samples, num_features), dtype=np.float32
        )

        # successive RR interval differences
        sd = np.diff(rri, axis=1)

        # calculate meanrr
        if bool({"meanrr", "nsdrr", "nrmssd"} & set(flist)):
            meanrr = np.mean(rri, axis=1)

        # calculate meanrr
        if bool({"sdrr", "nsdrr"} & set(flist)):
            sdrr = np.std(rri, axis=1)

        # calculate rmssd
        if bool({"rmssd", "nrmssd"} & set(flist)):
            rmssd = np.sqrt(np.mean(sd**2, axis=1))

        # calculate hr
        if bool({"meanhr", "maxhr", "minhr", "medianhr", "sdhr"} & set(flist)):
            hr = 60000 / rri

        if "meanrr" in flist:
            ix = flist.index("meanrr")
            features_arr[:, ix] = meanrr

        if "sdrr" in flist:
            ix = flist.index("sdrr")
            features_arr[:, ix] = sdrr

        if "medianrr" in flist:
            ix = flist.index("medianrr")
            features_arr[:, ix] = np.ma.median(rri, axis=1)

        if "rangerr" in flist:
            ix = flist.index("rangerr")
            features_arr[:, ix] = np.amax(rri, axis=1) - np.amin(rri, axis=1)

        if "nsdrr" in flist:
            ix = flist.index("nsdrr")
            features_arr[:, ix] = sdrr / meanrr

        if "sdsd" in flist:
            ix = flist.index("sdsd")
            features_arr[:, ix] = np.std(sd, axis=1)

        # Root mean square of successive RR interval differences
        if "rmssd" in flist:
            ix = flist.index("rmssd")
            features_arr[:, ix] = rmssd

        if "nrmssd" in flist:
            ix = flist.index("nrmssd")
            features_arr[:, ix] = rmssd / meanrr

        if "prr50" in flist:
            ix = flist.index("prr50")
            prr50 = 100 * np.sum(np.abs(sd) > 5, axis=1) / sd.shape[1]
            features_arr[:, ix] = prr50

        if "meanhr" in flist:
            ix = flist.index("meanhr")
            features_arr[:, ix] = np.mean(hr, axis=1)

        if "maxhr" in flist:
            ix = flist.index("maxhr")
            features_arr[:, ix] = np.amax(hr, axis=1)

        if "minhr" in flist:
            ix = flist.index("minhr")
            features_arr[:, ix] = np.amin(hr, axis=1)

        if "medianhr" in flist:
            ix = flist.index("medianhr")
            features_arr[:, ix] = np.median(hr, axis=1)

        if "sdhr" in flist:
            ix = flist.index("sdhr")
            features_arr[:, ix] = np.std(hr, axis=1)

        return features_arr

    elif only_names:
        return flist


def get_wf_feats(sig=None, interval=None, only_names=False):
    """Computes features for the signal waveform.

    The input signal is segmented into sub-signals
    based on the given interval parameter.

    Parameters
    ----------
    sig : numpy.array
        A 1D numpy array.
    interval : int
        Length of subsegments.

    Returns
    -------
    features_arr : numpy.array
        A 2D numpy array with the shape (#subsegments, #features).
    """

    flist = [
        "max",
        "min",
        "mean",
        "std",
        "median",
        "skew",
        "kurtosis",
        "max(squared)",
        "min(squared)",
        "mean(squared)",
        "std(squared)",
        "median(squared)",
        "skew(squared)",
        "kurtosis(squared)",
    ]

    if not only_names:
        b = int(len(sig) / interval)
        subseqs = []
        for i in range(b):
            subseq = sig[i * interval : (i + 1) * interval]
            subseqs.append(subseq)
        subseqs = np.array(subseqs)  # interval=36---> (300,36)
        f1 = np.amax(subseqs, axis=1)
        f2 = np.amin(subseqs, axis=1)
        f3 = np.mean(subseqs, axis=1)
        f4 = np.std(subseqs, axis=1)
        f5 = np.median(subseqs, axis=1)
        f6 = stats.skew(subseqs, axis=1)
        f7 = stats.kurtosis(subseqs, axis=1)
        sqr = subseqs**2
        f8 = np.amax(sqr, axis=1)
        f9 = np.amin(sqr, axis=1)
        f10 = np.mean(sqr, axis=1)
        f11 = np.std(sqr, axis=1)
        f12 = np.median(sqr, axis=1)
        f13 = stats.skew(sqr, axis=1)
        f14 = stats.kurtosis(sqr, axis=1)
        features_arr = np.vstack(
            (f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14)
        )

        return features_arr.T
    elif only_names:
        return flist
