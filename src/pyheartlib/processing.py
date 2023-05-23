import math
import numpy as np
from tqdm import tqdm
from scipy.signal import spectrogram
from scipy.signal import medfilt
from scipy.signal import butter, sosfilt, sosfreqz, sosfiltfilt


class Processing:
    """Methods for processing signals"""

    @staticmethod
    def apply(processors, signal):
        """Applies processors in order.

        Parameters
        ----------
        processors : list
            List of tuples of ('processor_name', params)
        signal : numpy.ndarray
            Signal
        """
        s = signal
        for p in processors:
            pname = p[0]
            pparams = p[1]
            s = getattr(__class__, pname)(s, **pparams)
        return s

    @staticmethod
    def remove_baseline(signal, sampling_rate=360):
        """Removes the signal baseline by applying two median filters.

        Parameters
        ----------
        signal : numpy.ndarray
            1D ndarray.
        sampling_rate : int, optional
            sampling frequency, by default 360

        Returns
        -------
        numpy.ndarray
            Baseline removed signal.
        """

        ker_size = int(0.2 * sampling_rate)
        if ker_size % 2 == 0:
            ker_size += 1
        baseline = medfilt(signal, kernel_size=ker_size)

        ker_size = int(0.6 * sampling_rate)
        if ker_size % 2 == 0:
            ker_size += 1
        baseline = medfilt(baseline, kernel_size=ker_size)
        modified_signal = signal - baseline

        return modified_signal

    @staticmethod
    def lowpass_filter_butter(signal, cutoff=45, sampling_rate=360, order=15):
        """Applies low pass filter to the signal.

        Parameters
        ----------
        signal : numpy.ndarray
            A one-dimensional signal.
        cutoff : int, optional
            Filter parameter, by default 45
        sampling_rate : int, optional
            Sampling frequency, by default 360
        order : int, optional
            Filter parameter, by default 15

        Returns
        -------
        numpy.ndarray
            Low pass filtered signal.
        """
        nyq = 0.5 * sampling_rate
        cf = cutoff / nyq
        sos = butter(order, cf, btype="low", output="sos", analog=False)
        sig = sosfiltfilt(sos, signal)
        return sig

    @staticmethod
    def denoise_signal(
        signal, remove_bl=True, lowpass=False, sampling_rate=360, cutoff=45, order=15
    ):
        """Denoises the signal by removing the baseline wander and/or applying a low pass filter.

        Parameters
        ----------
        signal : numpy.ndarray
            A one-dimensional signal.
        remove_bl : bool, optional
            If True, removes baseline wander, by default True
        lowpass : bool, optional
            If True, applies a low pass filter, by default False
        sampling_rate : int, optional
            Sampling frequency, by default 360
        cutoff : int, optional
            Low pass filter parameter, by default 45
        order : int, optional
            Low pass filter parameter, by default 15

        Returns
        -------
        numpy.ndarray
            Denoised signal.
        """

        if remove_bl and not lowpass:
            y = __class__.remove_baseline(signal, sampling_rate=sampling_rate)
        if lowpass and not remove_bl:
            y = __class__.lowpass_filter_butter(
                signal, cutoff=cutoff, sampling_rate=sampling_rate, order=order
            )
        if remove_bl and lowpass:
            y = __class__.remove_baseline(signal)
            y = __class__.lowpass_filter_butter(
                y, cutoff=cutoff, sampling_rate=sampling_rate, order=order
            )
        if not remove_bl and not lowpass:
            y = signal
        return y


class STFT:
    """
    Short time fourier transform.

    Example
    -------
    >>> dpr = STFT()
    >>> features = dpr.specgram(x, sampling_rate=360, nperseg=127, noverlap=122)
    """

    def __init__(self):
        pass

    def specgram(self, signals, sampling_rate=None, nperseg=None, noverlap=None):
        """Applies Short time fourier transform on the signals.

        Parameters
        ----------
        signals : numpy.ndarray
            2D array of raw signals. Each row is one signal.
        sampling_rate : int, optional
            sampling_rate, by default None
        nperseg : int, optional
            Window size (parameter of STFT), by default None
        noverlap : int, optional
            Overlap (parameter of STFT), by default None

        Returns
        -------
        numpy.ndarray
            3D array of transformed signals.
        """
        if sampling_rate is None:
            sampling_rate = 360
        if nperseg is None:
            nperseg = 64
        if noverlap is None:
            noverlap = int(nperseg / 2)
        list_all = []
        for i in tqdm(range(len(signals))):
            f, t, Sxx = spectrogram(
                signals[i],
                fs=sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap,
                mode="psd",
            )
            list_all.append(Sxx.T[:, :].tolist())
        out = np.array(list_all)
        return out

    def calc_feat_dim(self, samp, win, overlap):
        """Calculates the 2D spectral feature size.

        Parameters
        ----------
        samp : int
            Number of samples.
        win : int
            Window size (parameter of STFT).
        overlap : int
            Overlap (parameter of STFT).

        Returns
        -------
        int
            Height and width.
        """
        hdim = math.floor((samp - overlap) / (win - overlap))
        vdim = math.floor(win / 2 + 1)
        return hdim, vdim


def clean_inf_nan(ds):
    """Cleans the dataset by removing inf and nan.

    Parameters
    ----------
    ds : dict
        Dataset as a dictionary.

    Returns
    -------
    dict
        Cleaned dataset.
    """

    yds = ds["labels"]
    xds = ds["waveforms"]
    rds = ds["beat_feats"]
    indexes = []
    # cleans feature array
    indexes.extend(np.where(np.isinf(rds))[0])
    indexes.extend(np.where(np.isnan(rds))[0])
    rds = np.delete(rds, indexes, axis=0)
    xds = np.delete(xds, indexes, axis=0)
    yds = np.delete(yds, indexes, axis=0)
    # ydsc = [it for ind,it in enumerate(yds) if ind not in indexes]

    return {"waveforms": xds, "beat_feats": rds, "labels": yds}


def clean_IQR(ds, factor=1.5, return_indexes=False):
    """Cleans the dataset by removing outliers using IQR method.

    Parameters
    ----------
    ds : dict
        Dataset.
    factor : float, optional
        Parameter of IQR method, by default 1.5
    return_indexes : bool, optional
        If True returns indexes of outliers, otherwise returns cleaned dataset, by default False

    Returns
    -------
    dict
        Cleaned dataset.
    """
    yds = ds["labels"]
    xds = ds["waveforms"]
    rds = ds["beat_feats"]
    # cleans a 2d array. Each column is a features, rows are samples. Only r.
    ind_outliers = []
    for i in range(rds.shape[1]):
        x = rds[:, i]
        Q1 = np.quantile(x, 0.25, axis=0)
        Q3 = np.quantile(x, 0.75, axis=0)
        IQR = Q3 - Q1
        inds = np.where((x > (Q3 + factor * IQR)) | (x < (Q1 - factor * IQR)))[0]
        # print(len(inds))
        ind_outliers.extend(inds)
    rds = np.delete(rds, ind_outliers, axis=0)
    xds = np.delete(xds, ind_outliers, axis=0)
    yds = np.delete(yds, ind_outliers, axis=0)
    if return_indexes is False:
        return {"waveforms": xds, "beat_feats": rds, "labels": yds}
    else:
        return ind_outliers


def append_ds(ds1, ds2):
    """Appends two datasets together.

    Parameters
    ----------
    ds1 : dict
        Dataset one.
    ds2 : dict
        Dataset two.

    Returns
    -------
    dict
        Final dataset.
    """
    dss = dict()
    dss["waveforms"] = np.vstack((ds1["waveforms"], ds2["waveforms"]))
    dss["beat_feats"] = np.vstack((ds1["beat_feats"], ds2["beat_feats"]))
    dss["labels"] = np.vstack(
        (ds1["labels"].reshape(-1, 1), ds2["labels"].reshape(-1, 1))
    ).flatten()
    return dss


def clean_IQR_class(ds, factor=1.5):
    """Cleans dataset by IQR method for every class separately.

    Parameters
    ----------
    ds : dict
        Dataset.
    factor : float, optional
        Parameter of IQR method, by default 1.5

    Returns
    -------
    dict
        Cleaned dataset.
    """
    for label in list(np.unique(ds["labels"])):
        sliced = slice_data(ds, [label])
        cleaned = clean_IQR(sliced, factor=factor)
        try:
            ds_all = append_ds(ds_all, cleaned)
        except NameError:
            ds_all = cleaned
    return ds_all
