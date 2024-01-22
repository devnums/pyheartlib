#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math  # noqa: E402

import numpy as np  # noqa: E402
from tensorflow.keras.utils import Sequence  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pyheartlib.data import Data, DataSeq  # noqa: E402
from pyheartlib.features import get_hrv_features, get_wf_feats  # noqa: E402
from pyheartlib.io import load_data  # noqa: E402

# import time  # noqa: E402


class RhythmData(Data, DataSeq):
    """Processes ECG records to make a dataset holding records along with
    metadata about signal excerpts.

    It has a method that can generate metadata for signal excerpts.
    The metadata are generated using the sliding window approach.
    For each excerpt of a signal the onset, offset, and its annotation
    is recorded. The metadata list for an excerpt is structured as:
    [record_id, onset, offset, annotation].
    Annotation for an excerpt is a single label.
    Example metadata for an excerpt: [10, 500, 800, 'AFIB'].

    Parameters
    ----------
    base_path : str, optional
        Path of the main directory for storing the original and
        processed data, by default None
    remove_bl : bool, optional
        If True, the baseline wander is removed from the original signals
        prior to extracting excerpts, by default False
    lowpass : bool, optional
        Whether or not to apply low-pass filter to the original signals,
        by default False
    cutoff : int, optional
        Parameter of the low pass-filter, by default 45
    order : int, optional
        Parameter of the low pass-filter, by default 15
    progress_bar : bool, optional
        Whether to display a progress bar, by default True
    processors : list, optional
        Ordered list of functions' names for preprocessing the raw signals.
        Each function takes a one-dimensional NumPy array as its input and
        returns an array of the same length.

    Example
    -------
    >>> from pyheartlib.data_rhythm import RhythmData
    >>> # Make an instance of the RhythmData
    >>> rhythm_data = RhythmData(
    >>>     base_path="data", remove_bl=False, lowpass=False,
    >>>     progress_bar=False)
    >>> # Define records
    >>> train_set = [201, 203]
    >>> # Create the dataset
    >>> rhythm_data.save_dataset(
    >>>   rec_list=train_set, file_name="train.arr", win_size=3600, stride=64
    >>> )

    """

    def __init__(
        self,
        base_path=None,
        remove_bl=False,
        lowpass=False,
        cutoff=45,
        order=15,
        progress_bar=True,
        **kwargs,
    ):
        super().__init__(
            base_path,
            remove_bl,
            lowpass,
            cutoff,
            order,
            **kwargs,
        )
        DataSeq.progress_bar = not progress_bar

    def full_annotate(self, record):
        """Returns a signal along with an annotation of the same length.

        Parameters
        ----------
        record : dict
            Record as a dictionary with keys: *signal*, *r_locations*,
            *r_labels*, *rhythms*, *rhythms_locations*.

        Returns
        -------
        tuple
            Two items: (signal, full_ann).

            First element is the original signal (1D ndarray).

            Second element is a list that has the same length as
            the original signal with rhythm types as its elements:
            ['(N','AFIB','AFIB', ...].
        """

        signal, _, _, rhythms, rhythms_locations = record.values()
        if rhythms is None or rhythms_locations is None:
            raise RuntimeError("Rhythms are not provided!")
        sig_length = signal.shape[0]
        full_ann = []
        full_ann = ["unlab"] * signal.shape[0]
        for i in range(len(rhythms_locations)):
            remained = sig_length - rhythms_locations[i]
            full_ann[rhythms_locations[i] :] = [rhythms[i]] * remained
        record_full = (signal, full_ann)
        return record_full

    def gen_samples_info(
        self, annotated_records, win_size=30 * 360, stride=36, **kwargs
    ):
        """Generates metadata for signal excerpts.

        The metadata are generated using the sliding window approach.
        For each excerpt of a signal the onset, offset, and its annotation
        is recorded. The metadata list for an excerpt is structured as:
        [record id, onset, offset, annotation]

        Parameters
        ----------
        annotated_records : list
            List of records ([rec1_dict, ...]).
            Each record is a dictionary with keys: *signal*, *r_locations*,
            *r_labels*, *rhythms*, *rhythms_locations*, *full_ann*.
        win_size : int, optional
            Sliding window length, by default 30*360
        stride : int, optional
            Stride of the sliding window, by default 36

        Returns
        -------
        list
            A nested list. Each inner list is structured as:
            [record id, onset, offset, annotation].
            E.g. : [[10,500,800,'AFIB'], [10,700,900,'(N'], ...]
        """

        stride = int(stride)
        win_size = int(win_size)

        samples_info = []
        for rec_id in tqdm(
            range(len(annotated_records)), disable=DataSeq.progress_bar
        ):
            signal = annotated_records[rec_id]["signal"]
            full_ann = annotated_records[rec_id]["full_ann"]
            assert signal.shape[0] == len(
                full_ann
            ), "signal and annotation must have the same length!"

            end = win_size
            while end < len(full_ann):
                start = int(end - win_size)
                # unique rhythm type in each segment
                if len(set(full_ann[start:end])) == 1:
                    ann = full_ann[start]
                    samples_info.append([rec_id, start, end, ann])
                end += stride
            # time.sleep(3)
        return samples_info


class ECGSequence(Sequence):
    """Generates samples of data in batches.

    The excerpt for each sample is extracted based on the provided metadata.
    The use of metadata instead of excerpts has the advantage of reducing
    the RAM requirement, especially when numerous excerpts are required from
    the raw signals. By using metadata about the excerpts, they are
    extracted in batches whenever they are needed.

    Parameters
    ----------
    data : list
        A list containing a dictionary for each record: [rec1,rec2,….].
        Each record is a dictionary with keys:
        *signal*, *r_locations*, *r_labels*,
        *rhythms*, *rhythms_locations*, *full_ann*.
    samples_info : list
        A nested list of metadata for excerpts. For each excerpt,the metadata
        is structured as a list: [record_id, onset, offset, annotation].
        E.g. : [[10,500,800,'AFIB'], ...].
    class_labels : list, optional
        Classes as a list for converting the output annotations to integers
        such as: ["(N", "(VT"] => [0,1], by default None
    batch_size : int, optional
        Number of samples in each batch, by default 128
    raw : bool, optional
        Whether to return the waveform or the computed features,
        by default True
    interval : int, optional
        Interval for sub-segmenting the signal for waveform
        feature computation, by default 36
    shuffle : bool, optional
        If True, after each epoch the samples are shuffled, by default True
    rri_output: bool, optional
        Whether to return RR-intervals and their features.
        If False, returns only waveforms.
    rri_length : int, optional
        Length of the output RR-intervals list.
        It is zero-padded on the right side, by default 150

    Examples
    --------
    >>> from pyheartlib.data_rhythm import ECGSequence
    >>> trainseq = ECGSequence(
    >>>    annotated_records,
    >>>    samples_info,
    >>>    class_labels=None,
    >>>    batch_size=3,
    >>>    raw=True,
    >>>    interval=36,
    >>>    shuffle=False,
    >>>    rri_output=True,
    >>>    rri_length=25
    >>> )

    Notes
    -----
    Returns a tuple containing two elements when its object is utilized in
    this way: ECGSequence_object[BatchNo].

    The first element (Batch_x) contains data samples and the
    second one (Batch_y) their associated annotation.

    If `rri_output` is True, Batch_x is a list of NumPy arrays of
    Batch_wave, Batch_rri, Batch_rri_feat.

    Batch_wave contains signal excerpts or their features, Batch_rri
    contains RR-intervals, and Batch_rri_feat contains RR-interval features.

    If `rri_output` is False, Batch_x contains Batch_wave only.

    If `raw` is False, Batch_wave has the shape of
    (Batch size, Number of channels, Number of sub-segments, Number
    of features), otherwise, it has the shape of
    (Batch size, Number of channels, Length of excerpt).

    Batch_y has the shape of (Batch size, ).
    """

    def __init__(
        self,
        data,
        samples_info,
        class_labels=None,
        batch_size=128,
        raw=True,
        interval=36,
        shuffle=True,
        rri_output=True,
        rri_length=150,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.raw = raw
        self.interval = interval
        self.data = data
        self.samples_info = samples_info
        self.class_labels = class_labels
        self.rri_output = rri_output
        self.rri_length = rri_length
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples_info) / self.batch_size)

    def __getitem__(self, idx):
        """
        Returns
        -------
        tuple
            Contains batch_x and batch_y.

            If `rri_output` is True, batch_x is a list of Numpy arrays of
            batch_wave, batch_rri, batch_rri_feat. If `rri_output` is
            False, batch_x contains batch_wave only.

            If `raw` is False, batch_wave has the shape of
            (batch_size, #channels, #sub-segments, #features), otherwise,
            it has the shape of (batch_size, #channels, wave_len).

            batch_y has the shape of (batch_size, 1).
        """
        batch_samples = self.samples_info[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_wave = []
        batch_annotation = []
        batch_rri = []
        for sample in batch_samples:
            # eg sample:[10,500,800,'AFIB'] ::: [rec,start,end,ann]
            rec_id = sample[0]
            start = sample[1]
            end = sample[2]
            ann = sample[3]
            anno = ann
            if self.class_labels is not None:
                anno = self.get_integer(ann)
            batch_annotation.append(anno)
            seq = self.data[rec_id]["signal"][start:end]
            # compute signal waveform features for fragments
            if self.raw is False:
                feats = []
                for ch in range(seq.shape[1]):
                    feats_ch = self.compute_wf_feats(seq[:, ch])
                    feats.append(list(feats_ch))
                batch_wave.append(feats)
            else:
                batch_wave.append(seq)
            if self.rri_output:
                rri = self.get_rri(rec_id, start, end)
                batch_rri.append(rri)
        batch_wave = np.array(batch_wave)
        if self.rri_output:
            batch_rri = np.array(batch_rri)
            batch_rri_feat = self.compute_rri_features(batch_rri * 1000)
        if self.raw is True:
            batch_wave = np.swapaxes(batch_wave, 1, 2)
        if self.rri_output:
            batch_x = [batch_wave, batch_rri, batch_rri_feat]
        else:
            batch_x = batch_wave
        batch_y = np.array(batch_annotation)
        return batch_x, batch_y

    def on_epoch_end(self):
        """After each epoch shuffles the samples."""
        if self.shuffle:
            np.random.shuffle(self.samples_info)

    def get_integer(self, ann):
        """Converts a text label to integer."""
        return self.class_labels.index(ann)

    def get_rri(self, rec_id, start, end):
        """Computes RR-intervals."""
        sampling_rate = RhythmData.sampling_rate
        r_locations = self.data[rec_id]["r_locations"]  # entire record
        if r_locations is None:
            raise RuntimeError("R-peak locations are not provided!")
        r_locations = np.asarray(r_locations)
        inds = np.where((r_locations >= start) & (r_locations < end))
        rpeak_locs = list(
            r_locations[inds]
        )  # rpeak locs within start to end of excerpt
        rri = [
            (rpeak_locs[i + 1] - rpeak_locs[i]) / float(sampling_rate)
            for i in range(0, len(rpeak_locs) - 1)  # calculate rr intervals
        ]
        # worst case senario for a 30sec with bpm of 300---rri len=150
        rri_zeropadded = np.zeros(self.rri_length)
        rri_zeropadded[: len(rri)] = rri
        rri_zeropadded = rri_zeropadded.tolist()
        return rri_zeropadded

    def compute_rri_features(self, rri_array):
        """Computes some statistical features for RR-intervals."""
        # mask zero padding in the array
        import numpy.ma as ma

        mask = np.invert(rri_array.astype(bool))
        rri_masked = ma.masked_array(rri_array, mask=mask)
        return get_hrv_features(rri=rri_masked)

    def get_rri_features_names(self):
        """Get RR-interval feature names."""
        return get_hrv_features(only_names=True)

    def compute_wf_feats(self, seq):
        """Computes waveform features."""
        return get_wf_feats(sig=seq, interval=self.interval, only_names=False)

    def get_wf_feats_names(self):
        """Get waveform feature names."""
        return get_wf_feats(only_names=True)


def load_dataset(file_path=None):
    """Loads the dataset.

    Parameters
    ----------
    file_path : str, optional
        Path of the dataset, by default None

    """
    return load_data(file_path)
