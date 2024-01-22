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

import numpy as np  # noqa
from tensorflow.keras.utils import Sequence  # noqa: E402
from tqdm import tqdm  # noqa: E402

from pyheartlib.data import Data, DataSeq  # noqa: E402
from pyheartlib.features import get_wf_feats  # noqa: E402
from pyheartlib.io import load_data  # noqa: E402

# import time  # noqa: E402


class RpeakData(Data, DataSeq):
    """Processes ECG records to make a dataset holding records along with
    metadata about signal excerpts.

    It has a method that can generate metadata for signal excerpts.
    The metadata are generated using the sliding window approach.
    For each excerpt of a signal the onset, offset, and its annotation
    is recorded. The metadata list for an excerpt is structured as:
    [record_id, onset, offset, annotation].
    Annotation for an excerpt is a list of zeros, except for any interval
    where there is an R-peak label.
    Example metadata for an excerpt:
    [10, 500, 800, [0, 0, 0, 'N', 0, 'N', 0, 'N', 0, ...]]

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
    >>> from pyheartlib.data_rpeak import RpeakData
    >>> # Make an instance of the RpeakData
    >>> rpeak_data = RpeakData(
    >>>     base_path="data", remove_bl=False, lowpass=False,
    >>>     progress_bar=False)
    >>> # Define records
    >>> train_set = [201, 203]
    >>> # Create the dataset
    >>> # The win_size specifies the length of the excerpts
    >>> rpeak_data.save_dataset(
    >>>     rec_list=train_set,
    >>>     file_name="train.rpeak",
    >>>     win_size=5 * 360,
    >>>     stride=360,
    >>>     interval=72,
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
            the original signal with zero elements except at any
            R-peak index which has the R-peak label instead.
            E.g.: [0,0,0,'N',0,'N',0,'N', ...]

        """

        signal, r_locations, r_labels, _, _ = record.values()
        if r_labels is None or r_locations is None:
            raise RuntimeError("R-peaks are not provided!")
        full_ann = [0] * signal.shape[0]
        for i, loc in enumerate(r_locations):
            full_ann[loc] = r_labels[i]

        record_full = (signal, full_ann)
        return record_full

    def gen_samples_info(
        self, annotated_records, win_size=30 * 360, stride=256, **kwargs
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
        interval : int, optional
            Controls the degree of granularity of labels in
            the annotation list, by default 36
        binary : bool, optional
            If True, any non-zero label will be converted to
            1 in the annotation list, by default False

        Returns
        -------
        list
            A nested list. Each inner list is structured as:
            [record id, onset, offset, annotation].

            Annotation is a list with zeros except for any interval that
            there is an R-peak label.

            E.g.: [[10, 500, 800, [0,0,0,'N',0,'N',0,'N',0,....] ], ...]

        """

        # interval : the output interval for labels.
        # binary : if True, 1 is replaced instead of labels
        # interval and binary are optional parameters, ECGSequence does it
        kargs = {"interval": 36, "binary": False}  # default values
        kargs.update(kwargs)
        interval = kargs["interval"]
        binary = kargs["binary"]

        stride = int(stride)
        win_size = int(win_size)
        interval = int(interval)

        samples_info = []

        # each record
        for rec_id in tqdm(
            range(len(annotated_records)), disable=DataSeq.progress_bar
        ):
            signal = annotated_records[rec_id]["signal"]
            full_ann = annotated_records[rec_id]["full_ann"]
            assert signal.shape[0] == len(
                full_ann
            ), "signal and annotation must have the same length!"

            # each extracted segment
            end = win_size
            while end < len(full_ann):
                start = int(end - win_size)
                seg = full_ann[start:end]
                labels_seq = []
                # this part is optional, ECGSequence does it
                for i in range(0, len(seg), interval):
                    subseg = seg[i : i + interval]
                    if any(subseg):
                        nonzero = [sm for sm in subseg if sm != 0]
                        lb = nonzero[0]
                        if binary:
                            lb = 1
                        labels_seq.append(lb)
                    else:
                        lb = 0
                        labels_seq.append(lb)

                samples_info.append([rec_id, start, end, labels_seq])
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
        E.g. : [[10, 500, 800, [0,0,0,'N',0,'N',0,'N',0,...] ], ...].
    class_labels : list, optional
        Classes as a list for converting the output annotations to integers
        such as: [0, "N", "V"] => [0,1,2], by default None
    batch_size : int, optional
        Number of samples in each batch, by default 128
    binary : bool, optional
        If True, any non-zero label will be converted to
        1 in the output annotation list, by default True
    raw : bool, optional
        Whether to return the waveform or the computed features,
        by default True
    interval : int, optional
        Sub-segmenting interval for feature computations and label
        assignments. Controls the degree of granularity for labels in the
        output annotation list and the number of sub-segments, by default 36
    shuffle : bool, optional
        If True, after each epoch the samples are shuffled, by default True

    Examples
    --------
    >>> from pyheartlib.data_rpeak import ECGSequence
    >>> trainseq = ECGSequence(
    >>>     annotated_records, samples_info, binary=False, batch_size=2,
    >>> raw=True, interval=72)

    Notes
    -----
    Returns a tuple containing two elements when its object is utilized in
    this way: ECGSequence_object[BatchNo].

    The first element (Batch_x) contains data samples and the
    second one (Batch_y) their associated annotation.

    Batch_x contains the signal excerpts or their features (Batch_wave).

    If `raw` is False, Batch_wave has the shape of
    (Batch size, Number of channels, Number of sub-segments, Number
    of features), otherwise, it has the shape of (Batch_size, Number
    of channels, Length of excerpt).

    Batch_y has the shape of (Batch size, Length of annotation list).

    """

    def __init__(
        self,
        data,
        samples_info,
        class_labels=None,
        binary=True,
        batch_size=128,
        raw=True,
        interval=36,
        shuffle=True,
    ):
        self.data = data
        self.samples_info = samples_info
        self.class_labels = class_labels
        self.binary = binary
        self.batch_size = batch_size
        self.raw = raw
        self.interval = interval
        self.shuffle = shuffle
        self.on_epoch_end()
        if self.class_labels is not None:
            self.binary = False

    def __len__(self):
        return math.ceil(len(self.samples_info) / self.batch_size)

    def __getitem__(self, idx):
        """
        Returns
        -------
        tuple
            Contains batch_x, batch_y as numpy arrays.

            batch_x has the shape of
            (batch_size, #channels, #sub-segments, #features)
            if raw is False, otherwise it has the shape of
            (batch_size, #channels, len_seq).

            batch_y has the shape of (batch_size,len_annotation_list)
        """
        batch_samples = self.samples_info[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = []
        batch_y = []
        for sample in batch_samples:
            # eg sample:[10,500,800,[0,0,0,'N',0,0...]>>[rec,start,end,ann]
            rec_id = sample[0]
            start = sample[1]
            end = sample[2]
            # ignoring provided labels in samples_info and get it
            # directly from the data
            # ann = sample[3]
            seq = self.data[rec_id]["signal"][start:end]
            full_ann_seq = self.data[rec_id]["full_ann"][start:end]
            ann = self.gen_annotation(full_ann_seq)
            # compute signal waveform features for fragments
            if self.raw is False:
                feats = []
                for ch in range(seq.shape[1]):
                    feats_ch = self.compute_wf_feats(seq[:, ch])
                    feats.append(list(feats_ch))
                batch_x.append(feats)
            else:
                batch_x.append(seq)
            if self.class_labels is not None:
                ann = self.get_integer(ann)
            if self.binary:
                ann = self.get_binary(ann)
            batch_y.append(ann)
        batch_x = np.array(batch_x)
        if self.raw is True:
            batch_x = np.swapaxes(batch_x, 1, 2)
        batch_y = np.array(batch_y)
        return batch_x, batch_y

    def on_epoch_end(self):
        """After each epoch shuffles the samples."""
        if self.shuffle:
            np.random.shuffle(self.samples_info)

    def get_integer(self, ann):
        """Converts labels in an annotation list to integers."""
        int_label = [list(self.class_labels).index(item) for item in ann]
        return int_label

    def get_binary(self, ann):
        """Converts labels in an annotation list to 0 or 1."""
        binary_label = [1 if item != 0 else 0 for item in ann]
        return binary_label

    def gen_annotation(self, seg):
        """Generate annotation list.

        The returned list contains zeros except for any interval that has
        an R-peak label.

        """

        # this function is used if intentions is to not use
        # provided labels in samples_info
        labels_seq = []
        # each subsegment
        for i in range(0, len(seg), self.interval):
            subseg = seg[i : i + self.interval]
            if any(subseg):
                nonzero = [sm for sm in subseg if sm != 0]
                lb = nonzero[0]
                # if self.binary:
                # lb=1
                labels_seq.append(lb)
            else:
                lb = 0
                labels_seq.append(lb)
        return labels_seq

    def compute_wf_feats(self, seq):
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
