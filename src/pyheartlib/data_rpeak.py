import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import time
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from pyheartlib.data import Data, DataSeq
from pyheartlib.features import get_wf_feats


class RpeakData(Data, DataSeq):
    """Provides signal data with annotations as a list.

    Parameters
    ----------
    base_path : str, optional
        Path of main directory for loading and saving data, by default None
    remove_bl : bool, optional
        If True, the baseline is removed from the raw signals before extracting
        beat excerpts, by default False
    lowpass : bool, optional
        Whether to apply low pass filtering to the raw signals, by default False
    sampling_rate : int, optional
        Sampling rate of the signals, by default 360
    cutoff : int, optional
        Parameter of the low pass filter, by default 45
    order : int, optional
        Parameter of the low pass filter, by default 15
    progress_bar : bool, optional
        If True, progress bar is shown, by default True
    """

    def __init__(
        self,
        base_path=None,
        remove_bl=False,
        lowpass=False,
        sampling_rate=360,
        cutoff=45,
        order=15,
        progress_bar=True,
    ):
        super().__init__(
            base_path,
            remove_bl,
            lowpass,
            sampling_rate,
            cutoff,
            order,
        )
        DataSeq.progress_bar = not progress_bar

    def full_annotate(self, record):
        """Fully annotate a signal.

        Parameters
        ----------
        record : dict
            Record as a dictionary with keys: 'signal', 'r_locations',
            'r_labels', 'rhythms', 'rhythms_locations'.

        Returns
        -------
        list
            A list of signal and full_ann: [signal, full_ann].

            First element is the original signal (1D ndarray).

            Second element is a list that has the same length as the original signal with
            zero elements except at any rpeak index which has the rpeak annotation instead.
            [00000N00000000000000L00000...]
        """

        signal, r_locations, r_labels, _, _ = record.values()
        full_ann = [0] * len(signal)
        for i, loc in enumerate(r_locations):
            full_ann[loc] = r_labels[i]

        record_full = [signal, full_ann]
        return record_full

    def make_samples_info(self, annotated_records, win_size=30 * 360, stride=256):
        """Creates a list of signal excerpts meta info and their corresponding annotation.

        For each excerpt, record id, onset, and offset point of it on the original signal is recorded.

        Parameters
        ----------
        annotated_records: [[signal1, full_ann1],[signal2, full_ann2],...]
        win_size : the length of each extracted sample
        stride : the stride for extracted samples.

        Returns
        -------
        list
            A 2d list. Each inner list is like [record_no, start_win, end_win, Label].
            Label is a list and its elements are rpeak annotations for each interval.
            E.g.: [[10,500,800,[000000N00000000L00]], ...]
        """
        interval = 36  # not necessary, will be calculated in EXGSEQUENCE
        binary = False  # remove it and only do in ECGSEQUENCE
        # interval : the output interval for labels.
        # binary : if True 1 is replaced instead of labels

        stride = int(stride)
        win_size = int(win_size)
        interval = int(interval)

        samples_info = []

        # each record
        for rec_id in tqdm(range(len(annotated_records)), disable=DataSeq.progress_bar):
            signal = annotated_records[rec_id]["signal"]
            full_ann = annotated_records[rec_id]["full_ann"]
            assert len(signal) == len(
                full_ann
            ), "signal and annotation must have the same length!"

            # each extracted segment
            end = win_size
            while end < len(full_ann):
                start = int(end - win_size)
                seg = full_ann[start:end]
                labels_seq = []  # not necesssary anymore,will be done in SEQUENCE
                # each subsegment
                for i in range(0, len(seg), interval):
                    subseg = seg[i : i + interval]
                    if any(subseg):
                        nonzero = [l for l in subseg if l != 0]
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
    """
    Generates batches of data according to the meta information provided for each sample.

    It is memory efficient and there is no need to put large amounts of data in the memory.

    Parameters
    ----------
    data: list
        A list containing a dict for each record, [rec1,rec2,....].
        Each rec is a dict with keys: 'signal','r_locations','r_labels',
        'rhythms','rhythms_locations', 'full_ann'.
    samples_info: list
        [[record_no,start_win,end_win,label],[record_no,start_win,end_win,[labels] ], ...]
        eg: [[10,500,800,[0,0,0,'N',0,0...],[],...]
    class_labels : list, optional
        Class labels to convert the output label list to integers
        such as [0, "N", "V"] where "V" will be converted to 2, by default None
    batch_size : int, optional
        Batch size, by default 128
    binary : bool, optional
        If True, any non-zero item will be converted to one in the output annotation, by default True
    raw : bool, optional
        Whether to return the full waveform or the computed features, by default True
    interval : int, optional
        Interval for sub-segmenting the waveform for feature and
        label computations, by default 36
    shuffle : bool, optional
        If True, after each epoch the samples are shuffled, by default True
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
        if self.class_labels != None:
            self.binary = False

    def __len__(self):
        return math.ceil(len(self.samples_info) / self.batch_size)

    def __getitem__(self, idx):
        """
        Returns
        -------
        Tuple
            Contains batch_x, batch_y as numpy arrays.

            batch_x has the shape of (batch_size,#sub-segments,#features) if raw is False,
            otherwise it has the shape of (batch_size,len_seq).

            batch_y has the shape of (batch_size,len_label_list)
        """
        batch_samples = self.samples_info[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_x = []
        batch_y = []
        for sample in batch_samples:
            # eg sample:[10,500,800,[0,0,0,'N',0,0...] >>[rec,start,end,label]
            rec_id = sample[0]
            start = sample[1]
            end = sample[2]
            # ignoring provided labels in samples_info and get it directly from the data
            # label = sample[3]
            seq = self.data[rec_id]["signal"][start:end]
            full_ann_seq = self.data[rec_id]["full_ann"][start:end]
            label = self.get_label(full_ann_seq)
            # compute signal waveform features for fragments
            if self.raw is False:
                seq = self.compute_wf_feats(seq)
                batch_x.append(list(seq))
            else:
                batch_x.append(seq)
            if self.class_labels is not None:
                label = self.get_integer(label)
            if self.binary:
                label = self.get_binary(label)
            batch_y.append(label)
        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        """After each epoch shuffles the samples."""
        if self.shuffle:
            np.random.shuffle(self.samples_info)

    def get_integer(self, label):
        # text label to integer >> 0,1,2,...
        int_label = [list(self.class_labels).index(item) for item in label]
        return int_label

    def get_binary(self, label):
        # text label to integer 0,1 label
        binary_label = [1 if item != 0 else 0 for item in label]
        return binary_label

    def get_label(self, seg):
        # this function is used if intentions is to not use provided labels in samples_info
        labels_seq = []
        # each subsegment
        for i in range(0, len(seg), self.interval):
            subseg = seg[i : i + self.interval]
            if any(subseg):
                nonzero = [l for l in subseg if l != 0]
                lb = nonzero[0]
                # if self.binary:
                # lb=1
                labels_seq.append(lb)
            else:
                lb = 0
                labels_seq.append(lb)
        return labels_seq

    def compute_wf_feats(self, seq):
        return get_wf_feats(seq, self.interval)
