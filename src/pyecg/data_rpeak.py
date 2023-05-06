import os
import math
import time
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from pyecg.data import Data, DataSeq
from pyecg.dataset_config import DATA_DIR, DS1
from scipy import stats


class RpeakData(Data, DataSeq):
    """Provides signal data with annotations as a list. 

    Parameters
    ----------
    base_path : str, optional
        Path of main directory for loading and saving data, by default os.getcwd()
    data_path : str, optional
        Relative path of raw input data regarding to the base_path.
    remove_bl : bool, optional
        If True removes the baseline from the raw signals before extracting beat excerpts, by default False
    lowpass : bool, optional
        Whether to apply low pass filtering to the raw signals, by default False
    sampling_rate : int, optional
        Sampling rate of the signals, by default 360
    cutoff : int, optional
        Parameter of the low pass filter, by default 45
    order : int, optional
        Parameter of the low pass filter, by default 15
    """

    def __init__(
        self,
        base_path=os.getcwd(),
        data_path=DATA_DIR,
        remove_bl=False,
        lowpass=False,
        sampling_rate=360,
        cutoff=45,
        order=15,
    ):
        super().__init__(
            base_path,
            data_path,
            remove_bl,
            lowpass,
            sampling_rate,
            cutoff,
            order,
        )

    def full_annotate(self, record):
        """fully annotate a single record signal.

        Parameters
        ----------
        record : dict
            keys: signal,r_locations,r_labels,rhythms,rhythms_locations
        Returns
        -------
        list
            a list of zeros. At any index with an rpeak the zero is changed to label.
            [00000N00000000000000L00000...]
        """

        signal, r_locations, r_labels, _, _ = record.values()
        full_seq = [0] * len(signal)
        for i, loc in enumerate(r_locations):
            full_seq[loc] = r_labels[i]

        record_full = [signal, full_seq]
        return record_full

    def make_samples_info(self, annotated_records, win_size=30 * 360, stride=256):
        """
        Parameters
        ----------
        annotated_records: [[signal1, full_ann1],[signal2, full_ann2],...]
        win_size : the length of each extracted sample
        stride : the stride for extracted samples.
        interval : the output interval for labels.
        binary : if True 1 is replaced instead of labels

        Returns
        -------
        list
            a 2d list. Each inner list: [record_no,start_win,end_win,label]
            [[record_no,start_win,end_win,label],[record_no,start_win,end_win,label], ...]
            label is a list. Elements are rpeak labels for each interval.
            eg: [[10,500,800, [000000N00000000L00] ],[],...]
        """
        interval = 36  # not necessary, will be calculated in EXGSEQUENCE
        binary = False  # remove it and only do in ECGSEQUENCE

        stride = int(stride)
        win_size = int(win_size)
        interval = int(interval)

        samples_info = []

        # each record
        for rec_id in tqdm(range(len(annotated_records))):
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
            time.sleep(3)

        return samples_info


class ECGSequence(Sequence):
    """
    Generates batches of data according to the meta information provided for each sample. 
    Therefore it is memory efficent and there is no need to put large amount of data in the memory.

    Parameters
    ----------
    data: list
        A list containing a dict for each record, [rec1,rec2,....].
        Each rec is a dict with keys: 'signal','r_locations','r_labels','rhythms','rhythms_locations', 'full_ann'.
    samples_info: list
        [[record_no,start_win,end_win,label],[record_no,start_win,end_win,[labels] ], ...]
        eg: [[10,500,800,[0,0,0,'N',0,0...],[],...]
    class_labels : list, optional
        the list of class labels to convert the output label list to integers, by default None
    batch_size : int, optional
        Batch size, by default 128
    binary : bool, optional
        If True return will be 1 instead of riginal str label, by default True
    raw : bool, optional
        Whether to return the full waveform or the computed features., by default True
    interval : int, optional
        interval for sub segmenting the waveform for feature and label computations, by default 36
    shuffle : bool, optional
        If True shuffle the sample data, by default True
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
        self.batch_size = batch_size
        self.binary = binary
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
            Tuple of numpy arrays (batch_x,batch_y) for sequences(or computed features) and their labels.
            batch_x: If raw is False: (batch,segments,features), otherwise (batch,seqs)
            batch_y: (batch,label_list)
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
            # processing steps on the signal fraggment
            if self.raw == False:
                seq = self.proc_steps(seq)
                batch_x.append(list(seq))
            else:
                batch_x.append(seq)
            if self.class_labels != None:
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

    def proc_steps(self, seq):
        # get a 1d seq and return a multidim list.
        # each feature is an aggragate of a subseq.
        b = int(len(seq) / self.interval)
        subseqs = []
        for i in range(b):
            subseq = seq[i * self.interval : (i + 1) * self.interval]
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
        feats = np.vstack((f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14))
        return feats.T
