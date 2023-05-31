import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import time
import numpy as np
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
from pyheartlib.data import Data, DataSeq
from pyheartlib.features import get_hrv_features, get_wf_feats


class ArrhythmiaData(Data, DataSeq):
    """Provides signal data with single value annotations.

    Parameters
    ----------
    base_path : str, optional
        Path of main directory for loading and saving data, by default None
    remove_bl : bool, optional
        If True, baseline will be removed from the raw signals before extracting beat excerpts, by default False
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
            arrhythmia types at each index:
            ['(N','(N', '(N','AFIB','AFIB','AFIB',...].
        """

        signal, _, _, rhythms, rhythms_locations = record.values()
        sig_length = len(signal)
        full_ann = []
        full_ann = ["unlab"] * len(signal)
        for i in range(len(rhythms_locations)):
            remained = sig_length - rhythms_locations[i]
            full_ann[rhythms_locations[i] :] = [rhythms[i]] * remained
        record_full = [signal, full_ann]
        return record_full

    def make_samples_info(self, annotated_records, win_size=30 * 360, stride=36):
        """Creates a list of signal excerpts meta info and their corresponding annotation.

        For each excerpt, record id, onset, and offset point of it on the original signal is recorded.

        Parameters
        ----------
        annotated_records : list
            A list containing a dict for each record. [rec1,rec2,....].
            Each rec is a dict with keys: 'signal', 'r_locations', 'r_labels',
            'rhythms', 'rhythms_locations', 'full_ann'.
        win_size : int, optional
            Windows size, by default 30*360
        stride : int, optional
            Stride, by default 36

        Returns
        -------
        list
            A 2d list. Each inner list is like [record_id, start_win, end_win, Label].
            E.g. : [[10,500,800,'AFIB'], [10,700,900,'(N'], ...]
        """

        stride = int(stride)
        win_size = int(win_size)

        samples_info = []
        for rec_id in tqdm(range(len(annotated_records)), disable=DataSeq.progress_bar):
            signal = annotated_records[rec_id]["signal"]
            full_ann = annotated_records[rec_id]["full_ann"]
            assert len(signal) == len(
                full_ann
            ), "signal and annotation must have the same length!"

            end = win_size
            while end < len(full_ann):
                start = int(end - win_size)
                # unique arrhythmia type in each segment
                if len(set(full_ann[start:end])) == 1:
                    label = full_ann[start]
                    samples_info.append([rec_id, start, end, label])
                end += stride
            # time.sleep(3)
        return samples_info


class ECGSequence(Sequence):
    """
    Generates batches of data according to the meta information provided for each sample.

    It is memory efficient and there is no need to put large amount of data in the memory.

    Parameters
    ----------
    data : list
        A list containing a dict for each record, [rec1,rec2,....].
        Each rec is a dict with keys: 'signal', 'r_locations', 'r_labels',
        'rhythms', 'rhythms_locations', 'full_ann'.
    samples_info : list
        A list of lists. Each inner list is like [record_id, start_win, end_win, label].
        E.g. : [[10,500,800,'AFIB'], [10,700,900,'(N'], ...].
    class_labels : list, optional
        Arrhythmia types as a list such as ["(N", "(VT"], by default None
    batch_size : int, optional
        Batch size, by default 128
    raw : bool, optional
        Whether to return the full waveform or the computed features, by default True
    interval : int, optional
        interval for sub-segmenting the signal for waveform feature computation, by default 36
    shuffle : bool, optional
        If True, after each epoch the samples are shuffled, by default True
    rri_length : int, optional
        Zero padding (right) rri as they have different length for each excerpt, by default 150
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
        rri_length=150,
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.raw = raw
        self.interval = interval
        self.data = data
        self.samples_info = samples_info
        self.class_labels = class_labels
        self.rri_length = rri_length
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.samples_info) / self.batch_size)

    def __getitem__(self, idx):
        """
        Returns
        -------
        Tuple
            Contains batch_x, batch_y, where batch_x is a list of numpy arrays
            of batch_seq, batch_rri, batch_rri_feat.

            If raw is False, batch_seq has the shape of (batch_size,#sub-segments,#features),
            otherwise, it has the shape of (batch_size,seq_len).

            batch_y has the shape of (batch_size,1).
        """
        batch_samples = self.samples_info[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_seq = []
        batch_label = []
        batch_rri = []
        for sample in batch_samples:
            # eg sample:[10,500,800,'AFIB'] ::: [rec,start,end,label]
            rec_id = sample[0]
            start = sample[1]
            end = sample[2]
            label = sample[3]
            labelm = label
            if self.class_labels is not None:
                labelm = self.get_integer(label)
            batch_label.append(labelm)
            seq = self.data[rec_id]["signal"][start:end]
            # compute signal waveform features for fragments
            if self.raw is False:
                seq = self.compute_wf_feats(seq)
                batch_seq.append(list(seq))
            else:
                batch_seq.append(seq)
            rri = self.get_rri(rec_id, start, end)
            batch_rri.append(rri)
        batch_rri = np.array(batch_rri)
        batch_rri_feat = self.compute_rri_features(batch_rri * 1000)
        # return np.array(batch_seq),np.array(batch_label)
        batch_x = [np.array(batch_seq), batch_rri, batch_rri_feat]
        batch_y = np.array(batch_label)
        return batch_x, batch_y

    def on_epoch_end(self):
        """After each epoch shuffles the samples."""
        if self.shuffle:
            np.random.shuffle(self.samples_info)

    def get_integer(self, label):
        """Converts text label to integer"""
        return self.class_labels.index(label)

    def get_rri(self, rec_id, start, end):
        """Computes RR intervals"""
        fs = 360
        r_locations = np.asarray(self.data[rec_id]["r_locations"])  # entire record
        inds = np.where((r_locations >= start) & (r_locations < end))
        rpeak_locs = list(
            r_locations[inds]
        )  # rpeak locs within start to end of excerpt
        rri = [
            (rpeak_locs[i + 1] - rpeak_locs[i]) / float(fs)
            for i in range(0, len(rpeak_locs) - 1)  # calculate rr intervals
        ]
        # worst case senario for a 30sec with bpm of 300---rri len=150
        rri_zeropadded = np.zeros(self.rri_length)
        rri_zeropadded[: len(rri)] = rri
        rri_zeropadded = rri_zeropadded.tolist()
        return rri_zeropadded

    def compute_rri_features(self, rri_array):
        """Computes some stat features for rr intervals"""
        # mask zero padding in the array
        import numpy.ma as ma

        mask = np.invert(rri_array.astype(bool))
        rri_masked = ma.masked_array(rri_array, mask=mask)
        return get_hrv_features(rri_masked)

    def compute_wf_feats(self, seq):
        return get_wf_feats(seq, self.interval)
