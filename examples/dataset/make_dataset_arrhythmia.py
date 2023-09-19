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
import pandas as pd

from pyheartlib.data_arrhythmia import ArrhythmiaData, ECGSequence
from pyheartlib.io import load_data

# AFIBs
train_set = [201, 203]
# test_set = [202, 210, 219, 221, 222]

arrhythmia_data = ArrhythmiaData(
    base_path="./data", remove_bl=False, lowpass=False, progress_bar=True
)
annotated_records, samples_info = arrhythmia_data.save_samples(
    rec_list=train_set, file_name="train.arr", win_size=3600, stride=64
)
annotated_records, samples_info = load_data("./data/train.arr")

labels = []
for sample in samples_info:
    labels.append(sample[3])
df = pd.DataFrame(
    np.unique(labels, return_counts=True), index=["Label", "Count"]
)
print(df)

class_labels = list(np.unique(labels))

# if raw=True
trainseq = ECGSequence(
    annotated_records, samples_info, class_labels=None, batch_size=3, raw=True
)
bt = 0  # batch number
batch_label = trainseq[bt][1]  # excerpt label
batch_seq = trainseq[bt][0][0]  # excerpt values
batch_rri = trainseq[bt][0][1]  # rr intervals
batch_rri_feat = trainseq[bt][0][2]  # calculated rri features
print("batch_label shape:", batch_label.shape)
print(
    "batch_seq shape:",
    batch_seq.shape,
    " , batch_rri shape:",
    batch_rri.shape,
    " , batch_rri_feat shape:",
    batch_rri_feat.shape,
)

# if raw=False
trainseq = ECGSequence(
    annotated_records, samples_info, class_labels=None, batch_size=3, raw=False
)
bt = 0  # batch number
batch_label = trainseq[bt][1]  # excerpt label
batch_seq = trainseq[bt][0][0]  # excerpt values
batch_rri = trainseq[bt][0][1]  # rr intervals
batch_rri_feat = trainseq[bt][0][2]  # calculated rri features
print("batch_label shape:", batch_label.shape)
print(
    "batch_seq shape:",
    batch_seq.shape,
    " , batch_rri shape:",
    batch_rri.shape,
    " , batch_rri_feat shape:",
    batch_rri_feat.shape,
)
