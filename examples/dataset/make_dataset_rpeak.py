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

from pyheartlib.data_rpeak import ECGSequence, RpeakData
from pyheartlib.io import load_data

train_set = [201, 203]

rpeak_data = RpeakData(
    base_path="./data", remove_bl=False, lowpass=False, progress_bar=True
)
annotated_records, samples_info = rpeak_data.save_samples(
    rec_list=train_set, file_name="train.rpeak", win_size=30 * 360, stride=360
)
annotated_records, samples_info = load_data("./data/train.rpeak")

labels = []
for sample in samples_info:
    labels.append(sample[3])
df = pd.DataFrame(
    np.unique(labels, return_counts=True), index=["Label", "Count"]
)
print(df)

# if raw=True
trainseq = ECGSequence(
    annotated_records,
    samples_info,
    binary=False,
    batch_size=3,
    raw=True,
    interval=72,
)
bt = 0  # batch number
batch_y = trainseq[bt][1]  # excerpt label
batch_x = trainseq[bt][0]  # excerpt values
print("batch_x shape:", batch_x.shape, ", batch_y shape:", batch_y.shape)


# if raw=False
trainseq = ECGSequence(
    annotated_records,
    samples_info,
    binary=False,
    batch_size=3,
    raw=False,
    interval=72,
)
i = 0
batch_y = trainseq[bt][1]  # excerpt label
batch_x = trainseq[bt][0]  # excerpt values
print("batch_x shape:", batch_x.shape, ", batch_y shape:", batch_y.shape)
