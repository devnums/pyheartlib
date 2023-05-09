import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecg.io import load_data
from pyecg.data_arrhythmia import ArrhythmiaData, ECGSequence


# AFIBs
train_set = [201, 203]
#test_set = [202, 210, 219, 221, 222]

arrhythmia_data = ArrhythmiaData(base_path="./data", remove_bl=False, lowpass=False)

annotated_records, samples_info = arrhythmia_data.save_samples(
    rec_list=train_set, file_name="train.arr", win_size=3600, stride=64
)
annotated_records, samples_info = load_data("./data/train.arr")

labels = []
for sample in samples_info:
    labels.append(sample[3])
df = pd.DataFrame(np.unique(labels, return_counts=True), index=['Label','Count'])
print(df)

class_labels = list(np.unique(labels))
training_generator = ECGSequence(
    annotated_records, samples_info, class_labels=None, batch_size=3
)

i = 0
batch_label = training_generator.__getitem__(i)[1]  # excerpt label
batch_seq = training_generator.__getitem__(i)[0][0]  # excerpt values
batch_rri = training_generator.__getitem__(i)[0][1]  # rr intervals
batch_rri_feat = training_generator.__getitem__(i)[0][2]  # calculated rri features
print("batch_label shape:", batch_label.shape)
print("batch_seq shape:", batch_seq.shape, " , batch_rri shape:", batch_rri.shape, \
    " , batch_rri_feat shape:", batch_rri_feat.shape)