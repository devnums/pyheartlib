import numpy as np
import pandas as pd
from pyheartlib.io import load_data
from pyheartlib.data_arrhythmia import ArrhythmiaData, ECGSequence


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
df = pd.DataFrame(np.unique(labels, return_counts=True), index=["Label", "Count"])
print(df)

class_labels = list(np.unique(labels))

# if raw=True
training_generator = ECGSequence(
    annotated_records, samples_info, class_labels=None, batch_size=3, raw=True
)
bt = 0
batch_label = training_generator.__getitem__(bt)[1]  # excerpt label
batch_seq = training_generator.__getitem__(bt)[0][0]  # excerpt values
batch_rri = training_generator.__getitem__(bt)[0][1]  # rr intervals
batch_rri_feat = training_generator.__getitem__(bt)[0][2]  # calculated rri features
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
training_generator = ECGSequence(
    annotated_records, samples_info, class_labels=None, batch_size=3, raw=False
)
i = 0
batch_label = training_generator.__getitem__(bt)[1]  # excerpt label
batch_seq = training_generator.__getitem__(bt)[0][0]  # excerpt values
batch_rri = training_generator.__getitem__(bt)[0][1]  # rr intervals
batch_rri_feat = training_generator.__getitem__(bt)[0][2]  # calculated rri features
print("batch_label shape:", batch_label.shape)
print(
    "batch_seq shape:",
    batch_seq.shape,
    " , batch_rri shape:",
    batch_rri.shape,
    " , batch_rri_feat shape:",
    batch_rri_feat.shape,
)
