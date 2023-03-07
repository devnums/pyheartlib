import numpy as np
import matplotlib.pyplot as plt
from pyecg.io import load_data
from pyecg.data_arrhythmia import ArrhythmiaData, ECGSequence


# AFIBs
train_set = [201, 203]
test_set = [202, 210, 219, 221, 222]

arrhythmia_data = ArrhythmiaData(base_path="./data", remove_bl=False, lowpass=False)

annotated_records, samples_info = arrhythmia_data.save_samples(
    rec_list=train_set, file_path="./data/train.arr", win_size=3600, stride=64
)
annotated_records, samples_info = load_data("./data/train.arr")

labels = []
for sample in samples_info:
    labels.append(sample[3])
print(np.unique(labels, return_counts=True))
class_labels = list(np.unique(labels))
print(class_labels)

training_generator = ECGSequence(
    annotated_records, samples_info, class_labels=class_labels, batch_size=1
)

i = 0
label = training_generator.__getitem__(i)[1]  # excerpt label
seq = training_generator.__getitem__(i)[0][0]  # excerpt values
rri = training_generator.__getitem__(i)[0][1]  # rr intervals
feats = training_generator.__getitem__(i)[0][2]  # calculated rri features
