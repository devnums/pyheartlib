import numpy as np
import matplotlib.pyplot as plt
from pyecg.io import load_data
from pyecg.data_rpeak import RpeakData, ECGSequence


train_set = [201, 203]
test_set = [202, 210, 219, 221, 222]

rpeak_data = RpeakData(base_path="./data", remove_bl=False, lowpass=False)

annotated_records, samples_info = rpeak_data.save_samples(
    rec_list=train_set, file_path="./data/train.rpeak", win_size=30 * 360, stride=360
)
annotated_records, samples_info = load_data("./data/train.rpeak")

labels = []
for sample in samples_info:
    labels.append(sample[3])
print(np.unique(labels, return_counts=True))
class_labels = list(np.unique(labels))
print(class_labels)

training_generator = ECGSequence(
    annotated_records, samples_info, binary=False, batch_size=1, raw=True, interval=72
)

i = 0
label = training_generator.__getitem__(i)[1]  # excerpt label
seq = training_generator.__getitem__(i)[0]  # excerpt values

print("len label:", len(label[0]), "len seq:", len(seq[0]))
