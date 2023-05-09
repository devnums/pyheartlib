import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyecg.io import load_data
from pyecg.data_rpeak import RpeakData, ECGSequence


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
df = pd.DataFrame(np.unique(labels, return_counts=True), index=["Label", "Count"])
print(df)

# if raw=True
training_generator = ECGSequence(
    annotated_records, samples_info, binary=False, batch_size=3, raw=True, interval=72
)
i = 0
batch_y = training_generator.__getitem__(i)[1]  # excerpt label
batch_x = training_generator.__getitem__(i)[0]  # excerpt values
print("batch_x shape:", batch_x.shape, ", batch_y shape:", batch_y.shape)


# if raw=False
training_generator = ECGSequence(
    annotated_records, samples_info, binary=False, batch_size=3, raw=False, interval=72
)
i = 0
batch_y = training_generator.__getitem__(i)[1]  # excerpt label
batch_x = training_generator.__getitem__(i)[0]  # excerpt values
print("batch_x shape:", batch_x.shape, ", batch_y shape:", batch_y.shape)
