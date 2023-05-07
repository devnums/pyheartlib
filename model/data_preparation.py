import numpy as np
import matplotlib.pyplot as plt
from pyecg.io import load_data
from pyecg.data_rpeak import RpeakData, ECGSequence
from pyecg.dataset_config import DS1, DS2


train_set = DS1[:18]
val_set = DS1[18:]
test_set = DS2


# create train, validation, and test sets
rpeak_data = RpeakData(base_path="./data", remove_bl=False, lowpass=False)
annotated_records, samples_info = rpeak_data.save_samples(
    rec_list=train_set, file_name="train.rpeak", win_size=10 * 360, stride=1440
)
annotated_records, samples_info = rpeak_data.save_samples(
    rec_list=val_set, file_name="val.rpeak", win_size=10 * 360, stride=1440
)
annotated_records, samples_info = rpeak_data.save_samples(
    rec_list=test_set, file_name="test.rpeak", win_size=10 * 360, stride=1440
)

# for verfication
annotated_records, samples_info = load_data("./data/train.rpeak")
print("number of generated sampels:", str(len(samples_info)))

training_generator = ECGSequence(
    annotated_records, samples_info, binary=True, batch_size=1, raw=True, interval=36
)

i = 0
label = training_generator.__getitem__(i)[1]  # excerpt label
seq = training_generator.__getitem__(i)[0]  # excerpt values

print("len label:", len(label[0]), "len seq:", len(seq[0]))
