# Prepare dataset
import os

from pyheartlib.data_rpeak import ECGSequence, RpeakData, load_dataset

cdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cdir)
print("Current directory changed to:\n", cdir)
data_dir = "data"
# create train, validation, and test sets
rpeak_data = RpeakData(base_path=data_dir, remove_bl=False, lowpass=False)

train_set = rpeak_data.config["DS1"][:18]
val_set = rpeak_data.config["DS1"][18:]
test_set = rpeak_data.config["DS2"]

rpeak_data.save_dataset(
    rec_list=train_set, file_name="train.rpeak", win_size=10 * 360, stride=1440
)
rpeak_data.save_dataset(
    rec_list=val_set, file_name="val.rpeak", win_size=10 * 360, stride=1440
)
rpeak_data.save_dataset(
    rec_list=test_set, file_name="test.rpeak", win_size=10 * 360, stride=1440
)

# for verfication
train_data = os.path.join(data_dir, "train.rpeak")
annotated_records, samples_info = load_dataset(train_data)
print("number of generated sampels:", str(len(samples_info)))

trainseq = ECGSequence(
    annotated_records,
    samples_info,
    binary=True,
    batch_size=1,
    raw=True,
    interval=6,
)

i = 0
label = trainseq[i][1]  # excerpt label
seq = trainseq[i][0]  # excerpt values

print("len label:", len(label[0]), "len seq:", len(seq[0]))
