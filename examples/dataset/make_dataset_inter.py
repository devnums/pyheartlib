"""
This example shows how to create the datasets inter-patient.
"""

from pyheartlib.data_beat import BeatData
from pyheartlib.beat_info import BeatInfo


# Inter-patient
# Lets create a BeatData object.the train set by creating
beatdata = BeatData(
    base_path="./data",
    win=[200, 200],
    remove_bl=False,
    lowpass=False,
    progress_bar=True,
)

# Create a BeatInfo object for features calculations.
beatinfo = BeatInfo(beat_loc=beatdata.beat_loc)


def F_new_feature1(self):
    return 1.09


def F_new_feature2(self):
    post = beatinfo.F_post_rri()
    pre = beatinfo.F_pre_rri()
    return post - pre


new_features = [F_new_feature1, F_new_feature2]
beatinfo.add_features(new_features)
print(beatinfo.available_features())


beatinfo.select_features(
    ["F_beat_max", "F_beat_min", "F_beat_skewness", "F_new_feature1", "F_new_feature2"]
)

# use the save_dataset_inter method to create the dataset file.
# The file will be saved in the base data directory.
beatdata.save_dataset_inter(["209"], beatinfo, file="train.beat")

# In a similar way for validation and test sets
beatdata.save_dataset_inter(["215", "220"], beatinfo, file="val.beat")
beatdata.save_dataset_inter(["103"], beatinfo, file="test.beat")

# Loading the sets
train_ds = beatdata.load_data(file_name="train.beat")
val_ds = beatdata.load_data(file_name="val.beat")
test_ds = beatdata.load_data(file_name="test.beat")

# Number of samples per class
stat_report = beatdata.report_stats_table(
    [train_ds["labels"], val_ds["labels"], test_ds["labels"]], ["Train", "Val", "Test"]
)
print(stat_report)
print(train_ds["beat_feats"].tail(3))
