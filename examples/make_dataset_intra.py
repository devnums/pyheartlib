"""
This example shows how to create the datasets intra-patient.
"""

from pyecg.dataset_config import DS1, DS2
from pyecg.data_beat import BeatData
from pyecg.beat_info import BeatInfo


# Intra-patient
# Lets create a BeatData object.the train set by creating
beatdata = BeatData(base_path="./data", win=[200, 200], remove_bl=False, lowpass=False, progress_bar=True)

# Create a BeatInfo object for features calculations.
beatinfo = BeatInfo(beat_loc=beatdata.beat_loc)


def F_new_feature1(self):
    return 1.07

def F_new_feature2(self):
    post = beatinfo.F_post_rri()
    pre = beatinfo.F_pre_rri()
    return post - pre

new_features = [F_new_feature1, F_new_feature2]
beatinfo.add_features(new_features)  
print(beatinfo.available_features()) 


beatinfo.select_features(["F_beat_max", "F_beat_min", "F_beat_skewness", "F_new_feature1", "F_new_feature2"])

# The file will be saved in the base data directory.
beatdata.save_dataset_intra(DS1[17:21], beatinfo)

# Loading the sets
train_ds = beatdata.load_data(file_name="intra_train.beat")
test_ds = beatdata.load_data(file_name="intra_test.beat")

# Number of samples per class
stat_report = beatdata.report_stats_table(
    [train_ds["labels"], test_ds["labels"]], ["Train", "Test"]
)
print(stat_report)
print(train_ds['beat_feats'].head(3)) 




