"""
This example shows how to create the datasets intra-patient.
"""

from pyecg.utils import *
from pyecg.data_beat import BeatData
from pyecg.data_beat import DS1, DS2

# Intra-patient
beatdata = BeatData(base_path="./data", win=[120, 180], remove_bl=False, lowpass=False)
beatdata.save_dataset_intra(records=DS1, split_ratio=0.7, save_file_prefix="intra")

# Loading the sets
train_ds = beatdata.load_data(file_name="intra_train.beat")
test_ds = beatdata.load_data(file_name="intra_test.beat")

# Number of samples per class
stat_report = beatdata.report_stats_table(
    [train_ds["labels"], test_ds["labels"]], ["Train", "Test"]
)
