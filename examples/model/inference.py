#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


# This file can be used to do inference using the trained example model
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from pyheartlib.io import load_data

cdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cdir)
print("Current directory changed to:\n", cdir)

data_dir = "../../data"
test_data = os.path.join(data_dir, "test.rpeak")
mdl_checkpoint = os.path.join(os.getcwd(), "checkpoint/keras.exp")
interval_value = 6

# load trained model
loaded_model = tf.keras.models.load_model(mdl_checkpoint)

# load test data
annotated_records_test, samples_info_test = load_data(test_data)
print(len(samples_info_test))
from pyheartlib.data_rpeak import ECGSequence  # noqa: E402

batch_size = 32
testseq = ECGSequence(
    annotated_records_test,
    samples_info_test,
    binary=True,
    batch_size=batch_size,
    raw=True,
    interval=interval_value,
    shuffle=False,
)
labels_true = []
labels_pred = []
for i in tqdm(range(round(len(samples_info_test) / batch_size))):
    samples = testseq[i]
    labels_true.extend(samples[1])
    probs = loaded_model.predict(samples[0], verbose=0)  # predict
    labels_pred.extend(list(np.argmax(probs, axis=-1)))
labels_true = np.array(labels_true)
labels_pred = np.array(labels_pred)
assert labels_true.shape == labels_pred.shape

# print classification_report and confusion_matrix
true = labels_true.flatten()
pre = labels_pred.flatten()
print(classification_report(true, pre))
print(confusion_matrix(true, pre))
with open("./result.txt", "w") as f:
    print(classification_report(true, pre), file=f)
    print("Confusion Matrix", file=f)
    print(confusion_matrix(true, pre), file=f)
