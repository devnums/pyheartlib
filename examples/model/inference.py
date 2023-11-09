# Inference using the trained example model
import os
import textwrap

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from pyheartlib.data_rpeak import load_dataset

cdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cdir)
print("Current directory changed to:\n", cdir)

data_dir = "data"
test_data = os.path.join(data_dir, "test.rpeak")
mdl_checkpoint = os.path.join(os.getcwd(), "checkpoint/keras.exp")
interval_value = 6

# load trained model
loaded_model = tf.keras.models.load_model(mdl_checkpoint)

# load test data
annotated_records_test, samples_info_test = load_dataset(test_data)
print(len(samples_info_test))
from pyheartlib.data_rpeak import ECGSequence  # noqa: E402

batch_size = 128
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
pred = labels_pred.flatten()
print(classification_report(true, pred))
print(confusion_matrix(true, pred))
with open("./result.txt", "w") as f:
    title = (
        "************************************************************"
        "Without considering adjacency of ground truth and prediction"
    )
    print(textwrap.fill(title, width=60, replace_whitespace=False), file=f)
    print("\n", file=f)
    print(classification_report(true, pred), file=f)
    print("Confusion Matrix", file=f)
    print(confusion_matrix(true, pred), file=f)
    print("\n", file=f)


def post_tolerance(true, pred):
    # Give a tolerance for nearby segments (False positive)
    # when a False Positve is very adjacent to ground truth peak
    new_pred = np.copy(pred)
    for indx, _ in enumerate(pred):
        if pred[indx] == 1:
            if true[indx] != 1:
                if true[indx + 1] == 1:
                    new_pred[indx + 1] = 1
                    new_pred[indx] = 0
                elif true[indx - 1] == 1:
                    new_pred[indx - 1] = 1
                    new_pred[indx] = 0
                elif true[indx + 2] == 1:
                    new_pred[indx + 2] = 1
                    new_pred[indx] = 0
                elif true[indx - 2] == 1:
                    new_pred[indx - 2] = 1
                    new_pred[indx] = 0
                if true[indx + 3] == 1:
                    new_pred[indx + 3] = 1
                    new_pred[indx] = 0
                elif true[indx - 3] == 1:
                    new_pred[indx - 3] = 1
                    new_pred[indx] = 0
                if true[indx + 4] == 1:
                    new_pred[indx + 4] = 1
                    new_pred[indx] = 0
                elif true[indx - 4] == 1:
                    new_pred[indx - 4] = 1
                    new_pred[indx] = 0
                if true[indx + 5] == 1:
                    new_pred[indx + 5] = 1
                    new_pred[indx] = 0
                elif true[indx - 5] == 1:
                    new_pred[indx - 5] = 1
                    new_pred[indx] = 0
    return new_pred


# print classification_report and confusion_matrix
new_pred = post_tolerance(true, pred)
pred = new_pred
print(classification_report(true, pred))
print(confusion_matrix(true, pred))
with open("./result.txt", "a") as f:
    title = (
        "************************************************************"
        "The results below were obtained by considering a very small"
        "  tolerance for the adjacency of ground truth and prediction"
    )
    print(textwrap.fill(title, width=60, replace_whitespace=False), file=f)
    print("\n", file=f)
    print(classification_report(true, pred), file=f)
    print("Confusion Matrix", file=f)
    print(confusion_matrix(true, pred), file=f)
