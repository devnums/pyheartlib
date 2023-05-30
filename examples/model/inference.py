# This file can be used to do inference using the trained example model
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pyheartlib.io import load_data



cdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cdir)
print("Current directory changed to:\n", cdir)

data_dir = "../../data"
test_data = os.path.join(data_dir, "test.rpeak")
mdl_checkpoint = os.path.join(os.getcwd(), "checkpoint/keras.exp")

# load trained model
loaded_model = tf.keras.models.load_model(mdl_checkpoint)

# load test data
annotated_records_test, samples_info_test = load_data(test_data)
print(len(samples_info_test))
from pyheartlib.data_rpeak import ECGSequence

batch_size = 128
test_generator = ECGSequence(
    annotated_records_test,
    samples_info_test,
    binary=True,
    batch_size=batch_size,
    raw=True,
    interval=36,
    shuffle=False,
)
labels_true = []
labels_pred = []
for i in tqdm(range(round(len(samples_info_test) / batch_size))):
    samples = test_generator.__getitem__(i)
    labels_true.extend(samples[1])
    probs = loaded_model.predict(samples[0], verbose=0)  # predict
    labels_pred.extend(list(np.argmax(probs, axis=-1)))
labels_true = np.array(labels_true)
labels_pred = np.array(labels_pred)
assert labels_true.shape == labels_pred.shape

# print classification_report and confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

true = labels_true.flatten()
pre = labels_pred.flatten()
print(classification_report(true, pre))
print(confusion_matrix(true, pre))
with open("./result.txt", "w") as f:
    print(classification_report(true, pre), file=f)
    print("Confusion Matrix", file=f)
    print(confusion_matrix(true, pre), file=f)


# plot misclassified
import matplotlib.pyplot as plt

batch_size = 1
labels_true = []
labels_pred = []
for i in tqdm(
    range(4680, 4700)
):  # range(round(len(samples_info_test)/batch_size)):
    test_generator = ECGSequence(
        annotated_records_test,
        [samples_info_test[i]],
        binary=True,
        batch_size=batch_size,
        raw=True,
        interval=36,
        shuffle=False,
    )
    samples = test_generator.__getitem__(0)
    labels_true = samples[1]
    probs = loaded_model.predict(samples[0], verbose=0)
    labels_pred = np.argmax(probs, axis=-1)
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    if not np.array_equal(labels_true, labels_pred):
        rec = samples_info_test[i][0]
        st = samples_info_test[i][1]
        en = samples_info_test[i][2]
        sig = annotated_records_test[rec]["signal"][st:en]
        plt.figure(figsize=(10, 5))
        plt.plot(sig, linewidth=1)
        for p in range(labels_true.shape[1]):
            if labels_true[0][p] == 1:
                plt.scatter(p * 36, 1.2, s=40, c="limegreen")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] == 1:
                plt.scatter(p * 36, 1.1, s=40, c="dodgerblue")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] != labels_true[0][p] and labels_true[0][p] == 0:
                plt.scatter(p * 36, 1.1, s=40, c="orange")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] != labels_true[0][p] and labels_true[0][p] == 1:
                plt.scatter(p * 36, 1.2, s=40, c="red")
        # add legend
        plt.scatter(0, 1.3, s=40, c="limegreen", label="True")
        plt.scatter(0, 1.3, s=40, c="dodgerblue", label="Pred")
        plt.scatter(0, 1.3, s=40, c="orange", label="False positive")
        plt.scatter(0, 1.3, s=40, c="red", label="False negative")
        plt.scatter(0, 1.3, s=60, c="white")
        plt.legend(loc="lower left")
        # save fig
        plt.title("Example: Heartbeat Detection")
        plt.rcParams.update({"font.size": 12})
        fig_path = "./plots/mis_" + str(i) + ".png"
        plt.savefig(fig_path, dpi=300)
