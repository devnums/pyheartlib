import numpy as np
import tensorflow as tf
from pyecg.io import load_data

# load the trained model
loaded_model = tf.keras.models.load_model("model/checkpoint/keras.exp")

# load test data
annotated_records_test, samples_info_test = load_data("./data/test.rpeak")

from pyecg.data_rpeak import ECGSequence

batch_size = 128
test_generator = ECGSequence(
    annotated_records_test,
    samples_info_test,
    binary=True,
    batch_size=batch_size,
    raw=True,
    interval=36,
)
labels_true = []
labels_pred = []
for i in range(round(len(samples_info_test) / batch_size)):
    samples = test_generator.__getitem__(i)
    labels_true.extend(samples[1])
    probs = loaded_model.predict(samples[0])
    labels_pred.extend(list(np.argmax(probs, axis=-1)))
labels_true = np.array(labels_true)
labels_pred = np.array(labels_pred)
assert labels_true.shape == labels_pred.shape

from sklearn.metrics import confusion_matrix, classification_report

true = labels_true.flatten()
pre = labels_pred.flatten()
print(classification_report(true, pre))
print(confusion_matrix(true, pre))

# plot if misclassified
import matplotlib.pyplot as plt

batch_size = 1
labels_true = []
labels_pred = []
for i in range(0, 20):  # range(round(len(samples_info_test)/batch_size)):
    test_generator = ECGSequence(
        annotated_records_test,
        [samples_info_test[i]],
        binary=True,
        batch_size=batch_size,
        raw=True,
        interval=36,
    )
    samples = test_generator.__getitem__(0)
    labels_true = samples[1]
    probs = loaded_model.predict(samples[0])
    labels_pred = np.argmax(probs, axis=-1)
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    if not np.array_equal(labels_true, labels_pred):
        print(i)
        rec = samples_info_test[i][0]
        st = samples_info_test[i][1]
        en = samples_info_test[i][2]
        sig = annotated_records_test[rec]["signal"][st:en]
        plt.figure(figsize=(20, 10))
        plt.plot(sig)
        for p in range(labels_true.shape[1]):
            if labels_true[0][p] == 1:
                plt.scatter(p * 36, labels_true[0][p], s=100, c="g")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] == 1:
                plt.scatter(p * 36, labels_pred[0][p] - 0.03, s=100, c="r")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] != labels_true[0][p]:
                plt.scatter(p * 36, labels_pred[0][p] - 0.06, s=100, c="orange")
        # add legend
        plt.scatter(0, 1.3, s=100, c="g", label="True")
        plt.scatter(0, 1.3, s=100, c="r", label="Pred")
        plt.scatter(0, 1.3, s=100, c="orange", label="False pred")
        plt.scatter(0, 1.3, s=110, c="white")
        plt.legend(loc="lower left")
        # save fig
        plt.title("Heartbeat detection")
        fig_path = "model/plots/mis_" + str(i) + ".png"
        plt.savefig(fig_path, dpi=200)
