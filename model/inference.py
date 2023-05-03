import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pyecg.io import load_data

# load trained model
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
for i in tqdm(range(round(len(samples_info_test) / batch_size))):
    samples = test_generator.__getitem__(i)
    labels_true.extend(samples[1])
    probs = loaded_model.predict(samples[0], verbose=0)  # predict
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
for i in tqdm(range(0, 20)):  # range(round(len(samples_info_test)/batch_size)):
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
        plt.plot(sig)
        for p in range(labels_true.shape[1]):
            if labels_true[0][p] == 1:
                plt.scatter(p * 36, 1, s=100, c="g")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] == 1:
                plt.scatter(p * 36, 0.92, s=100, c="r")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] != labels_true[0][p] and labels_true[0][p] == 0:
                plt.scatter(p * 36, 0.92, s=100, c="orange")
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] != labels_true[0][p] and labels_true[0][p] == 1:
                plt.scatter(p * 36, 1, s=100, c="black")        
        # add legend
        plt.scatter(0, 1.3, s=100, c="g", label="True")
        plt.scatter(0, 1.3, s=100, c="r", label="Pred")
        plt.scatter(0, 1.3, s=100, c="orange", label="False positive")
        plt.scatter(0, 1.3, s=100, c="black", label="False negative")
        plt.scatter(0, 1.3, s=110, c="white")
        plt.legend(loc="lower left")
        # save fig
        plt.title("Heartbeat detection")
        fig_path = "model/plots/mis_" + str(i) + ".png"
        plt.savefig(fig_path, dpi=300)
