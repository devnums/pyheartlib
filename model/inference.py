import numpy as np
import tensorflow as tf
from pyecg.io import load_data


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
    probs = loaded_model.predict_generator(samples[0])
    labels_pred.extend(list(np.argmax(probs, axis=-1)))

labels_true = np.array(labels_true)
labels_pred = np.array(labels_pred)
assert labels_true.shape == labels_pred.shape


from sklearn.metrics import confusion_matrix, classification_report

true = labels_true.flatten()
pre = labels_pred.flatten()
print(classification_report(true, pre))
print(confusion_matrix(true, pre))
