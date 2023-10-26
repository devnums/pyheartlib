# Inference using the trained example model -- plots
import os
import pathlib

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pyheartlib.io import load_data

cdir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cdir)
print("Current directory changed to:\n", cdir)

data_dir = "data"
test_data = os.path.join(data_dir, "test.rpeak")
mdl_checkpoint = os.path.join(os.getcwd(), "checkpoint/keras.exp")
interval_value = 6
opt = {1: "mis", 2: "correct", 3: "all"}
to_plot = opt[3]
rang = range(3790, 3800)

# load trained model
loaded_model = tf.keras.models.load_model(mdl_checkpoint)

# load test data
annotated_records_test, samples_info_test = load_data(test_data)
print(len(samples_info_test))
# plot
import matplotlib.pyplot as plt  # noqa: E402

from pyheartlib.data_rpeak import ECGSequence  # noqa: E402

batch_size = 1
labels_true = []
labels_pred = []
for i in tqdm(rang):  # range(round(len(samples_info_test)/batch_size)):
    testseq = ECGSequence(
        annotated_records_test,
        [samples_info_test[i]],
        binary=True,
        batch_size=batch_size,
        raw=True,
        interval=interval_value,
        shuffle=False,
    )
    samples = testseq[0]
    labels_true = samples[1]
    probs = loaded_model.predict(samples[0], verbose=0)
    labels_pred = np.argmax(probs, axis=-1)
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    cond = not np.array_equal(labels_true, labels_pred)
    if to_plot == "mis":
        cond = not np.array_equal(labels_true, labels_pred)
    elif to_plot == "correct":
        cond = np.array_equal(labels_true, labels_pred)
    elif to_plot == "all":
        cond = True
    if cond:
        rec = samples_info_test[i][0]
        st = samples_info_test[i][1]
        en = samples_info_test[i][2]
        sig = annotated_records_test[rec]["signal"][st:en]
        # figure
        plt.figure(figsize=(10, 5))
        # plot signal
        plt.plot(sig, linewidth=0.8, color="#118ed6", zorder=2)
        # set axis range
        plt.xlim(-10, 3610)
        plt.ylim(-1.7, 2.0)
        plt.xlabel("Samples")
        plt.ylabel("Voltage(mV)")
        # add grid lines
        plt.grid(
            which="major",
            c="#bdbebf",
            linestyle="dashed",
            linewidth=0.5,
            zorder=1,
        )
        plt.grid(
            which="minor",
            c="#bdbebf",
            linestyle="dashed",
            linewidth=0.3,
            zorder=1,
        )
        plt.minorticks_on()
        # marker size
        mrkr_size = 30
        for p in range(labels_true.shape[1]):
            if labels_true[0][p] == 1:
                plt.scatter(
                    (p + 0.5) * interval_value,
                    1.2,
                    s=mrkr_size,
                    c="red",
                    zorder=3,
                )
        for p in range(labels_pred.shape[1]):
            if labels_pred[0][p] == 1:
                plt.scatter(
                    (p + 0.5) * interval_value,
                    1.1,
                    s=mrkr_size,
                    c="#044dcc",
                    zorder=3,
                )
        fp = False
        for p in range(labels_pred.shape[1]):
            if (
                labels_pred[0][p] != labels_true[0][p]
                and labels_true[0][p] == 0
            ):
                fp = True
                plt.scatter(
                    (p + 0.5) * interval_value,
                    1.1,
                    s=mrkr_size,
                    c="orange",
                    zorder=3,
                )
        """
        for p in range(labels_pred.shape[1]):
            if (
                labels_pred[0][p] != labels_true[0][p]
                and labels_true[0][p] == 1
            ):
                plt.scatter((p + 0.5) * interval_value, 1.2, s=mrkr_size,
                            c="#3ab505", zorder=3)
        """

        # add legend
        lbl = "Manually annotated R-peaks (Ground truth)"
        plt.scatter(-50, 1.3, s=mrkr_size, c="red", zorder=3, label=lbl)
        if fp:
            lbl = "Incorrectly detected R-peaks (False positive)"
            plt.scatter(-50, 1.3, s=mrkr_size, c="orange", zorder=3, label=lbl)
        lbl = "Correctly detected R-peaks (True positive)"
        plt.scatter(-50, 1.3, s=mrkr_size, c="#044dcc", zorder=3, label=lbl)
        # plt.scatter(0, 1.3, s=mrkr_size, c="#3ab505", label="Ground truth
        # R-peaks missed by the model (False negative)")
        lbl = (
            "Markers represent only the horizontal position of the R-peaks."
            " The deep learning model was"
        )
        plt.scatter(-50, 1.3, s=0, c="white", zorder=3, label=lbl)
        lbl = (
            "designed to predict whether a segment of the signal, which"
            " spans six samples, contains an R-peak."
        )
        plt.scatter(-50, 1.3, s=0, c="white", zorder=3, label=lbl)
        plt.legend(loc="lower left", fontsize=10, labelspacing=0.14)
        # add title
        titl = (
            "Example use case of the pyheartlib package for"
            " the R-peak detection task"
        )
        plt.title(titl, fontsize=12)
        plt.rcParams.update({"font.size": 12})
        plt.rcParams["axes.axisbelow"] = True
        # save fig
        pathlib.Path("plots").mkdir(parents=True, exist_ok=True)
        fig_path = "./plots/" + str(i) + ".png"
        plt.savefig(fig_path, dpi=600)
