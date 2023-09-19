#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


# from scipy.spatial import distance
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import pandas as pd


def reset_seed(seed_value=22):
    import os
    import random as python_random

    import tensorflow as tf
    from numpy import random

    # must run every time
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    python_random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)


# def integer_mapping(y, map, inverse=False):
#     y = list(y)
#     if not inverse:
#         out = [map[i] for i in y]
#     else:
#         invmap = {v: k for k, v in map.items()}
#         out = [invmap[i] for i in y]
#     return np.array(out)


# def mapping_AAMI(y, map):
#     out = [map[i] for i in y]
#     return out


# def plot_loss(model_history, p=False):
#     # Plot training & validation loss
#     import matplotlib.pyplot as plt

#     if p == True:
#         plt.style.use("plotstyle.txt")
#     plt.plot(model_history.history["loss"])
#     plt.plot(model_history.history["val_loss"], "--")
#     plt.title("Training Loss")
#     plt.ylabel("Loss")
#     plt.xlabel("Epoch")
#     plt.legend(["Train", "Validation"], loc="upper right")
#     plt.show()


# def plot_spectogram(sig, sampling_rate=360, win=127, overlap=122):
#     import matplotlib.pyplot as plt
#     from scipy import signal

#     # f,t,Sxx= signal.spectrogram(sig, fs=sampling_rate, nperseg=win,
#                                   nfft=win, noverlap=overlap, mode='psd')
#     # Sxx_log = 10*np.log10(Sxx)
#     # plt.pcolormesh(t, f, Sxx_log)
#     # plt.ylabel('Frequency [Hz]')
#     # plt.xlabel('Time [sec]')
#     # plt.show()

#     import matplotlib.pyplot as plt

#     Pxx, freqs, bins, im = plt.specgram(
#         sig, Fs=sampling_rate, NFFT=win, noverlap=overlap, mode="psd"
#     )
#     print(Pxx.shape)
#     expected = (len(sig) - overlap) / (win - overlap)
#     print("Expexted:" + str(expected))


# def plot_signal(data):
#     import plotly.express as px

#     # data: 1d signal as a pandas dataframe, numpy array, or a list
#     if isinstance(data, pd.DataFrame):
#         pass
#     elif isinstance(data, (np.ndarray, list)):
#         data = pd.DataFrame({"Time": range(len(data)), "Signal": data})
#     fig = px.line(data, title="ECG", x="Time", y="Signal")

#     """
#     import plotly.graph_objects as go
#     fig = go.Figure()
# 	fig.add_trace(
# 		go.Scatter(
# 			x=dfsignal['Index'],
# 			y=dfsignal['Signal'],
# 			#text=R_types,
# 			mode="lines",
# 			line=go.scatter.Line(color="black"),
# 			showlegend=False)
# 	)
# 	fig.add_trace(
# 		go.Scatter(
# 			x=R_locations,
# 			y=signal[R_locations],
# 			text=R_types,
# 			mode="markers",
# 			line=go.scatter.Line(color="red"),
# 			showlegend=False)
# 	)
# 	"""
#     fig.update_xaxes(rangeslider_visible=True)
#     fig.show()


# def calc_class_weights(flag, y):
#     if flag:
#         from sklearn.utils import class_weight

#         class_weights = class_weight.compute_class_weight("balanced",
#                                                            np.unique(y), y)
#         class_weights = dict(zip(np.unique(y), class_weights))
#         # dict(zip(FINAL_MAP_DICT, class_weights))
#         return class_weights
#     else:
#         return None

# import tensorflow as tf
# cosine_similarity = tf.keras.metrics.CosineSimilarity(axis=-1)


# def normalized_cross_corr(a, b):
#     if len(a.shape) == 1:
#         a = (a - np.mean(a)) / (np.std(a) * len(a))
#         b = (b - np.mean(b)) / (np.std(b))
#         cor = np.correlate(a, b)
#     elif len(a.shape) == 2:
#         cor = []
#         for i in range(a.shape[0]):
#             t = a[i]
#             q = b[i]
#             t = (t - np.mean(t)) / (np.std(t) * len(t))
#             q = (q - np.mean(q)) / (np.std(q))
#             c = np.correlate(t, q)
#             cor.append(c)
#         cor = np.array(cor)
#     return np.mean(cor)


# def sig_similarity_report(x_true, x_pred, y_true):
#     from fastdtw import fastdtw
#     from pyheartlib.data_beat import BeatData

#     beatdata = BeatData()
#     mse = []
#     syms = []
#     corr = []
#     coss = []
#     dtww = []
#     for s in np.unique(y_true):
#         ind = beatdata.search_label(y_true, sym=s)
#         true = x_true[ind]
#         pred = x_pred[ind]
#         err = mean_squared_error(true, pred)
#         cor = normalized_cross_corr(true, pred)
#         cos = cosine_similarity(true, pred).numpy()  # mayb wrong
#         dtw = fastdtw(true, pred)[0] / len(true)
#         mse.append(err)
#         corr.append(cor)
#         coss.append(cos)
#         dtww.append(dtw)
#         syms.append(s)

#     dict_res_x = {"sym": syms, "mse_x": mse, "cor": corr,
#                   "cos": coss, "dtw": dtww}
#     df_x = pd.DataFrame(dict_res_x)
#     print(df_x.sort_values(by="cor"))


# def sig_similarity_hist(x1, x2, y):
#     from pyheartlib.data_beat import BeatData

#     beatdata = BeatData()
#     MSE = []
#     COR = []
#     COS = []
#     DTW = []
#     SYMS = []
#     for s in np.unique(y):
#         ind = beatdata.search_label(y, sym=s)
#         x1_selected = x1[ind]
#         x2_selected = x2[ind]
#         for i in range(len(x1_selected)):
#             s1 = x1_selected[i].reshape((1, -1))
#             s2 = x2_selected[i].reshape((1, -1))
#             mse = mean_squared_error(s1, s2)
#             cor = normalized_cross_corr(s1, s2)
#             cos = 1 - (distance.cosine(s1, s2))
#             dtw = fastdtw(s1, s2)[0]
#             MSE.append(mse)
#             COR.append(cor)
#             COS.append(cos)
#             DTW.append(dtw)
#             SYMS.append(s)

#     dict_res = {"SYM": SYMS, "MSE": MSE, "COR": COR,
#                 "COS": COS, "DTW": DTW}
#     df_res = pd.DataFrame(dict_res)
#     return df_res


# def get_var_size(var_type="all"):
#     print("hhh")
#     if var_type == "all":
#         varss = set(dir())
#     elif var_type == "local":
#         varss = set(locals())
#     elif var_type == "global":
#         varss = set(globals())

#     import sys

#     ob = {}
#     for var in varss:
#         ob[var] = sys.getsizeof(var)
#     return ob
