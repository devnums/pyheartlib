#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


# from sklearn.metrics import (
#     confusion_matrix,
#     ConfusionMatrixDisplay,
#     classification_report,
# )
# import matplotlib.pylab as pylab
# import pandas as pd
# import numpy as np


# class Reports:
#     """
#     Generate reports for the model
#     """

#     def __init__(self, y_true, y_pred, labels=None):
#         self.y_true = y_true
#         self.y_pred = y_pred
#         self.labels = labels

#     def conf_matrix(self, normalize=None):
#         return confusion_matrix(self.y_true,
#                                 self.y_pred, normalize=normalize)

#     def plot_confusion_matrix(self, normalize=None,
#                               cmap="Blues", values_format=".2%"):
#         # print(pylab.rcParams)
#         params = {
#             "legend.fontsize": 4,
#             "figure.figsize": (4, 4),
#             "axes.labelsize": 4,
#             "axes.titlesize": 4,
#             "xtick.labelsize": 4,
#             "ytick.labelsize": 4,
#             "font.size": 4.0,
#             "figure.dpi": 200,
#             "legend.frameon": False,
#         }
#         pylab.rcParams.update(params)
#         if not self.labels:
#             print("Labels are not provided!")
#         cm = self.conf_matrix(normalize=normalize)
#         disp = ConfusionMatrixDisplay(
#             confusion_matrix=cm, display_labels=sorted(self.labels)
#         )
#         disp.plot(include_values=True, cmap=cmap,
#                   values_format=values_format)

#     def the_classification_report(self, digits=4):
#         report = classification_report(self.y_true, self.y_pred,
#                                        digits=digits)
#         return report

#     def metrics(self):
#         # arrays with length equal to num classes
#         cfm = self.conf_matrix()
#         FP = cfm.sum(axis=0) - np.diag(cfm)
#         FN = cfm.sum(axis=1) - np.diag(cfm)
#         TP = np.diag(cfm)
#         TN = cfm.sum() - (FP + FN + TP)

#         TPR = TP / (TP + FN) * 100  # Sensitivity, recall
#         TNR = TN / (TN + FP) * 100  # Specificity, true negative rate
#         # Precision, positive predictive value (PPV)
#         PPV = TP / (TP + FP) * 100
#         NPV = TN / (TN + FN) * 100  # Negative predictive value
#         FPR = FP / (FP + TN) * 100  # False positive rate
#         FNR = FN / (TP + FN) * 100  # False negative rate
#         # Accuracy of each class
#         ACC = (TP + TN) / (TP + FP + FN + TN) * 100
#         out = {
#             "Class": sorted(self.labels),
#             "(PPV)Precision": PPV,
#             "(Sensitivity)Recall": TPR,
#             "Specificity": TNR,
#             "Accuracy": ACC,
#         }
#         return out

#     def metrics_table(self):
#         mt = self.metrics()
#         df = pd.DataFrame(mt).round(2)
#         return df


# """
#  from sklearn.metrics import confusion_matrix
# np.set_printoptions(suppress=True,
#                     formatter={'float_kind':'{:.2f}'.format})
# cfm=confusion_matrix(y_true_AAMI, y_pred_AAMI, labels=sorted(labels_AAMI),
#                      normalize='true')*100
# cfm
# import pandas as pd
# pd.DataFrame(cfm).round(2)
# """
