import numpy as np
import random
import copy
from pyecg.dataset_config import *


def search_type(x, y, sym="N"):
    """Search for cutted signal with a patricular type"""
    indexes = [i for i, item in enumerate(y) if item == sym]
    return x[indexes], indexes


def aug_decrease(ds, label="N", desired_size=21000):
    # DATA Augmentation-decrease N
    ds = copy.deepcopy(ds)
    xx = ds["waveforms"]
    rr = ds["beat_feats"]
    yy = ds["labels"]
    _, ind = search_type(xx, yy, sym=label)
    random.shuffle(ind)
    nn = len(ind) - desired_size
    ind_remove = ind[0:nn]
    x_train_aug = np.delete(xx, ind_remove, axis=0)
    r_train_aug = np.delete(rr, ind_remove, axis=0)
    y_train_aug = np.delete(yy, ind_remove)
    print(x_train_aug.shape, y_train_aug.shape)

    return {"waveforms": x_train_aug, "beat_feats": r_train_aug, "labels": y_train_aug}


def aug_increase(ds, desired_size=21000):
    # DATA Augmentation-Increase minority
    import copy

    x_aug = ds["waveforms"]
    r_aug = ds["beat_feats"]
    y_aug = ds["labels"].tolist()
    for sym in list(MAP_AAMI.keys()):
        # print(sym)
        #
        try:
            _, ind_minor = search_type(ds["waveforms"], ds["labels"], sym=sym)
            minority = np.take(ds["waveforms"], ind_minor, axis=0)
            minority_r = np.take(ds["beat_feats"], ind_minor, axis=0)
            minority_labels = [sym] * len(minority)
            if len(minority) > 0 and len(ind_minor) < desired_size:
                times = desired_size // len(minority)
                if times > 0:
                    arr = copy.deepcopy(minority)
                    arr_r = copy.deepcopy(minority_r)
                    list_appnd = copy.deepcopy(minority_labels)
                    for i in range(times - 1):
                        arr = np.append(arr, minority, axis=0)
                        arr_r = np.append(arr_r, minority_r, axis=0)
                        list_appnd = list_appnd + minority_labels
                    x_aug = np.append(x_aug, arr, axis=0)
                    r_aug = np.append(r_aug, arr_r, axis=0)
                    y_aug = y_aug + list_appnd
                else:
                    print(sym)
        except:
            print("label zero")

    return {"waveforms": x_aug, "beat_feats": r_aug, "labels": np.array(y_aug)}


"""
def slice_data(ds, lable_list):
  #only keep the provided lables
  sliced_x = copy.deepcopy(ds['x'])
  sliced_y = copy.deepcopy(ds['y'])
  indexes_keep = []
  for sym in lable_list:
    inds = [i for i,item in enumerate(y) if item==sym]
    indexes_keep = indexes_keep+inds
  print(len(indexes_keep))
  sliced_x = sliced_x[indexes_keep]
  sliced_y = [sliced_y[i] for i in indexes_keep]

  return {'x':sliced_x, 'y':sliced_y, 'r':ds['r']}
"""


def binarize_lables(y, positive_lable, pos=1, neg=-1):
    # y a list of lables
    # positive_lable: positive class lable
    new_y = [pos if item == positive_lable else neg for item in y]
    return new_y
