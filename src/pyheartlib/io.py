#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


import pickle

import numpy as np
import pandas as pd
import wfdb


def get_data(record_path, config, return_dict=True):
    """Reads a record and returns its components.

    Parameters
    ----------
    record_path : str
        Path of the record.
    config : dict
        Config of the dataset.
    return_dict : bool
        If True, returns a dict, otherwise returns a pandas dataframe.

    Returns
    -------
    dict or dataframe
        If return_dict is True, it returns a dictionary with keys:
        *signal*, *r_locations*, *r_labels*, *rhythms*, *rhythms_locations*.

        If return_dict is False, it returns a dataframe containing
        the time, raw signal, and a list of equal size to the raw signal
        with None values except at annotations locations.

    """
    try:
        chnls = ""
        chnls = config["CHANNEL"]
    except KeyError:
        print("Names of channels are not provided!")
    if isinstance(chnls, list):
        channels = chnls
    else:
        channels = []
        channels.append(str(chnls))
    all_channels = []
    for ch in channels:
        try:
            record = wfdb.rdrecord(record_path, channel_names=[ch])
            all_channels.append(record.p_signal.T.tolist()[0])
        except Exception:
            errmsg = (
                f"Channel {ch} was not read correctly! Record "
                f"path: {record_path}"
            )
            raise RuntimeError(errmsg)
    signal = np.array(all_channels)
    signal = signal.T
    if signal.shape[1] != len(channels):
        raise ValueError(f"Channels are not right! Record path: {record_path}")
    annotation = wfdb.rdann(record_path, "atr")
    ann_locations = annotation.sample
    if len(ann_locations) < 1:
        errmsg = (
            f"Record was not read correctly or does not have "
            f"annotations! Record path: {record_path}"
        )
        raise RuntimeError(errmsg)
    symbol = annotation.symbol
    aux = annotation.aux_note
    aux = [txt.rstrip("\x00") for txt in aux]

    r_labels = []
    r_locations = []
    rhythms = []
    rhythms_locations = []

    try:
        syms = config["BEAT_TYPES"]
        for i in range(len(ann_locations)):
            if symbol[i] in syms:
                r_labels.append(symbol[i])
                r_locations.append(ann_locations[i])
    except KeyError:
        r_labels = None
        r_locations = None

    try:
        auxs = config["RHYTHM_TYPES"]
        for i in range(len(ann_locations)):
            if aux[i] in auxs:
                rhythms.append(aux[i])
                rhythms_locations.append(ann_locations[i])
    except KeyError:
        rhythms = None
        rhythms_locations = None

    if return_dict is True:
        return {
            "signal": signal,
            "r_locations": r_locations,
            "r_labels": r_labels,
            "rhythms": rhythms,
            "rhythms_locations": rhythms_locations,
        }
    else:
        annots = [None] * signal.shape[0]
        auxiliary = [None] * signal.shape[0]
        for i in range(len(ann_locations)):
            annots[ann_locations[i]] = symbol[i]
            if symbol[i] == "+":
                auxiliary[ann_locations[i]] = aux[i]
        sig_dict = {
            "time": range(signal.shape[0]),
            "annots": annots,
            "auxiliary": auxiliary,
        }
        for ch in channels:
            sig_dict[ch] = signal[:, ch]
        return pd.DataFrame(sig_dict)


def save_data(data, file_path):
    """Saves the data into a file."""
    if file_path is None:
        raise ValueError("Save file path is not provided!")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print("File saved at: " + str(file_path))


def load_data(file_path=None):
    """Loads the data from a file."""
    if file_path is None:
        print("File path must be provided")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print("File loaded from: " + file_path)
    return data
