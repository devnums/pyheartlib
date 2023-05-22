import pandas as pd
import wfdb
import pickle


def get_data(record_path, config, return_dict=True):
    """Loads a record and returns its components.

    Parameters
    ----------
    record_path : str
        Path of the record.
    config : dict
        Config of the dataset.
    return_dict : bool
        If True returns as a dict, otherwise returns as a pandas dataframe.

    Returns
    -------
    dict or dataframe
        If return_dict is True, it returns a dictionary
        with keys: 'signal','r_locations','r_labels','rhythms','rhythms_locations'.

        If return_dict is False, it returns a dataframe containing the time, raw signal, and
        a list of equal size to the raw signal with None values except at annotations locations.
    """
    record = wfdb.rdrecord(record_path, channel_names=["MLII"])
    annotation = wfdb.rdann(record_path, "atr")
    signal = record.p_signal[:, 0]  # numpy.ndarray
    ann_locations = annotation.sample
    symbol = annotation.symbol
    aux = annotation.aux_note
    aux = [txt.rstrip("\x00") for txt in aux]

    r_labels = []
    r_locations = []
    rhythms = []
    rhythms_locations = []
    syms = config["BEAT_TYPES"]
    for i in range(len(ann_locations)):
        if symbol[i] in syms:
            r_labels.append(symbol[i])
            r_locations.append(ann_locations[i])

        if aux[i] in config["RHYTHM_TYPES"]:
            rhythms.append(aux[i])
            rhythms_locations.append(ann_locations[i])

    if return_dict is True:
        return {
            "signal": signal,
            "r_locations": r_locations,
            "r_labels": r_labels,
            "rhythms": rhythms,
            "rhythms_locations": rhythms_locations,
        }
    else:
        annots = [None] * len(signal)
        for i in range(len(ann_locations)):
            annots[ann_locations[i]] = symbol[i]
            if aux[i] != "":
                annots[ann_locations[i]] = aux[i]
        sig_dict = {"time": range(len(signal)), "signal": signal, "annots": annots}
        return pd.DataFrame(sig_dict)


def save_data(data, file_path):
    """Saves data"""
    if file_path is None:
        raise ValueError("Save file path is not provided!")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print("File saved at: " + str(file_path))


def load_data(file_path=None):
    """Loads data"""
    if file_path is None:
        print("File path must be provided")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print("File loaded from: " + file_path)
    return data
