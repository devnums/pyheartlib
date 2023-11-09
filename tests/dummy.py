#############################################################################
# Copyright (c) 2023 Pyheartlib team. - All Rights Reserved                 #
# Project repo: https://github.com/devnums/pyheartlib                       #
# Contact: devnums.code@gmail.com                                           #
#                                                                           #
# This file is part of the Pyheartlib project.                              #
# To see the complete LICENSE file visit:                                   #
# https://github.com/devnums/pyheartlib/blob/main/LICENSE                   #
#############################################################################


from pathlib import Path

import numpy as np
import wfdb


class DummyData:
    """Creates fake records."""

    def __init__(
        self,
        save_dir="./tests/dummy_data",
        sampling_rate=360,
        channels=["I", "MLII"],
    ):
        self.save_dir = save_dir
        self.sampling_rate = sampling_rate
        self.channels = channels

        self.annotations = [
            (30, "+", "(N\x00"),
            (90, "N", ""),
            (450, "N", ""),
            (810, "V", ""),
            (900, "+", "(VT\x00"),
            (1170, "N", ""),
            (1530, "A", ""),
            (1600, "N", ""),
            (1730, "N", ""),
            (1810, "N", ""),
            (1930, "N", ""),
            (2230, "N", ""),
            (2430, "N", ""),
            (2550, "N", ""),
            (2610, "N", ""),
            (2750, "N", ""),
            (2970, "N", ""),
            (3150, "N", ""),
            (3400, "N", ""),
        ]

    def save(self, record_name="dummy101"):
        self.make_dir()
        self.record_name = record_name
        self.save_sig()
        self.save_ann()

    def save_sig(self):
        signal, _ = self.gen_sig(cycles=10)
        wfdb.wrsamp(
            self.record_name,
            fs=self.sampling_rate,
            units=["mV", "mV"],
            sig_name=self.channels,
            p_signal=signal,
            fmt=["16", "16"],
            write_dir=self.save_dir,
        )

    def save_ann(self):
        samples = np.array([ann[0] for ann in self.annotations])
        symbols = np.array([ann[1] for ann in self.annotations])
        aux_notes = [ann[2] for ann in self.annotations]
        ann = wfdb.Annotation(
            record_name=self.record_name,
            extension="atr",
            sample=samples,
            symbol=symbols,
            aux_note=aux_notes,
        )

        ann.wrann(write_fs=False, write_dir=self.save_dir)

    def gen_sig(self, cycles):
        x = list(range(0, 360 * cycles))
        y = np.sin(np.radians(x)) + 0.5
        peaks = [p * 360 + 90 for p in range(cycles)]
        y = y.reshape(-1, 1)
        sigs = np.repeat(y, len(self.channels), axis=1)
        return (sigs, peaks)

    def make_dir(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
