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

    def save(self, record_name="dummy101"):
        self.make_dir()
        self.record_name = record_name
        self.save_sig()
        self.save_ann()

    def save_sig(self):
        signal, _ = self.gen_sig(cycles=5)
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
        symbol = ["N", "N", "V", "N", "A"]
        sample = [90, 450, 810, 1170, 1530]
        sample.insert(0, 30)
        symbol.insert(0, "+")
        sample.insert(3, 700)
        symbol.insert(3, "+")
        sample = np.array(sample)
        aux_note = ["(N\x00", "", "(VT\x00", "", "", "", ""]

        wfdb.wrann(
            self.record_name,
            "atr",
            sample,
            symbol=symbol,
            aux_note=aux_note,
            fs=self.sampling_rate,
            write_dir=self.save_dir,
        )

    def gen_sig(self, cycles):
        x = list(range(0, 360 * cycles))
        y = np.sin(np.radians(x)) + 0.5
        peaks = [p * 360 + 90 for p in range(cycles)]
        y = y.reshape(-1, 1)
        sigs = np.repeat(y, len(self.channels), axis=1)
        return (sigs, peaks)

    def make_dir(self):
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)