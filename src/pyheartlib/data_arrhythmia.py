from warnings import warn

from pyheartlib.data_rhythm import RhythmData

msg = (
    "Module data_arrhythmia will be deprecated in future versions. "
    "Use module data_rhythm instead."
)
warn(
    msg,
    DeprecationWarning,
    stacklevel=2,
)

msg = "ArrhythmiaData class will be deprecated. Use RhythmData class instead."


class ArrhythmiaData(RhythmData):
    def __init__(self, *args, **kwargs):
        warn(
            msg,
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
