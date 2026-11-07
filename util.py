import multiprocessing as mp
from typing import Optional

import numpy as np
from scipy.signal import firwin, lfilter


class SharedState:
    """Shared process state for the continuous phase-coded SSVEP task."""

    MAX_TARGETS = 9
    EEG_CHANNELS = 9
    STATE_BASE_COLUMNS = [
        "frame_idx",
        "condition_m",
        "true_label",
        "target_phase_rad",
        "predict_label",
        "is_training",
        "start_predict",
        "current_freq",
        "cue_elapsed_sec",
        "cue_duration_sec",
        "paradigm_mode",
        "cue_position",
        "phase_label_at_position_1",
        "phase_label_at_position_2",
    ]
    STATE_COLUMNS = STATE_BASE_COLUMNS + [
        f"intensity_{i}" for i in range(1, MAX_TARGETS + 1)
    ]

    def __init__(
        self,
        window_size: int = 104,
        frequency: float = 10.0,
        fs: int = 2000,
        task_m: int = 9,
        paradigm_mode: str = "active",
        online: bool = True,
        block_count: int = 3,
        cue_duration_min: float = 3.0,
        cue_duration_max: float = 5.0,
        min_samples_per_class: int = 10,
        training_skip_after_cue_ms: float = 500.0,
        update_stride: int = 20,
        fusion_window_edges: int = 1,
        min_decode_interval_ms: float = 0.0,
        debug_decoder: bool = True,
    ):
        if task_m not in (2, 4, 9):
            raise ValueError("task_m must be 2, 4, or 9")
        paradigm_mode = str(paradigm_mode).lower()
        if paradigm_mode not in {"active", "passive"}:
            raise ValueError("paradigm_mode must be 'active' or 'passive'")
        if paradigm_mode == "passive" and task_m != 2:
            raise ValueError("passive paradigm requires task_m=2")
        if cue_duration_min <= 0 or cue_duration_max < cue_duration_min:
            raise ValueError("Invalid cue duration range")
        if training_skip_after_cue_ms < 0:
            raise ValueError("training_skip_after_cue_ms must be non-negative")

        self.window_size = mp.Value("i", int(window_size))
        self.fs = mp.Value("i", int(fs))
        self.frequency = mp.Value("f", float(frequency))
        self.current_freq = self.frequency
        self.task_m = mp.Value("i", int(task_m))
        self.paradigm_mode = paradigm_mode
        self.online = mp.Value("b", bool(online))
        self.block_count = mp.Value("i", int(block_count))
        self.cue_duration_min = mp.Value("f", float(cue_duration_min))
        self.cue_duration_max = mp.Value("f", float(cue_duration_max))
        self.min_samples_per_class = mp.Value("i", int(min_samples_per_class))
        self.training_skip_after_cue_ms = mp.Value(
            "d", float(training_skip_after_cue_ms)
        )
        self.update_stride = mp.Value("i", int(update_stride))
        self.fusion_window_edges = mp.Value("i", int(fusion_window_edges))
        self.min_decode_interval_ms = mp.Value("d", float(min_decode_interval_ms))
        self.debug_decoder = mp.Value("b", bool(debug_decoder))

        self.is_running = mp.Value("b", True)
        self.recorder_ready = mp.Value("b", False)
        self.start_requested = mp.Value("b", False)
        self.is_training = mp.Value("b", False)
        self.start_predict = mp.Value("b", False)

        self.frame_idx = mp.Value("i", 0)
        self.true_label = mp.Value("i", 0)
        self.target_phase = mp.Value("f", 0.0)
        self.predict_label = mp.Value("i", 0)
        self.cue_elapsed_sec = mp.Value("f", 0.0)
        self.cue_duration_sec = mp.Value("f", 0.0)
        self.cue_position = mp.Value("i", 0)
        self.phase_label_at_position_1 = mp.Value("i", 0)
        self.phase_label_at_position_2 = mp.Value("i", 0)
        self.stimulus_intensity = mp.Value("f", 0.0)
        self.intensities = mp.Array("f", self.MAX_TARGETS)

        self.state_width = len(self.STATE_COLUMNS)
        self.buff_width = self.EEG_CHANNELS + self.state_width
        # Four windows gives the centered-edge decoder enough history and slack.
        self.buff_len = max(int(window_size) * 4, int(window_size) + 8)
        self.buff_data = mp.Array("f", self.buff_len * self.buff_width)
        self.buff_idx = mp.Value("i", 0)

    @classmethod
    def state_columns(cls, task_m: int):
        return cls.STATE_BASE_COLUMNS + [
            f"intensity_{i}" for i in range(1, int(task_m) + 1)
        ]

    def get_state_vector(self):
        vals = [
            float(self.frame_idx.value),
            float(self.task_m.value),
            float(self.true_label.value),
            float(self.target_phase.value),
            float(self.predict_label.value),
            float(self.is_training.value),
            float(self.start_predict.value),
            float(self.frequency.value),
            float(self.cue_elapsed_sec.value),
            float(self.cue_duration_sec.value),
            float(self.paradigm_mode == "passive"),
            float(self.cue_position.value),
            float(self.phase_label_at_position_1.value),
            float(self.phase_label_at_position_2.value),
        ]
        vals.extend(float(v) for v in self.intensities[:])
        return np.asarray(vals, dtype=np.float32)


class BandpassFIR:
    def __init__(
        self,
        fs: int = 2000,
        lowcut: float = 5.0,
        highcut: float = 30.0,
        numtaps: int = 101,
        window: str = "hann",
    ):
        self.fs = fs
        if numtaps % 2 == 0:
            numtaps += 1
        self.b = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs, window=window)
        self.zi = None

    def _ensure_zi(self, n_ch: int):
        m = len(self.b) - 1
        if self.zi is None or self.zi.shape[1] != n_ch:
            self.zi = np.zeros((m, n_ch), dtype=float)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("BandpassFIR.process expects 2D array (n_samples, n_channels)")
        self._ensure_zi(x.shape[1])
        y, self.zi = lfilter(self.b, 1.0, x, axis=0, zi=self.zi)
        return y

    def reset(self, zi_value: Optional[float] = 0.0):
        if self.zi is not None:
            self.zi.fill(zi_value)


class NotchFIR:
    def __init__(
        self,
        fs: int = 2000,
        f0: float = 50.0,
        bandwidth: float = 2.0,
        numtaps: int = 401,
        window: str = "hann",
    ):
        self.fs = fs
        if numtaps % 2 == 0:
            numtaps += 1
        low = max(0.0, f0 - bandwidth / 2.0)
        high = min(fs / 2.0, f0 + bandwidth / 2.0)
        self.b = firwin(numtaps, [low, high], pass_zero="bandstop", fs=fs, window=window)
        self.zi = None

    def _ensure_zi(self, n_ch: int):
        m = len(self.b) - 1
        if self.zi is None or self.zi.shape[1] != n_ch:
            self.zi = np.zeros((m, n_ch), dtype=float)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError("NotchFIR.process expects 2D array (n_samples, n_channels)")
        self._ensure_zi(x.shape[1])
        y, self.zi = lfilter(self.b, 1.0, x, axis=0, zi=self.zi)
        return y

    def reset(self, zi_value: Optional[float] = 0.0):
        if self.zi is not None:
            self.zi.fill(zi_value)
