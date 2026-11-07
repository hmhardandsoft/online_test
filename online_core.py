"""Hardware-independent, causal online SSVEP decoding.

Epochs: [band, EEG channel, time]. Labels and edge IDs are zero based here.
All public sample indices refer to the original 2000 Hz stream, stop exclusive.
"""
from dataclasses import dataclass, asdict
from collections import deque
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


METHODS = ("Aug_FB_LDA", "Aug_FB_TRCA", "Aug_FB_eTRCA")
WAIT, TRAIN, DRAIN, FIT, CATCHUP, ONLINE, END = range(7)
WAIT_TEST, WARMUP = 7, 8  # Preserve stage codes in existing recordings.
STAGES = ("waiting", "training", "draining", "fitting", "catchup", "online", "ended",
          "waiting_test", "test_warmup")
FLICKER_STAGES = (TRAIN, WARMUP, ONLINE)
# Metadata columns followed by M intensity columns.
LABEL, TRAIN_VALID, STAGE, SEGMENT, VALID, INTENSITY = range(6)


def metadata_columns(task_m):
    return ["label", "train_valid", "stage", "segment", "valid",
            *[f"intensity_{i+1}" for i in range(task_m)], "stimulus_id", "stimulus_frame"]


@dataclass
class Config:
    task_m: int = 4
    method: str = "Aug_FB_TRCA"
    fs: int = 2000
    window_size: int = 200
    decimation: int = 2
    eeg_channels: tuple = (0, 2, 3, 4, 5, 6)
    enable_rereference: bool = True
    reference_channels: tuple = (1, 7)
    enable_notch: bool = True
    notch_hz: float = 50.0
    notch_q: float = 30.0
    filter_order: int = 4
    bands: tuple = ((5., 95.), (12., 95.), (19., 95.), (27., 95.),
                    (35., 95.), (43., 95.), (51., 95.), (59., 95.),
                    (67., 95.), (75., 95.))
    training_blocks: int = 10
    max_extra_blocks: int = 2
    train_cue_seconds: tuple = (3., 5.)
    skip_seconds: float = .5
    min_samples: int = 20
    online_mode: str = "cued"
    online_seconds: float = 120.
    online_cue_seconds: tuple = (3., 5.)
    test_warmup_seconds: float = .5
    feedback: bool = True
    refresh_rate: int = 120
    frequency: float = 10.
    ring_seconds: float = 60.
    catchup_max_seconds: float = .02
    catchup_hold_seconds: float = .25
    max_prediction_lag_seconds: float = .1
    fit_timeout_seconds: float = 45.
    no_data_timeout_seconds: float = 3.
    seed: int = 20261104

    def validate(self):
        if self.fs != 2000 or self.task_m not in (2, 4) or self.method not in METHODS:
            raise ValueError("Require FS=2000, M=2/4 and a supported method")
        if self.window_size <= 0 or self.window_size % (2 * self.decimation):
            raise ValueError("window_size must be positive and divisible by 2*decimation")
        if self.decimation not in (1, 2) or self.filter_order < 1:
            raise ValueError("Use decimation 1/2 and positive filter order")
        for name, channels in (("EEG", self.eeg_channels), ("reference", self.reference_channels)):
            if (name == "EEG" or self.enable_rereference) and not channels:
                raise ValueError(f"Empty {name} channels")
            if len(set(channels)) != len(channels) or any(i not in range(8) for i in channels):
                raise ValueError(f"Invalid {name} channels (zero based EEG 0..7; trigger excluded)")
        if not self.bands or any(not 0 < lo < hi < self.fs / (2*self.decimation)
                                 for lo, hi in self.bands):
            raise ValueError("Invalid filter bank / decimated Nyquist")
        if not 0 < self.notch_hz < self.fs/2 or self.notch_q <= 0:
            raise ValueError("Invalid notch configuration")
        if self.training_blocks < 1 or self.max_extra_blocks < 0 or self.min_samples < 2:
            raise ValueError("Invalid training counts")
        if self.online_mode not in ("cued", "uncued") or self.online_seconds <= 0:
            raise ValueError("Invalid online mode/duration")
        if not np.isfinite(self.test_warmup_seconds) or self.test_warmup_seconds < 0:
            raise ValueError("test_warmup_seconds must be finite and nonnegative")
        for limits in (self.train_cue_seconds, self.online_cue_seconds):
            if len(limits) != 2 or not 0 < limits[0] <= limits[1]:
                raise ValueError("Invalid cue duration range")
        if self.skip_seconds < 0 or self.train_cue_seconds[0] <= self.skip_seconds + self.window_size/self.fs:
            raise ValueError("Training cue too short for skip + full window")
        if min(self.refresh_rate, self.frequency, self.catchup_max_seconds,
               self.catchup_hold_seconds, self.max_prediction_lag_seconds,
               self.fit_timeout_seconds, self.no_data_timeout_seconds) <= 0:
            raise ValueError("Timing values must be positive")
        if self.ring_seconds <= self.fit_timeout_seconds + 5:
            raise ValueError("Ring must exceed fit timeout by at least 5 s")
        return self

    def to_dict(self):
        return asdict(self)


class CausalFilterBank:
    def __init__(self, cfg):
        self.cfg = cfg
        ch = len(cfg.eeg_channels)
        self.notch = tf2sos(*iirnotch(cfg.notch_hz, cfg.notch_q, fs=cfg.fs))
        self.notch_zi = np.zeros((len(self.notch), 2, ch))
        self.sos = [butter(cfg.filter_order, band, btype="bandpass", fs=cfg.fs, output="sos")
                    for band in cfg.bands]
        self.zi = [np.zeros((len(s), 2, ch)) for s in self.sos]
        self.position = 0

    def process(self, raw):
        cfg = self.cfg
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 2 or raw.shape[1] < 8 or not np.isfinite(raw).all():
            raise ValueError("Invalid/nonfinite EEG; cannot silently preserve continuity")
        x = raw[:, cfg.eeg_channels].copy()
        if cfg.enable_rereference:
            x -= raw[:, cfg.reference_channels].mean(axis=1, keepdims=True)
        if cfg.enable_notch:
            x, self.notch_zi = sosfilt(self.notch, x, axis=0, zi=self.notch_zi)
        indices = np.arange(self.position, self.position + len(raw))
        keep = indices % cfg.decimation == 0
        filtered = []
        for b, sos in enumerate(self.sos):
            y, self.zi[b] = sosfilt(sos, x, axis=0, zi=self.zi[b])
            filtered.append(y[keep].T)
        self.position += len(raw)
        return indices[keep], np.stack(filtered)  # B,C,T


@dataclass
class Epoch:
    edge: int
    edge_sample: int
    start: int
    stop: int
    x: np.ndarray
    label: int
    segment: int
    training: bool
    online: bool


class EpochStream:
    """Causal filtering, fixed global decimation and delayed centered epochs.

Raw metadata for ALL 200 samples is checked, including discarded odd samples.
The window [edge-100, edge+100) becomes available only after its last raw sample.
"""
    def __init__(self, cfg):
        self.cfg = cfg
        self.filters = CausalFilterBank(cfg)
        self.raw_base = 0
        self.meta = np.empty((0, len(metadata_columns(cfg.task_m))))
        self.ds_indices = np.empty(0, dtype=np.int64)
        self.data = np.empty((len(cfg.bands), len(cfg.eeg_channels), 0))
        self.previous = None
        self.previous_stimulus = 0
        self.pending = deque()
        self.total = 0

    def feed(self, raw, metadata):
        cfg = self.cfg
        metadata = np.asarray(metadata, dtype=float)
        if metadata.shape != (len(raw), len(metadata_columns(cfg.task_m))):
            raise ValueError("Metadata shape mismatch")
        if not len(raw):
            return []
        indices, data = self.filters.process(raw)
        self.meta = np.concatenate((self.meta, metadata))
        self.ds_indices = np.concatenate((self.ds_indices, indices))
        self.data = np.concatenate((self.data, data), axis=2)
        half = cfg.window_size // 2
        stimulus_col = INTENSITY + cfg.task_m
        for i in indices:
            row = metadata[i-self.total]
            now = row[INTENSITY:stimulus_col] >= .5
            stimulus = int(row[stimulus_col]) if row[STAGE] in FLICKER_STAGES else 0
            # A black -> bright startup is not a periodic rising edge. Seed the
            # baseline on the first sampled row of each stimulus presentation.
            if stimulus and stimulus == self.previous_stimulus and self.previous is not None:
                for edge in np.flatnonzero(now & ~self.previous):
                    self.pending.append((int(i), int(edge)))
            self.previous = now if stimulus else None
            self.previous_stimulus = stimulus
        self.total += len(raw)
        result = []
        while self.pending and self.pending[0][0] + half <= self.total:
            edge_sample, edge = self.pending.popleft()
            start, stop = edge_sample-half, edge_sample+half
            if start < self.raw_base:
                continue
            rows = self.meta[start-self.raw_base:stop-self.raw_base]
            if len(rows) != cfg.window_size or not np.all(rows[:, VALID] == 1):
                continue
            center = rows[half]
            same_stimulus = bool(center[stimulus_col] > 0
                                 and np.all(rows[:, stimulus_col] == center[stimulus_col]))
            training = bool(same_stimulus and np.all(rows[:, STAGE] == TRAIN)
                            and np.all(rows[:, TRAIN_VALID] == 1)
                            and np.all(rows[:, SEGMENT] == center[SEGMENT])
                            and np.all(rows[:, LABEL] == center[LABEL])
                            and 1 <= center[LABEL] <= cfg.task_m)
            # Cues may change inside an online window. Black screens, restarts
            # and calibration data may not. Late warmup data can seed online.
            online = bool(same_stimulus and center[STAGE] == ONLINE
                          and np.all(np.isin(rows[:, STAGE], (WARMUP, ONLINE))))
            if not (training or online):
                continue
            a, b = np.searchsorted(self.ds_indices, (start, stop))
            x = self.data[:, :, a:b].copy()
            if x.shape[-1] != cfg.window_size // cfg.decimation:
                raise RuntimeError("Decimation/window alignment lost")
            result.append(Epoch(edge, edge_sample, start, stop, x,
                                int(center[LABEL])-1, int(center[SEGMENT]), training, online))
        retain = max(0, self.total - cfg.window_size - cfg.decimation)
        self.meta = self.meta[retain-self.raw_base:]
        self.raw_base = retain
        cut = np.searchsorted(self.ds_indices, retain)
        self.ds_indices = self.ds_indices[cut:]
        self.data = self.data[:, :, cut:]
        return result


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    a, b = a-a.mean(), b-b.mean()
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    if denominator < 1e-12:
        raise ValueError("Degenerate TRCA projection (no valid correlation)")
    return float(np.dot(a, b)/denominator)


class TRCABand:
    def __init__(self, ensemble=False):
        self.ensemble = ensemble

    def fit(self, groups):
        self.templates = np.stack([x.mean(axis=0) for x in groups])
        weights = []
        for x in groups:
            dc = x - x.mean(axis=2, keepdims=True)
            sx = dc.sum(axis=0)
            ux = dc.transpose(1, 2, 0).reshape(x.shape[1], -1, order="F")
            eye = np.eye(x.shape[1])*1e-9
            _, vectors = eigh(sx@sx.T + eye, ux@ux.T + eye)
            w = vectors[:, -1]
            weights.append(w/(np.linalg.norm(w)+1e-12))
        self.weights = np.stack(weights, axis=1)  # channels, classes
        # Validate templates; do not silently substitute uniform spatial weights.
        for c, t in enumerate(self.templates):
            projection = self.weights.T@t if self.ensemble else self.weights[:, c]@t
            correlation(projection, projection)
        return self

    def scores(self, x):
        if self.ensemble:
            projected = self.weights.T@x
            return np.array([correlation(projected, self.weights.T@t) for t in self.templates])
        return np.array([correlation(self.weights[:, c]@x, self.weights[:, c]@t)
                         for c, t in enumerate(self.templates)])


def decision_columns(model, x):
    value = model.decision_function(x)
    return value[:, None] if value.ndim == 1 else value


class EdgeModelBank:
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = [[] for _ in range(cfg.task_m)]
        self.labels = [[] for _ in range(cfg.task_m)]
        self.counts = np.zeros((cfg.task_m, cfg.task_m), dtype=int)
        self.banks = []
        self.frozen = False
        self.training_index = []

    def collect(self, epoch):
        if self.frozen:
            raise RuntimeError("Attempt to train a frozen model")
        if not epoch.training or not 0 <= epoch.label < self.cfg.task_m:
            raise ValueError("Invalid training epoch")
        self.samples[epoch.edge].append(epoch.x)
        self.labels[epoch.edge].append(epoch.label)
        self.counts[epoch.edge, epoch.label] += 1
        self.training_index.append((epoch.edge, epoch.label, epoch.segment,
                                    epoch.edge_sample, epoch.start, epoch.stop))

    def ready(self):
        return bool(np.all(self.counts >= self.cfg.min_samples))

    def fit(self):
        if self.frozen or not self.ready():
            raise RuntimeError(f"Fit requires unfrozen complete training groups: {self.counts}")
        cfg = self.cfg
        for r in range(cfg.task_m):
            x, y = np.stack(self.samples[r]), np.asarray(self.labels[r])
            models = []
            stacked = []
            for b in range(len(cfg.bands)):
                if cfg.method == "Aug_FB_LDA":
                    flat = x[:, b].reshape(len(x), -1)
                    model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(flat, y)
                    stacked.append(decision_columns(model, flat))
                else:
                    model = TRCABand(cfg.method == "Aug_FB_eTRCA").fit(
                        [x[y == c, b] for c in range(cfg.task_m)])
                models.append(model)
            final = None
            if cfg.method == "Aug_FB_LDA":
                # Exactly the offline in-sample stacking rule, not OOF stacking.
                final = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
                    np.concatenate(stacked, axis=1), y)
            self.banks.append((models, final))
        self.frozen = True
        self.samples = self.labels = None
        return self

    def predict(self, edge, x):
        if not self.frozen:
            raise RuntimeError("Prediction before freezing")
        expected = (len(self.cfg.bands), len(self.cfg.eeg_channels),
                    self.cfg.window_size // self.cfg.decimation)
        if x.shape != expected or not np.isfinite(x).all():
            raise ValueError("Invalid prediction epoch")
        models, final = self.banks[edge]
        if final is not None:
            z = np.concatenate([decision_columns(model, x[b].reshape(1, -1))
                                for b, model in enumerate(models)], axis=1)
            decision = final.decision_function(z).ravel()
            scores = np.array([-decision[0]/2, decision[0]/2]) if self.cfg.task_m == 2 else decision
        else:
            scores = sum(((b+1)**(-1.25) + .25)*model.scores(x[b])
                         for b, model in enumerate(models))
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite model scores")
        return int(np.argmax(scores)), scores


def time_accuracy(cues, predictions, start, stop, cued=True):
    """Integrate piecewise held outputs at actual publication times, not window times."""
    if stop <= start:
        return dict(duration_seconds=0., correct_seconds=None, time_accuracy=None)
    if not cued:
        return dict(duration_seconds=stop-start, correct_seconds=None, time_accuracy=None)
    changes = [(float(t), 0, int(label)) for t, label in cues]
    changes += [(float(t), 1, int(label)) for t, label in predictions]
    target = prediction = 0
    last, correct = start, 0.
    for t, kind, label in sorted(changes):
        if t > stop:
            break
        if t > start:
            if target > 0 and prediction == target:
                correct += t-last
            last = t
        if kind == 0:
            target = label
        else:
            prediction = label
    if target > 0 and prediction == target:
        correct += stop-last
    return dict(duration_seconds=stop-start, correct_seconds=correct,
                time_accuracy=correct/(stop-start))
