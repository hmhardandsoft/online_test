#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continuous online decoder for same-frequency phase-coded SSVEP.

The first implementation uses rising-edge aligned samples only. For an M-class
task it trains M edge-type LDA models; each model is still an M-class classifier.
"""

from collections import Counter, deque
import multiprocessing as mp
import time

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from util import BandpassFIR, NotchFIR, SharedState


class RisingEdgeLDADecoder:
    def __init__(
        self,
        task_m: int,
        feature_window: int,
        n_features: int,
        min_samples_per_class: int,
        update_stride: int,
        debug: bool = True,
    ):
        self.task_m = int(task_m)
        self.feature_window = int(feature_window)
        self.n_features = int(n_features)
        self.min_samples_per_class = int(min_samples_per_class)
        self.update_stride = int(update_stride)
        self.debug = bool(debug)

        self.train_data = [
            np.empty((0, self.feature_window, self.n_features), dtype=np.float32)
            for _ in range(self.task_m)
        ]
        self.train_labels = [[] for _ in range(self.task_m)]
        self.models = [LinearDiscriminantAnalysis(solver="svd") for _ in range(self.task_m)]
        self.ready = [False for _ in range(self.task_m)]
        self.fitted = [False for _ in range(self.task_m)]
        self.samples_since_fit = [0 for _ in range(self.task_m)]

        self._debug(
            f"init M={self.task_m}, edge_models={self.task_m}, "
            f"min_per_class={self.min_samples_per_class}, update_stride={self.update_stride}"
        )

    def _debug(self, msg: str):
        if self.debug:
            print(f"[Decoder] {msg}", flush=True)

    def _class_counts(self, edge_idx: int):
        labels = self.train_labels[edge_idx]
        counts = Counter(labels)
        return [counts.get(label, 0) for label in range(1, self.task_m + 1)]

    def _is_ready(self, edge_idx: int):
        # sklearn LDA requires n_samples > n_classes. The per-class threshold
        # normally satisfies this, but keep the guard for low debug thresholds.
        return (
            all(c >= self.min_samples_per_class for c in self._class_counts(edge_idx))
            and len(self.train_labels[edge_idx]) > self.task_m
        )

    def ready_count(self):
        return sum(1 for is_ready in self.ready if is_ready)

    def all_ready(self):
        return self.ready_count() == self.task_m

    def _fit(self, edge_idx: int):
        x_train = self.train_data[edge_idx].reshape(self.train_data[edge_idx].shape[0], -1)
        y_train = np.asarray(self.train_labels[edge_idx], dtype=int)
        self.models[edge_idx].fit(x_train, y_train)
        self.fitted[edge_idx] = True
        self.samples_since_fit[edge_idx] = 0
        self._debug(
            f"fit edge={edge_idx + 1}/{self.task_m}, samples={len(y_train)}, "
            f"counts={self._class_counts(edge_idx)}"
        )

    def add_sample(self, edge_idx: int, feature: np.ndarray, label: int):
        if not 1 <= label <= self.task_m:
            return

        edge_idx = int(edge_idx)
        self.train_data[edge_idx] = np.append(
            self.train_data[edge_idx], [feature.astype(np.float32, copy=True)], axis=0
        )
        self.train_labels[edge_idx].append(int(label))
        self.samples_since_fit[edge_idx] += 1

        label_count = self._class_counts(edge_idx)[label - 1]
        if label_count == 1 or label_count % self.min_samples_per_class == 0:
            self._debug(
                f"sample edge={edge_idx + 1}, label={label}, "
                f"label_count={label_count}, counts={self._class_counts(edge_idx)}"
            )

        if not self.ready[edge_idx] and self._is_ready(edge_idx):
            self.ready[edge_idx] = True
            self._debug(
                f"ready edge={edge_idx + 1}/{self.task_m}, "
                f"ready_models={self.ready_count()}/{self.task_m}"
            )
            self._fit(edge_idx)
            return

        if self.ready[edge_idx] and self.samples_since_fit[edge_idx] >= self.update_stride:
            self._fit(edge_idx)

    def predict(self, edge_idx: int, feature: np.ndarray):
        edge_idx = int(edge_idx)
        if not self.fitted[edge_idx]:
            return 0
        x_test = feature.reshape(1, -1)
        return int(self.models[edge_idx].predict(x_test)[0])


class SSVEPdecoderProcess(mp.Process):
    def __init__(self, shared_state: SharedState):
        super().__init__()
        self.share = shared_state

    @staticmethod
    def _vote(recent_preds):
        if not recent_preds:
            return 0
        counts = Counter(recent_preds)
        best_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == best_count}
        for label in reversed(recent_preds):
            if label in tied:
                return int(label)
        return int(recent_preds[-1])

    def run(self):
        window_size = int(self.share.window_size.value)
        half_window = window_size // 2
        fs = int(self.share.fs.value)
        task_m = int(self.share.task_m.value)
        min_samples = int(self.share.min_samples_per_class.value)
        training_skip_after_cue_ms = float(
            self.share.training_skip_after_cue_ms.value
        )
        update_stride = int(self.share.update_stride.value)
        fusion_window_edges = max(1, int(self.share.fusion_window_edges.value))
        min_decode_interval_ms = float(self.share.min_decode_interval_ms.value)
        if min_decode_interval_ms > 0:
            min_decode_interval_samples = int(
                round(fs * min_decode_interval_ms / 1000.0)
            )
        else:
            min_decode_interval_samples = 0
        debug = bool(self.share.debug_decoder.value)

        buff_len = int(self.share.buff_len)
        buff_width = int(self.share.buff_width)
        raw_cols = SharedState.EEG_CHANNELS
        state_start = raw_cols
        true_label_col = state_start + SharedState.STATE_COLUMNS.index("true_label")
        is_training_col = state_start + SharedState.STATE_COLUMNS.index("is_training")
        intensity_start = state_start + SharedState.STATE_COLUMNS.index("intensity_1")

        buff_data = np.frombuffer(
            self.share.buff_data.get_obj(), dtype=np.float32
        ).reshape((buff_len, buff_width))

        bands = [(5.0, 95.0), (12.0, 95.0), (19.0, 95.0)]
        notch = NotchFIR(fs=fs, f0=50.0, bandwidth=4.0, numtaps=101)
        filters = [BandpassFIR(fs=fs, lowcut=low, highcut=high, numtaps=101) for low, high in bands]
        n_channels = 5
        n_bands = len(filters)
        feature_window = int(np.ceil(window_size / 8.0))
        decoder = RisingEdgeLDADecoder(
            task_m=task_m,
            feature_window=feature_window,
            n_features=n_channels * n_bands,
            min_samples_per_class=min_samples,
            update_stride=update_stride,
            debug=debug,
        )

        data_filter = np.zeros((1, n_channels), dtype=np.float32)
        data_window = np.zeros((window_size, n_channels * n_bands), dtype=np.float32)
        recent_preds = deque(maxlen=fusion_window_edges)
        predict_idx = 0
        last_decode_abs_idx = None
        has_reported_all_ready = False
        prediction_count = 0

        if debug:
            print(
                f"[Decoder] run fs={fs}, window={window_size}, half={half_window}, "
                f"channels=raw[1:6], fusion_window_edges={fusion_window_edges}, "
                f"training_skip_after_cue_ms={training_skip_after_cue_ms}, "
                f"min_decode_interval_ms={min_decode_interval_ms}, "
                f"min_decode_interval_samples={min_decode_interval_samples}",
                flush=True,
            )

        while self.share.is_running.value:
            if predict_idx >= self.share.buff_idx.value:
                time.sleep(0.001)
                continue

            row = buff_data[predict_idx % buff_len]
            # Keep the original decoder's five-channel choice: raw columns 1..5.
            data_filter[0, :] = row[1:6]
            data_filter = notch.process(data_filter)
            data_window[:-1, :] = data_window[1:, :]
            for band_idx, filt in enumerate(filters):
                filtered = filt.process(data_filter)
                start = band_idx * n_channels
                data_window[-1, start : start + n_channels] = filtered[0]

            current_abs_idx = predict_idx
            predict_idx += 1

            if not self.share.start_requested.value or current_abs_idx < window_size:
                continue

            # Check the edge half a window in the past so the edge peak is centered.
            edge_abs_idx = current_abs_idx - half_window
            prev_edge_row = buff_data[(edge_abs_idx - 1) % buff_len]
            edge_row = buff_data[edge_abs_idx % buff_len]
            label = int(edge_row[true_label_col])
            if not 1 <= label <= task_m:
                continue

            feature = data_window[::8, :]
            if feature.shape[0] != feature_window:
                # This only protects unusual window/downsample combinations.
                feature = feature[:feature_window, :]

            detected_edges = []
            for edge_idx in range(task_m):
                prev_intensity = prev_edge_row[intensity_start + edge_idx]
                curr_intensity = edge_row[intensity_start + edge_idx]
                if prev_intensity < 0.5 <= curr_intensity:
                    detected_edges.append(edge_idx)

            if not detected_edges:
                continue

            if min_decode_interval_samples <= 0 or last_decode_abs_idx is None:
                enough_interval = True
            else:
                enough_interval = (
                    current_abs_idx - last_decode_abs_idx
                    >= min_decode_interval_samples
                )

            can_predict = decoder.all_ready()
            if can_predict and enough_interval:
                preds = [decoder.predict(edge_idx, feature) for edge_idx in detected_edges]
                valid = [pred for pred in preds if pred > 0]
                if valid:
                    instant_pred = self._vote(valid)
                    recent_preds.append(instant_pred)
                    fused_pred = self._vote(list(recent_preds))
                    self.share.predict_label.value = fused_pred
                    last_decode_abs_idx = current_abs_idx
                    prediction_count += 1
                    if debug and (prediction_count <= 10 or prediction_count % 20 == 0):
                        edge_names = [edge_idx + 1 for edge_idx in detected_edges]
                        print(
                            f"[Decoder] predict edges={edge_names}, preds={preds}, "
                            f"instant={instant_pred}, fused={fused_pred}, "
                            f"votes={list(recent_preds)}, abs_idx={current_abs_idx}",
                            flush=True,
                        )
            elif not can_predict:
                self.share.start_predict.value = False
                self.share.predict_label.value = 0

            # Gate training with the state captured at the centered edge. Prediction
            # remains active during the post-cue transition interval.
            if edge_row[is_training_col] >= 0.5:
                for edge_idx in detected_edges:
                    decoder.add_sample(edge_idx, feature, label)

            if decoder.all_ready() and not has_reported_all_ready:
                has_reported_all_ready = True
                self.share.start_predict.value = True
                print("[Decoder] all edge models ready; prediction enabled.", flush=True)
