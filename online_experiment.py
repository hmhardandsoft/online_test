#!/usr/bin/env python
"""Continuous SSVEP: cued calibration -> frozen model -> cued/uncued online.

Run --help. Hardware/display are opened only by main(), never on import.
"""
import os
# Small covariance problems should not occupy every CPU while the display runs.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import csv
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import random
import secrets
import time
import traceback
from uuid import uuid4
import numpy as np
from scipy.io import savemat, loadmat
import joblib
from threadpoolctl import threadpool_limits
from util import SharedState
from timing_diagnostics import TimingDiagnostics
from online_core import (Config, METHODS, WAIT, TRAIN, DRAIN, FIT, CATCHUP, ONLINE, END,
                         WAIT_TEST, WARMUP, FLICKER_STAGES, metadata_columns,
                         STAGES, LABEL, TRAIN_VALID, STAGE, SEGMENT, VALID, INTENSITY,
                         EpochStream, EdgeModelBank, fit_all_methods, time_accuracy)


ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class OnlineState(SharedState):
    def __init__(self, cfg):
        super().__init__(window_size=cfg.window_size, frequency=cfg.frequency, fs=cfg.fs,
                         task_m=cfg.task_m, paradigm_mode="active", online=False,
                         block_count=cfg.training_blocks, cue_duration_min=cfg.train_cue_seconds[0],
                         cue_duration_max=cfg.train_cue_seconds[1], min_samples_per_class=cfg.min_samples,
                         training_skip_after_cue_ms=cfg.skip_seconds*1000, min_decode_interval_ms=0)
        self.snapshot_lock = mp.RLock()
        self.ring_lock = mp.RLock()
        self.prediction_lock = mp.RLock()
        self.stage = mp.Value("i", WAIT)
        self.segment = mp.Value("q", 0)
        self.stimulus_id = mp.Value("q", 0)
        self.stimulus_frame = mp.Value("q", -1)
        self.buff_width = 9 + len(self.STATE_COLUMNS) + len(metadata_columns(cfg.task_m))
        self.buff_len = int(cfg.fs*cfg.ring_seconds)
        self.buff_data = mp.RawArray("d", self.buff_len*self.buff_width)
        self.buff_idx = mp.Value("q", 0)
        self.read_idx = mp.Value("q", 0)
        self.close_idx = mp.Value("q", -1)
        self.drained = mp.Value("b", False)
        self.train_ready = mp.Value("b", False)
        self.fit_requested = mp.Value("b", False)
        self.frozen = mp.Value("b", False)
        self.failed = mp.Value("b", False)
        self.last_batch = mp.Value("d", 0.)
        self.prediction_time = mp.Value("d", 0.)
        self.test_id = mp.Value("i", 0)
        self.method_index = mp.Value("i", -1)


def training_ready(share, cfg, now):
    """Only show the first prompt after amplifier warmup and a fresh published batch."""
    return bool(share.recorder_ready.value and share.buff_idx.value > 0
                and now-share.last_batch.value <= cfg.no_data_timeout_seconds)


def ring_view(share):
    return np.frombuffer(share.buff_data, dtype=np.float64).reshape(share.buff_len, share.buff_width)


def publish_rows(share, rows):
    """Preserve row-wise writes/index publication; lock prevents torn/overwritten reads."""
    with share.ring_lock:
        start = share.buff_idx.value
        if start + len(rows) - share.read_idx.value > share.buff_len:
            raise RuntimeError("EEG ring overrun; stopping instead of silently overwriting unread data")
        ring = ring_view(share)
        for row in rows:
            ring[share.buff_idx.value % share.buff_len] = row
            share.buff_idx.value += 1
        return start


def read_rows(share, maximum=256):
    with share.ring_lock:
        start, stop = share.read_idx.value, share.buff_idx.value
        if stop-start > share.buff_len:
            raise RuntimeError("EEG ring overrun")
        stop = min(stop, start+maximum)
        indices = np.arange(start, stop) % share.buff_len
        return start, ring_view(share)[indices].copy()


def fail(share, output, worker, exc):
    write_json(Path(output)/f"{worker}_error.json", dict(error=str(exc), traceback=traceback.format_exc()))
    print(f"[{worker}] ERROR: {exc}", flush=True)
    share.failed.value = True
    share.is_running.value = False


def acquisition_worker(share, cfg, output, simulate, device_name):
    recorder = None
    chunks, diagnostics = [], []
    sample_count = 0
    started = time.perf_counter()
    try:
        parser = None
        if not simulate:
            # Existing tested SDK connection helper; acquisition still uses get_data(with_q=True).
            from usb_acquisition_benchmark import connect_real, bounded_call
            args = argparse.Namespace(device_type="USB8", fs=cfg.fs, device_name=device_name,
                                      connect_timeout=15)
            recorder = connect_real(args)
            bounded_call(lambda: recorder.start_acquisition_data(with_q=True), 10, "start acquisition")
            until = time.perf_counter()+3.
            while time.perf_counter() < until and share.is_running.value:
                recorder.get_data(timeout=.01)
            parser = getattr(recorder, "_iRecorder__parser", None)
            if parser is None or not hasattr(parser, "_drop_count"):
                raise RuntimeError("Installed SDK has no expected drop counter; inspect SDK before running")
        baseline_drops = int(parser._drop_count) if parser else 0
        write_json(Path(output)/"acquisition_start.json", dict(simulated=simulate,
                   warmup_drop_events=baseline_drops, fs=cfg.fs,
                   timestamp_meaning="host get_data return / state snapshot / ring publication; not ADC time"))
        rng = np.random.default_rng(cfg.seed)
        next_batch = time.perf_counter()
        last_data = next_batch
        share.last_batch.value = last_data
        share.recorder_ready.value = True
        while share.is_running.value:
            if simulate:
                next_batch += .005
                time.sleep(max(0., next_batch-time.perf_counter()))
                frames = np.zeros((10, 9))
                t = np.arange(sample_count, sample_count+10)/cfg.fs
                label = max(1, share.true_label.value)
                phase = 2*np.pi*(label-1)/cfg.task_m
                for ch in range(8):
                    frames[:, ch] = ((1+.1*ch)*np.sin(2*np.pi*cfg.frequency*t+phase)
                                      + .3*np.sin(4*np.pi*cfg.frequency*t+2*phase)
                                      + rng.normal(0, .15, len(t)))
            else:
                frames = recorder.get_data(timeout=.01)
            returned = time.perf_counter()
            if frames is None or len(frames) == 0:
                if returned-last_data > cfg.no_data_timeout_seconds:
                    raise RuntimeError("Amplifier stopped delivering data")
                continue
            raw = np.asarray(frames, dtype=float)
            if raw.ndim != 2 or raw.shape[1] < 9 or not np.isfinite(raw).all():
                raise RuntimeError("Malformed/nonfinite amplifier batch")
            raw = raw[:, :9]
            drops = int(parser._drop_count)-baseline_drops if parser else 0
            snapshot_before = time.perf_counter()
            with share.snapshot_lock:
                state = share.get_state_vector()
                meta = np.array([share.true_label.value, share.is_training.value,
                                 share.stage.value, share.segment.value, int(drops == 0),
                                 *share.intensities[:cfg.task_m], share.stimulus_id.value,
                                 share.stimulus_frame.value, share.test_id.value], dtype=float)
            snapshot_after = time.perf_counter()
            rows = np.column_stack((raw, np.tile(np.r_[state, meta], (len(raw), 1))))
            # Save the anomalous batch as evidence, but never train/predict across it.
            chunks.append(rows)
            if drops:
                diagnostics.append(dict(start_sample=sample_count, stop_sample=sample_count+len(raw),
                                        get_return=returned, snapshot_before=snapshot_before,
                                        snapshot_after=snapshot_after, publish_time=None, drop_events=drops))
                raise RuntimeError(f"SDK reported {drops} sequence/checksum events after warmup; saved and stopped")
            start = publish_rows(share, rows)
            published = time.perf_counter()
            diagnostics.append(dict(start_sample=start, stop_sample=start+len(raw), get_return=returned,
                                    snapshot_before=snapshot_before, snapshot_after=snapshot_after,
                                    publish_time=published, drop_events=drops))
            sample_count += len(raw)
            last_data = returned
            share.last_batch.value = returned
    except BaseException as exc:
        fail(share, output, "recorder", exc)
    finally:
        try:
            if recorder is not None:
                from usb_acquisition_benchmark import bounded_call
                bounded_call(recorder.stop_acquisition, 5, "stop acquisition")
                bounded_call(recorder.close_dev, 5, "close amplifier")
        except Exception as exc:
            write_json(Path(output)/"recorder_cleanup_error.json", dict(error=str(exc)))
        try:
            if chunks:
                rows = np.concatenate(chunks)
                state_end = 9+len(SharedState.STATE_COLUMNS)
                np.savez_compressed(Path(output)/"recording.npz", eeg=rows[:, :9],
                                    states=rows[:, 9:state_end], metadata=rows[:, state_end:],
                                    metadata_columns=np.array(metadata_columns(cfg.task_m)),
                                    state_columns=np.array(SharedState.STATE_COLUMNS), fs=cfg.fs)
                savemat(Path(output)/"EEG.mat", {"data": rows[:, :9], "Fs": cfg.fs})
                savemat(Path(output)/"States.mat", {"states": rows[:, 9:state_end], "Fs": cfg.fs,
                         "metadata": rows[:, state_end:],
                         "metadata_columns": np.asarray(metadata_columns(cfg.task_m), dtype=object),
                         "state_columns": np.asarray(SharedState.STATE_COLUMNS, dtype=object)})
            write_csv(Path(output)/"batches.csv", diagnostics)
            write_json(Path(output)/"acquisition_summary.json", dict(samples=sum(len(x) for x in chunks),
                       published_samples=sample_count, elapsed=time.perf_counter()-started))
        except BaseException as exc:
            fail(share, output, "recorder_save", exc)


def decoder_worker(share, cfg, output, method_order):
    stream, bank = EpochStream(cfg), EdgeModelBank(cfg)
    models = None
    predictions, diagnostics = [], []
    fit_seconds = None
    try:
        with threadpool_limits(limits=1):
            while share.is_running.value:
                if share.fit_requested.value and models is None:
                    started = time.perf_counter()
                    models = fit_all_methods(bank)
                    joblib.dump(dict(config=cfg.to_dict(), models=models), Path(output)/"frozen_model.joblib")
                    fit_seconds = time.perf_counter()-started
                    share.frozen.value = True
                    print(f"[Decoder] All methods frozen in {fit_seconds:.3f}s; counts={bank.counts.tolist()}", flush=True)
                start, rows = read_rows(share)
                if len(rows):
                    processed_at = time.perf_counter()
                    epochs = stream.feed(rows[:, :9], rows[:, 9+len(SharedState.STATE_COLUMNS):])
                    for epoch in epochs:
                        if epoch.training and not bank.frozen:
                            bank.collect(epoch)
                        elif epoch.online and models is not None:
                            lag = (share.buff_idx.value-epoch.stop)/cfg.fs
                            test_id = epoch.test_id
                            method = method_order[test_id-1] if 1 <= test_id <= len(method_order) else None
                            current = (share.stage.value == ONLINE and share.test_id.value == test_id
                                       and share.stimulus_id.value == epoch.stimulus_id and method is not None
                                       and share.method_index.value == METHODS.index(method))
                            if lag > cfg.max_prediction_lag_seconds or not current:
                                predictions.append(dict(edge=epoch.edge+1, edge_sample=epoch.edge_sample,
                                    start=epoch.start, stop=epoch.stop, publication_time=None, prediction=0,
                                    scores=None, lag_seconds=lag, test_id=test_id, method=method,
                                    stimulus_id=epoch.stimulus_id, status="skipped_stale_or_ended"))
                                continue
                            label, scores = models[method].predict(epoch.edge, epoch.x)
                            # Recheck AFTER computation as well; never publish a stale result knowingly.
                            lag = (share.buff_idx.value-epoch.stop)/cfg.fs
                            published = None
                            with share.prediction_lock:
                                if (lag <= cfg.max_prediction_lag_seconds and share.stage.value == ONLINE
                                        and share.test_id.value == test_id
                                        and share.stimulus_id.value == epoch.stimulus_id
                                        and share.method_index.value == METHODS.index(method)):
                                    published = time.perf_counter()
                                    share.predict_label.value = label+1
                                    share.prediction_time.value = published
                            predictions.append(dict(edge=epoch.edge+1, edge_sample=epoch.edge_sample,
                                start=epoch.start, stop=epoch.stop, publication_time=published,
                                prediction=label+1, scores=scores.tolist(), lag_seconds=lag,
                                test_id=test_id, method=method, stimulus_id=epoch.stimulus_id,
                                status="published" if published is not None else "skipped_stale_or_ended"))
                    share.read_idx.value = start+len(rows)
                    share.train_ready.value = bank.ready()
                    diagnostics.append(dict(start=start, stop=start+len(rows),
                        processing_seconds=time.perf_counter()-processed_at,
                        backlog_samples=share.buff_idx.value-share.read_idx.value))
                with share.ring_lock:
                    # One full window beyond the boundary drains centered windows and
                    # a recorder batch whose snapshot preceded the stage transition.
                    if share.close_idx.value >= 0 and share.read_idx.value >= share.close_idx.value+cfg.window_size:
                        share.drained.value = True
                if not len(rows):
                    time.sleep(.001)
    except BaseException as exc:
        fail(share, output, "decoder", exc)
    finally:
        try:
            write_json(Path(output)/"predictions.json", predictions)
            write_csv(Path(output)/"decoder_batches.csv", diagnostics)
            write_json(Path(output)/"training.json", dict(counts=bank.counts.tolist(),
                       frozen=bool(share.frozen.value), methods=list(models) if share.frozen.value else [],
                       fit_seconds=fit_seconds,
                       index_columns=["edge_zero_based", "class_zero_based", "segment", "edge_sample", "start", "stop"],
                       windows=bank.training_index))
        except BaseException as exc:
            fail(share, output, "decoder_save", exc)


class Schedule:
    """Wall-clock cue timing. A cue starts at its first software swap timestamp."""
    def __init__(self, cfg):
        self.cfg, self.rng = cfg, random.Random(cfg.seed)
        self.test_seed = secrets.randbits(64)
        order_rng = random.Random(self.test_seed)
        self.method_order = list(METHODS)
        order_rng.shuffle(self.method_order)
        self.test_seeds = [order_rng.getrandbits(64) for _ in METHODS]
        self.test_rngs = [random.Random(seed) for seed in self.test_seeds]
        self.test_records = [dict(test_id=i+1, method=method, cue_seed=self.test_seeds[i],
                                  stop_mode=cfg.online_stop_mode, requested_seconds=cfg.online_seconds,
                                  requested_cue_count=cfg.online_cue_count, cue_count_presented=0,
                                  cue_count_completed=0, ready_time=None, space_accepted_at=None,
                                  stimulus_start=None, online_start=None, online_end=None,
                                  stop_reason=None)
                             for i, method in enumerate(self.method_order)]
        self.test_index = 0
        self.pending_finish = None
        self.online_cue_count = 0
        self.cues = []
        self.blocks = 0
        for _ in range(cfg.training_blocks):
            self.add_block()
        self.index = 0
        self.stage = TRAIN
        self.segment = 1
        self.cue_start = None
        self.online_start = None
        self.stage_start = None
        self.caught_since = None
        self.online_cue = None
        self.test_ready_time = None
        self.test_space_accepted_at = None
        self.test_stimulus_start = None
        self.stimulus_serial = 0
        self.stimulus_id = 0
        self.stimulus_frame = -1

    def add_block(self):
        labels = list(range(1, self.cfg.task_m+1))
        self.rng.shuffle(labels)
        if self.cues and labels[0] == self.cues[-1][0]:
            labels[0], labels[1] = labels[1], labels[0]
        self.cues.extend((label, self.rng.uniform(*self.cfg.train_cue_seconds)) for label in labels)
        self.blocks += 1

    def next_online(self):
        rng = self.test_rngs[self.test_index]
        previous = self.online_cue[0] if self.online_cue else 0
        label = rng.choice([i for i in range(1, self.cfg.task_m+1) if i != previous])
        self.online_cue = (label, rng.uniform(*self.cfg.online_cue_seconds))
        self.online_cue_count += 1
        self.test_records[self.test_index]["cue_count_presented"] = self.online_cue_count
        self.segment += 1
        self.cue_start = None

    def current_test_id(self):
        return self.test_index+1 if self.stage in (WARMUP, ONLINE) else 0

    def current_method(self):
        return self.method_order[self.test_index] if self.stage in (WARMUP, ONLINE) else None

    def cue(self):
        if self.stage == TRAIN:
            return self.cues[self.index]
        if self.stage in (WARMUP, ONLINE) and self.cfg.online_mode == "cued":
            return self.online_cue
        return (0, 0.)

    def next_stimulus_frame(self):
        """Keep global frame count elsewhere; restart only a new stimulus run."""
        if self.stage in FLICKER_STAGES:
            if self.stimulus_id == 0:
                self.stimulus_serial += 1
                self.stimulus_id = self.stimulus_serial
                self.stimulus_frame = 0
            else:
                self.stimulus_frame += 1
        else:
            self.stimulus_id, self.stimulus_frame = 0, -1

    def decoder_ready(self, now, share):
        return bool(share.frozen.value
                    and (share.buff_idx.value-share.read_idx.value)/self.cfg.fs <= self.cfg.catchup_max_seconds
                    and now-share.last_batch.value <= max(.1, 2*self.cfg.catchup_max_seconds))

    def advance(self, now, share, start_test=False):
        if self.stage == TRAIN and self.cue_start is not None:
            if now-self.cue_start >= self.cue()[1]:
                self.index += 1
                self.segment += 1
                self.cue_start = None
                if self.index == len(self.cues):
                    self.stage = DRAIN
        elif self.stage == DRAIN and share.drained.value:
            if share.train_ready.value:
                self.stage = FIT
                share.fit_requested.value = True
            elif self.blocks < self.cfg.training_blocks+self.cfg.max_extra_blocks:
                # Reset barrier while holding the same lock as decoder's acknowledgement.
                with share.ring_lock:
                    share.close_idx.value = -1
                    share.drained.value = False
                self.add_block()
                self.stage = TRAIN
                print(f"[Training] Added block {self.blocks}: more valid epochs needed", flush=True)
            else:
                raise RuntimeError("Insufficient samples after maximum extra training blocks")
        elif self.stage == FIT:
            if share.frozen.value:
                self.stage = CATCHUP
            elif self.stage_start is not None and now-self.stage_start > self.cfg.fit_timeout_seconds:
                raise RuntimeError("Model fit exceeded configured timeout")
        elif self.stage == CATCHUP:
            if self.decoder_ready(now, share):
                self.caught_since = now if self.caught_since is None else self.caught_since
                if now-self.caught_since >= self.cfg.catchup_hold_seconds:
                    self.stage = WAIT_TEST
            else:
                self.caught_since = None
            if self.stage_start is not None and now-self.stage_start > 60:
                raise RuntimeError("Frozen decoder could not catch up with acquisition")
        elif self.stage == WAIT_TEST:
            if not self.decoder_ready(now, share):
                # Hide the prompt until readiness is restored; an old keypress
                # is discarded, and the subject must press again after readiness.
                self.stage = CATCHUP
                self.caught_since = None
                self.test_ready_time = None
            elif start_test and self.test_ready_time is not None:
                self.test_space_accepted_at = now
                self.test_records[self.test_index]["space_accepted_at"] = now
                self.stage = WARMUP
                self.online_cue = None
                self.online_cue_count = 0
                self.online_start = None
                self.cue_start = None
                self.test_stimulus_start = None
                if self.cfg.online_mode == "cued":
                    self.next_online()
                with share.prediction_lock:
                    share.predict_label.value = 0
                    share.prediction_time.value = 0.
        elif self.stage == WARMUP:
            if (self.test_stimulus_start is not None
                    and now-self.test_stimulus_start >= self.cfg.test_warmup_seconds
                    and self.decoder_ready(now, share)):
                self.stage = ONLINE
                # The first cue remains visible; its scored duration starts now.
                self.cue_start = None
        elif self.stage == ONLINE and self.online_start is not None:
            reached_time = (self.cfg.online_stop_mode == "time"
                            and now-self.online_start >= self.cfg.online_seconds)
            cue_done = (self.cfg.online_mode == "cued" and self.cue_start is not None
                        and now-self.cue_start >= self.cue()[1])
            reached_count = (self.cfg.online_stop_mode == "segments" and cue_done
                             and self.online_cue_count >= self.cfg.online_cue_count)
            if reached_time or reached_count:
                if cue_done:
                    self.test_records[self.test_index]["cue_count_completed"] += 1
                self.test_records[self.test_index]["stop_reason"] = (
                    "time_limit" if reached_time else "completed_cue_count")
                self.pending_finish = self.test_index
                self.test_index += 1
                self.stage = END if self.test_index == len(self.method_order) else CATCHUP
                self.caught_since = None
            elif cue_done:
                self.test_records[self.test_index]["cue_count_completed"] += 1
                self.next_online()


def present_state(share, cfg, schedule, frame, intensities, timestamp):
    label, duration = schedule.cue()
    elapsed = 0. if schedule.cue_start is None else max(0., timestamp-schedule.cue_start)
    with share.snapshot_lock:
        share.stage.value = schedule.stage
        share.segment.value = schedule.segment
        share.stimulus_id.value = schedule.stimulus_id
        share.stimulus_frame.value = schedule.stimulus_frame
        share.test_id.value = schedule.current_test_id()
        share.method_index.value = (METHODS.index(schedule.current_method())
                                    if schedule.current_method() else -1)
        share.frame_idx.value = frame
        share.true_label.value = label
        share.target_phase.value = 2*np.pi*(label-1)/cfg.task_m if label else 0.
        share.cue_position.value = label
        share.phase_label_at_position_1.value = 1
        share.phase_label_at_position_2.value = 2
        share.cue_elapsed_sec.value = elapsed
        share.cue_duration_sec.value = duration
        share.is_training.value = schedule.stage == TRAIN and elapsed >= cfg.skip_seconds
        share.start_predict.value = schedule.stage == ONLINE
        for i in range(SharedState.MAX_TARGETS):
            share.intensities[i] = intensities[i] if i < cfg.task_m else 0.
        share.stimulus_intensity.value = intensities[0]


def run_live(cfg, output, simulate=False, headless=False, device_name=None):
    if headless and not simulate:
        raise ValueError("--headless is only for synthetic software tests")
    share = OnlineState(cfg)
    workers, frames, events = [], [], []
    stim = graphics = glfw = None
    completed, error = False, None
    schedule = Schedule(cfg)
    online_end = None
    train_ready_time = train_prompt_time = train_space_accepted_at = None
    timing = TimingDiagnostics(cfg.timing_diagnostics)
    try:
        if not headless:
            import glfw
            import SSVEP_stimulator as graphics
            stim = graphics.SSVEPStimulator(share, required_refresh_rate=cfg.refresh_rate)
            if stim.refresh_rate != cfg.refresh_rate:
                raise RuntimeError(f"Display is {stim.refresh_rate} Hz, require {cfg.refresh_rate} Hz")
        timing.attach(glfw, stim.window if stim else None)
        workers = [mp.Process(target=acquisition_worker, name="online-recorder",
                              args=(share, cfg, str(output), simulate, device_name)),
                   mp.Process(target=decoder_worker, name="online-decoder",
                              args=(share, cfg, str(output), schedule.method_order))]
        for worker in workers:
            worker.start()
        deadline = time.perf_counter()+30
        return_code = None
        train_prompt_visible = False
        train_ready_time = train_prompt_time = train_space_accepted_at = None
        while True:
            now = time.perf_counter()
            ready = training_ready(share, cfg, now)
            if ready and train_ready_time is None:
                train_ready_time = now
            if ready and (headless or (train_prompt_visible and stim._space_pressed)):
                train_space_accepted_at = now
                break
            if share.failed.value or any(not p.is_alive() for p in workers):
                raise RuntimeError("Worker failed during startup; see worker error JSON")
            if not ready and now > deadline:
                raise RuntimeError("Amplifier startup timed out")
            if stim:
                if glfw.window_should_close(stim.window):
                    return_code = 0
                    break
                graphics.glClear(graphics.GL_COLOR_BUFFER_BIT)
                if ready:
                    if not train_prompt_visible:
                        stim._space_pressed = False  # Discard any key pressed during connection/warmup.
                    stim._draw_centered_text_5x7("PRESS SPACE TO START", pixel_size=3.)
                else:
                    stim._space_pressed = False
                glfw.swap_buffers(stim.window)
                if ready and not train_prompt_visible:
                    train_prompt_time = time.perf_counter()
                glfw.poll_events()
                train_prompt_visible = bool(ready)
            time.sleep(.01)
        if return_code is not None:
            return return_code
        share.start_requested.value = True
        if stim:
            stim._space_pressed = False  # Consume the calibration-start key.
        last_stage, previous_swap, previous_segment = None, None, None
        frame = 0
        headless_deadline = time.perf_counter()
        overload_since = None
        while share.is_running.value:
            timing.begin(frame)
            now = time.perf_counter()
            if share.failed.value or any(not p.is_alive() for p in workers):
                raise RuntimeError("Worker failed; see worker error JSON")
            if now-share.last_batch.value > cfg.no_data_timeout_seconds:
                raise RuntimeError("No fresh amplifier data")
            if stim and glfw.window_should_close(stim.window):
                break
            start_test = False
            if stim:
                start_test = stim._space_pressed
                stim._space_pressed = False  # Never latch a key from fitting/warmup.
            elif schedule.stage == WAIT_TEST and schedule.test_ready_time is not None:
                # Software tests have no keyboard; explicitly emulate a NEW key.
                start_test = now-schedule.test_ready_time >= .25
            schedule.advance(now, share, start_test=start_test)
            lag = (share.buff_idx.value-share.read_idx.value)/cfg.fs
            if schedule.stage in (WARMUP, ONLINE) and lag > cfg.max_prediction_lag_seconds:
                overload_since = now if overload_since is None else overload_since
                if now-overload_since > cfg.no_data_timeout_seconds:
                    raise RuntimeError("Decoder cannot sustain realtime processing")
            else:
                overload_since = None
            timing.mark('control_end')
            label, duration = schedule.cue()
            schedule.next_stimulus_frame()
            timing.mark('prediction_read_begin')
            with share.prediction_lock:
                pred = share.predict_label.value if schedule.stage == ONLINE else 0
            timing.mark('prediction_read_end')
            if stim:
                intensities = (stim._phase_intensities_for_frame(schedule.stimulus_frame)
                               if schedule.stimulus_id else [0.]*cfg.task_m)
                draw_frame(stim, graphics, cfg, schedule.stage, label, pred, intensities, timing)
                swap_before = time.perf_counter()
                glfw.swap_buffers(stim.window)
                swapped = time.perf_counter()
                timing.mark('swap_before', swap_before)
                timing.mark('swap_after', swapped)
                glfw.poll_events()
                timing.mark('poll_end')
            else:
                # Exact rational phase rule for integer refresh rate and 10 Hz.
                from fractions import Fraction
                step = Fraction(str(cfg.frequency))/cfg.refresh_rate
                den = step.denominator*cfg.task_m
                intensities = ([float(2*((schedule.stimulus_frame*step.numerator*cfg.task_m+i*step.denominator)%den)<den)
                                for i in range(cfg.task_m)] if schedule.stimulus_id else [0.]*cfg.task_m)
                timing.mark('draw_begin')
                timing.mark('draw_end')
                timing.mark('flush_end')
                swap_before = time.perf_counter()
                headless_deadline += 1/cfg.refresh_rate
                time.sleep(max(0., headless_deadline-time.perf_counter()))
                swapped = time.perf_counter()
                timing.mark('swap_before', swap_before)
                timing.mark('swap_after', swapped)
                timing.mark('poll_end')
            if schedule.cue_start is None and label:
                schedule.cue_start = swapped
            if schedule.stage == ONLINE and schedule.online_start is None:
                schedule.online_start = swapped
                schedule.test_records[schedule.test_index]["online_start"] = swapped
            if schedule.stage == WARMUP and schedule.test_stimulus_start is None:
                schedule.test_stimulus_start = swapped
                schedule.test_records[schedule.test_index]["stimulus_start"] = swapped
            if schedule.stage == WAIT_TEST and last_stage != WAIT_TEST:
                schedule.test_ready_time = swapped
                schedule.test_records[schedule.test_index]["ready_time"] = swapped
                print(f"[Online] Test {schedule.test_index+1}/3: {schedule.method_order[schedule.test_index]} ready. Press SPACE.", flush=True)
            if schedule.pending_finish is not None:
                schedule.test_records[schedule.pending_finish]["online_end"] = swapped
                schedule.pending_finish = None
            if schedule.stage != last_stage:
                schedule.stage_start = swapped
                print(f"[Stage] {STAGES[schedule.stage]}", flush=True)
            timing.mark('state_begin')
            present_state(share, cfg, schedule, frame, intensities, swapped)
            timing.mark('state_end')
            if schedule.stage == DRAIN and last_stage != DRAIN:
                with share.ring_lock:
                    share.close_idx.value = share.buff_idx.value
                    share.drained.value = False
            if schedule.stage != last_stage or schedule.segment != previous_segment:
                events.append(dict(time=swapped, stage=STAGES[schedule.stage], segment=schedule.segment,
                                   cue=label, duration_requested=duration, sample_write=share.buff_idx.value,
                                   stimulus_id=schedule.stimulus_id, stimulus_frame=schedule.stimulus_frame,
                                   test_id=schedule.current_test_id(), method=schedule.current_method()))
            timing.mark('events_end')
            frames.append(dict(frame=frame, swap_before=swap_before, swap_after=swapped,
                interval=None if previous_swap is None else swapped-previous_swap,
                stage=STAGES[schedule.stage], segment=schedule.segment, cue=label,
                test_id=schedule.current_test_id(), method=schedule.current_method(),
                stimulus_id=schedule.stimulus_id, stimulus_frame=schedule.stimulus_frame,
                displayed_prediction=pred if cfg.feedback else 0,
                intensity=json.dumps(intensities), decoder_backlog_samples=share.buff_idx.value-share.read_idx.value))
            timing.finish()
            previous_swap, last_stage, previous_segment = swapped, schedule.stage, schedule.segment
            frame += 1
            if schedule.stage == END:
                completed = True
                break
        if share.failed.value:
            raise RuntimeError("Worker failed; inspect error JSON")
    except (Exception, KeyboardInterrupt) as exc:
        error = str(exc) or "Interrupted"
        print(f"[Online] {error}", flush=True)
    finally:
        timing.detach()
        if stim:
            try:
                graphics.glClear(graphics.GL_COLOR_BUFFER_BIT)
                glfw.swap_buffers(stim.window)
            except Exception:
                pass
        online_end = time.perf_counter()
        if (schedule.test_index < len(schedule.test_records)
                and schedule.test_records[schedule.test_index]["online_start"] is not None
                and schedule.test_records[schedule.test_index]["online_end"] is None):
            schedule.test_records[schedule.test_index]["online_end"] = online_end
        with share.snapshot_lock:
            share.stage.value = END
            share.start_predict.value = False
            share.is_training.value = False
            share.stimulus_id.value = 0
            share.stimulus_frame.value = -1
            share.test_id.value = 0
            share.method_index.value = -1
            share.intensities[:] = [0.]*SharedState.MAX_TARGETS
            share.stimulus_intensity.value = 0.
        share.is_running.value = False
        for worker in workers:
            if worker.pid is None:
                continue
            worker.join(timeout=cfg.fit_timeout_seconds+5 if worker.name == "online-decoder" else 20)
            if worker.is_alive():
                worker.terminate()
                worker.join(3)
                error = (error or "") + f" {worker.name} required termination; outputs may be incomplete"
            elif worker.exitcode != 0:
                error = (error or "") + f" {worker.name} exited {worker.exitcode}"
        if share.failed.value and error is None:
            error = "Worker failed; inspect error JSON"
        if stim:
            glfw.destroy_window(stim.window)
            glfw.terminate()
        prediction_file = Path(output)/"predictions.json"
        predictions = json.loads(prediction_file.read_text(encoding="utf-8")) if prediction_file.exists() else []
        tests = []
        for record in schedule.test_records:
            item = dict(record)
            start, stop = item["online_start"], item["online_end"]
            published = [(p["publication_time"], p["prediction"]) for p in predictions
                         if p["publication_time"] is not None and p["test_id"] == item["test_id"]]
            cues = [(e["time"], e["cue"]) for e in events
                    if e["stage"] == "online" and e["test_id"] == item["test_id"]]
            item["prediction_count"] = len(published)
            item["test_start_input"] = "automatic_synthetic_test" if headless else "space_key"
            item["warmup_actual_seconds"] = (start-item["stimulus_start"]
                if start is not None and item["stimulus_start"] is not None else None)
            item["completed"] = stop is not None and item["stop_reason"] is not None
            item.update(time_accuracy(cues, published, start or online_end, stop or online_end,
                                      cfg.online_mode == "cued"))
            tests.append(item)
        intervals = [f["interval"] for f in frames if f["interval"] is not None]
        training_segments_presented = sum(e["stage"] == "training" for e in events)
        summary = dict(completed=completed and error is None, error=error, simulated=simulate,
                       headless=headless, physical_display_timing_verified=False,
                       model_frozen=bool(share.frozen.value), training_blocks_scheduled=schedule.blocks,
                       training_blocks_used=(training_segments_presented+cfg.task_m-1)//cfg.task_m,
                       training_segments_presented=training_segments_presented,
                       train_ready_time=train_ready_time, train_prompt_time=train_prompt_time,
                       train_space_accepted_at=train_space_accepted_at,
                       test_seed=schedule.test_seed, test_method_order=schedule.method_order,
                       tests=tests, online_stop_mode=cfg.online_stop_mode,
                       metadata_columns=metadata_columns(cfg.task_m), stage_codes=dict(enumerate(STAGES)),
                       accuracy_basis="held decoder output at host publication time; cue at software swap return",
                       prediction_count=sum(t["prediction_count"] for t in tests), config=cfg.to_dict(),
                       frame_interval_p95_ms=float(np.percentile(intervals, 95)*1000) if intervals else None,
                       long_frame_intervals=sum(t>1.5/cfg.refresh_rate for t in intervals))
        write_json(Path(output)/"summary.json", summary)
        write_json(Path(output)/"events.json", events)
        write_csv(Path(output)/"frames.csv", frames)
        timing.save(output, frames, cfg.refresh_rate, headless=headless)
        print(f"[Online] Results: {output}\n" + json.dumps([
            dict(method=t["method"], time_accuracy=t["time_accuracy"]) for t in tests]), flush=True)
    return 1 if error else 0


def draw_prediction(stim, graphics, label):
    x, y = stim.positions[label-1]
    width, base_y = stim.square_side*.28, y+stim.square_side*.78
    graphics.glColor3f(0., 1., 0.)
    graphics.glBegin(graphics.GL_TRIANGLES)
    graphics.glVertex2f(x-width/2, base_y)
    graphics.glVertex2f(x+width/2, base_y)
    graphics.glVertex2f(x, base_y-stim.square_side*.22)
    graphics.glEnd()


def draw_frame(stim, graphics, cfg, stage, label, prediction, intensities, timing=None):
    if timing is not None:
        timing.mark('draw_begin')
    graphics.glClear(graphics.GL_COLOR_BUFFER_BIT)
    graphics.glLoadIdentity()
    if stage in FLICKER_STAGES:
        for position, intensity in zip(stim.positions, intensities):
            stim._draw_target_square(position, intensity)
        if label:
            stim._draw_cue(label)
        if stage == ONLINE and cfg.feedback and prediction:
            draw_prediction(stim, graphics, prediction)
    elif stage == WAIT_TEST:
        stim._draw_centered_text_5x7("PRESS SPACE TO START", pixel_size=3.)
    if timing is not None:
        timing.mark('draw_end')
    graphics.glFlush()
    if timing is not None:
        timing.mark('flush_end')


def run_replay(cfg, output, eeg_path, state_path, train_segments):
    """Fast sequential MAT replay; no claim to reconstruct host/physical latency."""
    eeg_mat, state_mat = loadmat(eeg_path), loadmat(state_path)
    raw, states = eeg_mat["data"], state_mat["states"]
    if int(eeg_mat["Fs"].item()) != cfg.fs or int(state_mat["Fs"].item()) != cfg.fs:
        raise ValueError("Replay sampling rate mismatch")
    if len(raw) != len(states) or states.shape[1] < 14+cfg.task_m:
        raise ValueError("Replay EEG/States shape mismatch")
    if not np.all(states[:, 1] == cfg.task_m):
        raise ValueError("Replay condition M does not match --task-m")
    labels = states[:, 2].astype(int)
    if not np.all((labels >= 1) & (labels <= cfg.task_m)):
        raise ValueError("Replay requires an original continuous cued recording (labels 1..M)")
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels))+1, len(raw)]
    if not 0 < train_segments < len(boundaries)-1:
        raise ValueError("Need whole training cue segments followed by held-out segments")
    split = int(boundaries[train_segments])
    metadata = np.zeros((len(raw), len(metadata_columns(cfg.task_m))))
    metadata[:, LABEL] = labels
    metadata[:, VALID] = 1
    metadata[:, INTENSITY:INTENSITY+cfg.task_m] = states[:, 14:14+cfg.task_m]
    # Original MAT runs have continuous stimulation; never invent a restart.
    metadata[:, INTENSITY+cfg.task_m] = 1
    metadata[:, INTENSITY+cfg.task_m+1] = states[:, 0]
    metadata[:, INTENSITY+cfg.task_m+2] = 1
    metadata[:split, STAGE], metadata[split:, STAGE] = TRAIN, ONLINE
    for segment, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        metadata[start:stop, SEGMENT] = segment
        valid_start = min(stop, start+int(np.ceil(cfg.skip_seconds*cfg.fs)))
        if segment < train_segments:
            metadata[valid_start:stop, TRAIN_VALID] = 1
    stream, bank, predictions = EpochStream(cfg), EdgeModelBank(cfg), []
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        # Split blocks at train/test boundary; no fit on any held-out sample.
        for start in range(0, split, 10):
            stop = min(start+10, split)
            for e in stream.feed(raw[start:stop], metadata[start:stop]):
                if e.training:
                    bank.collect(e)
        bank.fit()
        joblib.dump(dict(config=cfg.to_dict(), model=bank), Path(output)/"frozen_model.joblib")
        for start in range(split, len(raw), 10):
            stop = min(start+10, len(raw))
            for e in stream.feed(raw[start:stop], metadata[start:stop]):
                if e.online and e.start >= split:
                    label, scores = bank.predict(e.edge, e.x)
                    predictions.append(dict(edge=e.edge+1, edge_sample=e.edge_sample, start=e.start,
                                            stop=e.stop, prediction=label+1, label_at_edge=e.label+1,
                                            scores=scores.tolist()))
    write_json(Path(output)/"replay_predictions.json", predictions)
    write_json(Path(output)/"training.json", dict(counts=bank.counts.tolist(), windows=bank.training_index))
    summary = dict(mode="fast_sequential_replay", config=cfg.to_dict(), eeg=str(eeg_path), states=str(state_path),
                   train_segments=train_segments, split_sample=split, samples=len(raw),
                   frozen=bank.frozen, predictions=len(predictions), runtime_seconds=time.perf_counter()-started,
                   diagnostic_window_accuracy=float(np.mean([p["prediction"] == p["label_at_edge"] for p in predictions]))
                   if predictions else None,
                   online_time_accuracy=None,
                   limitation="Overlapping-window diagnostic only; not realtime accuracy or physical latency")
    write_json(Path(output)/"summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration; see online_config.json")
    parser.add_argument("--method", choices=METHODS, help="Single-model MAT replay only; live run tests all three")
    parser.add_argument("--task-m", type=int, choices=(2, 4))
    parser.add_argument("--online-mode", choices=("cued", "uncued"))
    parser.add_argument("--online-seconds", type=float)
    parser.add_argument("--online-stop-mode", choices=("time", "segments"))
    parser.add_argument("--online-cue-count", type=int)
    parser.add_argument("--training-blocks", type=int)
    parser.add_argument("--train-cue-seconds", nargs=2, type=float)
    parser.add_argument("--skip-seconds", type=float)
    parser.add_argument("--test-warmup-seconds", type=float)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--timing-diagnostics", action=argparse.BooleanOptionalAction, default=None,
                        help="Main-thread timing, GC and window events (default: enabled)")
    parser.add_argument("--device-name", help="SDK exact amplifier name (not the COM port)")
    parser.add_argument("--simulate", action="store_true", help="Synthetic EEG, no amplifier; not a performance validation")
    parser.add_argument("--headless", action="store_true", help="Synthetic test only; no display")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay-eeg", type=Path)
    parser.add_argument("--replay-states", type=Path)
    parser.add_argument("--replay-train-segments", type=int, default=10)
    args = parser.parse_args(argv)
    cfg = Config(**json.loads(args.config.read_text(encoding="utf-8-sig"))) if args.config else Config()
    for key in ("method", "task_m", "online_mode", "online_seconds", "online_stop_mode",
                "online_cue_count", "training_blocks",
                "train_cue_seconds", "skip_seconds", "test_warmup_seconds", "min_samples", "timing_diagnostics"):
        if getattr(args, key) is not None:
            setattr(cfg, key, getattr(args, key))
    cfg.validate()
    if args.method is not None and not args.replay_eeg:
        parser.error("--method applies only to MAT replay; a live run tests all three methods")
    if bool(args.replay_eeg) != bool(args.replay_states):
        parser.error("Specify both --replay-eeg and --replay-states")
    if args.headless and not args.simulate:
        parser.error("--headless requires --simulate")
    output = args.output or ROOT/"online_results"/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_"+uuid4().hex[:6])
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"config.json", cfg.to_dict())
    if args.replay_eeg:
        return run_replay(cfg, output, args.replay_eeg, args.replay_states, args.replay_train_segments)
    return run_live(cfg, output, args.simulate, args.headless, args.device_name)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
