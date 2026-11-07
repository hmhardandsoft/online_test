#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""USB amplifier acquisition benchmark; optional rendering, no EEG decoding.

From D:/experiment/0710:
  .venv/Scripts/python.exe usb_acquisition_benchmark.py
  .venv/Scripts/python.exe usb_acquisition_benchmark.py --seconds 10 --repeat 1
  .venv/Scripts/python.exe usb_acquisition_benchmark.py --simulate --seconds 2 --repeat 1
  .venv/Scripts/python.exe usb_acquisition_benchmark.py --stimulus --refresh-rate 120

The three modes share a timestamp probe AFTER the installed SDK's parser:
  getdata_row:    original SDK get_data(), per-sample shared-buffer writes
  getdata_block:  original SDK get_data(), block shared-buffer writes
  callback_block: SDK callback, timestamped local thread queue, block writes

No SDK files, hardware packet size, sample rate settings beyond --fs, stimulus
code, or decoder code are changed. Probe attachment is temporary and in memory.
--stimulus runs rendering in a separate process. Swap timestamps do not prove
that its window was visible on the participant's screen or when pixels lit up.
Reported latency starts after parsing, NOT at hardware acquisition or display.
SDK drop counts count error events, NOT the number of lost EEG samples.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from dataclasses import dataclass
from datetime import datetime
import importlib.metadata
import json
import multiprocessing as mp
from pathlib import Path
import queue
import random
import threading
import time

import numpy as np


MODES = ("getdata_row", "getdata_block", "callback_block")
MODE_DESCRIPTIONS = {
    "getdata_row": "SDK get_data + per-sample writes (original recorder style)",
    "getdata_block": "SDK get_data + block writes",
    "callback_block": "SDK callback + local queue + block writes",
}


@dataclass(frozen=True)
class ChunkStamp:
    first_sample: int
    samples: int
    ready_ns: int

    @property
    def end_sample(self):
        return self.first_sample + self.samples


class ParserProbe:
    """Keep raw payload untouched; timestamp the same SDK location in all modes."""

    def __init__(self, recorder):
        self.recorder = recorder
        self.parser = getattr(recorder, "_iRecorder__parser", None)
        if self.parser is None or not callable(getattr(self.parser, "parse_data", None)):
            raise RuntimeError("Unsupported eConEXG layout: cannot attach the common parser probe")
        self.original = self.parser.parse_data
        self.pending = deque()
        self.lock = threading.Lock()
        self.next_sample = 0
        self.measure_start_ns = 0
        self.measure_end_ns = 0
        self.measured_samples = 0
        self.last_measured_end = 0
        self.last_ready_ns = time.perf_counter_ns()
        self.max_pending_chunks = 0

    def attach(self):
        self.parser.parse_data = self.parse

    def restore(self):
        self.parser.parse_data = self.original

    def parse(self, raw):
        result = self.original(raw)
        if result:
            with self.lock:
                now = time.perf_counter_ns()
                stamp = ChunkStamp(self.next_sample, len(result), now)
                self.next_sample = stamp.end_sample
                self.last_ready_ns = now
                self.pending.append(stamp)
                self.max_pending_chunks = max(self.max_pending_chunks, len(self.pending))
                if self.measure_start_ns <= now < self.measure_end_ns:
                    self.measured_samples += stamp.samples
                    self.last_measured_end = stamp.end_sample
        return result

    def take(self, samples):
        result = []
        remaining = samples
        with self.lock:
            while remaining:
                if not self.pending:
                    raise RuntimeError("SDK payload has no matching timestamp metadata")
                stamp = self.pending.popleft()
                if stamp.samples > remaining:
                    raise RuntimeError("SDK split a parser chunk unexpectedly; benchmark invalid")
                result.append(stamp)
                remaining -= stamp.samples
        return result


class CallbackReceiver:
    def __init__(self, probe, capacity=2048):
        self.probe = probe
        # Thread queue inside the recorder process, NOT an mp.Queue for EEG.
        self.pending = queue.Queue(maxsize=capacity)
        self.error = None
        self.max_depth = 0

    def __call__(self, array):
        try:
            stamps = self.probe.take(len(array))
            # This SDK creates a fresh ndarray before invoking the callback.
            self.pending.put_nowait((array, stamps))
            self.max_depth = max(self.max_depth, self.pending.qsize())
        except Exception as exc:
            # Do not block the SDK receiver or hide a lost chunk.
            self.error = f"Callback queue/metadata error: {exc}"


class SharedRing:
    """Reproduce the recorder's raw+state row width with a static dummy state."""

    def __init__(self, fs, channels, seconds=30):
        self.channels = channels
        self.state_width = 23  # SharedState.STATE_COLUMNS in the current project.
        self.width = channels + self.state_width
        self.length = max(1, round(fs * seconds))
        self.storage = mp.Array("f", self.length * self.width)
        self.data = np.frombuffer(self.storage.get_obj(), dtype=np.float32).reshape(
            self.length, self.width
        )
        self.index = mp.Value("q", 0)
        self.state = np.zeros(self.state_width, dtype=np.float32)
        self.state[1] = 4

    def write(self, eeg, block):
        n = len(eeg)
        if n > self.length:
            raise RuntimeError("A single input batch exceeds the diagnostic ring capacity")
        if not block:
            states = np.repeat(self.state[None, :], n, axis=0)
            for row in range(n):
                offset = self.index.value % self.length
                self.data[offset, :self.channels] = eeg[row]
                self.data[offset, self.channels:] = states[row]
                self.index.value += 1
            return
        start = self.index.value
        offset = start % self.length
        first = min(n, self.length - offset)
        self.data[offset:offset + first, :self.channels] = eeg[:first]
        self.data[offset:offset + first, self.channels:] = self.state
        if first < n:
            self.data[:n - first, :self.channels] = eeg[first:]
            self.data[:n - first, self.channels:] = self.state
        # Publish only after the complete batch has been written.
        self.index.value = start + n


class SimulatedParser:
    def parse_data(self, samples):
        return samples


class SimulatedRecorder:
    """Hardware-free exercise of SDK queue merging and callback branching."""

    def __init__(self, fs, channels):
        self.fs, self.channels = fs, channels
        self._iRecorder__parser = SimulatedParser()
        self.pending = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = None
        self.callback = None
        self.status = "IDLE"

    def set_update_functions(self, callback):
        self.callback = callback

    def start_acquisition_data(self, with_q=True):
        self.stop_event.clear()
        self.status = "SIGNAL"

        def produce():
            sample = 0
            start = time.perf_counter()
            count = 10
            while not self.stop_event.wait(max(0, start + (sample + count) / self.fs - time.perf_counter())):
                array = np.repeat(np.arange(sample, sample + count)[:, None], self.channels, axis=1)
                result = self._iRecorder__parser.parse_data(array.tolist())
                if with_q:
                    self.pending.put(result)
                else:
                    self.callback(np.array(result))
                sample += count
            self.status = "IDLE"

        self.thread = threading.Thread(target=produce, daemon=True)
        self.thread.start()

    def get_data(self, timeout=0.02):
        try:
            data = self.pending.get(timeout=timeout)
        except queue.Empty:
            return []
        while not self.pending.empty():
            data.extend(self.pending.get())
        return data

    def get_dev_status(self):
        return self.status

    def get_packet_drop_times(self):
        return 0

    def stop_acquisition(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def close_dev(self):
        self.stop_acquisition()
        self.status = "TERMINATE"


def bounded_call(function, timeout, description):
    """SDK start/close methods contain unbounded waits; bound our caller's wait."""
    result = queue.Queue()

    def invoke():
        try:
            result.put((True, function()))
        except BaseException as exc:
            result.put((False, exc))

    threading.Thread(target=invoke, daemon=True).start()
    try:
        success, value = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError(f"Timed out during {description}; aborting this benchmark") from None
    if not success:
        raise value
    return value


def percentiles(values):
    if not values:
        return dict(count=0, mean=None, p50=None, p95=None, p99=None, maximum=None)
    array = np.asarray(values, dtype=float)
    return dict(count=len(values), mean=float(array.mean()),
                p50=float(np.percentile(array, 50)), p95=float(np.percentile(array, 95)),
                p99=float(np.percentile(array, 99)), maximum=float(array.max()))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def connect_real(args):
    from eConEXG import iRecorder
    recorder = iRecorder(dev_type=args.device_type)
    try:
        recorder.set_frequency(args.fs)
        recorder.find_devs()
        until = time.perf_counter() + args.connect_timeout
        seen = []
        while time.perf_counter() < until:
            seen.extend(name for name in recorder.get_devs() if name not in seen)
            matches = [name for name in seen if args.device_name is None or name == args.device_name]
            if matches:
                if len(matches) > 1 and args.device_name is None:
                    raise RuntimeError(f"Multiple amplifiers found {matches}; use --device-name")
                name = matches[0]
                print(f"Connecting {name} ({args.device_type}, {args.fs} Hz)", flush=True)
                bounded_call(lambda: recorder.connect_device(name), args.connect_timeout, "device connection")
                info = recorder.get_dev_info()
                if info.get("fs") != args.fs:
                    raise RuntimeError(f"SDK selected Fs={info.get('fs')}, requested {args.fs}")
                return recorder
            time.sleep(0.05)
        raise RuntimeError(f"No matching USB amplifier found. Devices seen: {seen}")
    except BaseException:
        try:
            bounded_call(recorder.close_dev, 8, "connection cleanup")
            finish = getattr(recorder, "_iRecorder__finish_search", None)
            if finish:
                bounded_call(finish, 3, "search cleanup")
        except Exception:
            pass
        raise


def stimulus_worker(stop_event, ready_event, output, expected_rate, task_m, frequency):
    """Call original rendering methods in a separate process, with no decoder."""
    import glfw
    import SSVEP_stimulator as graphics
    from util import SharedState

    swaps = []
    stim = None
    error = None
    actual_rate = None
    graphics_info = {}
    try:
        share = SharedState(online=False, task_m=task_m, frequency=frequency, block_count=3)
        stim = graphics.SSVEPStimulator(share, required_refresh_rate=expected_rate)
        from OpenGL.GL import glGetString, GL_VENDOR, GL_RENDERER, GL_VERSION
        for name, token in (("vendor", GL_VENDOR), ("renderer", GL_RENDERER), ("version", GL_VERSION)):
            value = glGetString(token)
            graphics_info[name] = value.decode(errors="replace") if value else None
        try:
            from OpenGL.WGL.EXT.swap_control import wglGetSwapIntervalEXT
            graphics_info["reported_swap_interval"] = int(wglGetSwapIntervalEXT())
        except Exception:
            graphics_info["reported_swap_interval"] = None
        actual_rate = stim.refresh_rate
        if actual_rate != expected_rate:
            raise RuntimeError(f"Display is {actual_rate} Hz, expected {expected_rate} Hz")
        cue_idx = 0
        cue_start = 0
        frame = 0
        while not stop_event.is_set() and not glfw.window_should_close(stim.window):
            cue = stim.cue_schedule[cue_idx]
            if frame - cue_start >= cue["duration_frames"]:
                cue_idx = (cue_idx + 1) % len(stim.cue_schedule)
                cue_start = frame
                cue = stim.cue_schedule[cue_idx]
            stim._render_frame(frame, cue, (frame - cue_start) / actual_rate)
            glfw.swap_buffers(stim.window)
            swaps.append(time.perf_counter_ns())
            glfw.poll_events()
            frame += 1
            if frame >= actual_rate:
                ready_event.set()
        if not stop_event.is_set():
            error = "Stimulus window closed or Esc pressed"
            stop_event.set()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        stop_event.set()
    finally:
        if stim is not None:
            try:
                graphics.glClear(graphics.GL_COLOR_BUFFER_BIT)
                glfw.swap_buffers(stim.window)
                glfw.destroy_window(stim.window)
            except Exception:
                pass
        glfw.terminate()
        np.save(Path(output) / "stimulus_swap_perf_ns.npy", np.asarray(swaps, dtype=np.int64))
        (Path(output) / "stimulus.json").write_text(json.dumps(dict(
            expected_refresh_rate=expected_rate, actual_refresh_rate=actual_rate,
            task_m=task_m, stimulus_frequency_hz=frequency, swap_count=len(swaps), error=error,
            graphics_info=graphics_info,
        ), indent=2), encoding="utf-8")


def add_stimulus_metrics(output, runs, refresh_rate):
    path = output / "stimulus_swap_perf_ns.npy"
    if not path.exists():
        return
    times = np.load(path)
    period_ms = 1000 / refresh_rate
    for run in runs:
        start, end = run.get("measurement_start_perf_ns"), run.get("measurement_end_perf_ns")
        if start is None or end is None:
            continue
        selected = times[(times >= start) & (times < end)]
        intervals = np.diff(selected) / 1e6
        long_count = int(np.sum(intervals > 1.5 * period_ms))
        swap_rate = (float(1e9 * (len(selected) - 1) / (selected[-1] - selected[0]))
                     if len(selected) > 1 else None)
        rate_matches = bool(swap_rate is not None and abs(swap_rate / refresh_rate - 1) <= 0.05)
        run["stimulus"] = dict(
            swap_intervals=len(intervals), frame_interval_ms=percentiles(intervals.tolist()),
            software_swap_rate_hz=swap_rate,
            software_rate_within_5pct_of_refresh=rate_matches,
            long_frame_intervals=long_count,
            long_frame_fraction=long_count / len(intervals) if len(intervals) else None,
            short_frame_intervals=int(np.sum(intervals < 0.5 * period_ms)),
            timing_note="Swap return times are not physical presentation timestamps; do not interpret rate as monitor Hz.",
            estimated_missed_refreshes=(int(np.maximum(0, np.rint(intervals / period_ms) - 1).sum())
                                       if rate_matches else None),
        )
        (output / (run["run_id"] + ".json")).write_text(json.dumps(run, indent=2), encoding="utf-8")


def run_mode(recorder, args, mode, run_id, channels, stop_event=None):
    probe = ParserProbe(recorder)
    callback = CallbackReceiver(probe)
    ring = SharedRing(args.fs, channels)
    chunk_rows, batch_rows, saved_eeg = [], [], []
    baseline_drops = None
    peak_drops = 0
    last_end = 0
    consumed_measured = 0
    empty_polls = 0
    status = "ok"
    error = None
    start_ns = end_ns = None
    probe.attach()
    recorder.set_update_functions(callback if mode == "callback_block" else None)
    cpu_start = time.process_time()
    try:
        bounded_call(lambda: recorder.start_acquisition_data(with_q=mode != "callback_block"),
                     args.connect_timeout, "start acquisition")
        acquisition_started_ns = time.perf_counter_ns()
        start_ns = acquisition_started_ns + round(args.warmup * 1e9)
        end_ns = start_ns + round(args.seconds * 1e9)
        with probe.lock:
            probe.measure_start_ns, probe.measure_end_ns = start_ns, end_ns
        next_stall_ns = start_ns + round(args.stall_every * 1e9)
        while True:
            if stop_event is not None and stop_event.is_set():
                raise KeyboardInterrupt
            now = time.perf_counter_ns()
            if now >= start_ns and baseline_drops is None:
                baseline_drops = recorder.get_packet_drop_times()
            if baseline_drops is not None:
                peak_drops = max(peak_drops, recorder.get_packet_drop_times() - baseline_drops)
            if callback.error:
                raise RuntimeError(callback.error)
            if recorder.get_dev_status() != "SIGNAL":
                raise RuntimeError(f"Acquisition stopped: {recorder.get_dev_status()}")
            if now - max(probe.last_ready_ns, acquisition_started_ns) > args.no_data_timeout * 1e9:
                raise RuntimeError("No parsed data received within --no-data-timeout")
            if now >= end_ns:
                with probe.lock:
                    target_end = probe.last_measured_end
                if last_end >= target_end:
                    break
                if now > end_ns + round(args.no_data_timeout * 1e9):
                    raise RuntimeError("Unable to drain the measured data before the deadline")
            if args.stall_ms and start_ns <= now < end_ns and now >= next_stall_ns:
                time.sleep(args.stall_ms / 1000)
                next_stall_ns = time.perf_counter_ns() + round(args.stall_every * 1e9)

            call_start = time.perf_counter_ns()
            if mode == "callback_block":
                try:
                    payload, stamps = callback.pending.get(timeout=0.02)
                except queue.Empty:
                    empty_polls += 1
                    continue
            else:
                payload = recorder.get_data(timeout=0.02)
                if not payload:
                    empty_polls += 1
                    continue
                stamps = None
            returned_ns = time.perf_counter_ns()
            if stamps is None:
                stamps = probe.take(len(payload))
            if stamps[0].first_sample != last_end:
                raise RuntimeError("Application data chunks are missing, duplicated, or out of order")
            last_end = stamps[-1].end_sample
            raw = np.asarray(payload, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != channels:
                raise RuntimeError(f"Expected {channels} columns (EEG+trigger), received {raw.shape}")
            if args.simulate:
                expected = np.arange(stamps[0].first_sample, last_end, dtype=np.float32)
                if not np.array_equal(raw[:, 0], expected):
                    raise RuntimeError("Simulation payload lost or reordered samples")
            write_start = time.perf_counter_ns()
            ring.write(raw, block=mode != "getdata_row")
            published_ns = time.perf_counter_ns()
            measured = [stamp for stamp in stamps if start_ns <= stamp.ready_ns < end_ns]
            if measured:
                batch_rows.append(dict(
                    return_time_s=(returned_ns - start_ns) / 1e9,
                    samples=len(raw), source_chunks=len(stamps),
                    get_call_ms=(returned_ns - call_start) / 1e6,
                    service_ms=(published_ns - returned_ns) / 1e6,
                    ring_write_ms=(published_ns - write_start) / 1e6,
                    software_pending_samples=max(0, probe.next_sample - last_end),
                ))
                for stamp in measured:
                    consumed_measured += stamp.samples
                    chunk_rows.append(dict(
                        first_sample=stamp.first_sample, samples=stamp.samples,
                        ready_time_s=(stamp.ready_ns - start_ns) / 1e9,
                        postparse_to_processing_ms=(returned_ns - stamp.ready_ns) / 1e6,
                        postparse_to_publish_ms=(published_ns - stamp.ready_ns) / 1e6,
                    ))
                if args.save_eeg:
                    first = measured[0].first_sample - stamps[0].first_sample
                    last = measured[-1].end_sample - stamps[0].first_sample
                    saved_eeg.append(raw[first:last].copy())
    except KeyboardInterrupt:
        status, error = "interrupted", "Stopped by Ctrl+C"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        capture_end = time.perf_counter_ns()
        cpu_seconds = time.process_time() - cpu_start
        try:
            if baseline_drops is not None:
                peak_drops = max(peak_drops, recorder.get_packet_drop_times() - baseline_drops)
            bounded_call(recorder.stop_acquisition, 8, "stop acquisition")
            stop_deadline = time.perf_counter() + 8
            while recorder.get_dev_status() not in ("IDLE", "TERMINATE"):
                if time.perf_counter() >= stop_deadline:
                    raise TimeoutError("SDK did not return to IDLE after stopping")
                time.sleep(0.02)
        except Exception as exc:
            status = "failed" if status == "ok" else status
            error = f"{error or ''} Cleanup: {exc}".strip()
        probe.restore()
        recorder.set_update_functions(None)

    elapsed = max(0, min(capture_end, end_ns or capture_end) - (start_ns or capture_end)) / 1e9
    measured_received = probe.measured_samples
    if status == "ok" and (consumed_measured != measured_received or measured_received == 0):
        status, error = "failed", "Measured data were not completely consumed"
    intervals = np.diff([row["ready_time_s"] for row in chunk_rows]) * 1000
    summary = dict(
        run_id=run_id, mode=mode, status=status, error=error,
        simulated=args.simulate, measurement_seconds=elapsed,
        measurement_start_perf_ns=start_ns, measurement_end_perf_ns=min(capture_end, end_ns or capture_end),
        parsed_samples=measured_received, consumed_samples=consumed_measured,
        parsed_samples_per_second=measured_received / elapsed if elapsed else None,
        sdk_drop_events=peak_drops, sdk_drop_events_before_measurement=baseline_drops,
        application_data_complete=measured_received > 0 and consumed_measured == measured_received,
        cpu_seconds_including_warmup_and_drain=cpu_seconds,
        source_chunk_samples=percentiles([row["samples"] for row in chunk_rows]),
        source_ready_interval_ms=percentiles(intervals.tolist()),
        application_batch_samples=percentiles([row["samples"] for row in batch_rows]),
        merged_source_chunks=percentiles([row["source_chunks"] for row in batch_rows]),
        postparse_to_processing_ms=percentiles([row["postparse_to_processing_ms"] for row in chunk_rows]),
        postparse_to_publish_ms=percentiles([row["postparse_to_publish_ms"] for row in chunk_rows]),
        application_service_ms=percentiles([row["service_ms"] for row in batch_rows]),
        ring_write_ms=percentiles([row["ring_write_ms"] for row in batch_rows]),
        software_pending_samples=percentiles([row["software_pending_samples"] for row in batch_rows]),
        empty_polls=empty_polls,
    )
    prefix = args.output / run_id
    write_csv(prefix.with_name(prefix.name + "_chunks.csv"), chunk_rows)
    write_csv(prefix.with_name(prefix.name + "_batches.csv"), batch_rows)
    if saved_eeg:
        np.save(prefix.with_name(prefix.name + "_EEG.npy"), np.concatenate(saved_eeg))
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_report(output, args, runs, sdk_version):
    config = vars(args).copy()
    config["output"] = str(args.output)
    payload = dict(config=config, sdk_version=sdk_version, modes=MODE_DESCRIPTIONS, runs=runs)
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for run in runs:
        rows.append(dict(
            run_id=run["run_id"], mode=run["mode"], status=run["status"],
            samples_per_second=run["parsed_samples_per_second"],
            sdk_drop_events=run["sdk_drop_events"],
            application_data_complete=run["application_data_complete"],
            queue_p95_ms=run["postparse_to_processing_ms"]["p95"],
            publish_p95_ms=run["postparse_to_publish_ms"]["p95"],
            publish_max_ms=run["postparse_to_publish_ms"]["maximum"],
            service_p95_ms=run["application_service_ms"]["p95"],
            application_batch_p95_samples=run["application_batch_samples"]["p95"],
            source_chunks_per_batch_p95=run["merged_source_chunks"]["p95"],
            sdk_drop_events_before_measurement=run.get("sdk_drop_events_before_measurement"),
            stimulus_software_swaps_per_second=run.get("stimulus", {}).get("software_swap_rate_hz"),
            stimulus_long_intervals=run.get("stimulus", {}).get("long_frame_intervals"),
            stimulus_frame_p95_ms=run.get("stimulus", {}).get("frame_interval_ms", {}).get("p95"),
        ))
    write_csv(output / "summary.csv", rows)

    def number(value):
        return "N/A" if value is None else f"{value:.3f}"

    lines = [
        "# USB 空采诊断报告", "",
        f"数据来源：{'模拟器（不是硬件实测）' if args.simulate else 'USB 放大器'}；eConEXG={sdk_version}。",
        "",
        "时间起点是 SDK 解析完成；不包含设备采样、USB/驱动缓存或解析之前的延迟。",
        "所有模式都有相同的解析时间探针，因此是带诊断开销的比较，不是未插桩代码的绝对性能。",
        ("独立进程调用原刺激绘制代码，未运行解码器；未验证窗口在用户屏幕可见，swap 返回时间不能证明物理呈现时刻。"
         if args.stimulus else "没有运行刺激或解码器，不能据此判断 600 Hz 掉帧、分类准确率或屏幕同步精度。"),
        "源批次指一次解析返回；应用批次指一次 get_data 返回或回调出队。回调使用进程内线程队列。",
        "SDK drop events 是校验/序号异常事件次数，不等于丢失样本数。",
        "", "| 运行 | 状态 | 样本/s | SDK丢包事件 | 排队P95 ms | 发布P95 ms | 处理P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['run_id']} | {row['status']} | {number(row['samples_per_second'])} | "
                     f"{row['sdk_drop_events']} | {number(row['queue_p95_ms'])} | "
                     f"{number(row['publish_p95_ms'])} | {number(row['service_p95_ms'])} |")
    if args.stimulus:
        lines += ["", "swap 返回频率不等于物理屏幕刷新率；若明显超过设定刷新率，不能据此声称刺激逐帧同步。",
                  "", "| 运行 | swap返回次数/秒 | swap间隔P95 ms | 超过1.5帧周期的间隔数 |",
                  "|---|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['run_id']} | {number(row['stimulus_software_swaps_per_second'])} | "
                         f"{number(row['stimulus_frame_p95_ms'])} | {row['stimulus_long_intervals']} |")
    lines += ["", "## 如何判读", "",
              "1. 先排除 failed/interrupted、SDK 丢包事件增加或应用数据不完整的运行。",
              "2. 同一模式比较多次重复；先比较 getdata_row 与 getdata_block，再比较 callback_block。",
              "3. 排队时间是解析完成到应用开始处理；发布时间是解析完成到共享缓冲区写入完成。",
              "4. 批次变大/合并数增加并伴随排队时间上升，支持软件积压的判断；不等于硬件延迟变大。",
              "5. 处理耗时按应用批次统计，批次大小不同时不要只比较这一列；结合发布延迟和源批次明细。",
              "6. --stall-ms 是人为阻塞消费者的压力测试，应与不注入阻塞的结果分开比较。",
              "7. 短记录的样本率可能有批次边界波动；所有延迟分位数按源批次统计。",
              "8. sdk_drop_events_before_measurement 单独记录预热结束前的 SDK 异常事件；正式段丢包列不包含这些事件。",
              "", "所有输出在计时段结束后写盘；--save-eeg 会增加测量段内的复制开销，三个模式一致启用。",
              "共同探针使用当前 SDK 的私有 parser 属性，但不会修改 site-packages 文件。", ""]
    for run in runs:
        if run["error"]:
            lines.append(f"- {run['run_id']}: {run['error']}")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def self_test():
    a, b = SharedRing(10, 3, seconds=1), SharedRing(10, 3, seconds=1)
    for start, size in ((0, 7), (7, 6), (13, 10), (23, 1)):
        x = np.arange(start * 3, (start + size) * 3, dtype=np.float32).reshape(size, 3)
        a.write(x, block=False)
        b.write(x, block=True)
        assert a.index.value == b.index.value
        np.testing.assert_array_equal(a.data, b.data)
    recorder = SimulatedRecorder(2000, 9)
    probe = ParserProbe(recorder)
    probe.attach()
    recorder._iRecorder__parser.parse_data([[1] * 9] * 10)
    recorder._iRecorder__parser.parse_data([[2] * 9] * 20)
    stamps = probe.take(30)
    assert [(s.first_sample, s.samples) for s in stamps] == [(0, 10), (10, 20)]
    assert not probe.pending
    probe.restore()
    # A full callback queue must invalidate the run, never silently drop data.
    overflow_probe = ParserProbe(recorder)
    receiver = CallbackReceiver(overflow_probe, capacity=1)
    overflow_probe.parse([[1] * 9] * 10)
    receiver(np.ones((10, 9)))
    overflow_probe.parse([[2] * 9] * 10)
    receiver(np.ones((10, 9)))
    assert receiver.error is not None
    print("Self-test passed: wraparound writes and merged-chunk metadata.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=20, help="Measured seconds per mode (default 20)")
    parser.add_argument("--repeat", type=int, default=2, help="Repeat all requested modes (default 2)")
    parser.add_argument("--warmup", type=float, default=2, help="Warmup seconds before each measurement")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--fs", type=int, default=2000)
    parser.add_argument("--device-type", choices=("USB8", "USB16", "USB32"), default="USB8")
    parser.add_argument("--device-name", help="Exact eConEXG device name, e.g. iRe8-...")
    parser.add_argument("--connect-timeout", type=float, default=15)
    parser.add_argument("--no-data-timeout", type=float, default=3)
    parser.add_argument("--seed", type=int, default=20261103,
                        help="Seed first mode order; later repeats rotate the order")
    parser.add_argument("--stall-ms", type=float, default=0, help="Optional simulated consumer pause")
    parser.add_argument("--stall-every", type=float, default=1, help="Seconds between injected pauses")
    parser.add_argument("--save-eeg", action="store_true", help="Save measured raw EEG+trigger as NPY")
    parser.add_argument("--simulate", action="store_true", help="Exercise software with synthetic input, no hardware")
    parser.add_argument("--stimulus", action="store_true",
                        help="Render with the original stimulator in a separate process; window visibility requires direct observation")
    parser.add_argument("--refresh-rate", type=int, default=120)
    parser.add_argument("--stimulus-start-timeout", type=float, default=60,
                        help="Allow cold OpenGL/window initialization before starting acquisition")
    parser.add_argument("--task-m", type=int, choices=(4, 9), default=4)
    parser.add_argument("--stim-frequency", type=float, default=10.0)
    parser.add_argument("--self-test", action="store_true", help="Check buffer and timestamp logic; no hardware")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if min(args.seconds, args.fs, args.repeat, args.connect_timeout, args.no_data_timeout, args.stall_every,
           args.refresh_rate, args.stim_frequency, args.stimulus_start_timeout) <= 0:
        parser.error("Durations, sample rate and repeat count must be positive")
    if args.warmup < 0 or args.stall_ms < 0:
        parser.error("Warmup and stall must be nonnegative")
    if args.output is None:
        args.output = Path(__file__).resolve().parent / "usb_benchmark_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    channels = int(args.device_type[3:]) + 1
    rng = random.Random(args.seed)
    initial_order = list(dict.fromkeys(args.modes))
    rng.shuffle(initial_order)
    runs = []
    recorder = None
    stimulus_process = None
    stop_event = None
    exit_code = 0
    try:
        sdk_version = importlib.metadata.version("econexg")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "unavailable"
    print("SIMULATION (no amplifier)" if args.simulate else "USB acquisition benchmark: no decoder", flush=True)
    print(f"Output: {args.output}", flush=True)
    try:
        if args.stimulus:
            stop_event, ready_event = mp.Event(), mp.Event()
            stimulus_process = mp.Process(target=stimulus_worker, args=(
                stop_event, ready_event, str(args.output), args.refresh_rate, args.task_m, args.stim_frequency,
            ))
            stimulus_process.start()
            until = time.perf_counter() + args.stimulus_start_timeout
            while not ready_event.wait(timeout=0.1):
                if stop_event.is_set() or not stimulus_process.is_alive() or time.perf_counter() > until:
                    raise RuntimeError("Stimulus startup failed; see stimulus.json")
        if not args.simulate:
            recorder = connect_real(args)
        for repeat in range(1, args.repeat + 1):
            shift = (repeat - 1) % len(initial_order)
            order = initial_order[shift:] + initial_order[:shift]
            for index, mode in enumerate(order, 1):
                if args.simulate:
                    recorder = SimulatedRecorder(args.fs, channels)
                run_id = f"r{repeat:02d}_{index}_{mode}"
                print(f"[{run_id}] warmup {args.warmup:g}s + measure {args.seconds:g}s", flush=True)
                result = run_mode(recorder, args, mode, run_id, channels, stop_event)
                runs.append(result)
                write_report(args.output, args, runs, sdk_version)
                p95 = result["postparse_to_publish_ms"]["p95"]
                print(f"  status={result['status']}, samples={result['consumed_samples']}, "
                      f"drop_events={result['sdk_drop_events']}, publish_p95_ms={p95}", flush=True)
                if result["status"] != "ok":
                    print(result["error"], flush=True)
                    return 130 if result["status"] == "interrupted" else 1
    except KeyboardInterrupt:
        exit_code = 130
        print("Stopped by Ctrl+C", flush=True)
    except Exception as exc:
        exit_code = 1
        (args.output / "startup_error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        print(f"ERROR: {exc}", flush=True)
    finally:
        if recorder is not None:
            try:
                bounded_call(recorder.close_dev, 10, "close device")
            except Exception as exc:
                print(f"Cleanup error: {exc}", flush=True)
                exit_code = 1
        if stimulus_process is not None:
            stop_event.set()
            stimulus_process.join(timeout=8)
            if stimulus_process.is_alive():
                stimulus_process.terminate()
                stimulus_process.join(timeout=3)
                print("Stimulus process did not exit normally", flush=True)
                exit_code = 1
            add_stimulus_metrics(args.output, runs, args.refresh_rate)
        write_report(args.output, args, runs, sdk_version)
    print(f"Report: {args.output / 'report.md'}", flush=True)
    return exit_code


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
