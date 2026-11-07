#!/usr/bin/env python
# -*- coding: utf-8 -*-
import datetime
import multiprocessing as mp
import time

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from util import SharedState

try:
    from eConEXG import iRecorder

    HARDWARE_LIBS_AVAILABLE = True
except Exception as exc:
    iRecorder = None
    HARDWARE_LIBS_AVAILABLE = False
    HARDWARE_IMPORT_ERROR = exc


class EEGRecordingProcess(mp.Process):
    """Record 9-channel amplifier data and save raw EEG and state separately."""

    def __init__(self, shared_state: SharedState):
        super().__init__()
        self.share = shared_state

    def run(self):
        fs = int(self.share.fs.value)
        task_m = int(self.share.task_m.value)
        paradigm_mode = self.share.paradigm_mode
        mode_name = paradigm_mode.capitalize()
        state_cols = SharedState.state_columns(task_m)
        state_width_to_save = len(state_cols)
        buff_len = int(self.share.buff_len)
        buff_width = int(self.share.buff_width)
        buff_data = np.frombuffer(
            self.share.buff_data.get_obj(), dtype=np.float32
        ).reshape((buff_len, buff_width))

        recorder = None
        all_eeg_frames = []
        all_state_frames = []

        try:
            if not HARDWARE_LIBS_AVAILABLE:
                print(f"[Recorder] Hardware library unavailable: {HARDWARE_IMPORT_ERROR}", flush=True)
                self.share.recorder_ready.value = True
                while self.share.is_running.value:
                    time.sleep(0.05)
                return

            recorder = iRecorder(dev_type="USB8")
            recorder.set_frequency(fs)
            recorder.find_devs()

            print("[Recorder] Waiting for amplifier...", flush=True)
            while self.share.is_running.value:
                available_devices = recorder.get_devs()
                if available_devices:
                    break
                time.sleep(0.05)
            if not self.share.is_running.value:
                return

            recorder.connect_device(available_devices[0])
            recorder.start_acquisition_data(with_q=True)
            for _ in tqdm(range(100), desc="Recorder warmup", ncols=80):
                time.sleep(0.03)
            for _ in range(10):
                recorder.get_data(timeout=0.01)

            self.share.recorder_ready.value = True
            print("[Recorder] Ready.", flush=True)

            while self.share.is_running.value:
                try:
                    frames = recorder.get_data(timeout=0.01)
                    if not frames:
                        continue

                    data_frames = np.asarray(frames, dtype=np.float32)
                    if data_frames.ndim != 2:
                        continue
                    if data_frames.shape[1] < SharedState.EEG_CHANNELS:
                        print(
                            f"[Recorder] Ignored frame block with {data_frames.shape[1]} channels.",
                            flush=True,
                        )
                        continue
                    data_frames = data_frames[:, : SharedState.EEG_CHANNELS]

                    state_full = self.share.get_state_vector()
                    state_frames_full = np.repeat(
                        state_full[None, :], data_frames.shape[0], axis=0
                    )

                    for i in range(data_frames.shape[0]):
                        idx = self.share.buff_idx.value % buff_len
                        buff_data[idx, : SharedState.EEG_CHANNELS] = data_frames[i]
                        buff_data[idx, SharedState.EEG_CHANNELS :] = state_frames_full[i]
                        self.share.buff_idx.value += 1

                    if self.share.start_requested.value:
                        all_eeg_frames.append(data_frames.copy())
                        all_state_frames.append(
                            state_frames_full[:, :state_width_to_save].copy()
                        )
                except Exception as exc:
                    print(f"[Recorder] Recording error: {exc}", flush=True)
                    continue
        finally:
            if all_eeg_frames:
                recorded_eeg = np.concatenate(all_eeg_frames, axis=0)
                recorded_states = np.concatenate(all_state_frames, axis=0)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_prefix = f"VEP_Exp2_M{task_m}_{mode_name}"
                eeg_file = f"{file_prefix}_EEG_{timestamp}.mat"
                state_file = f"{file_prefix}_States_{timestamp}.mat"
                savemat(
                    eeg_file,
                    {"data": recorded_eeg, "Fs": fs, "paradigm_mode": paradigm_mode},
                )
                savemat(
                    state_file,
                    {
                        "states": recorded_states,
                        "Fs": fs,
                        "paradigm_mode": paradigm_mode,
                        "state_columns": np.asarray(state_cols, dtype=object),
                    },
                )
                print(f"[Recorder] Saved {eeg_file} and {state_file}", flush=True)
            else:
                print("[Recorder] No session data saved.", flush=True)

            if HARDWARE_LIBS_AVAILABLE and recorder is not None:
                try:
                    recorder.stop_acquisition()
                    recorder.close_dev()
                except Exception as exc:
                    print(f"[Recorder] Cleanup error: {exc}", flush=True)
