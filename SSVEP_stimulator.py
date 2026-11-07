#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Continuous phase-coded SSVEP stimulator.

All targets flicker continuously at the same frequency. A red cue rectangle
selects the attended target for 3-5 s; cue changes do not reset stimulus phase.
"""

import math
import random
import sys
import time
from fractions import Fraction

import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_MODELVIEW,
    GL_PROJECTION,
    GL_QUADS,
    GL_TRIANGLES,
    GL_TRUE,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glEnd,
    glFlush,
    glLoadIdentity,
    glMatrixMode,
    glOrtho,
    glVertex2f,
    glViewport,
)

from SSVEP_decoder import SSVEPdecoderProcess
from SSVEP_recorder import EEGRecordingProcess
from util import SharedState


class SSVEPStimulator:
    """GLFW + PyOpenGL continuous SSVEP stimulator."""

    _DEFAULT_SQUARE_SCALE_BY_M = {2: 0.22, 4: 0.22, 9: 0.22}
    _DEFAULT_GAP_SCALE_BY_M = {2: 0.45, 4: 0.45, 9: 0.45}
    _FONT_5X7 = {
        "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
        "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
        "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
        "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
        "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
        "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
        "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
        "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    }

    def __init__(
        self,
        shared_state: SharedState,
        required_refresh_rate: int = 600,
        square_scale_by_m=None,
        gap_scale_by_m=None,
    ):
        self.share = shared_state
        self.required_refresh_rate = int(required_refresh_rate)
        self.task_m = int(self.share.task_m.value)
        self.paradigm_mode = self.share.paradigm_mode
        self.passive_m2 = self.paradigm_mode == "passive"
        self.online = bool(self.share.online.value)
        self.stim_freq = float(self.share.frequency.value)
        self.square_scale_by_m = dict(self._DEFAULT_SQUARE_SCALE_BY_M)
        self.gap_scale_by_m = dict(self._DEFAULT_GAP_SCALE_BY_M)
        if square_scale_by_m is not None:
            self.square_scale_by_m.update(square_scale_by_m)
        if gap_scale_by_m is not None:
            self.gap_scale_by_m.update(gap_scale_by_m)
        self._space_pressed = False

        if not glfw.init():
            sys.exit("GLFW init failed")

        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.refresh_rate = int(mode.refresh_rate)
        width, height = mode.size.width, mode.size.height

        glfw.window_hint(glfw.DOUBLEBUFFER, GL_TRUE)
        glfw.window_hint(glfw.SAMPLES, 0)
        glfw.window_hint(glfw.SRGB_CAPABLE, GL_TRUE)
        glfw.window_hint(glfw.REFRESH_RATE, self.refresh_rate)

        print(f"[Stim] Display: {width}x{height} @ {self.refresh_rate} Hz", flush=True)
        if self.refresh_rate != self.required_refresh_rate:
            print(
                f"[Stim] WARNING: expected {self.required_refresh_rate} Hz, "
                f"detected {self.refresh_rate} Hz. Continuing.",
                flush=True,
            )

        self.window = glfw.create_window(width, height, "SSVEP Phase Coding", monitor, None)
        if not self.window:
            glfw.terminate()
            sys.exit("Window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
        glfw.set_key_callback(self.window, self._key_callback)

        self.screen_width = width
        self.screen_height = height
        square_scale = self._scale_for_task(self.square_scale_by_m, "square")
        gap_scale = self._scale_for_task(self.gap_scale_by_m, "gap")
        self.square_side = min(width, height) * square_scale
        self.gap = self.square_side * gap_scale
        self.positions = self._make_positions()
        self._warn_if_layout_too_large()
        self.phases = [2.0 * math.pi * i / self.task_m for i in range(self.task_m)]
        cycles_per_frame = (
            Fraction(str(self.stim_freq)).limit_denominator(1_000_000)
            / self.refresh_rate
        )
        self.phase_step_num = cycles_per_frame.numerator
        self.phase_step_den = cycles_per_frame.denominator
        self.phase_denominator = self.phase_step_den * self.task_m
        self.cue_schedule = self._make_cue_schedule()
        self.total_frames = sum(item["duration_frames"] for item in self.cue_schedule)

        glClearColor(0.0, 0.0, 0.0, 1.0)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-width / 2, width / 2, -height / 2, height / 2, -1.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glClear(GL_COLOR_BUFFER_BIT)
        glfw.swap_buffers(self.window)
        glfw.poll_events()

        period_frames = self.refresh_rate / self.stim_freq
        print(
            f"[Stim] Task M={self.task_m}, mode={self.paradigm_mode}, "
            f"f={self.stim_freq:.3f} Hz, "
            f"period={period_frames:.3f} frames, blocks={self.share.block_count.value}, "
            f"cues={len(self.cue_schedule)}, square={self.square_side:.1f}px, gap={self.gap:.1f}px",
            flush=True,
        )

        nominal_period = self.refresh_rate / self.stim_freq
        rounded_period = int(round(nominal_period))
        if abs(nominal_period - rounded_period) < 1e-9:
            if rounded_period % self.task_m != 0:
                print(
                    f"[Stim] WARNING: {rounded_period} frames/cycle is not divisible "
                    f"by M={self.task_m}; phase spacing will be frame-quantized.",
                    flush=True,
                )

        self.recording_process = None
        self.decoder_process = None
        if self.online:
            print("[Stim] Starting recorder...", flush=True)
            self.recording_process = EEGRecordingProcess(self.share)
            self.recording_process.start()
            while not self.share.recorder_ready.value and not glfw.window_should_close(self.window):
                time.sleep(0.05)
                glfw.poll_events()
            print("[Stim] Recorder ready.", flush=True)

            print("[Stim] Starting decoder...", flush=True)
            self.decoder_process = SSVEPdecoderProcess(self.share)
            self.decoder_process.start()
            print("[Stim] Decoder process started.", flush=True)
        else:
            print("[Stim] Offline preview mode: recorder/decoder disabled.", flush=True)

    def _scale_for_task(self, values_by_m, name):
        value = float(values_by_m.get(self.task_m, values_by_m.get("default", 0.0)))
        if value <= 0.0:
            raise ValueError(f"{name} scale must be positive for task_m={self.task_m}")
        return value

    def _warn_if_layout_too_large(self):
        if not self.positions:
            return

        half = self.square_side / 2.0
        min_x = min(x - half for x, _ in self.positions)
        max_x = max(x + half for x, _ in self.positions)
        min_y = min(y - half for _, y in self.positions)
        max_y = max(y + half for _, y in self.positions)
        width_margin = self.screen_width * 0.5 - max(abs(min_x), abs(max_x))
        height_margin = self.screen_height * 0.5 - max(abs(min_y), abs(max_y))
        cue_margin = self.square_side * 0.35

        if width_margin < 0 or height_margin < cue_margin:
            print(                                                                                                                                                                                                          
                "[Stim] WARNING: stimulus layout is close to or beyond the screen bounds. "
                f"width_margin={width_margin:.1f}px, height_margin={height_margin:.1f}px",
                flush=True,
            )

    def _make_positions(self):
        s = self.square_side
        step = s + self.gap
        if self.task_m == 2:
            return [(-step / 2.0, 0.0), (step / 2.0, 0.0)]
        if self.task_m == 4:
            return [
                (-step / 2.0, step / 2.0),
                (step / 2.0, step / 2.0),
                (-step / 2.0, -step / 2.0),
                (step / 2.0, -step / 2.0),
            ]
        return [
            (x * step, y * step)
            for y in (1, 0, -1)
            for x in (-1, 0, 1)
        ]

    def _make_cue_schedule(self):
        if self.passive_m2:
            return self._make_passive_m2_schedule()

        schedule = []
        block_count = int(self.share.block_count.value)
        d_min = float(self.share.cue_duration_min.value)
        d_max = float(self.share.cue_duration_max.value)
        for block_idx in range(block_count):
            labels = list(range(1, self.task_m + 1))
            random.shuffle(labels)
            if schedule and labels[0] == schedule[-1]["label"]:
                for idx, label in enumerate(labels[1:], start=1):
                    if label != schedule[-1]["label"]:
                        labels[0], labels[idx] = labels[idx], labels[0]
                        break
            for label in labels:
                duration_sec = random.uniform(d_min, d_max)
                schedule.append(
                    {
                        "block_idx": block_idx,
                        "label": label,
                        "phase": self.phases[label - 1],
                        "cue_position": label,
                        "phase_labels_by_position": tuple(range(1, self.task_m + 1)),
                        "duration_sec": duration_sec,
                        "duration_frames": max(1, int(round(duration_sec * self.refresh_rate))),
                    }
                )
        return schedule

    def _make_passive_m2_schedule(self):
        schedule = []
        repetitions_per_label = int(self.share.block_count.value)
        d_min = float(self.share.cue_duration_min.value)
        d_max = float(self.share.cue_duration_max.value)
        fixed_cue_position = random.choice((1, 2))
        first_label = random.choice((1, 2))

        for trial_idx in range(repetitions_per_label * 2):
            label = first_label if trial_idx % 2 == 0 else 3 - first_label
            phase_labels_by_position = [0, 0]
            phase_labels_by_position[fixed_cue_position - 1] = label
            phase_labels_by_position[1 - (fixed_cue_position - 1)] = 3 - label
            duration_sec = random.uniform(d_min, d_max)
            schedule.append(
                {
                    "block_idx": trial_idx // 2,
                    "label": label,
                    "phase": self.phases[label - 1],
                    "cue_position": fixed_cue_position,
                    "phase_labels_by_position": tuple(phase_labels_by_position),
                    "duration_sec": duration_sec,
                    "duration_frames": max(1, int(round(duration_sec * self.refresh_rate))),
                }
            )
        return schedule

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            self._space_pressed = True

    @staticmethod
    def _draw_quad(x1, y1, x2, y2):
        glBegin(GL_QUADS)
        glVertex2f(x1, y1)
        glVertex2f(x2, y1)
        glVertex2f(x2, y2)
        glVertex2f(x1, y2)
        glEnd()

    def _draw_text_5x7(self, text, x, y, pixel_size):
        cursor_x = float(x)
        pixel_size = float(pixel_size)
        gap = pixel_size
        for char in text.upper():
            if char == " ":
                cursor_x += pixel_size * 4.0
                continue

            glyph = self._FONT_5X7.get(char)
            if glyph is None:
                cursor_x += pixel_size * 6.0
                continue

            for row_idx, row in enumerate(glyph):
                for col_idx, val in enumerate(row):
                    if val != "1":
                        continue
                    x1 = cursor_x + col_idx * pixel_size
                    y1 = y - row_idx * pixel_size
                    self._draw_quad(x1, y1, x1 + pixel_size, y1 - pixel_size)
            cursor_x += pixel_size * 5.0 + gap

    def _draw_centered_text_5x7(self, text, center_y=0.0, pixel_size=12.0):
        text_width = 0.0
        for char in text.upper():
            text_width += pixel_size * (4.0 if char == " " else 6.0)
        text_width -= pixel_size
        text_height = pixel_size * 7.0
        x = -text_width / 2.0
        y = center_y + text_height / 2.0
        glColor3f(0.85, 0.85, 0.85)
        self._draw_text_5x7(text, x, y, pixel_size)

    def _draw_target_square(self, center, intensity):
        x, y = center
        h = self.square_side / 2.0
        glColor3f(float(intensity), float(intensity), float(intensity))
        self._draw_quad(x - h, y - h, x + h, y + h)

    def _draw_cue(self, label):
        x, y = self.positions[label - 1]
        h = self.square_side / 2.0
        cue_h = self.square_side * 0.08
        cue_gap = self.square_side * 0.18
        glColor3f(1.0, 0.0, 0.0)
        self._draw_quad(
            x - h,
            y - h - cue_gap - cue_h,
            x + h,
            y - h - cue_gap,
        )

    @staticmethod
    def _position_index_for_phase(cue, phase_label):
        return cue["phase_labels_by_position"].index(phase_label)

    def _draw_feedback(self, cue):
        pred = int(self.share.predict_label.value)
        if not self.online or not self.share.start_predict.value or not 1 <= pred <= self.task_m:
            return

        position_idx = self._position_index_for_phase(cue, pred)
        x, y = self.positions[position_idx]
        h = self.square_side / 2.0
        tri_w = self.square_side * 0.28
        tri_h = self.square_side * 0.22
        base_y = y + h + self.square_side * 0.28
        tip_y = base_y - tri_h
        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_TRIANGLES)
        glVertex2f(x - tri_w / 2.0, base_y)
        glVertex2f(x + tri_w / 2.0, base_y)
        glVertex2f(x, tip_y)
        glEnd()

    def _phase_intensities_for_frame(self, frame_idx):
        """Return deterministic square-wave samples for phases 2*pi*i/M."""
        base_tick = int(frame_idx) * self.phase_step_num * self.task_m
        return [
            1.0
            if 2 * ((base_tick + phase_idx * self.phase_step_den) % self.phase_denominator)
            < self.phase_denominator
            else 0.0
            for phase_idx in range(self.task_m)
        ]

    def _set_shared_state(self, frame_idx, cue, cue_elapsed_sec):
        phase_intensities = self._phase_intensities_for_frame(frame_idx)

        # Decoder references stay phase ordered; only the on-screen mapping swaps.
        display_intensities = [
            phase_intensities[label - 1]
            for label in cue["phase_labels_by_position"]
        ]

        self.share.frame_idx.value = int(frame_idx)
        self.share.true_label.value = int(cue["label"])
        self.share.target_phase.value = float(cue["phase"])
        skip_seconds = float(self.share.training_skip_after_cue_ms.value) / 1000.0
        self.share.is_training.value = cue_elapsed_sec >= skip_seconds
        self.share.cue_elapsed_sec.value = float(cue_elapsed_sec)
        self.share.cue_duration_sec.value = float(cue["duration_sec"])
        self.share.cue_position.value = int(cue["cue_position"])
        self.share.phase_label_at_position_1.value = int(cue["phase_labels_by_position"][0])
        self.share.phase_label_at_position_2.value = int(cue["phase_labels_by_position"][1])
        self.share.stimulus_intensity.value = float(phase_intensities[0])
        for i in range(SharedState.MAX_TARGETS):
            self.share.intensities[i] = (
                float(phase_intensities[i]) if i < self.task_m else 0.0
            )
        return display_intensities

    def _render_frame(self, frame_idx, cue, cue_elapsed_sec):
        display_intensities = self._set_shared_state(frame_idx, cue, cue_elapsed_sec)

        glClear(GL_COLOR_BUFFER_BIT)
        glLoadIdentity()
        for center, intensity in zip(self.positions, display_intensities):
            self._draw_target_square(center, intensity)
        self._draw_cue(cue["cue_position"])
        self._draw_feedback(cue)
        glFlush()

    def _wait_for_space(self):
        print("[Stim] Press SPACE to start; ESC to quit.", flush=True)
        text_pixel = max(2.0, min(self.screen_width, self.screen_height) * 0.012)
        while not glfw.window_should_close(self.window) and not self._space_pressed:
            glClear(GL_COLOR_BUFFER_BIT)
            glLoadIdentity()
            self._draw_centered_text_5x7("PRESS SPACE TO START", pixel_size=3.0)
            glFlush()
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(0.01)

    def run(self):
        self._wait_for_space()
        if glfw.window_should_close(self.window):
            self.share.is_running.value = False
            self._cleanup(0, time.perf_counter(), 0)
            return

        self.share.start_requested.value = True
        frame_idx = 0
        cue_idx = 0
        cue_start_frame = 0
        t_start = time.perf_counter()
        t_prev = t_start
        expected_dt = 1.0 / self.refresh_rate
        dropped_frames = 0

        first_cue = self.cue_schedule[0]
        print(
            f"[Stim] cue 1/{len(self.cue_schedule)}: "
            f"label={first_cue['label']}, duration={first_cue['duration_sec']:.2f}s",
            flush=True,
        )

        try:
            while not glfw.window_should_close(self.window):
                if frame_idx >= self.total_frames or not self.share.is_running.value:
                    break

                cue = self.cue_schedule[cue_idx]
                if frame_idx - cue_start_frame >= cue["duration_frames"]:
                    cue_idx += 1
                    cue_start_frame = frame_idx
                    if cue_idx >= len(self.cue_schedule):
                        break
                    cue = self.cue_schedule[cue_idx]
                    print(
                        f"[Stim] cue {cue_idx + 1}/{len(self.cue_schedule)}: "
                        f"label={cue['label']}, duration={cue['duration_sec']:.2f}s",
                        flush=True,
                    )

                cue_elapsed_sec = (frame_idx - cue_start_frame) / self.refresh_rate
                self._render_frame(frame_idx, cue, cue_elapsed_sec)

                glfw.swap_buffers(self.window)
                glfw.poll_events()

                t_now = time.perf_counter()
                dt = t_now - t_prev
                if frame_idx > 5 and dt > expected_dt * 1.5:
                    dropped_frames += 1
                t_prev = t_now
                frame_idx += 1
        finally:
            self.share.is_running.value = False
            glClear(GL_COLOR_BUFFER_BIT)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
            time.sleep(1.0)
            self._cleanup(frame_idx, t_start, dropped_frames)

    def _cleanup(self, frame_idx, t_start, dropped_frames):
        try:
            glfw.destroy_window(self.window)
        except Exception:
            pass
        glfw.terminate()

        if self.recording_process is not None and self.recording_process.is_alive():
            self.recording_process.join(timeout=10)
        if self.decoder_process is not None and self.decoder_process.is_alive():
            self.decoder_process.join(timeout=10)

        total_time = time.perf_counter() - t_start
        avg_fps = frame_idx / max(total_time, 1e-9)
        drop_rate = dropped_frames / max(frame_idx, 1) * 100
        print("\n========== SSVEP Stimulator Summary ==========")
        print(f"Duration:      {total_time:.2f} s")
        print(f"Total frames:  {frame_idx}")
        print(f"Average FPS:   {avg_fps:.2f}")
        print(f"Dropped frames:{dropped_frames}  ({drop_rate:.3f} %)")
        print("==============================================", flush=True)
