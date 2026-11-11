"""Main-process timing probes. No per-frame disk I/O or background worker."""
import csv
import gc
import json
import threading
import time
from pathlib import Path

import numpy as np


BASE_COLUMNS = ('frame', 'loop_start', 'control_end', 'prediction_read_begin',
           'prediction_read_end', 'draw_begin', 'draw_end', 'flush_end',
           'swap_before', 'swap_after', 'poll_end', 'state_begin', 'state_end',
           'events_end', 'frame_log_end', 'diagnostic_end')
DRAW_STEPS = ('clear_end', 'identity_end', 'target_1_end', 'target_2_end',
              'target_3_end', 'target_4_end', 'cue_end', 'feedback_end', 'prompt_end')
# Optional columns are appended: M2, black screens and headless frames have NaNs.
COLUMNS = BASE_COLUMNS + DRAW_STEPS + ('draw_cpu_start', 'draw_cpu_end')
INDEX = {name: i for i, name in enumerate(COLUMNS)}


def interval_parts(previous, current):
    """Partition swap(n-1)..swap(n), including both iterations and residual gaps."""
    p, c = previous, current
    spans = [
        ('previous_poll_events', p['swap_after'], p['poll_end']),
        ('previous_stage_bookkeeping', p['poll_end'], p['state_begin']),
        ('previous_present_state', p['state_begin'], p['state_end']),
        ('previous_event_log', p['state_end'], p['events_end']),
        ('previous_frame_log', p['events_end'], p['frame_log_end']),
        ('previous_diagnostic_store', p['frame_log_end'], p['diagnostic_end']),
        ('between_iterations', p['diagnostic_end'], c['loop_start']),
        ('control_and_schedule', c['loop_start'], c['control_end']),
        ('cue_and_phase_setup', c['control_end'], c['prediction_read_begin']),
        ('prediction_lock_and_read', c['prediction_read_begin'], c['prediction_read_end']),
        ('intensity_setup', c['prediction_read_end'], c['draw_begin']),
        ('draw_commands', c['draw_begin'], c['draw_end']),
        ('gl_flush', c['draw_end'], c['flush_end']),
        ('before_swap_gap', c['flush_end'], c['swap_before']),
        ('swap_call', c['swap_before'], c['swap_after']),
    ]
    return {name: (b-a)*1000 for name, a, b in spans}


def draw_details(current):
    """Host call spans, including Python/probe gaps; not GPU execution timings."""
    start, end = current['draw_begin'], current['draw_end']
    steps = [(name, current.get(name, float('nan'))) for name in DRAW_STEPS]
    steps = [(name, value) for name, value in steps if np.isfinite(value)]
    parts, previous = {}, start
    for name, value in steps:
        parts[name.removesuffix('_end')] = (value-previous)*1000
        previous = value
    if steps:
        parts['draw_tail'] = (end-previous)*1000
    cpu_start = current.get('draw_cpu_start', float('nan'))
    cpu_end = current.get('draw_cpu_end', float('nan'))
    cpu_ms = (cpu_end-cpu_start)*1000 if np.isfinite([cpu_start, cpu_end]).all() else None
    wall_ms = (end-start)*1000
    return dict(draw_components_ms=parts,
                largest_draw_component=max(parts, key=parts.get) if parts else None,
                draw_wall_ms=wall_ms, draw_cpu_ms=cpu_ms,
                draw_cpu_fraction=cpu_ms/wall_ms if cpu_ms is not None and wall_ms > 0 else None)


class TimingDiagnostics:
    def __init__(self, enabled=True, chunk_frames=8192):
        self.enabled = enabled
        self.chunk_frames = chunk_frames
        self.chunks = []
        self.count = 0
        self.frame = -1
        self.row = None
        self.gc_events = []
        self.window_events = []
        self.gc_starts = {}
        self.attached = False
        self.window_callbacks = []
        self._gc_callback = self._on_gc  # Stable identity for remove().
        self.draw_thread_id = None
        self.clock_info = ({name: vars(time.get_clock_info(name))
                            for name in ('perf_counter', 'thread_time')} if enabled else {})

    def begin(self, frame):
        if not self.enabled:
            return
        started = time.perf_counter()
        self.frame = frame
        if self.count % self.chunk_frames == 0:
            self.chunks.append(np.full((self.chunk_frames, len(COLUMNS)), np.nan))
        self.row = self.chunks[-1][self.count % self.chunk_frames]
        self.row[0] = frame
        self.row[1] = started
        self.count += 1

    def mark(self, name, timestamp=None):
        if self.enabled and self.row is not None:
            self.row[INDEX[name]] = time.perf_counter() if timestamp is None else timestamp

    def begin_draw(self):
        if self.enabled and self.row is not None:
            if self.draw_thread_id is None:
                self.draw_thread_id = threading.get_native_id()
            # CPU readings enclose the wall interval and a small amount of probe work.
            self.row[INDEX['draw_cpu_start']] = time.thread_time()
            self.mark('draw_begin')

    def end_draw(self):
        if self.enabled and self.row is not None:
            self.mark('draw_end')
            self.row[INDEX['draw_cpu_end']] = time.thread_time()

    def finish(self):
        self.mark('frame_log_end')
        self.mark('diagnostic_end')

    def _on_gc(self, phase, info):
        t = time.perf_counter()
        key = (threading.get_ident(), info['generation'])
        if phase == 'start':
            self.gc_starts[key] = (t, self.frame)
        elif phase == 'stop':
            start, frame = self.gc_starts.pop(key, (t, self.frame))
            self.gc_events.append(dict(start=start, stop=t, generation=info['generation'],
                frame_at_start=frame, frame_at_stop=self.frame, thread_id=key[0],
                collected=info.get('collected', 0), uncollectable=info.get('uncollectable', 0)))

    def attach(self, glfw=None, window=None):
        if not self.enabled or self.attached:
            return
        gc.callbacks.append(self._gc_callback)
        self.attached = True
        if glfw is None:
            return
        for attr in ('FOCUSED', 'ICONIFIED'):
            self.window_events.append(dict(time=time.perf_counter(), frame=-1,
                kind='initial_'+attr.lower(), values=[glfw.get_window_attrib(window, getattr(glfw, attr))]))
        for kind, setter_name in [('focus', 'set_window_focus_callback'),
                                  ('iconify', 'set_window_iconify_callback'),
                                  ('refresh', 'set_window_refresh_callback'),
                                  ('framebuffer_size', 'set_framebuffer_size_callback')]:
            setter = getattr(glfw, setter_name)
            previous = [None]
            def callback(win, *values, _kind=kind, _previous=previous):
                self.window_events.append(dict(time=time.perf_counter(), frame=self.frame,
                                                kind=_kind, values=list(values)))
                if _previous[0] is not None:
                    _previous[0](win, *values)
            previous[0] = setter(window, callback)
            self.window_callbacks.append((setter, window, previous[0], callback))

    def detach(self):
        if not self.attached:
            return
        if self._gc_callback in gc.callbacks:
            gc.callbacks.remove(self._gc_callback)
        for setter, window, previous, _callback in self.window_callbacks:
            setter(window, previous)
        self.window_callbacks.clear()
        self.attached = False

    def save(self, output, frames, refresh_rate, headless=False):
        if not self.enabled:
            return
        output = Path(output)
        # Serialize only after acquisition/stimulation have stopped.
        with (output/'frame_timing.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            for n in range(self.count):
                row = self.chunks[n//self.chunk_frames][n % self.chunk_frames]
                writer.writerow([int(row[0])]+[float(x) if np.isfinite(x) else '' for x in row[1:]])
        for filename, data in [('gc_events.json', self.gc_events), ('window_events.json', self.window_events)]:
            (output/filename).write_text(json.dumps(data, indent=2), encoding='utf-8')
        indexed = {f['frame']: f for f in frames}
        anomalies = []
        complete_rows = 0
        previous = None
        for n in range(self.count):
            row = self.chunks[n//self.chunk_frames][n % self.chunk_frames]
            if not np.isfinite(row[:len(BASE_COLUMNS)]).all() or int(row[0]) not in indexed:
                previous = None
                continue
            complete_rows += 1
            current = dict(zip(COLUMNS, row))
            if previous is not None:
                gap = current['swap_after']-previous['swap_after']
                if gap > 1.5/refresh_rate:
                    parts = interval_parts(previous, current)
                    a, b = previous['swap_after'], current['swap_after']
                    anomalies.append(dict(frame=int(current['frame']),
                        stage=indexed[int(current['frame'])]['stage'],
                        test_id=indexed[int(current['frame'])].get('test_id', 0),
                        method=indexed[int(current['frame'])].get('method'),
                        previous_test_id=indexed[int(previous['frame'])].get('test_id', 0),
                        previous_method=indexed[int(previous['frame'])].get('method'),
                        start=a, stop=b,
                        interval_ms=gap*1000, largest_component=max(parts, key=parts.get),
                        components_ms=parts,
                        **draw_details(current),
                        overlapping_gc=[i for i,e in enumerate(self.gc_events) if e['start']<b and e['stop']>a],
                        nearby_window_events=[i for i,e in enumerate(self.window_events) if a-.05<=e['time']<=b+.05]))
            previous = current
        result = dict(enabled=True, schema_version=2, headless=headless, clock='time.perf_counter seconds',
            clock_info=self.clock_info, draw_thread_native_id=self.draw_thread_id,
            scope='main process only; elapsed time includes waiting and OS scheduling',
            threshold_ms=1500/refresh_rate, frame_rows=self.count, presented_frames=len(frames),
            complete_frame_rows=complete_rows, incomplete_frame_rows=self.count-complete_rows,
            numeric_buffer_bytes=sum(c.nbytes for c in self.chunks),
            gc_event_count=len(self.gc_events), window_event_count=len(self.window_events),
            long_interval_count=len(anomalies),
            long_interval_count_by_test={str(test_id): sum(
                a['test_id'] == test_id or a['previous_test_id'] == test_id for a in anomalies)
                for test_id in sorted({f.get('test_id', 0) for f in frames}) if test_id},
            long_intervals=sorted(anomalies, key=lambda x:x['interval_ms'], reverse=True),
            note=('Draw components subdivide draw_commands; do not add both totals. '
                  'CPU time covers draw_begin..draw_end plus probe overhead, excluding glFlush/swap. '
                  'Clock precision is platform dependent; short CPU deltas may be zero. '
                  'Low CPU use does not distinguish driver blocking from OS descheduling. '
                  'Headless frames have no detailed GL/CPU measurements; swap includes synthetic sleep. '
                  'GC/window overlap indicates coincidence, not proof of cause.'))
        (output/'timing_diagnostics_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
