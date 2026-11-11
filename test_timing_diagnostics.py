import gc
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from timing_diagnostics import BASE_COLUMNS, COLUMNS, INDEX, TimingDiagnostics, interval_parts, draw_details


class TimingTests(unittest.TestCase):
    def test_partition_includes_previous_frame_and_recovers_interval(self):
        p = dict(zip(COLUMNS, range(len(COLUMNS))))
        c = dict(zip(COLUMNS, range(20, 20+len(COLUMNS))))
        parts = interval_parts(p, c)
        self.assertEqual(sum(parts.values()), (c['swap_after']-p['swap_after'])*1000)
        # A pause after swap(n-1) must be attributed to interval ending at n.
        p['poll_end'] += .7
        parts = interval_parts(p, c)
        self.assertAlmostEqual(parts['previous_poll_events'], 1700.)
        self.assertEqual(sum(parts.values()), (c['swap_after']-p['swap_after'])*1000)

    def test_gc_window_callback_chain_restore_and_export(self):
        calls=[]
        class GLFW:
            FOCUSED=1; ICONIFIED=2
            def __init__(self):
                self.callbacks={k:None for k in ('focus','iconify','refresh','framebuffer_size')}
                self.original=lambda *args:calls.append(args)
                self.callbacks['focus']=self.original
            def get_window_attrib(self,w,k):return int(k==self.FOCUSED)
            def __getattr__(self,name):
                key=name.removeprefix('set_window_').removeprefix('set_').removesuffix('_callback')
                if key not in self.callbacks:raise AttributeError(name)
                def setter(w, callback):
                    old=self.callbacks[key];self.callbacks[key]=callback;return old
                return setter
        fake=GLFW();d=TimingDiagnostics(True,chunk_frames=2)
        original_callbacks=list(gc.callbacks)
        try:
            d.attach(fake, 'window')
            frames=[]
            for frame in range(3):
                d.begin(frame)
                for name in BASE_COLUMNS[2:-2]:
                    d.mark(name)
                    if frame==0 and name=='swap_after':
                        fake.callbacks['focus']('window',0)
                        gc.collect(2)
                        time.sleep(.004)
                d.finish()
                frames.append(dict(frame=frame,stage='training'))
        finally:
            d.detach()
        self.assertEqual(gc.callbacks, original_callbacks)
        self.assertIs(fake.callbacks['focus'],fake.original)
        self.assertEqual(calls,[('window',0)])
        self.assertTrue(d.gc_events)
        with tempfile.TemporaryDirectory() as path:
            d.save(path,frames,600)
            result=json.loads((Path(path)/'timing_diagnostics_summary.json').read_text())
            found=[x for x in result['long_intervals'] if x['frame']==1]
            self.assertEqual(len(found),1)
            self.assertEqual(found[0]['largest_component'],'previous_poll_events')
            self.assertTrue(found[0]['overlapping_gc'])
            self.assertTrue(found[0]['nearby_window_events'])
            self.assertAlmostEqual(sum(found[0]['components_ms'].values()),found[0]['interval_ms'])
            self.assertEqual(result['frame_rows'],3)
            self.assertEqual(result['complete_frame_rows'],3)
            self.assertIsNone(found[0]['draw_cpu_ms'])

    def test_disabled_and_incomplete_frame(self):
        d=TimingDiagnostics(False)
        old=list(gc.callbacks)
        d.attach();d.begin(0);d.mark('swap_after');d.finish();d.detach()
        self.assertEqual(d.count,0);self.assertEqual(d.chunks,[]);self.assertEqual(gc.callbacks,old)
        with tempfile.TemporaryDirectory() as path:
            d.save(path,[],600)
            self.assertEqual(list(Path(path).iterdir()),[])
            d=TimingDiagnostics(True);d.begin(0);d.mark('control_end')
            d.save(path,[],600)
            result=json.loads((Path(path)/'timing_diagnostics_summary.json').read_text())
            self.assertEqual(result['long_interval_count'],0)

    def test_draw_steps_locate_injected_target_pause_and_cpu_span(self):
        from online_experiment import draw_frame
        from online_core import Config, TRAIN, WAIT_TEST, FIT, ONLINE
        clock = SimpleNamespace(wall=10., cpu=1.)
        calls = []
        def call(name, duration=.00001):
            calls.append(name)
            clock.wall += duration
            clock.cpu += .000001
        graphics = SimpleNamespace(GL_COLOR_BUFFER_BIT=1, GL_TRIANGLES=2,
            glClear=lambda *a: call('clear'), glLoadIdentity=lambda: call('identity'),
            glFlush=lambda: call('flush'), glColor3f=lambda *a: call('color'),
            glBegin=lambda *a: call('begin'), glVertex2f=lambda *a: call('vertex'),
            glEnd=lambda: call('end'))
        for m, stage in ((2, TRAIN), (4, TRAIN), (4, ONLINE), (4, WAIT_TEST), (4, FIT)):
            stim = SimpleNamespace(positions=[(i, 0) for i in range(m)], square_side=1,
                _draw_target_square=lambda p, i: call(f'target{p[0]+1}', .057 if p[0] == 1 else .00001),
                _draw_cue=lambda label: call('cue'),
                _draw_centered_text_5x7=lambda *a, **k: call('prompt'))
            cfg = Config(task_m=m)
            d = TimingDiagnostics(True)
            with patch('timing_diagnostics.time.perf_counter', lambda: clock.wall), \
                 patch('timing_diagnostics.time.thread_time', lambda: clock.cpu):
                d.begin(0)
                calls.clear()
                draw_frame(stim, graphics, cfg, stage, 1, 1, [1]*m, d)
                observed = list(calls)
                details = draw_details(dict(zip(COLUMNS, d.row)))
                calls.clear()
                draw_frame(stim, graphics, cfg, stage, 1, 1, [1]*m)
                self.assertEqual(calls, observed)  # Instrumentation preserves draw commands/order.
            self.assertIsNotNone(d.draw_thread_id)
            self.assertAlmostEqual(sum(details['draw_components_ms'].values()), details['draw_wall_ms'])
            # Every pre-flush fake call uses 1 us CPU, including the simulated 57 ms wait.
            self.assertAlmostEqual(details['draw_cpu_ms'], (len(observed)-1)*.001, places=8)
            if stage in (TRAIN, ONLINE):
                self.assertEqual(details['largest_draw_component'], 'target_2')
                self.assertAlmostEqual(details['draw_components_ms']['target_2'], 57.)
                self.assertLess(details['draw_cpu_fraction'], .01)
            else:
                self.assertFalse(any(key.startswith('target_') for key in details['draw_components_ms']))
            if m == 2:
                self.assertTrue(np.isnan(d.row[INDEX['target_3_end']]))
            if stage == WAIT_TEST:
                self.assertIn('prompt', details['draw_components_ms'])
            if stage == ONLINE:
                self.assertIn('feedback', details['draw_components_ms'])

    def test_optional_fields_do_not_hide_anomaly_and_incomplete_frame(self):
        d = TimingDiagnostics(True, chunk_frames=2)
        for frame in range(3):
            d.begin(frame)
            for name in BASE_COLUMNS[1:]:
                d.mark(name, frame*.010+BASE_COLUMNS.index(name)*.00001)
            if frame == 1:
                d.mark('clear_end', d.row[INDEX['draw_begin']]+.000003)
                d.mark('identity_end', d.row[INDEX['draw_end']]-.000002)
                d.mark('draw_cpu_start', 1.)
                d.mark('draw_cpu_end', 1.000001)
        d.begin(3)  # Window close after begin(): no corresponding presented frame.
        with tempfile.TemporaryDirectory() as path:
            frames = [dict(frame=i, stage='training', test_id=0) for i in range(3)]
            d.save(path, frames, 600)
            result = json.loads((Path(path)/'timing_diagnostics_summary.json').read_text())
            self.assertEqual(result['schema_version'], 2)
            self.assertEqual(result['complete_frame_rows'], 3)
            self.assertEqual(result['incomplete_frame_rows'], 1)
            self.assertEqual(result['long_interval_count'], 2)
            by_frame = {a['frame']: a for a in result['long_intervals']}
            self.assertAlmostEqual(by_frame[1]['draw_cpu_ms'], .001)
            self.assertAlmostEqual(sum(by_frame[1]['draw_components_ms'].values()),
                                   by_frame[1]['components_ms']['draw_commands'])
            self.assertIsNone(by_frame[2]['draw_cpu_ms'])
            self.assertIn('resolution', result['clock_info']['thread_time'])

    def test_disabled_draw_probes_do_not_read_clocks(self):
        d = TimingDiagnostics(False)
        with patch('timing_diagnostics.time.perf_counter', side_effect=AssertionError('wall clock read')), \
             patch('timing_diagnostics.time.thread_time', side_effect=AssertionError('CPU clock read')):
            d.begin(0)
            d.begin_draw()
            d.mark('clear_end')
            d.end_draw()
        self.assertEqual(d.count, 0)


if __name__=='__main__':
    unittest.main()
