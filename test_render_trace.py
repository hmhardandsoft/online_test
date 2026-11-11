import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from render_trace import RenderTraceContext, WindowsMarker
from export_render_trace_windows import trace_windows


class FakeMarker:
    frequency = 10_000_000
    error = None
    def __init__(self): self.messages = []; self.closed = False
    def ticks(self): return 123456789
    def enabled(self): return True
    def emit(self, text): self.messages.append(text); return 0
    def close(self): self.closed = True


class TraceTests(unittest.TestCase):
    def test_trace_time_mapping_includes_marker_uncertainty(self):
        context = dict(pid=123, main_thread_native_id=456, anchors=[dict(
            name='workers_started', provider_enabled=True, event_status=0,
            emit_before=100., emit_after=100.002, emit_bracket_ms=2.)])
        diagnostics = dict(long_intervals=[dict(frame=12, stage='training',
            interval_ms=50., largest_component='draw_commands', start=105., stop=105.05)])
        row = trace_windows(context, diagnostics, marker_seconds=20.)[0]
        self.assertAlmostEqual(row['start_seconds_low'], 24.998)
        self.assertAlmostEqual(row['start_seconds_high'], 25.)
        self.assertAlmostEqual(row['stop_seconds_high'], 25.05)
        with self.assertRaisesRegex(ValueError, 'named run_stop'):
            trace_windows(context, diagnostics, marker_name='run_stop')
        context['anchors'][0]['event_status'] = 5
        with self.assertRaisesRegex(ValueError, 'Start WPR'):
            trace_windows(context, diagnostics)
        context['anchors'][0]['event_status'] = 0
        context['anchors'][0]['provider_enabled'] = False
        with self.assertRaisesRegex(ValueError, 'Start WPR'):
            trace_windows(context, diagnostics)

    def test_anchors_identity_uncertainty_and_save(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = RenderTraceContext(folder, marker_factory=FakeMarker)
            ctx.workers([type('Worker', (), dict(name='recorder', pid=123))()])
            ctx.anchor('workers_started')
            ctx.save()
            ctx.close()
            data = json.loads((Path(folder)/'render_trace_context.json').read_text())
            self.assertEqual(data['pid'], os.getpid())
            self.assertEqual(data['workers'], [dict(name='recorder', pid=123)])
            self.assertEqual(len(data['anchors']), 2)
            self.assertTrue(ctx.marker.closed)
            for anchor in data['anchors']:
                self.assertLessEqual(anchor['emit_before'], anchor['emit_after'])
                self.assertLessEqual(anchor['qpc_perf_before'], anchor['qpc_perf_after'])
                self.assertTrue(anchor['provider_enabled'])
                self.assertEqual(anchor['event_status'], 0)
            self.assertIn('workers_started', ctx.marker.messages[0])

    def test_disabled_has_no_markers_or_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('render_trace.WindowsMarker', side_effect=AssertionError):
                ctx = RenderTraceContext(folder, enabled=False,
                    marker_factory=lambda: self.fail('must not initialize ETW'))
                ctx.anchor('unused'); ctx.save(); ctx.close()
            self.assertFalse(list(Path(folder).iterdir()))

    def test_failure_does_not_stop_experiment(self):
        with tempfile.TemporaryDirectory() as folder:
            ctx = RenderTraceContext(folder, marker_factory=FakeMarker)
            ctx.marker.emit = lambda _: (_ for _ in ()).throw(OSError('test failure'))
            ctx.anchor('broken')
            ctx.close()
            self.assertEqual(len(ctx.data['errors']), 2)
            self.assertTrue(ctx.marker.closed)
            self.assertTrue((Path(folder)/'render_trace_context.json').exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows API smoke test')
    def test_real_windows_provider_without_starting_capture(self):
        marker = WindowsMarker()
        try:
            self.assertIsNone(marker.error)
            self.assertGreater(marker.frequency, 0)
            self.assertGreater(marker.ticks(), 0)
            self.assertEqual(marker.emit('SSVEP_RENDER unit test'), 0)
        finally:
            marker.close()


if __name__ == '__main__':
    unittest.main()
