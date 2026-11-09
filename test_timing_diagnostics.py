import gc
import json
import tempfile
import time
import unittest
from pathlib import Path

from timing_diagnostics import COLUMNS, TimingDiagnostics, interval_parts


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
                for name in COLUMNS[2:-2]:
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


if __name__=='__main__':
    unittest.main()
