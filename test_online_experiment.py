"""Meaningful software checks; no amplifier or stimulus window is opened."""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[key] = "1"
import ast
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
import scipy
from scipy.signal import butter, iirnotch, sosfilt, tf2sos
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from threadpoolctl import threadpool_limits
from online_core import *
from online_experiment import OnlineState, publish_rows, read_rows, Schedule, draw_frame


class OnlineTests(unittest.TestCase):
    def test_causal_filter_matches_offline_and_packet_partition(self):
        cfg = Config()
        raw = np.random.default_rng(1).normal(size=(2003, 9))
        indices, whole = CausalFilterBank(cfg).process(raw)
        stream, chunks, kept = CausalFilterBank(cfg), [], []
        for a, b in zip((0, 7, 93, 499, 1201), (7, 93, 499, 1201, 2003)):
            ix, data = stream.process(raw[a:b])
            chunks.append(data)
            kept.extend(ix)
        np.testing.assert_array_equal(indices, kept)
        np.testing.assert_allclose(whole, np.concatenate(chunks, axis=2), atol=1e-12)
        x = raw[:, cfg.eeg_channels]-raw[:, cfg.reference_channels].mean(axis=1, keepdims=True)
        x = sosfilt(tf2sos(*iirnotch(50, 30, fs=2000)), x, axis=0)
        expected = np.stack([sosfilt(butter(4, band, btype="bandpass", fs=2000, output="sos"),
                                       x, axis=0)[::2].T for band in cfg.bands])
        np.testing.assert_allclose(expected, whole, atol=1e-12)
        off = replace(cfg, enable_rereference=False, enable_notch=False)
        _, actual = CausalFilterBank(off).process(raw)
        expected = sosfilt(butter(4, cfg.bands[0], btype="bandpass", fs=2000, output="sos"),
                          raw[:, cfg.eeg_channels], axis=0)[::2].T
        np.testing.assert_allclose(actual[0], expected)

    def fixture(self, m=4, length=2000):
        raw = np.random.default_rng(5).normal(size=(length, 9))
        meta = np.zeros((length, len(metadata_columns(m))))
        meta[:, LABEL] = 3 if m == 4 else 2
        meta[:, TRAIN_VALID] = 1
        meta[:, STAGE] = TRAIN
        meta[:, SEGMENT] = 1
        meta[:, VALID] = 1
        meta[:, INTENSITY+m] = 1
        meta[:, INTENSITY+m+1] = np.arange(length)//16
        for r in range(m):
            meta[:, INTENSITY+r] = ((np.arange(length)+r*200//m) % 200 >= 100)
        return raw, meta

    def test_all_edges_same_gaze_and_exact_window_availability(self):
        cfg = Config()
        raw, meta = self.fixture()
        stream, epochs = EpochStream(cfg), []
        for a in range(0, len(raw), 7):
            epochs.extend(stream.feed(raw[a:a+7], meta[a:a+7]))
            self.assertTrue(all(e.stop <= min(a+7, len(raw)) for e in epochs))
        self.assertEqual({e.edge for e in epochs}, {0, 1, 2, 3})
        self.assertTrue(all(e.label == 2 for e in epochs))
        self.assertTrue(all(e.stop-e.start == 200 and e.x.shape == (10, 6, 100) for e in epochs))
        _, whole = CausalFilterBank(cfg).process(raw)
        for e in epochs:
            np.testing.assert_allclose(e.x, whole[:, :, e.start//2:e.stop//2], atol=1e-12)

    def test_reject_raw_odd_sample_boundary_but_allow_online_cue_switch(self):
        cfg = Config()
        raw, meta = self.fixture()
        meta[501:, LABEL] = 4
        meta[501:, SEGMENT] = 2
        meta[1101, TRAIN_VALID] = 0  # Odd raw sample must not disappear in downsampling.
        epochs = EpochStream(cfg).feed(raw, meta)
        self.assertFalse(any(e.start <= 500 and e.stop > 501 for e in epochs))
        self.assertFalse(any(e.start <= 1101 < e.stop for e in epochs))
        meta[:, STAGE] = ONLINE
        epochs = EpochStream(cfg).feed(raw, meta)
        self.assertTrue(any(e.start <= 500 and e.stop > 501 for e in epochs))
        meta[1101, VALID] = 0
        epochs = EpochStream(cfg).feed(raw, meta)
        self.assertFalse(any(e.start <= 1101 < e.stop for e in epochs))

    def test_trca_matches_actual_offline_implementation(self):
        path = Path(__file__).parent/"switch_event"/"label_switch_accuracy_analysis_1.py"
        source = ast.parse(path.read_text(encoding="utf-8-sig"))
        nodes = [n for n in source.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and n.name in ("_safe_corr", "TRCA", "eTRCA")]
        scope = {"np": np, "scipy": scipy}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        rng = np.random.default_rng(11)
        groups = [rng.normal(size=(8, 6, 100)) for _ in range(4)]
        test = rng.normal(size=(6, 100))
        for ensemble, name in ((False, "TRCA"), (True, "eTRCA")):
            old = scope[name]()
            old.fit(groups)
            new = TRCABand(ensemble).fit(groups)
            np.testing.assert_allclose(new.scores(test), old.predict_corr(test), atol=1e-10)

    def test_all_methods_m2_m4_freeze_shapes_and_classification(self):
        rng = np.random.default_rng(2)
        with threadpool_limits(limits=1):
            for m in (2, 4):
                for method in METHODS:
                    cfg = replace(Config(), task_m=m, method=method, min_samples=3,
                                  eeg_channels=(0, 1), bands=((5., 95.), (12., 95.)))
                    bank = EdgeModelBank(cfg)
                    templates = rng.normal(size=(m, m, 2, 2, 100))
                    for r in range(m):
                        for c in range(m):
                            for _ in range(5):
                                x = templates[r, c]+rng.normal(0, .01, (2, 2, 100))
                                bank.collect(Epoch(r, 100, 0, 200, x, c, c, True, False))
                    bank.fit()
                    for r in range(m):
                        for c in range(m):
                            label, scores = bank.predict(r, templates[r, c])
                            self.assertEqual(label, c, (m, method, r, c))
                            self.assertEqual(scores.shape, (m,))
                        if method == "Aug_FB_LDA":
                            self.assertEqual(bank.banks[r][1].n_features_in_, 2 if m == 2 else 8)
                    with self.assertRaises(RuntimeError):
                        bank.fit()
                    with self.assertRaises(RuntimeError):
                        bank.collect(Epoch(0, 100, 0, 200, x, 0, 0, True, False))

    def test_lda_stacking_matches_independent_reference(self):
        rng = np.random.default_rng(7)
        for m in (2, 4):
            cfg = replace(Config(), task_m=m, method="Aug_FB_LDA", min_samples=2,
                          eeg_channels=(0,), bands=((5., 95.), (12., 95.)))
            bank = EdgeModelBank(cfg)
            x = rng.normal(size=(m*4, 2, 1, 100))
            y = np.repeat(np.arange(m), 4)
            for r in range(m):
                for sample, c in zip(x, y):
                    bank.collect(Epoch(r, 100, 0, 200, sample, c, 0, True, False))
            bank.fit()
            stacked, test_stack = [], []
            test = rng.normal(size=(2, 1, 100))
            for b in range(2):
                base = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(x[:, b].reshape(len(x), -1), y)
                stacked.append(base.decision_function(x[:, b].reshape(len(x), -1)).reshape(len(x), -1))
                test_stack.append(base.decision_function(test[b].reshape(1, -1)).reshape(1, -1))
            final = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(np.concatenate(stacked, axis=1), y)
            expected = final.predict(np.concatenate(test_stack, axis=1))[0]
            self.assertEqual(bank.predict(0, test)[0], expected)

    def test_time_accuracy_uses_publication_hold_and_includes_missing_output(self):
        result = time_accuracy([(10, 1), (12, 2)], [(11, 1), (13, 2)], 10, 14)
        self.assertEqual(result["correct_seconds"], 2)
        self.assertEqual(result["time_accuracy"], .5)
        self.assertIsNone(time_accuracy([], [(11, 1)], 10, 14, cued=False)["time_accuracy"])
        # Publications before start establish held state; after stop cannot change the score.
        self.assertEqual(time_accuracy([(9, 1)], [(8, 1), (15, 2)], 10, 14)["time_accuracy"], 1.)

    def test_ring_wrap_and_overrun(self):
        share = OnlineState(replace(Config(), ring_seconds=.005))
        rows = np.arange(8*share.buff_width).reshape(8, -1)
        publish_rows(share, rows)
        _, got = read_rows(share)
        np.testing.assert_array_equal(rows, got)
        share.read_idx.value = 8
        publish_rows(share, rows)
        start, got = read_rows(share)
        self.assertEqual(start, 8)
        np.testing.assert_array_equal(rows, got)
        with self.assertRaises(RuntimeError):
            publish_rows(share, rows)

    def test_schedule_wall_clock_blocks_and_adjacent_cues(self):
        cfg = Config()
        schedule = Schedule(cfg)
        self.assertEqual(len(schedule.cues), 40)
        self.assertTrue(all(a[0] != b[0] for a, b in zip(schedule.cues, schedule.cues[1:])))
        share = OnlineState(replace(cfg, ring_seconds=.005))
        schedule.cue_start = 100
        duration = schedule.cue()[1]
        schedule.advance(100+duration-.01, share)
        self.assertEqual(schedule.index, 0)
        schedule.advance(100+duration+.01, share)  # No frame count is supplied.
        self.assertEqual(schedule.index, 1)
        for m in (2, 4):
            schedule = Schedule(replace(cfg, task_m=m))
            previous = None
            for _ in range(100):
                schedule.next_online()
                self.assertNotEqual(previous, schedule.online_cue[0])
                previous = schedule.online_cue[0]

    def test_degenerate_templates_and_bad_config_fail(self):
        with self.assertRaises(ValueError):
            TRCABand().fit([np.zeros((3, 6, 100)), np.zeros((3, 6, 100))])
        with self.assertRaises(ValueError):
            replace(Config(), reference_channels=()).validate()
        with self.assertRaises(ValueError):
            replace(Config(), fs=1000).validate()

    def test_fit_and_catchup_gate_and_extra_block_limit(self):
        cfg = replace(Config(), task_m=2, training_blocks=1, max_extra_blocks=1)
        share = OnlineState(replace(cfg, ring_seconds=.005))
        schedule = Schedule(cfg)
        schedule.stage = DRAIN
        schedule.index = len(schedule.cues)
        share.drained.value = True
        schedule.advance(1., share)
        self.assertEqual(schedule.stage, TRAIN)
        self.assertEqual(schedule.blocks, 2)
        self.assertFalse(share.drained.value)
        schedule.stage = DRAIN
        share.drained.value = True
        with self.assertRaises(RuntimeError):
            schedule.advance(2., share)
        share.train_ready.value = True
        schedule.advance(3., share)
        self.assertTrue(share.fit_requested.value)
        self.assertEqual(schedule.stage, FIT)
        schedule.advance(4., share)
        self.assertEqual(schedule.stage, FIT)
        share.frozen.value = True
        schedule.advance(5., share)
        self.assertEqual(schedule.stage, CATCHUP)
        share.buff_idx.value = 1000
        share.last_batch.value = 6.
        schedule.advance(6., share)
        self.assertEqual(schedule.stage, CATCHUP)
        share.read_idx.value = 1000
        share.last_batch.value = 7.
        schedule.advance(7., share)
        share.last_batch.value = 7.2
        schedule.advance(7.2, share)
        self.assertEqual(schedule.stage, CATCHUP)
        share.last_batch.value = 7.3
        schedule.advance(7.3, share, start_test=True)  # Early key cannot start test.
        self.assertEqual(schedule.stage, WAIT_TEST)
        self.assertIsNone(schedule.online_cue)
        schedule.advance(7.31, share, start_test=True)  # Prompt not presented yet.
        self.assertEqual(schedule.stage, WAIT_TEST)
        schedule.test_ready_time = 7.32
        schedule.advance(7.33, share)
        self.assertEqual(schedule.stage, WAIT_TEST)
        schedule.advance(7.34, share, start_test=True)
        self.assertEqual(schedule.stage, WARMUP)
        first_cue = schedule.cue()
        self.assertIsNone(schedule.online_start)
        schedule.test_stimulus_start = 7.35
        share.last_batch.value = 7.8
        schedule.advance(7.8, share)
        self.assertEqual(schedule.stage, WARMUP)
        share.last_batch.value = 7.86
        schedule.advance(7.86, share)
        self.assertEqual(schedule.stage, ONLINE)
        self.assertEqual(schedule.cue(), first_cue)
        self.assertIsNone(schedule.cue_start)

    def test_black_restart_excludes_boundary_and_preserves_filters(self):
        cfg = Config()
        raw, meta = self.fixture(length=1800)
        # Black 0..399; stimulus restarts at400, then scored from1000.
        meta[:400, STAGE] = WAIT_TEST
        meta[:400, INTENSITY:INTENSITY+4] = 0
        meta[:400, INTENSITY+4] = 0
        meta[400:1000, STAGE] = WARMUP
        meta[1000:, STAGE] = ONLINE
        meta[400:, INTENSITY+4] = 2
        meta[400:410, INTENSITY:INTENSITY+4] = 1  # Startup lights, not periodic edges.
        stream = EpochStream(cfg)
        self.assertEqual(stream.feed(raw[:400], meta[:400]), [])
        stream.feed(raw[400:410], meta[400:410])
        self.assertEqual(len(stream.pending), 0)
        epochs = stream.feed(raw[410:], meta[410:])
        self.assertTrue(epochs)
        self.assertTrue(all(e.online and not e.training and e.edge_sample >= 1000 and e.start >= 400 for e in epochs))
        _, full = CausalFilterBank(cfg).process(raw)
        for e in epochs:
            np.testing.assert_allclose(e.x, full[:, :, e.start//2:e.stop//2], atol=1e-12)
        self.assertEqual(stream.filters.position, len(raw))
        # Even if a sampled black frame is missing, an ID change blocks cross-run windows.
        meta[:, STAGE] = ONLINE
        meta[:, INTENSITY+4] = 1
        meta[800:, INTENSITY+4] = 2
        epochs = EpochStream(cfg).feed(raw, meta)
        self.assertFalse(any(e.start < 800 < e.stop for e in epochs))

    def test_phase_frame_resets_only_when_flicker_restarts(self):
        s = Schedule(Config())
        s.next_stimulus_frame()
        self.assertEqual((s.stimulus_id, s.stimulus_frame), (1, 0))
        s.next_stimulus_frame()
        s.segment += 1
        s.next_stimulus_frame()
        self.assertEqual((s.stimulus_id, s.stimulus_frame), (1, 2))
        for stage in (DRAIN, FIT, CATCHUP, WAIT_TEST):
            s.stage = stage
            s.next_stimulus_frame()
            self.assertEqual((s.stimulus_id, s.stimulus_frame), (0, -1))
        s.stage = WARMUP
        s.next_stimulus_frame()
        self.assertEqual((s.stimulus_id, s.stimulus_frame), (2, 0))
        s.next_stimulus_frame()
        s.stage = ONLINE
        s.next_stimulus_frame()
        self.assertEqual((s.stimulus_id, s.stimulus_frame), (2, 2))

    def test_waiting_key_is_rejected_when_decoder_or_data_not_ready(self):
        cfg = Config()
        share = OnlineState(replace(cfg, ring_seconds=.005))
        share.frozen.value = True
        s = Schedule(cfg)
        s.stage = WAIT_TEST
        s.test_ready_time = 1.
        share.last_batch.value = 1.
        s.advance(2., share, start_test=True)  # No fresh samples for1s.
        self.assertEqual(s.stage, CATCHUP)
        self.assertIsNone(s.test_space_accepted_at)
        s.stage = WAIT_TEST
        share.last_batch.value = 2.
        share.buff_idx.value = 100
        s.advance(2., share, start_test=True)  # 50ms backlog, above20ms readiness.
        self.assertEqual(s.stage, CATCHUP)
        self.assertIsNone(s.test_space_accepted_at)

    def test_draw_black_prompt_and_warmup_without_feedback(self):
        from unittest.mock import Mock
        stim = Mock()
        stim.positions = [(0, 0)]*4
        graphics = Mock()
        for stage in (DRAIN, FIT, CATCHUP):
            draw_frame(stim, graphics, Config(), stage, 0, 0, [0]*4)
        stim._draw_target_square.assert_not_called()
        stim._draw_centered_text_5x7.assert_not_called()
        draw_frame(stim, graphics, Config(), WAIT_TEST, 0, 0, [0]*4)
        stim._draw_centered_text_5x7.assert_called_once_with("PRESS SPACE TO START", pixel_size=3.)
        draw_frame(stim, graphics, Config(), WARMUP, 3, 2, [1, 1, 0, 0])
        self.assertEqual(stim._draw_target_square.call_count, 4)
        stim._draw_cue.assert_called_once_with(3)
        graphics.glBegin.assert_not_called()  # No prediction marker in warmup.


if __name__ == "__main__":
    unittest.main()
