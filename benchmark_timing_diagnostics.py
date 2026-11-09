"""Recording-only overhead, not a 600 Hz display/amplifier performance test."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from timing_diagnostics import TimingDiagnostics, COLUMNS


def run(mode, count):
    frames=[]
    d=TimingDiagnostics(mode=='enabled')
    gc.collect()
    d.attach()
    elapsed=np.empty(count)
    start=time.perf_counter()
    for i in range(count):
        t=time.perf_counter()
        if mode!='legacy':
            d.begin(i)
            for name in COLUMNS[2:-2]:d.mark(name)
        # Same allocation/serialization pattern as the existing frame log.
        frames.append(dict(frame=i,swap_before=t,swap_after=t,interval=1/600,
            stage='training',segment=1,cue=1,stimulus_id=1,stimulus_frame=i,
            displayed_prediction=0,intensity=json.dumps([1.,1.,0.,0.]),decoder_backlog_samples=0))
        if mode!='legacy':d.finish()
        elapsed[i]=(time.perf_counter()-t)*1e6
    total=time.perf_counter()-start
    d.detach()
    return dict(mode=mode,frames=count,total_seconds=total,
        median_us=float(np.median(elapsed)),p95_us=float(np.percentile(elapsed,95)),
        p99_us=float(np.percentile(elapsed,99)),max_us=float(elapsed.max()),
        numeric_buffer_bytes=sum(c.nbytes for c in d.chunks),gc_events=len(d.gc_events))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames',type=int,default=100000)
    parser.add_argument('--output',type=Path,default=Path('online_results/timing_probe_overhead.json'))
    args=parser.parse_args()
    results=[]
    for mode in ('legacy','enabled','disabled','disabled','enabled','legacy'):
        r=run(mode,args.frames);results.append(r);print(json.dumps(r),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scope='Recording-only CPU/memory benchmark; no hardware or display',
                                         runs=results),indent=2),encoding='utf-8')
