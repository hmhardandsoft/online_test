"""Export long-frame intervals relative to an ETW marker, with clock bounds."""
import argparse
import csv
import json
from pathlib import Path


def trace_windows(context, diagnostics, marker_seconds=0.0, marker_name='workers_started'):
    anchors = [a for a in context['anchors']
               if a['provider_enabled'] and a['event_status'] == 0]
    if not anchors:
        raise ValueError('No enabled ETW marker was recorded. Start WPR before the experiment.')
    anchor = next((a for a in anchors if a['name'] == marker_name), None)
    if anchor is None:
        raise ValueError(f'No enabled, successful ETW marker named {marker_name}')
    rows = []
    for item in diagnostics['long_intervals']:
        row = dict(frame=item['frame'], stage=item['stage'], interval_ms=item['interval_ms'],
                   largest_component=item['largest_component'],
                   largest_draw_component=item.get('largest_draw_component'),
                   marker_name=anchor['name'], pid=context['pid'],
                   tid=context['main_thread_native_id'],
                   marker_bracket_ms=anchor['emit_bracket_ms'])
        for name in ('start', 'stop'):
            # ETW event lies somewhere inside [emit_before, emit_after].
            row[name+'_seconds_low'] = marker_seconds + item[name] - anchor['emit_after']
            row[name+'_seconds_high'] = marker_seconds + item[name] - anchor['emit_before']
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    parser.add_argument('--marker-seconds', type=float, default=0.,
                        help='Matching ETW marker time in WPA seconds; default exports offsets from marker')
    parser.add_argument('--marker-name', choices=('workers_started', 'run_stop'), default='workers_started')
    args = parser.parse_args()
    context = json.loads((args.run_directory/'render_trace_context.json').read_text(encoding='utf-8'))
    diag = json.loads((args.run_directory/'timing_diagnostics_summary.json').read_text(encoding='utf-8'))
    rows = trace_windows(context, diag, args.marker_seconds, args.marker_name)
    if not rows:
        print('No long-frame intervals.'); return
    output = args.run_directory/'render_trace_windows.csv'
    with output.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'{output}: {len(rows)} intervals; verify the matching run_id/marker in ETL.')


if __name__ == '__main__':
    main()
