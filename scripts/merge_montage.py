#!/usr/bin/env python3
"""
Merge verified montage CSVs into an exact-length clip list.

Combines multiple verified candidate sets, drops duplicate video moments, and
selects exactly N clips ranked by view count so the most-watched creators are
favoured. Named creators are pinned in first regardless of reach, since a
low-view creator would otherwise never survive a popularity ranking.

Clip length is set to song_duration / N so the assembled video ends with the
song.

Usage:
    python scripts/merge_montage.py --csv a.csv --csv b.csv --clips 100 \
        --song "/path/song.wav" --pin chronically.roxii:8 --out final.csv
"""
import sys
import csv
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def song_duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration', '-of', 'default=nw=1:nk=1', path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', action='append', required=True)
    p.add_argument('--song', required=True)
    p.add_argument('--clips', type=int, default=100)
    p.add_argument('--pin', action='append', default=[], metavar='HANDLE:N',
                   help='Reserve N slots for a creator (repeatable)')
    p.add_argument('--max-per-creator', type=int, default=3)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    dur = song_duration(args.song)
    clip_len = dur / args.clips

    rows, seen = [], set()
    for path in args.csv:
        for r in csv.DictReader(open(path, encoding='utf-8')):
            key = (r['url'], str(round(float(r['clip_start_s']), 2)))
            if key in seen:
                continue
            seen.add(key); rows.append(r)
    print(f"{len(rows)} unique verified clip(s) pooled from {len(args.csv)} file(s)")

    pins = {}
    for spec in args.pin:
        h, _, n = spec.partition(':')
        pins[h.lower().lstrip('@')] = int(n or 1)

    def handle(r):
        return (r.get('creator') or '').lower().lstrip('@')

    # Pinned creators first, strongest semantic match among their clips.
    chosen, per_creator = [], {}
    for h, n in pins.items():
        cands = sorted((r for r in rows if handle(r) == h),
                       key=lambda r: -float(r.get('similarity') or 0))
        for r in cands[:n]:
            chosen.append(r); per_creator[h] = per_creator.get(h, 0) + 1
        print(f"  pinned @{h}: {min(n, len(cands))} clip(s)")

    # Remainder by reach, respecting the per-creator cap.
    rest = sorted((r for r in rows if r not in chosen),
                  key=lambda r: -(int(r['views'] or 0)))
    for r in rest:
        if len(chosen) >= args.clips:
            break
        h = handle(r)
        if per_creator.get(h, 0) >= args.max_per_creator:
            continue
        chosen.append(r); per_creator[h] = per_creator.get(h, 0) + 1

    if len(chosen) < args.clips:
        clip_len = dur / len(chosen)
        print(f"\nOnly {len(chosen)} clip(s) available; clip length adjusted "
              f"to {clip_len:.3f}s to still fill the song.")

    for r in chosen:
        r['clip_len_s'] = round(clip_len, 3)
        r['clip_end_s'] = round(float(r['clip_start_s']) + clip_len, 2)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0].keys()),
                           extrasaction='ignore')
        w.writeheader(); w.writerows(chosen)

    views = sorted((int(r['views'] or 0) for r in chosen), reverse=True)
    print(f"\n{len(chosen)} clip(s), {clip_len:.3f}s each = "
          f"{len(chosen)*clip_len:.1f}s (song {dur:.1f}s)")
    print(f"creators: {len(per_creator)}   "
          f"median views: {views[len(views)//2]:,}   top: {views[0]:,}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
