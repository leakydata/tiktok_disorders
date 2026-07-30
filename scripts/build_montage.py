#!/usr/bin/env python3
"""
Select N unique clips to fill a song, maximising creator diversity.

Unlike fill_gaps.py, which picks one clip per lyric line, this builds a montage:
a fixed number of equal-length clips spread across as many different creators as
possible. Clip length is derived from the song duration divided by the clip
count, so the assembled video lands on the song's end without padding.

Selection is greedy and diversity-first: it walks the lyric lines in song order
taking each creator's best unused moment, capped per creator, and only takes a
second clip from a creator once every eligible creator has had one. Ranking
within a line prefers higher view counts, so the most-watched creators appear
first.

Every candidate is drawn from videos whose recorded permissions allow reuse and
which have extracted symptoms, so output needs no further filtering.

Usage:
    python scripts/build_montage.py --lyrics data/lyrics/close_enough_to_fine.md \
        --song "/home/scholyx/Music/Close Enough to Fine.wav" \
        --clips 100 --out data/lyrics/montage.csv
"""
import sys
import csv
import json
import re
import argparse
import subprocess
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import RealDictCursor
from database import get_connection
from config import OLLAMA_URL

EMBED_MODEL = 'nomic-embed-text'
QUERY_PREFIX = 'search_query: '


def embed(texts):
    body = json.dumps({'model': EMBED_MODEL,
                       'input': [QUERY_PREFIX + t for t in texts]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/embed", body,
                                {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=300))['embeddings']


def parse_lyrics(path):
    section, out = 'unknown', []
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        m = re.fullmatch(r'\[(.+?)\]', line)
        if m:
            section = m.group(1); continue
        out.append((section, line))
    return out


def song_duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                        'format=duration', '-of', 'default=nw=1:nk=1', path],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lyrics', required=True)
    p.add_argument('--song', required=True, help='Audio file, to derive length')
    p.add_argument('--clips', type=int, default=100)
    p.add_argument('--out', required=True)
    p.add_argument('--min-symptoms', type=int, default=2)
    p.add_argument('--min-views', type=int, default=0)
    p.add_argument('--max-per-creator', type=int, default=3)
    p.add_argument('--candidates-per-line', type=int, default=40,
                   help='Pool size fetched per lyric line before selection')
    p.add_argument('--max-distance', type=float, default=0.55)
    p.add_argument('--rank-by', choices=('views','similarity'), default='views',
                   help='Ordering within each line pool (default: views)')
    p.add_argument('--boost-creator', action='append', default=[],
                   metavar='HANDLE:N',
                   help='Allow a creator more clips than the cap, '
                        'e.g. chronically.roxii:6 (repeatable)')
    p.add_argument('--exclude-creator', action='append', default=[])
    args = p.parse_args()

    dur = song_duration(args.song)
    clip_len = dur / args.clips
    print(f"song {dur:.2f}s / {args.clips} clips = {clip_len:.3f}s per clip\n")

    lyrics = parse_lyrics(args.lyrics)
    excl = {c.lower().lstrip('@') for c in args.exclude_creator}
    vecs = embed([l for _, l in lyrics])

    # Candidate pool per lyric line, ordered by semantic distance.
    pools = []
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for (section, line), qv in zip(lyrics, vecs):
            cur.execute("""
                WITH scored AS (
                    SELECT v.author, v.url, v.view_count,
                           se.start_s, se.end_s, se.text,
                           se.embedding <=> %s::vector AS distance,
                           (SELECT count(*) FROM symptoms s
                             WHERE s.video_id = v.id) AS n_symptoms
                    FROM segment_embeddings se
                    JOIN videos v ON v.id = se.video_id
                    WHERE (v.duet_enabled OR v.stitch_enabled)
                      AND v.view_count >= %s
                )
                SELECT * FROM scored
                WHERE distance < %s AND n_symptoms >= %s
                  AND end_s - start_s >= %s
                ORDER BY distance ASC
                LIMIT %s
            """, (str(qv), args.min_views, args.max_distance, args.min_symptoms,
                  clip_len * 0.6, args.candidates_per_line))
            cands = [r for r in cur.fetchall()
                     if (r['author'] or '').lower().lstrip('@') not in excl]
            if args.rank_by == 'views':
                # Semantic distance already gated the pool; ordering by reach
                # inside it surfaces the most-watched creators first.
                cands.sort(key=lambda r: -(r['view_count'] or 0))
            pools.append({'section': section, 'lyric': line, 'cands': cands})

    total_cands = sum(len(p['cands']) for p in pools)
    creators = {r['author'] for p in pools for r in p['cands']}
    print(f"candidate pool: {total_cands} clip(s) across {len(creators)} creator(s)")

    # Greedy diversity-first selection. Pass 1 allows one clip per creator, so
    # breadth is established before any creator gets a second slot; later passes
    # raise the cap until the target count is met.
    # Per-creator overrides, e.g. --boost-creator chronically.roxii:6
    boosts = {}
    for spec in args.boost_creator:
        h, _, n = spec.partition(':')
        boosts[h.lower().lstrip('@')] = int(n or args.max_per_creator + 3)

    chosen, used_moments, per_creator = [], set(), {}
    max_cap = max([args.max_per_creator] + list(boosts.values()))
    for cap in range(1, max_cap + 1):
        for pool in pools:
            if len(chosen) >= args.clips:
                break
            for r in pool['cands']:
                handle = r['author']
                limit = boosts.get((handle or '').lower().lstrip('@'),
                                   args.max_per_creator)
                moment = (r['url'], round(float(r['start_s']), 2))
                if moment in used_moments:
                    continue
                if per_creator.get(handle, 0) >= min(cap, limit):
                    continue
                used_moments.add(moment)
                per_creator[handle] = per_creator.get(handle, 0) + 1
                sim = 1 - float(r['distance'])
                chosen.append({
                    'section': pool['section'], 'lyric': pool['lyric'],
                    'creator': handle, 'url': r['url'],
                    'views': r['view_count'],
                    'clip_start_s': round(float(r['start_s']), 2),
                    'clip_end_s': round(float(r['start_s']) + clip_len, 2),
                    'clip_len_s': round(clip_len, 3),
                    'similarity': round(sim, 4),
                    'spoken_text': r['text'].strip(),
                    'match_quality': ('strong' if sim >= 0.68 else
                                      'moderate' if sim >= 0.60 else 'weak'),
                })
                break
        if len(chosen) >= args.clips:
            break

    if len(chosen) < args.clips:
        print(f"\nOnly {len(chosen)} unique clip(s) available "
              f"(wanted {args.clips}).")
        print("Raise --max-per-creator or --max-distance, or lower "
              "--min-symptoms / --min-views.")
        clip_len = dur / len(chosen)
        print(f"Recomputed clip length: {clip_len:.3f}s to still fill the song.")
        for c in chosen:
            c['clip_len_s'] = round(clip_len, 3)
            c['clip_end_s'] = round(c['clip_start_s'] + clip_len, 2)

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(chosen[0].keys()))
        w.writeheader(); w.writerows(chosen)

    q = {}
    for c in chosen:
        q[c['match_quality']] = q.get(c['match_quality'], 0) + 1
    print(f"\nSelected {len(chosen)} unique clip(s) from "
          f"{len(per_creator)} creator(s), {clip_len:.3f}s each "
          f"= {len(chosen) * clip_len:.1f}s")
    print("  quality: " + "   ".join(f"{k}: {v}" for k, v in sorted(q.items())))
    top = sorted(per_creator.items(), key=lambda kv: -kv[1])[:6]
    print("  most used: " + ", ".join(f"@{k}({v})" for k, v in top))
    print(f"\nWrote {args.out}")
    print(f"\nNext:\n  uv run scripts/make_clips.py --csv {args.out} \\\n"
          f"    --out-dir data/clips_montage --min-len {clip_len:.2f} "
          f"--max-len {clip_len:.2f} --lead 0 --tail 0")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
