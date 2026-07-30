#!/usr/bin/env python3
"""
Append a specific creator's best on-theme clips to a montage CSV.

build_montage.py ranks candidates by view count so the most-watched creators
surface first. A low-view creator therefore never appears, even with a raised
per-creator cap. This selects a named creator's strongest matches by semantic
similarity instead, so they can be featured deliberately.

Usage:
    python scripts/add_creator_clips.py --creator chronically.roxii --count 8 \
        --themes data/lyrics/suffering_themes.md \
        --csv data/lyrics/montage_final.csv
"""
import sys
import csv
import json
import re
import argparse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import RealDictCursor
from database import get_connection
from config import OLLAMA_URL


def embed(texts):
    body = json.dumps({'model': 'nomic-embed-text',
                       'input': ['search_query: ' + t for t in texts]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/embed", body,
                                {'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=300))['embeddings']


def parse_themes(path):
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--creator', required=True)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--themes', required=True)
    p.add_argument('--csv', required=True, help='Montage CSV to append to')
    p.add_argument('--min-symptoms', type=int, default=2)
    p.add_argument('--max-distance', type=float, default=0.55)
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    fields = list(rows[0].keys())
    used = {(r['url'], str(round(float(r['clip_start_s']), 2))) for r in rows}

    themes = parse_themes(args.themes)
    vecs = embed([t for _, t in themes])

    # Best match per theme, then the strongest overall, so the chosen clips
    # span different aspects of hardship rather than clustering on one.
    found = []
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for (section, theme), qv in zip(themes, vecs):
            cur.execute("""
                WITH scored AS (
                    SELECT v.author, v.url, v.view_count,
                           se.start_s, se.end_s, se.text,
                           se.embedding <=> %s::vector AS distance,
                           (SELECT count(*) FROM symptoms s
                             WHERE s.video_id = v.id) AS n_symptoms
                    FROM segment_embeddings se
                    JOIN videos v ON v.id = se.video_id
                    WHERE v.author ILIKE %s
                      AND (v.duet_enabled OR v.stitch_enabled)
                )
                SELECT * FROM scored
                WHERE distance < %s AND n_symptoms >= %s
                ORDER BY distance ASC LIMIT 3
            """, (str(qv), args.creator, args.max_distance, args.min_symptoms))
            for r in cur.fetchall():
                key = (r['url'], str(round(float(r['start_s']), 2)))
                if key in used:
                    continue
                found.append((float(r['distance']), section, theme, r))
                break

    found.sort(key=lambda x: x[0])
    picked, seen = [], set()
    for dist, section, theme, r in found:
        key = (r['url'], str(round(float(r['start_s']), 2)))
        if key in seen:
            continue
        seen.add(key)
        sim = 1 - dist
        picked.append({
            'section': section, 'lyric': theme,
            'creator': r['author'], 'url': r['url'], 'views': r['view_count'],
            'clip_start_s': round(float(r['start_s']), 2),
            'clip_end_s': round(float(r['end_s']), 2),
            'clip_len_s': round(float(r['end_s']) - float(r['start_s']), 3),
            'similarity': round(sim, 4),
            'spoken_text': r['text'].strip(),
            'match_quality': ('strong' if sim >= 0.68 else
                              'moderate' if sim >= 0.60 else 'weak'),
            'verify_reason': 'creator featured by request',
        })
        if len(picked) >= args.count:
            break

    if not picked:
        print(f"No unused on-theme clips found for @{args.creator}")
        return 1

    print(f"Adding {len(picked)} clip(s) from @{args.creator}:")
    for c in picked:
        print(f"    sim={c['similarity']:.3f} {c['clip_start_s']:>7.1f}s  "
              f"\"{c['spoken_text'][:58]}\"")

    with open(args.csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows + picked)
    print(f"\n{len(rows) + len(picked)} total clip(s) in {args.csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
