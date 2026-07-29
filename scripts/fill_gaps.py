#!/usr/bin/env python3
"""
Backfill lyric lines that no candidate reached at the strict threshold.

semantic_match.py applies one distance ceiling to every line, so figurative
lines end up with nothing while literal ones get several candidates. This walks
the uncovered lines only, progressively loosening the ceiling until each finds a
match that is permitted for reuse and on topic, then merges the results into
one CSV in song order.

Only videos whose recorded permissions allow reuse are considered, so the
output needs no further permission filtering.

Usage:
    python scripts/fill_gaps.py --lyrics data/lyrics/close_enough_to_fine.md \
        --csv data/lyrics/editlist_full.csv \
        --out data/lyrics/editlist_complete.csv --min-symptoms 2
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

EMBED_MODEL = 'nomic-embed-text'
QUERY_PREFIX = 'search_query: '

# Tried in order. A line that only matches at 0.55 is a weak match and is
# labelled as such in the output rather than silently mixed in with the rest.
LADDER = [0.32, 0.38, 0.44, 0.50, 0.55]


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lyrics', required=True)
    p.add_argument('--csv', required=True, help='Existing match CSV to extend')
    p.add_argument('--out', required=True)
    p.add_argument('--min-symptoms', type=int, default=2)
    p.add_argument('--min-views', type=int, default=0)
    p.add_argument('--exclude-creator', action='append', default=[],
                   help='Creator to skip (repeatable)')
    args = p.parse_args()

    lyrics = parse_lyrics(args.lyrics)
    existing = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    fields = list(existing[0].keys()) if existing else []
    if 'match_quality' not in fields:
        fields.append('match_quality')

    # Best existing row per line, so the merged output is one clip per line.
    best = {}
    for r in existing:
        key = (r['section'], r['lyric'])
        if key not in best or float(r['similarity']) > float(best[key]['similarity']):
            r.setdefault('match_quality', 'strong')
            best[key] = r

    missing = [(s, l) for s, l in lyrics if (s, l) not in best]
    print(f"{len(lyrics)} lyric line(s): {len(best)} covered, "
          f"{len(missing)} to backfill\n")

    if missing:
        vecs = embed([l for _, l in missing])
        excl = [c.lower().lstrip('@') for c in args.exclude_creator]
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            for (section, line), qv in zip(missing, vecs):
                found = None
                for ceiling in LADDER:
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
                        ORDER BY distance ASC LIMIT 5
                    """, (str(qv), args.min_views, ceiling, args.min_symptoms))
                    for r in cur.fetchall():
                        if (r['author'] or '').lower().lstrip('@') in excl:
                            continue
                        found = (r, ceiling); break
                    if found:
                        break

                if not found:
                    print(f"  [{section}] \"{line[:44]}\"  -> STILL EMPTY")
                    continue
                r, ceiling = found
                sim = 1 - float(r['distance'])
                quality = ('strong' if ceiling <= 0.32 else
                           'moderate' if ceiling <= 0.44 else 'weak')
                best[(section, line)] = {
                    'section': section, 'lyric': line,
                    'creator': r['author'], 'url': r['url'],
                    'views': r['view_count'],
                    'clip_start_s': round(float(r['start_s']), 2),
                    'clip_end_s': round(float(r['end_s']), 2),
                    'clip_len_s': round(float(r['end_s']) - float(r['start_s']), 2),
                    'similarity': round(sim, 4),
                    'spoken_text': r['text'].strip(),
                    'match_quality': quality,
                }
                print(f"  [{section}] \"{line[:40]}\"")
                print(f"       {quality:8} sim={sim:.3f} @{r['author']} "
                      f"{float(r['start_s']):.1f}s sym={r['n_symptoms']} "
                      f"\"{r['text'].strip()[:52]}\"")

    ordered = [best[(s, l)] for s, l in lyrics if (s, l) in best]
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for row in ordered:
            row.setdefault('match_quality', 'strong')
            w.writerow(row)

    q = {}
    for row in ordered:
        q[row.get('match_quality', 'strong')] = q.get(row.get('match_quality', 'strong'), 0) + 1
    print(f"\nWrote {len(ordered)}/{len(lyrics)} line(s) to {args.out}")
    print("  " + "   ".join(f"{k}: {v}" for k, v in sorted(q.items())))
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
