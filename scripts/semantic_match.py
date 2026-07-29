#!/usr/bin/env python3
"""
Semantic lyric-to-clip matching over transcript segments using pgvector.

Keyword search fails on figurative lyrics: "a life that doesn't heal" shares no
vocabulary with how anyone actually speaks, and surface overlap produces
inverted matches (a clip about *pretending* to be sick scoring against a lyric
about *hiding* illness). Embeddings compare meaning instead.

Segments are embedded with a local Ollama model and stored in pgvector, so
matching is a cosine-distance query.

Output is a candidate edit list. Clips from creators who have not given
permission still need their Duet/Stitch setting verified in the TikTok app.

Usage:
    python scripts/semantic_match.py init
    python scripts/semantic_match.py embed --creator chronically.roxii
    python scripts/semantic_match.py embed --min-views 500000 --limit 4000
    python scripts/semantic_match.py match --lyrics song.md --per-line 3 \
        --creator chronically.roxii --out editlist.csv
"""
import sys
import csv
import json
import argparse
import re
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import RealDictCursor, execute_values
from database import get_connection
from config import OLLAMA_URL

EMBED_MODEL = 'nomic-embed-text'
DIMS = 768

# nomic-embed-text is trained with task prefixes; using the right one on each
# side materially improves retrieval.
DOC_PREFIX = 'search_document: '
QUERY_PREFIX = 'search_query: '

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS segment_embeddings (
    id           BIGSERIAL PRIMARY KEY,
    video_id     INTEGER NOT NULL,
    transcript_id INTEGER NOT NULL,
    seg_index    INTEGER NOT NULL,
    start_s      REAL NOT NULL,
    end_s        REAL NOT NULL,
    text         TEXT NOT NULL,
    embedding    vector({DIMS}) NOT NULL,
    UNIQUE (transcript_id, seg_index)
);

CREATE INDEX IF NOT EXISTS idx_segemb_video ON segment_embeddings (video_id);
-- IVFFlat needs training data, so it is created separately once rows exist.
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_segemb_vec ON segment_embeddings
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""


def embed(texts, prefix):
    """Embed a batch of texts via Ollama."""
    body = json.dumps({'model': EMBED_MODEL,
                       'input': [prefix + t for t in texts]}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL.rstrip('/')}/api/embed", body,
                                {'Content-Type': 'application/json'})
    d = json.load(urllib.request.urlopen(req, timeout=300))
    return d['embeddings']


def cmd_init(args):
    with get_connection() as conn:
        conn.cursor().execute(SCHEMA)
    print("Created segment_embeddings")
    return 0


def cmd_embed(args):
    """Embed transcript segments into pgvector."""
    where = ["t.segments IS NOT NULL", "t.song_lyrics_ratio < 0.2"]
    params = []
    if args.creator:
        where.append("v.author ILIKE %s"); params.append(args.creator)
    if args.min_views:
        where.append("v.view_count >= %s"); params.append(args.min_views)

    sql = f"""SELECT t.id AS transcript_id, t.video_id, t.segments
              FROM transcripts t JOIN videos v ON v.id = t.video_id
              WHERE {' AND '.join(where)}
                AND NOT EXISTS (SELECT 1 FROM segment_embeddings se
                                WHERE se.transcript_id = t.id)
              ORDER BY v.view_count DESC NULLS LAST"""
    if args.limit:
        sql += " LIMIT %s"; params.append(args.limit)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        print("Nothing to embed (already done, or no rows matched).")
        return 0

    # Flatten to segments. Very short segments carry no usable meaning and only
    # add noise to retrieval, so they are skipped.
    pending = []
    for r in rows:
        segs = r['segments']
        if isinstance(segs, str):
            try: segs = json.loads(segs)
            except Exception: continue
        for i, s in enumerate(segs or []):
            txt = (s.get('text') or '').strip()
            if len(txt.split()) < 4:
                continue
            pending.append((r['video_id'], r['transcript_id'], i,
                            float(s.get('start', 0)), float(s.get('end', 0)), txt))

    print(f"{len(rows)} transcript(s) -> {len(pending)} segment(s) to embed")
    done = 0
    B = args.batch
    for i in range(0, len(pending), B):
        chunk = pending[i:i+B]
        try:
            vecs = embed([c[5] for c in chunk], DOC_PREFIX)
        except Exception as ex:
            print(f"  batch {i//B}: {type(ex).__name__} {str(ex)[:90]} - skipped")
            continue
        with get_connection() as conn:
            execute_values(conn.cursor(), """
                INSERT INTO segment_embeddings
                    (video_id, transcript_id, seg_index, start_s, end_s, text, embedding)
                VALUES %s ON CONFLICT (transcript_id, seg_index) DO NOTHING
            """, [(c[0], c[1], c[2], c[3], c[4], c[5], str(v))
                  for c, v in zip(chunk, vecs)])
        done += len(chunk)
        print(f"  {done}/{len(pending)}", end='\r', flush=True)

    print(f"\nEmbedded {done} segment(s)")
    with get_connection() as conn:
        conn.cursor().execute(INDEX_SQL)
    print("Vector index ready")
    return 0


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


def cmd_match(args):
    lyrics = parse_lyrics(args.lyrics)
    print(f"Parsed {len(lyrics)} lyric line(s)\n")

    # One embedding call for all lyric lines.
    vecs = embed([l for _, l in lyrics], QUERY_PREFIX)

    rows_out, unmatched = [], 0
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for (section, line), qv in zip(lyrics, vecs):
            # Placeholder order must match the SQL exactly: the query vector
            # appears once in the CTE, then the optional filters, then the
            # distance ceiling and row limit.
            filters, filter_params = [], []
            if args.creator:
                filters.append("v.author ILIKE %s"); filter_params.append(args.creator)
            if args.min_views:
                filters.append("v.view_count >= %s"); filter_params.append(args.min_views)
            filter_sql = (' AND ' + ' AND '.join(filters)) if filters else ''

            # Compute distance once in a CTE so it can be filtered and ordered
            # without repeating the vector literal.
            cur.execute(f"""
                WITH scored AS (
                    SELECT v.author, v.url, v.view_count,
                           se.start_s, se.end_s, se.text,
                           se.embedding <=> %s::vector AS distance
                    FROM segment_embeddings se
                    JOIN videos v ON v.id = se.video_id
                    WHERE TRUE{filter_sql}
                )
                SELECT * FROM scored
                WHERE distance < %s
                ORDER BY distance ASC
                LIMIT %s
            """, [str(qv), *filter_params, args.max_distance, args.per_line])

            got = cur.fetchall()
            print(f"[{section}] {line}")
            if not got:
                unmatched += 1
                print("    (no match)\n")
                continue
            for r in got:
                sim = 1 - float(r['distance'])
                rows_out.append({
                    'section': section, 'lyric': line,
                    'creator': r['author'], 'url': r['url'],
                    'views': r['view_count'],
                    'clip_start_s': round(float(r['start_s']), 2),
                    'clip_end_s': round(float(r['end_s']), 2),
                    'clip_len_s': round(float(r['end_s']) - float(r['start_s']), 2),
                    'similarity': round(sim, 4),
                    'spoken_text': r['text'].strip(),
                })
                print(f"    sim={sim:.3f}  @{(r['author'] or '?'):20} "
                      f"{float(r['start_s']):6.1f}-{float(r['end_s']):6.1f}s  "
                      f"{(r['view_count'] or 0):>9,}  \"{r['text'].strip()[:66]}\"")
            print()

    print(f"Coverage: {len(lyrics)-unmatched}/{len(lyrics)} lyric lines matched")
    if args.out and rows_out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader(); w.writerows(rows_out)
        print(f"Wrote {len(rows_out)} candidate clip(s) to {args.out}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help='Create the embeddings table')

    e = sub.add_parser('embed', help='Embed transcript segments')
    e.add_argument('--creator'); e.add_argument('--min-views', type=int)
    e.add_argument('--limit', type=int); e.add_argument('--batch', type=int, default=64)

    m = sub.add_parser('match', help='Match lyrics semantically')
    m.add_argument('--lyrics', required=True)
    m.add_argument('--per-line', type=int, default=3)
    m.add_argument('--creator'); m.add_argument('--min-views', type=int)
    m.add_argument('--max-distance', type=float, default=0.55,
                   help='Cosine distance ceiling; lower is stricter (default 0.55)')
    m.add_argument('--out')

    args = p.parse_args()
    return {'init': cmd_init, 'embed': cmd_embed, 'match': cmd_match}[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
