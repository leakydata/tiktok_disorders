#!/usr/bin/env python3
"""
Match song lyrics to transcript segments, producing a candidate edit list.

For each lyric line, finds transcript segments whose spoken content expresses
the same idea, and reports the exact in-video timestamp. Output is a *candidate*
list only -- every clip still needs its Duet/Stitch permission verified in the
TikTok app before use, because that setting is not recorded in this database.

Matching is two-stage: PostgreSQL full-text search shortlists transcripts, then
each transcript's `segments` array is scanned to locate the specific segment and
its start/end offsets.

Usage:
    python scripts/lyric_match.py index          # one-off: build the FTS index
    python scripts/lyric_match.py match --lyrics song.md --per-line 5
    python scripts/lyric_match.py match --lyrics song.md --min-views 100000 \
        --out editlist.csv
"""
import sys
import csv
import json
import argparse
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import RealDictCursor
from database import get_connection

# Theme expansion. Lyrics are figurative; transcripts are literal. A line like
# "close enough to fine" never appears verbatim, so distinctive lyric phrases
# are mapped to the vocabulary people actually use. Keys are matched as
# substrings against the lowercased lyric line.
THEME_EXPANSIONS = {
    'better at hiding':      ['masking', 'pretend', 'hide how i feel', 'act normal',
                              'put on a face', 'nobody knows how bad'],
    'look better':           ['you look fine', 'you look healthy', 'look so good',
                              "don't look sick", 'invisible illness'],
    'close enough to fine':  ['normal results', 'labs came back normal',
                              'tests were normal', 'nothing wrong', 'borderline',
                              'within range', 'all my tests'],
    'thanked the doctor':    ['doctor dismissed', 'dismissed me', 'gaslighting',
                              'not taken seriously', 'told me it was anxiety',
                              'medical gaslighting'],
    'another door that closes': ['another specialist', 'referred me', 'waiting list',
                                 'no answers', 'still no diagnosis'],
    'used to make plans':    ['cancel plans', 'had to cancel', 'pacing',
                              'spoon theory', 'energy budget', 'flare up'],
    'miss a version of myself': ['who i used to be', 'before i got sick',
                                 'the old me', 'grieving', 'lost years'],
    'so strong':             ['you are so strong', 'so brave', 'inspiration',
                              "don't feel strong", 'i had no choice'],
    'just tired':            ['exhausted', 'fatigue', 'no energy', 'bone tired',
                              'crushing fatigue'],
    'stop answering':        ['isolated', 'stopped texting', 'lost friends',
                              'nobody checks in', 'alone'],
    'sit beside me':         ['support', 'just listen', 'believe me',
                              'validation', 'someone who gets it'],
    'cried':                 ['cried in the car', 'broke down', 'crying'],
    'life that doesn':       ['no cure', 'chronic', 'forever', 'manage it',
                              'never get better'],
}

STOPWORDS = set("""a an the and or but if so as at by for in of on to with i im i'm me my
mine you your we they them their that this it its is was were be been being do does did
not no dont don't cant can't just how what when who why like from out up down about
into over then than there here now got get gets getting say says said""".split())

FTS_SQL = """
ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS text_fts tsvector
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(text,''))) STORED;
CREATE INDEX IF NOT EXISTS idx_transcripts_text_fts ON transcripts USING GIN (text_fts);
"""


def cmd_index(args):
    """Create a generated tsvector column and GIN index for fast phrase search."""
    print("Building full-text index (one-off, may take a minute on 79k rows)...")
    with get_connection() as conn:
        conn.cursor().execute(FTS_SQL)
    print("Done: transcripts.text_fts + idx_transcripts_text_fts")
    return 0


def parse_lyrics(path):
    """Parse a lyrics file into (section, line) pairs, skipping headers."""
    section = 'unknown'
    out = []
    for raw in Path(path).read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        m = re.fullmatch(r'\[(.+?)\]', line)
        if m:
            section = m.group(1)
            continue
        out.append((section, line))
    return out


def build_terms(line):
    """Content words from the lyric plus any matching theme expansions."""
    low = line.lower()
    phrases = []
    for key, expansions in THEME_EXPANSIONS.items():
        if key in low:
            phrases.extend(expansions)
    words = [w for w in re.findall(r"[a-z']+", low)
             if w not in STOPWORDS and len(w) > 3]
    return words, phrases


def best_segment(segments, terms):
    """Return the segment with the highest term overlap, plus its offsets."""
    if not segments:
        return None
    if isinstance(segments, str):
        try:
            segments = json.loads(segments)
        except Exception:
            return None
    best, best_score = None, 0
    for seg in segments:
        txt = (seg.get('text') or '').lower()
        score = sum(1 for t in terms if t in txt)
        if score > best_score:
            best, best_score = seg, score
    return (best, best_score) if best else None


def cmd_match(args):
    lyrics = parse_lyrics(args.lyrics)
    if not lyrics:
        print("No lyric lines parsed.")
        return 1
    print(f"Parsed {len(lyrics)} lyric line(s) from {args.lyrics}\n")

    rows_out = []
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for section, line in lyrics:
            words, phrases = build_terms(line)
            if not words and not phrases:
                continue

            # OR-query over content words and expansion phrases. websearch
            # syntax keeps multi-word phrases intact.
            query = ' or '.join(
                [f'"{p}"' for p in phrases] + words[:8]
            )
            cur.execute("""
                SELECT v.author, v.url, v.view_count, v.upload_date,
                       t.segments, t.text,
                       ts_rank(t.text_fts, websearch_to_tsquery('english', %s)) AS rank
                FROM transcripts t
                JOIN videos v ON v.id = t.video_id
                WHERE t.text_fts @@ websearch_to_tsquery('english', %s)
                  AND t.song_lyrics_ratio < 0.2
                  AND t.segments IS NOT NULL
                  AND v.view_count >= %s
                  AND (%s IS NULL OR v.author ILIKE %s)
                ORDER BY v.view_count DESC NULLS LAST
                LIMIT %s
            """, (query, query, args.min_views, args.creator, args.creator,
                  args.per_line * 4))

            terms = [w for w in words] + [p for p in phrases]
            picked = 0
            print(f"[{section}] {line}")
            for r in cur.fetchall():
                hit = best_segment(r['segments'], terms)
                if not hit:
                    continue
                seg, score = hit
                if score < args.min_overlap:
                    continue
                start, end = float(seg.get('start', 0)), float(seg.get('end', 0))
                rows_out.append({
                    'section': section,
                    'lyric': line,
                    'creator': r['author'],
                    'url': r['url'],
                    'views': r['view_count'],
                    'clip_start_s': round(start, 2),
                    'clip_end_s': round(end, 2),
                    'clip_len_s': round(end - start, 2),
                    'match_score': score,
                    'spoken_text': (seg.get('text') or '').strip(),
                })
                print(f"    @{(r['author'] or '?'):22} {start:6.1f}-{end:6.1f}s  "
                      f"{(r['view_count'] or 0):>9,} views  "
                      f"\"{(seg.get('text') or '').strip()[:70]}\"")
                picked += 1
                if picked >= args.per_line:
                    break
            if not picked:
                print("    (no match)")
            print()

    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else
                               ['section', 'lyric', 'creator', 'url', 'views',
                                'clip_start_s', 'clip_end_s', 'clip_len_s',
                                'match_score', 'spoken_text'])
            w.writeheader(); w.writerows(rows_out)
        print(f"Wrote {len(rows_out)} candidate clip(s) to {args.out}")

    print("\nREMINDER: these are candidates only. Verify Duet/Stitch is enabled")
    print("on each video in the TikTok app, and assemble using TikTok's native")
    print("Duet/Stitch tools so the creator's permission setting is honoured.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('index', help='Build the full-text search index (one-off)')
    m = sub.add_parser('match', help='Match lyrics to transcript segments')
    m.add_argument('--lyrics', required=True, help='Path to lyrics file')
    m.add_argument('--per-line', type=int, default=5, help='Candidates per lyric line')
    m.add_argument('--min-views', type=int, default=0, help='Minimum view count')
    m.add_argument('--min-overlap', type=int, default=1,
                   help='Minimum term overlap for a segment to qualify')
    m.add_argument('--creator',
                   help='Restrict matches to one creator (use when you have '
                        'that creator\'s direct permission)')
    m.add_argument('--out', help='Write candidates to this CSV')
    args = p.parse_args()
    return {'index': cmd_index, 'match': cmd_match}[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
