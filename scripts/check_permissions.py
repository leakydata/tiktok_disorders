#!/usr/bin/env python3
"""
Fetch and record each video's reuse permissions from TikTok.

`duetEnabled` / `stitchEnabled` / `downloadSetting` are per-video creator
settings. They are NOT exposed by yt-dlp and do not affect whether a video
appears in search or on a profile, so a scraped corpus contains a mix of
permissioned and non-permissioned videos with no way to tell them apart. This
reads the settings from the page's rehydration payload and stores them, turning
"assume reuse is allowed" into a filterable fact.

Populates: videos.duet_enabled, videos.stitch_enabled, videos.download_enabled,
videos.permissions_checked_at

Usage:
    python scripts/check_permissions.py init
    python scripts/check_permissions.py check --creator chronically.roxii
    python scripts/check_permissions.py check --from-csv data/lyrics/editlist.csv
    python scripts/check_permissions.py report
"""
import sys
import csv
import json
import re
import time
import random
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from psycopg2.extras import RealDictCursor
from database import get_connection

SCHEMA = """
ALTER TABLE videos ADD COLUMN IF NOT EXISTS duet_enabled BOOLEAN;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS stitch_enabled BOOLEAN;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS download_enabled BOOLEAN;
ALTER TABLE videos ADD COLUMN IF NOT EXISTS permissions_checked_at TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_videos_duet ON videos (duet_enabled, stitch_enabled);
"""

BLOB_RE = re.compile(
    r'__UNIVERSAL_DATA_FOR_REHYDRATION__[^>]*>(\{.*?\})</script>', re.S)


def cmd_init(args):
    with get_connection() as conn:
        conn.cursor().execute(SCHEMA)
    print("Added duet_enabled / stitch_enabled / download_enabled / "
          "permissions_checked_at to videos")
    return 0


def fetch_permissions(url, timeout=40):
    """Return the reuse flags for one video, or None if unreadable.

    Impersonation is required -- without it TikTok blocks the request. This is
    also why curl-cffi must stay below 0.14 for yt-dlp compatibility.
    """
    from curl_cffi import requests
    r = requests.get(url, impersonate='chrome', timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    m = BLOB_RE.search(r.text)
    if not m:
        raise RuntimeError("no rehydration payload (removed or login-walled?)")
    s = json.dumps(json.loads(m.group(1)))

    # No itemStruct means TikTok returned an error payload rather than video
    # data -- commonly statusCode 10204 (author unavailable: private,
    # suspended, deleted or region-locked). That is UNREADABLE, not "reuse
    # denied", and must not be reported as a permission decision.
    if '"itemStruct"' not in s:
        code = re.search(r'"statusCode":\s*(\d+)', s)
        msg = re.search(r'"statusMsg":\s*"([^"]{0,40})', s)
        raise LookupError(
            f"no video data (statusCode={code.group(1) if code else '?'} "
            f"{msg.group(1) if msg else ''})")

    def flag(key):
        hit = re.search(rf'"{key}":\s*(true|false)', s)
        return (hit.group(1) == 'true') if hit else None

    dl = re.search(r'"downloadSetting":\s*(\d+)', s)
    return {
        'duet': flag('duetEnabled'),
        'stitch': flag('stitchEnabled'),
        # downloadSetting 0 = allowed; anything else restricts it.
        'download': (dl.group(1) == '0') if dl else None,
    }


def cmd_check(args):
    where, params = ["v.url IS NOT NULL"], []
    urls = None

    if args.from_csv:
        seen = []
        for r in csv.DictReader(open(args.from_csv, encoding='utf-8')):
            u = r.get('url')
            if u and u not in seen:
                seen.append(u)
        urls = seen
    else:
        if args.creator:
            where.append("v.author ILIKE %s"); params.append(args.creator)
        if not args.recheck:
            where.append("v.permissions_checked_at IS NULL")
        sql = (f"SELECT v.id, v.url, v.author FROM videos v "
               f"WHERE {' AND '.join(where)} "
               f"ORDER BY v.view_count DESC NULLS LAST")
        if args.limit:
            sql += " LIMIT %s"; params.append(args.limit)
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(sql, params)
            rows = cur.fetchall()

    if urls is not None:
        with get_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT id, url, author FROM videos WHERE url = ANY(%s)",
                        (urls,))
            rows = cur.fetchall()
        if args.limit:
            rows = rows[:args.limit]

    if not rows:
        print("Nothing to check.")
        return 0

    print(f"Checking {len(rows)} video(s)\n")
    ok = blocked = failed = unknown = 0
    for i, r in enumerate(rows, 1):
        try:
            p = fetch_permissions(r['url'])
            with get_connection() as conn:
                conn.cursor().execute("""
                    UPDATE videos SET duet_enabled=%s, stitch_enabled=%s,
                        download_enabled=%s, permissions_checked_at=now()
                    WHERE id=%s
                """, (p['duet'], p['stitch'], p['download'], r['id']))
            if p['duet'] is None and p['stitch'] is None:
                verdict = 'UNKNOWN - check manually'
                unknown += 1
            elif p['duet'] or p['stitch']:
                verdict = 'REUSE OK'; ok += 1
            else:
                verdict = 'NO REUSE'; blocked += 1
            print(f"[{i}/{len(rows)}] @{(r['author'] or '?'):22} "
                  f"duet={str(p['duet']):5} stitch={str(p['stitch']):5} "
                  f"download={str(p['download']):5} {verdict}")
        except Exception as ex:
            failed += 1
            print(f"[{i}/{len(rows)}] @{(r['author'] or '?'):22} "
                  f"FAILED {type(ex).__name__}: {str(ex)[:70]}")
        time.sleep(random.uniform(args.min_delay, args.max_delay))

    print(f"\nreuse allowed: {ok}   no reuse: {blocked}   "
          f"unknown: {unknown}   fetch failed: {failed}")
    if unknown or failed:
        print("Unknown/failed are NOT denials - the page had no readable video "
              "data.\nCheck those in the app, then allow them explicitly with "
              "make_clips.py --also-allow.")
    return 0


def cmd_report(args):
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT count(*) AS total,
                   count(permissions_checked_at) AS checked,
                   count(*) FILTER (WHERE duet_enabled) AS duet_ok,
                   count(*) FILTER (WHERE stitch_enabled) AS stitch_ok,
                   count(*) FILTER (WHERE download_enabled) AS download_ok,
                   count(*) FILTER (WHERE permissions_checked_at IS NOT NULL
                                      AND NOT COALESCE(duet_enabled,false)
                                      AND NOT COALESCE(stitch_enabled,false))
                       AS no_reuse
            FROM videos
        """)
        s = cur.fetchone()
    print("Video reuse permissions")
    print(f"  videos total        : {s['total']}")
    print(f"  permissions checked : {s['checked']}")
    print(f"    duet enabled      : {s['duet_ok']}")
    print(f"    stitch enabled    : {s['stitch_ok']}")
    print(f"    download allowed  : {s['download_ok']}")
    print(f"    NO reuse allowed  : {s['no_reuse']}")
    if s['checked'] and s['no_reuse']:
        pct = 100.0 * s['no_reuse'] / s['checked']
        print(f"\n  {pct:.1f}% of checked videos do NOT permit reuse.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('init', help='Add the permission columns')
    c = sub.add_parser('check', help='Fetch permissions for videos')
    c.add_argument('--creator'); c.add_argument('--limit', type=int)
    c.add_argument('--from-csv', help='Check only URLs appearing in this CSV')
    c.add_argument('--recheck', action='store_true',
                   help='Re-check videos already checked')
    c.add_argument('--min-delay', type=float, default=1.5)
    c.add_argument('--max-delay', type=float, default=3.5)
    sub.add_parser('report', help='Summarise permission coverage')
    args = p.parse_args()
    return {'init': cmd_init, 'check': cmd_check, 'report': cmd_report}[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
