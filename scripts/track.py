#!/usr/bin/env python3
"""
Creator watchlist management for longitudinal tracking.

Maintains a list of TikTok creators to follow over time, with a per-creator
polling cadence. `check` discovers new videos for creators whose next check is
due and appends them to the pending URL file for the pipeline to process.

Usage:
    python scripts/track.py init
    python scripts/track.py seed --top 100 --by symptoms
    python scripts/track.py seed --stratified 10 --min-videos 10 --active-days 90
    python scripts/track.py add chronic.kaleigh --cohort strain_pilot
    python scripts/track.py list --due
    python scripts/track.py check --limit 50
    python scripts/track.py status
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from psycopg2.extras import RealDictCursor

from database import get_connection

# discover.py lives alongside this script and owns all scraping logic.
from discover import (
    discover_user_videos,
    load_existing_urls,
    append_urls_incrementally,
)

MIGRATION = Path(__file__).parent / 'migrations' / '001_tracked_creators.sql'
DEFAULT_URL_FILE = Path(__file__).parent.parent / 'urls.txt'

# Follower thresholds for creator_tier, mirroring
# VideoDownloader._calculate_creator_tier so backfilled rows agree with rows
# written at download time.
#
# KNOWN LIMITATION (verified 2026-07-29): yt-dlp returns None for
# channel_follower_count / uploader_follower_count on TikTok, from both video
# and profile URLs. creator_tier is therefore NULL for all 79,369 collected
# videos and cannot be backfilled through this path. The COALESCE below is
# retained so tiers populate automatically if a future extractor version or an
# alternate source (e.g. tiktokapipy) supplies the field, but as of now it is
# inert. Any STRAIN creator-influence analysis needs another follower source.
TIER_THRESHOLDS = [
    (10_000, 'nano'), (100_000, 'micro'),
    (500_000, 'mid'), (1_000_000, 'macro'),
]

# Seeding criteria -> ORDER BY expression against user_profiles.
SEED_CRITERIA = {
    'videos': 'video_count DESC',
    'symptoms': 'total_symptoms_reported DESC',
    'diagnoses': 'unique_diagnoses_count DESC',
    'recent': 'last_video_date DESC',
}


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Apply the watchlist migration."""
    if not MIGRATION.exists():
        print(f"Migration not found: {MIGRATION}")
        return 1

    sql = MIGRATION.read_text()
    with get_connection() as conn:
        conn.cursor().execute(sql)

    print(f"Applied {MIGRATION.name}")
    print("Created table tracked_creators and view creators_due_for_check")
    return 0


def _require_table(cur):
    cur.execute("SELECT to_regclass('public.tracked_creators') IS NOT NULL AS ok")
    if not cur.fetchone()['ok']:
        print("tracked_creators does not exist. Run: python scripts/track.py init")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------

def _build_filters(args):
    """Shared WHERE clauses for seeding from user_profiles."""
    where = ['username IS NOT NULL']
    params = []

    if args.min_videos:
        where.append('video_count >= %s')
        params.append(args.min_videos)

    if args.active_days:
        where.append("last_video_date >= CURRENT_DATE - (%s * INTERVAL '1 day')")
        params.append(args.active_days)

    if args.condition:
        where.append('primary_condition = ANY(%s)')
        params.append(args.condition)

    return ' AND '.join(where), params


def cmd_seed(args):
    """Populate the watchlist from user_profiles using a stated rule."""
    if not args.top and not args.stratified:
        print("Specify --top N or --stratified N")
        return 1

    where_sql, params = _build_filters(args)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        _require_table(cur)

        if args.stratified:
            # Even allocation across primary_condition so one prolific
            # condition group cannot dominate the cohort.
            order = SEED_CRITERIA[args.by]
            cur.execute(f"""
                WITH ranked AS (
                    SELECT username, primary_condition,
                           ROW_NUMBER() OVER (
                               PARTITION BY primary_condition ORDER BY {order}
                           ) AS rn
                    FROM user_profiles
                    WHERE {where_sql} AND primary_condition IS NOT NULL
                )
                SELECT username, primary_condition FROM ranked WHERE rn <= %s
            """, params + [args.stratified])
            reason = (f'stratified_{args.stratified}_per_condition_by_{args.by}'
                      f'{_filter_suffix(args)}')
        else:
            order = SEED_CRITERIA[args.by]
            cur.execute(f"""
                SELECT username, primary_condition
                FROM user_profiles
                WHERE {where_sql}
                ORDER BY {order} NULLS LAST
                LIMIT %s
            """, params + [args.top])
            reason = f'top_{args.top}_by_{args.by}{_filter_suffix(args)}'

        rows = cur.fetchall()
        if not rows:
            print("No creators matched the filters.")
            return 0

        # ON CONFLICT DO NOTHING: seeding never overwrites a creator that was
        # added manually or assigned to a different cohort.
        inserted = 0
        for row in rows:
            cur.execute("""
                INSERT INTO tracked_creators
                    (username, cohort, added_reason, priority, check_interval_days)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (username) DO NOTHING
            """, (row['username'], args.cohort, reason,
                  args.priority, args.interval))
            inserted += cur.rowcount

    print(f"Matched {len(rows)} creators, added {inserted} new "
          f"({len(rows) - inserted} already tracked)")
    print(f"Selection rule recorded as: {reason}")
    return 0


def _filter_suffix(args):
    bits = []
    if args.min_videos:
        bits.append(f'minvid{args.min_videos}')
    if args.active_days:
        bits.append(f'active{args.active_days}d')
    if args.condition:
        bits.append('cond' + '+'.join(args.condition))
    return ('_' + '_'.join(bits)) if bits else ''


def cmd_add(args):
    """Add a single creator by username."""
    username = args.username.lstrip('@')
    with get_connection() as conn:
        cur = conn.cursor()
        _require_table(conn.cursor(cursor_factory=RealDictCursor))
        cur.execute("""
            INSERT INTO tracked_creators
                (username, cohort, added_reason, priority, check_interval_days, notes)
            VALUES (%s, %s, 'manual', %s, %s, %s)
            ON CONFLICT (username) DO UPDATE SET
                cohort = COALESCE(EXCLUDED.cohort, tracked_creators.cohort),
                is_active = true
        """, (username, args.cohort, args.priority, args.interval, args.notes))
    print(f"Tracking @{username}")
    return 0


def cmd_remove(args):
    """Deactivate a creator without deleting their history."""
    username = args.username.lstrip('@')
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE tracked_creators SET is_active = false WHERE username = %s",
            (username,))
        n = cur.rowcount
    print(f"Deactivated @{username}" if n else f"Not tracked: @{username}")
    return 0


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------

def cmd_list(args):
    """List tracked creators."""
    source = 'creators_due_for_check' if args.due else 'tracked_creators'
    where, params = ('', [])
    if args.cohort:
        where = 'WHERE cohort = %s'
        params = [args.cohort]

    order = '' if args.due else 'ORDER BY priority, username'

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        _require_table(cur)
        cur.execute(
            f"SELECT username, cohort, priority, check_interval_days, is_active, "
            f"last_checked_at, new_videos_found, consecutive_empty "
            f"FROM {source} {where} {order} LIMIT %s",
            params + [args.limit])
        rows = cur.fetchall()

    if not rows:
        print("No creators due." if args.due else "Watchlist is empty.")
        return 0

    print(f"{'username':28} {'cohort':16} {'pri':>3} {'every':>6} "
          f"{'last check':12} {'new':>5} {'empty':>5}")
    print("-" * 84)
    for r in rows:
        last = r['last_checked_at'].strftime('%Y-%m-%d') if r['last_checked_at'] else 'never'
        print(f"{r['username'][:28]:28} {(r['cohort'] or '-')[:16]:16} "
              f"{r['priority']:>3} {str(r['check_interval_days'])+'d':>6} "
              f"{last:12} {r['new_videos_found']:>5} {r['consecutive_empty']:>5}")
    print(f"\n{len(rows)} creator(s)")
    return 0


def cmd_status(args):
    """Summarize watchlist state."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        _require_table(cur)
        cur.execute("""
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE is_active) AS active,
                   count(*) FILTER (WHERE last_checked_at IS NULL AND is_active) AS never,
                   sum(new_videos_found) AS found,
                   count(*) FILTER (WHERE consecutive_empty >= 3) AS dormant
            FROM tracked_creators
        """)
        s = cur.fetchone()
        cur.execute("SELECT count(*) AS due FROM creators_due_for_check")
        due = cur.fetchone()['due']
        cur.execute("""
            SELECT cohort, count(*) AS n FROM tracked_creators
            WHERE is_active GROUP BY cohort ORDER BY n DESC
        """)
        cohorts = cur.fetchall()

    print("Watchlist")
    print(f"  tracked         : {s['total']} ({s['active']} active)")
    print(f"  due for check   : {due}")
    print(f"  never checked   : {s['never']}")
    print(f"  new videos found: {s['found'] or 0}")
    print(f"  dormant (3+ empty checks): {s['dormant']}")
    if cohorts:
        print("\nBy cohort:")
        for c in cohorts:
            print(f"  {(c['cohort'] or '(none)'):24} {c['n']}")
    return 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def cmd_check(args):
    """Discover new videos for creators whose next check is due."""
    url_file = Path(args.output) if args.output else DEFAULT_URL_FILE
    delay_range = (args.min_delay, args.max_delay)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        _require_table(cur)
        sql = "SELECT username, videos_at_last_check FROM creators_due_for_check"
        params = []
        if args.cohort:
            sql += " WHERE cohort = %s"
            params.append(args.cohort)
        if args.limit:
            sql += " LIMIT %s"
            params.append(args.limit)
        cur.execute(sql, params)
        due = cur.fetchall()

    if not due:
        print("No creators due for check.")
        return 0

    print("=" * 60)
    print(f"Checking {len(due)} creator(s)")
    print(f"Output: {url_file} (append)")
    print("=" * 60)

    existing = load_existing_urls(url_file)
    total_new = 0

    for i, row in enumerate(due, 1):
        username = row['username']
        print(f"\n[{i}/{len(due)}] @{username}")

        error = None
        new_count = 0
        found = 0
        try:
            urls = discover_user_videos(
                username, args.max_videos, delay_range,
                args.date_after, None)
            found = len(urls)
            if urls:
                new_count, existing = append_urls_incrementally(
                    urls, url_file, existing, f"@{username}")
        except Exception as exc:
            # One creator failing must not abort the run.
            error = f"{type(exc).__name__}: {exc}"[:500]
            print(f"    error: {error}")

        total_new += new_count

        # Persist per-creator so an interrupted run keeps its progress.
        with get_connection() as conn:
            conn.cursor().execute("""
                UPDATE tracked_creators SET
                    last_checked_at      = now(),
                    videos_at_last_check = %s,
                    new_videos_found     = new_videos_found + %s,
                    last_new_video_at    = CASE WHEN %s > 0 THEN now()
                                                ELSE last_new_video_at END,
                    consecutive_empty    = CASE WHEN %s > 0 THEN 0
                                                ELSE consecutive_empty + 1 END,
                    last_error           = %s
                WHERE username = %s
            """, (found, new_count, new_count, new_count, error, username))

    print("\n" + "=" * 60)
    print(f"Checked {len(due)} creator(s), {total_new} new URL(s) -> {url_file}")
    if total_new:
        print(f"\nNext: uv run pipeline.py run --urls-file {url_file} "
              f"--tags EDS MCAS POTS")
    return 0


# ---------------------------------------------------------------------------
# Engagement snapshots
# ---------------------------------------------------------------------------

def _creator_tier(follower_count):
    if not follower_count:
        return None
    for threshold, name in TIER_THRESHOLDS:
        if follower_count < threshold:
            return name
    return 'mega'


def _fetch_engagement(url):
    """Fetch current engagement counts for a single video URL."""
    import yt_dlp
    with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True,
                           'extract_flat': False}) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        'view_count': info.get('view_count'),
        'like_count': info.get('like_count'),
        'comment_count': info.get('comment_count'),
        # TikTok reposts are what the schema calls shares.
        'share_count': info.get('repost_count'),
        'follower_count': (info.get('channel_follower_count')
                           or info.get('uploader_follower_count')),
    }


def cmd_snapshot(args):
    """Re-poll engagement counts for tracked videos into engagement_snapshots.

    Each run appends a new row per video rather than overwriting, so repeated
    runs build the time series that diffusion analysis needs.
    """
    import time
    import random

    where = ['v.url IS NOT NULL']
    params = []

    if args.creator:
        where.append('v.author = %s')
        params.append(args.creator.lstrip('@'))
    else:
        # Default to videos by creators on the active watchlist.
        where.append("""v.author IN (
            SELECT username FROM tracked_creators WHERE is_active
        )""")
        if args.cohort:
            where[-1] = """v.author IN (
                SELECT username FROM tracked_creators
                WHERE is_active AND cohort = %s
            )"""
            params.append(args.cohort)

    if args.since_days:
        where.append("v.upload_date >= CURRENT_DATE - (%s * INTERVAL '1 day')")
        params.append(args.since_days)

    # Re-snapshot only if the newest existing snapshot is older than the
    # interval, so a repeated run does not double-sample the same day.
    if args.min_interval_days:
        where.append("""NOT EXISTS (
            SELECT 1 FROM engagement_snapshots es
            WHERE es.video_id = v.id
              AND es.snapshot_at > now() - (%s * INTERVAL '1 day')
        )""")
        params.append(args.min_interval_days)

    sql = f"""SELECT v.id, v.url, v.author, v.view_count AS prev_views
              FROM videos v WHERE {' AND '.join(where)}
              ORDER BY v.upload_date DESC NULLS LAST"""
    if args.limit:
        sql += ' LIMIT %s'
        params.append(args.limit)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        _require_table(cur)
        cur.execute(sql, params)
        videos = cur.fetchall()

    if not videos:
        print("No videos match. Seed the watchlist first, or pass --creator.")
        return 0

    print("=" * 60)
    print(f"Snapshotting {len(videos)} video(s)")
    print("=" * 60)

    taken = failed = 0
    for i, v in enumerate(videos, 1):
        try:
            stats = _fetch_engagement(v['url'])
        except Exception as exc:
            failed += 1
            print(f"[{i}/{len(videos)}] {v['url'][:60]} - {type(exc).__name__}")
            time.sleep(random.uniform(args.min_delay, args.max_delay))
            continue

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO engagement_snapshots
                    (video_id, view_count, like_count, comment_count,
                     share_count, snapshot_at)
                VALUES (%s, %s, %s, %s, %s, now())
            """, (v['id'], stats['view_count'], stats['like_count'],
                  stats['comment_count'], stats['share_count']))

            # Keep the denormalized counts on videos current too, and
            # backfill creator_tier, which is NULL for historically
            # collected rows.
            cur.execute("""
                UPDATE videos SET
                    view_count = COALESCE(%s, view_count),
                    like_count = COALESCE(%s, like_count),
                    comment_count = COALESCE(%s, comment_count),
                    share_count = COALESCE(%s, share_count),
                    author_follower_count = COALESCE(%s, author_follower_count),
                    creator_tier = COALESCE(%s, creator_tier),
                    updated_at = now()
                WHERE id = %s
            """, (stats['view_count'], stats['like_count'],
                  stats['comment_count'], stats['share_count'],
                  stats['follower_count'],
                  _creator_tier(stats['follower_count']), v['id']))

        taken += 1
        delta = ''
        if v['prev_views'] and stats['view_count']:
            diff = stats['view_count'] - v['prev_views']
            delta = f"  ({diff:+,} views)"
        print(f"[{i}/{len(videos)}] @{v['author']} "
              f"{(stats['view_count'] or 0):,} views{delta}")

        time.sleep(random.uniform(args.min_delay, args.max_delay))

    print("\n" + "=" * 60)
    print(f"Recorded {taken} snapshot(s), {failed} failed")
    return 0


def cmd_trend(args):
    """Show engagement growth from the accumulated snapshots."""
    where, params = ['true'], []
    if args.creator:
        where.append('v.author = %s')
        params.append(args.creator.lstrip('@'))
    if args.cohort:
        where.append("""v.author IN (
            SELECT username FROM tracked_creators WHERE cohort = %s
        )""")
        params.append(args.cohort)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"""
            WITH bounds AS (
                SELECT es.video_id,
                       min(es.snapshot_at) AS first_at,
                       max(es.snapshot_at) AS last_at,
                       count(*) AS n
                FROM engagement_snapshots es
                GROUP BY es.video_id
                HAVING count(*) >= 2
            )
            SELECT v.author, v.title, b.n AS snapshots,
                   f.view_count AS first_views, l.view_count AS last_views,
                   l.view_count - f.view_count AS growth,
                   EXTRACT(EPOCH FROM (b.last_at - b.first_at))/86400 AS days
            FROM bounds b
            JOIN videos v ON v.id = b.video_id
            JOIN engagement_snapshots f
              ON f.video_id = b.video_id AND f.snapshot_at = b.first_at
            JOIN engagement_snapshots l
              ON l.video_id = b.video_id AND l.snapshot_at = b.last_at
            WHERE {' AND '.join(where)}
            ORDER BY growth DESC NULLS LAST
            LIMIT %s
        """, params + [args.limit])
        rows = cur.fetchall()

    if not rows:
        print("No videos have 2+ snapshots yet. Run `snapshot` at least twice,")
        print("separated by enough time for counts to move.")
        return 0

    print(f"{'creator':24} {'snaps':>5} {'first':>12} {'latest':>12} "
          f"{'growth':>12} {'days':>6}")
    print("-" * 78)
    for r in rows:
        print(f"{(r['author'] or '?')[:24]:24} {r['snapshots']:>5} "
              f"{(r['first_views'] or 0):>12,} {(r['last_views'] or 0):>12,} "
              f"{(r['growth'] or 0):>+12,} {(r['days'] or 0):>6.1f}")
    return 0


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Creator watchlist for longitudinal tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Usage:')[1] if 'Usage:' in __doc__ else None)
    sub = parser.add_subparsers(dest='command', required=True)

    sub.add_parser('init', help='Create the watchlist table and view')

    p = sub.add_parser('seed', help='Populate from user_profiles')
    p.add_argument('--top', type=int, help='Take the top N creators overall')
    p.add_argument('--stratified', type=int, metavar='N',
                   help='Take N creators per primary_condition')
    p.add_argument('--by', choices=sorted(SEED_CRITERIA), default='videos',
                   help='Ranking criterion (default: videos)')
    p.add_argument('--min-videos', type=int, help='Require at least N videos')
    p.add_argument('--active-days', type=int,
                   help='Require a post within the last N days')
    p.add_argument('--condition', nargs='+',
                   help='Restrict to these primary_condition values')
    p.add_argument('--cohort', help='Cohort label to assign')
    p.add_argument('--priority', type=int, default=3)
    p.add_argument('--interval', type=int, default=7,
                   help='Check interval in days (default: 7)')

    p = sub.add_parser('add', help='Track a single creator')
    p.add_argument('username')
    p.add_argument('--cohort')
    p.add_argument('--priority', type=int, default=3)
    p.add_argument('--interval', type=int, default=7)
    p.add_argument('--notes')

    p = sub.add_parser('remove', help='Deactivate a creator')
    p.add_argument('username')

    p = sub.add_parser('list', help='List tracked creators')
    p.add_argument('--due', action='store_true', help='Only those due for check')
    p.add_argument('--cohort')
    p.add_argument('--limit', type=int, default=50)

    sub.add_parser('status', help='Summarize watchlist state')

    p = sub.add_parser('check', help='Discover new videos for due creators')
    p.add_argument('--limit', type=int, help='Max creators this run')
    p.add_argument('--cohort')
    p.add_argument('--max-videos', type=int,
                   help='Max videos per creator (default: all)')
    p.add_argument('--date-after', help='Only videos after YYYYMMDD')
    p.add_argument('--output', help=f'URL file (default: {DEFAULT_URL_FILE.name})')
    p.add_argument('--min-delay', type=float, default=2.0)
    p.add_argument('--max-delay', type=float, default=5.0)

    p = sub.add_parser('snapshot',
                       help='Record current engagement counts for tracked videos')
    p.add_argument('--creator', help='Single creator (default: whole watchlist)')
    p.add_argument('--cohort')
    p.add_argument('--limit', type=int, help='Max videos this run')
    p.add_argument('--since-days', type=int,
                   help='Only videos uploaded within the last N days')
    p.add_argument('--min-interval-days', type=int, default=1,
                   help='Skip videos snapshotted more recently than this '
                        '(default: 1)')
    p.add_argument('--min-delay', type=float, default=2.0)
    p.add_argument('--max-delay', type=float, default=5.0)

    p = sub.add_parser('trend', help='Show engagement growth from snapshots')
    p.add_argument('--creator')
    p.add_argument('--cohort')
    p.add_argument('--limit', type=int, default=25)

    args = parser.parse_args()
    return {
        'init': cmd_init, 'seed': cmd_seed, 'add': cmd_add,
        'remove': cmd_remove, 'list': cmd_list, 'status': cmd_status,
        'check': cmd_check, 'snapshot': cmd_snapshot, 'trend': cmd_trend,
    }[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
