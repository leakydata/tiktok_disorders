#!/usr/bin/env python3
"""
Cut video clips from a lyric match CSV, ready for assembly against a song.

Reads the output of semantic_match.py, downloads each source video, and cuts
the matched moment with ffmpeg. Clip windows are adjusted rather than used
verbatim: Whisper segment boundaries are speech-detection artefacts, not edit
points, so they run anywhere from under a second to 17s. Short segments are
padded outward and long ones trimmed inward to land in a musically usable
range.

Permissions are enforced per video, not per creator. With no --creators-file
or --creator, the allowed set is read from videos.duet_enabled /
stitch_enabled / download_enabled as recorded by check_permissions.py, and
anything unchecked stops the run. --require-download additionally excludes
videos whose creator disabled downloads while still allowing duet/stitch.

Usage:
    # per-video permissions from the database (run check_permissions.py first)
    python scripts/make_clips.py --csv data/lyrics/editlist_full.csv \\
        --min-similarity 0.70 --out-dir data/clips

    # or an explicit allowlist, e.g. a creator who gave direct permission
    python scripts/make_clips.py --csv ... --creator chronically.roxii
"""
import sys
import csv
import json
import argparse
import shutil
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Vertical 1080x1920 so clips from different creators concatenate without
# resolution mismatches. Sources vary; ffmpeg concat requires uniform streams.
TARGET_W, TARGET_H = 1080, 1920
FPS = 30


def adjust_window(start, end, min_len, max_len, lead, tail):
    """Turn a Whisper segment boundary into an edit point.

    Segment starts tend to clip the first phoneme and ends tend to cut a
    trailing breath, so a little is taken before and after. Then the window is
    grown or shrunk into [min_len, max_len]: grown symmetrically so the phrase
    stays centred, shrunk from the end so the phrase's opening is kept (the
    matched words are usually near the start).
    """
    s = max(0.0, start - lead)
    e = end + tail
    dur = e - s

    if dur < min_len:                      # too short: grow both ways
        grow = (min_len - dur) / 2.0
        s = max(0.0, s - grow)
        e = s + min_len
    elif dur > max_len:                    # too long: keep the opening
        e = s + max_len

    return round(s, 3), round(e - s, 3)    # (seek, duration)


def video_permissions(urls):
    """Look up recorded reuse permissions for a set of video URLs."""
    from database import get_connection as _gc
    from psycopg2.extras import RealDictCursor as _RDC
    out = {}
    with _gc() as conn:
        cur = conn.cursor(cursor_factory=_RDC)
        cur.execute("""SELECT url, duet_enabled, stitch_enabled, download_enabled
                       FROM videos
                       WHERE url = ANY(%s) AND permissions_checked_at IS NOT NULL""",
                    (list(set(urls)),))
        for r in cur.fetchall():
            out[r['url']] = {'duet': r['duet_enabled'],
                             'stitch': r['stitch_enabled'],
                             'download': r['download_enabled']}
    return out


def video_topicality(urls):
    """Per-video popularity and on-topic signal.

    The extraction pipeline already records symptoms and claimed diagnoses per
    video. A video with no extracted symptoms matched the lyric on phrasing
    alone -- a mortician, a podcast, a clinician explaining anatomy -- rather
    than someone describing their own illness. Symptom count is therefore the
    usable on-topic filter, and it is a property of the video's content, not a
    judgement about the person.
    """
    from database import get_connection as _gc
    from psycopg2.extras import RealDictCursor as _RDC
    out = {}
    with _gc() as conn:
        cur = conn.cursor(cursor_factory=_RDC)
        cur.execute("""
            SELECT v.url, v.view_count,
                   (SELECT count(*) FROM symptoms s WHERE s.video_id = v.id)
                       AS n_symptoms,
                   (SELECT count(*) FROM claimed_diagnoses d WHERE d.video_id = v.id)
                       AS n_dx
            FROM videos v WHERE v.url = ANY(%s)
        """, (list(set(urls)),))
        for r in cur.fetchall():
            out[r['url']] = {'views': r['view_count'] or 0,
                             'symptoms': r['n_symptoms'],
                             'dx': r['n_dx']}
    return out


def load_permitted(path):
    if not path:
        return None
    handles = set()
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        h = line.strip().lstrip('@')
        if h and not h.startswith('#'):
            handles.add(h.lower())
    return handles


def download_video(url, dest_dir):
    """Fetch the source video (not audio) once, cached by video id."""
    import yt_dlp
    vid = url.rstrip('/').split('/')[-1].split('?')[0]
    cached = list(dest_dir.glob(f"{vid}.*"))
    if cached:
        return cached[0]
    opts = {
        'quiet': True, 'no_warnings': True,
        'format': 'mp4/best',
        'outtmpl': str(dest_dir / f'{vid}.%(ext)s'),
    }
    with yt_dlp.YoutubeDL(opts) as y:
        y.download([url])
    hits = list(dest_dir.glob(f"{vid}.*"))
    return hits[0] if hits else None


def cut_clip(src, seek, dur, out_path):
    """Cut and normalise one clip.

    Re-encodes rather than stream-copying: a stream copy snaps to the nearest
    keyframe, which on short clips can shift the cut by a second or more and
    lose the matched words entirely.
    """
    vf = (f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=decrease,"
          f"pad={TARGET_W}:{TARGET_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
          f"fps={FPS},setsar=1")
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-ss', str(seek), '-i', str(src), '-t', str(dur),
        '-vf', vf,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '128k', '-ar', '48000', '-ac', '2',
        str(out_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or '')[-300:])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True, help='Match CSV from semantic_match.py')
    p.add_argument('--out-dir', default='data/clips')
    p.add_argument('--creators-file', help='One permitted handle per line')
    p.add_argument('--creator', help='Single permitted handle')
    p.add_argument('--min-len', type=float, default=1.6)
    p.add_argument('--max-len', type=float, default=4.0)
    p.add_argument('--lead', type=float, default=0.30,
                   help='Seconds taken before the segment start (default 0.30)')
    p.add_argument('--tail', type=float, default=0.20,
                   help='Seconds taken after the segment end (default 0.20)')
    p.add_argument('--min-similarity', type=float, default=0.0)
    p.add_argument('--best-per-line', action='store_true',
                   help='Keep only the strongest surviving candidate per lyric '
                        'line, so excluded clips are backfilled automatically')
    p.add_argument('--min-views', type=int, default=0,
                   help='Drop clips from videos below this view count')
    p.add_argument('--min-symptoms', type=int, default=0,
                   help='Require the video to have at least N extracted '
                        'symptoms, i.e. someone actually describing illness '
                        'rather than a phrasing coincidence')
    p.add_argument('--also-allow', metavar='FILE',
                   help='File of handles separately confirmed, one per line. '
                        'Overrides unreadable or missing permissions.')
    p.add_argument('--also-allow-handle', action='append', metavar='HANDLE',
                   help='Handle separately confirmed (repeatable)')
    p.add_argument('--require-download', action='store_true',
                   help='Also require the creator to have left downloads '
                        'enabled, not just duet/stitch')
    p.add_argument('--limit', type=int)
    p.add_argument('--dry-run', action='store_true',
                   help='Show the planned cuts without downloading')
    args = p.parse_args()

    if not shutil.which('ffmpeg'):
        print("ffmpeg not found on PATH"); return 1

    permitted = load_permitted(args.creators_file)
    if args.creator:
        permitted = (permitted or set()) | {args.creator.lower().lstrip('@')}

    all_rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8'))
                if float(r.get('similarity') or 0) >= args.min_similarity]

    # Popularity and on-topic filters, applied before the permission pass so
    # the exclusion reporting stays readable.
    if args.min_views or args.min_symptoms:
        topic = video_topicality([r['url'] for r in all_rows if r.get('url')])
        kept, dropped = [], []
        for r in all_rows:
            t = topic.get(r['url'], {'views': 0, 'symptoms': 0, 'dx': 0})
            if args.min_views and t['views'] < args.min_views:
                dropped.append((r, f"only {t['views']:,} views")); continue
            if args.min_symptoms and t['symptoms'] < args.min_symptoms:
                dropped.append((r, f"{t['symptoms']} symptoms - off topic")); continue
            kept.append(r)
        if dropped:
            print(f"Dropped {len(dropped)} clip(s) on popularity/topic:")
            for r, why in dropped[:8]:
                print(f"    @{r['creator']:24} {why}")
            if len(dropped) > 8:
                print(f"    ... and {len(dropped)-8} more")
            print()
        all_rows = kept

    if permitted is not None:
        rows = [r for r in all_rows
                if (r.get('creator') or '').lower().lstrip('@') in permitted]
    else:
        # Derive the allowed set from the per-video settings recorded by
        # check_permissions.py. This is stricter than a creator-level list:
        # permissions are per video, so one creator can allow reuse on some
        # uploads and not others.
        urls = [r['url'] for r in all_rows if r.get('url')]
        perms = video_permissions(urls)
        unchecked = [u for u in urls if u not in perms]
        if unchecked:
            print(f"{len(unchecked)} of {len(set(urls))} video(s) have no recorded "
                  f"permissions. Run first:")
            print(f"  uv run scripts/check_permissions.py check --from-csv {args.csv}")
            return 1
        # Creators separately confirmed by the user. Needed because some pages
        # return no video data at all (statusCode 10204 -- author private,
        # suspended or region-locked). That is unreadable, not denied, so it
        # must not be treated as a permission decision either way.
        also = load_permitted(args.also_allow) or set()
        if args.also_allow_handle:
            also |= {h.lower().lstrip('@') for h in args.also_allow_handle}

        rows, excluded = [], []
        for r in all_rows:
            p = perms.get(r['url'], {})
            handle = (r.get('creator') or '').lower().lstrip('@')
            reuse = bool(p.get('duet') or p.get('stitch'))
            dl = bool(p.get('download'))
            unreadable = p.get('duet') is None and p.get('stitch') is None

            if handle in also:
                rows.append(r); continue          # explicit confirmation wins
            if unreadable:
                excluded.append((r, 'permissions unreadable - verify in app'))
                continue
            if not reuse:
                excluded.append((r, 'duet+stitch disabled')); continue
            if args.require_download and not dl:
                excluded.append((r, 'download disabled')); continue
            rows.append(r)
        if excluded:
            print(f"Excluded {len(excluded)} clip(s) on permissions:")
            for r, why in excluded[:8]:
                print(f"    @{r['creator']:22} {why}")
            if len(excluded) > 8:
                print(f"    ... and {len(excluded)-8} more")
            print()

    # Song order, strongest match first within each lyric line.
    seen_order, order_idx = {}, 0
    for r in csv.DictReader(open(args.csv, encoding='utf-8')):
        key = (r['section'], r['lyric'])
        if key not in seen_order:
            seen_order[key] = order_idx; order_idx += 1
    rows.sort(key=lambda r: (seen_order.get((r['section'], r['lyric']), 999),
                             -float(r.get('similarity') or 0)))

    # One clip per lyric line, strongest surviving candidate. Because the
    # filters above run before this, a line whose best match was excluded on
    # permissions, popularity or topic automatically falls through to its next
    # candidate rather than going empty -- which is what --per-line spares are
    # for.
    if args.best_per_line:
        picked, by_line = [], set()
        for r in rows:
            key = (r['section'], r['lyric'])
            if key in by_line:
                continue
            by_line.add(key); picked.append(r)
        replaced = len(rows) - len(picked)
        print(f"Selected {len(picked)} clip(s), one per lyric line "
              f"({replaced} lower-ranked alternate(s) held in reserve)\n")
        rows = picked

    if args.limit:
        rows = rows[:args.limit]

    if not rows:
        print("No rows matched the permitted creators / similarity floor.")
        return 1

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = out_dir / '_sources'; src_dir.mkdir(exist_ok=True)

    print(f"{len(rows)} clip(s) planned  |  window {args.min_len}-{args.max_len}s  "
          f"|  lead {args.lead}s tail {args.tail}s\n")

    manifest, attribution, made, failed = [], [], 0, 0
    for i, r in enumerate(rows, 1):
        seek, dur = adjust_window(float(r['clip_start_s']), float(r['clip_end_s']),
                                  args.min_len, args.max_len, args.lead, args.tail)
        name = f"{i:03d}_{r['creator']}_{seek:.1f}s.mp4".replace('/', '_')
        dest = out_dir / name
        orig = float(r['clip_end_s']) - float(r['clip_start_s'])

        print(f"[{i}/{len(rows)}] {r['section']:14} \"{r['lyric'][:34]}\"")
        print(f"    @{r['creator']}  {orig:.1f}s -> {dur:.1f}s @ {seek:.1f}s  "
              f"sim={r.get('similarity')}")

        if args.dry_run:
            continue
        try:
            src = download_video(r['url'], src_dir)
            if not src:
                raise RuntimeError('download produced no file')
            cut_clip(src, seek, dur, dest)
            manifest.append(dest.name)
            attribution.append({'clip': dest.name, 'creator': r['creator'],
                                'url': r['url'], 'lyric': r['lyric'],
                                'section': r['section'],
                                'spoken_text': r.get('spoken_text', '')})
            made += 1
        except Exception as ex:
            failed += 1
            print(f"    FAILED: {type(ex).__name__} {str(ex)[:140]}")

    if args.dry_run:
        print("\nDry run - nothing downloaded or cut.")
        return 0

    # concat demuxer list, in song order
    (out_dir / 'concat.txt').write_text(
        ''.join(f"file '{n}'\n" for n in manifest), encoding='utf-8')
    (out_dir / 'attribution.json').write_text(
        json.dumps(attribution, indent=2), encoding='utf-8')

    print(f"\nCut {made} clip(s), {failed} failed -> {out_dir}")
    print(f"  concat.txt        ffmpeg concat list, in song order")
    print(f"  attribution.json  creator + source URL per clip")
    print("\nAssemble silent video, then lay the song over it:")
    print(f"  ffmpeg -f concat -safe 0 -i {out_dir}/concat.txt -c copy "
          f"{out_dir}/assembled.mp4")
    print(f"  ffmpeg -i {out_dir}/assembled.mp4 -i song.mp3 -map 0:v -map 1:a "
          f"-shortest -c:v copy -c:a aac {out_dir}/final.mp4")
    print("\nCredit every creator in the post description - attribution.json "
          "has the list.")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
