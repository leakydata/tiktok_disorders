#!/usr/bin/env python3
"""
Audio preprocessing A/B experiment for transcription quality.

Re-transcribes a stratified sample of already-collected videos under several
preprocessing arms and reports quality metrics per arm, so a preprocessing
change can be justified with evidence rather than assumed to help.

Arms:
    baseline    faster-whisper as originally configured (no VAD)
    vad         Silero VAD filtering before decoding
    demucs      Demucs vocal isolation, then VAD  (requires `demucs`)

Nothing is written to `transcripts`; results go to `audio_experiment_results`
so the production corpus is untouched and arms stay comparable.

Usage:
    python scripts/audio_experiment.py init
    python scripts/audio_experiment.py run --sample 60 --arms baseline vad
    python scripts/audio_experiment.py run --sample 30 --arms baseline vad demucs
    python scripts/audio_experiment.py report
"""
import sys
import time
import argparse
import tempfile
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from psycopg2.extras import RealDictCursor

# Imported for its CUDA preload side effect before faster_whisper loads.
import transcriber  # noqa: F401
from transcriber import MEDICAL_VOCABULARY_PROMPT
from database import get_connection
from config import (
    WHISPER_VAD_THRESHOLD, WHISPER_VAD_MIN_SPEECH_MS,
    WHISPER_VAD_MIN_SILENCE_MS, WHISPER_VAD_SPEECH_PAD_MS,
)
from detect_song_lyrics import detect_song_lyrics_heuristic

ARMS = ('baseline', 'vad', 'demucs')

SCHEMA = """
CREATE TABLE IF NOT EXISTS audio_experiment_results (
    id              SERIAL PRIMARY KEY,
    run_label       TEXT NOT NULL,
    arm             TEXT NOT NULL,
    video_id        INTEGER NOT NULL,
    stratum         TEXT NOT NULL,
    baseline_song_ratio  REAL,
    text            TEXT,
    word_count      INTEGER,
    song_ratio      REAL,
    song_confidence REAL,
    elapsed_s       REAL,
    error           TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    UNIQUE (run_label, arm, video_id)
);
CREATE INDEX IF NOT EXISTS idx_audio_exp_run ON audio_experiment_results (run_label, arm);
"""

# Strata over the existing song_lyrics_ratio, so effects can be read separately
# for music-heavy and clean-speech content. A preprocessing step that helps the
# former can still hurt the latter.
STRATA = {
    'music_heavy':  't.song_lyrics_ratio >= 0.5',
    'music_some':   't.song_lyrics_ratio >= 0.2 AND t.song_lyrics_ratio < 0.5',
    'speech_clean': 't.song_lyrics_ratio < 0.2',
}


def cmd_init(args):
    with get_connection() as conn:
        conn.cursor().execute(SCHEMA)
    print("Created audio_experiment_results")
    return 0


def _sample(per_stratum):
    """Draw a reproducible sample per stratum."""
    rows = []
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        for name, predicate in STRATA.items():
            # md5 ordering gives a deterministic pseudo-random draw, so the
            # same sample is reproduced on re-runs without a seed column.
            cur.execute(f"""
                SELECT v.id, v.audio_path, t.song_lyrics_ratio
                FROM transcripts t JOIN videos v ON v.id = t.video_id
                WHERE {predicate}
                  AND v.audio_path IS NOT NULL
                  AND t.song_lyrics_ratio IS NOT NULL
                ORDER BY md5(v.id::text)
                LIMIT %s
            """, (per_stratum,))
            for r in cur.fetchall():
                r['stratum'] = name
                rows.append(r)
    return rows


def _demucs_vocals(audio_path, workdir):
    """Isolate the vocal stem. Returns a path, or None if separation failed."""
    out = Path(workdir) / 'sep'
    subprocess.run(
        [sys.executable, '-m', 'demucs', '--two-stems=vocals',
         '-n', 'htdemucs', '-o', str(out), str(audio_path)],
        check=True, capture_output=True, timeout=900)
    hits = list(out.rglob('vocals.*'))
    return hits[0] if hits else None


def cmd_run(args):
    from faster_whisper import WhisperModel

    arms = args.arms or ['baseline', 'vad']
    if 'demucs' in arms:
        try:
            import demucs  # noqa: F401
        except ImportError:
            print("demucs is not installed; drop it from --arms or "
                  "install with: uv pip install demucs")
            return 1

    sample = _sample(args.sample)
    if not sample:
        print("No videos matched. Has detect_song_lyrics.py been run?")
        return 1

    counts = {}
    for r in sample:
        counts[r['stratum']] = counts.get(r['stratum'], 0) + 1
    print(f"Sample: {len(sample)} videos  " +
          "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"Arms: {', '.join(arms)}   Label: {args.label}\n")

    vad_params = {
        'threshold': WHISPER_VAD_THRESHOLD,
        'min_speech_duration_ms': WHISPER_VAD_MIN_SPEECH_MS,
        'min_silence_duration_ms': WHISPER_VAD_MIN_SILENCE_MS,
        'speech_pad_ms': WHISPER_VAD_SPEECH_PAD_MS,
    }

    model = WhisperModel(args.model, device='cuda', compute_type='float16')

    for arm in arms:
        print(f"===== arm: {arm} =====")
        for i, row in enumerate(sample, 1):
            audio = row['audio_path']
            if not Path(audio).exists():
                _store(args.label, arm, row, error='audio file missing')
                continue

            text, elapsed, err = '', 0.0, None
            tmp = None
            try:
                t0 = time.time()
                src = audio
                if arm == 'demucs':
                    tmp = tempfile.TemporaryDirectory()
                    stem = _demucs_vocals(audio, tmp.name)
                    if stem is None:
                        raise RuntimeError('demucs produced no vocal stem')
                    src = str(stem)

                opts = {}
                if arm in ('vad', 'demucs'):
                    opts = {'vad_filter': True, 'vad_parameters': vad_params}

                segs, _info = model.transcribe(
                    src, language='en', beam_size=5,
                    initial_prompt=MEDICAL_VOCABULARY_PROMPT, **opts)
                text = ' '.join(s.text for s in segs).strip()
                elapsed = time.time() - t0
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"[:400]
            finally:
                if tmp:
                    tmp.cleanup()

            ratio, conf = (detect_song_lyrics_heuristic(text)
                           if text else (None, None))
            _store(args.label, arm, row, text=text, elapsed=elapsed,
                   song_ratio=ratio, song_conf=conf, error=err)

            flag = f" ERROR {err}" if err else ''
            print(f"  [{i}/{len(sample)}] {row['stratum']:12} "
                  f"words={len(text.split()):>4} "
                  f"song={ratio if ratio is None else round(ratio, 2)} "
                  f"{elapsed:>5.1f}s{flag}")
        print()

    print("Done. Report with: python scripts/audio_experiment.py report "
          f"--label {args.label}")
    return 0


def _store(label, arm, row, text='', elapsed=None, song_ratio=None,
           song_conf=None, error=None):
    with get_connection() as conn:
        conn.cursor().execute("""
            INSERT INTO audio_experiment_results
                (run_label, arm, video_id, stratum, baseline_song_ratio,
                 text, word_count, song_ratio, song_confidence, elapsed_s, error)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (run_label, arm, video_id) DO UPDATE SET
                text = EXCLUDED.text, word_count = EXCLUDED.word_count,
                song_ratio = EXCLUDED.song_ratio, elapsed_s = EXCLUDED.elapsed_s,
                error = EXCLUDED.error, created_at = now()
        """, (label, arm, row['id'], row['stratum'], row['song_lyrics_ratio'],
              text, len(text.split()), song_ratio, song_conf, elapsed, error))


def cmd_report(args):
    where, params = ('', [])
    if args.label:
        where, params = ('WHERE run_label = %s', [args.label])

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(f"""
            SELECT stratum, arm,
                   count(*) AS n,
                   count(*) FILTER (WHERE error IS NOT NULL) AS errors,
                   round(avg(word_count)::numeric, 1)  AS avg_words,
                   round(avg(song_ratio)::numeric, 3)  AS avg_song_ratio,
                   count(*) FILTER (WHERE word_count = 0) AS empty,
                   round(avg(elapsed_s)::numeric, 2)   AS avg_s
            FROM audio_experiment_results {where}
            GROUP BY stratum, arm ORDER BY stratum, arm
        """, params)
        rows = cur.fetchall()

    if not rows:
        print("No results. Run the experiment first.")
        return 0

    print(f"{'stratum':13} {'arm':9} {'n':>4} {'err':>4} {'words':>7} "
          f"{'song':>7} {'empty':>6} {'sec':>7}")
    print("-" * 62)
    last = None
    for r in rows:
        if last and r['stratum'] != last:
            print()
        last = r['stratum']
        print(f"{r['stratum']:13} {r['arm']:9} {r['n']:>4} {r['errors']:>4} "
              f"{str(r['avg_words']):>7} {str(r['avg_song_ratio']):>7} "
              f"{r['empty']:>6} {str(r['avg_s']):>7}")

    print("\nReading this table:")
    print("  song  - lower is better; it is the share of output that looks")
    print("          like song lyrics rather than speech.")
    print("  empty - for music_heavy this is a WIN (correctly rejected music);")
    print("          for speech_clean it is a LOSS (real speech was dropped).")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    sub.add_parser('init', help='Create the results table')

    r = sub.add_parser('run', help='Run the experiment')
    r.add_argument('--sample', type=int, default=20,
                   help='Videos per stratum (default: 20)')
    r.add_argument('--arms', nargs='+', choices=ARMS,
                   help='Arms to run (default: baseline vad)')
    r.add_argument('--model', default='large-v3')
    r.add_argument('--label', default='exp1', help='Run label')

    rep = sub.add_parser('report', help='Summarize results')
    rep.add_argument('--label')

    args = p.parse_args()
    return {'init': cmd_init, 'run': cmd_run, 'report': cmd_report}[args.command](args)


if __name__ == '__main__':
    sys.exit(main() or 0)
