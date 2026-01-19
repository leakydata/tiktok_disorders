#!/usr/bin/env python3
"""
Backfill database rows from transcript JSON files.

Usage:
    python scripts/backfill_transcripts.py --dir data/transcripts
    python scripts/backfill_transcripts.py --dir data/transcripts --extract --provider ollama --model gpt-oss:20b
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRANSCRIPT_DIR, MIN_CONFIDENCE_SCORE
from database import insert_video, insert_transcript, get_video_by_url, get_transcript, get_connection
from extractor import SymptomExtractor


def _load_transcript(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_video(transcript_data: Dict[str, Any], source_id: str) -> int:
    url = f"local:transcript:{source_id}"
    existing = get_video_by_url(url)
    if existing:
        return existing["id"]

    metadata = {
        "title": transcript_data.get("title") or source_id,
        "author": transcript_data.get("author"),
        "duration": transcript_data.get("duration"),
        "upload_date": transcript_data.get("upload_date"),
        "tags": transcript_data.get("tags", []),
        "audio_path": transcript_data.get("audio_path"),
        "audio_size_bytes": transcript_data.get("audio_size_bytes"),
    }
    return insert_video(
        url=url,
        platform="local",
        video_id=source_id,
        metadata=metadata,
    )


def _get_videos_missing_symptoms() -> List[int]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT t.video_id
            FROM transcripts t
            LEFT JOIN symptoms s ON t.video_id = s.video_id
            WHERE s.id IS NULL
        """)
        return [row[0] for row in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill DB from transcript JSON files")
    parser.add_argument("--dir", default=str(TRANSCRIPT_DIR),
                        help="Directory containing transcript JSON files")
    parser.add_argument("--extract", action="store_true",
                        help="Run symptom extraction after backfill")
    parser.add_argument("--provider", choices=["anthropic", "ollama"],
                        help="Model provider (default: from EXTRACTOR_PROVIDER)")
    parser.add_argument("--model", help="Model name (default: provider-specific)")
    parser.add_argument("--ollama-url", help="Ollama base URL (default: from OLLAMA_URL)")
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE_SCORE,
                        help="Minimum confidence score to save")
    parser.add_argument("--parallel", action="store_true", help="Enable parallel extraction")
    parser.add_argument("--max-workers", type=int, default=10, help="Max parallel workers")

    args = parser.parse_args()

    transcript_dir = Path(args.dir)
    if not transcript_dir.exists():
        print(f"✗ Transcript directory not found: {transcript_dir}")
        return 1

    files = sorted(transcript_dir.glob("*.json"))
    if not files:
        print(f"✗ No transcript JSON files found in: {transcript_dir}")
        return 1

    inserted = 0
    skipped = 0

    for path in files:
        try:
            data = _load_transcript(path)
        except Exception as e:
            print(f"✗ Failed to read {path.name}: {e}")
            continue

        source_id = str(data.get("video_id") or path.stem)
        video_db_id = _ensure_video(data, source_id)

        if get_transcript(video_db_id):
            skipped += 1
            continue

        insert_transcript(
            video_id=video_db_id,
            text=data.get("text", ""),
            language=data.get("language"),
            model_used=data.get("model", "unknown"),
            segments=data.get("segments"),
        )
        inserted += 1

    print(f"✓ Backfill complete: {inserted} inserted, {skipped} skipped")

    if args.extract:
        video_ids = _get_videos_missing_symptoms()
        if not video_ids:
            print("✓ No transcripts pending symptom extraction")
            return 0

        extractor = SymptomExtractor(
            max_workers=args.max_workers,
            provider=args.provider,
            model=args.model,
            ollama_url=args.ollama_url,
        )
        extractor.extract_batch(
            video_ids,
            min_confidence=args.min_confidence,
            parallel=args.parallel,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
