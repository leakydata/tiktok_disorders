#!/usr/bin/env python3
"""
Extract symptoms from transcripts using Claude or Ollama.

Usage:
    python scripts/extract_symptoms.py --video-id 1
    python scripts/extract_symptoms.py --all --parallel
    python scripts/extract_symptoms.py --all --min-confidence 0.7
    python scripts/extract_symptoms.py --video-id 1 --provider ollama --model gpt-oss:20b
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from extractor import SymptomExtractor
from database import get_connection


def main():
    parser = argparse.ArgumentParser(description='Extract symptoms from transcripts')
    parser.add_argument('--video-id', type=int, help='Video database ID to process')
    parser.add_argument('--all', action='store_true', help='Process all transcribed videos')
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum confidence score to save (default: 0.6)')
    parser.add_argument('--max-workers', type=int, default=10,
                       help='Max parallel workers (default: 10)')
    parser.add_argument('--provider', choices=['anthropic', 'ollama'],
                       help='Model provider (default: from EXTRACTOR_PROVIDER)')
    parser.add_argument('--model',
                       help='Model name (default: provider-specific)')
    parser.add_argument('--ollama-url',
                       help='Ollama base URL (default: from OLLAMA_URL)')

    args = parser.parse_args()

    if not args.video_id and not args.all:
        parser.error("Either --video-id or --all is required")

    # Initialize extractor
    extractor = SymptomExtractor(
        max_workers=args.max_workers,
        provider=args.provider,
        model=args.model,
        ollama_url=args.ollama_url,
    )

    try:
        if args.video_id:
            # Single video
            result = extractor.extract_symptoms(args.video_id, args.min_confidence)
            if result.get('success'):
                print(f"\n✓ Extraction complete!")
                print(f"  Symptoms found: {result['symptoms_found']}")
                print(f"  Symptoms saved: {result['symptoms_saved']}")
            else:
                print(f"✗ Extraction failed: {result.get('error')}")

        else:
            # All videos with transcripts
            with get_connection() as conn:
                cur = conn.cursor()
                # Get videos that have transcripts but no symptoms
                cur.execute("""
                    SELECT DISTINCT t.video_id
                    FROM transcripts t
                    LEFT JOIN symptoms s ON t.video_id = s.video_id
                    WHERE s.id IS NULL
                """)
                video_ids = [row[0] for row in cur.fetchall()]

            if not video_ids:
                print("✓ All transcripts already processed!")
                return

            print(f"Found {len(video_ids)} videos to process")
            if args.parallel:
                print(f"Using parallel processing with {args.max_workers} workers")

            results = extractor.extract_batch(
                video_ids,
                min_confidence=args.min_confidence,
                parallel=args.parallel
            )

            success = sum(1 for r in results if r.get('success'))
            total_symptoms = sum(r.get('symptoms_saved', 0) for r in results if r.get('success'))

            print(f"\n✓ Processed {success}/{len(video_ids)} videos")
            print(f"  Total symptoms extracted: {total_symptoms}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
