#!/usr/bin/env python3
"""
Transcribe audio files using Whisper.

Usage:
    python scripts/transcribe.py --video-id 1
    python scripts/transcribe.py --video-id 1 --model large-v3
    python scripts/transcribe.py --all --model medium
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from transcriber import AudioTranscriber
from database import get_connection


def main():
    parser = argparse.ArgumentParser(description='Transcribe audio files')
    parser.add_argument('--video-id', type=int, help='Video database ID to transcribe')
    parser.add_argument('--all', action='store_true', help='Transcribe all videos')
    parser.add_argument('--model', default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model to use (default: large-v3 for RTX 4090)')
    parser.add_argument('--language', help='Force language (e.g., en)')

    args = parser.parse_args()

    if not args.video_id and not args.all:
        parser.error("Either --video-id or --all is required")

    # Initialize transcriber
    transcriber = AudioTranscriber(model_size=args.model)

    try:
        if args.video_id:
            # Single video
            result = transcriber.transcribe(args.video_id, language=args.language)
            print(f"\n✓ Transcription complete!")
            print(f"  Transcript ID: {result['transcript_id']}")
            print(f"  Language: {result['language']}")
            print(f"  Word count: {result['word_count']}")
            print(f"\nFirst 300 characters:")
            print(result['text'][:300] + "...")

        else:
            # All videos
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT v.id FROM videos v
                    LEFT JOIN transcripts t ON v.id = t.video_id
                    WHERE t.id IS NULL
                """)
                video_ids = [row[0] for row in cur.fetchall()]

            if not video_ids:
                print("✓ All videos already transcribed!")
                return

            print(f"Found {len(video_ids)} videos to transcribe")
            results = transcriber.transcribe_batch(video_ids, language=args.language)

            success = sum(1 for r in results if r['success'])
            print(f"\n✓ Transcribed {success}/{len(video_ids)} videos")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
