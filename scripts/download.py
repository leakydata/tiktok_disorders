#!/usr/bin/env python3
"""
Download videos and extract audio.

Usage:
    python scripts/download.py --url "https://youtube.com/watch?v=..."
    python scripts/download.py --file urls.txt --tags EDS,POTS
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from downloader import VideoDownloader


def main():
    parser = argparse.ArgumentParser(description='Download videos and extract audio')
    parser.add_argument('--url', help='Single video URL to download')
    parser.add_argument('--file', help='File containing URLs (one per line)')
    parser.add_argument('--tags', help='Comma-separated tags', default='')

    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Either --url or --file is required")

    # Parse tags
    tags = [t.strip() for t in args.tags.split(',') if t.strip()] if args.tags else None

    # Initialize downloader
    downloader = VideoDownloader()

    try:
        if args.url:
            # Single URL
            result = downloader.download_audio(args.url, tags)
            print(f"\n✓ Download complete!")
            print(f"  Video ID: {result['video_id']}")
            print(f"  Audio path: {result['audio_path']}")
            if not result.get('already_existed'):
                print(f"  File size: {result['file_size'] / 1024 / 1024:.2f} MB")

        else:
            # Multiple URLs from file
            results = downloader.download_from_file(args.file, tags)
            success = sum(1 for r in results if r['success'])
            print(f"\n✓ Downloaded {success}/{len(results)} videos")

    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
