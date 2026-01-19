"""
Video and audio downloader module using yt-dlp.
Supports YouTube, TikTok, and other platforms supported by yt-dlp.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yt_dlp
from typing import Optional, Dict, Any, List
from datetime import datetime
import re

from config import AUDIO_DIR, ensure_directories
from database import insert_video, get_video_by_url


class VideoDownloader:
    """Downloads videos and extracts audio using yt-dlp."""

    def __init__(self, audio_dir: Optional[Path] = None):
        """
        Initialize the downloader.

        Args:
            audio_dir: Directory to save audio files (defaults to config.AUDIO_DIR)
        """
        self.audio_dir = audio_dir or AUDIO_DIR
        ensure_directories()

    def _get_platform(self, url: str) -> str:
        """Determine the platform from the URL."""
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'youtube'
        elif 'tiktok.com' in url:
            return 'tiktok'
        elif 'instagram.com' in url:
            return 'instagram'
        elif 'facebook.com' in url:
            return 'facebook'
        else:
            return 'other'

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to remove problematic characters."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename

    def get_video_info(self, url: str) -> Dict[str, Any]:
        """
        Retrieve video metadata without downloading.

        Args:
            url: Video URL

        Returns:
            Dictionary containing video metadata
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            # Parse upload date
            upload_date = None
            if info.get('upload_date'):
                try:
                    upload_date = datetime.strptime(info['upload_date'], '%Y%m%d').date()
                except (ValueError, TypeError):
                    pass

            return {
                'video_id': info.get('id'),
                'title': info.get('title'),
                'author': info.get('uploader') or info.get('channel'),
                'duration': info.get('duration'),  # in seconds
                'upload_date': upload_date,
                'description': info.get('description'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'platform': self._get_platform(url),
            }

    def download_audio(self, url: str, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Download video and extract audio as MP3.

        Args:
            url: Video URL
            tags: Optional list of tags to associate with the video

        Returns:
            Dictionary with video database ID and file path
        """
        # Check if already downloaded
        existing = get_video_by_url(url)
        if existing and existing.get('audio_path'):
            print(f"✓ Video already downloaded: {existing['audio_path']}")
            return {
                'video_id': existing['id'],
                'audio_path': existing['audio_path'],
                'already_existed': True
            }

        # Get video info first
        info = self.get_video_info(url)
        video_id = info['video_id']
        platform = info['platform']

        # Create filename
        safe_title = self._sanitize_filename(info['title'] or video_id)
        output_filename = f"{platform}_{video_id}_{safe_title}.mp3"
        output_path = self.audio_dir / output_filename

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': str(self.audio_dir / f"{platform}_{video_id}_{safe_title}.%(ext)s"),
            'quiet': False,
            'no_warnings': False,
        }

        # Download and extract audio
        print(f"Downloading audio from {url}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Verify file exists
        if not output_path.exists():
            raise FileNotFoundError(f"Audio file not created: {output_path}")

        file_size = output_path.stat().st_size

        # Store in database
        metadata = {
            'title': info['title'],
            'author': info['author'],
            'duration': info['duration'],
            'upload_date': info['upload_date'],
            'tags': tags or [],
            'audio_path': str(output_path),
            'audio_size_bytes': file_size
        }

        db_id = insert_video(url, platform, video_id, metadata)

        print(f"✓ Audio downloaded: {output_path} ({file_size / 1024 / 1024:.2f} MB)")

        return {
            'video_id': db_id,
            'audio_path': str(output_path),
            'file_size': file_size,
            'already_existed': False
        }

    def download_from_file(self, file_path: str, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Download multiple videos from a text file (one URL per line).

        Args:
            file_path: Path to text file containing URLs
            tags: Optional tags to apply to all videos

        Returns:
            List of results for each video
        """
        results = []
        with open(file_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print(f"Found {len(urls)} URLs to download")

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Processing: {url}")
            try:
                result = self.download_audio(url, tags)
                results.append({'url': url, 'success': True, 'data': result})
            except Exception as e:
                print(f"✗ Error downloading {url}: {e}")
                results.append({'url': url, 'success': False, 'error': str(e)})

        # Summary
        success_count = sum(1 for r in results if r['success'])
        print(f"\n✓ Successfully downloaded {success_count}/{len(urls)} videos")

        return results


if __name__ == '__main__':
    # Example usage
    downloader = VideoDownloader()

    # Test with a video URL
    test_url = input("Enter a video URL to download: ").strip()
    if test_url:
        try:
            result = downloader.download_audio(test_url, tags=['EDS', 'test'])
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"Error: {e}")
