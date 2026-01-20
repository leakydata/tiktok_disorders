#!/usr/bin/env python3
"""
TikTok video discovery script.

Discover videos from TikTok users, hashtags, and keyword searches.
Outputs video URLs to a file for processing by the pipeline.

Usage:
    # From a video URL - extracts username and fetches all their videos
    python scripts/discover.py --url "https://tiktok.com/@user/video/123"

    # From a username directly
    python scripts/discover.py --user chronicallychillandhot

    # Expand all users from existing urls.txt
    python scripts/discover.py --expand-users urls.txt

    # Find videos with specific hashtags
    python scripts/discover.py --hashtag ehlersdanlos --max-videos 100

    # Multiple hashtags
    python scripts/discover.py --hashtag EDS --hashtag POTS --hashtag MCAS

    # Search by keyword/phrase
    python scripts/discover.py --search "ehlers danlos syndrome" --max-videos 50

    # Output options
    python scripts/discover.py --user someuser --output new_urls.txt
    python scripts/discover.py --user someuser --append  # Append to urls.txt
"""
import sys
import argparse
import re
import time
import random
from pathlib import Path
from typing import List, Set, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_video_by_url

# Flag to track if we should use yt-dlp fallback
USE_YTDLP_FALLBACK = True


def extract_username_from_url(url: str) -> Optional[str]:
    """Extract the TikTok username from a video URL."""
    # Pattern: https://www.tiktok.com/@username/video/...
    match = re.search(r'tiktok\.com/@([^/]+)', url)
    if match:
        return match.group(1)
    return None


def load_existing_urls(file_path: Path) -> Set[str]:
    """Load existing URLs from a file to avoid duplicates."""
    existing = set()
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Strip inline comments
                    if ' #' in line:
                        line = line.split(' #', 1)[0].strip()
                    if ' //' in line:
                        line = line.split(' //', 1)[0].strip()
                    if line:
                        existing.add(line)
    return existing


def check_url_in_database(url: str) -> bool:
    """Check if a URL already exists in the database."""
    try:
        video = get_video_by_url(url)
        return video is not None
    except Exception:
        return False


def discover_user_videos_ytdlp(username: str, max_videos: Optional[int] = None,
                               date_after: Optional[str] = None,
                               date_before: Optional[str] = None) -> List[str]:
    """
    Discover video URLs from a TikTok user's profile using yt-dlp.
    
    This is more reliable than tiktokapipy for user profiles.
    
    Args:
        username: TikTok username
        max_videos: Maximum number of videos to fetch
        date_after: Only include videos after this date (YYYYMMDD)
        date_before: Only include videos before this date (YYYYMMDD)
    """
    import yt_dlp
    
    username = username.lstrip('@')
    urls = []
    
    date_info = ""
    if date_after:
        date_info += f" after {date_after}"
    if date_before:
        date_info += f" before {date_before}"
    
    print(f"  Fetching videos for @{username}{date_info} (via yt-dlp)...")
    
    user_url = f"https://www.tiktok.com/@{username}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'playlist_items': f'1:{max_videos}' if max_videos else None,
    }
    
    # Add date filtering
    if date_after:
        ydl_opts['dateafter'] = date_after.replace('-', '')
    if date_before:
        ydl_opts['datebefore'] = date_before.replace('-', '')
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(user_url, download=False)
            
            if result and 'entries' in result:
                for entry in result['entries']:
                    if entry and entry.get('url'):
                        urls.append(entry['url'])
                    elif entry and entry.get('id'):
                        video_url = f"https://www.tiktok.com/@{username}/video/{entry['id']}"
                        urls.append(video_url)
                        
        print(f"  Found {len(urls)} videos for @{username}")
        
    except Exception as e:
        print(f"  Error fetching @{username}: {e}")
    
    return urls


def discover_user_videos_tiktokapipy(username: str, max_videos: Optional[int] = None,
                                      delay_range: tuple = (2, 5)) -> List[str]:
    """
    Discover all video URLs from a TikTok user's profile using tiktokapipy.
    
    Fallback method if yt-dlp doesn't work.
    """
    try:
        from tiktokapipy.api import TikTokAPI
    except ImportError:
        print("  tiktokapipy not available, using yt-dlp")
        return discover_user_videos_ytdlp(username, max_videos)

    username = username.lstrip('@')
    urls = []

    print(f"  Fetching videos for @{username} (via tiktokapipy)...")

    try:
        with TikTokAPI() as api:
            user = api.user(username)

            count = 0
            for video in user.videos:
                if max_videos and count >= max_videos:
                    break

                video_url = f"https://www.tiktok.com/@{username}/video/{video.id}"
                urls.append(video_url)
                count += 1

                # Rate limiting
                if count % 10 == 0:
                    delay = random.uniform(*delay_range)
                    time.sleep(delay)

            print(f"  Found {len(urls)} videos for @{username}")

    except Exception as e:
        print(f"  Error with tiktokapipy for @{username}: {e}")
        print(f"  Falling back to yt-dlp...")
        return discover_user_videos_ytdlp(username, max_videos)

    return urls


def discover_user_videos(username: str, max_videos: Optional[int] = None,
                         delay_range: tuple = (2, 5),
                         date_after: Optional[str] = None,
                         date_before: Optional[str] = None) -> List[str]:
    """
    Discover all video URLs from a TikTok user's profile.
    
    Tries yt-dlp first (more reliable), falls back to tiktokapipy.
    
    Args:
        username: TikTok username
        max_videos: Maximum number of videos to fetch
        delay_range: Random delay range between requests
        date_after: Only include videos after this date (YYYYMMDD)
        date_before: Only include videos before this date (YYYYMMDD)
    """
    # Try yt-dlp first as it's more reliable (and supports date filtering)
    urls = discover_user_videos_ytdlp(username, max_videos, date_after, date_before)
    
    # If yt-dlp failed, try tiktokapipy (no date filtering available)
    if not urls:
        if date_after or date_before:
            print("    Note: tiktokapipy fallback doesn't support date filtering")
        urls = discover_user_videos_tiktokapipy(username, max_videos, delay_range)
    
    return urls


def discover_hashtag_videos_ytdlp(hashtag: str, max_videos: int = 100) -> List[str]:
    """
    Discover video URLs from a TikTok hashtag page using yt-dlp.
    
    Uses the TikTok tag URL format: https://www.tiktok.com/tag/HASHTAG
    """
    import yt_dlp
    
    hashtag = hashtag.lstrip('#').lower()
    urls = []
    
    print(f"  Fetching videos for #{hashtag} (via yt-dlp)...")
    
    tag_url = f"https://www.tiktok.com/tag/{hashtag}"
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': 'in_playlist',
        'playlist_items': f'1:{max_videos}',
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(tag_url, download=False)
            
            if result and 'entries' in result:
                for entry in result['entries']:
                    if entry:
                        if entry.get('url'):
                            urls.append(entry['url'])
                        elif entry.get('webpage_url'):
                            urls.append(entry['webpage_url'])
                        elif entry.get('id') and entry.get('uploader_id'):
                            video_url = f"https://www.tiktok.com/@{entry['uploader_id']}/video/{entry['id']}"
                            urls.append(video_url)
                            
        print(f"  Found {len(urls)} videos for #{hashtag}")
        
    except Exception as e:
        error_msg = str(e)
        if 'Unsupported URL' in error_msg:
            print(f"  yt-dlp doesn't support hashtag URLs directly. Try --user instead.")
        else:
            print(f"  Error fetching #{hashtag}: {e}")
    
    return urls


def discover_hashtag_videos_tiktokapipy(hashtag: str, max_videos: int = 100,
                                         delay_range: tuple = (2, 5)) -> List[str]:
    """
    Discover video URLs from a TikTok hashtag page using tiktokapipy.
    """
    try:
        from tiktokapipy.api import TikTokAPI
    except ImportError:
        print("  tiktokapipy not available")
        return []

    hashtag = hashtag.lstrip('#')
    urls = []

    print(f"  Fetching videos for #{hashtag} (via tiktokapipy)...")

    try:
        with TikTokAPI() as api:
            challenge = api.challenge(hashtag)

            count = 0
            for video in challenge.videos:
                if count >= max_videos:
                    break

                # Get username from video author
                author = video.author.unique_id if hasattr(video, 'author') and video.author else None
                if author:
                    video_url = f"https://www.tiktok.com/@{author}/video/{video.id}"
                    urls.append(video_url)
                    count += 1

                # Rate limiting
                if count % 10 == 0:
                    delay = random.uniform(*delay_range)
                    time.sleep(delay)

            print(f"  Found {len(urls)} videos for #{hashtag}")

    except Exception as e:
        print(f"  Error fetching #{hashtag}: {e}")

    return urls


def discover_hashtag_videos_playwright(hashtag: str, max_videos: int = 100,
                                        scroll_pause: float = 2.0,
                                        headless: bool = False) -> List[str]:
    """
    Discover video URLs from a TikTok hashtag page using Playwright directly.
    
    Uses a real browser with human-like scrolling behavior to avoid detection.
    
    Args:
        hashtag: TikTok hashtag (without #)
        max_videos: Maximum number of videos to collect
        scroll_pause: Pause between scrolls (seconds)
        headless: Run browser in headless mode (visible browser is harder to detect)
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  Playwright not available. Run: uv run playwright install")
        return []
    
    hashtag = hashtag.lstrip('#').lower()
    urls = set()
    
    print(f"  Fetching videos for #{hashtag} (via Playwright browser)...")
    print(f"  Browser will open - please don't interact with it.")
    
    tag_url = f"https://www.tiktok.com/tag/{hashtag}"
    
    try:
        with sync_playwright() as p:
            # Launch visible browser (harder to detect as bot)
            browser = p.chromium.launch(
                headless=headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--no-sandbox',
                ]
            )
            
            # Create context with realistic viewport and user agent
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            page = context.new_page()
            
            # Navigate to hashtag page
            page.goto(tag_url, wait_until='networkidle', timeout=30000)
            
            # Wait for content to load
            time.sleep(3)
            
            # Check if we hit an actual visible captcha
            captcha_visible = page.query_selector('div[class*="captcha"]') or \
                              page.query_selector('div[class*="Captcha"]') or \
                              page.query_selector('iframe[src*="captcha"]')
            
            if captcha_visible:
                print("  TikTok is showing a captcha. Please solve it manually in the browser.")
                print("  Waiting 30 seconds for you to solve it...")
                time.sleep(30)
            
            # Check if page loaded but no content (might be blocked)
            no_content = page.query_selector('div[class*="empty"]') or \
                         page.query_selector('div[class*="no-content"]')
            if no_content and not page.query_selector('a[href*="/video/"]'):
                print("  Page appears empty - TikTok may be blocking. Continuing anyway...")
            
            
            # Scroll and collect video links
            last_height = 0
            scroll_attempts = 0
            max_scroll_attempts = 50  # Prevent infinite scrolling
            
            while len(urls) < max_videos and scroll_attempts < max_scroll_attempts:
                # Extract video links from current page
                links = page.query_selector_all('a[href*="/video/"]')
                
                for link in links:
                    href = link.get_attribute('href')
                    if href and '/video/' in href:
                        # Normalize URL
                        if href.startswith('/'):
                            href = f"https://www.tiktok.com{href}"
                        if 'tiktok.com' in href and '/video/' in href:
                            urls.add(href)
                
                print(f"    Found {len(urls)} videos so far...", end='\r')
                
                if len(urls) >= max_videos:
                    break
                
                # Scroll down with human-like behavior
                page.evaluate('window.scrollBy(0, window.innerHeight * 0.8)')
                
                # Random pause (human-like)
                time.sleep(scroll_pause + random.uniform(0.5, 1.5))
                
                # Check if we've reached the bottom
                new_height = page.evaluate('document.body.scrollHeight')
                if new_height == last_height:
                    scroll_attempts += 1
                    if scroll_attempts >= 3:
                        print(f"\n  Reached end of page after {len(urls)} videos")
                        break
                else:
                    scroll_attempts = 0
                    last_height = new_height
            
            browser.close()
            
        print(f"\n  Found {len(urls)} videos for #{hashtag}")
        
    except Exception as e:
        error_msg = str(e)
        if 'Timeout' in error_msg:
            print(f"  Page load timeout - TikTok may be slow or blocking")
        else:
            print(f"  Error fetching #{hashtag}: {e}")
    
    return list(urls)[:max_videos]


def discover_hashtag_videos(hashtag: str, max_videos: int = 100,
                            delay_range: tuple = (2, 5),
                            use_browser: bool = False) -> List[str]:
    """
    Discover video URLs from a TikTok hashtag page.
    
    Tries multiple methods:
    1. If use_browser=True, uses Playwright with visible browser (most reliable)
    2. yt-dlp (fast but often blocked)
    3. tiktokapipy (fallback)
    """
    # If browser mode requested, use Playwright directly
    if use_browser:
        return discover_hashtag_videos_playwright(hashtag, max_videos)
    
    # Try yt-dlp first
    urls = discover_hashtag_videos_ytdlp(hashtag, max_videos)
    
    # If yt-dlp failed, try tiktokapipy
    if not urls:
        urls = discover_hashtag_videos_tiktokapipy(hashtag, max_videos, delay_range)
    
    # If all else failed, suggest browser mode
    if not urls:
        print(f"  Tip: Try --browser flag for better hashtag discovery")
    
    return urls


def discover_search_videos(query: str, max_videos: int = 50,
                           delay_range: tuple = (2, 5)) -> List[str]:
    """
    Discover video URLs from a TikTok search.
    
    Note: TikTok search is harder to scrape. This uses tiktokapipy.
    """
    try:
        from tiktokapipy.api import TikTokAPI
    except ImportError:
        print("  tiktokapipy not available for search")
        return []

    urls = []

    print(f"  Searching for '{query}' (via tiktokapipy)...")

    try:
        with TikTokAPI() as api:
            # Search returns videos matching the query
            results = api.search.videos(query)

            count = 0
            for video in results:
                if count >= max_videos:
                    break

                author = video.author.unique_id if hasattr(video, 'author') and video.author else None
                if author:
                    video_url = f"https://www.tiktok.com/@{author}/video/{video.id}"
                    urls.append(video_url)
                    count += 1

                # Rate limiting
                if count % 10 == 0:
                    delay = random.uniform(*delay_range)
                    time.sleep(delay)

            print(f"  Found {len(urls)} videos for search '{query}'")

    except Exception as e:
        print(f"  Error searching '{query}': {e}")
        print("  Note: TikTok search requires Playwright browsers. Run: uv run playwright install")

    return urls


def extract_unique_users_from_file(file_path: Path) -> Set[str]:
    """Extract unique usernames from a URLs file."""
    users = set()
    urls = load_existing_urls(file_path)

    for url in urls:
        username = extract_username_from_url(url)
        if username:
            users.add(username)

    return users


def write_urls_to_file(urls: List[str], output_path: Path, append: bool = False,
                       existing_urls: Optional[Set[str]] = None) -> int:
    """
    Write URLs to a file, optionally appending.

    Args:
        urls: List of URLs to write
        output_path: Path to output file
        append: Whether to append to existing file
        existing_urls: Set of existing URLs to skip (for deduplication)

    Returns:
        Number of new URLs written
    """
    existing = existing_urls or set()
    if append and output_path.exists():
        existing = existing.union(load_existing_urls(output_path))

    new_urls = [url for url in urls if url not in existing]

    if not new_urls:
        return 0

    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        if not append:
            f.write("# TikTok URLs discovered by discover.py\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        else:
            f.write(f"\n# Added by discover.py on {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        for url in new_urls:
            f.write(url + '\n')

    return len(new_urls)


def append_urls_incrementally(urls: List[str], output_path: Path, 
                               existing_urls: Set[str], source_label: str) -> tuple:
    """
    Append URLs to file immediately after discovery (safer - survives crashes).
    
    Args:
        urls: List of URLs discovered
        output_path: Path to output file
        existing_urls: Set of URLs already in file (updated in place)
        source_label: Label for the source (e.g., "@username" or "#hashtag")
    
    Returns:
        Tuple of (new_urls_count, existing_urls set updated)
    """
    new_urls = [url for url in urls if url not in existing_urls]
    
    if not new_urls:
        print(f"    -> 0 new URLs (all duplicates)")
        return 0, existing_urls
    
    # Append to file immediately
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(f"\n# {source_label} - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        for url in new_urls:
            f.write(url + '\n')
    
    # Update the existing set
    existing_urls.update(new_urls)
    
    print(f"    -> Saved {len(new_urls)} new URLs to {output_path}")
    return len(new_urls), existing_urls


def main():
    parser = argparse.ArgumentParser(
        description='Discover TikTok videos from users, hashtags, and searches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get all videos from a user
  python scripts/discover.py --user chronicallychillandhot

  # Get videos from a video URL's author
  python scripts/discover.py --url "https://tiktok.com/@user/video/123"

  # Expand all users in urls.txt
  python scripts/discover.py --expand-users urls.txt --append

  # Search by hashtag
  python scripts/discover.py --hashtag EDS --hashtag POTS --max-videos 200

  # Search by keyword
  python scripts/discover.py --search "ehlers danlos" --max-videos 100
        """
    )

    # Input sources (at least one required)
    input_group = parser.add_argument_group('Input sources')
    input_group.add_argument('--url', action='append', dest='urls',
                             help='Video URL (extracts username and fetches all their videos)')
    input_group.add_argument('--user', action='append', dest='users',
                             help='TikTok username to fetch all videos from')
    input_group.add_argument('--expand-users', dest='expand_file',
                             help='File with URLs - extract unique users and fetch all their videos')
    input_group.add_argument('--hashtag', action='append', dest='hashtags',
                             help='Hashtag to search (without #)')
    input_group.add_argument('--search', action='append', dest='searches',
                             help='Search query')

    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument('--output', '-o', default='urls.txt',
                              help='Output file path (default: urls.txt)')
    output_group.add_argument('--append', '-a', action='store_true', default=True,
                              help='Append to existing output file (default: True)')
    output_group.add_argument('--overwrite', action='store_true',
                              help='Overwrite output file instead of appending')

    # Limits and behavior
    parser.add_argument('--max-videos', type=int, default=None,
                        help='Maximum videos per user/hashtag/search (default: unlimited for users, 100 for hashtags/searches)')
    parser.add_argument('--after', dest='date_after',
                        help='Only include videos uploaded after this date (YYYYMMDD or YYYY-MM-DD)')
    parser.add_argument('--before', dest='date_before',
                        help='Only include videos uploaded before this date (YYYYMMDD or YYYY-MM-DD)')
    parser.add_argument('--days', type=int,
                        help='Only include videos from the last N days (shortcut for --after)')
    parser.add_argument('--min-delay', type=float, default=2.0,
                        help='Minimum delay between requests in seconds (default: 2.0)')
    parser.add_argument('--max-delay', type=float, default=5.0,
                        help='Maximum delay between requests in seconds (default: 5.0)')
    parser.add_argument('--skip-db-check', action='store_true',
                        help='Skip checking database for existing videos')
    parser.add_argument('--browser', action='store_true',
                        help='Use visible browser for hashtag discovery (slower but more reliable)')
    parser.add_argument('--headless', action='store_true',
                        help='Run browser in headless mode (use with --browser)')

    args = parser.parse_args()

    # Validate that at least one input source is provided
    has_input = any([
        args.urls,
        args.users,
        args.expand_file,
        args.hashtags,
        args.searches
    ])

    if not has_input:
        parser.error("At least one input source is required (--url, --user, --expand-users, --hashtag, or --search)")

    delay_range = (args.min_delay, args.max_delay)
    output_path = Path(args.output)
    
    # --overwrite takes precedence over --append
    append_mode = not args.overwrite
    existing_urls = load_existing_urls(output_path) if append_mode else set()
    total_new_urls = 0
    total_found = 0
    
    # Handle date filtering
    date_after = args.date_after
    date_before = args.date_before
    
    # --days is a shortcut for --after (last N days)
    if args.days:
        from datetime import datetime, timedelta
        cutoff = datetime.now() - timedelta(days=args.days)
        date_after = cutoff.strftime('%Y%m%d')

    print("=" * 60)
    print("TikTok Video Discovery")
    print("=" * 60)
    print(f"Output: {output_path} ({'append' if append_mode else 'overwrite'} mode)")
    if date_after or date_before:
        date_range = []
        if date_after:
            date_range.append(f"after {date_after}")
        if date_before:
            date_range.append(f"before {date_before}")
        print(f"Date filter: {' and '.join(date_range)}")
    print("URLs are saved incrementally after each source (crash-safe)")

    # Initialize file if not appending
    if not append_mode:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# TikTok URLs discovered by discover.py\n")
            f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Process video URLs (extract username and fetch all videos)
    if args.urls:
        print(f"\nProcessing {len(args.urls)} video URL(s)...")
        for url in args.urls:
            username = extract_username_from_url(url)
            if username:
                urls = discover_user_videos(username, args.max_videos, delay_range,
                                           date_after, date_before)
                total_found += len(urls)
                if urls:
                    count, existing_urls = append_urls_incrementally(
                        urls, output_path, existing_urls, f"@{username}"
                    )
                    total_new_urls += count
            else:
                print(f"  Could not extract username from: {url}")

    # Process direct usernames
    if args.users:
        print(f"\nProcessing {len(args.users)} username(s)...")
        for username in args.users:
            urls = discover_user_videos(username, args.max_videos, delay_range,
                                       date_after, date_before)
            total_found += len(urls)
            if urls:
                count, existing_urls = append_urls_incrementally(
                    urls, output_path, existing_urls, f"@{username}"
                )
                total_new_urls += count

    # Expand users from file
    if args.expand_file:
        expand_path = Path(args.expand_file)
        if not expand_path.exists():
            print(f"Error: File not found: {expand_path}")
            sys.exit(1)

        users = extract_unique_users_from_file(expand_path)
        print(f"\nExpanding {len(users)} unique user(s) from {expand_path}...")

        for i, username in enumerate(sorted(users), 1):
            print(f"\n[{i}/{len(users)}] @{username}")
            urls = discover_user_videos(username, args.max_videos, delay_range,
                                       date_after, date_before)
            total_found += len(urls)
            if urls:
                count, existing_urls = append_urls_incrementally(
                    urls, output_path, existing_urls, f"@{username}"
                )
                total_new_urls += count

            # Longer delay between users
            delay = random.uniform(3, 7)
            time.sleep(delay)

    # Process hashtags
    if args.hashtags:
        if args.browser:
            print(f"\nProcessing {len(args.hashtags)} hashtag(s) using browser...")
            print("  A browser window will open. Please don't interact with it.")
        else:
            print(f"\nProcessing {len(args.hashtags)} hashtag(s)...")
        max_vids = args.max_videos or 100
        for hashtag in args.hashtags:
            if args.browser:
                urls = discover_hashtag_videos_playwright(
                    hashtag, max_vids, 
                    scroll_pause=2.0,
                    headless=args.headless
                )
            else:
                urls = discover_hashtag_videos(hashtag, max_vids, delay_range)
            total_found += len(urls)
            if urls:
                count, existing_urls = append_urls_incrementally(
                    urls, output_path, existing_urls, f"#{hashtag}"
                )
                total_new_urls += count

            # Delay between hashtags
            delay = random.uniform(3, 7)
            time.sleep(delay)

    # Process searches
    if args.searches:
        print(f"\nProcessing {len(args.searches)} search(es)...")
        max_vids = args.max_videos or 50
        for query in args.searches:
            urls = discover_search_videos(query, max_vids, delay_range)
            total_found += len(urls)
            if urls:
                count, existing_urls = append_urls_incrementally(
                    urls, output_path, existing_urls, f"search:{query}"
                )
                total_new_urls += count

            # Delay between searches
            delay = random.uniform(3, 7)
            time.sleep(delay)

    print(f"\n{'=' * 60}")
    print(f"Discovery complete!")
    print(f"  Total URLs found: {total_found}")
    print(f"  New URLs saved: {total_new_urls}")
    print(f"  Duplicates skipped: {total_found - total_new_urls}")
    print(f"  Output file: {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
