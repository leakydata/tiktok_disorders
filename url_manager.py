"""
URL management utilities for tracking processed and pending URLs.

This module provides functions to:
- Read URLs from both pending (urls.txt) and processed (urls_processed.txt) files
- Move URLs from pending to processed after successful pipeline completion
- Check if a URL has already been discovered/processed
"""

from pathlib import Path
from typing import List, Set, Optional
from datetime import datetime
import re


# Default file paths
DEFAULT_PENDING_FILE = "urls.txt"
DEFAULT_PROCESSED_FILE = "urls_processed.txt"
DEFAULT_FAILED_FILE = "urls_failed.txt"


def read_url_file(path: str) -> List[str]:
    """
    Read URLs from a file, handling comments and blank lines.
    
    Returns empty list if file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return []
    
    urls = []
    for raw in p.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('#') or line.startswith('//'):
            continue
        # Allow inline comments
        if ' #' in line:
            line = line.split(' #', 1)[0].strip()
        if ' //' in line:
            line = line.split(' //', 1)[0].strip()
        if line:
            urls.append(line)
    
    return urls


def get_pending_urls(pending_file: str = DEFAULT_PENDING_FILE) -> List[str]:
    """Get URLs that are pending processing."""
    return read_url_file(pending_file)


def get_processed_urls(processed_file: str = DEFAULT_PROCESSED_FILE) -> List[str]:
    """Get URLs that have been processed."""
    return read_url_file(processed_file)


def get_all_known_urls(pending_file: str = DEFAULT_PENDING_FILE,
                       processed_file: str = DEFAULT_PROCESSED_FILE) -> Set[str]:
    """
    Get all known URLs from both pending and processed files.
    
    Useful for deduplication during discovery.
    """
    pending = set(get_pending_urls(pending_file))
    processed = set(get_processed_urls(processed_file))
    return pending | processed


def is_url_known(url: str, 
                 pending_file: str = DEFAULT_PENDING_FILE,
                 processed_file: str = DEFAULT_PROCESSED_FILE) -> bool:
    """Check if a URL exists in either pending or processed files."""
    url = url.strip()
    all_urls = get_all_known_urls(pending_file, processed_file)
    return url in all_urls


def normalize_url(url: str) -> str:
    """Normalize a TikTok URL for consistent comparison."""
    url = url.strip()
    # Remove trailing slashes
    url = url.rstrip('/')
    # Remove query parameters
    if '?' in url:
        url = url.split('?')[0]
    return url


def mark_url_as_processed(url: str,
                          pending_file: str = DEFAULT_PENDING_FILE,
                          processed_file: str = DEFAULT_PROCESSED_FILE) -> bool:
    """
    Move a URL from pending to processed.
    
    - Removes the URL from pending file
    - Appends it to processed file with timestamp
    
    Returns True if URL was moved, False if it wasn't in pending.
    """
    url = url.strip()
    pending_path = Path(pending_file)
    processed_path = Path(processed_file)
    
    # Read current pending URLs
    if not pending_path.exists():
        return False
    
    lines = pending_path.read_text(encoding='utf-8').splitlines()
    
    # Find and remove the URL
    new_lines = []
    found = False
    for line in lines:
        clean = line.strip()
        # Extract URL from line (handle inline comments)
        url_part = clean
        if ' #' in url_part:
            url_part = url_part.split(' #', 1)[0].strip()
        if ' //' in url_part:
            url_part = url_part.split(' //', 1)[0].strip()
        
        if url_part == url:
            found = True
            # Don't add this line to new_lines (removes it)
        else:
            new_lines.append(line)
    
    if not found:
        # URL wasn't in pending file
        return False
    
    # Write updated pending file
    pending_path.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
    
    # Append to processed file with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(processed_path, 'a', encoding='utf-8') as f:
        f.write(f"{url} # processed {timestamp}\n")
    
    return True


def mark_urls_as_processed(urls: List[str],
                           pending_file: str = DEFAULT_PENDING_FILE,
                           processed_file: str = DEFAULT_PROCESSED_FILE) -> int:
    """
    Move multiple URLs from pending to processed (batch operation).
    
    More efficient than calling mark_url_as_processed repeatedly.
    
    Returns count of URLs moved.
    """
    # Normalize URLs for matching (strip whitespace, remove trailing slashes)
    def normalize(u):
        u = u.strip().rstrip('/')
        return u
    
    urls_to_move = set(normalize(u) for u in urls if u)
    pending_path = Path(pending_file)
    processed_path = Path(processed_file)
    
    # Create processed file if it doesn't exist
    if not processed_path.exists():
        processed_path.touch()
    
    if not pending_path.exists():
        # No pending file, but still record the URLs as processed
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(processed_path, 'a', encoding='utf-8') as f:
            for url in urls_to_move:
                f.write(f"{url} # processed {timestamp}\n")
        return len(urls_to_move)
    
    lines = pending_path.read_text(encoding='utf-8').splitlines()
    
    new_lines = []
    moved_urls = []
    
    for line in lines:
        clean = line.strip()
        url_part = clean
        if ' #' in url_part:
            url_part = url_part.split(' #', 1)[0].strip()
        if ' //' in url_part:
            url_part = url_part.split(' //', 1)[0].strip()
        
        # Normalize for comparison
        normalized_url = normalize(url_part)
        
        if normalized_url in urls_to_move:
            moved_urls.append(url_part)  # Keep original URL format
        else:
            new_lines.append(line)
    
    # If no URLs found in pending file, still record them as processed
    # (they might have been processed via command line or already removed)
    if not moved_urls and urls_to_move:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(processed_path, 'a', encoding='utf-8') as f:
            for url in urls_to_move:
                f.write(f"{url} # processed {timestamp}\n")
        return len(urls_to_move)
    
    if not moved_urls:
        return 0
    
    # Write updated pending file
    pending_path.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
    
    # Append to processed file
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(processed_path, 'a', encoding='utf-8') as f:
        for url in moved_urls:
            f.write(f"{url} # processed {timestamp}\n")
    
    return len(moved_urls)


def add_url_to_pending(url: str,
                       pending_file: str = DEFAULT_PENDING_FILE,
                       processed_file: str = DEFAULT_PROCESSED_FILE,
                       check_processed: bool = True) -> bool:
    """
    Add a URL to the pending file if it's not already known.
    
    Args:
        url: URL to add
        pending_file: Path to pending URLs file
        processed_file: Path to processed URLs file
        check_processed: If True, also check processed file for duplicates
        
    Returns:
        True if URL was added, False if it already existed
    """
    url = url.strip()
    
    # Check for duplicates
    pending_urls = set(get_pending_urls(pending_file))
    if url in pending_urls:
        return False
    
    if check_processed:
        processed_urls = set(get_processed_urls(processed_file))
        if url in processed_urls:
            return False
    
    # Add to pending
    pending_path = Path(pending_file)
    with open(pending_path, 'a', encoding='utf-8') as f:
        f.write(f"{url}\n")
    
    return True


def add_urls_to_pending(urls: List[str],
                        pending_file: str = DEFAULT_PENDING_FILE,
                        processed_file: str = DEFAULT_PROCESSED_FILE,
                        check_processed: bool = True) -> int:
    """
    Add multiple URLs to pending file, skipping duplicates.
    
    Returns count of new URLs added.
    """
    urls = [u.strip() for u in urls if u.strip()]
    
    # Get existing URLs
    existing = set(get_pending_urls(pending_file))
    if check_processed:
        existing |= set(get_processed_urls(processed_file))
    
    # Filter to new URLs only
    new_urls = [u for u in urls if u not in existing]
    
    if not new_urls:
        return 0
    
    # Append new URLs
    pending_path = Path(pending_file)
    with open(pending_path, 'a', encoding='utf-8') as f:
        for url in new_urls:
            f.write(f"{url}\n")
    
    return len(new_urls)


def get_failed_urls(failed_file: str = DEFAULT_FAILED_FILE) -> List[str]:
    """Get URLs that have failed processing."""
    return read_url_file(failed_file)


def mark_url_as_failed(url: str,
                       error: str = "",
                       pending_file: str = DEFAULT_PENDING_FILE,
                       failed_file: str = DEFAULT_FAILED_FILE) -> bool:
    """
    Move a URL from pending to failed.
    
    - Removes the URL from pending file
    - Appends it to failed file with timestamp and error
    
    Returns True if URL was moved, False if it wasn't in pending.
    """
    url = url.strip()
    pending_path = Path(pending_file)
    failed_path = Path(failed_file)
    
    # Create failed file if it doesn't exist
    if not failed_path.exists():
        failed_path.touch()
    
    # Read current pending URLs
    if not pending_path.exists():
        # No pending file, but still record as failed
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        error_comment = f" | {error}" if error else ""
        with open(failed_path, 'a', encoding='utf-8') as f:
            f.write(f"{url} # failed {timestamp}{error_comment}\n")
        return True
    
    lines = pending_path.read_text(encoding='utf-8').splitlines()
    
    # Find and remove the URL
    new_lines = []
    found = False
    for line in lines:
        clean = line.strip()
        # Extract URL from line (handle inline comments)
        url_part = clean
        if ' #' in url_part:
            url_part = url_part.split(' #', 1)[0].strip()
        if ' //' in url_part:
            url_part = url_part.split(' //', 1)[0].strip()
        
        if url_part == url:
            found = True
            # Don't add this line to new_lines (removes it)
        else:
            new_lines.append(line)
    
    # Write updated pending file
    if found:
        pending_path.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
    
    # Append to failed file with timestamp and error
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    error_comment = f" | {error}" if error else ""
    with open(failed_path, 'a', encoding='utf-8') as f:
        f.write(f"{url} # failed {timestamp}{error_comment}\n")
    
    return True


def mark_urls_as_failed(urls_with_errors: List[tuple],
                        pending_file: str = DEFAULT_PENDING_FILE,
                        failed_file: str = DEFAULT_FAILED_FILE) -> int:
    """
    Move multiple URLs from pending to failed (batch operation).
    
    Args:
        urls_with_errors: List of (url, error_message) tuples
        pending_file: Path to pending URLs file
        failed_file: Path to failed URLs file
    
    Returns count of URLs moved.
    """
    if not urls_with_errors:
        return 0
    
    # Normalize URLs for matching
    def normalize(u):
        u = u.strip().rstrip('/')
        return u
    
    urls_to_move = {normalize(u): err for u, err in urls_with_errors if u}
    pending_path = Path(pending_file)
    failed_path = Path(failed_file)
    
    # Create failed file if it doesn't exist
    if not failed_path.exists():
        failed_path.touch()
    
    moved_count = 0
    
    if pending_path.exists():
        lines = pending_path.read_text(encoding='utf-8').splitlines()
        
        new_lines = []
        moved_urls = []
        
        for line in lines:
            clean = line.strip()
            url_part = clean
            if ' #' in url_part:
                url_part = url_part.split(' #', 1)[0].strip()
            if ' //' in url_part:
                url_part = url_part.split(' //', 1)[0].strip()
            
            # Normalize for comparison
            normalized_url = normalize(url_part)
            
            if normalized_url in urls_to_move:
                moved_urls.append((url_part, urls_to_move[normalized_url]))
            else:
                new_lines.append(line)
        
        if moved_urls:
            # Write updated pending file
            pending_path.write_text('\n'.join(new_lines) + ('\n' if new_lines else ''), encoding='utf-8')
            moved_count = len(moved_urls)
    else:
        # No pending file, record all as failed
        moved_urls = [(u, err) for u, err in urls_with_errors if u]
    
    # Append to failed file
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(failed_path, 'a', encoding='utf-8') as f:
        for url, error in (moved_urls if pending_path.exists() else [(u, err) for u, err in urls_with_errors if u]):
            error_comment = f" | {error}" if error else ""
            f.write(f"{url} # failed {timestamp}{error_comment}\n")
    
    return moved_count if moved_count else len(urls_with_errors)


def filter_unprocessed_urls(urls: List[str],
                            processed_file: str = DEFAULT_PROCESSED_FILE,
                            failed_file: str = DEFAULT_FAILED_FILE) -> List[str]:
    """
    Filter out URLs that have already been processed or failed.
    
    Returns only URLs that haven't been seen before.
    """
    # Normalize for comparison
    def normalize(u):
        return u.strip().rstrip('/')
    
    processed = set(normalize(u) for u in get_processed_urls(processed_file))
    failed = set(normalize(u) for u in get_failed_urls(failed_file))
    already_seen = processed | failed
    
    return [u for u in urls if normalize(u) not in already_seen]


def get_stats(pending_file: str = DEFAULT_PENDING_FILE,
              processed_file: str = DEFAULT_PROCESSED_FILE,
              failed_file: str = DEFAULT_FAILED_FILE) -> dict:
    """Get statistics about pending, processed, and failed URLs."""
    pending = get_pending_urls(pending_file)
    processed = get_processed_urls(processed_file)
    failed = get_failed_urls(failed_file)
    
    return {
        'pending_count': len(pending),
        'processed_count': len(processed),
        'failed_count': len(failed),
        'total_known': len(pending) + len(processed) + len(failed),
        'pending_file': pending_file,
        'processed_file': processed_file,
        'failed_file': failed_file
    }


if __name__ == '__main__':
    # Quick stats when run directly
    stats = get_stats()
    print(f"URL Statistics:")
    print(f"  Pending: {stats['pending_count']}")
    print(f"  Processed: {stats['processed_count']}")
    print(f"  Failed: {stats['failed_count']}")
    print(f"  Total known: {stats['total_known']}")
