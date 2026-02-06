#!/usr/bin/env python3
"""
Clean transcripts by removing repeated phrases.

Whisper sometimes hallucinates and repeats phrases many times.
This script detects and removes those repetitions.

Features:
- Preserves original text in `original_text` column for revert
- Tracks when cleaning was done in `cleaned_at` column
- Can revert all cleaned transcripts back to original
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_connection
from transcriber import _remove_repeated_phrases


def clean_all_transcripts(dry_run: bool = False, limit: int = None, verbose: bool = False,
                          min_reduction: float = 0.0):
    """Clean all transcripts in the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Only process transcripts that haven't been cleaned yet
        query = """
            SELECT id, video_id, text, word_count 
            FROM transcripts 
            WHERE original_text IS NULL
            ORDER BY id
        """
        if limit:
            query = query.replace("ORDER BY id", f"ORDER BY id LIMIT {limit}")
        
        cur.execute(query)
        rows = cur.fetchall()
        
        print(f"Checking {len(rows)} uncleaned transcripts for repetitions...")
        
        cleaned_count = 0
        skipped_count = 0
        total_chars_saved = 0
        
        for id, video_id, text, word_count in rows:
            if not text:
                continue
                
            original_len = len(text)
            new_text = _remove_repeated_phrases(text)
            new_len = len(new_text)
            
            if new_text != text:
                chars_saved = original_len - new_len
                reduction_pct = (chars_saved / original_len) * 100
                
                # Skip if reduction is below minimum threshold
                if reduction_pct < min_reduction:
                    skipped_count += 1
                    continue
                
                if verbose or reduction_pct > 20:
                    print(f"\n{'='*80}")
                    print(f"[Video {video_id}] Transcript {id}:")
                    print(f"  Before: {original_len} chars, {word_count} words")
                    print(f"  After: {new_len} chars ({reduction_pct:.1f}% reduction)")
                    if verbose:
                        print(f"\n  --- ORIGINAL ---")
                        print(f"  {text}")
                        print(f"\n  --- CLEANED ---")
                        print(f"  {new_text}")
                        print(f"\n  --- WHAT WAS REMOVED ---")
                        print(f"  {chars_saved} characters of repeated content removed")
                
                if not dry_run:
                    new_word_count = len(new_text.split())
                    cur.execute("""
                        UPDATE transcripts 
                        SET text = %s, 
                            word_count = %s,
                            original_text = %s,
                            original_word_count = %s,
                            cleaned_at = %s
                        WHERE id = %s
                    """, (new_text, new_word_count, text, word_count, datetime.now(), id))
                
                cleaned_count += 1
                total_chars_saved += chars_saved
        
        if not dry_run:
            conn.commit()
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Summary:")
        print(f"  Transcripts cleaned: {cleaned_count}")
        if skipped_count > 0:
            print(f"  Skipped (below {min_reduction}% threshold): {skipped_count}")
        print(f"  Total characters removed: {total_chars_saved:,}")
        if cleaned_count > 0:
            print(f"  Average reduction: {total_chars_saved / cleaned_count:.0f} chars per transcript")
        
        if not dry_run and cleaned_count > 0:
            print(f"\nOriginal text preserved in 'original_text' column")
            print(f"  To revert: uv run python scripts/clean_transcripts.py --revert")


def revert_cleaned_transcripts(dry_run: bool = False, limit: int = None, verbose: bool = False):
    """Revert all cleaned transcripts back to their original text."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        query = """
            SELECT id, video_id, text, original_text, word_count, original_word_count
            FROM transcripts 
            WHERE original_text IS NOT NULL
            ORDER BY id
        """
        if limit:
            query = query.replace("ORDER BY id", f"ORDER BY id LIMIT {limit}")
        
        cur.execute(query)
        rows = cur.fetchall()
        
        if not rows:
            print("No cleaned transcripts found to revert.")
            return
        
        print(f"Found {len(rows)} cleaned transcripts to revert...")
        
        reverted_count = 0
        
        for id, video_id, current_text, original_text, current_wc, original_wc in rows:
            if verbose:
                print(f"\n[Video {video_id}] Transcript {id}:")
                print(f"  Current: {len(current_text)} chars, {current_wc} words")
                print(f"  Original: {len(original_text)} chars, {original_wc} words")
            
            if not dry_run:
                cur.execute("""
                    UPDATE transcripts 
                    SET text = original_text,
                        word_count = original_word_count,
                        original_text = NULL,
                        original_word_count = NULL,
                        cleaned_at = NULL
                    WHERE id = %s
                """, (id,))
            
            reverted_count += 1
        
        if not dry_run:
            conn.commit()
        
        print(f"\n{'[DRY RUN] ' if dry_run else ''}Reverted {reverted_count} transcripts to original")


def show_stats():
    """Show statistics about cleaned vs uncleaned transcripts."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE original_text IS NOT NULL) as cleaned,
                COUNT(*) FILTER (WHERE original_text IS NULL) as uncleaned,
                SUM(LENGTH(original_text) - LENGTH(text)) FILTER (WHERE original_text IS NOT NULL) as chars_saved
            FROM transcripts
        """)
        row = cur.fetchone()
        total, cleaned, uncleaned, chars_saved = row
        
        print(f"Transcript Cleaning Statistics:")
        print(f"  Total transcripts: {total}")
        print(f"  Cleaned (have original preserved): {cleaned}")
        print(f"  Uncleaned: {uncleaned}")
        if chars_saved:
            print(f"  Total characters saved: {chars_saved:,}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Clean transcripts by removing repeated phrases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview changes (dry run)
  uv run python scripts/clean_transcripts.py --dry-run --verbose
  
  # Clean all transcripts
  uv run python scripts/clean_transcripts.py
  
  # Only clean transcripts with >10% reduction
  uv run python scripts/clean_transcripts.py --min-reduction 10
  
  # Revert all cleaned transcripts
  uv run python scripts/clean_transcripts.py --revert
  
  # Show statistics
  uv run python scripts/clean_transcripts.py --stats
        """
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without modifying database')
    parser.add_argument('--limit', type=int,
                       help='Only process first N transcripts')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show full text for each cleaned transcript')
    parser.add_argument('--min-reduction', type=float, default=0.0,
                       help='Only clean transcripts with at least this %% reduction (default: 0)')
    parser.add_argument('--revert', action='store_true',
                       help='Revert cleaned transcripts back to original')
    parser.add_argument('--stats', action='store_true',
                       help='Show cleaning statistics')
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    elif args.revert:
        revert_cleaned_transcripts(
            dry_run=args.dry_run,
            limit=args.limit,
            verbose=args.verbose
        )
    else:
        clean_all_transcripts(
            dry_run=args.dry_run,
            limit=args.limit,
            verbose=args.verbose,
            min_reduction=args.min_reduction
        )


if __name__ == '__main__':
    main()
