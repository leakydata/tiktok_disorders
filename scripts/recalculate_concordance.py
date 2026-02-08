#!/usr/bin/env python3
"""
Recalculate symptom concordance for all videos.

Run this after updating expected_symptoms in database.py and running init_db.py.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import (
    get_connection, calculate_symptom_concordance,
    get_expected_symptoms
)
from psycopg2.extras import RealDictCursor


def get_all_diagnosis_video_pairs():
    """Get all (video_id, diagnosis_id, condition_code) pairs."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT cd.id as diagnosis_id, cd.video_id, cd.condition_code
            FROM claimed_diagnoses cd
            ORDER BY cd.video_id
        """)
        return [dict(row) for row in cur.fetchall()]


def get_pairs_for_user(username: str):
    """Get diagnosis pairs for a specific user."""
    # Remove @ prefix if present
    username = username.lstrip('@')
    
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT cd.id as diagnosis_id, cd.video_id, cd.condition_code
            FROM claimed_diagnoses cd
            JOIN videos v ON cd.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            ORDER BY cd.video_id
        """, (username,))
        return [dict(row) for row in cur.fetchall()]


def clear_concordance(video_ids=None, username=None):
    """Clear existing concordance records."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        if username:
            username = username.lstrip('@')
            cur.execute("""
                DELETE FROM symptom_concordance
                WHERE video_id IN (
                    SELECT id FROM videos WHERE LOWER(author) = LOWER(%s)
                )
            """, (username,))
        elif video_ids:
            cur.execute("""
                DELETE FROM symptom_concordance
                WHERE video_id = ANY(%s)
            """, (video_ids,))
        else:
            cur.execute("DELETE FROM symptom_concordance")
        
        deleted = cur.rowcount
        conn.commit()
        return deleted


def main():
    parser = argparse.ArgumentParser(
        description='Recalculate symptom concordance after updating expected symptoms'
    )
    parser.add_argument('--user', help='Recalculate only for specific user (e.g., @username)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--clear-only', action='store_true', help='Only clear existing concordance, do not recalculate')
    args = parser.parse_args()
    
    # Show what conditions are available
    print("Expected symptoms available for conditions:")
    for condition in ['EDS', 'MCAS', 'POTS', 'FIBROMYALGIA', 'CFS', 'CIRS', 'GASTROPARESIS', 'SIBO']:
        expected = get_expected_symptoms(condition)
        core_count = len(expected.get('core', []))
        common_count = len(expected.get('common', []))
        print(f"  {condition}: {core_count} core, {common_count} common symptoms")
    print()
    
    # Get pairs to process
    if args.user:
        pairs = get_pairs_for_user(args.user)
        print(f"Found {len(pairs)} diagnosis records for user {args.user}")
    else:
        pairs = get_all_diagnosis_video_pairs()
        print(f"Found {len(pairs)} total diagnosis records")
    
    if not pairs:
        print("No diagnoses found to process.")
        return
    
    # Group by condition for stats
    conditions = {}
    for p in pairs:
        cond = p['condition_code']
        conditions[cond] = conditions.get(cond, 0) + 1
    
    print("\nDiagnoses by condition:")
    for cond, count in sorted(conditions.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {count}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would recalculate concordance for these records.")
        return
    
    # Clear existing concordance
    print("\nClearing existing concordance records...")
    if args.user:
        deleted = clear_concordance(username=args.user)
    else:
        deleted = clear_concordance()
    print(f"  Deleted {deleted} existing concordance records")
    
    if args.clear_only:
        print("Clear only mode - not recalculating.")
        return
    
    # Recalculate
    print("\nRecalculating concordance...")
    success = 0
    skipped = 0
    errors = 0
    
    for i, pair in enumerate(pairs, 1):
        video_id = pair['video_id']
        diagnosis_id = pair['diagnosis_id']
        condition = pair['condition_code']
        
        if i % 100 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] Processing...")
        
        # Check if condition has expected symptoms
        expected = get_expected_symptoms(condition)
        if not expected.get('all'):
            skipped += 1
            continue
        
        try:
            result = calculate_symptom_concordance(video_id, diagnosis_id)
            success += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error for video {video_id}, diagnosis {diagnosis_id}: {e}")
    
    print(f"\nComplete!")
    print(f"  Success: {success}")
    print(f"  Skipped (no expected symptoms for condition): {skipped}")
    print(f"  Errors: {errors}")


if __name__ == '__main__':
    main()
