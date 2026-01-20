#!/usr/bin/env python3
"""
Initialize the database schema.

Usage:
    python scripts/init_db.py          # Create tables (if not exist)
    python scripts/init_db.py --reset  # Drop all tables and recreate
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import init_db, get_connection


def drop_all_tables():
    """Drop all tables in the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Get all table names
        cur.execute("""
            SELECT tablename FROM pg_tables 
            WHERE schemaname = 'public'
        """)
        tables = [row[0] for row in cur.fetchall()]
        
        if not tables:
            print("No tables to drop.")
            return
        
        print(f"Dropping {len(tables)} tables...")
        
        # Drop all tables with CASCADE
        for table in tables:
            print(f"  Dropping {table}...")
            cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')
        
        conn.commit()
        cur.close()
    print("All tables dropped.")


def main():
    parser = argparse.ArgumentParser(description='Initialize database schema')
    parser.add_argument('--reset', action='store_true',
                        help='Drop all tables and recreate (WARNING: deletes all data!)')
    args = parser.parse_args()
    
    if args.reset:
        print("=" * 60)
        print("WARNING: This will delete ALL data in the database!")
        print("=" * 60)
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm != 'yes':
            print("Aborted.")
            sys.exit(0)
        
        print("\nDropping all tables...")
        drop_all_tables()
        print()
    
    print("Initializing database schema...")
    try:
        init_db()
        print("\n[OK] Database initialized successfully!")
        print("\nYou can now:")
        print("  1. Discover videos: uv run python scripts/discover.py --hashtag EDS")
        print("  2. Run pipeline: uv run python pipeline.py run --urls-file urls.txt")
    except Exception as e:
        print(f"\n[ERROR] Error initializing database: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is installed and running")
        print("  2. DATABASE_URL is correctly configured in .env")
        print("  3. The database exists (createdb tiktok_disorders)")
        sys.exit(1)


if __name__ == '__main__':
    main()
