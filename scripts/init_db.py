#!/usr/bin/env python3
"""
Initialize the database schema.

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import init_db


def main():
    print("Initializing database schema...")
    try:
        init_db()
        print("\n✓ Database initialized successfully!")
        print("\nYou can now:")
        print("  1. Download videos: python scripts/download.py --url <url>")
        print("  2. Run full pipeline: python scripts/run_pipeline.py --file urls.txt")
    except Exception as e:
        print(f"\n✗ Error initializing database: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is installed and running")
        print("  2. DATABASE_URL is correctly configured in .env")
        print("  3. The database exists (createdb tiktok_disorders)")
        sys.exit(1)


if __name__ == '__main__':
    main()
