#!/usr/bin/env python3
"""
Display database statistics and insights.

Usage:
    python scripts/stats.py
    python scripts/stats.py --detailed
"""
import sys
import argparse
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_symptom_statistics, get_connection


def main():
    parser = argparse.ArgumentParser(description='Display database statistics')
    parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    parser.add_argument('--export-json', help='Export statistics to JSON file')

    args = parser.parse_args()

    try:
        stats = get_symptom_statistics()

        print("\n" + "="*80)
        print("DATABASE STATISTICS")
        print("="*80)

        print(f"\nOverview:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Total symptoms: {stats['total_symptoms']}")

        if stats['total_symptoms'] > 0:
            avg_symptoms_per_video = stats['total_symptoms'] / stats['total_videos']
            print(f"  Average symptoms per video: {avg_symptoms_per_video:.1f}")

        print(f"\nTop 10 Most Common Symptoms:")
        for i, symptom in enumerate(stats['top_symptoms'], 1):
            print(f"  {i:2}. {symptom['symptom']:40} "
                  f"(count: {symptom['count']}, avg conf: {symptom['avg_confidence']:.3f})")

        print(f"\nSymptoms by Category:")
        for cat in stats['by_category']:
            print(f"  {cat['category']:20} {cat['count']:5} symptoms")

        if args.detailed:
            # Get additional detailed stats
            with get_connection() as conn:
                cur = conn.cursor()

                # Videos with most symptoms
                cur.execute("""
                    SELECT v.title, COUNT(s.id) as symptom_count
                    FROM videos v
                    JOIN symptoms s ON v.id = s.video_id
                    GROUP BY v.id, v.title
                    ORDER BY symptom_count DESC
                    LIMIT 10
                """)
                top_videos = cur.fetchall()

                print(f"\nTop 10 Videos by Symptom Count:")
                for i, (title, count) in enumerate(top_videos, 1):
                    title_short = (title[:50] + '...') if len(title) > 50 else title
                    print(f"  {i:2}. {title_short:55} ({count} symptoms)")

                # High-confidence symptoms
                cur.execute("""
                    SELECT symptom, confidence, category
                    FROM symptoms
                    WHERE confidence >= 0.95
                    ORDER BY confidence DESC
                    LIMIT 10
                """)
                high_conf = cur.fetchall()

                if high_conf:
                    print(f"\nTop 10 Highest Confidence Symptoms:")
                    for i, (symptom, conf, category) in enumerate(high_conf, 1):
                        print(f"  {i:2}. {symptom:40} ({category}, conf: {conf:.3f})")

                # Transcription stats
                cur.execute("""
                    SELECT
                        COUNT(*) as total_transcripts,
                        AVG(word_count) as avg_word_count,
                        SUM(word_count) as total_words
                    FROM transcripts
                """)
                trans_stats = cur.fetchone()

                if trans_stats[0] > 0:
                    print(f"\nTranscription Statistics:")
                    print(f"  Total transcripts: {trans_stats[0]}")
                    print(f"  Average words per transcript: {trans_stats[1]:.0f}")
                    print(f"  Total words transcribed: {trans_stats[2]:,}")

        # Export to JSON if requested
        if args.export_json:
            with open(args.export_json, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            print(f"\n✓ Statistics exported to: {args.export_json}")

        print("\n" + "="*80)

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
