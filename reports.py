"""
Comprehensive analysis and reporting for the EDS/MCAS/POTS research database.
Generates various reports and exports data for further analysis.
"""
import sys
sys.path.insert(0, '.')

from database import get_connection
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List
import json
from datetime import datetime
from pathlib import Path
import csv


def get_full_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics from the database."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        stats = {}

        # Basic counts
        cur.execute("SELECT COUNT(*) as count FROM videos")
        stats['total_videos'] = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM transcripts")
        stats['total_transcripts'] = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM symptoms")
        stats['total_symptoms'] = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM claimed_diagnoses")
        stats['total_diagnoses'] = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM treatments")
        stats['total_treatments'] = cur.fetchone()['count']

        # Average transcript quality
        cur.execute("""
            SELECT AVG(quality_score) as avg_quality,
                   AVG(medical_term_density) as avg_medical_density,
                   AVG(clarity_score) as avg_clarity
            FROM transcript_quality
        """)
        quality = cur.fetchone()
        stats['avg_transcript_quality'] = float(quality['avg_quality']) if quality['avg_quality'] else None
        stats['avg_medical_density'] = float(quality['avg_medical_density']) if quality['avg_medical_density'] else None

        # Videos by platform
        cur.execute("""
            SELECT platform, COUNT(*) as count
            FROM videos
            GROUP BY platform
            ORDER BY count DESC
        """)
        stats['by_platform'] = [dict(r) for r in cur.fetchall()]

        return stats


def report_diagnoses() -> Dict[str, Any]:
    """Detailed diagnosis analysis."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Diagnoses by condition with confidence stats
        cur.execute("""
            SELECT condition_code,
                   COUNT(*) as count,
                   AVG(confidence) as avg_confidence,
                   MIN(confidence) as min_confidence,
                   MAX(confidence) as max_confidence
            FROM claimed_diagnoses
            GROUP BY condition_code
            ORDER BY count DESC
        """)
        by_condition = [dict(r) for r in cur.fetchall()]

        # Diagnosis type distribution
        cur.execute("""
            SELECT condition_name, COUNT(*) as count
            FROM claimed_diagnoses
            GROUP BY condition_name
            ORDER BY count DESC
        """)
        by_type = [dict(r) for r in cur.fetchall()]

        # Co-occurring diagnoses (from comorbidity table)
        cur.execute("""
            SELECT condition_a, condition_b, video_count
            FROM comorbidity_pairs
            ORDER BY video_count DESC
            LIMIT 20
        """)
        comorbidities = [dict(r) for r in cur.fetchall()]

        # Concordance analysis
        cur.execute("""
            SELECT cd.condition_code,
                   AVG(sc.concordance_score) as avg_concordance,
                   AVG(sc.core_symptom_score) as avg_core_score,
                   COUNT(*) as sample_size
            FROM symptom_concordance sc
            JOIN claimed_diagnoses cd ON sc.diagnosis_id = cd.id
            GROUP BY cd.condition_code
            ORDER BY avg_concordance DESC
        """)
        concordance = [dict(r) for r in cur.fetchall()]

        return {
            'by_condition': by_condition,
            'by_type': by_type,
            'comorbidities': comorbidities,
            'concordance': concordance
        }


def report_symptoms() -> Dict[str, Any]:
    """Detailed symptom analysis."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Most common symptoms
        cur.execute("""
            SELECT symptom, category, COUNT(*) as count,
                   AVG(confidence) as avg_confidence
            FROM symptoms
            GROUP BY symptom, category
            ORDER BY count DESC
            LIMIT 50
        """)
        top_symptoms = [dict(r) for r in cur.fetchall()]

        # Symptoms by category
        cur.execute("""
            SELECT category, COUNT(*) as count,
                   AVG(confidence) as avg_confidence
            FROM symptoms
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = [dict(r) for r in cur.fetchall()]

        # Severity distribution
        cur.execute("""
            SELECT severity, COUNT(*) as count
            FROM symptoms
            WHERE severity IS NOT NULL
            GROUP BY severity
            ORDER BY count DESC
        """)
        by_severity = [dict(r) for r in cur.fetchall()]

        # Temporal patterns
        cur.execute("""
            SELECT temporal_pattern, COUNT(*) as count
            FROM symptoms
            WHERE temporal_pattern IS NOT NULL
            GROUP BY temporal_pattern
            ORDER BY count DESC
        """)
        by_temporal = [dict(r) for r in cur.fetchall()]

        # Symptoms per condition (via concordance)
        cur.execute("""
            SELECT cd.condition_code,
                   COUNT(DISTINCT s.symptom) as unique_symptoms,
                   COUNT(s.id) as total_mentions
            FROM symptoms s
            JOIN claimed_diagnoses cd ON s.video_id = cd.video_id
            GROUP BY cd.condition_code
            ORDER BY unique_symptoms DESC
        """)
        symptoms_per_condition = [dict(r) for r in cur.fetchall()]

        return {
            'top_symptoms': top_symptoms,
            'by_category': by_category,
            'by_severity': by_severity,
            'by_temporal': by_temporal,
            'per_condition': symptoms_per_condition
        }


def report_treatments() -> Dict[str, Any]:
    """Detailed treatment analysis."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Most mentioned treatments
        cur.execute("""
            SELECT treatment_name, treatment_type,
                   COUNT(*) as mention_count,
                   AVG(confidence) as avg_confidence
            FROM treatments
            GROUP BY treatment_name, treatment_type
            ORDER BY mention_count DESC
            LIMIT 30
        """)
        top_treatments = [dict(r) for r in cur.fetchall()]

        # By type
        cur.execute("""
            SELECT treatment_type, COUNT(*) as count
            FROM treatments
            GROUP BY treatment_type
            ORDER BY count DESC
        """)
        by_type = [dict(r) for r in cur.fetchall()]

        # Effectiveness ratings
        cur.execute("""
            SELECT treatment_name, effectiveness, COUNT(*) as count
            FROM treatments
            WHERE effectiveness != 'unspecified'
            GROUP BY treatment_name, effectiveness
            ORDER BY treatment_name, count DESC
        """)
        effectiveness_raw = cur.fetchall()

        # Aggregate effectiveness by treatment
        effectiveness_by_treatment = {}
        for row in effectiveness_raw:
            name = row['treatment_name']
            if name not in effectiveness_by_treatment:
                effectiveness_by_treatment[name] = {}
            effectiveness_by_treatment[name][row['effectiveness']] = row['count']

        # Treatments by target condition
        cur.execute("""
            SELECT target_condition, treatment_name, COUNT(*) as count
            FROM treatments
            WHERE target_condition IS NOT NULL
            GROUP BY target_condition, treatment_name
            ORDER BY target_condition, count DESC
        """)
        by_condition = [dict(r) for r in cur.fetchall()]

        # Side effects mentioned
        cur.execute("""
            SELECT treatment_name, side_effects
            FROM treatments
            WHERE side_effects IS NOT NULL AND array_length(side_effects, 1) > 0
        """)
        side_effects = {}
        for row in cur.fetchall():
            name = row['treatment_name']
            if name not in side_effects:
                side_effects[name] = []
            side_effects[name].extend(row['side_effects'])

        # Deduplicate and count side effects
        for name in side_effects:
            from collections import Counter
            side_effects[name] = dict(Counter(side_effects[name]))

        return {
            'top_treatments': top_treatments,
            'by_type': by_type,
            'effectiveness': effectiveness_by_treatment,
            'by_condition': by_condition,
            'side_effects': side_effects
        }


def report_creators() -> Dict[str, Any]:
    """Analysis by video creator/author."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Top creators by video count
        cur.execute("""
            SELECT author, COUNT(*) as video_count,
                   SUM(view_count) as total_views,
                   AVG(duration) as avg_duration
            FROM videos
            WHERE author IS NOT NULL
            GROUP BY author
            ORDER BY video_count DESC
            LIMIT 20
        """)
        top_creators = [dict(r) for r in cur.fetchall()]

        # Creators by condition focus
        cur.execute("""
            SELECT v.author, cd.condition_code, COUNT(*) as videos
            FROM videos v
            JOIN claimed_diagnoses cd ON v.id = cd.video_id
            WHERE v.author IS NOT NULL
            GROUP BY v.author, cd.condition_code
            ORDER BY videos DESC
            LIMIT 50
        """)
        creators_by_condition = [dict(r) for r in cur.fetchall()]

        return {
            'top_creators': top_creators,
            'by_condition': creators_by_condition
        }


def export_symptoms_csv(output_path: str = "data/exports/symptoms.csv"):
    """Export all symptoms to CSV for external analysis."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT s.*, v.author, v.platform, v.url
            FROM symptoms s
            JOIN videos v ON s.video_id = v.id
            ORDER BY s.video_id, s.id
        """)
        rows = cur.fetchall()

        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Exported {len(rows)} symptoms to {output_path}")
            return len(rows)
    return 0


def export_diagnoses_csv(output_path: str = "data/exports/diagnoses.csv"):
    """Export all diagnoses to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT cd.*, v.author, v.platform, v.url,
                   sc.concordance_score, sc.core_symptom_score
            FROM claimed_diagnoses cd
            JOIN videos v ON cd.video_id = v.id
            LEFT JOIN symptom_concordance sc ON cd.id = sc.diagnosis_id
            ORDER BY cd.video_id, cd.id
        """)
        rows = cur.fetchall()

        if rows:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Exported {len(rows)} diagnoses to {output_path}")
            return len(rows)
    return 0


def export_treatments_csv(output_path: str = "data/exports/treatments.csv"):
    """Export all treatments to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT t.*, v.author, v.platform
            FROM treatments t
            JOIN videos v ON t.video_id = v.id
            ORDER BY t.video_id, t.id
        """)
        rows = cur.fetchall()

        if rows:
            # Convert arrays to strings for CSV
            for row in rows:
                if row.get('side_effects'):
                    row['side_effects'] = '; '.join(row['side_effects'])
                if row.get('target_symptoms'):
                    row['target_symptoms'] = '; '.join(row['target_symptoms'])

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Exported {len(rows)} treatments to {output_path}")
            return len(rows)
    return 0


def generate_full_report(output_dir: str = "data/reports"):
    """Generate a complete analysis report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Generating comprehensive analysis report...")
    print("=" * 60)

    report = {
        'generated_at': datetime.now().isoformat(),
        'statistics': get_full_statistics(),
        'diagnoses': report_diagnoses(),
        'symptoms': report_symptoms(),
        'treatments': report_treatments(),
        'creators': report_creators()
    }

    # Save full JSON report
    json_path = f"{output_dir}/full_report_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report saved to: {json_path}")

    # Print summary
    stats = report['statistics']
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Total transcripts: {stats['total_transcripts']}")
    print(f"  Total symptoms extracted: {stats['total_symptoms']}")
    print(f"  Total diagnoses extracted: {stats['total_diagnoses']}")
    print(f"  Total treatments extracted: {stats['total_treatments']}")

    if stats['avg_transcript_quality']:
        print(f"  Avg transcript quality: {stats['avg_transcript_quality']:.2f}")

    print(f"\nPlatform breakdown:")
    for p in stats['by_platform']:
        print(f"  {p['platform']}: {p['count']} videos")

    diag = report['diagnoses']
    if diag['by_condition']:
        print(f"\nTop diagnoses:")
        for d in diag['by_condition'][:5]:
            print(f"  {d['condition_code']}: {d['count']} (avg conf: {d['avg_confidence']:.2f})")

    if diag['concordance']:
        print(f"\nConcordance scores (symptom match):")
        for c in diag['concordance'][:5]:
            print(f"  {c['condition_code']}: {float(c['avg_concordance']):.2f} ({c['sample_size']} samples)")

    symptoms = report['symptoms']
    if symptoms['by_category']:
        print(f"\nSymptoms by category:")
        for s in symptoms['by_category'][:5]:
            print(f"  {s['category']}: {s['count']}")

    treatments = report['treatments']
    if treatments['top_treatments']:
        print(f"\nTop treatments:")
        for t in treatments['top_treatments'][:5]:
            print(f"  {t['treatment_name']} ({t['treatment_type']}): {t['mention_count']} mentions")

    print('='*60)

    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate analysis reports')
    parser.add_argument('--full', action='store_true', help='Generate full JSON report')
    parser.add_argument('--export-csv', action='store_true', help='Export data to CSV files')
    parser.add_argument('--diagnoses', action='store_true', help='Show diagnosis report')
    parser.add_argument('--symptoms', action='store_true', help='Show symptoms report')
    parser.add_argument('--treatments', action='store_true', help='Show treatments report')
    parser.add_argument('--creators', action='store_true', help='Show creators report')

    args = parser.parse_args()

    if args.export_csv:
        print("Exporting data to CSV...")
        export_symptoms_csv()
        export_diagnoses_csv()
        export_treatments_csv()
        print("Done!")
    elif args.diagnoses:
        result = report_diagnoses()
        print(json.dumps(result, indent=2, default=str))
    elif args.symptoms:
        result = report_symptoms()
        print(json.dumps(result, indent=2, default=str))
    elif args.treatments:
        result = report_treatments()
        print(json.dumps(result, indent=2, default=str))
    elif args.creators:
        result = report_creators()
        print(json.dumps(result, indent=2, default=str))
    else:
        # Default: generate full report
        generate_full_report()
