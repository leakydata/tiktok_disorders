"""
User-Level Analysis Module for Social Contagion Research

This module provides comprehensive analysis tools for studying individual TikTok creators'
health narratives over time. It supports the STRAIN framework by:
1. Tracking how users' claimed diagnoses evolve
2. Measuring symptom concordance consistency
3. Identifying narrative pattern changes
4. Detecting potential social contagion signals

Usage:
    uv run python user_analysis.py profile @username
    uv run python user_analysis.py timeline @username
    uv run python user_analysis.py concordance-report
    uv run python user_analysis.py low-concordance --threshold 0.3
    uv run python user_analysis.py export-all --output data/user_profiles.json
"""

import sys
sys.path.insert(0, '.')

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from database import (
    get_user_profile, get_user_timeline, get_user_concordance_over_time,
    get_all_users_summary, get_diagnosis_acquisition_patterns,
    get_symptom_consistency_analysis, get_users_with_low_concordance,
    get_connection
)
from psycopg2.extras import RealDictCursor


def format_date(d) -> str:
    """Format a date/datetime object for display."""
    if d is None:
        return "N/A"
    if isinstance(d, str):
        return d[:10]
    return d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)


def print_user_profile(username: str, verbose: bool = False):
    """Print comprehensive user profile to console."""
    profile = get_user_profile(username)
    
    if not profile:
        print(f"No data found for user: {username}")
        return
    
    print("=" * 70)
    print(f"USER PROFILE: @{profile['username']}")
    print("=" * 70)
    
    # Basic stats
    print(f"\nBASIC STATISTICS")
    print(f"   Videos analyzed: {profile['video_count']}")
    print(f"   Date range: {format_date(profile['first_video_date'])} to {format_date(profile['last_video_date'])}")
    if profile.get('follower_count'):
        print(f"   Followers: {profile['follower_count']:,}")
    if profile.get('total_views'):
        print(f"   Total views: {profile['total_views']:,}")
    
    # Diagnoses
    if profile.get('diagnoses'):
        print(f"\nCLAIMED DIAGNOSES ({len(profile['diagnoses'])} unique)")
        for d in profile['diagnoses']:
            # Determine diagnosis source
            has_self = d.get('ever_self_diagnosed', False)
            has_pro = d.get('ever_professionally_diagnosed', False)
            if has_self and has_pro:
                source = "[BOTH]"  # Claimed both self and professional at different times
            elif has_self:
                source = "[SELF]"
            else:
                source = "[PRO]"
            
            video_count = d.get('video_count', d.get('mention_count', 0))
            print(f"   - {d['condition_code']}: {d['mention_count']}x mentions in {video_count} videos {source}")
            print(f"     First: {format_date(d['first_mentioned'])} -> Last: {format_date(d['last_mentioned'])}")
    
    # Concordance
    if profile.get('concordance'):
        print(f"\nCONCORDANCE SCORES")
        for c in profile['concordance']:
            score = c['avg_concordance'] or 0
            level = "[HIGH]" if score >= 0.5 else "[MED]" if score >= 0.3 else "[LOW]"
            print(f"   {level} {c['condition_code']}: {score:.2f} avg concordance")
            print(f"      Core symptom score: {c.get('avg_core_score', 0):.2f}")
            print(f"      Range: {c.get('min_concordance', 0):.2f} - {c.get('max_concordance', 0):.2f}")
    
    # Top symptoms
    if profile.get('symptoms'):
        print(f"\nTOP SYMPTOMS ({len(profile['symptoms'])} unique)")
        for s in profile['symptoms'][:50]:
            severities = s.get('severities_reported', [])
            sev_str = ", ".join([str(x) for x in severities if x]) if severities else "unspecified"
            print(f"   - {s['symptom']} ({s['category']}): {s['mention_count']}x - {sev_str}")
    
    # STRAIN indicators
    strain = profile.get('strain_indicators', {})
    if strain and strain.get('total_narratives', 0) > 0:
        print(f"\nSTRAIN INDICATORS (from {strain.get('total_narratives', 0)} videos)")
        print(f"   Self-diagnosis mentions: {strain.get('self_diagnosis_mentions', 0)}")
        print(f"   Professional diagnosis mentions: {strain.get('professional_diagnosis_mentions', 0)}")
        print(f"   Doctor dismissal/gaslighting: {strain.get('doctor_dismissal_mentions', 0) + strain.get('medical_gaslighting_mentions', 0)}")
        print(f"   Stress trigger mentions: {strain.get('stress_trigger_mentions', 0)}")
        print(f"   Online community mentions: {strain.get('online_community_mentions', 0)}")
        print(f"   Learned from TikTok: {strain.get('learned_from_tiktok_mentions', 0)}")
    
    # Treatments
    if profile.get('treatments'):
        print(f"\nTREATMENTS MENTIONED ({len(profile['treatments'])})")
        for t in profile['treatments'][:30]:
            eff = t.get('reported_effectiveness', [])
            eff_str = ", ".join([str(x) for x in eff if x]) if eff else "unspecified"
            print(f"   - {t['treatment_name']} ({t['treatment_type']}): {t['mention_count']}x - {eff_str}")
    
    print("\n" + "=" * 70)


def print_user_timeline(username: str):
    """Print chronological timeline for a user."""
    timeline = get_user_timeline(username)
    
    if not timeline:
        print(f"No timeline data found for user: {username}")
        return
    
    print("=" * 70)
    print(f"TIMELINE: @{username}")
    print("=" * 70)
    
    for entry in timeline:
        date_str = format_date(entry.get('upload_date'))
        print(f"\n📅 {date_str} | Video #{entry['video_id']}")
        
        if entry.get('title'):
            title = entry['title'][:60] + "..." if len(entry.get('title', '')) > 60 else entry.get('title', '')
            print(f"   {title}")
        
        if entry.get('diagnoses_mentioned'):
            print(f"   🏥 Diagnoses: {', '.join(entry['diagnoses_mentioned'])}")
        
        if entry.get('symptom_count'):
            categories = entry.get('symptom_categories', [])
            cat_str = ", ".join([str(c) for c in categories if c]) if categories else "N/A"
            print(f"   🩺 Symptoms: {entry['symptom_count']} ({cat_str})")
        
        if entry.get('content_type'):
            print(f"   📝 Content type: {entry['content_type']}")
        
        flags = []
        if entry.get('mentions_self_diagnosis'):
            flags.append("self-diagnosis")
        if entry.get('mentions_professional_diagnosis'):
            flags.append("professional diagnosis")
        if flags:
            print(f"   🚩 Flags: {', '.join(flags)}")
    
    print("\n" + "=" * 70)


def print_concordance_report():
    """Print comprehensive concordance analysis report."""
    users = get_all_users_summary()
    
    print("=" * 70)
    print("CONCORDANCE ANALYSIS REPORT")
    print("=" * 70)
    
    # Overall stats
    with_concordance = [u for u in users if u.get('avg_concordance') is not None]
    
    if not with_concordance:
        print("No concordance data available yet.")
        return
    
    avg_overall = sum(u['avg_concordance'] for u in with_concordance) / len(with_concordance)
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total users analyzed: {len(users)}")
    print(f"   Users with concordance data: {len(with_concordance)}")
    print(f"   Overall average concordance: {avg_overall:.2f}")
    
    # Distribution
    high = [u for u in with_concordance if u['avg_concordance'] >= 0.5]
    medium = [u for u in with_concordance if 0.3 <= u['avg_concordance'] < 0.5]
    low = [u for u in with_concordance if u['avg_concordance'] < 0.3]
    
    print(f"\n📈 CONCORDANCE DISTRIBUTION")
    print(f"   🟢 High (≥0.5): {len(high)} users ({len(high)/len(with_concordance)*100:.1f}%)")
    print(f"   🟡 Medium (0.3-0.5): {len(medium)} users ({len(medium)/len(with_concordance)*100:.1f}%)")
    print(f"   🔴 Low (<0.3): {len(low)} users ({len(low)/len(with_concordance)*100:.1f}%)")
    
    # Top high concordance users
    if high:
        print(f"\n🏆 TOP HIGH-CONCORDANCE USERS")
        sorted_high = sorted(high, key=lambda x: x['avg_concordance'], reverse=True)[:10]
        for u in sorted_high:
            print(f"   @{u['username']}: {u['avg_concordance']:.2f} ({u['video_count']} videos, {u['unique_diagnoses']} diagnoses)")
    
    # Users of concern (low concordance)
    if low:
        print(f"\n⚠️ LOW-CONCORDANCE USERS (potential social contagion)")
        sorted_low = sorted(low, key=lambda x: x['avg_concordance'])[:15]
        for u in sorted_low:
            print(f"   @{u['username']}: {u['avg_concordance']:.2f} ({u['video_count']} videos, {u['unique_diagnoses']} diagnoses)")
    
    print("\n" + "=" * 70)


def print_low_concordance_users(threshold: float = 0.3):
    """Print users with consistently low concordance scores."""
    results = get_users_with_low_concordance(threshold)
    
    print("=" * 70)
    print(f"LOW CONCORDANCE USERS (threshold < {threshold})")
    print("=" * 70)
    
    if not results:
        print(f"No users found with concordance < {threshold}")
        return
    
    print(f"\nFound {len(results)} user-condition pairs with low concordance:\n")
    
    # Group by user
    by_user = {}
    for r in results:
        if r['username'] not in by_user:
            by_user[r['username']] = []
        by_user[r['username']].append(r)
    
    for username, conditions in by_user.items():
        print(f"@{username}")
        for c in conditions:
            self_diag = "🔴" if c.get('includes_self_diagnosed') else "🟢"
            print(f"   {self_diag} {c['condition_code']}: {c['avg_concordance']:.2f} concordance, "
                  f"{c['avg_core_score']:.2f} core score ({c['video_count']} videos)")
        print()
    
    print("=" * 70)


def print_diagnosis_patterns():
    """Print analysis of how users acquire diagnoses over time."""
    patterns = get_diagnosis_acquisition_patterns()
    
    print("=" * 70)
    print("DIAGNOSIS ACQUISITION PATTERNS")
    print("=" * 70)
    
    if not patterns:
        print("No diagnosis pattern data available.")
        return
    
    # Group by user
    by_user = {}
    for p in patterns:
        if p['username'] not in by_user:
            by_user[p['username']] = []
        by_user[p['username']].append(p)
    
    # Find users with multiple diagnoses
    multi_diag_users = {u: d for u, d in by_user.items() if len(d) > 1}
    
    print(f"\n📊 OVERVIEW")
    print(f"   Total users with diagnoses: {len(by_user)}")
    print(f"   Users with multiple diagnoses: {len(multi_diag_users)}")
    
    # Common diagnosis sequences
    print(f"\n🔄 COMMON DIAGNOSIS SEQUENCES")
    sequences = {}
    for user, diagnoses in multi_diag_users.items():
        sorted_diag = sorted(diagnoses, key=lambda x: x['first_mention_date'] or datetime.min)
        seq = " → ".join([d['condition_code'] for d in sorted_diag])
        sequences[seq] = sequences.get(seq, 0) + 1
    
    for seq, count in sorted(sequences.items(), key=lambda x: -x[1])[:15]:
        print(f"   {seq}: {count} users")
    
    # Example users with diagnosis progression
    print(f"\n👥 EXAMPLE USERS WITH DIAGNOSIS PROGRESSION")
    for username, diagnoses in list(multi_diag_users.items())[:10]:
        sorted_diag = sorted(diagnoses, key=lambda x: x['first_mention_date'] or datetime.min)
        print(f"\n   @{username}:")
        for d in sorted_diag:
            date_str = format_date(d['first_mention_date'])
            print(f"      {date_str}: {d['condition_code']}")
    
    print("\n" + "=" * 70)


def export_all_profiles(output_path: str):
    """Export all user profiles to JSON."""
    users = get_all_users_summary()
    
    all_profiles = []
    for i, user in enumerate(users):
        if user.get('username'):
            print(f"[{i+1}/{len(users)}] Processing @{user['username']}...")
            profile = get_user_profile(user['username'])
            if profile:
                # Convert dates to strings for JSON
                profile = json.loads(json.dumps(profile, default=str))
                all_profiles.append(profile)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Exported {len(all_profiles)} user profiles to {output_path}")


def print_symptom_consistency(username: str):
    """Print symptom consistency analysis for a user."""
    analysis = get_symptom_consistency_analysis(username)
    
    if not analysis or analysis['video_count'] == 0:
        print(f"No data found for user: {username}")
        return
    
    print("=" * 70)
    print(f"SYMPTOM CONSISTENCY ANALYSIS: @{username}")
    print("=" * 70)
    
    print(f"\n📊 OVERVIEW")
    print(f"   Videos analyzed: {analysis['video_count']}")
    print(f"   Unique symptoms: {analysis['total_unique_symptoms']}")
    print(f"   Symptoms with inconsistent severity: {analysis['symptoms_with_inconsistent_severity']}")
    
    if analysis['inconsistent_symptoms']:
        print(f"\n⚠️ INCONSISTENT SEVERITY REPORTING")
        for s in analysis['inconsistent_symptoms']:
            severities = s.get('all_severities', [])
            sev_str = ", ".join([str(x) for x in severities if x])
            print(f"   • {s['symptom']}: reported as {sev_str}")
            print(f"     Mentioned {s['times_mentioned']}x, first: {format_date(s['first_mentioned'])}, last: {format_date(s['last_mentioned'])}")
    
    print("\n" + "=" * 70)


def detect_narrative_inconsistencies(username: str) -> Dict[str, Any]:
    """Detect narrative inconsistencies for a user (conflicting claims across videos)."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        inconsistencies = {
            'username': username,
            'diagnosis_inconsistencies': [],
            'treatment_inconsistencies': [],
            'severity_inconsistencies': [],
            'total_flags': 0
        }
        
        # Check for diagnosis type conflicts (self vs professional across videos)
        cur.execute("""
            SELECT 
                cd.condition_code,
                BOOL_OR(cd.is_self_diagnosed) as ever_self_diagnosed,
                BOOL_OR(NOT cd.is_self_diagnosed) as ever_professionally_diagnosed,
                COUNT(DISTINCT CASE WHEN cd.is_self_diagnosed THEN v.id END) as self_diag_videos,
                COUNT(DISTINCT CASE WHEN NOT cd.is_self_diagnosed THEN v.id END) as prof_diag_videos
            FROM claimed_diagnoses cd
            JOIN videos v ON cd.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY cd.condition_code
            HAVING BOOL_OR(cd.is_self_diagnosed) AND BOOL_OR(NOT cd.is_self_diagnosed)
        """, (username,))
        
        for row in cur.fetchall():
            inconsistencies['diagnosis_inconsistencies'].append({
                'type': 'diagnosis_source_conflict',
                'condition': row['condition_code'],
                'description': f"Claims both self-diagnosed ({row['self_diag_videos']} videos) and professionally diagnosed ({row['prof_diag_videos']} videos)",
                'severity': 'high'
            })
        
        # Check for treatment effectiveness conflicts
        cur.execute("""
            SELECT 
                t.treatment_name,
                ARRAY_AGG(DISTINCT t.effectiveness) as effectiveness_claims,
                COUNT(DISTINCT t.effectiveness) as claim_variations
            FROM treatments t
            JOIN videos v ON t.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s) AND t.effectiveness IS NOT NULL
            GROUP BY t.treatment_name
            HAVING COUNT(DISTINCT t.effectiveness) > 1
        """, (username,))
        
        for row in cur.fetchall():
            claims = [str(c) for c in row['effectiveness_claims'] if c]
            # Flag if dramatically contradictory (helpful vs harmful)
            helpful = any('helpful' in str(c).lower() for c in claims)
            harmful = any('worse' in str(c).lower() or 'harmful' in str(c).lower() for c in claims)
            
            if helpful and harmful:
                inconsistencies['treatment_inconsistencies'].append({
                    'type': 'treatment_effectiveness_conflict',
                    'treatment': row['treatment_name'],
                    'claims': claims,
                    'description': f"Reports both helpful AND harmful effects for same treatment",
                    'severity': 'high'
                })
            else:
                inconsistencies['treatment_inconsistencies'].append({
                    'type': 'treatment_effectiveness_variation',
                    'treatment': row['treatment_name'],
                    'claims': claims,
                    'description': f"Varying effectiveness reports: {', '.join(claims)}",
                    'severity': 'low'
                })
        
        # Check for symptom severity inconsistencies (same symptom, wildly different severities)
        cur.execute("""
            SELECT 
                s.symptom,
                ARRAY_AGG(DISTINCT s.severity) as severities,
                COUNT(DISTINCT s.severity) as severity_count
            FROM symptoms s
            JOIN videos v ON s.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s) AND s.severity IS NOT NULL
            GROUP BY s.symptom
            HAVING COUNT(DISTINCT s.severity) > 1
        """, (username,))
        
        for row in cur.fetchall():
            severities = [str(s) for s in row['severities'] if s]
            # Flag if ranges from mild to severe
            has_mild = 'mild' in severities
            has_severe = 'severe' in severities
            
            if has_mild and has_severe:
                inconsistencies['severity_inconsistencies'].append({
                    'type': 'severity_range_extreme',
                    'symptom': row['symptom'],
                    'severities': severities,
                    'description': f"Reports same symptom as both 'mild' and 'severe'",
                    'severity': 'medium'
                })
        
        # Count total flags
        inconsistencies['total_flags'] = (
            len(inconsistencies['diagnosis_inconsistencies']) +
            len([t for t in inconsistencies['treatment_inconsistencies'] if t['severity'] == 'high']) +
            len(inconsistencies['severity_inconsistencies'])
        )
        
        return inconsistencies


def print_narrative_inconsistencies(username: str):
    """Print narrative inconsistency analysis for a user."""
    analysis = detect_narrative_inconsistencies(username)
    
    print("=" * 70)
    print(f"NARRATIVE CONSISTENCY ANALYSIS: @{username}")
    print("=" * 70)
    
    if analysis['total_flags'] == 0:
        print("\n✅ No significant inconsistencies detected.")
        print("=" * 70)
        return
    
    print(f"\n⚠️ TOTAL FLAGS: {analysis['total_flags']}")
    
    if analysis['diagnosis_inconsistencies']:
        print(f"\n🏥 DIAGNOSIS INCONSISTENCIES ({len(analysis['diagnosis_inconsistencies'])})")
        for inc in analysis['diagnosis_inconsistencies']:
            print(f"   🔴 {inc['condition']}: {inc['description']}")
    
    if analysis['treatment_inconsistencies']:
        high_sev = [t for t in analysis['treatment_inconsistencies'] if t['severity'] == 'high']
        low_sev = [t for t in analysis['treatment_inconsistencies'] if t['severity'] == 'low']
        
        if high_sev:
            print(f"\n💊 TREATMENT CONFLICTS (HIGH SEVERITY)")
            for inc in high_sev:
                print(f"   🔴 {inc['treatment']}: {inc['description']}")
        
        if low_sev:
            print(f"\n💊 TREATMENT VARIATIONS (LOW SEVERITY)")
            for inc in low_sev[:5]:
                print(f"   🟡 {inc['treatment']}: {inc['description']}")
    
    if analysis['severity_inconsistencies']:
        print(f"\n📊 SEVERITY INCONSISTENCIES ({len(analysis['severity_inconsistencies'])})")
        for inc in analysis['severity_inconsistencies'][:10]:
            print(f"   🟡 {inc['symptom']}: {inc['description']}")
    
    print("\n" + "=" * 70)


def refresh_user_data(username: str):
    """Refresh all longitudinal tracking data for a user."""
    from database import update_user_profile, update_diagnosis_timeline, update_symptom_consistency
    
    print(f"Refreshing data for @{username}...")
    
    profile = update_user_profile(username)
    if profile:
        print(f"   ✓ User profile updated")
    
    timeline = update_diagnosis_timeline(username)
    print(f"   ✓ Diagnosis timeline updated ({len(timeline)} entries)")
    
    consistency = update_symptom_consistency(username)
    print(f"   ✓ Symptom consistency updated ({len(consistency)} symptoms)")
    
    return {'profile': profile, 'timeline': timeline, 'consistency': consistency}


def refresh_all_data():
    """Refresh longitudinal tracking data for all users."""
    from database import refresh_all_user_profiles
    
    print("Refreshing all user profiles...")
    count = refresh_all_user_profiles()
    print(f"✓ Updated {count} user profiles")
    return count


def main():
    parser = argparse.ArgumentParser(
        description="User-level analysis for social contagion research"
    )
    subparsers = parser.add_subparsers(dest='command', help='Analysis command')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Show user profile')
    profile_parser.add_argument('username', help='TikTok username (with or without @)')
    profile_parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    # Timeline command
    timeline_parser = subparsers.add_parser('timeline', help='Show user timeline')
    timeline_parser.add_argument('username', help='TikTok username')
    
    # Concordance report
    subparsers.add_parser('concordance-report', help='Show overall concordance analysis')
    
    # Low concordance users
    low_conc_parser = subparsers.add_parser('low-concordance', help='Find users with low concordance')
    low_conc_parser.add_argument('--threshold', type=float, default=0.3, help='Concordance threshold')
    
    # Diagnosis patterns
    subparsers.add_parser('diagnosis-patterns', help='Analyze diagnosis acquisition patterns')
    
    # Symptom consistency
    consistency_parser = subparsers.add_parser('consistency', help='Symptom consistency analysis')
    consistency_parser.add_argument('username', help='TikTok username')
    
    # Export all
    export_parser = subparsers.add_parser('export-all', help='Export all user profiles')
    export_parser.add_argument('--output', default='data/exports/user_profiles.json', help='Output file path')
    
    # Summary
    subparsers.add_parser('summary', help='Show summary of all users')
    
    # Narrative inconsistencies
    inconsist_parser = subparsers.add_parser('inconsistencies', help='Detect narrative inconsistencies')
    inconsist_parser.add_argument('username', help='TikTok username')
    
    # Refresh user data
    refresh_parser = subparsers.add_parser('refresh', help='Refresh longitudinal tracking data')
    refresh_parser.add_argument('--username', help='Specific user (omit for all users)')
    
    args = parser.parse_args()
    
    if args.command == 'profile':
        username = args.username.lstrip('@')
        print_user_profile(username, args.verbose)
    
    elif args.command == 'timeline':
        username = args.username.lstrip('@')
        print_user_timeline(username)
    
    elif args.command == 'concordance-report':
        print_concordance_report()
    
    elif args.command == 'low-concordance':
        print_low_concordance_users(args.threshold)
    
    elif args.command == 'diagnosis-patterns':
        print_diagnosis_patterns()
    
    elif args.command == 'consistency':
        username = args.username.lstrip('@')
        print_symptom_consistency(username)
    
    elif args.command == 'export-all':
        export_all_profiles(args.output)
    
    elif args.command == 'summary':
        users = get_all_users_summary()
        print("=" * 70)
        print("ALL USERS SUMMARY")
        print("=" * 70)
        print(f"\nTotal users: {len(users)}\n")
        
        for u in users[:50]:
            conc = f"{u['avg_concordance']:.2f}" if u.get('avg_concordance') else "N/A"
            print(f"@{u['username']}: {u['video_count']} videos, {u['unique_diagnoses']} diagnoses, "
                  f"{u['unique_symptoms']} symptoms, concordance: {conc}")
        
        if len(users) > 50:
            print(f"\n... and {len(users) - 50} more users")
    
    elif args.command == 'inconsistencies':
        username = args.username.lstrip('@')
        print_narrative_inconsistencies(username)
    
    elif args.command == 'refresh':
        if args.username:
            refresh_user_data(args.username.lstrip('@'))
        else:
            refresh_all_data()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
