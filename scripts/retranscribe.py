#!/usr/bin/env python3
"""
Re-transcribe and re-extract all videos for complete dataset consistency.

This script:
1. Re-transcribes videos with improved medical vocabulary prompt
2. Applies 200+ post-processing corrections for medical terms
3. Clears old extraction data (symptoms, diagnoses, treatments, narrative elements)
4. Re-extracts using the configured LLM

Usage:
    # See what would be done (no changes)
    uv run python scripts/retranscribe.py --dry-run
    
    # Re-transcribe and re-extract all videos
    uv run python scripts/retranscribe.py --provider ollama --model gpt-oss:20b
    
    # Re-transcribe only (no re-extraction)
    uv run python scripts/retranscribe.py --transcribe-only
    
    # Re-transcribe with backup of old data
    uv run python scripts/retranscribe.py --backup --provider ollama --model gpt-oss:20b
    
    # Test with just 5 videos first
    uv run python scripts/retranscribe.py --limit 5 --provider ollama --model gpt-oss:20b
    
    # Re-process specific videos
    uv run python scripts/retranscribe.py --video-ids 1 2 3 --provider ollama --model gpt-oss:20b
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_connection, get_video_by_id
from transcriber import AudioTranscriber, _get_author_dir, _apply_transcription_corrections
from extractor import SymptomExtractor
from config import TRANSCRIPT_DIR


def get_videos_with_transcripts(limit: int = None, video_ids: list = None):
    """Get all videos that have transcripts."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        if video_ids:
            placeholders = ','.join(['%s'] * len(video_ids))
            cur.execute(f"""
                SELECT v.id, v.audio_path, v.author, t.id as transcript_id, t.text
                FROM videos v
                JOIN transcripts t ON v.id = t.video_id
                WHERE v.id IN ({placeholders})
                ORDER BY v.id
            """, video_ids)
        else:
            query = """
                SELECT v.id, v.audio_path, v.author, t.id as transcript_id, t.text
                FROM videos v
                JOIN transcripts t ON v.id = t.video_id
                ORDER BY v.id
            """
            if limit:
                query += f" LIMIT {limit}"
            cur.execute(query)
        
        return cur.fetchall()


def backup_transcript(video_id: int, transcript_id: int, text: str, author: str):
    """Save a backup of the old transcript."""
    backup_dir = TRANSCRIPT_DIR / '_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = backup_dir / f"transcript_{video_id}_backup_{timestamp}.json"
    
    backup_data = {
        'video_id': video_id,
        'transcript_id': transcript_id,
        'author': author,
        'text': text,
        'backed_up_at': datetime.now().isoformat()
    }
    
    with open(backup_file, 'w', encoding='utf-8') as f:
        json.dump(backup_data, f, indent=2, ensure_ascii=False)
    
    return backup_file


def backup_extractions(video_id: int):
    """Backup and return old extraction data before clearing."""
    backup_dir = TRANSCRIPT_DIR / '_backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Get old symptoms
        cur.execute("SELECT * FROM symptoms WHERE video_id = %s", (video_id,))
        columns = [desc[0] for desc in cur.description]
        symptoms = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Get old diagnoses
        cur.execute("SELECT * FROM claimed_diagnoses WHERE video_id = %s", (video_id,))
        columns = [desc[0] for desc in cur.description]
        diagnoses = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Get old treatments
        cur.execute("SELECT * FROM treatments WHERE video_id = %s", (video_id,))
        columns = [desc[0] for desc in cur.description]
        treatments = [dict(zip(columns, row)) for row in cur.fetchall()]
        
        # Get old narrative elements
        cur.execute("SELECT * FROM narrative_elements WHERE video_id = %s", (video_id,))
        columns = [desc[0] for desc in cur.description]
        narrative = [dict(zip(columns, row)) for row in cur.fetchall()]
    
    if symptoms or diagnoses or treatments or narrative:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = backup_dir / f"extractions_{video_id}_backup_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        def serialize(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            return str(obj)
        
        backup_data = {
            'video_id': video_id,
            'symptoms': symptoms,
            'diagnoses': diagnoses,
            'treatments': treatments,
            'narrative_elements': narrative,
            'backed_up_at': datetime.now().isoformat()
        }
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False, default=serialize)
        
        return backup_file
    
    return None


def clear_extractions(video_id: int):
    """Clear all extraction data for a video."""
    with get_connection() as conn:
        cur = conn.cursor()
        
        # Delete in order to respect foreign keys
        cur.execute("DELETE FROM symptom_concordance WHERE diagnosis_id IN (SELECT id FROM claimed_diagnoses WHERE video_id = %s)", (video_id,))
        cur.execute("DELETE FROM narrative_elements WHERE video_id = %s", (video_id,))
        cur.execute("DELETE FROM treatments WHERE video_id = %s", (video_id,))
        cur.execute("DELETE FROM claimed_diagnoses WHERE video_id = %s", (video_id,))
        cur.execute("DELETE FROM symptoms WHERE video_id = %s", (video_id,))
        
        conn.commit()
        
        return True


def update_transcript_in_db(transcript_id: int, new_text: str, model: str):
    """Update the transcript text in the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE transcripts 
            SET text = %s, 
                model = %s,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (new_text, model, transcript_id))
        conn.commit()


def retranscribe_video(transcriber: AudioTranscriber, video_id: int, audio_path: str, 
                       author: str, old_transcript_id: int, old_text: str,
                       do_backup: bool = False, dry_run: bool = False):
    """Re-transcribe a single video."""
    
    # Check if audio file exists
    if not audio_path or not Path(audio_path).exists():
        return {
            'video_id': video_id,
            'success': False,
            'error': f'Audio file not found: {audio_path}'
        }
    
    if do_backup and not dry_run:
        backup_file = backup_transcript(video_id, old_transcript_id, old_text, author)
        print(f"    Backed up to: {backup_file}")
    
    if dry_run:
        # Just show what the corrections would do to existing text
        corrected = _apply_transcription_corrections(old_text)
        changes = corrected != old_text
        return {
            'video_id': video_id,
            'success': True,
            'dry_run': True,
            'would_change': changes,
            'old_length': len(old_text),
            'new_length': len(corrected) if changes else len(old_text)
        }
    
    # Actually re-transcribe
    print(f"    Transcribing: {audio_path}")
    
    try:
        # Use faster-whisper transcribe directly (bypass the database check)
        from transcriber import MEDICAL_VOCABULARY_PROMPT
        
        segments_iter, info = transcriber.model.transcribe(
            str(audio_path),
            language=None,
            beam_size=5,
            temperature=0.0,
            initial_prompt=MEDICAL_VOCABULARY_PROMPT,
        )
        
        text_parts = []
        for segment in segments_iter:
            corrected_text = _apply_transcription_corrections(segment.text)
            text_parts.append(corrected_text)
        
        new_text = " ".join(text_parts).strip()
        detected_language = info.language if info and info.language else 'en'
        
        # Update in database
        update_transcript_in_db(old_transcript_id, new_text, transcriber.model_size)
        
        # Also save to file
        author_dir = _get_author_dir(author or '_unknown')
        output_filename = f"transcript_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = author_dir / output_filename
        
        transcript_data = {
            'video_id': video_id,
            'text': new_text,
            'language': detected_language,
            'model': transcriber.model_size,
            'backend': transcriber.backend,
            'retranscribed_at': datetime.now().isoformat(),
            'word_count': len(new_text.split())
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)
        
        return {
            'video_id': video_id,
            'success': True,
            'old_length': len(old_text),
            'new_length': len(new_text),
            'word_count': len(new_text.split()),
            'output_file': str(output_path)
        }
        
    except Exception as e:
        return {
            'video_id': video_id,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Re-transcribe videos with improved medical vocabulary.'
    )
    parser.add_argument('--video-ids', type=int, nargs='+',
                        help='Specific video IDs to re-transcribe')
    parser.add_argument('--limit', type=int,
                        help='Limit to first N videos')
    parser.add_argument('--backup', action='store_true',
                        help='Backup old transcripts before overwriting')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--whisper-model', default='large-v3',
                        help='Whisper model to use (default: large-v3)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Re-transcribe Videos with Medical Vocabulary")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")
    
    # Get videos to process
    print("Finding videos with transcripts...")
    videos = get_videos_with_transcripts(limit=args.limit, video_ids=args.video_ids)
    
    if not videos:
        print("No videos found to re-transcribe.")
        return 0
    
    print(f"Found {len(videos)} video(s) to re-transcribe")
    
    if args.backup:
        print("Backup mode: Old transcripts will be saved to data/transcripts/_backups/")
    
    # Initialize transcriber (only if not dry-run, to save time)
    transcriber = None
    if not args.dry_run:
        print(f"\nLoading Whisper model '{args.whisper_model}'...")
        transcriber = AudioTranscriber(model_size=args.whisper_model)
    
    # Process videos
    print(f"\n{'=' * 60}")
    success_count = 0
    error_count = 0
    would_change_count = 0
    
    for i, (video_id, audio_path, author, transcript_id, old_text) in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Video {video_id} (@{author or 'unknown'})")
        
        result = retranscribe_video(
            transcriber=transcriber,
            video_id=video_id,
            audio_path=audio_path,
            author=author,
            old_transcript_id=transcript_id,
            old_text=old_text,
            do_backup=args.backup,
            dry_run=args.dry_run
        )
        
        if result['success']:
            success_count += 1
            if args.dry_run:
                if result.get('would_change'):
                    would_change_count += 1
                    print(f"    Would change: {result['old_length']} -> {result['new_length']} chars")
                else:
                    print(f"    No changes needed (corrections already match)")
            else:
                print(f"    Done: {result['word_count']} words")
        else:
            error_count += 1
            print(f"    Error: {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    if args.dry_run:
        print(f"  Videos checked: {len(videos)}")
        print(f"  Would change: {would_change_count}")
        print(f"  No changes needed: {success_count - would_change_count}")
        print(f"  Errors: {error_count}")
        print(f"\nRun without --dry-run to apply changes.")
    else:
        print(f"  Successfully re-transcribed: {success_count}")
        print(f"  Errors: {error_count}")
        if args.backup:
            print(f"  Backups saved to: data/transcripts/_backups/")
    
    print("=" * 60)
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
