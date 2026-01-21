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
                model_used = %s,
                word_count = %s,
                transcribed_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """, (new_text, model, len(new_text.split()), transcript_id))
        conn.commit()


def reprocess_video(transcriber: AudioTranscriber, extractor: SymptomExtractor,
                    video_id: int, audio_path: str, author: str, 
                    old_transcript_id: int, old_text: str,
                    do_backup: bool = False, dry_run: bool = False,
                    transcribe_only: bool = False):
    """Re-transcribe and optionally re-extract a single video."""
    
    result = {
        'video_id': video_id,
        'success': False,
        'transcribe': {},
        'extract': {}
    }
    
    # Check if audio file exists
    if not audio_path or not Path(audio_path).exists():
        result['error'] = f'Audio file not found: {audio_path}'
        return result
    
    # Backup old data
    if do_backup and not dry_run:
        backup_file = backup_transcript(video_id, old_transcript_id, old_text, author)
        print(f"    Backed up transcript to: {backup_file}")
        
        if not transcribe_only:
            extract_backup = backup_extractions(video_id)
            if extract_backup:
                print(f"    Backed up extractions to: {extract_backup}")
    
    if dry_run:
        # Just show what the corrections would do
        corrected = _apply_transcription_corrections(old_text)
        changes = corrected != old_text
        result['success'] = True
        result['dry_run'] = True
        result['transcribe'] = {
            'would_change': changes,
            'old_length': len(old_text),
            'new_length': len(corrected) if changes else len(old_text)
        }
        if not transcribe_only:
            result['extract'] = {'would_reextract': True}
        return result
    
    # ===== STAGE 1: Re-transcribe =====
    print(f"    [1/2] Transcribing: {Path(audio_path).name}")
    
    try:
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
        
        # Save to file
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
        
        result['transcribe'] = {
            'success': True,
            'word_count': len(new_text.split()),
            'output_file': str(output_path)
        }
        print(f"        Done: {len(new_text.split())} words")
        
    except Exception as e:
        result['transcribe'] = {'success': False, 'error': str(e)}
        result['error'] = f'Transcription failed: {e}'
        return result
    
    # ===== STAGE 2: Re-extract (if not transcribe-only) =====
    if transcribe_only:
        result['success'] = True
        result['extract'] = {'skipped': True}
        return result
    
    print(f"    [2/2] Extracting symptoms, diagnoses, treatments...")
    
    try:
        # Clear old extraction data
        clear_extractions(video_id)
        
        # Re-extract with force=True to bypass the "already extracted" check
        extract_result = extractor.extract_all(video_id, force=True)
        
        result['extract'] = {
            'success': extract_result.get('success', False),
            'symptoms': extract_result.get('symptoms_saved', 0),
            'diagnoses': extract_result.get('diagnoses_saved', 0),
            'treatments': extract_result.get('treatments_saved', 0)
        }
        
        if extract_result.get('success'):
            print(f"        Done: {extract_result.get('symptoms_saved', 0)} symptoms, "
                  f"{extract_result.get('diagnoses_saved', 0)} diagnoses, "
                  f"{extract_result.get('treatments_saved', 0)} treatments")
            result['success'] = True
        else:
            result['error'] = f"Extraction failed: {extract_result.get('error', 'Unknown')}"
        
    except Exception as e:
        result['extract'] = {'success': False, 'error': str(e)}
        result['error'] = f'Extraction failed: {e}'
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Re-transcribe and re-extract videos for dataset consistency.'
    )
    parser.add_argument('--video-ids', type=int, nargs='+',
                        help='Specific video IDs to reprocess')
    parser.add_argument('--limit', type=int,
                        help='Limit to first N videos')
    parser.add_argument('--backup', action='store_true',
                        help='Backup old data before overwriting')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--transcribe-only', action='store_true',
                        help='Only re-transcribe, skip extraction')
    parser.add_argument('--whisper-model', default='large-v3',
                        help='Whisper model to use (default: large-v3)')
    parser.add_argument('--provider', default='ollama',
                        help='LLM provider for extraction (anthropic or ollama)')
    parser.add_argument('--model', default='gpt-oss:20b',
                        help='LLM model for extraction (default: gpt-oss:20b)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Re-transcribe and Re-extract Videos for Dataset Consistency")
    print("=" * 70)
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")
    
    if args.transcribe_only:
        print("Mode: Transcribe only (extraction will be skipped)")
    else:
        print(f"Mode: Full reprocess (transcribe + extract with {args.provider}/{args.model})")
    
    # Get videos to process
    print("\nFinding videos with transcripts...")
    videos = get_videos_with_transcripts(limit=args.limit, video_ids=args.video_ids)
    
    if not videos:
        print("No videos found to reprocess.")
        return 0
    
    print(f"Found {len(videos)} video(s) to reprocess")
    
    if args.backup:
        print("Backup mode: Old data will be saved to data/transcripts/_backups/")
    
    # Initialize components (only if not dry-run)
    transcriber = None
    extractor = None
    
    if not args.dry_run:
        print(f"\nLoading Whisper model '{args.whisper_model}'...")
        transcriber = AudioTranscriber(model_size=args.whisper_model)
        
        if not args.transcribe_only:
            print(f"Initializing extractor ({args.provider}/{args.model})...")
            extractor = SymptomExtractor(
                provider=args.provider,
                model=args.model,
                max_workers=1  # Sequential for reprocessing to avoid overwhelming
            )
    
    # Process videos
    print(f"\n{'=' * 70}")
    success_count = 0
    error_count = 0
    total_symptoms = 0
    total_diagnoses = 0
    total_treatments = 0
    
    for i, (video_id, audio_path, author, transcript_id, old_text) in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Video {video_id} (@{author or 'unknown'})")
        
        result = reprocess_video(
            transcriber=transcriber,
            extractor=extractor,
            video_id=video_id,
            audio_path=audio_path,
            author=author,
            old_transcript_id=transcript_id,
            old_text=old_text,
            do_backup=args.backup,
            dry_run=args.dry_run,
            transcribe_only=args.transcribe_only
        )
        
        if result['success']:
            success_count += 1
            if not args.dry_run and not args.transcribe_only:
                total_symptoms += result['extract'].get('symptoms', 0)
                total_diagnoses += result['extract'].get('diagnoses', 0)
                total_treatments += result['extract'].get('treatments', 0)
        else:
            error_count += 1
            print(f"    ERROR: {result.get('error', 'Unknown error')}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    
    if args.dry_run:
        print(f"  Videos to reprocess: {len(videos)}")
        print(f"  Mode: {'Transcribe only' if args.transcribe_only else 'Transcribe + Extract'}")
        print(f"\nRun without --dry-run to apply changes.")
    else:
        print(f"  Successfully reprocessed: {success_count}/{len(videos)}")
        print(f"  Errors: {error_count}")
        if not args.transcribe_only:
            print(f"\n  Total symptoms extracted: {total_symptoms}")
            print(f"  Total diagnoses extracted: {total_diagnoses}")
            print(f"  Total treatments extracted: {total_treatments}")
        if args.backup:
            print(f"\n  Backups saved to: data/transcripts/_backups/")
    
    print("=" * 70)
    return 0 if error_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
