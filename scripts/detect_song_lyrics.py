#!/usr/bin/env python3
"""
Script to detect and flag song lyrics in existing transcripts.

TikTok videos often have songs playing instead of the creator speaking.
This script uses heuristics + LLM to accurately classify each transcript
as either spoken content or song lyrics, then updates the database.

Optimized for accuracy over token efficiency (designed for local GPU inference).

This is a backfill script for already-transcribed videos. New videos
will be automatically checked during the extraction process.

Usage:
    # Check all unchecked transcripts
    uv run python scripts/detect_song_lyrics.py
    
    # Limit to first 100
    uv run python scripts/detect_song_lyrics.py --limit 100
    
    # Dry run (don't update database)
    uv run python scripts/detect_song_lyrics.py --dry-run
    
    # Show current statistics
    uv run python scripts/detect_song_lyrics.py --stats
    
    # Use a specific model
    uv run python scripts/detect_song_lyrics.py --model llama3:8b
"""
import argparse
import signal
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    if shutdown_requested:
        print("\n\nForce quitting...")
        sys.exit(1)
    shutdown_requested = True
    print("\n\nCtrl+C detected - finishing current tasks and stopping...")
    print("  (Press Ctrl+C again to force quit)\n")

from config import OLLAMA_URL, OLLAMA_MODEL
from database import (
    get_transcripts_needing_song_check,
    update_transcript_song_lyrics_ratio,
    get_song_lyrics_stats
)


def detect_song_lyrics_llm(transcript_text: str, model: str, ollama_url: str, verbose: bool = False) -> tuple[float, bool]:
    """
    Use LLM to determine what percentage of transcript is song lyrics.
    
    Returns (song_lyrics_ratio, success).
    - song_lyrics_ratio: 0.0 (pure spoken) to 1.0 (pure lyrics)
    - success: True if LLM returned valid response, False on error
    """
    # Allow more context since we're running locally on RTX 4090
    if len(transcript_text) > 5000:
        transcript_text = transcript_text[:5000] + "..."
    
    if verbose:
        print(f"    [LLM] Sending {len(transcript_text)} chars to {model}")
    
    prompt = f"""Analyze this transcript from a TikTok video. Estimate what PERCENTAGE of the content is SONG LYRICS vs SPOKEN WORDS from a person.

SONG LYRICS characteristics:
- Repetitive phrases or choruses
- Rhyming patterns typical of music
- Poetic/emotional/abstract language typical of songs
- Music-related sounds like "yeah", "oh", "baby", "la la", repeated syllables
- Lacks conversational structure

SPOKEN CONTENT characteristics:
- Conversational language with a clear speaker
- Personal stories, experiences, or opinions
- Medical/health information (symptoms, diagnoses, treatments)
- Questions, answers, or direct address to viewers
- Natural speech with filler words like "um", "like", "you know"
- First-person narrative ("I went to", "My diagnosis", "I've been dealing with")

MIXED CONTENT examples:
- Someone talking over background music = mostly spoken (10-30% lyrics)
- Someone commenting briefly then playing a song = mixed (40-60%)
- Someone lip-syncing with occasional comments = mostly lyrics (70-90%)

Transcript:
---
{transcript_text}
---

What percentage of this transcript is SONG LYRICS (not spoken words)?
Answer with ONLY a number from 0 to 100 (no % sign, no other text).
Examples: 0, 15, 50, 85, 100"""

    import re
    
    def extract_number(text: str) -> float:
        """Extract a number (0-100) from text and convert to ratio (0.0-1.0)."""
        # Look for numbers in the text
        numbers = re.findall(r'\b(\d{1,3})\b', text)
        for num_str in numbers:
            num = int(num_str)
            if 0 <= num <= 100:
                return num / 100.0
        return None
    
    try:
        start_time = time.time()
        # Use /api/chat endpoint (same as extractor.py)
        response = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "num_ctx": 32768,  # Match extractor settings
                    "num_predict": 500,  # Generous room for reasoning + answer
                    "temperature": 0.0,
                }
            },
            timeout=120,
        )
        elapsed_ms = (time.time() - start_time) * 1000
        response.raise_for_status()
        
        payload = response.json()
        message = payload.get("message", {})
        answer = message.get("content", "").strip()
        thinking = message.get("thinking", "").strip()
        
        if verbose:
            print(f"    [LLM] Response in {elapsed_ms:.0f}ms")
            print(f"    [LLM] Content: '{answer[:100]}'" if answer else "    [LLM] Content: (empty)")
            if thinking:
                thinking_preview = thinking[:200] + "..." if len(thinking) > 200 else thinking
                print(f"    [LLM] Thinking: {thinking_preview}")
        
        # Try to extract number from answer first
        ratio = extract_number(answer) if answer else None
        
        # If no number in content, try thinking field (for reasoning models)
        if ratio is None and thinking:
            # Look for conclusion patterns in thinking
            # Check for explicit percentage mentions
            ratio = extract_number(thinking[-100:])  # Check end of thinking
            if ratio is None:
                # Fallback: look for keywords to estimate
                thinking_lower = thinking.lower()
                if "pure spoken" in thinking_lower or "0%" in thinking_lower or "no lyrics" in thinking_lower:
                    ratio = 0.0
                elif "pure lyrics" in thinking_lower or "100%" in thinking_lower or "all lyrics" in thinking_lower:
                    ratio = 1.0
                elif "mostly spoken" in thinking_lower:
                    ratio = 0.2
                elif "mostly lyrics" in thinking_lower:
                    ratio = 0.8
                elif "mixed" in thinking_lower or "50" in thinking_lower:
                    ratio = 0.5
        
        if ratio is not None:
            if verbose:
                print(f"    [LLM] Extracted ratio: {ratio:.0%}")
            return ratio, True
        
        # Couldn't parse - log and return failure
        print(f"  Could not extract number from LLM response: '{answer[:50] if answer else '(empty)'}...'")
        if thinking:
            print(f"      Thinking excerpt: '{thinking[-100:]}'")
        return 0.0, False
            
    except requests.exceptions.Timeout:
        print(f"  LLM timeout (model may be too slow)")
        return 0.0, False
    except Exception as e:
        print(f"  LLM error: {e}")
        return 0.0, False


def detect_song_lyrics_heuristic(transcript_text: str, verbose: bool = False) -> tuple[float, float]:
    """
    Use heuristics to estimate song lyrics ratio (no LLM needed).
    
    Returns (song_lyrics_ratio, confidence).
    - song_lyrics_ratio: 0.0 (pure spoken) to 1.0 (pure lyrics)
    - confidence: How sure we are about this estimate (0.0-1.0)
    """
    words = transcript_text.lower().split()
    word_count = len(words)
    text_lower = transcript_text.lower()
    
    if verbose:
        print(f"    [Heuristic] Analyzing {word_count} words")
    
    if word_count < 10:
        if verbose:
            print(f"    [Heuristic] Too short ({word_count} words) -> assuming spoken (low confidence)")
        return 0.1, 0.3  # Too short to tell, assume mostly spoken
    
    # Calculate a score based on multiple factors
    lyrics_score = 0.0  # Will accumulate evidence
    spoken_score = 0.0
    
    # Check for high phrase repetition (songs repeat)
    lines = transcript_text.split('\n')
    if len(lines) > 3:
        unique_lines = set(line.strip().lower() for line in lines if line.strip())
        repetition_ratio = 1 - (len(unique_lines) / len(lines))
        if verbose:
            print(f"    [Heuristic] Line repetition: {repetition_ratio:.1%} ({len(unique_lines)} unique / {len(lines)} total)")
        if repetition_ratio > 0.5:
            lyrics_score += 0.4
        elif repetition_ratio > 0.3:
            lyrics_score += 0.2
    
    # Check for common song indicators
    song_indicators = [
        'yeah yeah', 'oh oh', 'la la', 'na na', 'da da',
        'ooh ooh', 'hey hey', 'baby baby', 'love love',
        'chorus', 'verse', 'bridge'
    ]
    found_indicators = [ind for ind in song_indicators if ind in text_lower]
    indicator_count = len(found_indicators)
    if verbose and found_indicators:
        print(f"    [Heuristic] Song indicators found: {found_indicators}")
    lyrics_score += min(indicator_count * 0.15, 0.4)  # Cap at 0.4
    
    # Check for medical/health terms (strong indicator of spoken content)
    medical_terms = [
        'doctor', 'diagnosis', 'symptom', 'treatment', 'medication',
        'pain', 'fatigue', 'chronic', 'condition', 'flare',
        'eds', 'pots', 'mcas', 'ehlers', 'danlos', 'mast cell',
        'hypermobility', 'dysautonomia', 'autoimmune', 'specialist',
        'prescribed', 'tested', 'blood work', 'mri', 'physical therapy'
    ]
    found_medical = [term for term in medical_terms if term in text_lower]
    medical_count = len(found_medical)
    if verbose and found_medical:
        print(f"    [Heuristic] Medical terms found: {found_medical[:5]}{'...' if len(found_medical) > 5 else ''}")
    spoken_score += min(medical_count * 0.15, 0.5)  # Medical terms are strong signal
    
    # Check for conversational markers
    conversational_markers = [
        "i'm ", "i am ", "i have ", "i was ", "i've been",
        "my doctor", "my diagnosis", "so basically", "you know",
        "for those who", "if you", "here's what", "let me tell you"
    ]
    found_conv = [marker for marker in conversational_markers if marker in text_lower]
    conv_count = len(found_conv)
    if verbose and found_conv:
        print(f"    [Heuristic] Conversational markers: {found_conv[:3]}{'...' if len(found_conv) > 3 else ''}")
    spoken_score += min(conv_count * 0.1, 0.3)
    
    # Calculate final ratio
    # Start at 0.3 (slight bias toward spoken for our use case)
    # Adjust based on evidence
    total_evidence = lyrics_score + spoken_score
    if total_evidence > 0:
        # Weighted ratio: lyrics_score pushes up, spoken_score pushes down
        ratio = 0.3 + (lyrics_score - spoken_score)
        ratio = max(0.0, min(1.0, ratio))  # Clamp to 0-1
        confidence = min(total_evidence, 0.9)  # More evidence = more confident
    else:
        ratio = 0.3  # Default slight spoken bias
        confidence = 0.3  # Low confidence
    
    if verbose:
        print(f"    [Heuristic] Scores - lyrics: {lyrics_score:.2f}, spoken: {spoken_score:.2f}")
        print(f"    [Heuristic] Estimated ratio: {ratio:.0%} lyrics (confidence: {confidence:.0%})")
    
    return ratio, confidence


def process_transcript(transcript: dict, model: str, ollama_url: str, 
                       dry_run: bool = False, heuristics_only: bool = False,
                       verbose: bool = False) -> dict:
    """
    Process a single transcript for song lyrics detection using ratio-based scoring.
    
    Uses BOTH heuristics and LLM to estimate song_lyrics_ratio (0.0-1.0).
    LLM ratio weighted more heavily than heuristic (2x weight).
    
    IMPORTANT: Uses original_text if available (for cleaned transcripts) because
    song detection relies on repetition patterns that cleaning removes.
    
    Args:
        transcript: Dict with video_id, text, original_text (optional), word_count
        model: Ollama model name
        ollama_url: Ollama server URL  
        dry_run: If True, don't update database
        heuristics_only: If True, skip LLM
        verbose: If True, print detailed logging
    """
    video_id = transcript['video_id']
    # Use original_text if available (pre-cleaning), otherwise use current text
    # This preserves song repetitions that cleaning would have removed
    text = transcript.get('original_text') or transcript['text']
    word_count = transcript.get('word_count', len(text.split()))
    
    if transcript.get('original_text') and verbose:
        print(f"  [Note] Using original_text (pre-cleaning) for song detection")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Processing Video ID: {video_id}")
        print(f"  Word count: {word_count}")
        # Show transcript preview
        preview = text[:300].replace('\n', ' ')
        if len(text) > 300:
            preview += "..."
        print(f"  Transcript preview: \"{preview}\"")
        print(f"{'='*60}")
    
    result = {
        'video_id': video_id,
        'word_count': word_count,
    }
    
    # Run heuristics - get ratio estimate
    heuristic_ratio, heuristic_conf = detect_song_lyrics_heuristic(text, verbose=verbose)
    result['heuristic_ratio'] = heuristic_ratio
    result['heuristic_conf'] = heuristic_conf
    
    # Skip LLM if heuristics-only mode
    if heuristics_only:
        final_ratio = heuristic_ratio
        result['song_lyrics_ratio'] = final_ratio
        result['method'] = 'heuristic_only'
        
        if verbose:
            print(f"    [Result] Heuristic only -> ratio={final_ratio:.0%}")
        
        if not dry_run:
            update_transcript_song_lyrics_ratio(video_id, final_ratio)
            if verbose:
                print(f"    [DB] Updated: song_lyrics_ratio={final_ratio:.2f}")
        
        return result
    
    # Run LLM - get ratio estimate
    llm_ratio, llm_success = detect_song_lyrics_llm(text, model, ollama_url, verbose=verbose)
    result['llm_ratio'] = llm_ratio
    result['llm_success'] = llm_success
    
    if not llm_success:
        # LLM failed - fall back to heuristics only
        final_ratio = heuristic_ratio
        result['song_lyrics_ratio'] = final_ratio
        result['method'] = 'heuristic_fallback'
        result['llm_error'] = True
        
        if verbose:
            print(f"    [Result] LLM failed, using heuristic -> ratio={final_ratio:.0%}")
        
        if not dry_run:
            update_transcript_song_lyrics_ratio(video_id, final_ratio)
            if verbose:
                print(f"    [DB] Updated: song_lyrics_ratio={final_ratio:.2f}")
        
        return result
    
    # Combine heuristic and LLM ratios (LLM weighted 2x more)
    # Weighted average: (heuristic * 1 + llm * 2) / 3
    final_ratio = (heuristic_ratio * 1 + llm_ratio * 2) / 3
    result['song_lyrics_ratio'] = final_ratio
    result['method'] = 'combined'
    
    if verbose:
        print(f"    [Combine] Heuristic={heuristic_ratio:.0%}, LLM={llm_ratio:.0%} -> Final={final_ratio:.0%}")
    
    if not dry_run:
        update_transcript_song_lyrics_ratio(video_id, final_ratio)
        if verbose:
            print(f"    [DB] Updated: song_lyrics_ratio={final_ratio:.2f}")
    elif verbose:
        print(f"    [DB] DRY RUN - would set song_lyrics_ratio={final_ratio:.2f}")
    
    return result


def main():
    # Register signal handler for graceful Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Detect and flag song lyrics in existing transcripts"
    )
    parser.add_argument('--limit', type=int, help='Maximum number of transcripts to process')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Check transcripts but do not update database')
    parser.add_argument('--stats', action='store_true',
                        help='Show current song lyrics statistics and exit')
    parser.add_argument('--model', type=str, default=OLLAMA_MODEL,
                        help=f'Ollama model to use (default: {OLLAMA_MODEL})')
    parser.add_argument('--ollama-url', type=str, default=OLLAMA_URL,
                        help=f'Ollama server URL (default: {OLLAMA_URL})')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--heuristics-only', action='store_true',
                        help='Use only heuristics (no LLM calls)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show detailed logging of each transcript analysis')
    
    args = parser.parse_args()
    
    # Force single worker in verbose mode to keep output readable
    if args.verbose and args.workers > 1:
        print(f"Verbose mode: forcing --workers=1 for readable output")
        args.workers = 1
    
    # Show statistics
    if args.stats:
        stats = get_song_lyrics_stats()
        print("\nSong Lyrics Ratio Statistics")
        print("=" * 45)
        print(f"Total transcripts:     {stats['total']}")
        print(f"Unchecked (no ratio):  {stats['unchecked']}")
        print(f"Checked (has ratio):   {stats.get('checked', 0)}")
        
        if stats.get('avg_ratio') is not None:
            print(f"\nRatio Breakdown:")
            print(f"  Average ratio:         {stats['avg_ratio']:.1%}")
            print(f"  Pure spoken (<20%):    {stats.get('pure_spoken', 0)}")
            print(f"  Mostly spoken (20-50%): {stats.get('mostly_spoken', 0)}")
            print(f"  Mixed (50-80%):        {stats.get('mixed', 0)}")
            print(f"  Mostly lyrics (>80%):  {stats.get('mostly_lyrics', 0)}")
            
            print(f"\nFilter tips:")
            print(f"  WHERE song_lyrics_ratio < 0.5  -- mostly spoken content")
            print(f"  WHERE song_lyrics_ratio < 0.8  -- include some mixed")
        return
    
    # Get transcripts needing check
    print("Finding transcripts to check...")
    transcripts = get_transcripts_needing_song_check(limit=args.limit)
    
    if not transcripts:
        print("All transcripts have been checked!")
        stats = get_song_lyrics_stats()
        print(f"  Song lyrics: {stats['song_lyrics']}, Spoken content: {stats['spoken_content']}")
        return
    
    print(f"Found {len(transcripts)} transcripts to check")
    if args.dry_run:
        print("DRY RUN - database will not be updated")
    
    if args.heuristics_only:
        print("Mode: Heuristics only (no LLM)")
    else:
        print(f"Mode: Combined (heuristics + LLM)")
        print(f"LLM model: {args.model}")
        print(f"Ollama URL: {args.ollama_url}")
    
    if args.verbose:
        print("Verbose: ON (detailed logging enabled)")
    print()
    
    # Process transcripts
    song_lyrics_count = 0
    spoken_count = 0
    error_count = 0
    
    # Method tracking
    method_counts = {
        'combined': 0,
        'heuristic_only': 0,
        'heuristic_fallback': 0,
    }
    
    # Ratio tracking
    all_ratios = []
    
    start_time = time.time()
    
    # Process with thread pool for parallel LLM calls
    processed_count = 0
    cancelled_count = 0
    
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_transcript, 
                    t, 
                    args.model, 
                    args.ollama_url, 
                    args.dry_run,
                    args.heuristics_only,
                    args.verbose
                ): t 
                for t in transcripts
            }
            
            for i, future in enumerate(as_completed(futures), 1):
                # Check for shutdown request
                if shutdown_requested:
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                            cancelled_count += 1
                    break
                
                transcript = futures[future]
                try:
                    result = future.result(timeout=1)
                    processed_count += 1
                    
                    if 'error' in result:
                        print(f"[{i}/{len(transcripts)}] FAILED Video {result['video_id']}: {result['error']}")
                        error_count += 1
                        continue
                    
                    method = result.get('method', 'unknown')
                    if method in method_counts:
                        method_counts[method] += 1
                    
                    # Track ratios
                    ratio = result.get('song_lyrics_ratio', 0)
                    all_ratios.append(ratio)
                    
                    # Categorize by ratio for display
                    if ratio >= 0.8:
                        symbol = "[SONG]"
                        song_lyrics_count += 1
                        print(f"[{i}/{len(transcripts)}] {symbol} Video {result['video_id']}: {ratio:.0%} lyrics")
                    else:
                        symbol = "[SPEAK]"
                        spoken_count += 1
                        # Only print every 10th for mostly-spoken content to reduce noise
                        if i % 10 == 0 or i == len(transcripts) or ratio >= 0.5:
                            print(f"[{i}/{len(transcripts)}] {symbol} Video {result['video_id']}: {ratio:.0%} lyrics")
                            
                except Exception as e:
                    print(f"[{i}/{len(transcripts)}] ERROR processing video {transcript['video_id']}: {e}")
                    error_count += 1
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    if shutdown_requested or cancelled_count > 0:
        print(f"\nStopped early. Processed {processed_count}/{len(transcripts)}, cancelled {cancelled_count}")
    
    elapsed = time.time() - start_time
    
    # Summary
    print()
    print("=" * 50)
    print("Detection Summary")
    print("=" * 50)
    total = len(transcripts)
    print(f"Total processed:      {total}")
    print(f"Mostly lyrics (>80%): {song_lyrics_count} ({song_lyrics_count/total*100:.1f}%)")
    print(f"Mostly spoken (<80%): {spoken_count} ({spoken_count/total*100:.1f}%)")
    print(f"Errors:               {error_count}")
    print()
    
    # Ratio statistics
    if all_ratios:
        avg_ratio = sum(all_ratios) / len(all_ratios)
        pure_spoken = sum(1 for r in all_ratios if r < 0.2)
        mostly_spoken = sum(1 for r in all_ratios if 0.2 <= r < 0.5)
        mixed = sum(1 for r in all_ratios if 0.5 <= r < 0.8)
        mostly_lyrics = sum(1 for r in all_ratios if r >= 0.8)
        
        print(f"Ratio Distribution:")
        print(f"  Average ratio:        {avg_ratio:.0%}")
        print(f"  Pure spoken (<20%):   {pure_spoken}")
        print(f"  Mostly spoken (20-50%): {mostly_spoken}")
        print(f"  Mixed (50-80%):       {mixed}")
        print(f"  Mostly lyrics (>80%): {mostly_lyrics}")
    print()
    
    print(f"Method breakdown:")
    for method, count in method_counts.items():
        if count > 0:
            print(f"  {method}: {count}")
    print()
    print(f"Time elapsed:         {elapsed:.1f}s ({elapsed/total:.2f}s per transcript)")
    
    if args.dry_run:
        print("\nDRY RUN - no database changes were made")
    else:
        print("\nDatabase updated successfully")
        
        # Show updated stats
        stats = get_song_lyrics_stats()
        print(f"\nUpdated totals: {stats['checked']} checked, {stats['unchecked']} remaining")
        if stats.get('avg_ratio') is not None:
            print(f"  Avg ratio: {stats['avg_ratio']:.0%} | <60%: {stats.get('pure_spoken', 0) + stats.get('mostly_spoken', 0)} | >=60%: {stats.get('mixed', 0) + stats.get('mostly_lyrics', 0)}")


if __name__ == '__main__':
    main()
