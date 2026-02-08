"""
Complete pipeline orchestration for the EDS/MCAS/POTS research project.
Coordinates all stages from download to analysis.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from downloader import VideoDownloader
from transcriber import AudioTranscriber
from extractor import SymptomExtractor
from analyzer import SymptomAnalyzer
from database import (
    get_all_videos_with_transcripts, get_symptom_statistics,
    start_processing_run, complete_processing_run,
    init_pipeline_progress, update_pipeline_progress,
    get_incomplete_urls, get_latest_run_id, get_run_progress_summary,
    get_treatment_statistics, get_comorbidity_matrix, get_strain_indicators_summary
)
import config


class ResearchPipeline:
    """Orchestrates the complete research pipeline."""

    def __init__(self,
                 whisper_model: str = 'large-v3',
                 min_confidence: float = 0.6,
                 parallel_extraction: bool = True,
                 extractor_provider: Optional[str] = None,
                 extractor_model: Optional[str] = None,
                 ollama_url: Optional[str] = None,
                 max_song_ratio: float = 0.6,
                 enable_thinking: bool = False):
        """
        Initialize the pipeline.

        Args:
            whisper_model: Whisper model for transcription (use large-v3 for RTX 4090)
            min_confidence: Minimum confidence score for symptoms
            parallel_extraction: Enable parallel symptom extraction
            max_song_ratio: Skip videos with song_lyrics_ratio >= this (default 0.6)
            enable_thinking: For Qwen3 models, use /think mode (slower, more thorough)
        """
        self.downloader = VideoDownloader()
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.extractor = SymptomExtractor(
            max_workers=10 if parallel_extraction else 1,
            provider=extractor_provider,
            model=extractor_model,
            ollama_url=ollama_url,
            max_song_ratio=max_song_ratio,
            enable_thinking=enable_thinking,
        )
        self.analyzer = SymptomAnalyzer(min_confidence=min_confidence)

        self.stats = {
            'videos_downloaded': 0,
            'videos_transcribed': 0,
            'symptoms_extracted': 0,
            'diagnoses_extracted': 0,
            'treatments_extracted': 0,
            'errors': []
        }
        self.current_run_id = None

    def process_url(self, url: str, tags: Optional[List[str]] = None,
                    skip_if_exists: bool = True) -> Dict[str, Any]:
        """
        Process a single URL through the complete pipeline.

        Args:
            url: Video URL to process
            tags: Optional tags for the video
            skip_if_exists: Skip processing if already complete

        Returns:
            Dictionary with processing results
        """
        result = {
            'url': url,
            'stages': {},
            'success': True
        }

        try:
            # Stage 1: Download
            print(f"\n{'='*80}")
            print(f"STAGE 1: Downloading video")
            print('='*80)
            download_result = self.downloader.download_audio(url, tags)
            result['stages']['download'] = download_result
            video_id = download_result['video_id']

            if not download_result.get('already_existed'):
                self.stats['videos_downloaded'] += 1

            # Stage 2: Transcribe
            print(f"\n{'='*80}")
            print(f"STAGE 2: Transcribing audio")
            print('='*80)
            transcript_result = self.transcriber.transcribe(video_id)
            result['stages']['transcribe'] = transcript_result

            if not transcript_result.get('already_existed'):
                self.stats['videos_transcribed'] += 1

            # Stage 3: Extract symptoms, diagnoses, and treatments
            print(f"\n{'='*80}")
            print(f"STAGE 3: Extracting symptoms, diagnoses, and treatments")
            print('='*80)
            extraction_result = self.extractor.extract_all(video_id)
            result['stages']['extract'] = extraction_result

            if extraction_result.get('success'):
                symptoms_data = extraction_result.get('symptoms', {})
                diagnoses_data = extraction_result.get('diagnoses', {})
                treatments_data = extraction_result.get('treatments', {})

                self.stats['symptoms_extracted'] += symptoms_data.get('symptoms_saved', 0)
                self.stats['diagnoses_extracted'] += diagnoses_data.get('diagnoses_saved', 0)
                self.stats['treatments_extracted'] += treatments_data.get('treatments_saved', 0)

            print(f"\n{'='*80}")
            print(f"✓ Pipeline complete for: {url}")
            print(f"  Downloaded: {download_result['audio_path']}")
            print(f"  Transcribed: {transcript_result.get('word_count', 'N/A')} words")
            if transcript_result.get('quality_score'):
                print(f"  Transcript quality: {transcript_result['quality_score']:.2f}")
            symptoms_data = extraction_result.get('symptoms', {})
            diagnoses_data = extraction_result.get('diagnoses', {})
            treatments_data = extraction_result.get('treatments', {})
            print(f"  Symptoms: {symptoms_data.get('symptoms_saved', 0)} extracted")
            print(f"  Diagnoses: {diagnoses_data.get('diagnoses_saved', 0)} extracted")
            print(f"  Treatments: {treatments_data.get('treatments_saved', 0)} extracted")
            if extraction_result.get('concordance'):
                for conc in extraction_result['concordance']:
                    print(f"  Concordance ({conc['condition']}): {conc['concordance_score']:.2f}")
            print('='*80)

        except Exception as e:
            print(f"\n✗ Pipeline failed for {url}: {e}")
            result['success'] = False
            result['error'] = str(e)
            self.stats['errors'].append({'url': url, 'error': str(e)})

        return result

    def process_batch(self, urls: List[str], tags: Optional[List[str]] = None,
                      resume_run_id: Optional[int] = None,
                      urls_file: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process multiple URLs through the pipeline with progress tracking.

        Args:
            urls: List of video URLs
            tags: Optional tags for all videos
            resume_run_id: Optional run ID to resume from (for interrupted runs)
            urls_file: Optional path to urls file (for incremental progress tracking)

        Returns:
            List of processing results
        """
        print(f"[DEBUG] process_batch called with urls_file={urls_file}")
        # Start or resume a processing run
        if resume_run_id:
            self.current_run_id = resume_run_id
            incomplete = get_incomplete_urls(resume_run_id)
            urls_to_process = [item['url'] for item in incomplete]
            print(f"\n{'#'*80}")
            print(f"# RESUMING RUN {resume_run_id}")
            print(f"# Found {len(urls_to_process)} incomplete URLs to process")
            progress = get_run_progress_summary(resume_run_id)
            print(f"# Previous progress: {progress['completed']} completed, {progress['failed']} failed")
            print('#'*80)
        else:
            config_snapshot = {
                'whisper_model': self.transcriber.model_size,
                'device': self.transcriber.device,
                'min_confidence': self.analyzer.min_confidence,
                'parallel_extraction': self.extractor.max_workers > 1,
                'extractor_provider': self.extractor.provider,
                'extractor_model': self.extractor.model
            }
            self.current_run_id = start_processing_run('batch', config_snapshot)
            init_pipeline_progress(self.current_run_id, urls)
            urls_to_process = urls

        print(f"\n{'#'*80}")
        print(f"# Starting pipeline for {len(urls_to_process)} videos (Run ID: {self.current_run_id})")
        print(f"# Configuration:")
        print(f"#   Whisper model: {self.transcriber.model_size}")
        print(f"#   Device: {self.transcriber.device}")
        print(f"#   Min confidence: {self.analyzer.min_confidence}")
        print(f"#   Parallel extraction: {self.extractor.max_workers > 1}")
        print(f"#   Extractor: {self.extractor.provider} / {self.extractor.model}")
        print('#'*80)

        results = []
        start_time = datetime.now()

        for i, url in enumerate(urls_to_process, 1):
            print(f"\n\n{'#'*80}")
            print(f"# VIDEO {i}/{len(urls_to_process)}")
            print('#'*80)

            try:
                update_pipeline_progress(self.current_run_id, url, 'downloading')
                result = self.process_url(url, tags)
                results.append(result)

                if result['success']:
                    update_pipeline_progress(self.current_run_id, url, 'completed',
                                           video_id=result['stages']['download']['video_id'])
                    # Immediately move successful URL to processed file
                    no_move = getattr(self, '_no_move_processed', False)
                    print(f"  [DEBUG] urls_file={urls_file}, no_move={no_move}")
                    if urls_file and not no_move:
                        from url_manager import mark_urls_as_processed
                        moved = mark_urls_as_processed([url], pending_file=urls_file)
                        print(f"  [DEBUG] Moved {moved} URL(s) to urls_processed.txt")
                else:
                    update_pipeline_progress(self.current_run_id, url, 'failed',
                                           error_message=result.get('error', 'Unknown error'))
            except Exception as e:
                print(f"✗ Unhandled error for {url}: {e}")
                update_pipeline_progress(self.current_run_id, url, 'failed', error_message=str(e))
                results.append({'url': url, 'success': False, 'error': str(e)})

        # Final statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        success_count = sum(1 for r in results if r.get('success', False))

        # Complete the processing run
        complete_processing_run(
            self.current_run_id,
            videos_processed=len(results),
            transcripts_created=self.stats['videos_transcribed'],
            symptoms_extracted=self.stats['symptoms_extracted'],
            diagnoses_extracted=self.stats['diagnoses_extracted'],
            errors=self.stats['errors'] if self.stats['errors'] else None
        )

        print(f"\n\n{'#'*80}")
        print(f"# PIPELINE COMPLETE (Run ID: {self.current_run_id})")
        print('#'*80)
        print(f"  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Success rate: {success_count}/{len(urls_to_process)} ({success_count/len(urls_to_process)*100:.1f}%)")
        print(f"  Videos downloaded: {self.stats['videos_downloaded']}")
        print(f"  Videos transcribed: {self.stats['videos_transcribed']}")
        print(f"  Total symptoms extracted: {self.stats['symptoms_extracted']}")
        print(f"  Total diagnoses extracted: {self.stats['diagnoses_extracted']}")
        print(f"  Total treatments extracted: {self.stats['treatments_extracted']}")
        if self.stats['errors']:
            print(f"  Errors: {len(self.stats['errors'])}")
        print(f"\n  To resume this run if interrupted: --resume {self.current_run_id}")
        print('#'*80)

        return results

    def analyze_all(self, cluster_method: str = 'kmeans',
                   viz_method: str = 'umap',
                   optimize_clusters: bool = True) -> Dict[str, Any]:
        """
        Run complete analysis on all collected data.

        Args:
            cluster_method: Clustering algorithm ('kmeans' or 'dbscan')
            viz_method: Visualization method ('pca', 'tsne', or 'umap')
            optimize_clusters: Auto-optimize cluster count

        Returns:
            Analysis results
        """
        print(f"\n{'='*80}")
        print("Running complete analysis on collected data")
        print('='*80)

        # Load data
        df = self.analyzer.load_symptom_data()

        if len(df) < 10:
            raise ValueError("Not enough data for analysis (need at least 10 symptoms)")

        # Prepare features
        features = self.analyzer.prepare_features(df, method='combined')

        # Cluster
        if cluster_method == 'kmeans':
            labels, metrics = self.analyzer.cluster_kmeans(features, optimize=optimize_clusters)
        elif cluster_method == 'dbscan':
            labels, metrics = self.analyzer.cluster_dbscan(features)
        else:
            raise ValueError(f"Unknown cluster method: {cluster_method}")

        # Visualize
        viz_path = self.analyzer.visualize_clusters(df, features, labels, method=viz_method)

        # Generate report
        report = self.analyzer.generate_cluster_report(df, labels)

        # Export
        export_path = self.analyzer.export_results(df, labels)

        print(f"\n✓ Analysis complete!")
        print(f"  Found {report['n_clusters']} clusters from {report['total_symptoms']} symptoms")
        print(f"  Visualization: {viz_path}")
        print(f"  Export: {export_path}")

        return {
            'metrics': metrics,
            'report': report,
            'visualization_path': str(viz_path),
            'export_path': str(export_path)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get current pipeline statistics."""
        db_stats = get_symptom_statistics()

        return {
            'database': db_stats,
            'pipeline_session': self.stats
        }


def read_urls_file(path: str) -> List[str]:
    """
    Read a text file containing one URL per line.
    - Ignores blank lines
    - Supports comments starting with # or //
    - Trims whitespace
    """
    from url_manager import read_url_file
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"URLs file not found: {p}")
    return read_url_file(path)


def merge_and_dedupe_urls(cli_urls: List[str], file_urls: List[str]) -> List[str]:
    """Keep first-seen order while removing duplicates."""
    seen = set()
    merged: List[str] = []
    for u in (file_urls + cli_urls):
        u2 = (u or "").strip()
        if not u2:
            continue
        if u2 in seen:
            continue
        seen.add(u2)
        merged.append(u2)
    return merged


# =============================================================================
# Subcommand handlers
# =============================================================================

def cmd_run(args):
    """Handle the 'run' subcommand - full pipeline processing."""
    from database import get_connection
    from url_manager import mark_urls_as_processed

    pipeline = ResearchPipeline(
        whisper_model=args.whisper_model,
        min_confidence=args.min_confidence,
        parallel_extraction=not args.no_parallel,
        extractor_provider=args.provider,
        extractor_model=args.model,
        max_song_ratio=args.max_song_ratio,
        enable_thinking=getattr(args, 'thinking', False)
    )
    pipeline._no_move_processed = args.no_move_processed

    if args.resume:
        results = pipeline.process_batch([], tags=args.tags, resume_run_id=args.resume,
                                         urls_file=args.urls_file)
    else:
        file_urls: List[str] = []
        if args.urls_file:
            file_urls = read_urls_file(args.urls_file)

        merged_urls = merge_and_dedupe_urls(args.urls or [], file_urls)

        if not merged_urls:
            print("Error: No URLs provided. Use --urls-file or provide URLs as arguments.")
            return 1

        results = pipeline.process_batch(merged_urls, tags=args.tags, urls_file=args.urls_file)

    # Show URL processing summary (URLs are moved incrementally during processing)
    if args.urls_file:
        successful_urls = [r['url'] for r in results if r.get('success')]
        print(f"\nURL Processing Summary:")
        print(f"  Total results: {len(results)}")
        print(f"  Successful: {len(successful_urls)}")
        if not args.no_move_processed:
            print(f"  URLs moved to urls_processed.txt as they completed")

    print("\n" + json.dumps(results, indent=2, default=str))
    return 0


def cmd_download(args):
    """Handle the 'download' subcommand - download only."""
    from scripts.discover import discover_user_videos
    from url_manager import (
        mark_urls_as_processed, mark_url_as_failed,
        filter_unprocessed_urls, get_stats
    )
    
    downloader = VideoDownloader()
    urls = []
    urls_file = None  # Track which file we're reading from for URL management

    if args.url:
        urls = [args.url]
    elif args.urls_file:
        urls_file = args.urls_file
        urls = read_urls_file(args.urls_file)
    elif getattr(args, 'user', None):
        # Discover videos for user(s) first
        print(f"Discovering videos for user(s): {', '.join(args.user)}")
        for username in args.user:
            try:
                user_urls = discover_user_videos(username, max_videos=getattr(args, 'max_videos', None))
                print(f"  Found {len(user_urls)} videos for @{username}")
                urls.extend(user_urls)
            except Exception as e:
                print(f"  Error discovering @{username}: {e}")
        
        if not urls:
            print("No videos found for the specified user(s)")
            return 1
    else:
        print("Error: Provide --url, --urls-file, or --user")
        return 1

    original_count = len(urls)
    
    # Filter out already processed or failed URLs (unless --force is used)
    skip_processed = not getattr(args, 'force', False)
    if skip_processed and urls_file:
        urls = filter_unprocessed_urls(urls)
        skipped = original_count - len(urls)
        if skipped > 0:
            print(f"Skipping {skipped} already processed/failed URLs")
            stats = get_stats()
            print(f"  (processed: {stats['processed_count']}, failed: {stats['failed_count']})")
    
    if not urls:
        print("No new URLs to download!")
        return 0

    print(f"\nDownloading {len(urls)} video(s)...")
    success_count = 0
    already_existed_count = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n[{i}/{len(urls)}] {url}")
        try:
            result = downloader.download_audio(url, args.tags)
            if result.get('already_existed'):
                print(f"  Already exists: {result['audio_path']}")
                already_existed_count += 1
            else:
                print(f"  Downloaded: {result['audio_path']}")
            success_count += 1
            
            # Move URL to processed file immediately after success
            if urls_file:
                mark_urls_as_processed([url], pending_file=urls_file)
                
        except Exception as e:
            error_msg = str(e)[:100]  # Truncate long errors
            print(f"  Error: {e}")
            
            # Move URL to failed file
            if urls_file:
                mark_url_as_failed(url, error=error_msg, pending_file=urls_file)

    print(f"\nDownload Summary:")
    print(f"  Total processed: {success_count}/{len(urls)}")
    print(f"  New downloads: {success_count - already_existed_count}")
    print(f"  Already existed: {already_existed_count}")
    print(f"  Failed: {len(urls) - success_count}")
    if urls_file:
        stats = get_stats()
        print(f"  Remaining in {urls_file}: {stats['pending_count']}")
    
    return 0


def cmd_transcribe(args):
    """Handle the 'transcribe' subcommand - transcribe only."""
    from database import get_connection

    transcriber = AudioTranscriber(model_size=args.whisper_model)

    if args.video_id:
        video_ids = [args.video_id]
    elif getattr(args, 'user', None):
        # Find untranscribed videos for specific user(s)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT v.id, v.author FROM videos v
                LEFT JOIN transcripts t ON v.id = t.video_id
                WHERE t.id IS NULL 
                  AND v.audio_path IS NOT NULL
                  AND v.author = ANY(%s)
            """, (args.user,))
            results = cur.fetchall()
            video_ids = [row[0] for row in results]
            
        if not video_ids:
            print(f"No untranscribed videos found for user(s): {', '.join(args.user)}")
            return 0
        
        print(f"Found {len(video_ids)} untranscribed video(s) for user(s): {', '.join(args.user)}")
    elif args.all:
        # Find videos without transcripts
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT v.id FROM videos v
                LEFT JOIN transcripts t ON v.id = t.video_id
                WHERE t.id IS NULL AND v.audio_path IS NOT NULL
            """)
            video_ids = [row[0] for row in cur.fetchall()]

        if not video_ids:
            print("All videos already transcribed!")
            return 0

        print(f"Found {len(video_ids)} video(s) to transcribe")
    else:
        print("Error: Provide --video-id, --user, or --all")
        return 1

    success_count = 0
    for i, vid in enumerate(video_ids, 1):
        print(f"\n[{i}/{len(video_ids)}] Transcribing video ID {vid}...")
        try:
            result = transcriber.transcribe(vid)
            if result.get('already_existed'):
                print(f"  Already transcribed: {result['word_count']} words")
            else:
                print(f"  Transcribed: {result['word_count']} words")
            success_count += 1
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nTranscribed {success_count}/{len(video_ids)} videos")
    return 0


def cmd_extract(args):
    """Handle the 'extract' subcommand - extract symptoms only."""
    from database import get_connection, clear_transcript_extracted

    extractor = SymptomExtractor(
        max_workers=10 if not args.no_parallel else 1,
        provider=args.provider,
        model=args.model,
        max_song_ratio=args.max_song_ratio,
        enable_thinking=getattr(args, 'thinking', False)
    )

    min_words = getattr(args, 'min_words', 20)
    force = getattr(args, 'force', False)
    
    if args.video_id:
        video_ids = [args.video_id]
        if force:
            clear_transcript_extracted(args.video_id)
            print(f"Force mode: cleared extraction status for video {args.video_id}")
    elif getattr(args, 'user', None):
        # Find transcripts for specific user(s)
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT DISTINCT t.video_id
                FROM transcripts t
                JOIN videos v ON t.video_id = v.id
                WHERE v.author = ANY(%s)
                  AND (t.extracted_at IS NULL OR %s)
                  AND (t.song_lyrics_ratio IS NULL OR t.song_lyrics_ratio < %s)
                  AND (t.word_count IS NULL OR t.word_count >= %s)
            """, (args.user, force, args.max_song_ratio, min_words))
            video_ids = [row[0] for row in cur.fetchall()]
            
            if force and video_ids:
                cur.execute("""
                    UPDATE transcripts SET extracted_at = NULL
                    WHERE video_id = ANY(%s)
                """, (video_ids,))
                conn.commit()
                print(f"Force mode: cleared extraction status for {len(video_ids)} videos")
        
        if not video_ids:
            print(f"No videos to extract for user(s): {', '.join(args.user)}")
            return 0
        
        print(f"Found {len(video_ids)} video(s) to extract for user(s): {', '.join(args.user)}")
    elif args.all:
        # Find videos with transcripts to extract
        # If --force, include already extracted; otherwise only unextracted
        with get_connection() as conn:
            cur = conn.cursor()
            
            if force:
                # Force mode: include all transcripts (ignore extracted_at)
                cur.execute("""
                    SELECT DISTINCT t.video_id
                    FROM transcripts t
                    WHERE (t.song_lyrics_ratio IS NULL OR t.song_lyrics_ratio < %s)
                      AND (t.word_count IS NULL OR t.word_count >= %s)
                """, (args.max_song_ratio, min_words))
                video_ids = [row[0] for row in cur.fetchall()]
                
                # Count skipped
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE t.song_lyrics_ratio >= %s) as skipped_lyrics,
                        COUNT(*) FILTER (WHERE t.word_count < %s) as skipped_short
                    FROM transcripts t
                """, (args.max_song_ratio, min_words))
                row = cur.fetchone()
                skipped_lyrics = row[0]
                skipped_short = row[1]
                already_extracted = 0
                
                # Clear extraction status for all selected videos
                if video_ids:
                    cur.execute("""
                        UPDATE transcripts SET extracted_at = NULL
                        WHERE video_id = ANY(%s)
                    """, (video_ids,))
                    conn.commit()
                    print(f"Force mode: cleared extraction status for {len(video_ids)} videos")
            else:
                # Normal mode: exclude already extracted
                cur.execute("""
                    SELECT DISTINCT t.video_id
                    FROM transcripts t
                    WHERE t.extracted_at IS NULL
                      AND (t.song_lyrics_ratio IS NULL OR t.song_lyrics_ratio < %s)
                      AND (t.word_count IS NULL OR t.word_count >= %s)
                """, (args.max_song_ratio, min_words))
                video_ids = [row[0] for row in cur.fetchall()]
                
                # Count how many were skipped for each reason
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE t.extracted_at IS NOT NULL) as already_extracted,
                        COUNT(*) FILTER (WHERE t.extracted_at IS NULL AND t.song_lyrics_ratio >= %s) as skipped_lyrics,
                        COUNT(*) FILTER (WHERE t.extracted_at IS NULL AND t.word_count < %s) as skipped_short
                    FROM transcripts t
                """, (args.max_song_ratio, min_words))
                row = cur.fetchone()
                already_extracted = row[0]
                skipped_lyrics = row[1]
                skipped_short = row[2]

        if not video_ids:
            print("All transcripts already processed!")
            if already_extracted > 0 or skipped_lyrics > 0 or skipped_short > 0:
                print(f"  ({already_extracted} already extracted, {skipped_lyrics} song lyrics, {skipped_short} too short)")
            return 0

        print(f"Found {len(video_ids)} video(s) to extract symptoms from")
        if already_extracted > 0 or skipped_lyrics > 0 or skipped_short > 0:
            print(f"  ({already_extracted} already extracted, {skipped_lyrics} song lyrics, {skipped_short} too short < {min_words} words)")
    else:
        print("Error: Provide --video-id, --user, or --all")
        return 1

    total_symptoms = 0
    success_count = 0
    for i, vid in enumerate(video_ids, 1):
        print(f"\n[{i}/{len(video_ids)}] Extracting from video ID {vid}...")
        try:
            result = extractor.extract_all(vid, force=force)
            if result.get('success'):
                symptoms = result.get('symptoms', {}).get('symptoms_saved', 0)
                diagnoses = result.get('diagnoses', {}).get('diagnoses_saved', 0)
                treatments = result.get('treatments', {}).get('treatments_saved', 0)
                print(f"  Extracted: {symptoms} symptoms, {diagnoses} diagnoses, {treatments} treatments")
                total_symptoms += symptoms
                success_count += 1
            else:
                print(f"  Failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"  Error: {e}")

    print(f"\nProcessed {success_count}/{len(video_ids)} videos")
    print(f"Total symptoms extracted: {total_symptoms}")
    return 0


def cmd_analyze(args):
    """Handle the 'analyze' subcommand - run analysis."""
    analyzer = SymptomAnalyzer(min_confidence=args.min_confidence)

    print("Loading symptom data...")
    df = analyzer.load_symptom_data()

    if len(df) < 10:
        print(f"Not enough data for analysis (need at least 10 symptoms, have {len(df)})")
        return 1

    print(f"Loaded {len(df)} symptoms")
    print(f"Preparing features using {args.feature_method} method...")
    features = analyzer.prepare_features(df, method=args.feature_method)

    print(f"Clustering using {args.cluster_method}...")
    if args.cluster_method == 'kmeans':
        labels, metrics = analyzer.cluster_kmeans(
            features,
            n_clusters=args.clusters,
            optimize=(args.clusters is None)
        )
    else:
        labels, metrics = analyzer.cluster_dbscan(features)

    print(f"Generating visualizations using {args.viz_method}...")
    viz_path = analyzer.visualize_clusters(df, features, labels, method=args.viz_method)

    print("Generating cluster report...")
    report = analyzer.generate_cluster_report(df, labels)
    export_path = analyzer.export_results(df, labels)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print('='*60)
    print(f"Total symptoms: {report['total_symptoms']}")
    print(f"Number of clusters: {report['n_clusters']}")
    print(f"Visualization: {viz_path}")
    print(f"Export: {export_path}")

    print(f"\nCluster Summary:")
    for cluster in report['clusters']:
        print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} symptoms")

    return 0


def cmd_stats(args):
    """Handle the 'stats' subcommand - show statistics."""
    stats = get_symptom_statistics()

    print(f"\n{'='*60}")
    print("DATABASE STATISTICS")
    print('='*60)
    print(f"\nOverview:")
    print(f"  Videos: {stats['total_videos']}")
    print(f"  Transcripts: {stats['total_transcripts']}")
    print(f"  Symptoms: {stats['total_symptoms']}")
    print(f"  Diagnoses: {stats['total_diagnoses']}")

    if stats['diagnoses_by_condition']:
        print(f"\nDiagnoses by Condition:")
        for d in stats['diagnoses_by_condition']:
            print(f"  {d['condition_code']}: {d['count']}")

    if stats['concordance_by_condition']:
        print(f"\nConcordance by Condition:")
        for c in stats['concordance_by_condition']:
            print(f"  {c['condition_code']}: {c['avg_concordance']:.2f} avg ({c['sample_size']} samples)")

    if args.detailed:
        treatment_stats = get_treatment_statistics()
        if treatment_stats['top_treatments']:
            print(f"\nTop Treatments:")
            for t in treatment_stats['top_treatments'][:10]:
                eff = f" (eff: {t['avg_effectiveness']:.2f})" if t['avg_effectiveness'] else ""
                print(f"  {t['treatment_name']} ({t['treatment_type']}): {t['mention_count']} mentions{eff}")

        comorbidity = get_comorbidity_matrix()
        if comorbidity:
            print(f"\nComorbidity Pairs:")
            for c in comorbidity[:10]:
                print(f"  {c['condition_a']} + {c['condition_b']}: {c['video_count']} videos")

        # STRAIN Framework Indicators
        try:
            strain_stats = get_strain_indicators_summary()
            if strain_stats.get('total_analyzed', 0) > 0:
                print(f"\n--- STRAIN Framework Indicators ---")
                print(f"  Videos analyzed: {strain_stats['total_analyzed']}")
                print(f"  Self-diagnosed: {strain_stats.get('self_diagnosed_count', 0)}")
                print(f"  Professional diagnosis: {strain_stats.get('professionally_diagnosed_count', 0)}")
                print(f"  Doctor dismissal mentioned: {strain_stats.get('doctor_dismissal_count', 0)}")
                print(f"  Medical gaslighting mentioned: {strain_stats.get('medical_gaslighting_count', 0)}")
                print(f"  Long diagnostic journey: {strain_stats.get('long_journey_count', 0)}")
                print(f"  Stress triggers mentioned: {strain_stats.get('stress_triggers_count', 0)}")
                print(f"  Symptom flares mentioned: {strain_stats.get('symptom_flares_count', 0)}")
                print(f"  Learned from TikTok: {strain_stats.get('learned_from_tiktok_count', 0)}")
                print(f"  Online community mention: {strain_stats.get('online_community_count', 0)}")
                
                if strain_stats.get('by_content_type'):
                    print(f"\n  Content Types:")
                    for ct in strain_stats['by_content_type']:
                        print(f"    {ct['content_type']}: {ct['count']}")
        except Exception as e:
            pass  # STRAIN stats may not be available yet

        # Creator tier breakdown
        try:
            from database import get_connection
            with get_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT creator_tier, COUNT(*) as count 
                    FROM videos 
                    WHERE creator_tier IS NOT NULL 
                    GROUP BY creator_tier 
                    ORDER BY count DESC
                """)
                tiers = cur.fetchall()
                if tiers:
                    print(f"\nCreator Influence Tiers:")
                    for tier, count in tiers:
                        print(f"  {tier}: {count} videos")
        except Exception:
            pass

        # Run status
        run_id = get_latest_run_id()
        if run_id:
            progress = get_run_progress_summary(run_id)
            print(f"\nLatest Run (ID {run_id}):")
            print(f"  Total URLs: {progress['total']}")
            print(f"  Completed: {progress['completed']}")
            print(f"  Failed: {progress['failed']}")

    print('='*60)
    return 0


def cmd_discover(args):
    """Handle the 'discover' subcommand - find new videos."""
    import subprocess
    import sys

    # Build the command to run discover.py
    cmd = [sys.executable, 'scripts/discover.py']

    if args.url:
        for url in args.url:
            cmd.extend(['--url', url])
    if args.user:
        for user in args.user:
            cmd.extend(['--user', user])
    if args.expand_users:
        cmd.extend(['--expand-users', args.expand_users])
    if args.hashtag:
        for tag in args.hashtag:
            cmd.extend(['--hashtag', tag])
    if args.search:
        for query in args.search:
            cmd.extend(['--search', query])
    if args.output:
        cmd.extend(['--output', args.output])
    if args.append:
        cmd.append('--append')
    if args.max_videos:
        cmd.extend(['--max-videos', str(args.max_videos)])

    # Run the discover script
    result = subprocess.run(cmd)
    return result.returncode


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='EDS/MCAS/POTS Research Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python pipeline.py run --urls-file urls.txt --tags EDS MCAS POTS

  # Granular operations
  python pipeline.py download --urls-file urls.txt
  python pipeline.py transcribe --all
  python pipeline.py extract --all

  # Analysis
  python pipeline.py analyze --cluster-method kmeans
  python pipeline.py stats --detailed

  # Discovery
  python pipeline.py discover --hashtag EDS --max-videos 100
        """
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # --- run subcommand ---
    run_parser = subparsers.add_parser('run', help='Run full pipeline (download, transcribe, extract)')
    run_parser.add_argument('urls', nargs='*', help='Video URLs to process')
    run_parser.add_argument('--urls-file', help='Path to text file with URLs')
    run_parser.add_argument('--tags', nargs='+', default=['EDS', 'MCAS', 'POTS'], help='Tags for videos')
    run_parser.add_argument('--resume', type=int, help='Resume a previous run by ID')
    run_parser.add_argument('--whisper-model', default='large-v3', help='Whisper model size')
    run_parser.add_argument('--provider', help='Extractor provider (anthropic or ollama)')
    run_parser.add_argument('--model', help='Extractor model name')
    run_parser.add_argument('--min-confidence', type=float, default=0.6, help='Min symptom confidence')
    run_parser.add_argument('--no-move-processed', action='store_true',
                           help='Do not move processed URLs to urls_processed.txt')
    run_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel extraction')
    run_parser.add_argument('--max-song-ratio', type=float, default=0.2,
                           help='Skip videos with song_lyrics_ratio >= this (default: 0.2)')
    run_parser.add_argument('--thinking', action='store_true',
                           help='Enable Qwen3 thinking mode (/think) for deeper reasoning (slower)')
    run_parser.set_defaults(func=cmd_run)

    # --- download subcommand ---
    dl_parser = subparsers.add_parser('download', help='Download videos only')
    dl_parser.add_argument('--url', help='Single video URL')
    dl_parser.add_argument('--urls-file', help='Path to text file with URLs')
    dl_parser.add_argument('--user', action='append', help='TikTok username(s) to discover and download')
    dl_parser.add_argument('--tags', nargs='+', default=[], help='Tags for videos')
    dl_parser.add_argument('--max-videos', type=int, help='Max videos per user (default: all)')
    dl_parser.add_argument('--force', action='store_true',
                          help='Process URLs even if already in urls_processed.txt or urls_failed.txt')
    dl_parser.set_defaults(func=cmd_download)

    # --- transcribe subcommand ---
    tr_parser = subparsers.add_parser('transcribe', help='Transcribe audio only')
    tr_parser.add_argument('--video-id', type=int, help='Single video database ID')
    tr_parser.add_argument('--all', action='store_true', help='Transcribe all untranscribed videos')
    tr_parser.add_argument('--user', action='append', help='TikTok username(s) to transcribe')
    tr_parser.add_argument('--whisper-model', default='large-v3', help='Whisper model size')
    tr_parser.set_defaults(func=cmd_transcribe)

    # --- extract subcommand ---
    ex_parser = subparsers.add_parser('extract', help='Extract symptoms only')
    ex_parser.add_argument('--video-id', type=int, help='Single video database ID')
    ex_parser.add_argument('--all', action='store_true', help='Extract from all unprocessed transcripts')
    ex_parser.add_argument('--user', action='append', help='TikTok username(s) to extract from')
    ex_parser.add_argument('--provider', help='Extractor provider (ollama, deepseek, or anthropic)')
    ex_parser.add_argument('--model', help='Extractor model name')
    ex_parser.add_argument('--min-confidence', type=float, default=0.6, help='Min symptom confidence')
    ex_parser.add_argument('--no-parallel', action='store_true', help='Disable parallel extraction')
    ex_parser.add_argument('--max-song-ratio', type=float, default=0.2,
                           help='Skip videos with song_lyrics_ratio >= this (default: 0.2)')
    ex_parser.add_argument('--min-words', type=int, default=20,
                           help='Skip transcripts with fewer than this many words (default: 20)')
    ex_parser.add_argument('--force', action='store_true',
                           help='Re-extract even if already processed (clears previous extraction)')
    ex_parser.add_argument('--thinking', action='store_true',
                           help='Enable Qwen3 thinking mode (/think) for deeper reasoning (slower)')
    ex_parser.set_defaults(func=cmd_extract)

    # --- analyze subcommand ---
    an_parser = subparsers.add_parser('analyze', help='Run clustering and visualization')
    an_parser.add_argument('--cluster-method', choices=['kmeans', 'dbscan'], default='kmeans')
    an_parser.add_argument('--viz-method', choices=['pca', 'tsne', 'umap'], default='umap')
    an_parser.add_argument('--clusters', type=int, help='Number of clusters (auto if not set)')
    an_parser.add_argument('--min-confidence', type=float, default=0.6, help='Min symptom confidence')
    an_parser.add_argument('--feature-method', choices=['tfidf', 'combined'], default='combined')
    an_parser.set_defaults(func=cmd_analyze)

    # --- stats subcommand ---
    st_parser = subparsers.add_parser('stats', help='Show database statistics')
    st_parser.add_argument('--detailed', action='store_true', help='Show detailed statistics')
    st_parser.set_defaults(func=cmd_stats)

    # --- discover subcommand ---
    disc_parser = subparsers.add_parser('discover', help='Discover new TikTok videos')
    disc_parser.add_argument('--url', action='append', help='Video URL to expand user from')
    disc_parser.add_argument('--user', action='append', help='TikTok username')
    disc_parser.add_argument('--expand-users', help='File to extract and expand users from')
    disc_parser.add_argument('--hashtag', action='append', help='Hashtag to search')
    disc_parser.add_argument('--search', action='append', help='Search query')
    disc_parser.add_argument('--output', default='urls.txt', help='Output file')
    disc_parser.add_argument('--append', action='store_true', help='Append to output file')
    disc_parser.add_argument('--max-videos', type=int, help='Max videos per source')
    disc_parser.set_defaults(func=cmd_discover)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Run the appropriate subcommand
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
