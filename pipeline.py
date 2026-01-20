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
    get_treatment_statistics, get_comorbidity_matrix
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
                 ollama_url: Optional[str] = None):
        """
        Initialize the pipeline.

        Args:
            whisper_model: Whisper model for transcription (use large-v3 for RTX 4090)
            min_confidence: Minimum confidence score for symptoms
            parallel_extraction: Enable parallel symptom extraction
        """
        self.downloader = VideoDownloader()
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.extractor = SymptomExtractor(
            max_workers=10 if parallel_extraction else 1,
            provider=extractor_provider,
            model=extractor_model,
            ollama_url=ollama_url,
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
            print(f"  Transcribed: {transcript_result['word_count']} words")
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
                      resume_run_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple URLs through the pipeline with progress tracking.

        Args:
            urls: List of video URLs
            tags: Optional tags for all videos
            resume_run_id: Optional run ID to resume from (for interrupted runs)

        Returns:
            List of processing results
        """
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


if __name__ == '__main__':
    # Example: Run pipeline on a sample URL
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='EDS/MCAS/POTS Research Pipeline')
    parser.add_argument('urls', nargs='*', help='Video URLs to process')
    parser.add_argument('--analyze', action='store_true', help='Run analysis on collected data')
    parser.add_argument('--resume', type=int, help='Resume a previous run by ID')
    parser.add_argument('--status', action='store_true', help='Show status of the latest run')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--whisper-model', default='large-v3', help='Whisper model size')
    parser.add_argument('--provider', default=None, help='Extractor provider (anthropic or ollama)')
    parser.add_argument('--model', default=None, help='Extractor model name')
    parser.add_argument('--tags', nargs='+', default=['EDS', 'MCAS', 'POTS'], help='Tags for videos')

    args = parser.parse_args()

    # Initialize pipeline (optimized for RTX 4090!)
    pipeline = ResearchPipeline(
        whisper_model=args.whisper_model,
        min_confidence=0.6,
        parallel_extraction=True,
        extractor_provider=args.provider,
        extractor_model=args.model
    )

    if args.status:
        # Show status of latest run
        run_id = args.resume or get_latest_run_id()
        if run_id:
            progress = get_run_progress_summary(run_id)
            print(f"\n{'='*60}")
            print(f"Run {run_id} Status:")
            print(f"  Total URLs: {progress['total']}")
            print(f"  Completed: {progress['completed']}")
            print(f"  Failed: {progress['failed']}")
            print(f"  In Progress: {progress['in_progress']}")
            print(f"\nBy stage:")
            for stage, count in progress['by_stage'].items():
                print(f"  {stage}: {count}")
            print('='*60)
        else:
            print("No processing runs found")

    elif args.stats:
        # Show database statistics
        stats = get_symptom_statistics()
        print(f"\n{'='*60}")
        print("Database Statistics:")
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

        # Treatment stats
        treatment_stats = get_treatment_statistics()
        if treatment_stats['top_treatments']:
            print(f"\nTop Treatments:")
            for t in treatment_stats['top_treatments'][:10]:
                eff = f" (eff: {t['avg_effectiveness']:.2f})" if t['avg_effectiveness'] else ""
                print(f"  {t['treatment_name']} ({t['treatment_type']}): {t['mention_count']} mentions{eff}")

        # Comorbidity
        comorbidity = get_comorbidity_matrix()
        if comorbidity:
            print(f"\nComorbidity Pairs:")
            for c in comorbidity[:10]:
                print(f"  {c['condition_a']} + {c['condition_b']}: {c['video_count']} videos")

        print('='*60)

    elif args.analyze:
        # Just run analysis
        results = pipeline.analyze_all()
        print(json.dumps(results, indent=2, default=str))

    elif args.resume:
        # Resume a previous run
        results = pipeline.process_batch([], tags=args.tags, resume_run_id=args.resume)
        print("\n" + json.dumps(results, indent=2, default=str))

    elif args.urls:
        # Process URLs
        results = pipeline.process_batch(args.urls, tags=args.tags)
        print("\n" + json.dumps(results, indent=2, default=str))

    else:
        parser.print_help()
