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
from database import get_all_videos_with_transcripts, get_symptom_statistics
import config


class ResearchPipeline:
    """Orchestrates the complete research pipeline."""

    def __init__(self,
                 whisper_model: str = 'large-v3',
                 min_confidence: float = 0.6,
                 parallel_extraction: bool = True):
        """
        Initialize the pipeline.

        Args:
            whisper_model: Whisper model for transcription (use large-v3 for RTX 4090)
            min_confidence: Minimum confidence score for symptoms
            parallel_extraction: Enable parallel symptom extraction
        """
        self.downloader = VideoDownloader()
        self.transcriber = AudioTranscriber(model_size=whisper_model)
        self.extractor = SymptomExtractor(max_workers=10 if parallel_extraction else 1)
        self.analyzer = SymptomAnalyzer(min_confidence=min_confidence)

        self.stats = {
            'videos_downloaded': 0,
            'videos_transcribed': 0,
            'symptoms_extracted': 0,
            'errors': []
        }

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

            # Stage 3: Extract symptoms
            print(f"\n{'='*80}")
            print(f"STAGE 3: Extracting symptoms")
            print('='*80)
            extraction_result = self.extractor.extract_symptoms(video_id)
            result['stages']['extract'] = extraction_result

            if extraction_result.get('success'):
                self.stats['symptoms_extracted'] += extraction_result.get('symptoms_saved', 0)

            print(f"\n{'='*80}")
            print(f"✓ Pipeline complete for: {url}")
            print(f"  Downloaded: {download_result['audio_path']}")
            print(f"  Transcribed: {transcript_result['word_count']} words")
            print(f"  Symptoms: {extraction_result.get('symptoms_saved', 0)} extracted")
            print('='*80)

        except Exception as e:
            print(f"\n✗ Pipeline failed for {url}: {e}")
            result['success'] = False
            result['error'] = str(e)
            self.stats['errors'].append({'url': url, 'error': str(e)})

        return result

    def process_batch(self, urls: List[str], tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple URLs through the pipeline.

        Args:
            urls: List of video URLs
            tags: Optional tags for all videos

        Returns:
            List of processing results
        """
        print(f"\n{'#'*80}")
        print(f"# Starting pipeline for {len(urls)} videos")
        print(f"# Configuration:")
        print(f"#   Whisper model: {self.transcriber.model_size}")
        print(f"#   Device: {self.transcriber.device}")
        print(f"#   Min confidence: {self.analyzer.min_confidence}")
        print(f"#   Parallel extraction: {self.extractor.max_workers > 1}")
        print('#'*80)

        results = []
        start_time = datetime.now()

        for i, url in enumerate(urls, 1):
            print(f"\n\n{'#'*80}")
            print(f"# VIDEO {i}/{len(urls)}")
            print('#'*80)

            result = self.process_url(url, tags)
            results.append(result)

        # Final statistics
        elapsed = (datetime.now() - start_time).total_seconds()
        success_count = sum(1 for r in results if r['success'])

        print(f"\n\n{'#'*80}")
        print(f"# PIPELINE COMPLETE")
        print('#'*80)
        print(f"  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"  Success rate: {success_count}/{len(urls)} ({success_count/len(urls)*100:.1f}%)")
        print(f"  Videos downloaded: {self.stats['videos_downloaded']}")
        print(f"  Videos transcribed: {self.stats['videos_transcribed']}")
        print(f"  Total symptoms extracted: {self.stats['symptoms_extracted']}")
        if self.stats['errors']:
            print(f"  Errors: {len(self.stats['errors'])}")
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

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <url1> [url2] [url3] ...")
        print("   or: python pipeline.py --analyze")
        sys.exit(1)

    # Initialize pipeline (optimized for RTX 4090!)
    pipeline = ResearchPipeline(
        whisper_model='large-v3',
        min_confidence=0.6,
        parallel_extraction=True
    )

    if sys.argv[1] == '--analyze':
        # Just run analysis
        results = pipeline.analyze_all()
        print(json.dumps(results, indent=2, default=str))
    else:
        # Process URLs
        urls = sys.argv[1:]
        results = pipeline.process_batch(urls, tags=['EDS', 'MCAS', 'POTS'])

        # Print summary
        print("\n" + json.dumps(results, indent=2, default=str))
