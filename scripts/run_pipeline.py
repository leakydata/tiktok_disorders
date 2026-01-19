#!/usr/bin/env python3
"""
Run the complete research pipeline on a set of URLs.

Usage:
    python scripts/run_pipeline.py --url "https://youtube.com/watch?v=..."
    python scripts/run_pipeline.py --file urls.txt --tags EDS,MCAS
    python scripts/run_pipeline.py --file urls.txt --analyze
"""
import sys
import argparse
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import ResearchPipeline


def main():
    parser = argparse.ArgumentParser(description='Run complete research pipeline')
    parser.add_argument('--url', help='Single video URL to process')
    parser.add_argument('--file', help='File containing URLs (one per line)')
    parser.add_argument('--tags', help='Comma-separated tags', default='')
    parser.add_argument('--analyze', action='store_true',
                       help='Run analysis after processing all videos')
    parser.add_argument('--whisper-model', default='large-v3',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model (default: large-v3 for RTX 4090)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum confidence for symptoms (default: 0.6)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel extraction')
    parser.add_argument('--provider', choices=['anthropic', 'ollama'],
                       help='Model provider (default: from EXTRACTOR_PROVIDER)')
    parser.add_argument('--model',
                       help='Model name (default: provider-specific)')
    parser.add_argument('--ollama-url',
                       help='Ollama base URL (default: from OLLAMA_URL)')
    parser.add_argument('--cluster-method', choices=['kmeans', 'dbscan'], default='kmeans',
                       help='Clustering method for analysis')
    parser.add_argument('--viz-method', choices=['pca', 'tsne', 'umap'], default='umap',
                       help='Visualization method for analysis')

    args = parser.parse_args()

    if not args.url and not args.file:
        parser.error("Either --url or --file is required")

    # Parse tags
    tags = [t.strip() for t in args.tags.split(',') if t.strip()] if args.tags else None

    # Initialize pipeline (optimized for RTX 4090!)
    print("Initializing pipeline...")
    print(f"  Whisper model: {args.whisper_model}")
    print(f"  Min confidence: {args.min_confidence}")
    print(f"  Parallel extraction: {not args.no_parallel}")
    if args.provider:
        print(f"  Provider: {args.provider}")
    if args.model:
        print(f"  Model: {args.model}")

    pipeline = ResearchPipeline(
        whisper_model=args.whisper_model,
        min_confidence=args.min_confidence,
        parallel_extraction=not args.no_parallel,
        extractor_provider=args.provider,
        extractor_model=args.model,
        ollama_url=args.ollama_url,
    )

    try:
        # Collect URLs
        if args.url:
            urls = [args.url]
        else:
            with open(args.file, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not urls:
            print("✗ No URLs found")
            sys.exit(1)

        # Process videos
        results = pipeline.process_batch(urls, tags=tags)

        # Save results
        output_file = Path('pipeline_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

        # Run analysis if requested
        if args.analyze:
            print("\n" + "="*80)
            print("Running analysis on all collected data...")
            print("="*80)

            try:
                analysis_results = pipeline.analyze_all(
                    cluster_method=args.cluster_method,
                    viz_method=args.viz_method
                )

                # Save analysis results
                analysis_file = Path('analysis_results.json')
                with open(analysis_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2, default=str)
                print(f"\nAnalysis results saved to: {analysis_file}")

            except ValueError as e:
                print(f"⚠ Cannot run analysis: {e}")

        # Print statistics
        stats = pipeline.get_statistics()
        print("\n" + "="*80)
        print("FINAL STATISTICS")
        print("="*80)
        print(f"Database totals:")
        print(f"  Videos: {stats['database']['total_videos']}")
        print(f"  Symptoms: {stats['database']['total_symptoms']}")
        print(f"\nThis session:")
        print(f"  Videos downloaded: {stats['pipeline_session']['videos_downloaded']}")
        print(f"  Videos transcribed: {stats['pipeline_session']['videos_transcribed']}")
        print(f"  Symptoms extracted: {stats['pipeline_session']['symptoms_extracted']}")
        if stats['pipeline_session']['errors']:
            print(f"  Errors: {len(stats['pipeline_session']['errors'])}")

    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
