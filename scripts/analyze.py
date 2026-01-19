#!/usr/bin/env python3
"""
Analyze extracted symptoms with clustering and visualization.

Usage:
    python scripts/analyze.py
    python scripts/analyze.py --cluster-method dbscan
    python scripts/analyze.py --viz-method tsne --clusters 8
"""
import sys
import argparse
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from analyzer import SymptomAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze and cluster symptoms')
    parser.add_argument('--cluster-method', choices=['kmeans', 'dbscan'], default='kmeans',
                       help='Clustering algorithm (default: kmeans)')
    parser.add_argument('--viz-method', choices=['pca', 'tsne', 'umap'], default='umap',
                       help='Visualization method (default: umap)')
    parser.add_argument('--clusters', type=int, help='Number of clusters (auto-optimized if not specified)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                       help='Minimum confidence score (default: 0.6)')
    parser.add_argument('--feature-method', choices=['tfidf', 'combined'], default='combined',
                       help='Feature extraction method (default: combined)')
    parser.add_argument('--export-json', action='store_true',
                       help='Export cluster report as JSON')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = SymptomAnalyzer(min_confidence=args.min_confidence)

    try:
        print("Loading symptom data...")
        df = analyzer.load_symptom_data()

        if len(df) < 10:
            print("✗ Not enough data for analysis (need at least 10 symptoms)")
            print(f"  Current count: {len(df)}")
            print("  Run the pipeline to collect more data first")
            sys.exit(1)

        # Prepare features
        print(f"\nPreparing features using {args.feature_method} method...")
        features = analyzer.prepare_features(df, method=args.feature_method)

        # Cluster
        print(f"\nClustering using {args.cluster_method}...")
        if args.cluster_method == 'kmeans':
            labels, metrics = analyzer.cluster_kmeans(
                features,
                n_clusters=args.clusters,
                optimize=(args.clusters is None)
            )
        else:
            labels, metrics = analyzer.cluster_dbscan(features)

        print(f"Clustering metrics: {metrics}")

        # Visualize
        print(f"\nGenerating visualizations using {args.viz_method}...")
        viz_path = analyzer.visualize_clusters(df, features, labels, method=args.viz_method)

        # Generate report
        print("\nGenerating cluster report...")
        report = analyzer.generate_cluster_report(df, labels)

        # Export
        export_path = analyzer.export_results(df, labels)

        # Print summary
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print('='*80)
        print(f"Total symptoms: {report['total_symptoms']}")
        print(f"Number of clusters: {report['n_clusters']}")
        print(f"\nVisualization: {viz_path}")
        print(f"Export: {export_path}")

        print(f"\nCluster Summary:")
        for cluster in report['clusters']:
            print(f"  Cluster {cluster['cluster_id']}: {cluster['size']} symptoms, "
                  f"avg confidence {cluster['avg_confidence']:.3f}")

        # Print top symptoms per cluster
        print(f"\nTop symptoms by cluster:")
        for cluster in report['clusters']:
            print(f"\n  Cluster {cluster['cluster_id']}:")
            for i, symptom in enumerate(cluster['top_symptoms'][:3], 1):
                print(f"    {i}. {symptom['symptom']} (conf: {symptom['confidence']:.3f})")

        # Export JSON report if requested
        if args.export_json:
            json_path = export_path.parent / f"report_{export_path.stem}.json"
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nJSON report: {json_path}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
