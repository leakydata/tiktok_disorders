"""
Advanced symptom analysis module with clustering and visualization.
Optimized for high-memory systems (389GB RAM) and RTX 4090 GPU.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
except ImportError:
    UMAP = None
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

from config import VISUALIZATION_DIR, CLUSTER_COUNT, MIN_CONFIDENCE_SCORE, ensure_directories
from database import get_connection, get_symptom_statistics


class SymptomAnalyzer:
    """Advanced analysis and clustering of extracted symptoms."""

    def __init__(self, min_confidence: Optional[float] = None):
        """
        Initialize the analyzer.

        Args:
            min_confidence: Minimum confidence score for symptoms to include
        """
        self.min_confidence = min_confidence or MIN_CONFIDENCE_SCORE
        ensure_directories()

        # Set style for beautiful visualizations
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (16, 10)
        plt.rcParams['font.size'] = 11

    def load_symptom_data(self) -> pd.DataFrame:
        """
        Load all symptoms from database as a pandas DataFrame.

        Returns:
            DataFrame with symptom data
        """
        with get_connection() as conn:
            query = """
                SELECT
                    s.id,
                    s.video_id,
                    s.category,
                    s.symptom,
                    s.confidence,
                    s.context,
                    s.extracted_at,
                    v.title as video_title,
                    v.author as video_author,
                    v.tags,
                    v.duration
                FROM symptoms s
                JOIN videos v ON s.video_id = v.id
                WHERE s.confidence >= %s
                ORDER BY s.confidence DESC
            """
            df = pd.read_sql(query, conn, params=(self.min_confidence,))

        print(f"✓ Loaded {len(df)} symptoms from {df['video_id'].nunique()} videos")
        return df

    def prepare_features(self, df: pd.DataFrame, method: str = 'tfidf') -> np.ndarray:
        """
        Convert symptoms to numerical features for clustering.

        Args:
            df: DataFrame with symptom data
            method: Feature extraction method ('tfidf', 'bow', 'combined')

        Returns:
            Feature matrix as numpy array
        """
        if method == 'tfidf':
            # TF-IDF on symptom text
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            features = vectorizer.fit_transform(df['symptom']).toarray()
            self.vectorizer = vectorizer

        elif method == 'combined':
            # Combine symptom text with category and confidence
            vectorizer = TfidfVectorizer(
                max_features=400,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )
            text_features = vectorizer.fit_transform(df['symptom']).toarray()

            # One-hot encode categories
            category_dummies = pd.get_dummies(df['category'], prefix='cat')

            # Combine all features
            features = np.hstack([
                text_features,
                category_dummies.values,
                df[['confidence']].values
            ])
            self.vectorizer = vectorizer

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        self.scaler = scaler

        print(f"✓ Created feature matrix: {features.shape}")
        return features

    def cluster_kmeans(self, features: np.ndarray, n_clusters: int = None,
                      optimize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Perform K-means clustering on symptom features.

        Args:
            features: Feature matrix
            n_clusters: Number of clusters (auto-optimized if None)
            optimize: Whether to find optimal k using elbow method

        Returns:
            Tuple of (cluster labels, metrics)
        """
        n_clusters = n_clusters or CLUSTER_COUNT

        if optimize and n_clusters is None:
            # Find optimal k using elbow method (with plenty of RAM, we can test many!)
            print("Optimizing cluster count...")
            inertias = []
            k_range = range(2, min(50, len(features) // 10))

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)

            # Find elbow point (simplified)
            diffs = np.diff(inertias)
            diff_ratios = diffs[:-1] / diffs[1:]
            optimal_k = k_range[np.argmax(diff_ratios) + 1]
            print(f"✓ Optimal cluster count: {optimal_k}")
            n_clusters = optimal_k

        # Perform clustering with high iteration count (we have time!)
        print(f"Performing K-means clustering with k={n_clusters}...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=50,  # Multiple initializations for best result
            max_iter=1000,
            algorithm='elkan'  # Faster for many features
        )
        labels = kmeans.fit_predict(features)

        metrics = {
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_,
            'algorithm': 'kmeans'
        }

        print(f"✓ Clustering complete: {n_clusters} clusters")
        return labels, metrics

    def cluster_dbscan(self, features: np.ndarray, eps: float = 0.5,
                      min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Perform DBSCAN clustering (density-based).

        Args:
            features: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood

        Returns:
            Tuple of (cluster labels, metrics)
        """
        print(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(features)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        metrics = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'algorithm': 'dbscan'
        }

        print(f"✓ Found {n_clusters} clusters, {n_noise} noise points")
        return labels, metrics

    def reduce_dimensions(self, features: np.ndarray, method: str = 'umap',
                         n_components: int = 2) -> np.ndarray:
        """
        Reduce feature dimensions for visualization.

        Args:
            features: High-dimensional feature matrix
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Target dimensionality (2 or 3)

        Returns:
            Reduced feature matrix
        """
        print(f"Reducing dimensions using {method.upper()}...")

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(features)
            explained_var = sum(reducer.explained_variance_ratio_) * 100
            print(f"✓ PCA complete: {explained_var:.1f}% variance explained")

        elif method == 'tsne':
            # t-SNE is compute-intensive but perfect for RTX 4090
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=min(30, len(features) - 1),
                n_iter=2000,  # High iteration count for quality
                learning_rate='auto',
                n_jobs=-1
            )
            reduced = reducer.fit_transform(features)
            print(f"✓ t-SNE complete")

        elif method == 'umap':
            # UMAP: best of both worlds - fast and high quality
            if UMAP is None:
                raise ImportError("UMAP is not installed. Add 'umap-learn' to use method='umap'.")
            reducer = UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=15,
                min_dist=0.1,
                metric='cosine'
            )
            reduced = reducer.fit_transform(features)
            print(f"✓ UMAP complete")

        else:
            raise ValueError(f"Unknown method: {method}")

        self.reducer = reducer
        return reduced

    def visualize_clusters(self, df: pd.DataFrame, features: np.ndarray,
                          labels: np.ndarray, method: str = 'umap',
                          save_path: Optional[Path] = None) -> Path:
        """
        Create beautiful cluster visualizations.

        Args:
            df: DataFrame with symptom data
            features: Feature matrix
            labels: Cluster labels
            method: Dimensionality reduction method
            save_path: Optional custom save path

        Returns:
            Path to saved visualization
        """
        # Reduce to 2D for visualization
        coords_2d = self.reduce_dimensions(features, method=method, n_components=2)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Symptom Cluster Analysis', fontsize=18, fontweight='bold')

        # Add cluster info to dataframe
        df_viz = df.copy()
        df_viz['cluster'] = labels
        df_viz['x'] = coords_2d[:, 0]
        df_viz['y'] = coords_2d[:, 1]

        # Plot 1: Clusters colored by cluster ID
        ax1 = axes[0, 0]
        scatter = ax1.scatter(
            df_viz['x'], df_viz['y'],
            c=df_viz['cluster'],
            cmap='tab20',
            alpha=0.6,
            s=50
        )
        ax1.set_title(f'Symptom Clusters ({method.upper()} projection)', fontsize=14)
        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        plt.colorbar(scatter, ax=ax1, label='Cluster ID')

        # Plot 2: Colored by category
        ax2 = axes[0, 1]
        categories = df_viz['category'].unique()
        for i, cat in enumerate(categories):
            mask = df_viz['category'] == cat
            ax2.scatter(
                df_viz[mask]['x'], df_viz[mask]['y'],
                label=cat,
                alpha=0.6,
                s=50
            )
        ax2.set_title('Symptoms by Category', fontsize=14)
        ax2.set_xlabel(f'{method.upper()} 1')
        ax2.set_ylabel(f'{method.upper()} 2')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        # Plot 3: Cluster sizes
        ax3 = axes[1, 0]
        cluster_sizes = df_viz['cluster'].value_counts().sort_index()
        ax3.bar(cluster_sizes.index, cluster_sizes.values, color='steelblue', alpha=0.7)
        ax3.set_title('Cluster Sizes', fontsize=14)
        ax3.set_xlabel('Cluster ID')
        ax3.set_ylabel('Number of Symptoms')
        ax3.grid(axis='y', alpha=0.3)

        # Plot 4: Category distribution per cluster
        ax4 = axes[1, 1]
        cluster_category = pd.crosstab(df_viz['cluster'], df_viz['category'])
        cluster_category.plot(kind='bar', stacked=True, ax=ax4, alpha=0.8)
        ax4.set_title('Category Distribution by Cluster', fontsize=14)
        ax4.set_xlabel('Cluster ID')
        ax4.set_ylabel('Count')
        ax4.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax4.tick_params(axis='x', rotation=0)

        plt.tight_layout()

        # Save
        if not save_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = VISUALIZATION_DIR / f'clusters_{method}_{timestamp}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")

        plt.close()
        return save_path

    def generate_cluster_report(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        Generate detailed report about clusters.

        Args:
            df: DataFrame with symptom data
            labels: Cluster labels

        Returns:
            Dictionary with cluster analysis
        """
        df_report = df.copy()
        df_report['cluster'] = labels

        report = {
            'total_symptoms': len(df_report),
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
            'clusters': []
        }

        for cluster_id in sorted(df_report['cluster'].unique()):
            cluster_data = df_report[df_report['cluster'] == cluster_id]

            # Top symptoms in this cluster
            top_symptoms = cluster_data.nlargest(10, 'confidence')[['symptom', 'confidence']].to_dict('records')

            # Category distribution
            category_dist = cluster_data['category'].value_counts().to_dict()

            # Average confidence
            avg_confidence = cluster_data['confidence'].mean()

            cluster_info = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'avg_confidence': float(avg_confidence),
                'top_symptoms': top_symptoms,
                'category_distribution': category_dist,
                'video_count': cluster_data['video_id'].nunique()
            }

            report['clusters'].append(cluster_info)

        return report

    def export_results(self, df: pd.DataFrame, labels: np.ndarray,
                      output_path: Optional[Path] = None) -> Path:
        """
        Export cluster results to CSV.

        Args:
            df: DataFrame with symptom data
            labels: Cluster labels
            output_path: Optional custom output path

        Returns:
            Path to exported file
        """
        df_export = df.copy()
        df_export['cluster'] = labels

        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = VISUALIZATION_DIR / f'symptom_clusters_{timestamp}.csv'

        df_export.to_csv(output_path, index=False)
        print(f"✓ Results exported: {output_path}")

        return output_path


if __name__ == '__main__':
    # Run analysis
    analyzer = SymptomAnalyzer()

    try:
        # Load data
        df = analyzer.load_symptom_data()

        if len(df) < 10:
            print("⚠ Not enough data for meaningful analysis (need at least 10 symptoms)")
        else:
            # Prepare features
            features = analyzer.prepare_features(df, method='combined')

            # Cluster
            labels, metrics = analyzer.cluster_kmeans(features, optimize=True)

            # Visualize
            viz_path = analyzer.visualize_clusters(df, features, labels, method='umap')

            # Generate report
            report = analyzer.generate_cluster_report(df, labels)
            print(f"\nCluster Report:")
            print(json.dumps(report, indent=2))

            # Export
            export_path = analyzer.export_results(df, labels)

            print(f"\n✓ Analysis complete!")
            print(f"  Visualization: {viz_path}")
            print(f"  Export: {export_path}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
