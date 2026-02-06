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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats
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
        Includes all enhanced schema fields for research analysis.

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
                    s.severity,
                    s.temporal_pattern,
                    s.body_location,
                    s.triggers,
                    s.is_personal_experience,
                    s.extractor_model,
                    s.extractor_provider,
                    s.extracted_at,
                    v.title as video_title,
                    v.author as video_author,
                    v.platform,
                    v.tags,
                    v.hashtags,
                    v.duration,
                    v.view_count,
                    v.like_count
                FROM symptoms s
                JOIN videos v ON s.video_id = v.id
                WHERE s.confidence >= %s
                ORDER BY s.confidence DESC
            """
            df = pd.read_sql(query, conn, params=(self.min_confidence,))

        print(f"Loaded {len(df)} symptoms from {df['video_id'].nunique()} videos")

        # Print summary of enhanced fields
        if len(df) > 0:
            personal_pct = df['is_personal_experience'].sum() / len(df) * 100 if 'is_personal_experience' in df else 0
            print(f"  Personal experiences: {personal_pct:.1f}%")
            if 'severity' in df and df['severity'].notna().any():
                severity_counts = df['severity'].value_counts()
                print(f"  Severity breakdown: {severity_counts.to_dict()}")

        return df

    def prepare_features(self, df: pd.DataFrame, method: str = 'tfidf') -> np.ndarray:
        """
        Convert symptoms to numerical features for clustering.

        Args:
            df: DataFrame with symptom data
            method: Feature extraction method ('tfidf', 'combined', 'enhanced')

        Returns:
            Feature matrix as numpy array
        """
        n_samples = len(df)

        # Adaptive min_df based on dataset size
        min_df = 1 if n_samples < 50 else 2

        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=500,
                ngram_range=(1, 3),
                min_df=min_df,
                max_df=0.8,
                stop_words='english'
            )
            features = vectorizer.fit_transform(df['symptom']).toarray()
            self.vectorizer = vectorizer

        elif method == 'combined':
            vectorizer = TfidfVectorizer(
                max_features=400,
                ngram_range=(1, 3),
                min_df=min_df,
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

        elif method == 'enhanced':
            # Use all available enhanced fields
            vectorizer = TfidfVectorizer(
                max_features=300,
                ngram_range=(1, 3),
                min_df=min_df,
                max_df=0.8,
                stop_words='english'
            )
            text_features = vectorizer.fit_transform(df['symptom']).toarray()

            # One-hot encode categories
            category_dummies = pd.get_dummies(df['category'], prefix='cat')

            # One-hot encode severity (if available)
            severity_dummies = pd.DataFrame()
            if 'severity' in df.columns and df['severity'].notna().any():
                severity_dummies = pd.get_dummies(df['severity'].fillna('unspecified'), prefix='sev')

            # One-hot encode temporal pattern (if available)
            temporal_dummies = pd.DataFrame()
            if 'temporal_pattern' in df.columns and df['temporal_pattern'].notna().any():
                temporal_dummies = pd.get_dummies(df['temporal_pattern'].fillna('unspecified'), prefix='temp')

            # Personal experience flag
            personal_flag = np.zeros((len(df), 1))
            if 'is_personal_experience' in df.columns:
                personal_flag = df['is_personal_experience'].fillna(True).astype(int).values.reshape(-1, 1)

            # Combine all features
            feature_list = [text_features, category_dummies.values, df[['confidence']].values, personal_flag]
            if len(severity_dummies) > 0:
                feature_list.append(severity_dummies.values)
            if len(temporal_dummies) > 0:
                feature_list.append(temporal_dummies.values)

            features = np.hstack(feature_list)
            self.vectorizer = vectorizer

        else:
            raise ValueError(f"Unknown method: {method}")

        # Normalize features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        self.scaler = scaler

        print(f"Created feature matrix: {features.shape}")
        return features

    def cluster_kmeans(self, features: np.ndarray, n_clusters: Optional[int] = None,
                      optimize: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Perform K-means clustering on symptom features.

        Args:
            features: Feature matrix
            n_clusters: Number of clusters (auto-optimized if None and optimize=True)
            optimize: Whether to find optimal k using elbow method

        Returns:
            Tuple of (cluster labels, metrics)
        """
        # Fix: Only use default if n_clusters is None AND we're not optimizing
        if n_clusters is None and not optimize:
            n_clusters = CLUSTER_COUNT

        if optimize and n_clusters is None:
            # Find optimal k using elbow method
            print("Optimizing cluster count...")
            max_k = min(50, len(features) // 5, len(features) - 1)
            if max_k < 3:
                print(f"Dataset too small for optimization, using k=2")
                n_clusters = 2
            else:
                k_range = range(2, max_k + 1)
                inertias = []
                silhouettes = []

                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(features)
                    inertias.append(kmeans.inertia_)
                    if k > 1:
                        silhouettes.append(silhouette_score(features, labels))

                # Find elbow point using second derivative
                if len(inertias) >= 3:
                    diffs = np.diff(inertias)
                    diffs2 = np.diff(diffs)
                    elbow_idx = np.argmax(diffs2) + 2  # +2 because of two diffs
                    optimal_k = list(k_range)[min(elbow_idx, len(k_range) - 1)]

                    # Also consider silhouette score
                    best_silhouette_k = list(k_range)[np.argmax(silhouettes) + 1] if silhouettes else optimal_k

                    # Average of elbow and silhouette suggestions
                    n_clusters = (optimal_k + best_silhouette_k) // 2
                    print(f"Optimal k: {n_clusters} (elbow: {optimal_k}, silhouette: {best_silhouette_k})")
                else:
                    n_clusters = 3
                    print(f"Using default k={n_clusters} (dataset too small for optimization)")

        # Perform clustering
        print(f"Performing K-means clustering with k={n_clusters}...")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=50,
            max_iter=1000,
            algorithm='elkan'
        )
        labels = kmeans.fit_predict(features)

        # Calculate validation metrics
        metrics = {
            'n_clusters': n_clusters,
            'inertia': float(kmeans.inertia_),
            'algorithm': 'kmeans'
        }

        if n_clusters > 1 and len(set(labels)) > 1:
            metrics['silhouette_score'] = float(silhouette_score(features, labels))
            metrics['davies_bouldin_score'] = float(davies_bouldin_score(features, labels))
            metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(features, labels))

            print(f"Clustering complete: {n_clusters} clusters")
            print(f"  Silhouette Score: {metrics['silhouette_score']:.3f} (higher is better, -1 to 1)")
            print(f"  Davies-Bouldin Index: {metrics['davies_bouldin_score']:.3f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.1f} (higher is better)")
        else:
            print(f"Clustering complete: {n_clusters} clusters (metrics require >1 cluster)")

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

        # Calculate validation metrics if we have valid clusters
        if n_clusters > 1:
            # Exclude noise points for metrics
            valid_mask = labels != -1
            if valid_mask.sum() > n_clusters:
                metrics['silhouette_score'] = float(silhouette_score(features[valid_mask], labels[valid_mask]))
                print(f"  Silhouette Score (excl. noise): {metrics['silhouette_score']:.3f}")

        print(f"Found {n_clusters} clusters, {n_noise} noise points")
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
            print(f"PCA complete: {explained_var:.1f}% variance explained")

        elif method == 'tsne':
            perplexity = min(30, max(5, len(features) // 4))
            reducer = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=perplexity,
                n_iter=2000,
                learning_rate='auto',
                n_jobs=-1
            )
            reduced = reducer.fit_transform(features)
            print(f"t-SNE complete (perplexity={perplexity})")

        elif method == 'umap':
            if UMAP is None:
                raise ImportError("UMAP is not installed. Add 'umap-learn' to use method='umap'.")
            n_neighbors = min(15, max(2, len(features) // 5))
            reducer = UMAP(
                n_components=n_components,
                random_state=42,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                metric='cosine'
            )
            reduced = reducer.fit_transform(features)
            print(f"UMAP complete (n_neighbors={n_neighbors})")

        else:
            raise ValueError(f"Unknown method: {method}")

        self.reducer = reducer
        return reduced

    def visualize_clusters(self, df: pd.DataFrame, features: np.ndarray,
                          labels: np.ndarray, method: str = 'umap',
                          save_path: Optional[Path] = None) -> Path:
        """
        Create cluster visualizations with enhanced fields.

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

        # Create figure with subplots (3x2 for enhanced visualizations)
        fig, axes = plt.subplots(3, 2, figsize=(20, 24))
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
        for cat in categories:
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

        # Plot 5: Severity distribution (if available)
        ax5 = axes[2, 0]
        if 'severity' in df_viz.columns and df_viz['severity'].notna().any():
            severity_order = ['mild', 'moderate', 'severe', 'unspecified']
            severity_counts = df_viz['severity'].value_counts()
            severity_counts = severity_counts.reindex([s for s in severity_order if s in severity_counts.index])
            colors = {'mild': '#90EE90', 'moderate': '#FFD700', 'severe': '#FF6B6B', 'unspecified': '#D3D3D3'}
            ax5.bar(severity_counts.index, severity_counts.values,
                   color=[colors.get(s, '#D3D3D3') for s in severity_counts.index], alpha=0.7)
            ax5.set_title('Symptom Severity Distribution', fontsize=14)
            ax5.set_xlabel('Severity')
            ax5.set_ylabel('Count')
        else:
            ax5.text(0.5, 0.5, 'Severity data not available', ha='center', va='center', fontsize=12)
            ax5.set_title('Symptom Severity Distribution', fontsize=14)

        # Plot 6: Temporal pattern distribution (if available)
        ax6 = axes[2, 1]
        if 'temporal_pattern' in df_viz.columns and df_viz['temporal_pattern'].notna().any():
            temporal_counts = df_viz['temporal_pattern'].value_counts()
            ax6.pie(temporal_counts.values, labels=temporal_counts.index, autopct='%1.1f%%',
                   colors=plt.cm.Set3.colors[:len(temporal_counts)])
            ax6.set_title('Temporal Pattern Distribution', fontsize=14)
        else:
            ax6.text(0.5, 0.5, 'Temporal pattern data not available', ha='center', va='center', fontsize=12)
            ax6.set_title('Temporal Pattern Distribution', fontsize=14)

        plt.tight_layout()

        # Save
        if not save_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = VISUALIZATION_DIR / f'clusters_{method}_{timestamp}.png'

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {save_path}")

        plt.close()
        return save_path

    def test_cluster_significance(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        Perform statistical tests on cluster distributions.

        Args:
            df: DataFrame with symptom data
            labels: Cluster labels

        Returns:
            Dictionary with statistical test results
        """
        df_test = df.copy()
        df_test['cluster'] = labels

        results = {}

        # Chi-square test for category distribution across clusters
        if 'category' in df_test.columns:
            contingency = pd.crosstab(df_test['cluster'], df_test['category'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results['category_chi2'] = {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            print(f"Category distribution chi2: {chi2:.2f}, p={p_value:.4f} {'(significant)' if p_value < 0.05 else ''}")

        # Chi-square for severity if available
        if 'severity' in df_test.columns and df_test['severity'].notna().any():
            contingency = pd.crosstab(df_test['cluster'], df_test['severity'].fillna('unspecified'))
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            results['severity_chi2'] = {
                'statistic': float(chi2),
                'p_value': float(p_value),
                'degrees_of_freedom': int(dof),
                'significant': p_value < 0.05
            }
            print(f"Severity distribution chi2: {chi2:.2f}, p={p_value:.4f} {'(significant)' if p_value < 0.05 else ''}")

        # ANOVA for confidence scores across clusters
        cluster_groups = [group['confidence'].values for _, group in df_test.groupby('cluster')]
        if len(cluster_groups) > 1 and all(len(g) > 0 for g in cluster_groups):
            f_stat, p_value = stats.f_oneway(*cluster_groups)
            results['confidence_anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
            print(f"Confidence ANOVA: F={f_stat:.2f}, p={p_value:.4f} {'(significant)' if p_value < 0.05 else ''}")

        return results

    def generate_cluster_report(self, df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        Generate detailed report about clusters including enhanced fields.

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

            # Add severity distribution if available
            if 'severity' in cluster_data.columns and cluster_data['severity'].notna().any():
                cluster_info['severity_distribution'] = cluster_data['severity'].value_counts().to_dict()

            # Add temporal pattern distribution if available
            if 'temporal_pattern' in cluster_data.columns and cluster_data['temporal_pattern'].notna().any():
                cluster_info['temporal_pattern_distribution'] = cluster_data['temporal_pattern'].value_counts().to_dict()

            # Add personal experience ratio
            if 'is_personal_experience' in cluster_data.columns:
                personal_count = cluster_data['is_personal_experience'].sum()
                cluster_info['personal_experience_ratio'] = float(personal_count / len(cluster_data))

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
        print(f"Results exported: {output_path}")

        return output_path


if __name__ == '__main__':
    # Run analysis
    analyzer = SymptomAnalyzer()

    try:
        # Load data
        df = analyzer.load_symptom_data()

        if len(df) < 10:
            print("Not enough data for meaningful analysis (need at least 10 symptoms)")
        else:
            # Prepare features using enhanced method
            features = analyzer.prepare_features(df, method='enhanced')

            # Cluster with optimization
            labels, metrics = analyzer.cluster_kmeans(features, optimize=True)

            # Statistical tests
            print("\n--- Statistical Tests ---")
            stats_results = analyzer.test_cluster_significance(df, labels)

            # Visualize
            viz_path = analyzer.visualize_clusters(df, features, labels, method='umap')

            # Generate report
            report = analyzer.generate_cluster_report(df, labels)
            report['validation_metrics'] = metrics
            report['statistical_tests'] = stats_results

            print(f"\nCluster Report:")
            print(json.dumps(report, indent=2))

            # Export
            export_path = analyzer.export_results(df, labels)

            print(f"\nAnalysis complete!")
            print(f"  Visualization: {viz_path}")
            print(f"  Export: {export_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
