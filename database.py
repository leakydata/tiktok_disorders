"""
Database module for the TikTok Disorders Research Pipeline.
Handles PostgreSQL connections and CRUD operations for videos, transcripts, and symptoms.

Research-grade schema with:
- Data provenance tracking for reproducibility
- Enhanced symptom tracking with severity and temporal patterns
- Video metadata enrichment with engagement metrics
- Symptom co-occurrence tracking
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json

from config import DATABASE_URL


@contextmanager
def get_connection():
    """
    Get a database connection as a context manager.

    Usage:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM videos")
    """
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema with research-grade tables."""
    with get_connection() as conn:
        cur = conn.cursor()

        # =============================================================================
        # Core Tables
        # =============================================================================

        # Videos table - enriched with engagement metrics and creator info
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                author TEXT,
                author_id TEXT,
                author_follower_count INTEGER,
                duration INTEGER,
                upload_date DATE,
                tags TEXT[],
                hashtags TEXT[],
                description TEXT,
                audio_path TEXT,
                audio_size_bytes BIGINT,

                -- Engagement metrics (snapshot at collection time)
                view_count BIGINT,
                like_count BIGINT,
                comment_count BIGINT,
                share_count BIGINT,

                -- Data provenance
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                collection_method TEXT DEFAULT 'yt-dlp',
                collection_version TEXT,

                -- Research metadata
                is_verified_creator BOOLEAN,
                content_warning TEXT,
                research_notes TEXT,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Transcripts table - with model provenance
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                language TEXT,
                language_confidence REAL,

                -- Model provenance
                model_used TEXT,
                model_backend TEXT,
                model_compute_type TEXT,
                transcription_device TEXT,

                -- Quality metrics
                word_count INTEGER,
                audio_duration_seconds REAL,
                words_per_minute REAL,

                segments JSONB,

                -- Timestamps
                transcribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time_seconds REAL,

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id)
            )
        """)

        # Symptoms table - enhanced with severity and temporal patterns
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptoms (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                symptom TEXT NOT NULL,
                confidence REAL NOT NULL,
                context TEXT,

                -- Enhanced symptom tracking
                severity TEXT CHECK (severity IN ('mild', 'moderate', 'severe', 'unspecified')),
                temporal_pattern TEXT CHECK (temporal_pattern IN ('acute', 'chronic', 'intermittent', 'progressive', 'unspecified')),
                body_location TEXT,
                triggers TEXT[],

                -- Whether speaker is describing personal experience vs general info
                is_personal_experience BOOLEAN DEFAULT TRUE,

                -- Model provenance
                extractor_model TEXT,
                extractor_provider TEXT,

                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =============================================================================
        # Research Enhancement Tables
        # =============================================================================

        # Symptom co-occurrence tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptom_cooccurrence (
                id SERIAL PRIMARY KEY,
                symptom_a_id INTEGER REFERENCES symptoms(id) ON DELETE CASCADE,
                symptom_b_id INTEGER REFERENCES symptoms(id) ON DELETE CASCADE,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                temporal_proximity_seconds REAL,
                mentioned_together BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symptom_a_id, symptom_b_id)
            )
        """)

        # Engagement metrics history (for tracking changes over time)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS engagement_snapshots (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                view_count BIGINT,
                like_count BIGINT,
                comment_count BIGINT,
                share_count BIGINT,
                snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Processing runs - track each pipeline execution
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_runs (
                id SERIAL PRIMARY KEY,
                run_type TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                videos_processed INTEGER DEFAULT 0,
                transcripts_created INTEGER DEFAULT 0,
                symptoms_extracted INTEGER DEFAULT 0,
                errors JSONB,
                config_snapshot JSONB,
                notes TEXT
            )
        """)

        # Research annotations - for manual tagging/review
        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                symptom_id INTEGER REFERENCES symptoms(id) ON DELETE CASCADE,
                annotation_type TEXT NOT NULL,
                annotation_value TEXT,
                annotator TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (video_id IS NOT NULL OR symptom_id IS NOT NULL)
            )
        """)

        # =============================================================================
        # Indexes
        # =============================================================================

        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_url ON videos(url)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_platform ON videos(platform)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_author ON videos(author)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_upload_date ON videos(upload_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_collected_at ON videos(collected_at)")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_language ON transcripts(language)")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_video_id ON symptoms(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_category ON symptoms(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_confidence ON symptoms(confidence)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_severity ON symptoms(severity)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_temporal ON symptoms(temporal_pattern)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_personal ON symptoms(is_personal_experience)")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_cooccur_video ON symptom_cooccurrence(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_engagement_video ON engagement_snapshots(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_video ON annotations(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_annotations_symptom ON annotations(symptom_id)")

        conn.commit()
        print("Database schema initialized successfully")


# =============================================================================
# Video Operations
# =============================================================================

def insert_video(url: str, platform: str, video_id: str, metadata: Dict[str, Any]) -> int:
    """
    Insert a video record into the database.

    Args:
        url: Video URL
        platform: Platform name (youtube, tiktok, etc.)
        video_id: Platform-specific video ID
        metadata: Dictionary containing video metadata

    Returns:
        Database ID of the inserted video
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO videos (
                url, platform, video_id, title, author, author_id, author_follower_count,
                duration, upload_date, tags, hashtags, description, audio_path, audio_size_bytes,
                view_count, like_count, comment_count, share_count,
                collection_method, collection_version, is_verified_creator, research_notes
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                title = COALESCE(EXCLUDED.title, videos.title),
                author = COALESCE(EXCLUDED.author, videos.author),
                author_id = COALESCE(EXCLUDED.author_id, videos.author_id),
                author_follower_count = COALESCE(EXCLUDED.author_follower_count, videos.author_follower_count),
                duration = COALESCE(EXCLUDED.duration, videos.duration),
                upload_date = COALESCE(EXCLUDED.upload_date, videos.upload_date),
                tags = COALESCE(EXCLUDED.tags, videos.tags),
                hashtags = COALESCE(EXCLUDED.hashtags, videos.hashtags),
                description = COALESCE(EXCLUDED.description, videos.description),
                audio_path = COALESCE(EXCLUDED.audio_path, videos.audio_path),
                audio_size_bytes = COALESCE(EXCLUDED.audio_size_bytes, videos.audio_size_bytes),
                view_count = COALESCE(EXCLUDED.view_count, videos.view_count),
                like_count = COALESCE(EXCLUDED.like_count, videos.like_count),
                comment_count = COALESCE(EXCLUDED.comment_count, videos.comment_count),
                share_count = COALESCE(EXCLUDED.share_count, videos.share_count),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            url,
            platform,
            video_id,
            metadata.get('title'),
            metadata.get('author'),
            metadata.get('author_id'),
            metadata.get('author_follower_count'),
            metadata.get('duration'),
            metadata.get('upload_date'),
            metadata.get('tags', []),
            metadata.get('hashtags', []),
            metadata.get('description'),
            metadata.get('audio_path'),
            metadata.get('audio_size_bytes'),
            metadata.get('view_count'),
            metadata.get('like_count'),
            metadata.get('comment_count'),
            metadata.get('share_count'),
            metadata.get('collection_method', 'yt-dlp'),
            metadata.get('collection_version'),
            metadata.get('is_verified_creator'),
            metadata.get('research_notes')
        ))
        result = cur.fetchone()
        return result[0]


def get_video_by_url(url: str) -> Optional[Dict[str, Any]]:
    """Get a video by its URL."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE url = %s", (url,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_video_by_id(video_id: int) -> Optional[Dict[str, Any]]:
    """Get a video by its database ID."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_all_videos() -> List[Dict[str, Any]]:
    """Get all videos from the database."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos ORDER BY collected_at DESC")
        return [dict(row) for row in cur.fetchall()]


def get_all_videos_with_transcripts() -> List[Dict[str, Any]]:
    """Get all videos that have transcripts."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT v.*, t.id as transcript_id, t.text as transcript_text, t.word_count
            FROM videos v
            JOIN transcripts t ON v.id = t.video_id
            ORDER BY v.collected_at DESC
        """)
        return [dict(row) for row in cur.fetchall()]


def save_engagement_snapshot(video_id: int) -> int:
    """Save current engagement metrics as a historical snapshot."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO engagement_snapshots (video_id, view_count, like_count, comment_count, share_count)
            SELECT id, view_count, like_count, comment_count, share_count
            FROM videos WHERE id = %s
            RETURNING id
        """, (video_id,))
        result = cur.fetchone()
        return result[0] if result else None


# =============================================================================
# Transcript Operations
# =============================================================================

def insert_transcript(video_id: int, text: str, language: Optional[str] = None,
                     model_used: Optional[str] = None, segments: Optional[List] = None,
                     **kwargs) -> int:
    """
    Insert a transcript record with full provenance tracking.

    Additional kwargs: model_backend, model_compute_type, transcription_device,
                      language_confidence, audio_duration_seconds, processing_time_seconds
    """
    word_count = len(text.split()) if text else 0
    audio_duration = kwargs.get('audio_duration_seconds')
    wpm = (word_count / (audio_duration / 60)) if audio_duration and audio_duration > 0 else None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcripts (
                video_id, text, language, language_confidence,
                model_used, model_backend, model_compute_type, transcription_device,
                word_count, audio_duration_seconds, words_per_minute,
                segments, processing_time_seconds
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                text = EXCLUDED.text,
                language = EXCLUDED.language,
                language_confidence = EXCLUDED.language_confidence,
                model_used = EXCLUDED.model_used,
                model_backend = EXCLUDED.model_backend,
                model_compute_type = EXCLUDED.model_compute_type,
                transcription_device = EXCLUDED.transcription_device,
                word_count = EXCLUDED.word_count,
                audio_duration_seconds = EXCLUDED.audio_duration_seconds,
                words_per_minute = EXCLUDED.words_per_minute,
                segments = EXCLUDED.segments,
                processing_time_seconds = EXCLUDED.processing_time_seconds,
                transcribed_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            video_id,
            text,
            language,
            kwargs.get('language_confidence'),
            model_used,
            kwargs.get('model_backend'),
            kwargs.get('model_compute_type'),
            kwargs.get('transcription_device'),
            word_count,
            audio_duration,
            wpm,
            json.dumps(segments) if segments else None,
            kwargs.get('processing_time_seconds')
        ))
        result = cur.fetchone()
        return result[0]


def get_transcript(video_id: int) -> Optional[Dict[str, Any]]:
    """Get transcript for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM transcripts WHERE video_id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


# =============================================================================
# Symptom Operations
# =============================================================================

def insert_symptom(video_id: int, category: str, symptom: str,
                  confidence: float, context: Optional[str] = None,
                  **kwargs) -> int:
    """
    Insert a symptom record with enhanced tracking.

    Additional kwargs: severity, temporal_pattern, body_location, triggers,
                      is_personal_experience, extractor_model, extractor_provider
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO symptoms (
                video_id, category, symptom, confidence, context,
                severity, temporal_pattern, body_location, triggers,
                is_personal_experience, extractor_model, extractor_provider
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            video_id,
            category,
            symptom,
            confidence,
            context,
            kwargs.get('severity', 'unspecified'),
            kwargs.get('temporal_pattern', 'unspecified'),
            kwargs.get('body_location'),
            kwargs.get('triggers', []),
            kwargs.get('is_personal_experience', True),
            kwargs.get('extractor_model'),
            kwargs.get('extractor_provider')
        ))
        result = cur.fetchone()
        return result[0]


def get_symptoms_by_video(video_id: int) -> List[Dict[str, Any]]:
    """Get all symptoms for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM symptoms
            WHERE video_id = %s
            ORDER BY confidence DESC
        """, (video_id,))
        return [dict(row) for row in cur.fetchall()]


def insert_symptom_cooccurrence(symptom_a_id: int, symptom_b_id: int, video_id: int,
                                temporal_proximity_seconds: Optional[float] = None,
                                mentioned_together: bool = False) -> int:
    """Record that two symptoms co-occurred in the same video."""
    # Ensure consistent ordering (smaller ID first)
    if symptom_a_id > symptom_b_id:
        symptom_a_id, symptom_b_id = symptom_b_id, symptom_a_id

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO symptom_cooccurrence (symptom_a_id, symptom_b_id, video_id,
                                             temporal_proximity_seconds, mentioned_together)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symptom_a_id, symptom_b_id) DO UPDATE SET
                temporal_proximity_seconds = EXCLUDED.temporal_proximity_seconds,
                mentioned_together = EXCLUDED.mentioned_together
            RETURNING id
        """, (symptom_a_id, symptom_b_id, video_id, temporal_proximity_seconds, mentioned_together))
        result = cur.fetchone()
        return result[0]


# =============================================================================
# Processing Run Tracking
# =============================================================================

def start_processing_run(run_type: str, config_snapshot: Optional[Dict] = None,
                        notes: Optional[str] = None) -> int:
    """Start tracking a new processing run."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO processing_runs (run_type, config_snapshot, notes)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (run_type, json.dumps(config_snapshot) if config_snapshot else None, notes))
        return cur.fetchone()[0]


def complete_processing_run(run_id: int, videos_processed: int = 0,
                           transcripts_created: int = 0, symptoms_extracted: int = 0,
                           errors: Optional[List[Dict]] = None):
    """Mark a processing run as complete with statistics."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE processing_runs
            SET completed_at = CURRENT_TIMESTAMP,
                videos_processed = %s,
                transcripts_created = %s,
                symptoms_extracted = %s,
                errors = %s
            WHERE id = %s
        """, (videos_processed, transcripts_created, symptoms_extracted,
              json.dumps(errors) if errors else None, run_id))


# =============================================================================
# Annotations
# =============================================================================

def add_annotation(annotation_type: str, annotation_value: str,
                  video_id: Optional[int] = None, symptom_id: Optional[int] = None,
                  annotator: Optional[str] = None) -> int:
    """Add a research annotation to a video or symptom."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO annotations (video_id, symptom_id, annotation_type, annotation_value, annotator)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (video_id, symptom_id, annotation_type, annotation_value, annotator))
        return cur.fetchone()[0]


# =============================================================================
# Statistics and Analysis Queries
# =============================================================================

def get_symptom_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics about extracted symptoms."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Total counts
        cur.execute("SELECT COUNT(*) as count FROM videos")
        total_videos = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM transcripts")
        total_transcripts = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM symptoms")
        total_symptoms = cur.fetchone()['count']

        # Symptoms by category
        cur.execute("""
            SELECT category, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM symptoms
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = [dict(row) for row in cur.fetchall()]

        # Symptoms by severity
        cur.execute("""
            SELECT severity, COUNT(*) as count
            FROM symptoms
            WHERE severity IS NOT NULL
            GROUP BY severity
            ORDER BY count DESC
        """)
        by_severity = [dict(row) for row in cur.fetchall()]

        # Symptoms by temporal pattern
        cur.execute("""
            SELECT temporal_pattern, COUNT(*) as count
            FROM symptoms
            WHERE temporal_pattern IS NOT NULL
            GROUP BY temporal_pattern
            ORDER BY count DESC
        """)
        by_temporal = [dict(row) for row in cur.fetchall()]

        # Personal vs informational
        cur.execute("""
            SELECT is_personal_experience, COUNT(*) as count
            FROM symptoms
            GROUP BY is_personal_experience
        """)
        by_personal = [dict(row) for row in cur.fetchall()]

        # Videos with symptoms
        cur.execute("SELECT COUNT(DISTINCT video_id) as count FROM symptoms")
        videos_with_symptoms = cur.fetchone()['count']

        # Platform breakdown
        cur.execute("""
            SELECT platform, COUNT(*) as video_count,
                   SUM(view_count) as total_views,
                   AVG(duration) as avg_duration
            FROM videos
            GROUP BY platform
        """)
        by_platform = [dict(row) for row in cur.fetchall()]

        avg_symptoms = total_symptoms / videos_with_symptoms if videos_with_symptoms > 0 else 0

        return {
            'total_videos': total_videos,
            'total_transcripts': total_transcripts,
            'total_symptoms': total_symptoms,
            'videos_with_symptoms': videos_with_symptoms,
            'avg_symptoms_per_video': round(avg_symptoms, 2),
            'by_category': by_category,
            'by_severity': by_severity,
            'by_temporal_pattern': by_temporal,
            'by_personal_experience': by_personal,
            'by_platform': by_platform
        }


def get_cooccurrence_matrix(min_occurrences: int = 2) -> List[Dict[str, Any]]:
    """Get symptom co-occurrence data for network analysis."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                s1.symptom as symptom_a,
                s1.category as category_a,
                s2.symptom as symptom_b,
                s2.category as category_b,
                COUNT(*) as cooccurrence_count,
                AVG(sc.temporal_proximity_seconds) as avg_temporal_proximity
            FROM symptom_cooccurrence sc
            JOIN symptoms s1 ON sc.symptom_a_id = s1.id
            JOIN symptoms s2 ON sc.symptom_b_id = s2.id
            GROUP BY s1.symptom, s1.category, s2.symptom, s2.category
            HAVING COUNT(*) >= %s
            ORDER BY cooccurrence_count DESC
        """, (min_occurrences,))
        return [dict(row) for row in cur.fetchall()]


if __name__ == '__main__':
    print("Initializing database...")
    init_db()

    try:
        stats = get_symptom_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Videos: {stats['total_videos']}")
        print(f"  Transcripts: {stats['total_transcripts']}")
        print(f"  Symptoms: {stats['total_symptoms']}")
        if stats['by_platform']:
            print(f"\n  By Platform:")
            for p in stats['by_platform']:
                print(f"    {p['platform']}: {p['video_count']} videos")
    except Exception as e:
        print(f"Could not fetch statistics: {e}")
