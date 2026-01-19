"""
Database module for the TikTok Disorders Research Pipeline.
Handles PostgreSQL connections and CRUD operations for videos, transcripts, and symptoms.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
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
    """Initialize the database schema."""
    with get_connection() as conn:
        cur = conn.cursor()

        # Videos table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                author TEXT,
                duration INTEGER,
                upload_date DATE,
                tags TEXT[],
                audio_path TEXT,
                audio_size_bytes BIGINT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Transcripts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                language TEXT,
                model_used TEXT,
                segments JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id)
            )
        """)

        # Symptoms table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptoms (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                symptom TEXT NOT NULL,
                confidence REAL NOT NULL,
                context TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes for better query performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_url ON videos(url)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_platform ON videos(platform)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_video_id ON symptoms(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_category ON symptoms(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_confidence ON symptoms(confidence)")

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
        metadata: Dictionary containing title, author, duration, upload_date, tags, audio_path, audio_size_bytes

    Returns:
        Database ID of the inserted video
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO videos (url, platform, video_id, title, author, duration, upload_date, tags, audio_path, audio_size_bytes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                title = EXCLUDED.title,
                author = EXCLUDED.author,
                duration = EXCLUDED.duration,
                upload_date = EXCLUDED.upload_date,
                tags = EXCLUDED.tags,
                audio_path = EXCLUDED.audio_path,
                audio_size_bytes = EXCLUDED.audio_size_bytes
            RETURNING id
        """, (
            url,
            platform,
            video_id,
            metadata.get('title'),
            metadata.get('author'),
            metadata.get('duration'),
            metadata.get('upload_date'),
            metadata.get('tags', []),
            metadata.get('audio_path'),
            metadata.get('audio_size_bytes')
        ))
        result = cur.fetchone()
        return result[0]


def get_video_by_url(url: str) -> Optional[Dict[str, Any]]:
    """
    Get a video by its URL.

    Args:
        url: Video URL

    Returns:
        Video record as dictionary, or None if not found
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE url = %s", (url,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_video_by_id(video_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a video by its database ID.

    Args:
        video_id: Database ID

    Returns:
        Video record as dictionary, or None if not found
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_all_videos() -> List[Dict[str, Any]]:
    """Get all videos from the database."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos ORDER BY created_at DESC")
        return [dict(row) for row in cur.fetchall()]


def get_all_videos_with_transcripts() -> List[Dict[str, Any]]:
    """Get all videos that have transcripts."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT v.*, t.id as transcript_id, t.text as transcript_text
            FROM videos v
            JOIN transcripts t ON v.id = t.video_id
            ORDER BY v.created_at DESC
        """)
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Transcript Operations
# =============================================================================

def insert_transcript(video_id: int, text: str, language: Optional[str] = None,
                     model_used: Optional[str] = None, segments: Optional[List] = None) -> int:
    """
    Insert a transcript record.

    Args:
        video_id: Database ID of the video
        text: Transcript text
        language: Detected language
        model_used: Whisper model used
        segments: Optional list of segment dictionaries

    Returns:
        Database ID of the inserted transcript
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcripts (video_id, text, language, model_used, segments)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                text = EXCLUDED.text,
                language = EXCLUDED.language,
                model_used = EXCLUDED.model_used,
                segments = EXCLUDED.segments
            RETURNING id
        """, (
            video_id,
            text,
            language,
            model_used,
            json.dumps(segments) if segments else None
        ))
        result = cur.fetchone()
        return result[0]


def get_transcript(video_id: int) -> Optional[Dict[str, Any]]:
    """
    Get transcript for a video.

    Args:
        video_id: Database ID of the video

    Returns:
        Transcript record as dictionary, or None if not found
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM transcripts WHERE video_id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


# =============================================================================
# Symptom Operations
# =============================================================================

def insert_symptom(video_id: int, category: str, symptom: str,
                  confidence: float, context: Optional[str] = None) -> int:
    """
    Insert a symptom record.

    Args:
        video_id: Database ID of the video
        category: Symptom category
        symptom: Symptom description
        confidence: Confidence score (0.0-1.0)
        context: Optional context from transcript

    Returns:
        Database ID of the inserted symptom
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO symptoms (video_id, category, symptom, confidence, context)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (video_id, category, symptom, confidence, context))
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


def get_symptom_statistics() -> Dict[str, Any]:
    """
    Get statistics about extracted symptoms.

    Returns:
        Dictionary with various statistics
    """
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

        # Videos with symptoms
        cur.execute("""
            SELECT COUNT(DISTINCT video_id) as count FROM symptoms
        """)
        videos_with_symptoms = cur.fetchone()['count']

        # Average symptoms per video
        avg_symptoms = total_symptoms / videos_with_symptoms if videos_with_symptoms > 0 else 0

        return {
            'total_videos': total_videos,
            'total_transcripts': total_transcripts,
            'total_symptoms': total_symptoms,
            'videos_with_symptoms': videos_with_symptoms,
            'avg_symptoms_per_video': round(avg_symptoms, 2),
            'by_category': by_category
        }


if __name__ == '__main__':
    # Initialize database when run directly
    print("Initializing database...")
    init_db()

    # Show statistics
    try:
        stats = get_symptom_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Videos: {stats['total_videos']}")
        print(f"  Transcripts: {stats['total_transcripts']}")
        print(f"  Symptoms: {stats['total_symptoms']}")
    except Exception as e:
        print(f"Could not fetch statistics: {e}")
