"""
Database module for managing PostgreSQL connections and schema.
Stores video metadata, audio files, transcripts, and extracted symptoms.
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
from contextlib import contextmanager

from config import DATABASE_URL


@contextmanager
def get_connection():
    """Context manager for database connections."""
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
    """Initialize database schema."""
    with get_connection() as conn:
        cur = conn.cursor()

        # Videos table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                platform TEXT NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                author TEXT,
                duration INTEGER,
                upload_date DATE,
                tags TEXT[],
                downloaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                audio_path TEXT,
                audio_size_bytes BIGINT
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
                transcribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                segments JSONB
            )
        """)

        # Symptoms table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptoms (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                symptom TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                context TEXT,
                timestamp_start INTEGER,
                timestamp_end INTEGER,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Clusters table for analysis results
        cur.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id SERIAL PRIMARY KEY,
                analysis_name TEXT NOT NULL,
                cluster_id INTEGER NOT NULL,
                symptoms TEXT[],
                video_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB
            )
        """)

        # Create indexes for better query performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_video_id ON videos(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_tags ON videos USING GIN(tags)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_video_id ON symptoms(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_category ON symptoms(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_confidence ON symptoms(confidence)")

        print("✓ Database schema initialized successfully")


def insert_video(url: str, platform: str, video_id: str, metadata: Dict[str, Any]) -> int:
    """Insert a new video record and return its ID."""
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
        return cur.fetchone()[0]


def insert_transcript(video_id: int, text: str, language: str, model_used: str, segments: Optional[List] = None) -> int:
    """Insert a transcript record and return its ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        word_count = len(text.split()) if text else 0
        cur.execute("""
            INSERT INTO transcripts (video_id, text, language, model_used, word_count, segments)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            video_id,
            text,
            language,
            model_used,
            word_count,
            json.dumps(segments) if segments else None
        ))
        return cur.fetchone()[0]


def insert_symptom(video_id: int, category: str, symptom: str, confidence: float,
                   context: Optional[str] = None, timestamp_start: Optional[int] = None,
                   timestamp_end: Optional[int] = None) -> int:
    """Insert a symptom record and return its ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO symptoms (video_id, category, symptom, confidence, context, timestamp_start, timestamp_end)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (video_id, category, symptom, confidence, context, timestamp_start, timestamp_end))
        return cur.fetchone()[0]


def get_video_by_url(url: str) -> Optional[Dict[str, Any]]:
    """Retrieve a video record by URL."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE url = %s", (url,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_video_by_id(video_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a video record by database ID."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_transcript(video_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve the transcript for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM transcripts WHERE video_id = %s ORDER BY transcribed_at DESC LIMIT 1", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_symptoms(video_id: int, min_confidence: float = 0.0) -> List[Dict[str, Any]]:
    """Retrieve symptoms for a video, optionally filtered by confidence."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM symptoms
            WHERE video_id = %s AND confidence >= %s
            ORDER BY confidence DESC
        """, (video_id, min_confidence))
        return [dict(row) for row in cur.fetchall()]


def get_all_videos_with_transcripts() -> List[Dict[str, Any]]:
    """Retrieve all videos that have been transcribed."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT v.*, t.id as transcript_id, t.text as transcript_text
            FROM videos v
            JOIN transcripts t ON v.id = t.video_id
            ORDER BY v.downloaded_at DESC
        """)
        return [dict(row) for row in cur.fetchall()]


def get_symptom_statistics() -> Dict[str, Any]:
    """Get statistics about symptoms in the database."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Total counts
        cur.execute("SELECT COUNT(*) as total_videos FROM videos")
        total_videos = cur.fetchone()['total_videos']

        cur.execute("SELECT COUNT(*) as total_symptoms FROM symptoms")
        total_symptoms = cur.fetchone()['total_symptoms']

        # Top symptoms
        cur.execute("""
            SELECT symptom, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM symptoms
            GROUP BY symptom
            ORDER BY count DESC
            LIMIT 10
        """)
        top_symptoms = [dict(row) for row in cur.fetchall()]

        # Symptoms by category
        cur.execute("""
            SELECT category, COUNT(*) as count
            FROM symptoms
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = [dict(row) for row in cur.fetchall()]

        return {
            'total_videos': total_videos,
            'total_symptoms': total_symptoms,
            'top_symptoms': top_symptoms,
            'by_category': by_category
        }


if __name__ == '__main__':
    # Test database connection and initialization
    try:
        init_db()
        stats = get_symptom_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Total videos: {stats['total_videos']}")
        print(f"  Total symptoms: {stats['total_symptoms']}")
    except Exception as e:
        print(f"✗ Database error: {e}")
