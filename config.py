"""
Configuration management for the TikTok Disorders Research Pipeline.
Loads settings from environment variables with sensible defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://localhost:5433/tiktok_disorders'
)

# API Keys
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Data Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
AUDIO_DIR = Path(os.getenv('AUDIO_DIR', DATA_DIR / 'audio'))
TRANSCRIPT_DIR = Path(os.getenv('TRANSCRIPT_DIR', DATA_DIR / 'transcripts'))
VISUALIZATION_DIR = Path(os.getenv('VISUALIZATION_DIR', DATA_DIR / 'visualizations'))

# Whisper Model Configuration
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']

# Analysis Configuration
MIN_CONFIDENCE_SCORE = float(os.getenv('MIN_CONFIDENCE_SCORE', '0.6'))
CLUSTER_COUNT = int(os.getenv('CLUSTER_COUNT', '5'))

# Disorder Tags
DISORDER_TAGS = ['EDS', 'MCAS', 'POTS', 'hEDS', 'Dysautonomia']


def ensure_directories():
    """Create necessary directories if they don't exist."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)


def validate_config():
    """Validate required configuration is present."""
    errors = []

    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is not set")

    if WHISPER_MODEL not in WHISPER_MODELS:
        errors.append(f"WHISPER_MODEL must be one of {WHISPER_MODELS}")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    ensure_directories()


if __name__ == '__main__':
    # Test configuration
    try:
        validate_config()
        print("✓ Configuration is valid")
        print(f"  Database: {DATABASE_URL}")
        print(f"  Audio directory: {AUDIO_DIR}")
        print(f"  Transcript directory: {TRANSCRIPT_DIR}")
        print(f"  Visualization directory: {VISUALIZATION_DIR}")
        print(f"  Whisper model: {WHISPER_MODEL}")
    except ValueError as e:
        print(f"✗ Configuration error:\n{e}")
