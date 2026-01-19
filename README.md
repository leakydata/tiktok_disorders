# EDS/MCAS/POTS Research Pipeline

A proof-of-concept research pipeline to collect and analyze videos about Ehlers-Danlos Syndrome (EDS), Mast Cell Activation Syndrome (MCAS), and Postural Orthostatic Tachycardia Syndrome (POTS).

## Project Overview

This pipeline enables researchers to:
- Download videos from various platforms (YouTube, TikTok, etc.)
- Extract and transcribe audio content
- Use AI to identify and categorize symptoms
- Perform cluster analysis to discover patterns
- Generate visualizations and export data

## Architecture

The project consists of seven main modules:

1. **database.py** - PostgreSQL database management for storing video metadata, audio files, transcripts, and symptoms
2. **downloader.py** - Video/audio download using yt-dlp
3. **transcriber.py** - Audio-to-text transcription using OpenAI Whisper
4. **extractor.py** - Symptom extraction using Claude API
5. **analyzer.py** - K-means clustering and visualization with scikit-learn
6. **pipeline.py** - Complete workflow orchestration
7. **scripts/** - Command-line interfaces for each stage

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials and API keys
```

## Database Setup

```bash
# Create PostgreSQL database
createdb tiktok_disorders

# Initialize schema
python -c "from database import init_db; init_db()"
```

## Usage

### Download videos
```bash
python scripts/download.py --url "https://youtube.com/watch?v=..." --tags "EDS,POTS"
```

### Transcribe audio
```bash
python scripts/transcribe.py --video-id <id> --model base
```

### Extract symptoms
```bash
python scripts/extract_symptoms.py --video-id <id>
```

### Analyze patterns
```bash
python scripts/analyze.py --min-videos 10
```

### Run complete pipeline
```bash
python scripts/run_pipeline.py --urls urls.txt
```

## Configuration

Configuration is managed through environment variables in `.env`:
- `DATABASE_URL` - PostgreSQL connection string
- `ANTHROPIC_API_KEY` - Claude API key for symptom extraction
- `AUDIO_DIR` - Directory for storing audio files (default: ./data/audio)
- `TRANSCRIPT_DIR` - Directory for storing transcripts (default: ./data/transcripts)

## Output

The pipeline generates:
- Database records with video metadata and extracted symptoms
- MP3 audio files
- Transcripts (text and JSON format)
- Cluster visualizations (PNG)
- Correlation plots
- CSV exports with symptoms by category and confidence scores

## Project Status

See [Issue #1](https://github.com/leakydata/tiktok_disorders/issues/1) for the complete implementation plan.

## License

Research use only.
