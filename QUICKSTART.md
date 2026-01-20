# Quick Start Guide

This guide will help you get started with the EDS/MCAS/POTS Research Pipeline.

This pipeline was developed for the research study: **[Public Health Narratives, Self-Diagnosis, and Symptom Attribution on TikTok](https://osf.io/5y46c)**

## Prerequisites

- **Python 3.9+**
- **PostgreSQL** database
- **FFmpeg** (for audio extraction)
- **CUDA-capable GPU** (RTX 4090 recommended for best performance)
- **Anthropic API key** or **Ollama** (for symptom extraction)

## Installation

### 1. Clone and set up the repository

```bash
cd tiktok_disorders
uv sync

# Optional: UMAP visualization support
uv sync --extra umap

# Install Playwright browsers for TikTok discovery
uv run playwright install
```

### 2. Install system dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg postgresql
```

**macOS:**
```bash
brew install ffmpeg postgresql
```

**Windows:**
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Download and install [PostgreSQL](https://www.postgresql.org/download/windows/)

### 3. Set up the database

```bash
# Create the database
createdb tiktok_disorders

# Or on Windows via psql:
# psql -U postgres
# CREATE DATABASE tiktok_disorders;
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:
- `DATABASE_URL` - Your PostgreSQL connection string
- `ANTHROPIC_API_KEY` - Your Claude API key (if using Anthropic)
- `EXTRACTOR_PROVIDER` - `anthropic` or `ollama`
- `OLLAMA_MODEL` - Ollama model name (if using Ollama)

Example:
```
DATABASE_URL=postgresql://username:password@localhost:5432/tiktok_disorders
ANTHROPIC_API_KEY=sk-ant-...
EXTRACTOR_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_URL=http://localhost:11434
```

### 5. Initialize the database schema

```bash
uv run python scripts/init_db.py
```

## Basic Usage

### Option 1: Start with Known Users (Recommended)

```bash
# 1. Get all videos from a specific user
uv run python scripts/discover.py --user chronicillnesswarrior

# 2. Or get only recent videos (last 6 months)
uv run python scripts/discover.py --user chronicillnesswarrior --days 180

# 3. Run the pipeline on discovered URLs
uv run python pipeline.py run --urls-file urls.txt --tags EDS MCAS POTS
```

### Option 2: Expand Existing URL List

```bash
# 1. Start with some seed URLs in urls.txt
# 2. Expand to get ALL videos from those users (last year)
uv run python scripts/discover.py --expand-users urls.txt --days 365

# 3. Process everything
uv run python pipeline.py run --urls-file urls.txt --tags EDS POTS
```

### Option 3: Process a Single Video

```bash
uv run python pipeline.py run "https://tiktok.com/@user/video/123" --tags EDS
```

## Discovery Commands

Find new videos to analyze:

```bash
# From a user's profile (via video URL)
uv run python scripts/discover.py --url "https://tiktok.com/@user/video/123"

# From a username directly
uv run python scripts/discover.py --user chronicallychillandhot

# Expand all users in existing file
uv run python scripts/discover.py --expand-users urls.txt

# From hashtags (browser window will open - don't interact with it)
uv run python scripts/discover.py --hashtag EDS --hashtag POTS --max-videos 200

# With date filtering (last 6 months)
uv run python scripts/discover.py --user someuser --days 180

# With date range (all of 2024)
uv run python scripts/discover.py --user someuser --after 2024-01-01 --before 2025-01-01
```

**Note**: Hashtag discovery opens a browser window that scrolls through pages. Don't interact with it - let it run.

## Pipeline Commands

All pipeline operations use subcommands:

```bash
# Full pipeline (download + transcribe + extract)
uv run python pipeline.py run --urls-file urls.txt

# Download only
uv run python pipeline.py download --urls-file urls.txt

# Transcribe all untranscribed videos
uv run python pipeline.py transcribe --all

# Extract symptoms from all unprocessed transcripts
uv run python pipeline.py extract --all

# Run analysis
uv run python pipeline.py analyze

# Show statistics
uv run python pipeline.py stats --detailed
```

## Recovery Workflow

If the pipeline fails partway through:

```bash
# 1. Check what's incomplete
uv run python pipeline.py stats

# 2. Complete any missing transcriptions
uv run python pipeline.py transcribe --all

# 3. Extract symptoms from completed transcripts
uv run python pipeline.py extract --all

# 4. Run analysis
uv run python pipeline.py analyze
```

## Discovery Tips

### Date Filtering

Limit discovery to specific time periods:

```bash
# Last 30 days
uv run python scripts/discover.py --user someone --days 30

# Last year
uv run python scripts/discover.py --expand-users urls.txt --days 365

# Specific date range
uv run python scripts/discover.py --user someone --after 2024-01-01 --before 2024-07-01
```

### Crash Safety

- URLs are saved incrementally after each user
- If discovery is interrupted, just run the command again
- Default mode appends to existing file (won't lose data)
- Duplicates are automatically skipped

## Configuration Tips

### For RTX 4090 (Recommended Settings)

The pipeline is optimized for your hardware! Use these settings for best performance:

```bash
uv run python pipeline.py run \
  --urls-file urls.txt \
  --whisper-model large-v3 \
  --min-confidence 0.6 \
  --tags EDS MCAS POTS
```

### For Limited Resources

If running on a less powerful system:

```bash
uv run python pipeline.py run \
  --urls-file urls.txt \
  --whisper-model base \
  --no-parallel
```

## Output Files

The pipeline generates:

- **Audio files**: `data/audio/*.mp3`
- **Transcripts**: `data/transcripts/*.json`
- **Visualizations**: `data/visualizations/*.png`
- **CSV exports**: `data/exports/*.csv`
- **Results**: `pipeline_results.json`, `analysis_results.json`

## Troubleshooting

### Database connection errors

- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check your `DATABASE_URL` in `.env`
- Verify the database exists: `psql -l`

### CUDA/GPU errors

- Verify CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, Whisper will fall back to CPU (slower but still works)

### FFmpeg errors

- Verify FFmpeg is installed: `ffmpeg -version`
- Add FFmpeg to your system PATH

### TikTok discovery errors

- Install Playwright browsers: `uv run playwright install`
- Hashtag discovery opens a browser window - don't interact with it
- If a captcha appears, solve it manually (script waits 30 seconds)
- If getting timeouts, increase delays: `--min-delay 5 --max-delay 10`

### API rate limits

If you hit Claude API rate limits:
- Use `--no-parallel` flag
- Process videos in smaller batches
- Switch to Ollama for local extraction

## Next Steps

1. **Discover videos**: Use user discovery to build your dataset
2. **Run the pipeline**: Process videos and extract symptoms
3. **Analyze patterns**: Use clustering to discover symptom patterns
4. **Iterate**: Adjust confidence thresholds and clustering parameters
5. **Export data**: Share CSV exports with your research team

## Getting Help

- Check the main [README.md](README.md) for full documentation
- Review module docstrings for API details
