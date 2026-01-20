# Quick Start Guide

This guide will help you get started with the EDS/MCAS/POTS Research Pipeline.

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

### Option 1: Discover and Process Videos (Recommended)

```bash
# 1. Find videos from hashtags
uv run python scripts/discover.py --hashtag EDS --hashtag POTS --max-videos 100 --output urls.txt

# 2. Expand to get all videos from discovered users
uv run python scripts/discover.py --expand-users urls.txt --append

# 3. Run the full pipeline
uv run python pipeline.py run --urls-file urls.txt --tags EDS POTS
```

### Option 2: Process a Single Video

```bash
uv run python pipeline.py run "https://tiktok.com/@user/video/123" --tags EDS
```

### Option 3: Process from a URL File

1. Create a file with URLs (one per line):

```bash
cp urls.txt.example urls.txt
# Edit urls.txt and add your video URLs
```

2. Run the pipeline:

```bash
uv run python pipeline.py run --urls-file urls.txt --tags EDS MCAS POTS
```

## Discovery Commands

Find new videos to analyze:

```bash
# From a user's profile (via video URL)
uv run python scripts/discover.py --url "https://tiktok.com/@user/video/123"

# From a username directly
uv run python scripts/discover.py --user chronicallychillandhot

# From hashtags
uv run python scripts/discover.py --hashtag ehlersdanlos --hashtag chronicillness

# From keyword search
uv run python scripts/discover.py --search "ehlers danlos syndrome"

# Expand all users in existing file
uv run python scripts/discover.py --expand-users urls.txt --append
```

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
- **CSV exports**: `data/visualizations/*.csv`
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
- If getting 403 errors, increase delays or try later

### API rate limits

If you hit Claude API rate limits:
- Use `--no-parallel` flag
- Process videos in smaller batches
- Switch to Ollama for local extraction

## Next Steps

1. **Discover videos**: Use hashtag and user discovery to build your dataset
2. **Run the pipeline**: Process videos and extract symptoms
3. **Analyze patterns**: Use clustering to discover symptom patterns
4. **Iterate**: Adjust confidence thresholds and clustering parameters
5. **Export data**: Share CSV exports with your research team

## Getting Help

- Check the main [README.md](README.md) for full documentation
- Review module docstrings for API details
