# Quick Start Guide

This guide will help you get started with the EDS/MCAS/POTS Research Pipeline.

## Prerequisites

- **Python 3.8+**
- **PostgreSQL** database
- **FFmpeg** (for audio extraction)
- **CUDA-capable GPU** (RTX 4090 recommended for best performance)
- **Anthropic API key** (for Claude symptom extraction)

## Installation

### 1. Clone and set up the repository

```bash
cd tiktok_disorders
uv sync
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
- `ANTHROPIC_API_KEY` - Your Claude API key

Example:
```
DATABASE_URL=postgresql://username:password@localhost:5432/tiktok_disorders
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Initialize the database schema

```bash
uv run python scripts/init_db.py
```

## Basic Usage

### Option 1: Process a single video

```bash
uv run python scripts/run_pipeline.py --url "https://youtube.com/watch?v=VIDEO_ID" --tags EDS,POTS
```

### Option 2: Process multiple videos

1. Create a file with URLs (one per line):

```bash
cp urls.txt.example urls.txt
# Edit urls.txt and add your video URLs
```

2. Run the pipeline:

```bash
uv run python scripts/run_pipeline.py --file urls.txt --tags EDS,MCAS,POTS --analyze
```

The `--analyze` flag will automatically run clustering analysis after processing all videos.

### Option 3: Run stages individually

**Download videos:**
```bash
uv run python scripts/download.py --url "https://youtube.com/watch?v=VIDEO_ID"
```

**Transcribe audio:**
```bash
uv run python scripts/transcribe.py --video-id 1 --model large-v3
```

**Extract symptoms:**
```bash
uv run python scripts/extract_symptoms.py --video-id 1 --min-confidence 0.6
```

**Analyze patterns:**
```bash
uv run python scripts/analyze.py --cluster-method kmeans --viz-method umap
```

## View Statistics

```bash
# Basic stats
uv run python scripts/stats.py

# Detailed stats
uv run python scripts/stats.py --detailed

# Export to JSON
uv run python scripts/stats.py --export-json stats.json
```

## Configuration Tips

### For RTX 4090 (Recommended Settings)

The pipeline is optimized for your hardware! Use these settings for best performance:

- **Whisper model**: `large-v3` (highest quality transcription)
- **Parallel extraction**: Enabled (processes multiple videos simultaneously)
- **Clustering**: Can handle large datasets with advanced algorithms

```bash
uv run python scripts/run_pipeline.py \
  --file urls.txt \
  --whisper-model large-v3 \
  --min-confidence 0.6 \
  --analyze
```

### For Limited Resources

If running on a less powerful system:

- **Whisper model**: `base` or `small`
- **Parallel extraction**: Disable with `--no-parallel`

```bash
uv run python scripts/run_pipeline.py \
  --file urls.txt \
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

### API rate limits

If you hit Claude API rate limits:
- Reduce `--max-workers` in extraction
- Use `--no-parallel` flag
- Process videos in smaller batches

## Next Steps

1. **Collect videos**: Find relevant YouTube/TikTok videos about EDS, MCAS, or POTS
2. **Run the pipeline**: Process videos and extract symptoms
3. **Analyze patterns**: Use clustering to discover symptom patterns
4. **Iterate**: Adjust confidence thresholds and clustering parameters
5. **Export data**: Share CSV exports with your research team

## Getting Help

- Check the main [README.md](README.md) for architecture details
- See [Issue #1](https://github.com/leakydata/tiktok_disorders/issues/1) for implementation notes
- Review module docstrings for API details

Happy researching! 🔬
