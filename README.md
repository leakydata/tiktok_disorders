# EDS/MCAS/POTS Research Pipeline

A research-grade pipeline to collect and analyze TikTok/YouTube videos about Ehlers-Danlos Syndrome (EDS), Mast Cell Activation Syndrome (MCAS), and Postural Orthostatic Tachycardia Syndrome (POTS).

## Features

- **Video Discovery** - Find videos from user profiles, hashtags, and keyword searches
- **Video Download** - Download from TikTok, YouTube, and other platforms via yt-dlp
- **GPU-Accelerated Transcription** - Whisper large-v3 on CUDA (optimized for RTX 4090)
- **AI-Powered Extraction** - Extract symptoms, diagnoses, and treatments using Claude or Ollama
- **Diagnosis Concordance** - Compare reported symptoms against expected symptoms for claimed conditions
- **Comorbidity Tracking** - Track which conditions appear together
- **Treatment Analysis** - Track medications, supplements, therapies with effectiveness ratings
- **Cluster Analysis** - K-means/DBSCAN clustering with silhouette validation
- **Resumable Runs** - Progress tracking allows interrupted runs to be resumed
- **Duplicate Detection** - Prevents downloading the same video twice
- **Granular Recovery** - Run individual pipeline stages to recover from failures

## Requirements

- Python 3.9+
- PostgreSQL
- FFmpeg (required for audio extraction)
- CUDA-capable GPU (optional, for fast transcription)

## Installation

```bash
# Install dependencies with uv
uv sync

# For GPU acceleration (RTX 4090, CUDA 12.1)
uv sync --group cuda

# Optional: UMAP for dimensionality reduction
uv sync --extra umap

# Install Playwright browsers for TikTok discovery (one-time)
uv run playwright install

# Set up environment
cp .env.example .env
# Edit .env with your credentials
```

### FFmpeg Installation

FFmpeg is required for extracting audio from videos:

**Windows:**
```bash
# Using winget
winget install ffmpeg

# Or using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add to PATH
```

**Linux:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

## Database Setup

```bash
# Create PostgreSQL database
createdb tiktok_disorders

# Initialize schema (creates all tables)
uv run python scripts/init_db.py
```

## Usage

The pipeline uses subcommands for different operations:

### Discover New Videos

Find videos from TikTok users, hashtags, and searches:

```bash
# Get all videos from a user (from a video URL)
uv run python scripts/discover.py --url "https://tiktok.com/@user/video/123"

# Get all videos from a username directly
uv run python scripts/discover.py --user chronicallychillandhot

# Expand all users from an existing URL file
uv run python scripts/discover.py --expand-users urls.txt --append

# Find videos by hashtag
uv run python scripts/discover.py --hashtag EDS --hashtag POTS --max-videos 200

# Search by keyword
uv run python scripts/discover.py --search "ehlers danlos syndrome" --max-videos 100

# Combine methods
uv run python scripts/discover.py --hashtag chronicillness --search "hypermobility" --append
```

### Run Full Pipeline

Process videos through all stages (download, transcribe, extract):

```bash
# Process from a file of URLs
uv run python pipeline.py run --urls-file urls.txt --tags EDS MCAS POTS

# Process specific URLs
uv run python pipeline.py run "https://tiktok.com/@user/video/123" --tags EDS

# Use Ollama instead of Claude for extraction
uv run python pipeline.py run --urls-file urls.txt --provider ollama --model llama3

# Resume an interrupted run
uv run python pipeline.py run --resume 5
```

### Granular Operations (Recovery)

Run individual stages when you need to recover from failures:

```bash
# Download only
uv run python pipeline.py download --urls-file urls.txt
uv run python pipeline.py download --url "https://tiktok.com/@user/video/123"

# Transcribe all untranscribed videos
uv run python pipeline.py transcribe --all

# Transcribe a specific video
uv run python pipeline.py transcribe --video-id 42

# Extract symptoms from all unprocessed transcripts
uv run python pipeline.py extract --all

# Re-extract with different settings
uv run python pipeline.py extract --all --min-confidence 0.8 --provider ollama
```

### Analysis and Statistics

```bash
# Show database statistics
uv run python pipeline.py stats
uv run python pipeline.py stats --detailed

# Run clustering analysis
uv run python pipeline.py analyze
uv run python pipeline.py analyze --cluster-method dbscan --viz-method tsne
```

### Generate Reports

```bash
# Full analysis report (JSON + summary)
uv run python reports.py

# Export all data to CSV for Excel/R/Python
uv run python reports.py --export-csv

# Individual reports
uv run python reports.py --diagnoses    # Diagnosis breakdown with concordance
uv run python reports.py --symptoms     # Symptom analysis by category/severity
uv run python reports.py --treatments   # Treatment effectiveness analysis
uv run python reports.py --creators     # Analysis by video creator
```

Reports are saved to `data/reports/` and CSV exports to `data/exports/`.

## Command Reference

| Command | Description |
|---------|-------------|
| `pipeline.py run` | Full pipeline (download, transcribe, extract) |
| `pipeline.py download` | Download videos only |
| `pipeline.py transcribe` | Transcribe audio only |
| `pipeline.py extract` | Extract symptoms only |
| `pipeline.py analyze` | Run clustering and visualization |
| `pipeline.py stats` | Show database statistics |
| `pipeline.py discover` | Find new TikTok videos |
| `scripts/discover.py` | Standalone discovery script |
| `scripts/init_db.py` | Initialize database schema |

## Example Workflows

### Initial Dataset Collection

```bash
# 1. Find videos about EDS from popular hashtags
uv run python scripts/discover.py --hashtag ehlersdanlos --max-videos 500 --output urls.txt
uv run python scripts/discover.py --hashtag hypermobility --max-videos 200 --append

# 2. Expand to get complete profiles for all discovered users
uv run python scripts/discover.py --expand-users urls.txt --append

# 3. Process everything
uv run python pipeline.py run --urls-file urls.txt --tags EDS
```

### Recovery After Failure

```bash
# Check what's missing
uv run python pipeline.py stats

# Finish any incomplete transcriptions
uv run python pipeline.py transcribe --all

# Extract symptoms from transcripts
uv run python pipeline.py extract --all

# Re-run analysis
uv run python pipeline.py analyze
```

### Adding a New Condition to Study

```bash
# Search for MCAS content
uv run python scripts/discover.py --hashtag MCAS --hashtag mastcellactivation --max-videos 300 --output mcas_urls.txt

# Process new videos
uv run python pipeline.py run --urls-file mcas_urls.txt --tags MCAS
```

## Research Hashtags

Suggested hashtags for chronic illness research:

- **EDS**: `#ehlersdanlos`, `#EDS`, `#hypermobility`, `#zebra`
- **POTS**: `#POTS`, `#dysautonomia`, `#posturaltachycardia`
- **MCAS**: `#MCAS`, `#mastcellactivation`, `#histamineintolerance`
- **General**: `#chronicillness`, `#spoonie`, `#invisibleillness`, `#chronicpain`

## Pipeline Stages

For each video, the pipeline:

1. **Download** - Extract audio from video (with duplicate detection)
2. **Transcribe** - Convert audio to text using Whisper (GPU-accelerated)
3. **Quality Assessment** - Score transcript clarity, completeness, medical term density
4. **Extract Symptoms** - Identify symptoms with severity, temporal patterns, triggers
5. **Extract Diagnoses** - Identify claimed conditions (EDS, MCAS, POTS, etc.)
6. **Extract Treatments** - Identify medications, supplements, therapies
7. **Concordance Analysis** - Compare reported symptoms vs expected symptoms
8. **Comorbidity Tracking** - Track condition co-occurrence

## Database Schema

### Core Tables
- `videos` - Video metadata, engagement metrics
- `transcripts` - Transcribed text with model provenance
- `symptoms` - Extracted symptoms with severity, temporal patterns
- `claimed_diagnoses` - Conditions the speaker claims to have
- `treatments` - Medications, supplements, therapies mentioned

### Analysis Tables
- `expected_symptoms` - Medical reference data for each condition
- `symptom_concordance` - How well reported symptoms match expected
- `comorbidity_pairs` - Which conditions appear together
- `transcript_quality` - Quality metrics for each transcript

### Progress Tables
- `processing_runs` - Track batch processing runs
- `pipeline_progress` - Per-URL progress for resumable runs

## Configuration

Environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/tiktok_disorders

# Extraction (choose one)
EXTRACTOR_PROVIDER=anthropic  # or 'ollama'
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Or for local extraction
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Transcription
WHISPER_MODEL=large-v3
TRANSCRIBER_BACKEND=faster-whisper
WHISPER_COMPUTE_TYPE=auto  # float16 for GPU, int8 for CPU

# Extraction thresholds
MIN_CONFIDENCE_SCORE=0.6
```

## Output

The pipeline generates:

- **Database Records** - All extracted data with full provenance
- **Audio Files** - MP3 files in `data/audio/`
- **Transcripts** - JSON files in `data/transcripts/`
- **Visualizations** - Cluster plots in `data/analysis/`
- **CSV Exports** - Symptom data for external analysis

## Example Output

```
# PIPELINE COMPLETE (Run ID: 5)
################################################################################
  Total time: 45.2 seconds (0.8 minutes)
  Success rate: 3/3 (100.0%)
  Videos downloaded: 3
  Videos transcribed: 3
  Total symptoms extracted: 24
  Total diagnoses extracted: 5
  Total treatments extracted: 8

  To resume this run if interrupted: --resume 5
################################################################################
```

## Troubleshooting

### FFmpeg Errors
If you see `unable to obtain file audio codec with ffprobe`:
1. Ensure FFmpeg is installed: `ffmpeg -version`
2. Ensure it's in your PATH
3. Restart your terminal after installation

### CUDA Not Available
If transcription runs on CPU:
```bash
# Reinstall with CUDA support
rm -rf .venv
uv sync --group cuda
```

### TikTok Discovery Issues
If discovery fails with 403 errors:
1. Install Playwright browsers: `uv run playwright install`
2. Increase delays: `--min-delay 5 --max-delay 10`
3. Try again later (TikTok rate limits)

### TikTok Impersonation Warning
The warning about impersonation is normal - curl_cffi is included to handle this.

## URL File Format

When using `--urls-file`, the file supports:

```text
# Full-line comments start with # or //
// This is also a comment

https://www.tiktok.com/@user1/video/123
https://www.tiktok.com/@user2/video/456  # inline comments work too
https://youtube.com/watch?v=abc  // this style too

# Blank lines are ignored
https://www.tiktok.com/@user3/video/789
```

- One URL per line
- Comments with `#` or `//` (full-line or inline)
- Blank lines ignored
- Whitespace trimmed
- Duplicates automatically removed

## License

Research use only.
