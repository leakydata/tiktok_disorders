# EDS/MCAS/POTS Research Pipeline

A research-grade pipeline to collect and analyze TikTok/YouTube videos about Ehlers-Danlos Syndrome (EDS), Mast Cell Activation Syndrome (MCAS), and Postural Orthostatic Tachycardia Syndrome (POTS).

## Features

- **Video Discovery** - Find ALL videos from user profiles with date filtering
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
- **Crash-Safe Discovery** - URLs saved incrementally to survive interruptions

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

## Video Discovery

The discovery script finds TikTok videos and saves them to `urls.txt` for processing.

### Basic Discovery

```bash
# Get ALL videos from a specific user
uv run python scripts/discover.py --user chronicallychillandhot

# Get videos from a video URL (extracts username automatically)
uv run python scripts/discover.py --url "https://tiktok.com/@user/video/123"

# Expand ALL users from your existing URL file (gets complete profiles)
uv run python scripts/discover.py --expand-users urls.txt
```

### Date Filtering

Limit discovery to videos from a specific time period:

```bash
# Only videos from the last 30 days
uv run python scripts/discover.py --user someuser --days 30

# Only videos from the last 6 months
uv run python scripts/discover.py --expand-users urls.txt --days 180

# Only videos from the last year
uv run python scripts/discover.py --expand-users urls.txt --days 365

# Only videos after a specific date (YYYY-MM-DD or YYYYMMDD)
uv run python scripts/discover.py --user someuser --after 2024-01-01

# Only videos before a specific date
uv run python scripts/discover.py --user someuser --before 2025-01-01

# Date range (e.g., all of 2024)
uv run python scripts/discover.py --user someuser --after 2024-01-01 --before 2025-01-01
```

### Output Options

```bash
# Default: appends to urls.txt (safe, won't lose existing URLs)
uv run python scripts/discover.py --user someuser

# Write to a different file
uv run python scripts/discover.py --user someuser --output my_urls.txt

# Overwrite instead of append (use with caution!)
uv run python scripts/discover.py --user someuser --overwrite
```

### Discovery Options Reference

| Option | Description |
|--------|-------------|
| `--user USERNAME` | Get all videos from a TikTok user |
| `--url URL` | Extract username from URL and get all their videos |
| `--expand-users FILE` | Get all videos from every user in the URL file |
| `--hashtag TAG` | Search for videos with hashtag (limited by TikTok) |
| `--search QUERY` | Search for videos by keyword (limited by TikTok) |
| `--days N` | Only include videos from the last N days |
| `--after DATE` | Only include videos after this date |
| `--before DATE` | Only include videos before this date |
| `--max-videos N` | Maximum videos per user (default: unlimited) |
| `--output FILE` | Output file (default: urls.txt) |
| `--overwrite` | Overwrite output file instead of appending |
| `--min-delay SEC` | Minimum delay between requests (default: 2.0) |
| `--max-delay SEC` | Maximum delay between requests (default: 5.0) |

### Notes on Discovery

- **Append is default**: New URLs are added to your existing file
- **Crash-safe**: URLs are saved after each user, so interruptions don't lose progress
- **Deduplication**: Duplicate URLs are automatically skipped
- **User profiles work best**: Hashtag/search discovery is often blocked by TikTok
- **Rate limiting**: Built-in delays prevent IP bans

## Running the Pipeline

### Full Pipeline

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
| `scripts/discover.py` | Find TikTok videos from users/hashtags |
| `scripts/init_db.py` | Initialize database schema |

## Example Workflows

### Complete Research Workflow

```bash
# 1. Start with a few seed videos you found manually
# Add them to urls.txt

# 2. Expand to get ALL videos from those users (last year only)
uv run python scripts/discover.py --expand-users urls.txt --days 365

# 3. Process everything through the pipeline
uv run python pipeline.py run --urls-file urls.txt --tags EDS MCAS POTS

# 4. Analyze the results
uv run python pipeline.py analyze
uv run python reports.py --export-csv
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

### Adding a New User to Study

```bash
# Get all their videos from the last 6 months
uv run python scripts/discover.py --user newusername --days 180

# Process the new URLs
uv run python pipeline.py run --urls-file urls.txt --tags EDS
```

### Studying Recent Content Only

```bash
# Get only videos from the last 30 days for all your users
uv run python scripts/discover.py --expand-users urls.txt --days 30 --output recent_urls.txt --overwrite

# Process just the recent content
uv run python pipeline.py run --urls-file recent_urls.txt
```

## Research Hashtags

Suggested hashtags for chronic illness research (note: hashtag discovery may be limited by TikTok):

- **EDS**: `#ehlersdanlos`, `#EDS`, `#hypermobility`, `#zebra`
- **POTS**: `#POTS`, `#dysautonomia`, `#posturaltachycardia`
- **MCAS**: `#MCAS`, `#mastcellactivation`, `#histamineintolerance`
- **CIRS**: `#CIRS`, `#moldillness`, `#biotoxin`
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
- `videos` - Video metadata, engagement metrics, author info
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
- **Visualizations** - Cluster plots in `data/visualizations/`
- **CSV Exports** - Symptom data for external analysis in `data/exports/`

## Example Output

```
============================================================
TikTok Video Discovery
============================================================
Output: urls.txt (append mode)
Date filter: after 20240101
URLs are saved incrementally after each source (crash-safe)

[1/85] @chronicallychillandhot
  Fetching videos for @chronicallychillandhot after 20240101 (via yt-dlp)...
  Found 127 videos for @chronicallychillandhot
    -> Saved 127 new URLs to urls.txt

[2/85] @ehlers_danlos_life
  Fetching videos for @ehlers_danlos_life after 20240101 (via yt-dlp)...
  Found 89 videos for @ehlers_danlos_life
    -> Saved 89 new URLs to urls.txt
...
```

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

**Hashtag/search discovery returns 0 videos:**
- This is normal - TikTok blocks automated hashtag scraping
- Use `--user` or `--expand-users` instead (these work reliably)

**403 errors or timeouts:**
1. Install Playwright browsers: `uv run playwright install`
2. Increase delays: `--min-delay 5 --max-delay 10`
3. Try again later (TikTok rate limits)

**Discovery interrupted:**
- Don't worry! URLs are saved after each user
- Just run the command again - it will skip already-saved URLs

### TikTok Impersonation Warning
The warning about impersonation is normal - curl_cffi is included to handle this.

## URL File Format

When using `--urls-file`, the file supports:

```text
# Full-line comments start with # or //
// This is also a comment

# @username - 2024-01-20 15:30:00
https://www.tiktok.com/@user1/video/123
https://www.tiktok.com/@user1/video/456

# @another_user - 2024-01-20 15:31:00
https://www.tiktok.com/@user2/video/789

# Blank lines are ignored
```

- One URL per line
- Comments with `#` or `//` (full-line or inline)
- Blank lines ignored
- Whitespace trimmed
- Duplicates automatically removed by pipeline

## License

Research use only.
