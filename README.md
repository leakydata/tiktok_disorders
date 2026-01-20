# EDS/MCAS/POTS Research Pipeline

A research-grade pipeline to collect and analyze TikTok/YouTube videos about Ehlers-Danlos Syndrome (EDS), Mast Cell Activation Syndrome (MCAS), and Postural Orthostatic Tachycardia Syndrome (POTS).

## Features

- **Video Download** - Download from TikTok, YouTube, and other platforms via yt-dlp
- **GPU-Accelerated Transcription** - Whisper large-v3 on CUDA (optimized for RTX 4090)
- **AI-Powered Extraction** - Extract symptoms, diagnoses, and treatments using Claude or Ollama
- **Diagnosis Concordance** - Compare reported symptoms against expected symptoms for claimed conditions
- **Comorbidity Tracking** - Track which conditions appear together
- **Treatment Analysis** - Track medications, supplements, therapies with effectiveness ratings
- **Cluster Analysis** - K-means/DBSCAN clustering with silhouette validation
- **Resumable Runs** - Progress tracking allows interrupted runs to be resumed
- **Duplicate Detection** - Prevents downloading the same video twice

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
python database.py
```

## Usage

### Process Videos

```bash
# Process a single video
python pipeline.py "https://www.tiktok.com/@user/video/123"

# Process multiple videos
python pipeline.py "url1" "url2" "url3"

# Process from a file (one URL per line)
python pipeline.py --urls-file urls.txt

# Combine file and CLI URLs (deduplicates automatically)
python pipeline.py --urls-file urls.txt "https://extra-url.com/video"

# Use Ollama instead of Claude for extraction
python pipeline.py --provider ollama --model llama3 "url"
```

### Check Status and Statistics

```bash
# Show database statistics
python pipeline.py --stats

# Show status of latest run
python pipeline.py --status

# Resume an interrupted run
python pipeline.py --resume <run_id>
```

### Run Analysis

```bash
# Cluster analysis on collected data
python pipeline.py --analyze
```

### Generate Reports

```bash
# Full analysis report (JSON + summary)
python reports.py

# Export all data to CSV for Excel/R/Python
python reports.py --export-csv

# Individual reports
python reports.py --diagnoses    # Diagnosis breakdown with concordance
python reports.py --symptoms     # Symptom analysis by category/severity
python reports.py --treatments   # Treatment effectiveness analysis
python reports.py --creators     # Analysis by video creator
```

Reports are saved to `data/reports/` and CSV exports to `data/exports/`.

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
