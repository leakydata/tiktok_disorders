# EDS/MCAS/POTS Research Pipeline

A research-grade pipeline to collect and analyze TikTok/YouTube videos about Ehlers-Danlos Syndrome (EDS), Mast Cell Activation Syndrome (MCAS), and Postural Orthostatic Tachycardia Syndrome (POTS).

## Research Project

This pipeline was developed for the research study:

**[Public Health Narratives, Self-Diagnosis, and Symptom Attribution on TikTok: A Narrative-Based Observational Study](https://osf.io/5y46c)**

### Research Goals

The study examines how chronic illness narratives spread on social media, with a focus on:
- Self-diagnosis patterns and symptom attribution
- Concordance between reported symptoms and clinical diagnostic criteria
- Comorbidity claims and treatment discussions
- The role of social media in shaping public health narratives

### STRAIN Framework Validation

This pipeline is designed to collect data for validating the **STRAIN (Stress-Activated Inflammatory Neuro-dysregulation)** framework, which proposes that many individuals reporting multisystem symptoms resembling EDS, MCAS, POTS, and CIRS may exhibit a phenotypic pattern characterized by:

- **Stress-reactivity**: Symptom flares linked to psychological stressors
- **Biomarker-silent inflammation**: Symptoms without sustained inflammatory markers
- **Multisystem migration**: Symptoms shifting between organ systems
- **Social/narrative context**: Symptom onset or framing after exposure to illness narratives
- **Diagnostic seeking**: Long journeys, negative testing, multiple doctors

The pipeline captures narrative elements from TikTok content to analyze:
- Self-diagnosis vs. professional diagnosis rates
- Mentions of doctor dismissal or "medical gaslighting"
- Stress-trigger and symptom flare patterns
- Online community influence on symptom framing
- Concordance between claimed symptoms and diagnostic criteria

The full research protocol, preregistration, and materials are available on OSF: https://osf.io/5y46c

## Features

- **Video Discovery** - Find ALL videos from user profiles with date filtering
- **Video Download** - Download from TikTok, YouTube, and other platforms via yt-dlp
- **GPU-Accelerated Transcription** - Whisper large-v3 on CUDA (optimized for RTX 4090)
- **AI-Powered Extraction** - Extract symptoms, diagnoses, treatments, and narrative elements using Claude or Ollama
- **STRAIN Framework Support** - Captures self-diagnosis patterns, doctor dismissal mentions, stress triggers, and social media influence
- **Creator Tier Analysis** - Automatically categorizes creators by influence (nano to mega based on follower count)
- **Diagnosis Concordance** - Compare reported symptoms against expected symptoms for claimed conditions
- **Comorbidity Tracking** - Track which conditions appear together
- **Treatment Analysis** - Track medications, supplements, therapies with effectiveness ratings
- **Cluster Analysis** - K-means/DBSCAN clustering with silhouette validation
- **Resumable Runs** - Progress tracking allows interrupted runs to be resumed
- **Duplicate Detection** - Prevents downloading the same video twice
- **Idempotent Processing** - Safe to re-run; skips already downloaded, transcribed, and extracted videos
- **URL Progress Tracking** - Successful URLs moved to `urls_processed.txt`, failed to `urls_failed.txt`
- **Song Lyrics Detection** - Automatically detects and skips song lyrics to avoid wasted extraction
- **Granular Recovery** - Run individual pipeline stages to recover from failures
- **Crash-Safe Discovery** - URLs saved incrementally to survive interruptions
- **Organized File Storage** - Audio and transcripts saved in username-based subfolders

## Requirements

- Python 3.9+
- PostgreSQL
- FFmpeg (required for audio extraction)
- CUDA-capable GPU (optional, for fast transcription)

## Installation

```bash
# Install dependencies with uv
uv sync

# Install Playwright browsers for TikTok discovery (one-time)
uv run playwright install

# Set up environment
cp .env.example .env
# Edit .env with your credentials
```

### GPU Setup (Recommended)

For fast transcription with NVIDIA GPUs:

```powershell
# Install PyTorch with CUDA 12.1 support
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Reinstall ctranslate2 to pick up CUDA
uv pip install --force-reinstall ctranslate2
```

This enables GPU-accelerated transcription (10-20x faster than CPU).

### Optional Dependencies

```bash
# UMAP for dimensionality reduction in analysis
uv sync --extra umap
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

# Reset database (WARNING: deletes all data!)
uv run python scripts/init_db.py --reset
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

# Find videos by hashtag (opens browser window, don't interact with it)
uv run python scripts/discover.py --hashtag EDS --hashtag POTS --max-videos 200
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
| `--hashtag TAG` | Search for videos with hashtag (uses browser) |
| `--search QUERY` | Search for videos by keyword |
| `--days N` | Only include videos from the last N days |
| `--after DATE` | Only include videos after this date |
| `--before DATE` | Only include videos before this date |
| `--max-videos N` | Maximum videos per user (default: unlimited) |
| `--output FILE` | Output file (default: urls.txt) |
| `--overwrite` | Overwrite output file instead of appending |
| `--headless` | Run browser in headless mode (faster but may get blocked) |
| `--no-browser` | Use API instead of browser for hashtags (often blocked) |
| `--min-delay SEC` | Minimum delay between requests (default: 2.0) |
| `--max-delay SEC` | Maximum delay between requests (default: 5.0) |

### Notes on Discovery

- **Append is default**: New URLs are added to your existing file
- **Browser is default for hashtags**: Opens a real browser window that scrolls through hashtag pages
- **Crash-safe**: URLs are saved after each user/hashtag, so interruptions don't lose progress
- **Deduplication**: Duplicate URLs are automatically skipped
- **Rate limiting**: Built-in delays prevent IP bans
- **Don't interact**: When the browser opens, let it scroll on its own

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
# Download only (with automatic URL tracking)
uv run python pipeline.py download --urls-file urls.txt
uv run python pipeline.py download --url "https://tiktok.com/@user/video/123"

# Re-process URLs that previously failed or were already processed
uv run python pipeline.py download --urls-file urls.txt --force

# Download all videos from a specific user (discovers + downloads)
uv run python pipeline.py download --user chronicallychillandhot
uv run python pipeline.py download --user user1 --user user2 --max-videos 50

# Transcribe all untranscribed videos
uv run python pipeline.py transcribe --all

# Transcribe a specific video
uv run python pipeline.py transcribe --video-id 42

# Transcribe only videos from a specific user
uv run python pipeline.py transcribe --user chronicallychillandhot

# Extract symptoms from all unprocessed transcripts
uv run python pipeline.py extract --all

# Extract only from a specific user's videos
uv run python pipeline.py extract --user chronicallychillandhot --provider deepseek

# Re-extract with different settings
uv run python pipeline.py extract --all --min-confidence 0.8 --provider ollama

# Skip song lyrics (default: >= 20% ratio) and short transcripts (default: < 20 words)
uv run python pipeline.py extract --all --max-song-ratio 0.3 --min-words 30
```

**Download command options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--url` | - | Single video URL to download |
| `--urls-file` | - | Path to text file with URLs |
| `--user` | - | TikTok username(s) to discover and download (can use multiple times) |
| `--tags` | - | Tags to associate with videos |
| `--max-videos` | all | Max videos per user |
| `--force` | - | Re-process URLs even if in urls_processed.txt or urls_failed.txt |

When using `--urls-file`, URLs are automatically tracked:
- Successful downloads are moved to `urls_processed.txt`
- Failed downloads are moved to `urls_failed.txt` with error message
- Re-running skips already processed/failed URLs (use `--force` to override)

**Extract command options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--user` | - | Filter by TikTok username(s) (can use multiple times) |
| `--max-song-ratio` | 0.2 | Skip videos with song_lyrics_ratio >= this |
| `--min-words` | 20 | Skip transcripts with fewer words (uses cleaned word count) |
| `--min-confidence` | 0.6 | Minimum confidence for symptoms |
| `--provider` | ollama | LLM provider (ollama, deepseek, or anthropic) |
| `--model` | gpt-oss:20b | LLM model name |
| `--force` | - | Re-extract all videos (clears previous extraction status) |
| `--thinking` | - | Enable Qwen3 `/think` mode for deeper reasoning (slower) |

**Note:** Videos are marked as "extracted" after processing (even if zero symptoms found). This prevents re-processing the same videos. Use `--force` to re-extract.

### Analysis and Statistics

```bash
# Show database statistics
uv run python pipeline.py stats

# Detailed stats with STRAIN indicators, creator tiers, and treatments
uv run python pipeline.py stats --detailed

# Run clustering analysis
uv run python pipeline.py analyze
uv run python pipeline.py analyze --cluster-method dbscan --viz-method tsne
```

The `--detailed` stats show:
- Diagnosis counts and concordance scores
- STRAIN framework indicators (self-diagnosis %, doctor dismissal mentions, etc.)
- Creator tier breakdown (how many videos from nano vs mega influencers)
- Top treatments and effectiveness ratings
- Comorbidity patterns

### User-Level Analysis (Social Contagion Research)

The `user_analysis.py` module provides comprehensive tools for studying individual creators' health narratives over time - essential for social contagion research.

```bash
# View comprehensive profile for a user
uv run python user_analysis.py profile @username

# View chronological timeline of a user's health narrative
uv run python user_analysis.py timeline @username

# Check symptom reporting consistency
uv run python user_analysis.py consistency @username

# Detect narrative inconsistencies (conflicting claims across videos)
uv run python user_analysis.py inconsistencies @username

# Overall concordance report (all users)
uv run python user_analysis.py concordance-report

# Find users with low concordance scores (potential social contagion)
uv run python user_analysis.py low-concordance --threshold 0.3

# Analyze diagnosis acquisition patterns
uv run python user_analysis.py diagnosis-patterns

# Summary of all users
uv run python user_analysis.py summary

# Export all user profiles to JSON
uv run python user_analysis.py export-all --output data/exports/user_profiles.json

# Refresh longitudinal tracking data
uv run python user_analysis.py refresh                    # All users
uv run python user_analysis.py refresh --username @user   # Specific user
```

**User Profile includes:**
- Video count and date range
- All claimed diagnoses with first-mention dates
- Top symptoms with frequency and severity variations
- Concordance scores per condition
- STRAIN narrative indicators
- Treatment mentions and effectiveness

**Concordance Analysis:**
- Compares reported symptoms to expected symptoms for each claimed condition
- Core symptom score (did they report the defining symptoms?)
- Flags users with consistently low concordance (< 0.3)
- **Fuzzy matching with 100+ synonym mappings** for EDS/MCAS/POTS terminology (e.g., "food sensitivity" ↔ "food reactions", "heart racing" ↔ "tachycardia")
- **Condition name filtering** - Excludes when the LLM extracts diagnosis names as "symptoms" (e.g., "mast cell activation" won't count as a symptom for MCAS)

**Narrative Inconsistencies Detection:**
- Diagnosis source conflicts (same condition claimed as both self-diagnosed AND professionally diagnosed)
- Treatment effectiveness conflicts (same treatment reported as both helpful AND harmful)
- Symptom severity inconsistencies (same symptom reported as "mild" in one video, "severe" in another)

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
| `scripts/retranscribe.py` | Re-transcribe and re-extract for consistency |
| `scripts/detect_song_lyrics.py` | Flag song lyrics transcripts (backfill) |
| `scripts/clean_transcripts.py` | Remove repeated phrases from transcripts |

## Example Workflows

### Complete Research Workflow

```bash
# 1. Find videos from hashtags (browser will open - don't interact)
uv run python scripts/discover.py --hashtag ehlersdanlos --hashtag POTS --max-videos 500

# 2. Expand to get ALL videos from discovered users (last year only)
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

## Medical Vocabulary & Transcription Accuracy

The transcriber includes specialized support for chronic illness terminology:

### Vocabulary Hints

Whisper receives context about expected medical terms before transcribing, including:
- Condition names (EDS, MCAS, POTS, CIRS, hEDS, dysautonomia)
- 50+ medications (Xolair, Rhapsido, Dupixent, cromolyn, ketotifen, midodrine, etc.)
- Supplements (quercetin, DAO enzyme, magnesium glycinate, LMNT, etc.)
- Medical devices (compression stockings, ring splints, PICC line, port-a-cath)
- Medical terms (interleukin, tryptase, subluxation, gastroparesis, etc.)

### Auto-Corrections

200+ post-processing corrections fix common Whisper mistakes:

| Whisper hears | Corrected to |
|---------------|--------------|
| mass cell | mast cell |
| Zolair | Xolair |
| wrap seedo | Rhapsido |
| em cass / MKAS | MCAS |
| ehler danlos | Ehlers-Danlos |
| sigh bo | SIBO |
| inter leukin | interleukin |
| gastro paresis | gastroparesis |

### Re-transcribe for Consistency

If you need to update existing transcripts with the improved vocabulary:

```powershell
# Preview what would change
uv run python scripts/retranscribe.py --dry-run

# Re-transcribe and re-extract all videos (with backup)
uv run python scripts/retranscribe.py --backup --provider ollama --model gpt-oss:20b

# Re-transcribe only (keep existing extractions)
uv run python scripts/retranscribe.py --transcribe-only --backup

# Test on 10 videos first
uv run python scripts/retranscribe.py --limit 10 --backup --provider ollama --model gpt-oss:20b

# Continue from a specific video (resume after interruption)
uv run python scripts/retranscribe.py --start-from 240 --backup --provider ollama --model gpt-oss:20b

# Retry only specific failed videos
uv run python scripts/retranscribe.py --video-ids 217 239 --backup --provider ollama --model gpt-oss:20b
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview without making changes |
| `--backup` | Save old data to `data/transcripts/_backups/` |
| `--transcribe-only` | Skip re-extraction |
| `--start-from N` | Start from video ID N (skip earlier) |
| `--video-ids` | Process only specific video IDs |
| `--limit N` | Process only first N videos |
| `--provider` | LLM provider: ollama, deepseek, or anthropic (default: ollama) |
| `--model` | LLM model (default: gpt-oss:20b) |

This ensures dataset consistency for publication.

### Clean Repeated Phrases

Whisper sometimes "hallucinates" and repeats the same phrase many times. The cleaning script detects and removes these repetitions while preserving the original text for revert.

```bash
# Preview what would be cleaned (dry run with full text)
uv run python scripts/clean_transcripts.py --dry-run --verbose

# Only clean transcripts with >10% reduction
uv run python scripts/clean_transcripts.py --min-reduction 10

# Clean all transcripts (originals preserved automatically)
uv run python scripts/clean_transcripts.py

# Show cleaning statistics
uv run python scripts/clean_transcripts.py --stats

# Revert all cleaned transcripts back to original
uv run python scripts/clean_transcripts.py --revert
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Preview without modifying database |
| `--verbose` | Show full original and cleaned text |
| `--min-reduction N` | Only clean if reduction >= N% |
| `--limit N` | Process only first N transcripts |
| `--revert` | Restore original text from backup |
| `--stats` | Show cleaning statistics |

The script automatically:
- Preserves original in `original_text` column
- Updates `word_count` with cleaned count
- Records `cleaned_at` timestamp

New transcriptions automatically have repetitions removed during transcription.

### Treatment Normalization

The extractor automatically normalizes LLM responses to valid database values:

**Treatment Types:** `diet` -> `lifestyle`, `drug` -> `medication`, `vitamin` -> `supplement`, etc.

**Effectiveness:** `flared_harder` -> `made_worse`, `amazing` -> `very_helpful`, `useless` -> `not_helpful`, etc.

This prevents database constraint errors when the LLM returns creative but non-standard values.

### Song Lyrics Detection

TikTok videos often have songs playing instead of the creator speaking. The pipeline uses **ratio-based scoring** to detect and handle mixed content (videos with both lyrics AND spoken words).

**How it works:**
1. **Heuristics** - Fast pattern matching estimates lyrics ratio (repetitive phrases, rhyming, medical terms, conversational markers)
2. **LLM** - Ollama asks "what percentage is song lyrics?" and returns 0-100
3. **Combined ratio** - Weighted average (LLM 2x weight) produces `song_lyrics_ratio` (0.0-1.0)
4. **Filter by ratio** - Use SQL to filter: `WHERE song_lyrics_ratio < 0.8` for mostly spoken content

**Database column:**
| Column | Type | Description |
|--------|------|-------------|
| `song_lyrics_ratio` | FLOAT | 0.0 (pure spoken) to 1.0 (pure lyrics) |

**Ratio categories:**
| Range | Category | Default Action |
|-------|----------|----------------|
| < 0.2 | Pure spoken | Extract (default) |
| >= 0.2 | Has lyrics | Skip (default threshold) |
| >= 0.5 | Mostly lyrics | Skip |
| >= 0.8 | Pure lyrics | Skip |

**Automatic detection during pipeline:**
- Extraction automatically skips videos with `song_lyrics_ratio >= 0.2` (configurable)
- Run `detect_song_lyrics.py` first to pre-classify before extraction
- Override threshold: `--max-song-ratio 0.5` to be more lenient

**Recommended workflow order:**
1. Transcribe videos
2. Run song lyrics detection (uses repetition patterns)
3. Clean transcripts (removes Whisper hallucination loops)
4. Extract symptoms

If transcripts were cleaned before song detection, the script automatically uses `original_text` (preserved during cleaning) to preserve the repetition patterns needed for accurate detection.

**Backfill existing transcripts:**
```powershell
# Check statistics (with ratio breakdown)
uv run python scripts/detect_song_lyrics.py --stats

# Dry run - preview without updating database
uv run python scripts/detect_song_lyrics.py --dry-run --limit 10

# Dry run with verbose output (see all details)
uv run python scripts/detect_song_lyrics.py -v --dry-run --limit 5

# Run for real - process all unchecked transcripts
uv run python scripts/detect_song_lyrics.py

# Limit to first 100
uv run python scripts/detect_song_lyrics.py --limit 100

# Use a specific model
uv run python scripts/detect_song_lyrics.py --model llama3:8b

# Heuristics only (no LLM, faster but less accurate)
uv run python scripts/detect_song_lyrics.py --heuristics-only
```

| Option | Description |
|--------|-------------|
| `--stats` | Show detection statistics with ratio breakdown |
| `--dry-run` | Check transcripts without updating database |
| `-v, --verbose` | Show detailed logging for each transcript |
| `--limit N` | Process only first N transcripts |
| `--model` | Ollama model to use (default: from config) |
| `--workers N` | Number of parallel workers (default: 4) |
| `--heuristics-only` | Skip LLM, use only heuristics |

**SQL to add column (if upgrading existing database):**
```sql
ALTER TABLE transcripts ADD COLUMN song_lyrics_ratio REAL DEFAULT NULL;
CREATE INDEX idx_transcripts_song_lyrics_ratio ON transcripts(song_lyrics_ratio);

-- If you have the old is_song_lyrics column, remove it:
ALTER TABLE transcripts DROP COLUMN IF EXISTS is_song_lyrics;
DROP INDEX IF EXISTS idx_transcripts_is_song_lyrics;
```

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

### Idempotent Processing (Safe to Re-run)

The pipeline is designed to be safe to re-run without creating duplicates:

| Stage | Behavior |
|-------|----------|
| Download | Skips if audio file already exists |
| Transcribe | Skips if transcript already exists |
| Extract | Skips if `extracted_at` timestamp is set (use `--force` to re-extract) |
| Extract | Skips if song_lyrics_ratio >= 0.2 (configurable via --max-song-ratio) |
| Extract | Skips if word_count < 20 (configurable via --min-words) |

This means you can:
- Restart an interrupted pipeline safely
- Add new URLs to `urls.txt` and re-run (only new videos processed)
- Re-run after fixing errors without worrying about duplicates
- Use `--force` to re-extract all videos if needed

## Database Schema

### Core Tables
- `videos` - Video metadata, engagement metrics, author info, creator tier
- `transcripts` - Transcribed text with model provenance, song lyrics ratio, extraction timestamp, `original_text` (preserved when cleaned), `cleaned_at`
- `symptoms` - Extracted symptoms with severity, temporal patterns
- `claimed_diagnoses` - Conditions the speaker claims to have
- `treatments` - Medications, supplements, therapies mentioned
- `narrative_elements` - STRAIN framework indicators (self-diagnosis, doctor dismissal, stress triggers, etc.)

### Symptom Categories
The extractor classifies symptoms into these categories (defined in `extractor.py`):

| Category | Description |
|----------|-------------|
| musculoskeletal | Joint pain, hypermobility, dislocations, subluxations, chronic pain |
| craniocervical | CCI, AAI, Chiari, skull settling, tethered cord |
| cardiovascular | Tachycardia, palpitations, blood pressure issues, chest pain |
| orthostatic_intolerance | Dizziness, fainting, blood pooling, POTS symptoms |
| autonomic | Dysautonomia, adrenaline surges, nervous system dysregulation |
| thermoregulation | Temperature regulation, Raynaud's, heat/cold intolerance |
| gastrointestinal | Nausea, gastroparesis, IBS, reflux, SIBO, motility issues |
| mast_cell_allergy_like | Flushing, hives, anaphylaxis, MCAS, histamine reactions |
| respiratory | Shortness of breath, asthma-like symptoms, breathing difficulties |
| ent | **Sinus issues**, sinus pain/pressure, tinnitus, ear fullness, post-nasal drip |
| neurological | Headaches, migraines, neuropathy, nerve pain, tremors |
| cognitive | Brain fog, memory issues, word-finding difficulty, confusion |
| fatigue | Chronic fatigue, post-exertional malaise, crashes, PEM |
| dermatological | Skin hyperextensibility, bruising, scarring, rashes |
| vascular_bleeding | Easy bruising, heavy periods, nosebleeds, vascular fragility |
| gynecologic | Menstrual issues, endometriosis, pelvic pain, PCOS |
| urological | Bladder issues, interstitial cystitis, incontinence |
| ocular | Vision problems, dry eyes, light sensitivity |
| dental | TMJ, jaw pain, dental fragility |
| psychological | Anxiety, depression, PTSD, panic attacks |
| immune | Frequent infections, slow healing, autoimmune symptoms |
| sleep | Insomnia, sleep apnea, unrefreshing sleep |
| other | Other symptoms not fitting above categories |

To add custom categories, edit the `SYMPTOM_CATEGORIES` dictionary in `extractor.py`.

### Analysis Tables
- `expected_symptoms` - Medical reference data for each condition (EDS, MCAS, POTS, CIRS)
- `symptom_concordance` - How well reported symptoms match expected
- `comorbidity_pairs` - Which conditions appear together
- `transcript_quality` - Quality metrics for each transcript

### Longitudinal Tracking Tables (for Social Contagion Research)
- `user_profiles` - Aggregated statistics per creator (concordance, STRAIN indicators, flags)
- `diagnosis_timeline` - When each user first claimed each diagnosis, diagnosis order, time between
- `symptom_consistency` - Tracks severity reporting consistency per symptom per user

### Progress Tables
- `processing_runs` - Track batch processing runs
- `pipeline_progress` - Per-URL progress for resumable runs

### Creator Tier Categories

The pipeline automatically categorizes creators by follower count:

| Tier | Followers | Description |
|------|-----------|-------------|
| nano | <10K | Small personal accounts |
| micro | 10K-100K | Growing influence |
| mid | 100K-500K | Significant reach |
| macro | 500K-1M | Major influencer |
| mega | >1M | Celebrity-level reach |

## Configuration

Environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost/tiktok_disorders

# Extraction (choose one: ollama, deepseek, or anthropic)
EXTRACTOR_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gpt-oss:20b

# Or use DeepSeek API (cost-effective cloud option)
EXTRACTOR_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat  # or deepseek-reasoner

# Or use Anthropic Claude
EXTRACTOR_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# Transcription
WHISPER_MODEL=large-v3
TRANSCRIBER_BACKEND=faster-whisper
WHISPER_COMPUTE_TYPE=auto  # float16 for GPU, int8 for CPU

# Extraction thresholds
MIN_CONFIDENCE_SCORE=0.6

# Hugging Face (for faster model downloads)
HF_TOKEN=hf_...
```

## Running with Ollama (Local LLM)

For local extraction without API costs, use Ollama with a capable model:

### Setup Ollama

```bash
# Install Ollama (if not already installed)
# Download from https://ollama.ai or use winget:
winget install Ollama.Ollama

# Pull OpenAI's gpt-oss model (RECOMMENDED - best quality)
ollama pull gpt-oss:20b

# Start Ollama server (runs in background)
ollama serve
```

### Run Pipeline with Ollama

```powershell
# Full pipeline with gpt-oss:20b (recommended)
uv run python pipeline.py run --urls-file urls.txt --provider ollama --model gpt-oss:20b --tags EDS MCAS POTS CIRS

# Extract symptoms only (if already downloaded/transcribed)
uv run python pipeline.py extract --all --provider ollama --model gpt-oss:20b

# Be more lenient with song lyrics (include up to 50% lyrics)
uv run python pipeline.py extract --all --max-song-ratio 0.5

# Require longer transcripts (at least 50 words)
uv run python pipeline.py extract --all --min-words 50
```

### Recommended Ollama Models

| Model | Size | Context | Quality | Notes |
|-------|------|---------|---------|-------|
| `qwen3:32b` | 32B | 32k (131k YaRN) | **Best** | Excels at colloquial TikTok language, /think mode |
| `gpt-oss:20b` | 20B | 128k | Excellent | OpenAI open-weight, optimized for reasoning |
| `alibayram/medgemma:27b` | 27B | 8-32k | Excellent | Medical terminology embedded, 87.7% MedQA |
| `gpt-oss:120b` | 120B | 128k | Excellent | Requires 80GB VRAM |
| `qwen2.5:20b` | 20B | 32k | Very Good | Good alternative |
| `llama3:70b` | 70B | 8k | Very Good | Large but shorter context |

### Qwen3 Thinking Mode

Qwen3 models support `/think` and `/no_think` modes for controlling reasoning depth:

```bash
# Fast extraction (default) - uses /no_think for efficient JSON output
uv run python pipeline.py extract --all --model qwen3:32b

# Deep reasoning mode - uses /think for complex/ambiguous cases (slower)
uv run python pipeline.py extract --all --model qwen3:32b --thinking
```

| Mode | Flag | Best For |
|------|------|----------|
| `/no_think` | (default) | Fast extraction, straightforward transcripts |
| `/think` | `--thinking` | Complex cases, ambiguous language, validation |

**Why Qwen3?** It excels at understanding informal TikTok language like "my body just does weird stuff" or "it's like my joints are made of rubber bands" that clinical models might miss.

### MedGemma for Medical Validation

For medical terminology normalization or validation passes:

```bash
# Use MedGemma for extraction
uv run python pipeline.py extract --all --model alibayram/medgemma:27b
```

MedGemma has 87.7% MedQA accuracy with medical terms deeply embedded from training on clinical data.

### Optimizations for High-Capability Models

When using `gpt-oss:20b` or similar high-capability models, the pipeline automatically:

1. **Combined Extraction** - All data (symptoms, diagnoses, treatments, narrative) extracted in a single API call (4x faster)
2. **Extended Context** - Uses 32k context window for complex prompts
3. **Parallel Processing** - 20 concurrent extractions (optimized for multi-core workstations)
4. **Extended Timeouts** - 5-minute timeout for thorough reasoning

The pipeline detects `qwen3`, `gpt-oss`, `qwen2.5:20b`, `llama3:70b`, `mixtral`, and `medgemma` as high-capability models.

## Running with DeepSeek API

DeepSeek offers a cost-effective cloud API with strong reasoning capabilities. It uses an OpenAI-compatible API format.

### Setup DeepSeek

1. Get an API key from [DeepSeek Platform](https://platform.deepseek.com/)
2. Add to your `.env` file:

```bash
DEEPSEEK_API_KEY=sk-your-api-key-here
```

### Run Pipeline with DeepSeek

```powershell
# DeepSeek V3.2 (fast, cost-effective)
uv run python pipeline.py extract --all --provider deepseek --model deepseek-chat

# DeepSeek V3.2 with thinking mode (deeper reasoning)
uv run python pipeline.py extract --all --provider deepseek --model deepseek-reasoner

# Full pipeline with DeepSeek
uv run python pipeline.py run --urls-file urls.txt --provider deepseek --model deepseek-chat --tags EDS MCAS POTS
```

### DeepSeek Models

| Model | Description | Pricing (approx) |
|-------|-------------|------------------|
| `deepseek-chat` | DeepSeek-V3.2 non-thinking mode, fast extraction | ~$0.14/M input, $0.28/M output |
| `deepseek-reasoner` | DeepSeek-V3.2 thinking mode, deep reasoning | ~$0.55/M input, $2.19/M output |

DeepSeek is a great middle-ground between free local models (Ollama) and premium APIs (Anthropic).

## Output

The pipeline generates:

- **Database Records** - All extracted data with full provenance
- **Audio Files** - MP3 files organized by username: `data/audio/{username}/`
- **Transcripts** - JSON files organized by username: `data/transcripts/{username}/`
- **Visualizations** - Cluster plots in `data/visualizations/`
- **Reports** - JSON analysis reports in `data/reports/`
- **CSV Exports** - Symptom data for external analysis in `data/exports/`

### File Organization

Files are organized by TikTok username to keep things manageable:

```
data/
  audio/
    chronicallychillandhot/
      tiktok_123456_video_title.mp3
      tiktok_789012_another_video.mp3
    zebra_warrior/
      tiktok_345678_my_story.mp3
  transcripts/
    chronicallychillandhot/
      transcript_1_20240120_153000.json
      transcript_2_20240120_153100.json
    zebra_warrior/
      transcript_3_20240120_154000.json
```

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

### Example Stats Output (with --detailed)

```
============================================================
DATABASE STATISTICS
============================================================

Overview:
  Videos: 150
  Transcripts: 148
  Symptoms: 1,247
  Diagnoses: 312

Diagnoses by Condition:
  EDS: 89
  POTS: 76
  MCAS: 54
  CIRS: 23

--- STRAIN Framework Indicators ---
  Videos analyzed: 148
  Self-diagnosed: 67
  Professional diagnosis: 81
  Doctor dismissal mentioned: 43
  Medical gaslighting mentioned: 28
  Long diagnostic journey: 52
  Stress triggers mentioned: 71
  Symptom flares mentioned: 89
  Learned from TikTok: 34
  Online community mention: 56

Creator Influence Tiers:
  nano: 78 videos
  micro: 45 videos
  mid: 19 videos
  macro: 6 videos
  mega: 2 videos

============================================================
```

## Troubleshooting

### FFmpeg Errors
If you see `unable to obtain file audio codec with ffprobe`:
1. Ensure FFmpeg is installed: `ffmpeg -version`
2. Ensure it's in your PATH
3. Restart your terminal after installation

### CUDA/GPU Not Available

If you see `Using model: large-v3 on cpu` instead of `cuda`:

```powershell
# Install PyTorch with CUDA support
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Reinstall ctranslate2 (used by faster-whisper)
uv pip install --force-reinstall ctranslate2

# Verify GPU is detected
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

You should see:
```
CUDA: True, GPU: NVIDIA GeForce RTX 4090
```

When running the pipeline, confirm GPU is active:
```
Loading Whisper model 'large-v3' on cuda (faster-whisper)...
Using GPU: NVIDIA GeForce RTX 4090 (24.0 GB)
```

### TikTok Discovery Issues

**Browser window opens for hashtags:**
- This is expected! The script uses a real browser to scroll through hashtag pages
- Don't interact with the browser - let it scroll on its own
- URLs are saved after each hashtag (crash-safe)

**Captcha appears:**
- If a captcha appears, solve it manually in the browser window
- The script will wait 30 seconds then continue

**403 errors or timeouts:**
1. Install Playwright browsers: `uv run playwright install`
2. Increase delays: `--min-delay 5 --max-delay 10`
3. Try again later (TikTok rate limits)

**Discovery interrupted:**
- Don't worry! URLs are saved after each user/hashtag
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

## URL Progress Tracking

The pipeline automatically tracks processed and failed URLs:

| File | Purpose |
|------|---------|
| `urls.txt` | Pending URLs to process |
| `urls_processed.txt` | Successfully processed URLs (with timestamps) |
| `urls_failed.txt` | Failed URLs (with timestamps and error messages) |

### How It Works

1. **Pipeline processes URLs** from `urls.txt`
2. **Successful URLs are moved** to `urls_processed.txt` with timestamp
3. **Failed URLs are moved** to `urls_failed.txt` with timestamp and error message
4. **Next run skips** URLs already in processed or failed files
5. **Discovery checks all files** to avoid re-discovering processed videos

After each download run, you'll see a summary:

```
Download Summary:
  Total processed: 95/100
  New downloads: 80
  Already existed: 15
  Failed: 5
  Remaining in urls.txt: 0
```

### Features

- **Automatic file creation**: `urls_processed.txt` and `urls_failed.txt` are created automatically
- **Smart URL matching**: URLs are normalized (trailing slashes, whitespace) for reliable matching
- **Immediate tracking**: URLs are moved as soon as each download completes (crash-safe)
- **Error logging**: Failed URLs include the error message for debugging
- **Skip already-tried**: Re-running skips URLs in processed/failed files automatically

### Benefits

- **Track progress**: See how many URLs are pending, completed, or failed
- **Avoid duplicates**: Discovery won't re-add already processed videos
- **Easy debugging**: Check `urls_failed.txt` to see why downloads failed
- **Clean retry**: Use `--force` to re-process failed URLs after fixing issues

### Check Status

```powershell
# Quick stats
uv run python url_manager.py

# Output:
# URL Statistics:
#   Pending: 150
#   Processed: 8500
#   Failed: 23
#   Total known: 8673
```

### Re-process Failed URLs

```powershell
# Re-try all URLs (ignore processed/failed files)
uv run python pipeline.py download --urls-file urls.txt --force

# Or manually move URLs from failed back to pending
# (edit urls_failed.txt, move desired URLs to urls.txt)
```

### Disable URL Moving

If you prefer to keep all URLs in `urls.txt`:

```powershell
uv run python pipeline.py run --urls-file urls.txt --no-move-processed
```

Note: The `download` command always tracks URLs. The `--no-move-processed` flag is for the `run` command.

## License

This software is provided for research purposes only.

## Acknowledgements

This research pipeline was developed through **AI-assisted pair programming** - a collaboration between the researcher and Claude (Anthropic's AI assistant). The entire codebase, from initial concept to production-ready pipeline, was built iteratively through natural language conversation.

### A Note to Fellow Researchers

If you have some programming knowledge but feel overwhelmed by the prospect of building custom research software, consider trying AI-assisted development. This pipeline - with its video scraping, GPU-accelerated transcription, LLM-powered extraction, and PostgreSQL database - was built entirely through conversational pair programming.

You don't need to be a software engineer. You need:
- A clear research question
- Basic familiarity with Python (or willingness to learn)
- The ability to describe what you want in plain language
- Patience to iterate and refine

Modern AI assistants can help you:
- Design database schemas for your specific research needs
- Write scrapers and data collection tools
- Integrate machine learning models (Whisper, LLMs)
- Handle edge cases and error recovery
- Write documentation and tests

The future of research software may not be researchers learning to code alone, but researchers collaborating with AI to build exactly the tools they need.

## Citation

If you use this pipeline in your research, please cite:

> Public Health Narratives, Self-Diagnosis, and Symptom Attribution on TikTok: A Narrative-Based Observational Study. OSF. https://osf.io/5y46c

## Contact

For questions about this research project, see the OSF page: https://osf.io/5y46c
