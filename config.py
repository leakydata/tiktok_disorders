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

# API Keys / Providers
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
EXTRACTOR_PROVIDER = os.getenv('EXTRACTOR_PROVIDER', 'anthropic')
ANTHROPIC_MODEL = os.getenv('ANTHROPIC_MODEL', 'claude-opus-4-5-20251101')
DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')  # or deepseek-reasoner
DEEPSEEK_URL = os.getenv('DEEPSEEK_URL', 'https://api.deepseek.com')
MINIMAX_API_KEY = os.getenv('MINIMAX_API_KEY')
MINIMAX_MODEL = os.getenv('MINIMAX_MODEL', 'MiniMax-M3')
# MiniMax exposes an OpenAI-compatible chat endpoint at this base; the SDK
# appends /chat/completions.
MINIMAX_URL = os.getenv('MINIMAX_URL', 'https://api.minimax.io/v1')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'gpt-oss:20b')

# llama.cpp server (llama-server). Despite also serving gpt-oss, this is NOT
# Ollama: it speaks the OpenAI wire format and has no /api/generate, /api/chat
# or /api/tags. The base URL therefore includes /v1 and the OpenAI SDK is used.
LLAMACPP_URL = os.getenv('LLAMACPP_URL', 'http://127.0.0.1:8081/v1')
LLAMACPP_MODEL = os.getenv('LLAMACPP_MODEL', 'gpt-oss-120b')
# No auth, but the OpenAI SDK refuses to construct without some key.
LLAMACPP_API_KEY = os.getenv('LLAMACPP_API_KEY', 'no-key-required')
# gpt-oss reasons before answering and those tokens are billed against
# max_tokens. Measured on this box for a classification prompt: "low" 72
# completion tokens / 9.7s vs "high" 568 / 93.6s for the same answer. The
# reasoning is pure overhead for structured extraction, so "low" is the default.
LLAMACPP_REASONING_EFFORT = os.getenv('LLAMACPP_REASONING_EFFORT', 'low')
# Max concurrent requests. The server exposes 4 slots, but slots are not free
# throughput: this model is launched with --n-cpu-moe 26, so 59GB of weights do
# not fit the 24GB card and the MoE layers run on CPU across --threads 20.
# Concurrent requests then contend for one saturated memory-bandwidth path.
# Measured on the same 4 transcripts: serial 278.8s vs 4-way 330.9s -- batching
# was 19% SLOWER, with per-request latency rising 69.7s -> 294.2s.
#
# So serial is the default. Raise this only if the model is ever fully resident
# on the GPU (drop --n-cpu-moe, or use a smaller quant), where the slots would
# genuinely parallelise.
LLAMACPP_CONCURRENCY = int(os.getenv('LLAMACPP_CONCURRENCY', '1'))
# Each slot gets its own 32k context; leave room for the transcript + prompt.
LLAMACPP_MAX_TOKENS = int(os.getenv('LLAMACPP_MAX_TOKENS', '8192'))

HF_TOKEN = os.getenv('HF_TOKEN')

# Set HF_TOKEN in environment for huggingface_hub to find it
if HF_TOKEN:
    os.environ['HF_TOKEN'] = HF_TOKEN

# Data Directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
AUDIO_DIR = Path(os.getenv('AUDIO_DIR', DATA_DIR / 'audio'))
TRANSCRIPT_DIR = Path(os.getenv('TRANSCRIPT_DIR', DATA_DIR / 'transcripts'))
VISUALIZATION_DIR = Path(os.getenv('VISUALIZATION_DIR', DATA_DIR / 'visualizations'))

# Whisper Model Configuration
WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'base')
WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large']
TRANSCRIBER_BACKEND = os.getenv('TRANSCRIBER_BACKEND', 'faster-whisper')
WHISPER_COMPUTE_TYPE = os.getenv('WHISPER_COMPUTE_TYPE', 'auto')

# Voice activity detection (faster-whisper only). Silero VAD drops non-speech
# regions before decoding, which suppresses Whisper's tendency to hallucinate
# text over music and silence. Roughly 20% of this corpus is music-heavy
# (song_lyrics_ratio >= 0.2), so this is on by default.
#
# Set WHISPER_VAD=false to reproduce pre-2026-07 transcription behaviour.
WHISPER_VAD = os.getenv('WHISPER_VAD', 'true').lower() not in ('false', '0', 'no')
# Speech probability threshold. Higher = stricter about what counts as speech.
WHISPER_VAD_THRESHOLD = float(os.getenv('WHISPER_VAD_THRESHOLD', '0.5'))
# Speech shorter than this (ms) is discarded.
WHISPER_VAD_MIN_SPEECH_MS = int(os.getenv('WHISPER_VAD_MIN_SPEECH_MS', '250'))
# Silence shorter than this (ms) does not split a speech region. Kept generous
# so natural pauses mid-sentence do not fragment segments.
WHISPER_VAD_MIN_SILENCE_MS = int(os.getenv('WHISPER_VAD_MIN_SILENCE_MS', '700'))
# Padding retained either side of detected speech (ms), so VAD does not clip
# quiet word onsets.
WHISPER_VAD_SPEECH_PAD_MS = int(os.getenv('WHISPER_VAD_SPEECH_PAD_MS', '200'))

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

    if EXTRACTOR_PROVIDER not in {'anthropic', 'ollama', 'deepseek', 'minimax',
                                  'llamacpp'}:
        errors.append(
            "EXTRACTOR_PROVIDER must be 'anthropic', 'ollama', 'deepseek', "
            "'minimax', or 'llamacpp'")

    if EXTRACTOR_PROVIDER == 'minimax' and not MINIMAX_API_KEY:
        errors.append("MINIMAX_API_KEY is not set")

    if EXTRACTOR_PROVIDER == 'anthropic' and not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY is not set")
    
    if EXTRACTOR_PROVIDER == 'deepseek' and not DEEPSEEK_API_KEY:
        errors.append("DEEPSEEK_API_KEY is not set")

    if TRANSCRIBER_BACKEND not in {'faster-whisper', 'openai-whisper'}:
        errors.append("TRANSCRIBER_BACKEND must be 'faster-whisper' or 'openai-whisper'")

    if WHISPER_MODEL not in WHISPER_MODELS:
        errors.append(f"WHISPER_MODEL must be one of {WHISPER_MODELS}")

    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    ensure_directories()


if __name__ == '__main__':
    # Test configuration
    try:
        validate_config()
        print("Configuration is valid")
        print(f"  Database: {DATABASE_URL}")
        print(f"  Audio directory: {AUDIO_DIR}")
        print(f"  Transcript directory: {TRANSCRIPT_DIR}")
        print(f"  Visualization directory: {VISUALIZATION_DIR}")
        print(f"  Whisper model: {WHISPER_MODEL}")
    except ValueError as e:
        print(f"Configuration error:\n{e}")
