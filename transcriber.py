"""
Advanced audio transcription module using Whisper.
Optimized for GPU acceleration (RTX 4090) with support for large models.
"""
try:
    import torch
except ImportError:
    torch = None

try:
    from faster_whisper import WhisperModel as FasterWhisperModel
except ImportError:
    FasterWhisperModel = None

try:
    import whisper as openai_whisper
except ImportError:
    openai_whisper = None
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

from config import (
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    TRANSCRIBER_BACKEND,
    WHISPER_COMPUTE_TYPE,
    ensure_directories,
)
from database import insert_transcript, get_video_by_id, get_transcript


class AudioTranscriber:
    """Transcribes audio files using OpenAI Whisper with GPU acceleration."""

    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
                       Default uses config.WHISPER_MODEL
            device: Device to use ('cuda' or 'cpu'). Auto-detects if not specified.
        """
        self.model_size = model_size or WHISPER_MODEL
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading Whisper model '{self.model_size}' on {self.device}...")
        self.model = whisper.load_model(self.model_size, device=self.device)

        # Display GPU info if using CUDA
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✓ Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

        ensure_directories()

    def transcribe(self, video_id: int, save_segments: bool = True,
                   temperature: float = 0.0, language: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio for a video.

        Args:
            video_id: Database ID of the video
            save_segments: Whether to save detailed segment information
            temperature: Sampling temperature (0 = deterministic, higher = more creative)
            language: Force a specific language (e.g., 'en'), or None for auto-detect
            **kwargs: Additional arguments passed to whisper.transcribe()

        Returns:
            Dictionary containing transcript and metadata
        """
        # Check if already transcribed
        existing = get_transcript(video_id)
        if existing:
            print(f"✓ Transcript already exists for video {video_id}")
            return {
                'transcript_id': existing['id'],
                'text': existing['text'],
                'language': existing['language'],
                'already_existed': True
            }

        # Get video info
        video = get_video_by_id(video_id)
        if not video:
            raise ValueError(f"Video {video_id} not found in database")

        audio_path = video.get('audio_path')
        if not audio_path or not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Transcribing: {audio_path}")
        print(f"Using model: {self.model_size} on {self.device}")

        # Transcribe with advanced options
        # For RTX 4090, we can use aggressive settings
        transcribe_options = {
            'language': language,
            'task': 'transcribe',
            'temperature': temperature,
            'best_of': 5 if temperature > 0 else 1,  # Multiple samples for non-zero temperature
            'beam_size': 5,  # Larger beam search for better quality
            'patience': 1.0,
            'length_penalty': 1.0,
            'suppress_tokens': "-1",
            'initial_prompt': "This is a video about chronic illness, specifically EDS, MCAS, or POTS. The speaker may discuss medical symptoms.",
            'condition_on_previous_text': True,
            'fp16': self.device == 'cuda',  # Use half precision on GPU for speed
            'verbose': True,
            **kwargs
        }

        result = self.model.transcribe(str(audio_path), **transcribe_options)

        # Extract results
        text = result['text'].strip()
        detected_language = result['language']
        segments = result.get('segments', []) if save_segments else None

        # Save to file
        output_filename = f"transcript_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = TRANSCRIPT_DIR / output_filename

        transcript_data = {
            'video_id': video_id,
            'text': text,
            'language': detected_language,
            'model': self.model_size,
            'device': self.device,
            'segments': segments,
            'transcribed_at': datetime.now().isoformat()
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f"✓ Transcript saved: {output_path}")

        # Store in database
        db_id = insert_transcript(
            video_id=video_id,
            text=text,
            language=detected_language,
            model_used=self.model_size,
            segments=segments
        )

        return {
            'transcript_id': db_id,
            'text': text,
            'language': detected_language,
            'segments': segments,
            'file_path': str(output_path),
            'word_count': len(text.split()),
            'already_existed': False
        }

    def transcribe_batch(self, video_ids: List[int], **kwargs) -> List[Dict[str, Any]]:
        """
        Transcribe multiple videos in batch.

        Args:
            video_ids: List of video database IDs
            **kwargs: Additional arguments passed to transcribe()

        Returns:
            List of transcription results
        """
        results = []
        print(f"Transcribing {len(video_ids)} videos...")

        for i, video_id in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}] Processing video {video_id}")
            try:
                result = self.transcribe(video_id, **kwargs)
                results.append({'video_id': video_id, 'success': True, 'data': result})
            except Exception as e:
                print(f"✗ Error transcribing video {video_id}: {e}")
                results.append({'video_id': video_id, 'success': False, 'error': str(e)})

        # Summary
        success_count = sum(1 for r in results if r['success'])
        print(f"\n✓ Successfully transcribed {success_count}/{len(video_ids)} videos")

        return results

    def get_available_models(self) -> List[str]:
        """Get list of available Whisper models."""
        return ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']

    def benchmark_models(self, audio_path: str):
        """
        Benchmark different Whisper models on a sample audio file.
        Useful for determining the best speed/quality tradeoff.

        Args:
            audio_path: Path to audio file for benchmarking
        """
        import time

        models_to_test = ['base', 'small', 'medium', 'large-v3']
        results = []

        for model_size in models_to_test:
            print(f"\n{'='*60}")
            print(f"Testing model: {model_size}")
            print('='*60)

            try:
                start_time = time.time()

                # Load model
                model = whisper.load_model(model_size, device=self.device)

                # Transcribe
                result = model.transcribe(audio_path, fp16=self.device == 'cuda')

                elapsed = time.time() - start_time

                results.append({
                    'model': model_size,
                    'time': elapsed,
                    'language': result['language'],
                    'text_length': len(result['text']),
                    'success': True
                })

                print(f"✓ Completed in {elapsed:.2f} seconds")
                print(f"  Text length: {len(result['text'])} characters")

            except Exception as e:
                print(f"✗ Error: {e}")
                results.append({
                    'model': model_size,
                    'success': False,
                    'error': str(e)
                })

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print('='*60)
        for r in results:
            if r['success']:
                print(f"{r['model']:12} - {r['time']:6.2f}s - {r['text_length']} chars")
            else:
                print(f"{r['model']:12} - FAILED: {r['error']}")

        return results


if __name__ == '__main__':
    # Test transcription
    print("Whisper Transcriber Test")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("⚠ CUDA not available, will use CPU")

    # Interactive test
    video_id = input("\nEnter video ID to transcribe (or 'q' to quit): ").strip()
    if video_id and video_id != 'q':
        try:
            transcriber = AudioTranscriber(model_size='large-v3')  # Use best model for RTX 4090
            result = transcriber.transcribe(int(video_id))
            print(f"\n✓ Transcription complete!")
            print(f"  Language: {result['language']}")
            print(f"  Word count: {result['word_count']}")
            print(f"\nFirst 500 characters:\n{result['text'][:500]}...")
        except Exception as e:
            print(f"✗ Error: {e}")
