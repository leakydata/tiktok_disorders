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
from database import insert_transcript, get_video_by_id, get_transcript, insert_transcript_quality


class AudioTranscriber:
    """Transcribes audio files using Whisper with GPU acceleration."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        backend: Optional[str] = None,
    ):
        """
        Initialize the transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
                       Default uses config.WHISPER_MODEL
            device: Device to use ('cuda' or 'cpu'). Auto-detects if not specified.
        """
        self.model_size = model_size or WHISPER_MODEL
        self.backend = (backend or TRANSCRIBER_BACKEND).lower()
        if self.backend not in {'faster-whisper', 'openai-whisper'}:
            raise ValueError("backend must be 'faster-whisper' or 'openai-whisper'")

        if device:
            self.device = device
        else:
            # Detailed CUDA detection
            if torch is None:
                print("⚠ PyTorch not installed - using CPU")
                self.device = 'cpu'
            elif not torch.cuda.is_available():
                print("⚠ CUDA not available - using CPU")
                print("  To enable GPU: rm -rf .venv && uv sync --group cuda")
                self.device = 'cpu'
            else:
                self.device = 'cuda'

        print(f"Loading Whisper model '{self.model_size}' on {self.device} ({self.backend})...")
        if self.backend == 'faster-whisper':
            if FasterWhisperModel is None:
                raise ImportError("faster-whisper is not installed")
            compute_type = WHISPER_COMPUTE_TYPE
            if compute_type == 'auto':
                compute_type = 'float16' if self.device == 'cuda' else 'int8'
            self.model = FasterWhisperModel(
                self.model_size,
                device=self.device,
                compute_type=compute_type,
            )
        else:
            if openai_whisper is None:
                raise ImportError("openai-whisper is not installed")
            self.model = openai_whisper.load_model(self.model_size, device=self.device)

        # Display GPU info if using CUDA
        if self.device == 'cuda' and torch:
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
            **kwargs: Additional arguments passed to the backend transcribe()

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
        print(f"Using model: {self.model_size} on {self.device} ({self.backend})")

        if self.backend == 'faster-whisper':
            segments_iter, info = self.model.transcribe(
                str(audio_path),
                language=language,
                beam_size=5,
                temperature=temperature,
                **kwargs,
            )

            segment_items = []
            text_parts = []
            for segment in segments_iter:
                text_parts.append(segment.text)
                if save_segments:
                    segment_items.append({
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'text': segment.text
                    })

            text = " ".join(text_parts).strip()
            detected_language = info.language if info and info.language else (language or 'unknown')
            segments = segment_items if save_segments else None
        else:
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
            'backend': self.backend,
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
            segments=segments,
            model_backend=self.backend,
            transcription_device=self.device
        )

        # Assess transcript quality
        quality_metrics = self._assess_quality(text, segments)
        if quality_metrics:
            insert_transcript_quality(db_id, quality_metrics)
            print(f"  Quality score: {quality_metrics['quality_score']:.2f}")

        return {
            'transcript_id': db_id,
            'text': text,
            'language': detected_language,
            'segments': segments,
            'file_path': str(output_path),
            'word_count': len(text.split()),
            'quality_score': quality_metrics.get('quality_score') if quality_metrics else None,
            'already_existed': False
        }

    def _assess_quality(self, text: str, segments: Optional[List] = None) -> Dict[str, Any]:
        """
        Assess transcript quality based on various metrics.

        Returns quality metrics dictionary.
        """
        if not text or len(text.strip()) < 10:
            return {'quality_score': 0.0, 'issues': ['transcript_too_short']}

        issues = []
        words = text.lower().split()
        word_count = len(words)

        # Filler words detection
        filler_words = {'um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally', 'so', 'right'}
        filler_count = sum(1 for w in words if w in filler_words)
        filler_ratio = filler_count / word_count if word_count > 0 else 0

        if filler_ratio > 0.15:
            issues.append('high_filler_ratio')

        # Medical term density (basic check)
        medical_terms = {
            'pain', 'fatigue', 'dizziness', 'nausea', 'headache', 'joint', 'muscle',
            'heart', 'blood', 'pressure', 'syndrome', 'chronic', 'diagnosis', 'doctor',
            'medication', 'symptoms', 'flare', 'eds', 'mcas', 'pots', 'hypermobility',
            'tachycardia', 'mast', 'cell', 'allergy', 'reaction', 'inflammation'
        }
        medical_count = sum(1 for w in words if w in medical_terms)
        medical_density = medical_count / word_count if word_count > 0 else 0

        # Segment confidence analysis (if available)
        avg_confidence = None
        low_confidence_segments = 0
        total_segments = 0

        if segments:
            total_segments = len(segments)
            # Check for segments with low confidence or issues
            for seg in segments:
                if isinstance(seg, dict):
                    # Placeholder - faster-whisper doesn't expose confidence directly
                    # but we can detect issues via text patterns
                    seg_text = seg.get('text', '')
                    if '[' in seg_text or '?' in seg_text or len(seg_text.strip()) < 3:
                        low_confidence_segments += 1

        # Completeness check (no obvious truncation)
        completeness_score = 1.0
        if text.strip().endswith('...') or text.strip().endswith('-'):
            completeness_score = 0.7
            issues.append('possible_truncation')

        if word_count < 50:
            completeness_score *= 0.8
            issues.append('very_short')

        # Clarity score (based on sentence structure)
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        avg_sentence_length = word_count / len(sentences) if sentences else 0

        clarity_score = 1.0
        if avg_sentence_length > 40:
            clarity_score = 0.7
            issues.append('long_sentences')
        elif avg_sentence_length < 5:
            clarity_score = 0.8
            issues.append('fragmented')

        # Overall quality score
        quality_score = (
            (1 - filler_ratio) * 0.2 +
            min(medical_density * 10, 1.0) * 0.2 +
            completeness_score * 0.3 +
            clarity_score * 0.3
        )

        # Penalize if too many issues
        quality_score = max(0.1, quality_score - len(issues) * 0.05)

        return {
            'quality_score': round(quality_score, 3),
            'clarity_score': round(clarity_score, 3),
            'completeness_score': round(completeness_score, 3),
            'medical_term_density': round(medical_density, 4),
            'filler_word_ratio': round(filler_ratio, 4),
            'avg_confidence': avg_confidence,
            'low_confidence_segments': low_confidence_segments,
            'total_segments': total_segments,
            'issues': issues
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

                if openai_whisper is None:
                    raise ImportError("openai-whisper is not installed")

                # Load model
                model = openai_whisper.load_model(model_size, device=self.device)

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
    if torch and torch.cuda.is_available():
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
