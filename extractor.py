"""
Symptom extraction module using Claude API.
Analyzes transcripts to identify and categorize symptoms with confidence scores.
Optimized for high-throughput parallel processing.
"""
import anthropic
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ANTHROPIC_API_KEY, MIN_CONFIDENCE_SCORE
from database import insert_symptom, get_transcript, get_video_by_id


# Comprehensive symptom categories for EDS, MCAS, and POTS
SYMPTOM_CATEGORIES = {
    'musculoskeletal': 'Joint pain, hypermobility, dislocations, subluxations, chronic pain',
    'cardiovascular': 'Tachycardia, palpitations, blood pressure issues, dizziness, fainting',
    'gastrointestinal': 'Nausea, vomiting, gastroparesis, IBS, reflux, constipation, diarrhea',
    'neurological': 'Brain fog, headaches, migraines, difficulty concentrating, memory issues',
    'autonomic': 'Temperature regulation, sweating issues, tremors, exercise intolerance',
    'allergic': 'Flushing, hives, itching, anaphylaxis, food sensitivities, chemical sensitivities',
    'dermatological': 'Skin hyperextensibility, easy bruising, scarring, skin fragility, rashes',
    'respiratory': 'Shortness of breath, asthma-like symptoms, breathing difficulties',
    'fatigue': 'Chronic fatigue, post-exertional malaise, exhaustion, sleep issues',
    'other': 'Other symptoms not fitting above categories'
}


class SymptomExtractor:
    """Extracts symptoms from transcripts using Claude API."""

    def __init__(self, api_key: Optional[str] = None, max_workers: int = 10):
        """
        Initialize the symptom extractor.

        Args:
            api_key: Anthropic API key (defaults to config.ANTHROPIC_API_KEY)
            max_workers: Maximum parallel API calls (with 389GB RAM, we can handle many!)
        """
        self.api_key = api_key or ANTHROPIC_API_KEY
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.max_workers = max_workers

    def _build_extraction_prompt(self, transcript: str) -> str:
        """Build the prompt for Claude to extract symptoms."""
        categories_str = "\n".join([f"- {cat}: {desc}" for cat, desc in SYMPTOM_CATEGORIES.items()])

        return f"""You are a medical research assistant analyzing a transcript about chronic illnesses (EDS, MCAS, POTS).

Extract ALL mentioned symptoms from the following transcript. For each symptom:
1. Identify the specific symptom or complaint
2. Categorize it using one of these categories:
{categories_str}

3. Assign a confidence score (0.0-1.0) based on:
   - 1.0: Explicitly stated by the speaker as their personal experience
   - 0.8-0.9: Strongly implied or clearly described
   - 0.6-0.7: Mentioned but with less detail or certainty
   - 0.4-0.5: Vaguely referenced or possibly relevant
   - Below 0.4: Not worth recording

4. Include surrounding context (the sentence or phrase where it was mentioned)

Return your response as a JSON array of objects with this structure:
[
  {{
    "symptom": "brief description of the symptom",
    "category": "category name from the list above",
    "confidence": 0.95,
    "context": "the relevant quote from the transcript"
  }},
  ...
]

TRANSCRIPT:
{transcript}

Return ONLY the JSON array, no additional text."""

    def extract_symptoms(self, video_id: int, min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract symptoms from a video's transcript.

        Args:
            video_id: Database ID of the video
            min_confidence: Minimum confidence score to save (defaults to config.MIN_CONFIDENCE_SCORE)

        Returns:
            Dictionary with extraction results
        """
        min_conf = min_confidence if min_confidence is not None else MIN_CONFIDENCE_SCORE

        # Get transcript
        transcript_data = get_transcript(video_id)
        if not transcript_data:
            raise ValueError(f"No transcript found for video {video_id}")

        transcript_text = transcript_data['text']

        # Check if transcript is too short
        if len(transcript_text.split()) < 20:
            print(f"⚠ Transcript too short ({len(transcript_text.split())} words), skipping")
            return {'video_id': video_id, 'symptoms_found': 0, 'symptoms_saved': 0}

        print(f"Extracting symptoms from video {video_id} ({len(transcript_text)} chars)...")

        # Call Claude API
        try:
            response = self.client.messages.create(
                model="claude-opus-4-5-20251101",  # Use the most capable model
                max_tokens=4096,
                temperature=0.0,  # Deterministic for consistency
                messages=[{
                    "role": "user",
                    "content": self._build_extraction_prompt(transcript_text)
                }]
            )

            # Parse response
            response_text = response.content[0].text.strip()

            # Extract JSON from response (in case there's extra text)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            symptoms = json.loads(response_text)

            # Validate and save symptoms
            saved_count = 0
            for symptom_data in symptoms:
                confidence = symptom_data.get('confidence', 0.0)

                if confidence >= min_conf:
                    insert_symptom(
                        video_id=video_id,
                        category=symptom_data.get('category', 'other'),
                        symptom=symptom_data['symptom'],
                        confidence=confidence,
                        context=symptom_data.get('context')
                    )
                    saved_count += 1

            print(f"✓ Extracted {len(symptoms)} symptoms, saved {saved_count} (confidence >= {min_conf})")

            return {
                'video_id': video_id,
                'symptoms_found': len(symptoms),
                'symptoms_saved': saved_count,
                'min_confidence': min_conf,
                'success': True
            }

        except json.JSONDecodeError as e:
            print(f"✗ Failed to parse Claude response: {e}")
            print(f"Response text: {response_text[:500]}...")
            return {
                'video_id': video_id,
                'success': False,
                'error': f"JSON parse error: {e}"
            }

        except Exception as e:
            print(f"✗ Error extracting symptoms: {e}")
            return {
                'video_id': video_id,
                'success': False,
                'error': str(e)
            }

    def extract_batch(self, video_ids: List[int], min_confidence: Optional[float] = None,
                     parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Extract symptoms from multiple videos.

        Args:
            video_ids: List of video database IDs
            min_confidence: Minimum confidence score to save
            parallel: Whether to process in parallel (with 389GB RAM, absolutely!)

        Returns:
            List of extraction results
        """
        print(f"Extracting symptoms from {len(video_ids)} videos...")

        if parallel and len(video_ids) > 1:
            # Parallel processing for maximum throughput
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_video = {
                    executor.submit(self.extract_symptoms, vid, min_confidence): vid
                    for vid in video_ids
                }

                for i, future in enumerate(as_completed(future_to_video), 1):
                    video_id = future_to_video[future]
                    try:
                        result = future.result()
                        results.append(result)
                        if result.get('success', False):
                            print(f"[{i}/{len(video_ids)}] ✓ Video {video_id}: {result['symptoms_saved']} symptoms")
                        else:
                            print(f"[{i}/{len(video_ids)}] ✗ Video {video_id}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"[{i}/{len(video_ids)}] ✗ Video {video_id}: {e}")
                        results.append({'video_id': video_id, 'success': False, 'error': str(e)})
        else:
            # Sequential processing
            results = []
            for i, video_id in enumerate(video_ids, 1):
                print(f"\n[{i}/{len(video_ids)}] Processing video {video_id}")
                result = self.extract_symptoms(video_id, min_confidence)
                results.append(result)

        # Summary
        success_count = sum(1 for r in results if r.get('success', False))
        total_symptoms = sum(r.get('symptoms_saved', 0) for r in results if r.get('success', False))
        print(f"\n✓ Successfully processed {success_count}/{len(video_ids)} videos")
        print(f"  Total symptoms extracted: {total_symptoms}")

        return results

    def reanalyze_with_custom_categories(self, video_id: int, custom_categories: Dict[str, str]) -> Dict[str, Any]:
        """
        Re-extract symptoms with custom categories.

        Args:
            video_id: Database ID of the video
            custom_categories: Dictionary of category names and descriptions

        Returns:
            Extraction results with custom categories
        """
        # Temporarily replace categories
        original_categories = SYMPTOM_CATEGORIES.copy()
        SYMPTOM_CATEGORIES.clear()
        SYMPTOM_CATEGORIES.update(custom_categories)

        try:
            result = self.extract_symptoms(video_id)
            return result
        finally:
            # Restore original categories
            SYMPTOM_CATEGORIES.clear()
            SYMPTOM_CATEGORIES.update(original_categories)


if __name__ == '__main__':
    # Test symptom extraction
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extractor.py <video_id>")
        sys.exit(1)

    video_id = int(sys.argv[1])

    try:
        extractor = SymptomExtractor()
        result = extractor.extract_symptoms(video_id)
        print(f"\nExtraction Result:")
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
