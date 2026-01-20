"""
Symptom extraction module using Claude or Ollama.
Analyzes transcripts to identify and categorize symptoms with confidence scores.
Optimized for high-throughput parallel processing.
"""
import anthropic
import requests
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    EXTRACTOR_PROVIDER,
    MIN_CONFIDENCE_SCORE,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from database import (
    insert_symptom, get_transcript, get_video_by_id,
    insert_claimed_diagnosis, get_diagnoses_by_video,
    calculate_symptom_concordance, insert_treatment,
    update_comorbidity_pairs
)


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
    """Extracts symptoms from transcripts using Claude or Ollama."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_workers: int = 10,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        ollama_url: Optional[str] = None,
    ):
        """
        Initialize the symptom extractor.

        Args:
            api_key: Anthropic API key (defaults to config.ANTHROPIC_API_KEY)
            max_workers: Maximum parallel API calls (with 389GB RAM, we can handle many!)
        """
        self.provider = (provider or EXTRACTOR_PROVIDER).lower()
        if self.provider not in {"anthropic", "ollama"}:
            raise ValueError("provider must be 'anthropic' or 'ollama'")

        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model = model or (ANTHROPIC_MODEL if self.provider == "anthropic" else OLLAMA_MODEL)
        self.ollama_url = (ollama_url or OLLAMA_URL).rstrip("/")

        if self.provider == "anthropic":
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
        self.max_workers = max_workers

    def _build_extraction_prompt(self, transcript: str) -> str:
        """Build the prompt for the model to extract symptoms with enhanced research data."""
        categories_str = "\n".join([f"- {cat}: {desc}" for cat, desc in SYMPTOM_CATEGORIES.items()])

        return f"""You are a medical research assistant analyzing a transcript about chronic illnesses (EDS, MCAS, POTS).

Extract ALL mentioned symptoms from the following transcript. For each symptom, provide:

1. **symptom**: Brief description of the symptom or complaint
2. **category**: One of these categories:
{categories_str}

3. **confidence** (0.0-1.0):
   - 1.0: Explicitly stated by the speaker as their personal experience
   - 0.8-0.9: Strongly implied or clearly described
   - 0.6-0.7: Mentioned but with less detail or certainty
   - 0.4-0.5: Vaguely referenced or possibly relevant
   - Below 0.4: Not worth recording

4. **severity**: One of: "mild", "moderate", "severe", "unspecified"
   - Based on how the speaker describes the impact on their life

5. **temporal_pattern**: One of: "acute", "chronic", "intermittent", "progressive", "unspecified"
   - acute: sudden onset, short duration
   - chronic: long-lasting, persistent
   - intermittent: comes and goes
   - progressive: getting worse over time

6. **body_location**: Specific body part if mentioned (e.g., "knees", "lower back", "hands")

7. **triggers**: Array of triggers if mentioned (e.g., ["standing", "heat", "certain foods"])

8. **is_personal_experience**: true if speaker describes their own experience, false if discussing general information

9. **context**: The relevant quote from the transcript

Return your response as a JSON array:
[
  {{
    "symptom": "joint hypermobility",
    "category": "musculoskeletal",
    "confidence": 0.95,
    "severity": "moderate",
    "temporal_pattern": "chronic",
    "body_location": "fingers and wrists",
    "triggers": ["repetitive motion"],
    "is_personal_experience": true,
    "context": "my fingers bend way back and my wrists are always giving out"
  }}
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

        # Call model API
        try:
            response_text = self._call_model(self._build_extraction_prompt(transcript_text))

            # Extract JSON from response (in case there's extra text)
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            symptoms = json.loads(response_text)

            # Validate and save symptoms with enhanced fields
            saved_count = 0
            for symptom_data in symptoms:
                confidence = symptom_data.get('confidence', 0.0)

                if confidence >= min_conf:
                    insert_symptom(
                        video_id=video_id,
                        category=symptom_data.get('category', 'other'),
                        symptom=symptom_data['symptom'],
                        confidence=confidence,
                        context=symptom_data.get('context'),
                        severity=symptom_data.get('severity', 'unspecified'),
                        temporal_pattern=symptom_data.get('temporal_pattern', 'unspecified'),
                        body_location=symptom_data.get('body_location'),
                        triggers=symptom_data.get('triggers', []),
                        is_personal_experience=symptom_data.get('is_personal_experience', True),
                        extractor_model=self.model,
                        extractor_provider=self.provider
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
            print(f"✗ Failed to parse model response: {e}")
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

    def extract_diagnoses(self, video_id: int) -> Dict[str, Any]:
        """
        Extract claimed diagnoses from a video's transcript.

        Args:
            video_id: Database ID of the video

        Returns:
            Dictionary with diagnosis extraction results
        """
        # Get transcript
        transcript_data = get_transcript(video_id)
        if not transcript_data:
            raise ValueError(f"No transcript found for video {video_id}")

        transcript_text = transcript_data['text']

        if len(transcript_text.split()) < 10:
            return {'video_id': video_id, 'diagnoses_found': 0, 'diagnoses_saved': 0}

        print(f"Extracting diagnoses from video {video_id}...")

        prompt = f"""Analyze this transcript and extract any medical conditions or diagnoses that the speaker claims to have.

Look for mentions of:
- EDS (Ehlers-Danlos Syndrome) - any type (hEDS, vEDS, classical, etc.)
- MCAS (Mast Cell Activation Syndrome)
- POTS (Postural Orthostatic Tachycardia Syndrome)
- Fibromyalgia
- CFS/ME (Chronic Fatigue Syndrome / Myalgic Encephalomyelitis)
- Any other chronic illnesses mentioned

For each claimed diagnosis, provide:
1. **condition_code**: Abbreviation (EDS, MCAS, POTS, FIBROMYALGIA, CFS, or OTHER)
2. **condition_name**: Full name as the speaker calls it
3. **confidence**: 0.0-1.0 based on how clearly they claim to have it
4. **is_self_diagnosed**: true if they mention self-diagnosis, false if doctor-diagnosed, null if unclear
5. **diagnosis_date_mentioned**: Any date/year mentioned for when diagnosed (or null)
6. **context**: The quote where they mention having this condition

Return a JSON array:
[
  {{
    "condition_code": "EDS",
    "condition_name": "hypermobile Ehlers-Danlos Syndrome",
    "confidence": 0.95,
    "is_self_diagnosed": false,
    "diagnosis_date_mentioned": "2020",
    "context": "I was diagnosed with hEDS in 2020 by a geneticist"
  }}
]

Return ONLY the JSON array. If no diagnoses are mentioned, return an empty array [].

TRANSCRIPT:
{transcript_text}"""

        try:
            response_text = self._call_model(prompt)

            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            diagnoses = json.loads(response_text)

            # Save diagnoses
            saved_count = 0
            diagnosis_ids = []
            for diag in diagnoses:
                if diag.get('confidence', 0) >= 0.5:
                    diag_id = insert_claimed_diagnosis(
                        video_id=video_id,
                        condition_code=diag.get('condition_code', 'OTHER'),
                        condition_name=diag.get('condition_name', 'Unknown'),
                        confidence=diag.get('confidence', 0.5),
                        context=diag.get('context'),
                        is_self_diagnosed=diag.get('is_self_diagnosed'),
                        diagnosis_date_mentioned=diag.get('diagnosis_date_mentioned'),
                        extractor_model=self.model,
                        extractor_provider=self.provider
                    )
                    diagnosis_ids.append(diag_id)
                    saved_count += 1

            print(f"✓ Extracted {len(diagnoses)} diagnoses, saved {saved_count}")

            return {
                'video_id': video_id,
                'diagnoses_found': len(diagnoses),
                'diagnoses_saved': saved_count,
                'diagnosis_ids': diagnosis_ids,
                'success': True
            }

        except Exception as e:
            print(f"✗ Error extracting diagnoses: {e}")
            return {
                'video_id': video_id,
                'success': False,
                'error': str(e)
            }

    def extract_treatments(self, video_id: int) -> Dict[str, Any]:
        """
        Extract treatments/medications mentioned in a video's transcript.

        Args:
            video_id: Database ID of the video

        Returns:
            Dictionary with treatment extraction results
        """
        # Get transcript
        transcript_data = get_transcript(video_id)
        if not transcript_data:
            raise ValueError(f"No transcript found for video {video_id}")

        transcript_text = transcript_data['text']

        if len(transcript_text.split()) < 10:
            return {'video_id': video_id, 'treatments_found': 0, 'treatments_saved': 0}

        print(f"Extracting treatments from video {video_id}...")

        prompt = f"""Analyze this transcript and extract any treatments, medications, supplements, or therapies mentioned.

For each treatment, provide:
1. **treatment_type**: One of: "medication", "supplement", "therapy", "lifestyle", "procedure", "device", "other"
2. **treatment_name**: The name of the treatment/medication
3. **dosage**: Dosage if mentioned (e.g., "10mg", "twice daily")
4. **frequency**: How often taken if mentioned
5. **effectiveness**: Speaker's assessment - "very_helpful", "somewhat_helpful", "not_helpful", "made_worse", or "unspecified"
6. **side_effects**: Array of side effects mentioned (can be empty)
7. **is_current**: true if currently using, false if stopped, null if unclear
8. **target_condition**: What condition it's for (e.g., "POTS", "pain", "sleep")
9. **target_symptoms**: Array of symptoms it addresses
10. **context**: The relevant quote
11. **confidence**: 0.0-1.0 based on how clearly mentioned

Common treatments for EDS/MCAS/POTS include:
- Medications: beta blockers, antihistamines, mast cell stabilizers, pain meds, etc.
- Supplements: salt tablets, electrolytes, vitamins, collagen, etc.
- Therapies: physical therapy, occupational therapy, etc.
- Lifestyle: compression garments, diet changes, pacing, etc.
- Devices: heart rate monitors, mobility aids, etc.

Return a JSON array:
[
  {{
    "treatment_type": "medication",
    "treatment_name": "propranolol",
    "dosage": "10mg",
    "frequency": "twice daily",
    "effectiveness": "very_helpful",
    "side_effects": ["fatigue"],
    "is_current": true,
    "target_condition": "POTS",
    "target_symptoms": ["tachycardia", "palpitations"],
    "context": "propranolol has been a game changer for my POTS",
    "confidence": 0.95
  }}
]

Return ONLY the JSON array. If no treatments are mentioned, return [].

TRANSCRIPT:
{transcript_text}"""

        try:
            response_text = self._call_model(prompt)

            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()

            treatments = json.loads(response_text)

            # Save treatments
            saved_count = 0
            for treatment in treatments:
                if treatment.get('confidence', 0) >= 0.4:
                    insert_treatment(
                        video_id=video_id,
                        treatment_type=treatment.get('treatment_type', 'other'),
                        treatment_name=treatment.get('treatment_name', 'Unknown'),
                        dosage=treatment.get('dosage'),
                        frequency=treatment.get('frequency'),
                        effectiveness=treatment.get('effectiveness', 'unspecified'),
                        side_effects=treatment.get('side_effects', []),
                        is_current=treatment.get('is_current'),
                        target_condition=treatment.get('target_condition'),
                        target_symptoms=treatment.get('target_symptoms', []),
                        context=treatment.get('context'),
                        confidence=treatment.get('confidence', 0.5),
                        extractor_model=self.model,
                        extractor_provider=self.provider
                    )
                    saved_count += 1

            print(f"✓ Extracted {len(treatments)} treatments, saved {saved_count}")

            return {
                'video_id': video_id,
                'treatments_found': len(treatments),
                'treatments_saved': saved_count,
                'success': True
            }

        except Exception as e:
            print(f"✗ Error extracting treatments: {e}")
            return {
                'video_id': video_id,
                'success': False,
                'error': str(e)
            }

    def extract_all(self, video_id: int, min_confidence: Optional[float] = None) -> Dict[str, Any]:
        """
        Extract symptoms, diagnoses, treatments, and calculate concordance for a video.

        Args:
            video_id: Database ID of the video
            min_confidence: Minimum confidence for symptoms

        Returns:
            Combined results with concordance analysis
        """
        # Extract symptoms
        symptom_result = self.extract_symptoms(video_id, min_confidence)

        # Extract diagnoses
        diagnosis_result = self.extract_diagnoses(video_id)

        # Extract treatments
        treatment_result = self.extract_treatments(video_id)

        # Calculate concordance for each diagnosis
        concordance_results = []
        if diagnosis_result.get('success') and diagnosis_result.get('diagnosis_ids'):
            for diag_id in diagnosis_result['diagnosis_ids']:
                try:
                    concordance = calculate_symptom_concordance(video_id, diag_id, self.model)
                    concordance_results.append(concordance)
                    print(f"  Concordance for diagnosis {diag_id}: {concordance['concordance_score']:.2f} "
                          f"(core: {concordance['core_symptom_score']:.2f})")
                except Exception as e:
                    print(f"  Could not calculate concordance for {diag_id}: {e}")

            # Update comorbidity pairs if multiple diagnoses
            if len(diagnosis_result['diagnosis_ids']) >= 2:
                try:
                    update_comorbidity_pairs(video_id)
                    print(f"  Updated comorbidity tracking")
                except Exception as e:
                    print(f"  Could not update comorbidity: {e}")

        return {
            'video_id': video_id,
            'symptoms': symptom_result,
            'diagnoses': diagnosis_result,
            'treatments': treatment_result,
            'concordance': concordance_results,
            'success': symptom_result.get('success', False) and diagnosis_result.get('success', False)
        }

    def _call_model(self, prompt: str) -> str:
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return response.content[0].text.strip()

        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        message = payload.get("message", {})
        content = message.get("content")
        if not content:
            raise ValueError("Ollama response missing message content")
        return str(content).strip()

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
