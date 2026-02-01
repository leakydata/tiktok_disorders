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

import re

from config import (
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    TRANSCRIBER_BACKEND,
    WHISPER_COMPUTE_TYPE,
    ensure_directories,
)
from database import insert_transcript, get_video_by_id, get_transcript, insert_transcript_quality


# Medical vocabulary prompt to help Whisper recognize specialized terms
# This provides context about expected terminology
MEDICAL_VOCABULARY_PROMPT = """This is a video about chronic illness conditions including:
- Ehlers-Danlos Syndrome (EDS), hypermobile EDS (hEDS), classical EDS, vascular EDS
- Mast Cell Activation Syndrome (MCAS), mast cells, histamine, antihistamines
- Postural Orthostatic Tachycardia Syndrome (POTS), dysautonomia, orthostatic intolerance
- Chronic Inflammatory Response Syndrome (CIRS), biotoxin illness, mold illness
- SIBO (small intestinal bacterial overgrowth), Lyme disease, fibromyalgia, chronic fatigue

Medications commonly discussed:
- Biologics: Xolair (omalizumab), Rhapsido (remibrutinib), Dupixent (dupilumab)
- Mast cell stabilizers: cromolyn, Gastrocrom, ketotifen, Zaditen
- H1 antihistamines: cetirizine, Zyrtec, loratadine, Claritin, fexofenadine, Allegra, 
  diphenhydramine, Benadryl, hydroxyzine, Vistaril, Atarax
- H2 blockers: famotidine, Pepcid, ranitidine, Zantac
- POTS medications: fludrocortisone, Florinef, midodrine, pyridostigmine, Mestinon,
  ivabradine, Corlanor, droxidopa, Northera, propranolol, metoprolol
- GI medications: low dose naltrexone (LDN), omeprazole, Prilosec, ondansetron, Zofran
- Pain medications: gabapentin, Neurontin, pregabalin, Lyrica, cyclobenzaprine, Flexeril
- CIRS medications: cholestyramine, Questran, Welchol, VIP nasal spray
- Supplements: quercetin, DAO enzyme, magnesium glycinate, magnesium threonate, 
  methylcobalamin, B12, vitamin D, L-theanine, electrolytes, LMNT, Liquid IV

Medical devices and products:
- Compression: compression stockings, TED hose, Jobst, Sigvaris, abdominal binder
- Bracing: ring splints, silver ring splints, Oval-8, kinesiology tape, KT Tape
- Mobility: rollator, wheelchair, forearm crutches, SmartCrutch
- IV access: port-a-cath, PICC line, IV fluids, saline infusion
- Monitoring: pulse oximeter, Kardia, AliveCor, Apple Watch
- Air quality: HEPA filter, Austin Air, IQAir

Medical terms: interleukin, cytokines, tryptase, prostaglandin D2, collagen, connective tissue,
hypermobility, subluxation, proprioception, gastroparesis, syncope, presyncope, tachycardia,
bradycardia, orthostatic, Beighton score, tilt table test, brain fog, flare-up, comorbid.

Abbreviations: EDS, hEDS, MCAS, POTS, CIRS, CFS, ME, SIBO, LDN, DAO, IV, BP, HR, GI, PGD2."""


# Common transcription errors and their corrections
TRANSCRIPTION_CORRECTIONS = {
    # ==========================================================================
    # CONDITION NAME VARIATIONS
    # ==========================================================================
    
    # MCAS variations
    'mass cell': 'mast cell',
    'mass cells': 'mast cells',
    'mass cell activation': 'mast cell activation',
    'M-CAS': 'MCAS',
    'M CAS': 'MCAS',
    'em cass': 'MCAS',
    'emcass': 'MCAS',
    'MCA S': 'MCAS',
    'MKAS': 'MCAS',
    'm cast': 'MCAS',
    
    # EDS variations
    'E-D-S': 'EDS',
    'E.D.S.': 'EDS',
    'ehler danlos': 'Ehlers-Danlos',
    'ehlers danlos': 'Ehlers-Danlos',
    'ehler-danlos': 'Ehlers-Danlos',
    'eller danlos': 'Ehlers-Danlos',
    'ehlers-danlos': 'Ehlers-Danlos',
    'H-E-D-S': 'hEDS',
    'H.E.D.S.': 'hEDS',
    'h eds': 'hEDS',
    'h-eds': 'hEDS',
    'hyper mobile eds': 'hypermobile EDS',
    
    # POTS variations
    'P-O-T-S': 'POTS',
    'P.O.T.S.': 'POTS',
    'pots syndrome': 'POTS syndrome',
    'postural orthostatic tachycardia': 'Postural Orthostatic Tachycardia',
    
    # CIRS variations
    'C-I-R-S': 'CIRS',
    'C.I.R.S.': 'CIRS',
    'sirs': 'CIRS',
    
    # ==========================================================================
    # PRESCRIPTION MEDICATIONS
    # ==========================================================================
    
    # Biologics / Injectables
    'Zolair': 'Xolair',
    'zolair': 'Xolair',
    'zolaire': 'Xolair',
    'xolaire': 'Xolair',
    'omaluzimab': 'omalizumab',
    'omalizimab': 'omalizumab',
    'wrap seedo': 'Rhapsido',
    'wrap-seedo': 'Rhapsido',
    'rapsido': 'Rhapsido',
    'rap seedo': 'Rhapsido',
    'rapseedo': 'Rhapsido',
    'rhapseedo': 'Rhapsido',
    'rhapsody': 'Rhapsido',
    'rhapsido': 'Rhapsido',
    'remibrutinib': 'remibrutinib',
    'remibritinib': 'remibrutinib',
    'Dupixent': 'Dupixent',
    'dupixant': 'Dupixent',
    'dupilumab': 'dupilumab',
    'dupilimab': 'dupilumab',
    
    # Mast cell stabilizers
    'cromo lyn': 'cromolyn',
    'chromolyn': 'cromolyn',
    'gastrocrom': 'Gastrocrom',
    'gastro crom': 'Gastrocrom',
    'keto tifen': 'ketotifen',
    'ketitifen': 'ketotifen',
    'zaditen': 'Zaditen',
    'zaditin': 'Zaditen',
    
    # Antihistamines - H1 blockers
    'cet irizine': 'cetirizine',
    'cetirazine': 'cetirizine',
    'zyrtec': 'Zyrtec',
    'zirtec': 'Zyrtec',
    'lora tadine': 'loratadine',
    'loratidine': 'loratadine',
    'claritin': 'Claritin',
    'clariton': 'Claritin',
    'fexo fenadine': 'fexofenadine',
    'fexofenidine': 'fexofenadine',
    'allegra': 'Allegra',
    'alegra': 'Allegra',
    'diphen hydramine': 'diphenhydramine',
    'diphenhydromine': 'diphenhydramine',
    'benadryl': 'Benadryl',
    'benedryl': 'Benadryl',
    'hydroxyzine': 'hydroxyzine',
    'hydroxy zine': 'hydroxyzine',
    'hydroxazine': 'hydroxyzine',
    'vistaril': 'Vistaril',
    'visteril': 'Vistaril',
    'atarax': 'Atarax',
    'aterax': 'Atarax',
    
    # Antihistamines - H2 blockers
    'famota dine': 'famotidine',
    'famotadine': 'famotidine',
    'pepcid': 'Pepcid',
    'pepsid': 'Pepcid',
    'rani tidine': 'ranitidine',
    'ranitadine': 'ranitidine',
    'zantac': 'Zantac',
    'zantec': 'Zantac',
    
    # POTS medications
    'fluda cortisone': 'fludrocortisone',
    'fludracortisone': 'fludrocortisone',
    'florinef': 'Florinef',
    'floranef': 'Florinef',
    'mido drine': 'midodrine',
    'midodreen': 'midodrine',
    'proamatine': 'ProAmatine',
    'pro amatine': 'ProAmatine',
    'pyridostigmine': 'pyridostigmine',
    'pyridostigmin': 'pyridostigmine',
    'mestinon': 'Mestinon',
    'mestanon': 'Mestinon',
    'ivabradine': 'ivabradine',
    'ivabradeen': 'ivabradine',
    'corlanor': 'Corlanor',
    'corlaner': 'Corlanor',
    'droxidopa': 'droxidopa',
    'droxadopa': 'droxidopa',
    'northera': 'Northera',
    'northara': 'Northera',
    
    # Beta blockers
    'propranolol': 'propranolol',
    'propanolol': 'propranolol',
    'metoprolol': 'metoprolol',
    'metopralol': 'metoprolol',
    'atenolol': 'atenolol',
    'atenalol': 'atenolol',
    
    # GI medications
    'low dose naltrexone': 'low dose naltrexone',
    'LDN': 'LDN',
    'ldn': 'LDN',
    'omeprazole': 'omeprazole',
    'omeprazol': 'omeprazole',
    'prilosec': 'Prilosec',
    'prilozec': 'Prilosec',
    'pantoprazole': 'pantoprazole',
    'pantoprazol': 'pantoprazole',
    'protonix': 'Protonix',
    'protonex': 'Protonix',
    'ondansetron': 'ondansetron',
    'ondansetran': 'ondansetron',
    'zofran': 'Zofran',
    'zophran': 'Zofran',
    'dicyclomine': 'dicyclomine',
    'dicyclomin': 'dicyclomine',
    'bentyl': 'Bentyl',
    'bentil': 'Bentyl',
    
    # Pain / anti-inflammatory
    'gabapentin': 'gabapentin',
    'gabapentine': 'gabapentin',
    'neurontin': 'Neurontin',
    'neurotin': 'Neurontin',
    'pregabalin': 'pregabalin',
    'pregabaline': 'pregabalin',
    'lyrica': 'Lyrica',
    'lirica': 'Lyrica',
    'tramadol': 'tramadol',
    'tremadol': 'tramadol',
    'cyclobenzaprine': 'cyclobenzaprine',
    'cyclobenzapreen': 'cyclobenzaprine',
    'flexeril': 'Flexeril',
    'flexerol': 'Flexeril',
    
    # CIRS / mold medications
    'cholestyramine': 'cholestyramine',
    'colestyramine': 'cholestyramine',
    'questran': 'Questran',
    'questron': 'Questran',
    'welchol': 'Welchol',
    'welcol': 'Welchol',
    'colesevelam': 'colesevelam',
    'colesevelum': 'colesevelam',
    'VIP nasal spray': 'VIP nasal spray',
    'vasoactive intestinal peptide': 'vasoactive intestinal peptide',
    
    # ==========================================================================
    # SUPPLEMENTS
    # ==========================================================================
    
    'quer cetin': 'quercetin',
    'quercitin': 'quercetin',
    'quercetine': 'quercetin',
    'DAO enzyme': 'DAO enzyme',
    'dao enzyme': 'DAO enzyme',
    'diamine oxidase': 'diamine oxidase',
    'diamene oxidase': 'diamine oxidase',
    'vitamin d': 'vitamin D',
    'vitamin d3': 'vitamin D3',
    'b twelve': 'B12',
    'b-twelve': 'B12',
    'b 12': 'B12',
    'methyl b12': 'methyl B12',
    'methylcobalamin': 'methylcobalamin',
    'methyl cobalamin': 'methylcobalamin',
    'magnesium glycinate': 'magnesium glycinate',
    'magnesium glysinate': 'magnesium glycinate',
    'magnesium theronate': 'magnesium threonate',
    'magnesium threonate': 'magnesium threonate',
    'l-theanine': 'L-theanine',
    'l theanine': 'L-theanine',
    'theanine': 'L-theanine',
    'electrolytes': 'electrolytes',
    'electrolites': 'electrolytes',
    'LMNT': 'LMNT',
    'element electrolytes': 'LMNT',
    'liquid iv': 'Liquid IV',
    'liquid i.v.': 'Liquid IV',
    'drip drop': 'DripDrop',
    'nuun': 'Nuun',
    'noon tablets': 'Nuun tablets',
    
    # ==========================================================================
    # MEDICAL DEVICES & PRODUCTS
    # ==========================================================================
    
    # Compression
    'compression stockings': 'compression stockings',
    'compression socks': 'compression socks',
    'ted hose': 'TED hose',
    'jobst': 'Jobst',
    'jobest': 'Jobst',
    'sigvaris': 'Sigvaris',
    'sigvares': 'Sigvaris',
    'abdominal binder': 'abdominal binder',
    'abdominal compression': 'abdominal compression',
    
    # Bracing / support
    'ring splints': 'ring splints',
    'silver ring splints': 'silver ring splints',
    'oval 8': 'Oval-8',
    'oval eight': 'Oval-8',
    'kinesio tape': 'kinesiology tape',
    'kinesiology tape': 'kinesiology tape',
    'k tape': 'KT Tape',
    'kt tape': 'KT Tape',
    'rock tape': 'RockTape',
    
    # Mobility aids
    'rollator': 'rollator',
    'rolator': 'rollator',
    'wheelchair': 'wheelchair',
    'wheel chair': 'wheelchair',
    'walking cane': 'walking cane',
    'forearm crutches': 'forearm crutches',
    'smart crutch': 'SmartCrutch',
    
    # IV / infusion
    'IV fluids': 'IV fluids',
    'i.v. fluids': 'IV fluids',
    'saline infusion': 'saline infusion',
    'normal saline': 'normal saline',
    'lactated ringers': "lactated Ringer's",
    'ringers lactate': "Ringer's lactate",
    'port-a-cath': 'port-a-cath',
    'port a cath': 'port-a-cath',
    'PICC line': 'PICC line',
    'pick line': 'PICC line',
    'pic line': 'PICC line',
    
    # Monitoring devices
    'pulse oximeter': 'pulse oximeter',
    'pulse ox': 'pulse ox',
    'blood pressure cuff': 'blood pressure cuff',
    'bp cuff': 'BP cuff',
    'heart rate monitor': 'heart rate monitor',
    'apple watch': 'Apple Watch',
    'kardia': 'Kardia',
    'cardia': 'Kardia',
    'alivecor': 'AliveCor',
    
    # Allergy / environmental
    'air purifier': 'air purifier',
    'HEPA filter': 'HEPA filter',
    'hepa filter': 'HEPA filter',
    'austin air': 'Austin Air',
    'intellipure': 'Intellipure',
    'iqair': 'IQAir',
    
    # ==========================================================================
    # MEDICAL TERMS
    # ==========================================================================
    
    'inter leukin': 'interleukin',
    'interleuken': 'interleukin',
    'interlukin': 'interleukin',
    'cyto kines': 'cytokines',
    'cytokins': 'cytokines',
    'hista mine': 'histamine',
    'histomine': 'histamine',
    'anti histamine': 'antihistamine',
    'antihistomine': 'antihistamine',
    'dys autonomia': 'dysautonomia',
    'disautonomia': 'dysautonomia',
    'gastro paresis': 'gastroparesis',
    'gastro-paresis': 'gastroparesis',
    'sigh bo': 'SIBO',
    'see bo': 'SIBO',
    'si bo': 'SIBO',
    'S-I-B-O': 'SIBO',
    'sub luxation': 'subluxation',
    'sub-luxation': 'subluxation',
    'hyper mobility': 'hypermobility',
    'hyper-mobility': 'hypermobility',
    'hyper mobile': 'hypermobile',
    'hyper-mobile': 'hypermobile',
    'trip tase': 'tryptase',
    'triptase': 'tryptase',
    'pro prio ception': 'proprioception',
    'propriaception': 'proprioception',
    'sin cope': 'syncope',
    'syncopee': 'syncope',
    'pre syncope': 'presyncope',
    'pre-syncope': 'presyncope',
    'ortho static': 'orthostatic',
    'taky cardia': 'tachycardia',
    'tachy cardia': 'tachycardia',
    'brady cardia': 'bradycardia',
    'bradi cardia': 'bradycardia',
    'fibro myalgia': 'fibromyalgia',
    'fibromialgia': 'fibromyalgia',
    'my algic': 'myalgic',
    'myaligic': 'myalgic',
    'encephalo myelitis': 'encephalomyelitis',
    'encephalomialitis': 'encephalomyelitis',
    'chronic fatigue': 'chronic fatigue',
    'brain fog': 'brain fog',
    'brainfog': 'brain fog',
    'flare up': 'flare-up',
    'flareup': 'flare-up',
    'co morbid': 'comorbid',
    'co-morbid': 'comorbid',
    'comorbidity': 'comorbidity',
    'co morbidity': 'comorbidity',
    
    # Tests
    'tilt table test': 'tilt table test',
    'tilt table': 'tilt table',
    'triptase test': 'tryptase test',
    'prostaglandin d2': 'prostaglandin D2',
    'PGD2': 'PGD2',
    'n-methylhistamine': 'N-methylhistamine',
    'beighton score': 'Beighton score',
    'brighton score': 'Beighton score',
    'bayton score': 'Beighton score',
}


def _apply_transcription_corrections(text: str) -> str:
    """Apply medical terminology corrections to transcribed text."""
    corrected = text
    for wrong, right in TRANSCRIPTION_CORRECTIONS.items():
        # Case-insensitive replacement while preserving surrounding context
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        corrected = pattern.sub(right, corrected)
    return corrected


def _remove_repeated_phrases(text: str, min_phrase_words: int = 4, max_repeats: int = 2) -> str:
    """
    Remove repeated consecutive phrases/sentences from transcription.
    
    Whisper sometimes gets stuck in loops and repeats the same phrase many times.
    This function detects and collapses these repetitions.
    
    Args:
        text: The transcribed text
        min_phrase_words: Minimum words in a phrase to consider for deduplication
        max_repeats: Maximum times a phrase should appear consecutively
    
    Returns:
        Cleaned text with repetitions removed
    """
    if not text or len(text) < 50:
        return text
    
    # Split into sentences (by period, question mark, exclamation)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) < 3:
        return text
    
    # Remove consecutive duplicate sentences
    cleaned_sentences = []
    prev_sentence = None
    repeat_count = 0
    
    for sentence in sentences:
        # Normalize for comparison (lowercase, strip extra whitespace)
        normalized = ' '.join(sentence.lower().split())
        
        if normalized == prev_sentence:
            repeat_count += 1
            if repeat_count < max_repeats:
                cleaned_sentences.append(sentence)
        else:
            cleaned_sentences.append(sentence)
            prev_sentence = normalized
            repeat_count = 0
    
    result = ' '.join(cleaned_sentences)
    
    # Also detect repeated phrases within sentences
    # Look for patterns like "phrase phrase phrase phrase"
    words = result.split()
    if len(words) < min_phrase_words * 3:
        return result
    
    # Try different phrase lengths (4-10 words)
    for phrase_len in range(min_phrase_words, min(11, len(words) // 3)):
        i = 0
        new_words = []
        while i < len(words):
            phrase = ' '.join(words[i:i + phrase_len])
            
            # Count how many times this phrase repeats consecutively
            repeats = 1
            j = i + phrase_len
            while j + phrase_len <= len(words):
                next_phrase = ' '.join(words[j:j + phrase_len])
                if next_phrase.lower() == phrase.lower():
                    repeats += 1
                    j += phrase_len
                else:
                    break
            
            # If repeated more than max_repeats, collapse
            if repeats > max_repeats:
                # Keep only max_repeats instances
                for _ in range(max_repeats):
                    new_words.extend(words[i:i + phrase_len])
                i = j  # Skip all the repetitions
            else:
                new_words.append(words[i])
                i += 1
        
        words = new_words
    
    return ' '.join(words)


def _get_author_dir(author: str) -> Path:
    """Get or create a subdirectory for the author/username."""
    if not author:
        author = "_unknown"
    # Sanitize author name for folder
    safe_author = re.sub(r'[<>:"/\\|?*]', '_', author)
    safe_author = safe_author.strip().strip('.')  # Remove trailing dots/spaces
    if not safe_author:
        safe_author = "_unknown"
    author_dir = TRANSCRIPT_DIR / safe_author
    author_dir.mkdir(parents=True, exist_ok=True)
    return author_dir


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
            text = existing.get('text', '')
            return {
                'transcript_id': existing['id'],
                'text': text,
                'language': existing['language'],
                'word_count': existing.get('word_count') or len(text.split()),
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
                initial_prompt=MEDICAL_VOCABULARY_PROMPT,
                **kwargs,
            )

            segment_items = []
            text_parts = []
            for segment in segments_iter:
                # Apply corrections to each segment
                corrected_text = _apply_transcription_corrections(segment.text)
                corrected_text = _remove_repeated_phrases(corrected_text)
                text_parts.append(corrected_text)
                if save_segments:
                    segment_items.append({
                        'start': float(segment.start),
                        'end': float(segment.end),
                        'text': corrected_text
                    })

            text = " ".join(text_parts).strip()
            # Apply deduplication to full text (repetitions may span segments)
            text = _remove_repeated_phrases(text)
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
                'initial_prompt': MEDICAL_VOCABULARY_PROMPT,
                'condition_on_previous_text': True,
                'fp16': self.device == 'cuda',  # Use half precision on GPU for speed
                'verbose': True,
                **kwargs
            }

            result = self.model.transcribe(str(audio_path), **transcribe_options)

            # Extract results and apply corrections
            text = _apply_transcription_corrections(result['text'].strip())
            text = _remove_repeated_phrases(text)
            detected_language = result['language']
            
            # Apply corrections to segments too
            segments = None
            if save_segments and result.get('segments'):
                segments = []
                for seg in result['segments']:
                    seg_copy = dict(seg)
                    seg_copy['text'] = _apply_transcription_corrections(seg['text'])
                    seg_copy['text'] = _remove_repeated_phrases(seg_copy['text'])
                    segments.append(seg_copy)

        # Save to file in author subdirectory
        author = video.get('author') or '_unknown'
        author_dir = _get_author_dir(author)
        output_filename = f"transcript_{video_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path = author_dir / output_filename

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
