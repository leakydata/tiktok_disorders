"""
Database module for the TikTok Disorders Research Pipeline.
Handles PostgreSQL connections and CRUD operations for videos, transcripts, symptoms, and diagnoses.

Research-grade schema with:
- Data provenance tracking for reproducibility
- Enhanced symptom tracking with severity and temporal patterns
- Video metadata enrichment with engagement metrics
- Symptom co-occurrence tracking
- Claimed diagnosis tracking with concordance analysis
- Expected symptoms per condition for validation
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import json

from config import DATABASE_URL


# =============================================================================
# Expected symptoms for each condition (based on medical literature)
# Used for concordance analysis
# =============================================================================

CONDITION_EXPECTED_SYMPTOMS = {

    'EDS': {
        'name': 'Ehlers-Danlos Syndrome',
        'core_symptoms': [
            # Clinical terms
            'generalized joint hypermobility',
            'joint instability',
            'recurrent joint dislocations',
            'recurrent subluxations',
            'chronic musculoskeletal pain',
            'skin hyperextensibility',
            'easy bruising',
            'poor wound healing',
            'soft or velvety skin',
            # Colloquial terms (TikTok language)
            'hypermobility',
            'hypermobile',
            'dislocations',
            'subluxations',
            'chronic pain',
            'joint pain',
            'fatigue',
            'bruising'
        ],
        'common_symptoms': [
            'chronic fatigue',
            'exhaustion',
            'scoliosis',
            'flat feet',
            'pes planus',
            'recurrent sprains',
            'sprains',
            'hernias',
            'pelvic organ prolapse',
            'prolapse',
            'stretch marks',
            'unexplained striae',
            'bilateral piezogenic papules of the heel',
            'atrophic scarring',
            'dental crowding',
            'high or narrow palate',
            'arachnodactyly',
            'mitral valve prolapse',
            'aortic root dilatation',
            'digestive issues',
            'POTS symptoms',
            'anxiety'
        ],
        'categories': [
            'connective_tissue_structural',
            'musculoskeletal',
            'dermatological',
            'fatigue',
            'vascular_bleeding',
            'dental_tmj',
            'cardiovascular'
        ]
    },

    'MCAS': {
        'name': 'Mast Cell Activation Syndrome',
        'core_symptoms': [
            # Clinical terms
            'episodic flushing',
            'urticaria',
            'pruritus',
            'angioedema',
            'anaphylaxis',
            'abdominal pain',
            'diarrhea',
            'nausea',
            # Colloquial terms (TikTok language)
            'flushing',
            'hives',
            'itching',
            'swelling',
            'food reactions',
            'chemical sensitivity',
            'medication sensitivity',
            'allergic reactions'
        ],
        'common_symptoms': [
            'nasal congestion',
            'nasal pruritus',
            'wheezing',
            'tachycardia',
            'hypotension',
            'syncope',
            'headache',
            'throat swelling',
            'brain fog',
            'fatigue',
            'anxiety',
            'rapid heart rate',
            'fainting',
            'skin rashes',
            'rashes',
            'bladder pain',
            'bone pain'
        ],
        'categories': [
            'mast_cell_allergy_like',
            'immune_inflammation',
            'gastrointestinal',
            'respiratory',
            'cardiovascular',
            'ent',
            'neurological'
        ]
    },

    'POTS': {
        'name': 'Postural Orthostatic Tachycardia Syndrome',
        'core_symptoms': [
            # Clinical terms
            'excessive heart rate increase on standing',
            'orthostatic intolerance',
            'lightheadedness',
            'palpitations',
            'presyncope',
            # Colloquial terms (TikTok language)
            'tachycardia',
            'rapid heart rate',
            'elevated heart rate',
            'heart racing',
            'dizziness',
            'dizzy',
            'fainting',
            'near fainting',
            'exercise intolerance'
        ],
        'common_symptoms': [
            'fatigue',
            'exhaustion',
            'tremulousness',
            'tremor',
            'shaking',
            'generalized weakness',
            'weakness',
            'blurred vision',
            'nausea',
            'cold extremities',
            'cold hands',
            'cold feet',
            'sleep disturbance',
            'migraine headaches',
            'brain fog',
            'blood pooling',
            'sweating',
            'adrenaline surges'
        ],
        'categories': [
            'autonomic',
            'dysautonomia',
            'cardiovascular',
            'orthostatic_intolerance',
            'fatigue',
            'sleep',
            'neurological'
        ]
    },

    'FIBROMYALGIA': {
        'name': 'Fibromyalgia',
        'core_symptoms': [
            'widespread musculoskeletal pain',
            'chronic pain',
            'fatigue',
            'non-restorative sleep',
            'cognitive dysfunction'
        ],
        'common_symptoms': [
            'headache',
            'waking unrefreshed',
            'lower abdominal pain or cramps',
            'irritable bowel symptoms',
            'morning stiffness',
            'paresthesias',
            'depression'
        ],
        'categories': [
            'fibromyalgia',
            'musculoskeletal',
            'pain_characterization',
            'fatigue',
            'sleep',
            'cognitive',
            'neurological',
            'gastrointestinal',
            'affective'
        ]
    },

    'CFS': {
        'name': 'Myalgic Encephalomyelitis / Chronic Fatigue Syndrome',
        'core_symptoms': [
            'substantial reduction in functional capacity',
            'post-exertional malaise',
            'unrefreshing sleep',
            'cognitive impairment',
            'orthostatic intolerance'
        ],
        'common_symptoms': [
            'muscle pain',
            'joint pain without swelling or redness',
            'headaches of a new type, pattern, or severity',
            'sore throat',
            'tender lymph nodes',
            'chills',
            'night sweats',
            'visual disturbances',
            'sensitivity to light',
            'sensitivity to sound',
            'nausea',
            'dizziness'
        ],
        'categories': [
            'fatigue',
            'functional_capacity',
            'neurological',
            'sleep',
            'orthostatic_intolerance',
            'sensory_overload',
            'lymphatic',
            'gastrointestinal'
        ]
    },

    'CIRS': {
        'name': 'Chronic Inflammatory Response Syndrome',
        'core_symptoms': [
            'chronic fatigue',
            'cognitive impairment',
            'headache',
            'myalgia'
        ],
        'common_symptoms': [
            'sinus congestion',
            'cough',
            'dyspnea',
            'abdominal pain',
            'diarrhea',
            'light sensitivity'
        ],
        'categories': [
            'immune_inflammation',
            'fatigue',
            'cognitive',
            'respiratory',
            'gastrointestinal'
        ]
    },

    'GASTROPARESIS': {
        'name': 'Gastroparesis',
        'core_symptoms': [
            'nausea',
            'vomiting',
            'early satiety',
            'bloating',
            'abdominal pain',
            'feeling full',
            'delayed gastric emptying'
        ],
        'common_symptoms': [
            'heartburn',
            'acid reflux',
            'weight loss',
            'malnutrition',
            'blood sugar fluctuations',
            'appetite loss',
            'stomach pain',
            'constipation',
            'food sitting in stomach'
        ],
        'categories': [
            'gastrointestinal'
        ]
    },

    'SIBO': {
        'name': 'Small Intestinal Bacterial Overgrowth',
        'core_symptoms': [
            'bloating',
            'abdominal distension',
            'diarrhea',
            'constipation',
            'abdominal pain',
            'gas',
            'flatulence',
            'nausea'
        ],
        'common_symptoms': [
            'fatigue',
            'brain fog',
            'weight loss',
            'malnutrition',
            'food intolerances',
            'joint pain',
            'skin issues',
            'vitamin deficiencies',
            'belching'
        ],
        'categories': [
            'gastrointestinal'
        ]
    }

}


@contextmanager
def get_connection():
    """Get a database connection as a context manager."""
    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Initialize the database schema with research-grade tables."""
    with get_connection() as conn:
        cur = conn.cursor()

        # =============================================================================
        # Core Tables
        # =============================================================================

        # Videos table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id SERIAL PRIMARY KEY,
                url TEXT UNIQUE NOT NULL,
                platform TEXT NOT NULL,
                video_id TEXT NOT NULL,
                title TEXT,
                author TEXT,
                author_id TEXT,
                author_follower_count INTEGER,
                duration INTEGER,
                upload_date DATE,
                tags TEXT[],
                hashtags TEXT[],
                description TEXT,
                audio_path TEXT,
                audio_size_bytes BIGINT,

                -- Engagement metrics
                view_count BIGINT,
                like_count BIGINT,
                comment_count BIGINT,
                share_count BIGINT,

                -- Data provenance
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                collection_method TEXT DEFAULT 'yt-dlp',
                collection_version TEXT,

                -- Research metadata
                is_verified_creator BOOLEAN,
                content_warning TEXT,
                research_notes TEXT,
                
                -- Creator influence tier (for STRAIN social contagion analysis)
                creator_tier TEXT CHECK (creator_tier IN ('nano', 'micro', 'mid', 'macro', 'mega')),
                -- nano: <10K, micro: 10-100K, mid: 100K-500K, macro: 500K-1M, mega: >1M

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Transcripts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                text TEXT NOT NULL,
                original_text TEXT DEFAULT NULL,  -- Stores original if cleaned (for revert)
                language TEXT,
                language_confidence REAL,
                model_used TEXT,
                model_backend TEXT,
                model_compute_type TEXT,
                transcription_device TEXT,
                word_count INTEGER,
                original_word_count INTEGER DEFAULT NULL,  -- Original word count if cleaned
                audio_duration_seconds REAL,
                words_per_minute REAL,
                segments JSONB,
                transcribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time_seconds REAL,
                song_lyrics_ratio REAL DEFAULT NULL,
                extracted_at TIMESTAMP DEFAULT NULL,
                cleaned_at TIMESTAMP DEFAULT NULL,  -- When repetitions were removed
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(video_id)
            )
        """)
        
        # Add columns for existing databases
        cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS original_text TEXT DEFAULT NULL")
        cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS original_word_count INTEGER DEFAULT NULL")
        cur.execute("ALTER TABLE transcripts ADD COLUMN IF NOT EXISTS cleaned_at TIMESTAMP DEFAULT NULL")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_extracted_at ON transcripts(extracted_at)")

        # Symptoms table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptoms (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                symptom TEXT NOT NULL,
                confidence REAL NOT NULL,
                context TEXT,
                severity TEXT CHECK (severity IN ('mild', 'moderate', 'severe', 'unspecified')),
                temporal_pattern TEXT CHECK (temporal_pattern IN ('acute', 'chronic', 'intermittent', 'progressive', 'unspecified')),
                body_location TEXT,
                triggers TEXT[],
                is_personal_experience BOOLEAN DEFAULT TRUE,
                extractor_model TEXT,
                extractor_provider TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =============================================================================
        # Diagnosis Tracking Tables (NEW)
        # =============================================================================

        # Claimed diagnoses - what conditions the speaker claims to have
        cur.execute("""
            CREATE TABLE IF NOT EXISTS claimed_diagnoses (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                condition_code TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                context TEXT,
                
                -- Diagnosis status (more nuanced than boolean)
                diagnosis_status TEXT CHECK (diagnosis_status IN (
                    'confirmed', 'self_diagnosed', 'suspected', 'clinical', 
                    'genetic', 'seeking', 'lost', 'unclear'
                )),
                is_self_diagnosed BOOLEAN,  -- Keep for backward compatibility
                diagnosis_date_mentioned TEXT,
                
                -- EDS-specific subtype
                eds_subtype TEXT CHECK (eds_subtype IN (
                    'hEDS', 'vEDS', 'cEDS', 'clEDS', 'kEDS', 'HSD', NULL
                )),
                
                -- Enhanced fields
                diagnosing_specialty TEXT,  -- geneticist, rheumatologist, cardiologist, PCP, allergist, self, etc.
                sentiment TEXT CHECK (sentiment IN (
                    'validated', 'frustrated', 'relieved', 'questioning', 'neutral', NULL
                )),
                mentioned_with TEXT[],  -- Other conditions mentioned in same context
                
                extractor_model TEXT,
                extractor_provider TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Expected symptoms reference table - symptoms expected for each condition
        cur.execute("""
            CREATE TABLE IF NOT EXISTS expected_symptoms (
                id SERIAL PRIMARY KEY,
                condition_code TEXT NOT NULL,
                symptom TEXT NOT NULL,
                is_core_symptom BOOLEAN DEFAULT FALSE,
                category TEXT,
                source TEXT DEFAULT 'medical_literature',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(condition_code, symptom)
            )
        """)

        # Symptom concordance - how well reported symptoms match expected symptoms
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptom_concordance (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                diagnosis_id INTEGER REFERENCES claimed_diagnoses(id) ON DELETE CASCADE,

                -- Concordance metrics
                total_symptoms_reported INTEGER DEFAULT 0,
                expected_symptoms_matched INTEGER DEFAULT 0,
                core_symptoms_matched INTEGER DEFAULT 0,
                unexpected_symptoms_count INTEGER DEFAULT 0,

                -- Calculated scores
                concordance_score REAL,
                core_symptom_score REAL,

                -- Lists for analysis
                matched_symptoms TEXT[],
                unmatched_expected TEXT[],
                unexpected_symptoms TEXT[],

                -- Analysis metadata
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analyzer_model TEXT,

                UNIQUE(video_id, diagnosis_id)
            )
        """)

        # =============================================================================
        # Research Enhancement Tables
        # =============================================================================

        # Symptom co-occurrence
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptom_cooccurrence (
                id SERIAL PRIMARY KEY,
                symptom_a_id INTEGER REFERENCES symptoms(id) ON DELETE CASCADE,
                symptom_b_id INTEGER REFERENCES symptoms(id) ON DELETE CASCADE,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                temporal_proximity_seconds REAL,
                mentioned_together BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symptom_a_id, symptom_b_id)
            )
        """)

        # Engagement snapshots
        cur.execute("""
            CREATE TABLE IF NOT EXISTS engagement_snapshots (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                view_count BIGINT,
                like_count BIGINT,
                comment_count BIGINT,
                share_count BIGINT,
                snapshot_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =============================================================================
        # User Longitudinal Tracking Tables
        # =============================================================================
        
        # User profiles - aggregated view per creator
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id SERIAL PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                first_video_date DATE,
                last_video_date DATE,
                video_count INTEGER DEFAULT 0,
                total_symptoms_reported INTEGER DEFAULT 0,
                unique_diagnoses_count INTEGER DEFAULT 0,
                avg_concordance_score REAL,
                avg_core_symptom_score REAL,
                primary_condition TEXT,
                follower_count BIGINT,
                
                -- STRAIN indicators aggregated
                self_diagnosis_ratio REAL,
                professional_diagnosis_ratio REAL,
                doctor_dismissal_mentions INTEGER DEFAULT 0,
                stress_trigger_mentions INTEGER DEFAULT 0,
                
                -- Flags for research
                is_flagged_low_concordance BOOLEAN DEFAULT FALSE,
                is_flagged_inconsistent_symptoms BOOLEAN DEFAULT FALSE,
                research_notes TEXT,
                
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles_username ON user_profiles(username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles_primary_condition ON user_profiles(primary_condition)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_user_profiles_concordance ON user_profiles(avg_concordance_score)")

        # Diagnosis timeline - when each user first claimed each diagnosis
        cur.execute("""
            CREATE TABLE IF NOT EXISTS diagnosis_timeline (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                condition_code TEXT NOT NULL,
                first_mentioned DATE NOT NULL,
                last_mentioned DATE,
                mention_count INTEGER DEFAULT 1,
                was_self_diagnosed BOOLEAN,
                diagnosis_order INTEGER,
                days_since_previous_diagnosis INTEGER,
                previous_diagnosis TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, condition_code)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_timeline_username ON diagnosis_timeline(username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_diagnosis_timeline_condition ON diagnosis_timeline(condition_code)")

        # Symptom consistency tracking
        cur.execute("""
            CREATE TABLE IF NOT EXISTS symptom_consistency (
                id SERIAL PRIMARY KEY,
                username TEXT NOT NULL,
                symptom TEXT NOT NULL,
                category TEXT,
                total_mentions INTEGER DEFAULT 1,
                severity_variations INTEGER DEFAULT 1,
                all_severities TEXT[],
                first_mentioned DATE,
                last_mentioned DATE,
                avg_confidence REAL,
                is_inconsistent BOOLEAN DEFAULT FALSE,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(username, symptom)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptom_consistency_username ON symptom_consistency(username)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptom_consistency_inconsistent ON symptom_consistency(is_inconsistent)")

        # Processing runs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS processing_runs (
                id SERIAL PRIMARY KEY,
                run_type TEXT NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                videos_processed INTEGER DEFAULT 0,
                transcripts_created INTEGER DEFAULT 0,
                symptoms_extracted INTEGER DEFAULT 0,
                diagnoses_extracted INTEGER DEFAULT 0,
                concordance_analyzed INTEGER DEFAULT 0,
                errors JSONB,
                config_snapshot JSONB,
                notes TEXT
            )
        """)

        # Annotations
        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                symptom_id INTEGER REFERENCES symptoms(id) ON DELETE CASCADE,
                diagnosis_id INTEGER REFERENCES claimed_diagnoses(id) ON DELETE CASCADE,
                annotation_type TEXT NOT NULL,
                annotation_value TEXT,
                annotator TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (video_id IS NOT NULL OR symptom_id IS NOT NULL OR diagnosis_id IS NOT NULL)
            )
        """)

        # =============================================================================
        # Comorbidity Tracking (NEW)
        # =============================================================================

        # Track which conditions appear together across videos
        cur.execute("""
            CREATE TABLE IF NOT EXISTS comorbidity_pairs (
                id SERIAL PRIMARY KEY,
                condition_a TEXT NOT NULL,
                condition_b TEXT NOT NULL,
                video_count INTEGER DEFAULT 1,
                avg_concordance_a REAL,
                avg_concordance_b REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(condition_a, condition_b),
                CHECK (condition_a < condition_b)
            )
        """)

        # =============================================================================
        # Treatment/Medication Tracking (NEW)
        # =============================================================================

        cur.execute("""
            CREATE TABLE IF NOT EXISTS treatments (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                treatment_type TEXT NOT NULL CHECK (treatment_type IN ('medication', 'supplement', 'therapy', 'lifestyle', 'procedure', 'device', 'other')),
                treatment_name TEXT NOT NULL,
                dosage TEXT,
                frequency TEXT,
                effectiveness TEXT CHECK (effectiveness IN ('very_helpful', 'somewhat_helpful', 'not_helpful', 'made_worse', 'unspecified')),
                side_effects TEXT[],
                is_current BOOLEAN,
                target_condition TEXT,
                target_symptoms TEXT[],
                context TEXT,
                confidence REAL DEFAULT 0.5,
                extractor_model TEXT,
                extractor_provider TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # =============================================================================
        # STRAIN Framework Tables - Narrative and Social Context Analysis
        # =============================================================================

        # Narrative elements for STRAIN validation
        cur.execute("""
            CREATE TABLE IF NOT EXISTS narrative_elements (
                id SERIAL PRIMARY KEY,
                video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
                
                -- Content type classification
                content_type TEXT CHECK (content_type IN (
                    'personal_story', 'educational', 'advice_giving', 
                    'awareness_advocacy', 'product_promotion', 'vent_rant', 'other'
                )),
                
                -- Diagnostic journey indicators (STRAIN-relevant)
                mentions_self_diagnosis BOOLEAN,
                mentions_professional_diagnosis BOOLEAN,
                mentions_negative_testing BOOLEAN,
                mentions_doctor_dismissal BOOLEAN,
                mentions_medical_gaslighting BOOLEAN,
                mentions_long_diagnostic_journey BOOLEAN,
                mentions_multiple_doctors BOOLEAN,
                years_to_diagnosis_mentioned INTEGER,
                
                -- Stress-symptom relationship (STRAIN core feature)
                mentions_stress_triggers BOOLEAN,
                mentions_symptom_flares BOOLEAN,
                mentions_symptom_migration BOOLEAN,
                
                -- Social/community context (STRAIN feature)
                mentions_online_community BOOLEAN,
                mentions_other_creators BOOLEAN,
                mentions_learning_from_tiktok BOOLEAN,
                cites_medical_sources BOOLEAN,
                
                -- Authority claims
                claims_healthcare_background BOOLEAN,
                claims_expert_knowledge BOOLEAN,
                
                -- Illness identity indicators
                uses_condition_as_identity BOOLEAN,
                mentions_chronic_illness_community BOOLEAN,
                
                -- Key extracted quotes
                diagnostic_journey_quotes TEXT[],
                stress_trigger_quotes TEXT[],
                
                confidence REAL DEFAULT 0.5,
                extractor_model TEXT,
                extractor_provider TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                UNIQUE(video_id)
            )
        """)

        # =============================================================================
        # Transcript Quality Metrics (NEW)
        # =============================================================================

        cur.execute("""
            CREATE TABLE IF NOT EXISTS transcript_quality (
                id SERIAL PRIMARY KEY,
                transcript_id INTEGER REFERENCES transcripts(id) ON DELETE CASCADE,
                quality_score REAL,
                clarity_score REAL,
                completeness_score REAL,
                medical_term_density REAL,
                filler_word_ratio REAL,
                avg_confidence REAL,
                low_confidence_segments INTEGER,
                total_segments INTEGER,
                issues TEXT[],
                assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(transcript_id)
            )
        """)

        # =============================================================================
        # Pipeline Progress Tracking (NEW - for resumable runs)
        # =============================================================================

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_progress (
                id SERIAL PRIMARY KEY,
                run_id INTEGER REFERENCES processing_runs(id) ON DELETE CASCADE,
                url TEXT NOT NULL,
                stage TEXT NOT NULL CHECK (stage IN ('queued', 'downloading', 'downloaded', 'transcribing', 'transcribed', 'extracting', 'completed', 'failed')),
                video_id INTEGER REFERENCES videos(id) ON DELETE SET NULL,
                error_message TEXT,
                retry_count INTEGER DEFAULT 0,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, url)
            )
        """)

        # =============================================================================
        # Indexes
        # =============================================================================

        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_url ON videos(url)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_platform ON videos(platform)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_author ON videos(author)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_video_id ON transcripts(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_video_id ON symptoms(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_category ON symptoms(category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_symptoms_confidence ON symptoms(confidence)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_video_id ON claimed_diagnoses(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_diagnoses_condition ON claimed_diagnoses(condition_code)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_expected_condition ON expected_symptoms(condition_code)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_concordance_video ON symptom_concordance(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_concordance_diagnosis ON symptom_concordance(diagnosis_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_treatments_video ON treatments(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_treatments_type ON treatments(treatment_type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comorbidity_conditions ON comorbidity_pairs(condition_a, condition_b)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_progress_run ON pipeline_progress(run_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pipeline_progress_stage ON pipeline_progress(stage)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcript_quality ON transcript_quality(transcript_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_narrative_elements_video ON narrative_elements(video_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_videos_creator_tier ON videos(creator_tier)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_song_lyrics_ratio ON transcripts(song_lyrics_ratio)")

        conn.commit()
        print("Database schema initialized successfully")

        # Populate expected symptoms reference table
        _populate_expected_symptoms(cur)
        conn.commit()
        print("Expected symptoms reference data populated")


def _populate_expected_symptoms(cur):
    """Populate the expected_symptoms table with medical reference data."""
    for condition_code, data in CONDITION_EXPECTED_SYMPTOMS.items():
        # Insert core symptoms
        for symptom in data['core_symptoms']:
            cur.execute("""
                INSERT INTO expected_symptoms (condition_code, symptom, is_core_symptom, category)
                VALUES (%s, %s, TRUE, %s)
                ON CONFLICT (condition_code, symptom) DO NOTHING
            """, (condition_code, symptom.lower(), data['categories'][0] if data['categories'] else None))

        # Insert common symptoms
        for symptom in data['common_symptoms']:
            cur.execute("""
                INSERT INTO expected_symptoms (condition_code, symptom, is_core_symptom, category)
                VALUES (%s, %s, FALSE, NULL)
                ON CONFLICT (condition_code, symptom) DO NOTHING
            """, (condition_code, symptom.lower()))


# =============================================================================
# Video Operations
# =============================================================================

def insert_video(url: str, platform: str, video_id: str, metadata: Dict[str, Any]) -> int:
    """Insert a video record into the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO videos (
                url, platform, video_id, title, author, author_id, author_follower_count,
                duration, upload_date, tags, hashtags, description, audio_path, audio_size_bytes,
                view_count, like_count, comment_count, share_count,
                collection_method, collection_version, is_verified_creator, research_notes,
                creator_tier
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO UPDATE SET
                title = COALESCE(EXCLUDED.title, videos.title),
                author = COALESCE(EXCLUDED.author, videos.author),
                duration = COALESCE(EXCLUDED.duration, videos.duration),
                audio_path = COALESCE(EXCLUDED.audio_path, videos.audio_path),
                creator_tier = COALESCE(EXCLUDED.creator_tier, videos.creator_tier),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            url, platform, video_id,
            metadata.get('title'), metadata.get('author'), metadata.get('author_id'),
            metadata.get('author_follower_count'), metadata.get('duration'),
            metadata.get('upload_date'), metadata.get('tags', []),
            metadata.get('hashtags', []), metadata.get('description'),
            metadata.get('audio_path'), metadata.get('audio_size_bytes'),
            metadata.get('view_count'), metadata.get('like_count'),
            metadata.get('comment_count'), metadata.get('share_count'),
            metadata.get('collection_method', 'yt-dlp'), metadata.get('collection_version'),
            metadata.get('is_verified_creator'), metadata.get('research_notes'),
            metadata.get('creator_tier')
        ))
        return cur.fetchone()[0]


def get_video_by_url(url: str) -> Optional[Dict[str, Any]]:
    """Get a video by its URL."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE url = %s", (url,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_video_by_id(video_id: int) -> Optional[Dict[str, Any]]:
    """Get a video by its database ID."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos WHERE id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_all_videos() -> List[Dict[str, Any]]:
    """Get all videos from the database."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM videos ORDER BY collected_at DESC")
        return [dict(row) for row in cur.fetchall()]


def get_all_videos_with_transcripts() -> List[Dict[str, Any]]:
    """Get all videos that have transcripts."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT v.*, t.id as transcript_id, t.text as transcript_text, t.word_count
            FROM videos v
            JOIN transcripts t ON v.id = t.video_id
            ORDER BY v.collected_at DESC
        """)
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Transcript Operations
# =============================================================================

def insert_transcript(video_id: int, text: str, language: Optional[str] = None,
                     model_used: Optional[str] = None, segments: Optional[List] = None,
                     **kwargs) -> int:
    """Insert a transcript record with full provenance tracking."""
    word_count = len(text.split()) if text else 0
    audio_duration = kwargs.get('audio_duration_seconds')
    wpm = (word_count / (audio_duration / 60)) if audio_duration and audio_duration > 0 else None

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcripts (
                video_id, text, language, language_confidence,
                model_used, model_backend, model_compute_type, transcription_device,
                word_count, audio_duration_seconds, words_per_minute,
                segments, processing_time_seconds
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                text = EXCLUDED.text,
                language = EXCLUDED.language,
                model_used = EXCLUDED.model_used,
                word_count = EXCLUDED.word_count,
                transcribed_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            video_id, text, language, kwargs.get('language_confidence'),
            model_used, kwargs.get('model_backend'), kwargs.get('model_compute_type'),
            kwargs.get('transcription_device'), word_count, audio_duration, wpm,
            json.dumps(segments) if segments else None, kwargs.get('processing_time_seconds')
        ))
        return cur.fetchone()[0]


def get_transcript(video_id: int) -> Optional[Dict[str, Any]]:
    """Get transcript for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM transcripts WHERE video_id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def update_transcript_song_lyrics_ratio(video_id: int, song_lyrics_ratio: float) -> bool:
    """Update the song_lyrics_ratio for a transcript.
    
    Args:
        video_id: The video ID
        song_lyrics_ratio: Float 0.0-1.0 (0=pure spoken, 1=pure lyrics)
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE transcripts
            SET song_lyrics_ratio = %s
            WHERE video_id = %s
            RETURNING id
        """, (song_lyrics_ratio, video_id))
        result = cur.fetchone()
        return result is not None


def mark_transcript_extracted(video_id: int) -> bool:
    """Mark a transcript as having been processed for extraction.
    
    This prevents re-processing videos that yielded zero symptoms.
    
    Args:
        video_id: The video ID
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE transcripts
            SET extracted_at = CURRENT_TIMESTAMP
            WHERE video_id = %s
            RETURNING id
        """, (video_id,))
        result = cur.fetchone()
        return result is not None


def clear_transcript_extracted(video_id: int) -> bool:
    """Clear the extracted_at timestamp to allow re-extraction.
    
    Args:
        video_id: The video ID
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE transcripts
            SET extracted_at = NULL
            WHERE video_id = %s
            RETURNING id
        """, (video_id,))
        result = cur.fetchone()
        return result is not None


def get_transcripts_needing_song_check(limit: int = None) -> List[Dict[str, Any]]:
    """Get transcripts that haven't been checked for song lyrics (ratio is NULL).
    
    Returns original_text if available (for cleaned transcripts) so song detection
    can use the pre-cleaned text with repetitions intact.
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        query = """
            SELECT t.id, t.video_id, t.text, t.original_text, t.word_count, v.title, v.author
            FROM transcripts t
            JOIN videos v ON t.video_id = v.id
            WHERE t.song_lyrics_ratio IS NULL
            ORDER BY t.id
        """
        if limit:
            query += f" LIMIT {limit}"
        cur.execute(query)
        return [dict(row) for row in cur.fetchall()]


def get_song_lyrics_stats() -> Dict[str, Any]:
    """Get statistics about song lyrics detection (ratio-based)."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE song_lyrics_ratio IS NULL) as unchecked,
                COUNT(*) FILTER (WHERE song_lyrics_ratio IS NOT NULL) as checked,
                AVG(song_lyrics_ratio) FILTER (WHERE song_lyrics_ratio IS NOT NULL) as avg_ratio,
                COUNT(*) FILTER (WHERE song_lyrics_ratio < 0.2) as pure_spoken,
                COUNT(*) FILTER (WHERE song_lyrics_ratio >= 0.2 AND song_lyrics_ratio < 0.5) as mostly_spoken,
                COUNT(*) FILTER (WHERE song_lyrics_ratio >= 0.5 AND song_lyrics_ratio < 0.8) as mixed,
                COUNT(*) FILTER (WHERE song_lyrics_ratio >= 0.8) as mostly_lyrics
            FROM transcripts
        """)
        return dict(cur.fetchone())


# =============================================================================
# Value Normalization (LLM outputs -> valid database values)
# =============================================================================

# Temporal pattern normalization
TEMPORAL_PATTERN_MAP = {
    'acute': 'acute',
    'acute (transient)': 'acute',
    'transient': 'acute',
    'sudden': 'acute',
    'chronic': 'chronic',
    'long-term': 'chronic',
    'persistent': 'chronic',
    'ongoing': 'chronic',
    'constant': 'chronic',
    'intermittent': 'intermittent',
    'episodic': 'intermittent',
    'fluctuating': 'intermittent',
    'comes and goes': 'intermittent',
    'flares': 'intermittent',
    'progressive': 'progressive',
    'worsening': 'progressive',
    'degenerative': 'progressive',
    'getting worse': 'progressive',
    'unspecified': 'unspecified',
    'unknown': 'unspecified',
    'unclear': 'unspecified',
    '': 'unspecified',
    None: 'unspecified',
}

# Diagnosis status normalization
DIAGNOSIS_STATUS_MAP = {
    'confirmed': 'confirmed',
    'diagnosed': 'confirmed',
    'professionally diagnosed': 'confirmed',
    'doctor diagnosed': 'confirmed',
    'official': 'confirmed',
    'self_diagnosed': 'self_diagnosed',
    'self-diagnosed': 'self_diagnosed',
    'self diagnosed': 'self_diagnosed',
    'suspected': 'suspected',
    'possible': 'suspected',
    'likely': 'suspected',
    'probably': 'suspected',
    'clinical': 'clinical',
    'clinically diagnosed': 'clinical',
    'genetic': 'genetic',
    'genetically confirmed': 'genetic',
    'seeking': 'seeking',
    'pursuing': 'seeking',
    'trying to get diagnosed': 'seeking',
    'waiting for diagnosis': 'seeking',
    'lost': 'lost',
    'dismissed': 'lost',
    'misdiagnosed': 'lost',
    'reversed': 'lost',
    'unclear': 'unclear',
    'unknown': 'unclear',
    'unspecified': 'unclear',
    '': 'unclear',
    None: 'unclear',
}

# EDS subtype normalization
EDS_SUBTYPE_MAP = {
    'heds': 'hEDS',
    'h-eds': 'hEDS',
    'hypermobile': 'hEDS',
    'hypermobile eds': 'hEDS',
    'hypermobile type': 'hEDS',
    'veds': 'vEDS',
    'v-eds': 'vEDS',
    'vascular': 'vEDS',
    'vascular eds': 'vEDS',
    'vascular type': 'vEDS',
    'ceds': 'cEDS',
    'c-eds': 'cEDS',
    'classical': 'cEDS',
    'classical eds': 'cEDS',
    'classical type': 'cEDS',
    'cleds': 'clEDS',
    'classical-like': 'clEDS',
    'keds': 'kEDS',
    'k-eds': 'kEDS',
    'kyphoscoliotic': 'kEDS',
    'hsd': 'HSD',
    'hypermobility spectrum': 'HSD',
    'hypermobility spectrum disorder': 'HSD',
    'unspecified': None,
    'unknown': None,
    'unclear': None,
    '': None,
    None: None,
}

# Severity normalization
SEVERITY_MAP = {
    'mild': 'mild',
    'slight': 'mild',
    'minor': 'mild',
    'moderate': 'moderate',
    'medium': 'moderate',
    'severe': 'severe',
    'extreme': 'severe',
    'debilitating': 'severe',
    'intense': 'severe',
    'unspecified': 'unspecified',
    'unknown': 'unspecified',
    'unclear': 'unspecified',
    '': 'unspecified',
    None: 'unspecified',
}

# Sentiment normalization
SENTIMENT_MAP = {
    'validated': 'validated',
    'relieved': 'relieved',
    'frustrated': 'frustrated',
    'angry': 'frustrated',
    'upset': 'frustrated',
    'questioning': 'questioning',
    'uncertain': 'questioning',
    'doubtful': 'questioning',
    'neutral': 'neutral',
    'unspecified': 'neutral',
    '': 'neutral',
    None: 'neutral',
}


def _normalize_value(value: Any, mapping: Dict, default: Any) -> Any:
    """Normalize a value using a mapping, with fallback to default."""
    if value is None:
        return mapping.get(None, default)
    
    # Try exact match first
    value_lower = str(value).lower().strip()
    if value_lower in mapping:
        return mapping[value_lower]
    
    # Try partial match
    for key, mapped_value in mapping.items():
        if key and value_lower in str(key).lower():
            return mapped_value
    
    return default


# =============================================================================
# Symptom Operations
# =============================================================================

def insert_symptom(video_id: int, category: str, symptom: str,
                  confidence: float, context: Optional[str] = None,
                  **kwargs) -> int:
    """Insert a symptom record with enhanced tracking."""
    # Normalize values to valid database constraints
    severity = _normalize_value(kwargs.get('severity'), SEVERITY_MAP, 'unspecified')
    temporal_pattern = _normalize_value(kwargs.get('temporal_pattern'), TEMPORAL_PATTERN_MAP, 'unspecified')
    
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO symptoms (
                video_id, category, symptom, confidence, context,
                severity, temporal_pattern, body_location, triggers,
                is_personal_experience, extractor_model, extractor_provider
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            video_id, category, symptom, confidence, context,
            severity,
            temporal_pattern,
            kwargs.get('body_location'), kwargs.get('triggers', []),
            kwargs.get('is_personal_experience', True),
            kwargs.get('extractor_model'), kwargs.get('extractor_provider')
        ))
        return cur.fetchone()[0]


def get_symptoms_by_video(video_id: int) -> List[Dict[str, Any]]:
    """Get all symptoms for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM symptoms WHERE video_id = %s ORDER BY confidence DESC", (video_id,))
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Diagnosis Operations (NEW)
# =============================================================================

def insert_claimed_diagnosis(video_id: int, condition_code: str, condition_name: str,
                            confidence: float, context: Optional[str] = None,
                            **kwargs) -> int:
    """Insert a claimed diagnosis record with enhanced fields."""
    # Normalize values to valid database constraints
    diagnosis_status = _normalize_value(kwargs.get('diagnosis_status'), DIAGNOSIS_STATUS_MAP, 'unclear')
    eds_subtype = _normalize_value(kwargs.get('eds_subtype'), EDS_SUBTYPE_MAP, None)
    sentiment = _normalize_value(kwargs.get('sentiment'), SENTIMENT_MAP, 'neutral')
    
    # Map is_self_diagnosed to diagnosis_status for backward compatibility
    is_self_diagnosed = kwargs.get('is_self_diagnosed')
    if kwargs.get('diagnosis_status') is None and is_self_diagnosed is not None:
        diagnosis_status = 'self_diagnosed' if is_self_diagnosed else 'confirmed'
    
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO claimed_diagnoses (
                video_id, condition_code, condition_name, confidence, context,
                diagnosis_status, is_self_diagnosed, diagnosis_date_mentioned,
                eds_subtype, diagnosing_specialty, sentiment, mentioned_with,
                extractor_model, extractor_provider
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            video_id, condition_code.upper(), condition_name, confidence, context,
            diagnosis_status,
            is_self_diagnosed,
            kwargs.get('diagnosis_date_mentioned'),
            eds_subtype,
            kwargs.get('diagnosing_specialty'),
            sentiment,
            kwargs.get('mentioned_with', []),
            kwargs.get('extractor_model'),
            kwargs.get('extractor_provider')
        ))
        return cur.fetchone()[0]


def get_diagnoses_by_video(video_id: int) -> List[Dict[str, Any]]:
    """Get all claimed diagnoses for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM claimed_diagnoses
            WHERE video_id = %s
            ORDER BY confidence DESC
        """, (video_id,))
        return [dict(row) for row in cur.fetchall()]


def get_expected_symptoms(condition_code: str) -> Dict[str, List[str]]:
    """Get expected symptoms for a condition."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT symptom, is_core_symptom
            FROM expected_symptoms
            WHERE condition_code = %s
        """, (condition_code.upper(),))
        results = cur.fetchall()

        core = [r['symptom'] for r in results if r['is_core_symptom']]
        common = [r['symptom'] for r in results if not r['is_core_symptom']]

        return {'core': core, 'common': common, 'all': core + common}


# =============================================================================
# Concordance Analysis (NEW)
# =============================================================================

# Symptom synonym mappings for better concordance matching
# Aligned with extractor.py SYMPTOM_CATEGORIES for the EDS/MCAS/POTS space
SYMPTOM_SYNONYMS = {
    # === MUSCULOSKELETAL & HYPERMOBILITY ===
    'hypermobility': ['joint laxity', 'loose joints', 'bendy', 'flexible', 'hyperextension', 'beighton'],
    'joint': ['joints', 'articulation'],
    'subluxation': ['subluxations', 'partial dislocation', 'slipping', 'unstable joint'],
    'dislocation': ['dislocations', 'joint dislocation', 'popping out'],
    'instability': ['unstable', 'laxity', 'loose', 'hypermobile'],
    'joint pain': ['arthralgia', 'joints hurt', 'joint ache', 'painful joints'],
    'muscle spasm': ['muscle spasms', 'cramping', 'muscle cramps', 'spasms'],
    
    # === CRANIOCERVICAL ===
    'cci': ['craniocervical instability', 'cranial instability'],
    'aai': ['atlantoaxial instability', 'c1 c2'],
    'chiari': ['chiari malformation', 'brain herniation', 'cerebellar'],
    'tethered cord': ['tethered spinal cord', 'filum terminale'],
    
    # === CARDIOVASCULAR & POTS ===
    'tachycardia': ['rapid heart rate', 'heart racing', 'fast heart', 'high heart rate'],
    'heart racing': ['rapid heart rate', 'tachycardia', 'palpitations', 'heart pounding'],
    'palpitations': ['heart racing', 'rapid heart rate', 'heart pounding', 'heart flutter'],
    'blood pressure': ['bp', 'hypotension', 'hypertension', 'low bp', 'high bp'],
    'chest pain': ['chest tightness', 'chest pressure', 'angina'],
    
    # === ORTHOSTATIC / POTS ===
    'dizziness': ['dizzy', 'lightheaded', 'lightheadedness', 'vertigo', 'woozy'],
    'lightheadedness': ['lightheaded', 'dizziness', 'dizzy', 'faint feeling'],
    'fainting': ['syncope', 'faint', 'passed out', 'blackout', 'loss of consciousness'],
    'presyncope': ['near fainting', 'almost fainted', 'graying out', 'near syncope'],
    'blood pooling': ['venous pooling', 'pooling', 'legs turn purple', 'mottled'],
    'orthostatic': ['standing', 'postural', 'position change', 'upright'],
    'pots': ['postural tachycardia', 'orthostatic tachycardia'],
    
    # === AUTONOMIC ===
    'dysautonomia': ['autonomic dysfunction', 'autonomic', 'ans dysfunction'],
    'adrenaline': ['epinephrine', 'adrenaline surge', 'hyperadrenergic', 'fight or flight'],
    
    # === THERMOREGULATION ===
    'raynaud': ['raynauds', 'fingers turn white', 'fingers turn blue', 'cold fingers'],
    'temperature': ['thermoregulation', 'heat intolerance', 'cold intolerance', 'overheating'],
    'sweating': ['hyperhidrosis', 'night sweats', 'excessive sweating'],
    
    # === GASTROINTESTINAL ===
    'gastrointestinal': ['abdominal pain', 'stomach', 'nausea', 'diarrhea', 'constipation', 'bloating', 'gi', 'digestive'],
    'stomach': ['gastrointestinal', 'abdominal', 'nausea', 'gi issues', 'digestive', 'gastric'],
    'gastroparesis': ['delayed gastric emptying', 'stomach emptying', 'gp', 'motility'],
    'motility': ['dysmotility', 'gi motility', 'slow motility', 'gastroparesis'],
    'sibo': ['small intestinal bacterial overgrowth', 'bacterial overgrowth'],
    'reflux': ['gerd', 'acid reflux', 'heartburn'],
    'nausea': ['nauseous', 'queasy', 'sick to stomach'],
    'bloating': ['bloated', 'distension', 'abdominal distension', 'pregnant belly'],
    'food sensitivity': ['food reactions', 'food intolerance', 'food allergy', 'dietary', 'food triggers'],
    'food intolerance': ['food reactions', 'food sensitivity', 'food allergy', 'cannot tolerate'],
    'malabsorption': ['not absorbing', 'nutrient deficiency', 'cannot absorb'],
    
    # === MAST CELL / ALLERGIC ===
    'flushing': ['facial flushing', 'face flushing', 'skin flushing', 'red face', 'flush'],
    'facial flushing': ['flushing', 'skin flushing', 'face on fire'],
    'hives': ['urticaria', 'skin rash', 'welts', 'raised welts'],
    'itching': ['itchy', 'pruritus', 'skin itching'],
    'swelling': ['angioedema', 'edema', 'swollen', 'inflammation'],
    'anaphylaxis': ['anaphylactic', 'severe allergic', 'anaphylactoid'],
    'histamine': ['histamine reaction', 'histamine intolerance', 'mast cell'],
    'mcas': ['mast cell activation', 'mast cell', 'mast cell disease'],
    'chemical sensitivity': ['mcs', 'fragrance sensitivity', 'chemical intolerance', 'smell sensitivity'],
    'medication sensitivity': ['drug sensitivity', 'medication reactions', 'med sensitivity'],
    'reaction': ['allergic reaction', 'mast cell reaction', 'histamine reaction', 'reacting'],
    'dermatographia': ['skin writing', 'dermatographic'],
    
    # === DERMATOLOGICAL ===
    'bruising': ['bruise', 'bruises', 'easy bruising', 'contusion'],
    'stretch marks': ['striae', 'skin marks'],
    'scarring': ['scars', 'abnormal scars', 'atrophic scars', 'poor healing'],
    'skin fragility': ['thin skin', 'fragile skin', 'skin tears'],
    
    # === RESPIRATORY ===
    'wheezing': ['wheeze', 'asthma', 'bronchospasm'],
    'breathing': ['dyspnea', 'shortness of breath', 'air hunger', 'breathless'],
    'throat': ['throat tightness', 'globus', 'lump in throat'],
    
    # === SINUS / ENT ===
    'sinus': ['nasal congestion', 'sinusitis', 'sinus congestion', 'nasal', 'sinuses'],
    'sinusitis': ['sinus', 'nasal congestion', 'chronic sinusitis', 'sinus infection'],
    'nasal congestion': ['sinus', 'stuffy nose', 'blocked nose', 'congestion'],
    'rhinitis': ['nasal congestion', 'runny nose', 'sinus', 'nasal inflammation'],
    'tinnitus': ['ear ringing', 'ringing ears', 'ears ringing'],
    'ear': ['ears', 'ear fullness', 'ear pain', 'eustachian'],
    'post nasal drip': ['drainage', 'nasal drip', 'constant drainage'],
    
    # === NEUROLOGICAL ===
    'headache': ['headaches', 'migraine', 'head pain', 'cephalalgia'],
    'migraine': ['headache', 'headaches', 'migraines'],
    'neuropathy': ['nerve pain', 'peripheral neuropathy', 'small fiber neuropathy', 'sfn'],
    'tingling': ['paresthesia', 'pins and needles', 'numbness', 'prickling'],
    'numbness': ['numb', 'loss of sensation', 'tingling'],
    'burning': ['burning pain', 'burning sensation', 'nerves on fire'],
    'tremor': ['tremors', 'shaking', 'trembling'],
    
    # === COGNITIVE ===
    'brain fog': ['cognitive dysfunction', 'mental fog', 'foggy', 'confusion', 'concentration', 'cognitive'],
    'memory': ['memory issues', 'forgetful', 'memory problems', 'forgetting'],
    'concentration': ['focus', 'attention', 'cannot concentrate', 'distracted'],
    'word finding': ['word finding difficulty', 'losing words', 'cannot find words'],
    'dissociation': ['dissociating', 'dpdr', 'depersonalization', 'derealization'],
    
    # === FATIGUE & PEM ===
    'fatigue': ['exhaustion', 'tired', 'low energy', 'chronic fatigue', 'fatigued', 'wiped out'],
    'exhaustion': ['fatigue', 'tired', 'wiped out', 'depleted'],
    'pem': ['post exertional malaise', 'crash', 'payback', 'post exertional'],
    'crash': ['crashes', 'crashing', 'pem', 'flare'],
    'flare': ['flares', 'flare up', 'flaring', 'symptom flare'],
    
    # === SLEEP ===
    'insomnia': ['cannot sleep', 'sleep issues', 'sleep problems'],
    'sleep': ['sleeping', 'rest', 'unrefreshing sleep'],
    
    # === PAIN ===
    'pain': ['ache', 'hurts', 'painful', 'chronic pain', 'aching'],
    'widespread pain': ['all over pain', 'total body pain', 'diffuse pain', 'everywhere hurts'],
    'allodynia': ['pain from touch', 'touch hurts', 'hypersensitive'],
    'hyperalgesia': ['increased pain', 'pain amplification', 'extra sensitive'],
    
    # === BLADDER / UROLOGICAL ===
    'bladder': ['urinary', 'interstitial cystitis', 'bladder pain', 'ic', 'bladder issues'],
    'interstitial cystitis': ['bladder pain', 'bladder issues', 'ic', 'painful bladder'],
    'urgency': ['urinary urgency', 'frequent urination', 'always peeing'],
    
    # === GYNECOLOGIC ===
    'menstrual': ['period', 'periods', 'menstruation', 'cycle'],
    'endometriosis': ['endo', 'endometrial'],
    'pelvic pain': ['pelvic', 'lower abdominal pain'],
    'pcos': ['polycystic ovarian', 'polycystic ovary'],
    
    # === OCULAR ===
    'dry eyes': ['eye dryness', 'eyes dry'],
    'photophobia': ['light sensitivity', 'light sensitive', 'bright lights hurt'],
    'blurry vision': ['blurred vision', 'vision problems', 'cannot focus'],
    
    # === DENTAL / TMJ ===
    'tmj': ['tmd', 'jaw pain', 'temporomandibular', 'jaw'],
    'jaw': ['jaw pain', 'tmj', 'jaw clicking', 'jaw locking'],
    'bruxism': ['teeth grinding', 'grinding teeth', 'clenching'],
}

# Condition names that should NOT be counted as symptoms
# (Sometimes the LLM extracts the diagnosis itself as a symptom)
CONDITION_NAME_PATTERNS = {
    'mast cell activation', 'mcas', 'mast cell disease', 'mast cell disorder',
    'ehlers danlos', 'eds', 'hypermobility syndrome', 'heds', 'veds', 'ceds',
    'pots', 'postural orthostatic tachycardia', 'postural tachycardia',
    'dysautonomia', 'autonomic dysfunction',
    'fibromyalgia', 'fibro',
    'chronic fatigue syndrome', 'cfs', 'me/cfs', 'mecfs',
    'gastroparesis',  # This is both a condition and symptom - keep for now
    'endometriosis', 'endo',
    'interstitial cystitis',  # Both condition and symptom
    'chiari', 'chiari malformation',
    'craniocervical instability', 'cci',
    'atlantoaxial instability', 'aai',
    'small fiber neuropathy', 'sfn',
    'long covid', 'post covid',
}


def _is_condition_name(symptom: str) -> bool:
    """Check if the 'symptom' is actually a condition name, not a symptom."""
    symptom_lower = symptom.lower().strip()
    for pattern in CONDITION_NAME_PATTERNS:
        if pattern in symptom_lower or symptom_lower in pattern:
            # But allow if it includes actual symptom descriptors
            symptom_words = set(symptom_lower.split())
            symptom_descriptors = {'pain', 'ache', 'fatigue', 'tired', 'dizzy', 
                                   'nausea', 'flare', 'reaction', 'episode'}
            if symptom_words & symptom_descriptors:
                return False  # Has symptom-like words, keep it
            return True
    return False


def _symptom_matches(reported: str, expected: str) -> bool:
    """Check if a reported symptom matches an expected symptom using fuzzy matching."""
    reported = reported.lower().strip()
    expected = expected.lower().strip()
    
    # Direct substring match
    if expected in reported or reported in expected:
        return True
    
    # Word overlap - if any significant word matches
    reported_words = set(reported.replace('-', ' ').replace('/', ' ').split())
    expected_words = set(expected.replace('-', ' ').replace('/', ' ').split())
    
    # Remove common filler words
    filler = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'with', 'from', 'issues', 'problems'}
    reported_words -= filler
    expected_words -= filler
    
    # If any significant word matches
    if reported_words & expected_words:
        return True
    
    # Check synonyms
    for word in reported_words:
        if word in SYMPTOM_SYNONYMS:
            for syn in SYMPTOM_SYNONYMS[word]:
                if syn in expected or expected in syn:
                    return True
                if set(syn.split()) & expected_words:
                    return True
    
    # Check if expected has synonyms that match reported
    for word in expected_words:
        if word in SYMPTOM_SYNONYMS:
            for syn in SYMPTOM_SYNONYMS[word]:
                if syn in reported or reported in syn:
                    return True
    
    return False


def calculate_symptom_concordance(video_id: int, diagnosis_id: int,
                                  analyzer_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculate how well reported symptoms match expected symptoms for a diagnosis.

    Returns concordance metrics and saves to database.
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get the diagnosis
        cur.execute("SELECT * FROM claimed_diagnoses WHERE id = %s", (diagnosis_id,))
        diagnosis = cur.fetchone()
        if not diagnosis:
            raise ValueError(f"Diagnosis {diagnosis_id} not found")

        condition_code = diagnosis['condition_code']

        # Get reported symptoms for this video
        cur.execute("""
            SELECT LOWER(symptom) as symptom, category, confidence
            FROM symptoms
            WHERE video_id = %s AND is_personal_experience = TRUE
        """, (video_id,))
        reported_symptoms = [r['symptom'] for r in cur.fetchall()]

        # Get expected symptoms for this condition
        expected = get_expected_symptoms(condition_code)
        all_expected = set(expected['all'])
        core_expected = set(expected['core'])

        # Filter out condition names from reported symptoms
        # (LLM sometimes extracts the diagnosis itself as a symptom)
        reported_set = {s for s in set(reported_symptoms) if not _is_condition_name(s)}

        matched = []
        for reported in reported_set:
            for exp in all_expected:
                if _symptom_matches(reported, exp):
                    matched.append(reported)
                    break

        matched_set = set(matched)
        core_matched = []
        for m in matched_set:
            for core in core_expected:
                if _symptom_matches(m, core):
                    core_matched.append(m)
                    break

        # Unexpected symptoms (reported but not in expected list)
        unexpected = [s for s in reported_set if s not in matched_set]

        # Calculate scores
        total_reported = len(reported_set)
        matched_count = len(matched_set)
        core_matched_count = len(set(core_matched))

        concordance_score = matched_count / total_reported if total_reported > 0 else 0
        core_score = core_matched_count / len(core_expected) if core_expected else 0

        # Unmatched expected (expected but not reported)
        unmatched_expected = list(all_expected - matched_set)

        # Save to database
        cur.execute("""
            INSERT INTO symptom_concordance (
                video_id, diagnosis_id,
                total_symptoms_reported, expected_symptoms_matched, core_symptoms_matched,
                unexpected_symptoms_count, concordance_score, core_symptom_score,
                matched_symptoms, unmatched_expected, unexpected_symptoms, analyzer_model
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id, diagnosis_id) DO UPDATE SET
                total_symptoms_reported = EXCLUDED.total_symptoms_reported,
                expected_symptoms_matched = EXCLUDED.expected_symptoms_matched,
                core_symptoms_matched = EXCLUDED.core_symptoms_matched,
                unexpected_symptoms_count = EXCLUDED.unexpected_symptoms_count,
                concordance_score = EXCLUDED.concordance_score,
                core_symptom_score = EXCLUDED.core_symptom_score,
                matched_symptoms = EXCLUDED.matched_symptoms,
                unmatched_expected = EXCLUDED.unmatched_expected,
                unexpected_symptoms = EXCLUDED.unexpected_symptoms,
                analyzed_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            video_id, diagnosis_id,
            total_reported, matched_count, core_matched_count,
            len(unexpected), concordance_score, core_score,
            list(matched_set), unmatched_expected[:20], unexpected[:20],
            analyzer_model
        ))

        conn.commit()

        return {
            'video_id': video_id,
            'diagnosis_id': diagnosis_id,
            'condition': condition_code,
            'total_symptoms_reported': total_reported,
            'expected_matched': matched_count,
            'core_matched': core_matched_count,
            'unexpected_count': len(unexpected),
            'concordance_score': round(concordance_score, 3),
            'core_symptom_score': round(core_score, 3),
            'matched_symptoms': list(matched_set),
            'missing_core_symptoms': [s for s in core_expected if s not in str(matched_set)],
            'unexpected_symptoms': unexpected[:10]
        }


def get_concordance_summary(video_id: int) -> List[Dict[str, Any]]:
    """Get concordance analysis summary for all diagnoses in a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                sc.*,
                cd.condition_code,
                cd.condition_name,
                cd.is_self_diagnosed
            FROM symptom_concordance sc
            JOIN claimed_diagnoses cd ON sc.diagnosis_id = cd.id
            WHERE sc.video_id = %s
            ORDER BY sc.concordance_score DESC
        """, (video_id,))
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Statistics
# =============================================================================

def get_symptom_statistics() -> Dict[str, Any]:
    """Get comprehensive statistics about extracted symptoms and diagnoses."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("SELECT COUNT(*) as count FROM videos")
        total_videos = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM transcripts")
        total_transcripts = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM symptoms")
        total_symptoms = cur.fetchone()['count']

        cur.execute("SELECT COUNT(*) as count FROM claimed_diagnoses")
        total_diagnoses = cur.fetchone()['count']

        # Diagnoses by condition
        cur.execute("""
            SELECT condition_code, COUNT(*) as count
            FROM claimed_diagnoses
            GROUP BY condition_code
            ORDER BY count DESC
        """)
        by_condition = [dict(row) for row in cur.fetchall()]

        # Average concordance by condition
        cur.execute("""
            SELECT
                cd.condition_code,
                AVG(sc.concordance_score) as avg_concordance,
                AVG(sc.core_symptom_score) as avg_core_score,
                COUNT(*) as sample_size
            FROM symptom_concordance sc
            JOIN claimed_diagnoses cd ON sc.diagnosis_id = cd.id
            GROUP BY cd.condition_code
            ORDER BY avg_concordance DESC
        """)
        concordance_by_condition = [dict(row) for row in cur.fetchall()]

        # Symptoms by category
        cur.execute("""
            SELECT category, COUNT(*) as count, AVG(confidence) as avg_confidence
            FROM symptoms
            GROUP BY category
            ORDER BY count DESC
        """)
        by_category = [dict(row) for row in cur.fetchall()]

        return {
            'total_videos': total_videos,
            'total_transcripts': total_transcripts,
            'total_symptoms': total_symptoms,
            'total_diagnoses': total_diagnoses,
            'diagnoses_by_condition': by_condition,
            'concordance_by_condition': concordance_by_condition,
            'symptoms_by_category': by_category
        }


def get_cooccurrence_matrix(min_occurrences: int = 2) -> List[Dict[str, Any]]:
    """Get symptom co-occurrence data for network analysis."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT
                s1.symptom as symptom_a,
                s1.category as category_a,
                s2.symptom as symptom_b,
                s2.category as category_b,
                COUNT(*) as cooccurrence_count
            FROM symptom_cooccurrence sc
            JOIN symptoms s1 ON sc.symptom_a_id = s1.id
            JOIN symptoms s2 ON sc.symptom_b_id = s2.id
            GROUP BY s1.symptom, s1.category, s2.symptom, s2.category
            HAVING COUNT(*) >= %s
            ORDER BY cooccurrence_count DESC
        """, (min_occurrences,))
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Utility functions
# =============================================================================

def insert_symptom_cooccurrence(symptom_a_id: int, symptom_b_id: int, video_id: int,
                                temporal_proximity_seconds: Optional[float] = None,
                                mentioned_together: bool = False) -> int:
    """Record symptom co-occurrence."""
    if symptom_a_id > symptom_b_id:
        symptom_a_id, symptom_b_id = symptom_b_id, symptom_a_id

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO symptom_cooccurrence (symptom_a_id, symptom_b_id, video_id,
                                             temporal_proximity_seconds, mentioned_together)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (symptom_a_id, symptom_b_id) DO UPDATE SET
                mentioned_together = EXCLUDED.mentioned_together
            RETURNING id
        """, (symptom_a_id, symptom_b_id, video_id, temporal_proximity_seconds, mentioned_together))
        return cur.fetchone()[0]


def start_processing_run(run_type: str, config_snapshot: Optional[Dict] = None,
                        notes: Optional[str] = None) -> int:
    """Start tracking a new processing run."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO processing_runs (run_type, config_snapshot, notes)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (run_type, json.dumps(config_snapshot) if config_snapshot else None, notes))
        return cur.fetchone()[0]


def complete_processing_run(run_id: int, **kwargs):
    """Mark a processing run as complete with statistics."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE processing_runs
            SET completed_at = CURRENT_TIMESTAMP,
                videos_processed = %s,
                transcripts_created = %s,
                symptoms_extracted = %s,
                diagnoses_extracted = %s,
                concordance_analyzed = %s,
                errors = %s
            WHERE id = %s
        """, (
            kwargs.get('videos_processed', 0),
            kwargs.get('transcripts_created', 0),
            kwargs.get('symptoms_extracted', 0),
            kwargs.get('diagnoses_extracted', 0),
            kwargs.get('concordance_analyzed', 0),
            json.dumps(kwargs.get('errors')) if kwargs.get('errors') else None,
            run_id
        ))


def add_annotation(annotation_type: str, annotation_value: str,
                  video_id: Optional[int] = None, symptom_id: Optional[int] = None,
                  diagnosis_id: Optional[int] = None, annotator: Optional[str] = None) -> int:
    """Add a research annotation."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO annotations (video_id, symptom_id, diagnosis_id, annotation_type, annotation_value, annotator)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (video_id, symptom_id, diagnosis_id, annotation_type, annotation_value, annotator))
        return cur.fetchone()[0]


def save_engagement_snapshot(video_id: int) -> int:
    """Save engagement metrics snapshot."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO engagement_snapshots (video_id, view_count, like_count, comment_count, share_count)
            SELECT id, view_count, like_count, comment_count, share_count
            FROM videos WHERE id = %s
            RETURNING id
        """, (video_id,))
        result = cur.fetchone()
        return result[0] if result else None


# =============================================================================
# Duplicate Detection (NEW)
# =============================================================================

def check_duplicate_video(url: str = None, video_id: str = None, title: str = None,
                          author: str = None, duration: int = None) -> Optional[Dict[str, Any]]:
    """
    Check if a video already exists using multiple criteria.

    Returns the existing video if found, None otherwise.
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Exact URL match
        if url:
            cur.execute("SELECT * FROM videos WHERE url = %s", (url,))
            result = cur.fetchone()
            if result:
                return {'match_type': 'exact_url', 'video': dict(result)}

        # Platform video ID match
        if video_id:
            cur.execute("SELECT * FROM videos WHERE video_id = %s", (video_id,))
            result = cur.fetchone()
            if result:
                return {'match_type': 'video_id', 'video': dict(result)}

        # Fuzzy match: same author + same duration (within 2 seconds) + similar title
        if author and duration and title:
            cur.execute("""
                SELECT * FROM videos
                WHERE author = %s
                AND ABS(duration - %s) <= 2
                AND title IS NOT NULL
            """, (author, duration))
            results = cur.fetchall()

            for row in results:
                # Simple similarity check - if titles share 70% of words
                existing_words = set(row['title'].lower().split()) if row['title'] else set()
                new_words = set(title.lower().split())
                if existing_words and new_words:
                    overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
                    if overlap > 0.7:
                        return {'match_type': 'fuzzy', 'video': dict(row), 'similarity': overlap}

        return None


def get_potential_duplicates(min_similarity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Find potential duplicate videos in the database.

    Returns groups of potentially duplicate videos.
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Find videos with same author and similar duration
        cur.execute("""
            SELECT v1.id as id1, v2.id as id2,
                   v1.title as title1, v2.title as title2,
                   v1.author, v1.duration as dur1, v2.duration as dur2,
                   v1.url as url1, v2.url as url2
            FROM videos v1
            JOIN videos v2 ON v1.author = v2.author
                          AND v1.id < v2.id
                          AND ABS(v1.duration - v2.duration) <= 2
            WHERE v1.author IS NOT NULL
        """)

        potential_dupes = []
        for row in cur.fetchall():
            # Calculate title similarity
            words1 = set(row['title1'].lower().split()) if row['title1'] else set()
            words2 = set(row['title2'].lower().split()) if row['title2'] else set()

            if words1 and words2:
                similarity = len(words1 & words2) / max(len(words1), len(words2))
                if similarity >= min_similarity:
                    potential_dupes.append({
                        'video_ids': [row['id1'], row['id2']],
                        'author': row['author'],
                        'titles': [row['title1'], row['title2']],
                        'urls': [row['url1'], row['url2']],
                        'similarity': round(similarity, 3)
                    })

        return potential_dupes


# =============================================================================
# Treatment Operations (NEW)
# =============================================================================

def insert_treatment(video_id: int, treatment_type: str, treatment_name: str,
                    **kwargs) -> int:
    """Insert a treatment/medication record."""
    
    # Normalize treatment_type to allowed values
    VALID_TREATMENT_TYPES = {'medication', 'supplement', 'therapy', 'lifestyle', 'procedure', 'device', 'other'}
    TYPE_MAPPINGS = {
        'diet': 'lifestyle',
        'food': 'lifestyle',
        'exercise': 'lifestyle',
        'sleep': 'lifestyle',
        'rest': 'lifestyle',
        'avoidance': 'lifestyle',
        'elimination': 'lifestyle',
        'drug': 'medication',
        'prescription': 'medication',
        'otc': 'medication',
        'vitamin': 'supplement',
        'mineral': 'supplement',
        'herb': 'supplement',
        'herbal': 'supplement',
        'physical_therapy': 'therapy',
        'pt': 'therapy',
        'counseling': 'therapy',
        'cbt': 'therapy',
        'surgery': 'procedure',
        'injection': 'procedure',
        'infusion': 'procedure',
        'iv': 'procedure',
        'brace': 'device',
        'compression': 'device',
        'mobility_aid': 'device',
    }
    
    normalized_type = treatment_type.lower().strip() if treatment_type else 'other'
    if normalized_type in TYPE_MAPPINGS:
        normalized_type = TYPE_MAPPINGS[normalized_type]
    elif normalized_type not in VALID_TREATMENT_TYPES:
        normalized_type = 'other'
    
    # Normalize effectiveness to allowed values
    VALID_EFFECTIVENESS = {'very_helpful', 'somewhat_helpful', 'not_helpful', 'made_worse', 'unspecified'}
    EFFECTIVENESS_MAPPINGS = {
        'helpful': 'somewhat_helpful',
        'effective': 'somewhat_helpful',
        'works': 'somewhat_helpful',
        'good': 'somewhat_helpful',
        'great': 'very_helpful',
        'excellent': 'very_helpful',
        'amazing': 'very_helpful',
        'life_changing': 'very_helpful',
        'lifesaver': 'very_helpful',
        'bad': 'not_helpful',
        'useless': 'not_helpful',
        'didnt_help': 'not_helpful',
        'no_effect': 'not_helpful',
        'worse': 'made_worse',
        'harmful': 'made_worse',
        'flared': 'made_worse',
        'flared_harder': 'made_worse',
        'triggered': 'made_worse',
        'reaction': 'made_worse',
        'side_effects': 'made_worse',
        'unknown': 'unspecified',
        'unsure': 'unspecified',
        'mixed': 'unspecified',
        'varies': 'unspecified',
        None: 'unspecified',
        '': 'unspecified',
    }
    
    raw_effectiveness = kwargs.get('effectiveness', 'unspecified')
    if raw_effectiveness:
        normalized_effectiveness = raw_effectiveness.lower().strip()
    else:
        normalized_effectiveness = 'unspecified'
    
    if normalized_effectiveness in EFFECTIVENESS_MAPPINGS:
        normalized_effectiveness = EFFECTIVENESS_MAPPINGS[normalized_effectiveness]
    elif normalized_effectiveness not in VALID_EFFECTIVENESS:
        normalized_effectiveness = 'unspecified'
    
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO treatments (
                video_id, treatment_type, treatment_name, dosage, frequency,
                effectiveness, side_effects, is_current, target_condition,
                target_symptoms, context, confidence, extractor_model, extractor_provider
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            video_id, normalized_type, treatment_name,
            kwargs.get('dosage'), kwargs.get('frequency'),
            normalized_effectiveness,
            kwargs.get('side_effects', []),
            kwargs.get('is_current'),
            kwargs.get('target_condition'),
            kwargs.get('target_symptoms', []),
            kwargs.get('context'),
            kwargs.get('confidence', 0.5),
            kwargs.get('extractor_model'),
            kwargs.get('extractor_provider')
        ))
        return cur.fetchone()[0]


def get_treatments_by_video(video_id: int) -> List[Dict[str, Any]]:
    """Get all treatments mentioned in a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM treatments
            WHERE video_id = %s
            ORDER BY confidence DESC
        """, (video_id,))
        return [dict(row) for row in cur.fetchall()]


def get_treatment_statistics() -> Dict[str, Any]:
    """Get statistics about treatments across all videos."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Most mentioned treatments
        cur.execute("""
            SELECT treatment_name, treatment_type, COUNT(*) as mention_count,
                   AVG(CASE WHEN effectiveness = 'very_helpful' THEN 1.0
                            WHEN effectiveness = 'somewhat_helpful' THEN 0.66
                            WHEN effectiveness = 'not_helpful' THEN 0.33
                            WHEN effectiveness = 'made_worse' THEN 0.0
                            ELSE NULL END) as avg_effectiveness
            FROM treatments
            GROUP BY treatment_name, treatment_type
            ORDER BY mention_count DESC
            LIMIT 50
        """)
        top_treatments = [dict(row) for row in cur.fetchall()]

        # By type
        cur.execute("""
            SELECT treatment_type, COUNT(*) as count
            FROM treatments
            GROUP BY treatment_type
            ORDER BY count DESC
        """)
        by_type = [dict(row) for row in cur.fetchall()]

        # By effectiveness
        cur.execute("""
            SELECT effectiveness, COUNT(*) as count
            FROM treatments
            WHERE effectiveness IS NOT NULL
            GROUP BY effectiveness
            ORDER BY count DESC
        """)
        by_effectiveness = [dict(row) for row in cur.fetchall()]

        return {
            'top_treatments': top_treatments,
            'by_type': by_type,
            'by_effectiveness': by_effectiveness
        }


# =============================================================================
# Comorbidity Operations (NEW)
# =============================================================================

def update_comorbidity_pairs(video_id: int):
    """
    Update comorbidity pairs based on diagnoses in a video.
    Call this after extracting diagnoses.
    """
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Get all diagnoses for this video
        cur.execute("""
            SELECT condition_code FROM claimed_diagnoses
            WHERE video_id = %s AND confidence >= 0.5
        """, (video_id,))
        conditions = [row['condition_code'] for row in cur.fetchall()]

        if len(conditions) < 2:
            return  # Need at least 2 conditions for comorbidity

        # Create pairs (sorted to ensure condition_a < condition_b)
        from itertools import combinations
        for cond_a, cond_b in combinations(sorted(conditions), 2):
            cur.execute("""
                INSERT INTO comorbidity_pairs (condition_a, condition_b, video_count)
                VALUES (%s, %s, 1)
                ON CONFLICT (condition_a, condition_b) DO UPDATE SET
                    video_count = comorbidity_pairs.video_count + 1,
                    updated_at = CURRENT_TIMESTAMP
            """, (cond_a, cond_b))

        conn.commit()


def get_comorbidity_matrix() -> List[Dict[str, Any]]:
    """Get comorbidity data for all condition pairs."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT condition_a, condition_b, video_count,
                   avg_concordance_a, avg_concordance_b
            FROM comorbidity_pairs
            ORDER BY video_count DESC
        """)
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Transcript Quality Operations (NEW)
# =============================================================================

def insert_transcript_quality(transcript_id: int, metrics: Dict[str, Any]) -> int:
    """Insert transcript quality assessment."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO transcript_quality (
                transcript_id, quality_score, clarity_score, completeness_score,
                medical_term_density, filler_word_ratio, avg_confidence,
                low_confidence_segments, total_segments, issues
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (transcript_id) DO UPDATE SET
                quality_score = EXCLUDED.quality_score,
                clarity_score = EXCLUDED.clarity_score,
                completeness_score = EXCLUDED.completeness_score,
                medical_term_density = EXCLUDED.medical_term_density,
                filler_word_ratio = EXCLUDED.filler_word_ratio,
                avg_confidence = EXCLUDED.avg_confidence,
                low_confidence_segments = EXCLUDED.low_confidence_segments,
                total_segments = EXCLUDED.total_segments,
                issues = EXCLUDED.issues,
                assessed_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            transcript_id,
            metrics.get('quality_score'),
            metrics.get('clarity_score'),
            metrics.get('completeness_score'),
            metrics.get('medical_term_density'),
            metrics.get('filler_word_ratio'),
            metrics.get('avg_confidence'),
            metrics.get('low_confidence_segments'),
            metrics.get('total_segments'),
            metrics.get('issues', [])
        ))
        return cur.fetchone()[0]


def get_transcript_quality(transcript_id: int) -> Optional[Dict[str, Any]]:
    """Get quality metrics for a transcript."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM transcript_quality
            WHERE transcript_id = %s
        """, (transcript_id,))
        result = cur.fetchone()
        return dict(result) if result else None


# =============================================================================
# Pipeline Progress Operations (NEW - for resumable runs)
# =============================================================================

def init_pipeline_progress(run_id: int, urls: List[str]):
    """Initialize progress tracking for a batch of URLs."""
    with get_connection() as conn:
        cur = conn.cursor()
        for url in urls:
            cur.execute("""
                INSERT INTO pipeline_progress (run_id, url, stage)
                VALUES (%s, %s, 'queued')
                ON CONFLICT (run_id, url) DO NOTHING
            """, (run_id, url))
        conn.commit()


def update_pipeline_progress(run_id: int, url: str, stage: str,
                            video_id: int = None, error_message: str = None):
    """Update progress for a URL in the pipeline."""
    with get_connection() as conn:
        cur = conn.cursor()

        if stage == 'failed':
            cur.execute("""
                UPDATE pipeline_progress
                SET stage = %s, error_message = %s, completed_at = CURRENT_TIMESTAMP,
                    retry_count = retry_count + 1
                WHERE run_id = %s AND url = %s
            """, (stage, error_message, run_id, url))
        elif stage in ('downloading', 'transcribing', 'extracting'):
            cur.execute("""
                UPDATE pipeline_progress
                SET stage = %s, started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                    video_id = COALESCE(%s, video_id)
                WHERE run_id = %s AND url = %s
            """, (stage, video_id, run_id, url))
        else:
            cur.execute("""
                UPDATE pipeline_progress
                SET stage = %s, video_id = COALESCE(%s, video_id),
                    completed_at = CASE WHEN %s = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE run_id = %s AND url = %s
            """, (stage, video_id, stage, run_id, url))

        conn.commit()


def get_incomplete_urls(run_id: int) -> List[Dict[str, Any]]:
    """Get URLs that haven't completed processing (for resume)."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT url, stage, video_id, retry_count, error_message
            FROM pipeline_progress
            WHERE run_id = %s AND stage NOT IN ('completed')
            ORDER BY
                CASE stage
                    WHEN 'failed' THEN 1
                    WHEN 'queued' THEN 2
                    WHEN 'downloading' THEN 3
                    WHEN 'downloaded' THEN 4
                    WHEN 'transcribing' THEN 5
                    WHEN 'transcribed' THEN 6
                    WHEN 'extracting' THEN 7
                END
        """, (run_id,))
        return [dict(row) for row in cur.fetchall()]


def get_latest_run_id() -> Optional[int]:
    """Get the most recent processing run ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id FROM processing_runs
            ORDER BY started_at DESC
            LIMIT 1
        """)
        result = cur.fetchone()
        return result[0] if result else None


def get_run_progress_summary(run_id: int) -> Dict[str, Any]:
    """Get summary of progress for a run."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT stage, COUNT(*) as count
            FROM pipeline_progress
            WHERE run_id = %s
            GROUP BY stage
        """, (run_id,))
        by_stage = {row['stage']: row['count'] for row in cur.fetchall()}

        cur.execute("""
            SELECT COUNT(*) as total FROM pipeline_progress WHERE run_id = %s
        """, (run_id,))
        total = cur.fetchone()['total']

        return {
            'run_id': run_id,
            'total': total,
            'completed': by_stage.get('completed', 0),
            'failed': by_stage.get('failed', 0),
            'in_progress': total - by_stage.get('completed', 0) - by_stage.get('failed', 0),
            'by_stage': by_stage
        }


# =============================================================================
# Narrative Elements Operations (STRAIN Framework)
# =============================================================================

def insert_narrative_elements(video_id: int, elements: Dict[str, Any]) -> int:
    """Insert narrative elements for STRAIN analysis."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO narrative_elements (
                video_id, content_type,
                mentions_self_diagnosis, mentions_professional_diagnosis,
                mentions_negative_testing, mentions_doctor_dismissal,
                mentions_medical_gaslighting, mentions_long_diagnostic_journey,
                mentions_multiple_doctors, years_to_diagnosis_mentioned,
                mentions_stress_triggers, mentions_symptom_flares,
                mentions_symptom_migration, mentions_online_community,
                mentions_other_creators, mentions_learning_from_tiktok,
                cites_medical_sources, claims_healthcare_background,
                claims_expert_knowledge, uses_condition_as_identity,
                mentions_chronic_illness_community,
                diagnostic_journey_quotes, stress_trigger_quotes,
                confidence, extractor_model, extractor_provider
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (video_id) DO UPDATE SET
                content_type = EXCLUDED.content_type,
                mentions_self_diagnosis = EXCLUDED.mentions_self_diagnosis,
                mentions_professional_diagnosis = EXCLUDED.mentions_professional_diagnosis,
                mentions_negative_testing = EXCLUDED.mentions_negative_testing,
                mentions_doctor_dismissal = EXCLUDED.mentions_doctor_dismissal,
                mentions_medical_gaslighting = EXCLUDED.mentions_medical_gaslighting,
                mentions_long_diagnostic_journey = EXCLUDED.mentions_long_diagnostic_journey,
                mentions_stress_triggers = EXCLUDED.mentions_stress_triggers,
                mentions_symptom_flares = EXCLUDED.mentions_symptom_flares,
                confidence = EXCLUDED.confidence,
                extracted_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (
            video_id,
            elements.get('content_type'),
            elements.get('mentions_self_diagnosis'),
            elements.get('mentions_professional_diagnosis'),
            elements.get('mentions_negative_testing'),
            elements.get('mentions_doctor_dismissal'),
            elements.get('mentions_medical_gaslighting'),
            elements.get('mentions_long_diagnostic_journey'),
            elements.get('mentions_multiple_doctors'),
            elements.get('years_to_diagnosis_mentioned'),
            elements.get('mentions_stress_triggers'),
            elements.get('mentions_symptom_flares'),
            elements.get('mentions_symptom_migration'),
            elements.get('mentions_online_community'),
            elements.get('mentions_other_creators'),
            elements.get('mentions_learning_from_tiktok'),
            elements.get('cites_medical_sources'),
            elements.get('claims_healthcare_background'),
            elements.get('claims_expert_knowledge'),
            elements.get('uses_condition_as_identity'),
            elements.get('mentions_chronic_illness_community'),
            elements.get('diagnostic_journey_quotes', []),
            elements.get('stress_trigger_quotes', []),
            elements.get('confidence', 0.5),
            elements.get('extractor_model'),
            elements.get('extractor_provider')
        ))
        return cur.fetchone()[0]


def get_narrative_elements(video_id: int) -> Optional[Dict[str, Any]]:
    """Get narrative elements for a video."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM narrative_elements WHERE video_id = %s", (video_id,))
        result = cur.fetchone()
        return dict(result) if result else None


def get_strain_indicators_summary() -> Dict[str, Any]:
    """Get summary statistics for STRAIN framework indicators."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Self-diagnosis vs professional diagnosis
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE mentions_self_diagnosis = TRUE) as self_diagnosed_count,
                COUNT(*) FILTER (WHERE mentions_professional_diagnosis = TRUE) as professionally_diagnosed_count,
                COUNT(*) FILTER (WHERE mentions_negative_testing = TRUE) as negative_testing_count,
                COUNT(*) FILTER (WHERE mentions_doctor_dismissal = TRUE) as doctor_dismissal_count,
                COUNT(*) FILTER (WHERE mentions_medical_gaslighting = TRUE) as medical_gaslighting_count,
                COUNT(*) FILTER (WHERE mentions_long_diagnostic_journey = TRUE) as long_journey_count,
                COUNT(*) FILTER (WHERE mentions_stress_triggers = TRUE) as stress_triggers_count,
                COUNT(*) FILTER (WHERE mentions_symptom_flares = TRUE) as symptom_flares_count,
                COUNT(*) FILTER (WHERE mentions_online_community = TRUE) as online_community_count,
                COUNT(*) FILTER (WHERE mentions_learning_from_tiktok = TRUE) as learned_from_tiktok_count,
                COUNT(*) as total_analyzed
            FROM narrative_elements
        """)
        indicators = dict(cur.fetchone())
        
        # Content type breakdown
        cur.execute("""
            SELECT content_type, COUNT(*) as count
            FROM narrative_elements
            WHERE content_type IS NOT NULL
            GROUP BY content_type
            ORDER BY count DESC
        """)
        by_content_type = [dict(row) for row in cur.fetchall()]
        
        indicators['by_content_type'] = by_content_type
        return indicators


# =============================================================================
# User-Level Analysis Functions (for longitudinal tracking)
# =============================================================================

def get_user_profile(username: str) -> Dict[str, Any]:
    """Get comprehensive profile for a TikTok user across all their videos."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Basic user stats
        cur.execute("""
            SELECT 
                author as username,
                COUNT(*) as video_count,
                MIN(upload_date) as first_video_date,
                MAX(upload_date) as last_video_date,
                SUM(view_count) as total_views,
                AVG(like_count) as avg_likes,
                MAX(author_follower_count) as follower_count
            FROM videos
            WHERE LOWER(author) = LOWER(%s)
            GROUP BY author
        """, (username,))
        basic_stats = cur.fetchone()
        if not basic_stats:
            return None
        
        profile = dict(basic_stats)
        
        # Get all video IDs for this user
        cur.execute("""
            SELECT id FROM videos WHERE LOWER(author) = LOWER(%s)
        """, (username,))
        video_ids = [row['id'] for row in cur.fetchall()]
        profile['video_ids'] = video_ids
        
        # All claimed diagnoses with first mention date (aggregated by condition_code only)
        cur.execute("""
            SELECT 
                cd.condition_code,
                MODE() WITHIN GROUP (ORDER BY cd.condition_name) as condition_name,
                COUNT(*) as mention_count,
                AVG(cd.confidence) as avg_confidence,
                BOOL_OR(cd.is_self_diagnosed) as ever_self_diagnosed,
                BOOL_OR(NOT cd.is_self_diagnosed) as ever_professionally_diagnosed,
                MIN(v.upload_date) as first_mentioned,
                MAX(v.upload_date) as last_mentioned,
                COUNT(DISTINCT v.id) as video_count
            FROM claimed_diagnoses cd
            JOIN videos v ON cd.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY cd.condition_code
            ORDER BY first_mentioned
        """, (username,))
        profile['diagnoses'] = [dict(row) for row in cur.fetchall()]
        
        # All symptoms with frequency
        cur.execute("""
            SELECT 
                s.symptom,
                s.category,
                COUNT(*) as mention_count,
                AVG(s.confidence) as avg_confidence,
                ARRAY_AGG(DISTINCT s.severity) as severities_reported
            FROM symptoms s
            JOIN videos v ON s.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY s.symptom, s.category
            ORDER BY mention_count DESC
        """, (username,))
        profile['symptoms'] = [dict(row) for row in cur.fetchall()]
        
        # Concordance scores per condition
        cur.execute("""
            SELECT 
                cd.condition_code,
                AVG(sc.concordance_score) as avg_concordance,
                AVG(sc.core_symptom_score) as avg_core_score,
                MIN(sc.concordance_score) as min_concordance,
                MAX(sc.concordance_score) as max_concordance,
                COUNT(*) as sample_count
            FROM symptom_concordance sc
            JOIN claimed_diagnoses cd ON sc.diagnosis_id = cd.id
            JOIN videos v ON sc.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY cd.condition_code
        """, (username,))
        profile['concordance'] = [dict(row) for row in cur.fetchall()]
        
        # STRAIN narrative indicators aggregated
        cur.execute("""
            SELECT 
                SUM(CASE WHEN mentions_self_diagnosis THEN 1 ELSE 0 END) as self_diagnosis_mentions,
                SUM(CASE WHEN mentions_professional_diagnosis THEN 1 ELSE 0 END) as professional_diagnosis_mentions,
                SUM(CASE WHEN mentions_doctor_dismissal THEN 1 ELSE 0 END) as doctor_dismissal_mentions,
                SUM(CASE WHEN mentions_medical_gaslighting THEN 1 ELSE 0 END) as medical_gaslighting_mentions,
                SUM(CASE WHEN mentions_stress_triggers THEN 1 ELSE 0 END) as stress_trigger_mentions,
                SUM(CASE WHEN mentions_online_community THEN 1 ELSE 0 END) as online_community_mentions,
                SUM(CASE WHEN mentions_learning_from_tiktok THEN 1 ELSE 0 END) as learned_from_tiktok_mentions,
                COUNT(*) as total_narratives
            FROM narrative_elements ne
            JOIN videos v ON ne.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
        """, (username,))
        profile['strain_indicators'] = dict(cur.fetchone() or {})
        
        # Treatment mentions
        cur.execute("""
            SELECT 
                t.treatment_type,
                t.treatment_name,
                COUNT(*) as mention_count,
                ARRAY_AGG(DISTINCT t.effectiveness) as reported_effectiveness
            FROM treatments t
            JOIN videos v ON t.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY t.treatment_type, t.treatment_name
            ORDER BY mention_count DESC
            LIMIT 30
        """, (username,))
        profile['treatments'] = [dict(row) for row in cur.fetchall()]
        
        return profile


def get_user_timeline(username: str) -> List[Dict[str, Any]]:
    """Get chronological timeline of a user's health narrative evolution."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                v.id as video_id,
                v.upload_date,
                v.title,
                v.view_count,
                t.word_count,
                t.song_lyrics_ratio,
                ARRAY_AGG(DISTINCT cd.condition_code) FILTER (WHERE cd.condition_code IS NOT NULL) as diagnoses_mentioned,
                COUNT(DISTINCT s.id) as symptom_count,
                ARRAY_AGG(DISTINCT s.category) FILTER (WHERE s.category IS NOT NULL) as symptom_categories,
                ne.mentions_self_diagnosis,
                ne.mentions_professional_diagnosis,
                ne.content_type
            FROM videos v
            LEFT JOIN transcripts t ON v.id = t.video_id
            LEFT JOIN claimed_diagnoses cd ON v.id = cd.video_id
            LEFT JOIN symptoms s ON v.id = s.video_id
            LEFT JOIN narrative_elements ne ON v.id = ne.video_id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY v.id, v.upload_date, v.title, v.view_count, t.word_count, t.song_lyrics_ratio,
                     ne.mentions_self_diagnosis, ne.mentions_professional_diagnosis, ne.content_type
            ORDER BY v.upload_date
        """, (username,))
        
        return [dict(row) for row in cur.fetchall()]


def get_user_concordance_over_time(username: str) -> List[Dict[str, Any]]:
    """Track how a user's concordance scores change over time."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                v.upload_date,
                v.id as video_id,
                cd.condition_code,
                sc.concordance_score,
                sc.core_symptom_score,
                sc.total_symptoms_reported,
                sc.expected_symptoms_matched,
                sc.unexpected_symptoms_count
            FROM symptom_concordance sc
            JOIN claimed_diagnoses cd ON sc.diagnosis_id = cd.id
            JOIN videos v ON sc.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            ORDER BY v.upload_date, cd.condition_code
        """, (username,))
        
        return [dict(row) for row in cur.fetchall()]


def get_all_users_summary() -> List[Dict[str, Any]]:
    """Get summary statistics for all users in the database."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                v.author as username,
                COUNT(DISTINCT v.id) as video_count,
                COUNT(DISTINCT cd.condition_code) as unique_diagnoses,
                COUNT(DISTINCT s.symptom) as unique_symptoms,
                AVG(sc.concordance_score) as avg_concordance,
                MIN(v.upload_date) as first_video,
                MAX(v.upload_date) as last_video,
                MAX(v.author_follower_count) as followers
            FROM videos v
            LEFT JOIN claimed_diagnoses cd ON v.id = cd.video_id
            LEFT JOIN symptoms s ON v.id = s.video_id
            LEFT JOIN symptom_concordance sc ON v.id = sc.video_id
            WHERE v.author IS NOT NULL
            GROUP BY v.author
            HAVING COUNT(DISTINCT v.id) >= 1
            ORDER BY video_count DESC
        """)
        
        return [dict(row) for row in cur.fetchall()]


def get_diagnosis_acquisition_patterns() -> List[Dict[str, Any]]:
    """Analyze patterns of how users acquire new diagnoses over time."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # For each user, find when they first mentioned each condition
        cur.execute("""
            WITH first_mentions AS (
                SELECT 
                    v.author,
                    cd.condition_code,
                    MIN(v.upload_date) as first_mention_date,
                    ROW_NUMBER() OVER (PARTITION BY v.author ORDER BY MIN(v.upload_date)) as diagnosis_order
                FROM claimed_diagnoses cd
                JOIN videos v ON cd.video_id = v.id
                WHERE v.author IS NOT NULL AND v.upload_date IS NOT NULL
                GROUP BY v.author, cd.condition_code
            )
            SELECT 
                author as username,
                condition_code,
                first_mention_date,
                diagnosis_order,
                LAG(condition_code) OVER (PARTITION BY author ORDER BY first_mention_date) as previous_diagnosis,
                LAG(first_mention_date) OVER (PARTITION BY author ORDER BY first_mention_date) as previous_date
            FROM first_mentions
            ORDER BY author, first_mention_date
        """)
        
        return [dict(row) for row in cur.fetchall()]


def get_symptom_consistency_analysis(username: str) -> Dict[str, Any]:
    """Analyze symptom reporting consistency for a user across their videos."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get symptom reporting patterns
        cur.execute("""
            SELECT 
                s.symptom,
                s.category,
                COUNT(*) as times_mentioned,
                COUNT(DISTINCT s.severity) as severity_variations,
                ARRAY_AGG(DISTINCT s.severity) as all_severities,
                AVG(s.confidence) as avg_confidence,
                MIN(v.upload_date) as first_mentioned,
                MAX(v.upload_date) as last_mentioned
            FROM symptoms s
            JOIN videos v ON s.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY s.symptom, s.category
            ORDER BY times_mentioned DESC
        """, (username,))
        symptom_patterns = [dict(row) for row in cur.fetchall()]
        
        # Find symptoms with inconsistent severity reporting
        inconsistent = [s for s in symptom_patterns if s['severity_variations'] > 1]
        
        # Get video count
        cur.execute("""
            SELECT COUNT(*) as count FROM videos WHERE LOWER(author) = LOWER(%s)
        """, (username,))
        video_count = cur.fetchone()['count']
        
        return {
            'username': username,
            'video_count': video_count,
            'total_unique_symptoms': len(symptom_patterns),
            'symptoms_with_inconsistent_severity': len(inconsistent),
            'inconsistent_symptoms': inconsistent[:10],  # Top 10
            'all_symptoms': symptom_patterns
        }


def get_users_with_low_concordance(threshold: float = 0.3) -> List[Dict[str, Any]]:
    """Find users with consistently low concordance scores (potential social contagion cases)."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                v.author as username,
                cd.condition_code,
                AVG(sc.concordance_score) as avg_concordance,
                AVG(sc.core_symptom_score) as avg_core_score,
                COUNT(*) as video_count,
                BOOL_OR(cd.is_self_diagnosed) as includes_self_diagnosed
            FROM symptom_concordance sc
            JOIN claimed_diagnoses cd ON sc.diagnosis_id = cd.id
            JOIN videos v ON sc.video_id = v.id
            WHERE v.author IS NOT NULL
            GROUP BY v.author, cd.condition_code
            HAVING AVG(sc.concordance_score) < %s AND COUNT(*) >= 2
            ORDER BY avg_concordance ASC
        """, (threshold,))
        
        return [dict(row) for row in cur.fetchall()]


# =============================================================================
# Longitudinal Tracking Update Functions
# =============================================================================

def update_user_profile(username: str) -> Dict[str, Any]:
    """Update or create a user profile with aggregated statistics."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Calculate all aggregated stats for this user
        cur.execute("""
            WITH user_stats AS (
                SELECT 
                    v.author as username,
                    MIN(v.upload_date) as first_video_date,
                    MAX(v.upload_date) as last_video_date,
                    COUNT(DISTINCT v.id) as video_count,
                    MAX(v.author_follower_count) as follower_count
                FROM videos v
                WHERE LOWER(v.author) = LOWER(%s)
                GROUP BY v.author
            ),
            symptom_stats AS (
                SELECT COUNT(*) as total_symptoms
                FROM symptoms s
                JOIN videos v ON s.video_id = v.id
                WHERE LOWER(v.author) = LOWER(%s)
            ),
            diagnosis_stats AS (
                SELECT 
                    COUNT(DISTINCT cd.condition_code) as unique_diagnoses,
                    MODE() WITHIN GROUP (ORDER BY cd.condition_code) as primary_condition
                FROM claimed_diagnoses cd
                JOIN videos v ON cd.video_id = v.id
                WHERE LOWER(v.author) = LOWER(%s)
            ),
            concordance_stats AS (
                SELECT 
                    AVG(sc.concordance_score) as avg_concordance,
                    AVG(sc.core_symptom_score) as avg_core_score
                FROM symptom_concordance sc
                JOIN videos v ON sc.video_id = v.id
                WHERE LOWER(v.author) = LOWER(%s)
            ),
            narrative_stats AS (
                SELECT 
                    COUNT(*) as total_narratives,
                    SUM(CASE WHEN mentions_self_diagnosis THEN 1 ELSE 0 END) as self_diag_count,
                    SUM(CASE WHEN mentions_professional_diagnosis THEN 1 ELSE 0 END) as prof_diag_count,
                    SUM(CASE WHEN mentions_doctor_dismissal THEN 1 ELSE 0 END) as dismissal_count,
                    SUM(CASE WHEN mentions_stress_triggers THEN 1 ELSE 0 END) as stress_count
                FROM narrative_elements ne
                JOIN videos v ON ne.video_id = v.id
                WHERE LOWER(v.author) = LOWER(%s)
            )
            SELECT 
                us.*,
                ss.total_symptoms,
                ds.unique_diagnoses,
                ds.primary_condition,
                cs.avg_concordance,
                cs.avg_core_score,
                ns.total_narratives,
                ns.self_diag_count,
                ns.prof_diag_count,
                ns.dismissal_count,
                ns.stress_count,
                CASE WHEN ns.total_narratives > 0 
                     THEN ns.self_diag_count::float / ns.total_narratives 
                     ELSE NULL END as self_diagnosis_ratio,
                CASE WHEN ns.total_narratives > 0 
                     THEN ns.prof_diag_count::float / ns.total_narratives 
                     ELSE NULL END as professional_diagnosis_ratio
            FROM user_stats us
            CROSS JOIN symptom_stats ss
            CROSS JOIN diagnosis_stats ds
            CROSS JOIN concordance_stats cs
            CROSS JOIN narrative_stats ns
        """, (username, username, username, username, username))
        
        stats = cur.fetchone()
        if not stats:
            return None
        
        # Determine flags
        is_low_concordance = stats['avg_concordance'] is not None and stats['avg_concordance'] < 0.3
        
        # Upsert the profile
        cur.execute("""
            INSERT INTO user_profiles (
                username, first_video_date, last_video_date, video_count,
                total_symptoms_reported, unique_diagnoses_count,
                avg_concordance_score, avg_core_symptom_score, primary_condition,
                follower_count, self_diagnosis_ratio, professional_diagnosis_ratio,
                doctor_dismissal_mentions, stress_trigger_mentions,
                is_flagged_low_concordance, last_updated
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (username) DO UPDATE SET
                first_video_date = EXCLUDED.first_video_date,
                last_video_date = EXCLUDED.last_video_date,
                video_count = EXCLUDED.video_count,
                total_symptoms_reported = EXCLUDED.total_symptoms_reported,
                unique_diagnoses_count = EXCLUDED.unique_diagnoses_count,
                avg_concordance_score = EXCLUDED.avg_concordance_score,
                avg_core_symptom_score = EXCLUDED.avg_core_symptom_score,
                primary_condition = EXCLUDED.primary_condition,
                follower_count = EXCLUDED.follower_count,
                self_diagnosis_ratio = EXCLUDED.self_diagnosis_ratio,
                professional_diagnosis_ratio = EXCLUDED.professional_diagnosis_ratio,
                doctor_dismissal_mentions = EXCLUDED.doctor_dismissal_mentions,
                stress_trigger_mentions = EXCLUDED.stress_trigger_mentions,
                is_flagged_low_concordance = EXCLUDED.is_flagged_low_concordance,
                last_updated = CURRENT_TIMESTAMP
            RETURNING *
        """, (
            stats['username'], stats['first_video_date'], stats['last_video_date'],
            stats['video_count'], stats['total_symptoms'], stats['unique_diagnoses'],
            stats['avg_concordance'], stats['avg_core_score'], stats['primary_condition'],
            stats['follower_count'], stats['self_diagnosis_ratio'], stats['professional_diagnosis_ratio'],
            stats['dismissal_count'], stats['stress_count'], is_low_concordance
        ))
        
        result = cur.fetchone()
        conn.commit()
        return dict(result) if result else None


def update_diagnosis_timeline(username: str) -> List[Dict[str, Any]]:
    """Update the diagnosis timeline for a user."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        # Get diagnosis history
        cur.execute("""
            WITH diagnosis_history AS (
                SELECT 
                    v.author as username,
                    cd.condition_code,
                    MIN(v.upload_date) as first_mentioned,
                    MAX(v.upload_date) as last_mentioned,
                    COUNT(*) as mention_count,
                    BOOL_OR(cd.is_self_diagnosed) as was_self_diagnosed,
                    ROW_NUMBER() OVER (ORDER BY MIN(v.upload_date)) as diagnosis_order
                FROM claimed_diagnoses cd
                JOIN videos v ON cd.video_id = v.id
                WHERE LOWER(v.author) = LOWER(%s) AND v.upload_date IS NOT NULL
                GROUP BY v.author, cd.condition_code
            )
            SELECT 
                dh.*,
                LAG(condition_code) OVER (ORDER BY first_mentioned) as previous_diagnosis,
                LAG(first_mentioned) OVER (ORDER BY first_mentioned) as previous_date
            FROM diagnosis_history dh
            ORDER BY first_mentioned
        """, (username,))
        
        results = []
        for row in cur.fetchall():
            days_since = None
            if row['previous_date'] and row['first_mentioned']:
                days_since = (row['first_mentioned'] - row['previous_date']).days
            
            cur.execute("""
                INSERT INTO diagnosis_timeline (
                    username, condition_code, first_mentioned, last_mentioned,
                    mention_count, was_self_diagnosed, diagnosis_order,
                    days_since_previous_diagnosis, previous_diagnosis
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username, condition_code) DO UPDATE SET
                    last_mentioned = EXCLUDED.last_mentioned,
                    mention_count = EXCLUDED.mention_count,
                    was_self_diagnosed = EXCLUDED.was_self_diagnosed
                RETURNING *
            """, (
                row['username'], row['condition_code'], row['first_mentioned'],
                row['last_mentioned'], row['mention_count'], row['was_self_diagnosed'],
                row['diagnosis_order'], days_since, row['previous_diagnosis']
            ))
            results.append(dict(cur.fetchone()))
        
        conn.commit()
        return results


def update_symptom_consistency(username: str) -> List[Dict[str, Any]]:
    """Update symptom consistency tracking for a user."""
    with get_connection() as conn:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT 
                %s as username,
                s.symptom,
                s.category,
                COUNT(*) as total_mentions,
                COUNT(DISTINCT s.severity) as severity_variations,
                ARRAY_AGG(DISTINCT s.severity) FILTER (WHERE s.severity IS NOT NULL) as all_severities,
                MIN(v.upload_date) as first_mentioned,
                MAX(v.upload_date) as last_mentioned,
                AVG(s.confidence) as avg_confidence
            FROM symptoms s
            JOIN videos v ON s.video_id = v.id
            WHERE LOWER(v.author) = LOWER(%s)
            GROUP BY s.symptom, s.category
        """, (username, username))
        
        results = []
        for row in cur.fetchall():
            is_inconsistent = row['severity_variations'] > 1
            
            cur.execute("""
                INSERT INTO symptom_consistency (
                    username, symptom, category, total_mentions,
                    severity_variations, all_severities, first_mentioned,
                    last_mentioned, avg_confidence, is_inconsistent
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (username, symptom) DO UPDATE SET
                    total_mentions = EXCLUDED.total_mentions,
                    severity_variations = EXCLUDED.severity_variations,
                    all_severities = EXCLUDED.all_severities,
                    last_mentioned = EXCLUDED.last_mentioned,
                    avg_confidence = EXCLUDED.avg_confidence,
                    is_inconsistent = EXCLUDED.is_inconsistent
                RETURNING *
            """, (
                row['username'], row['symptom'], row['category'],
                row['total_mentions'], row['severity_variations'],
                row['all_severities'], row['first_mentioned'],
                row['last_mentioned'], row['avg_confidence'], is_inconsistent
            ))
            results.append(dict(cur.fetchone()))
        
        conn.commit()
        return results


def refresh_all_user_profiles() -> int:
    """Refresh all user profiles in the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT author FROM videos WHERE author IS NOT NULL")
        usernames = [row[0] for row in cur.fetchall()]
    
    count = 0
    for username in usernames:
        try:
            update_user_profile(username)
            update_diagnosis_timeline(username)
            update_symptom_consistency(username)
            count += 1
        except Exception as e:
            print(f"Error updating {username}: {e}")
    
    return count


if __name__ == '__main__':
    print("Initializing database...")
    init_db()

    try:
        stats = get_symptom_statistics()
        print(f"\nDatabase Statistics:")
        print(f"  Videos: {stats['total_videos']}")
        print(f"  Transcripts: {stats['total_transcripts']}")
        print(f"  Symptoms: {stats['total_symptoms']}")
        print(f"  Diagnoses: {stats['total_diagnoses']}")
        if stats['diagnoses_by_condition']:
            print(f"\n  Diagnoses by Condition:")
            for d in stats['diagnoses_by_condition']:
                print(f"    {d['condition_code']}: {d['count']}")
        if stats['concordance_by_condition']:
            print(f"\n  Concordance by Condition:")
            for c in stats['concordance_by_condition']:
                print(f"    {c['condition_code']}: {c['avg_concordance']:.2f} avg concordance ({c['sample_size']} samples)")
    except Exception as e:
        print(f"Could not fetch statistics: {e}")
