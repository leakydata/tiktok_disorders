-- Creator watchlist for longitudinal tracking.
--
-- Separates "creators we intend to follow over time" from user_profiles, which
-- is a derived aggregate rebuilt from videos. A creator stays on the watchlist
-- even before any of their videos are collected, and carries its own polling
-- cadence and bookkeeping.
--
-- Apply with: python scripts/track.py init

CREATE TABLE IF NOT EXISTS tracked_creators (
    username              TEXT PRIMARY KEY,

    -- Cohort assignment. Free-form so cohorts can be defined per research
    -- question ('strain_pilot', 'mcas_core'). Keep the selection rule in
    -- added_reason so a cohort can be reconstructed later.
    cohort                TEXT,
    added_reason          TEXT,
    added_at              TIMESTAMP NOT NULL DEFAULT now(),

    -- Polling cadence. Lower priority number = checked first when a run is
    -- capped with --limit.
    priority              SMALLINT NOT NULL DEFAULT 3,
    check_interval_days   INTEGER  NOT NULL DEFAULT 7,
    is_active             BOOLEAN  NOT NULL DEFAULT true,

    -- Bookkeeping, maintained by `track.py check`.
    last_checked_at       TIMESTAMP,
    last_new_video_at     TIMESTAMP,
    videos_at_last_check  INTEGER,
    new_videos_found      INTEGER NOT NULL DEFAULT 0,
    consecutive_empty     INTEGER NOT NULL DEFAULT 0,
    last_error            TEXT,

    notes                 TEXT
);

CREATE INDEX IF NOT EXISTS idx_tracked_creators_active
    ON tracked_creators (is_active, priority, last_checked_at);

CREATE INDEX IF NOT EXISTS idx_tracked_creators_cohort
    ON tracked_creators (cohort);

-- Creators whose next check is due. A creator that has never been checked is
-- always due. Ordered so that --limit takes the most urgent first.
CREATE OR REPLACE VIEW creators_due_for_check AS
SELECT
    tc.*,
    (tc.last_checked_at IS NULL) AS never_checked,
    now() - tc.last_checked_at   AS since_last_check
FROM tracked_creators tc
WHERE tc.is_active
  AND (
      tc.last_checked_at IS NULL
      OR tc.last_checked_at + (tc.check_interval_days * INTERVAL '1 day') < now()
  )
ORDER BY tc.priority ASC, tc.last_checked_at ASC NULLS FIRST;
