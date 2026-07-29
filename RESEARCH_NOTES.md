# Research Notes

Running log of methodological findings, dead ends, and open decisions across
the two projects that share the `tiktok_disorders` database. Intended to
outlive any single analysis session — record things here that would otherwise
have to be rediscovered.

Convention: newest section first, absolute dates, and state the *consequence*
of a finding, not just the finding.

---

## 2026-07-29 — yt-dlp impersonation was silently disabled (TikTok blocks)

`uv sync` resolved **curl-cffi 0.14.0**, but yt-dlp 2025.12.08 accepts only
`0.5.10` or `>=0.10,<0.14` and raises `ImportError` on anything else. That
import failure is swallowed, leaving **0 impersonate targets** and only a
non-fatal warning per download:

    WARNING: [TikTok] The extractor is attempting impersonation, but no
    impersonate target is available.

Without impersonation TikTok blocks a large share of requests — this is the
likely cause of the "Your IP address is blocked from accessing this post"
errors seen during engagement snapshots, and of `urls_failed.txt` having
grown to 1,628 entries.

Fixed by pinning `curl-cffi>=0.13.0,<0.14` in `pyproject.toml` (was
`>=0.13.0`, which let the incompatible 0.14 in). After the pin: **31
impersonate targets**, warning gone, downloads succeed.

Check after any dependency update:

    .venv/bin/python -c "from yt_dlp import YoutubeDL; \
      print(len(YoutubeDL({'quiet':True})._get_available_impersonate_targets()))"

Zero means impersonation is off and collection is degraded.


## 2026-07-29 — Extraction provider benchmark (measured, not estimated)

All four candidates run against the **same** combined-extraction system prompt
(14,943 chars / ~3,735 tokens) and the same transcript, `temperature=0`:

| provider / model | latency | valid JSON | prompt tok | completion tok | reasoning tok |
|---|---|---|---|---|---|
| **MiniMax-M3** | **4.9s** | yes | 3,907 | **486** | **0** |
| MiniMax-M2.7-highspeed | 12.5s | yes | 3,751 | 736 | 505 |
| deepseek-v4-flash | 20.1s | yes | 3,950 | 2,274 | 1,200 |
| deepseek-v4-pro | 57.1s | **no** (truncated) | 3,950 | 4,096 (capped) | 3,260 |
| local gemma4:31b (Ollama) | **>10 min** | — | — | — | — |

**MiniMax-M3 is the clear choice** and is now the configured default
(`MINIMAX_MODEL`). It emits **zero reasoning tokens** on this task, making it
both fastest and ~4.7x cheaper on output than deepseek-v4-flash. For structured
extraction, reasoning is pure overhead — v4-pro spent 3,260 tokens thinking,
hit the token cap mid-JSON, and returned unparseable output.

**Local inference is not competitive here, and the reason is concurrency, not
per-call speed.** `SymptomExtractor.max_workers = 20`, but gemma4:31b occupies
29GB of 32GB total VRAM, so Ollama serializes all 20 workers through one model
instance. A single local extraction did not complete in 10 minutes. Across
6,923 transcripts that is roughly: **MiniMax ~30 min, DeepSeek ~2 h, local
~1 month.** Do not assume "local is free" implies "local is cheap" — wall-clock
is the binding constraint. A smaller model (phi4 at 9GB) could run 2-3 in
parallel if local inference is ever required.

### Provider inventory (verified live, 2026-07-29)

| provider | status | models actually served |
|---|---|---|
| MiniMax | working, credit available | `MiniMax-M3`, `MiniMax-M2.7`, `MiniMax-M2.7-highspeed`, `MiniMax-M2.5`, `MiniMax-Text-01` |
| DeepSeek | **$18.71** remaining | `deepseek-v4-flash`, `deepseek-v4-pro` |
| OpenAI | key valid, 129 models | — |
| Anthropic | key set (extracted the existing corpus) | — |

**`deepseek-chat` and `deepseek-reasoner` no longer exist.** `config.py` still
defaults `DEEPSEEK_MODEL` to `deepseek-chat`, so any DeepSeek run must pass
`--model deepseek-v4-flash` explicitly or it will fail. Left as-is rather than
silently repointing a documented default.

Balances are only checkable programmatically for DeepSeek
(`GET https://api.deepseek.com/user/balance`). MiniMax, OpenAI and Anthropic
expose no balance endpoint — check their dashboards.

---

## 2026-07-29 — Combined extraction was broken for every Ollama model

`_call_model` passed the `system` prompt to Anthropic (as a cached system
block) and to DeepSeek (as a system role message), but the **Ollama branch
silently dropped it**. Local models therefore received only
`TRANSCRIPT: <text>` with no schema and no instruction to emit JSON, replied
with prose, and every extraction failed with
`Expecting value: line 1 column 1 (char 0)`.

This was never noticed because the existing corpus was extracted through the
Anthropic API. It surfaced only when gemma4 was tried locally.

Fixes in `extractor.py`:

1. System prompt now sent as a `system` role message for Ollama.
2. `"format": "json"` added to the Ollama request so decoding is constrained —
   fenced or prose-wrapped answers cannot reach the parser at all.
3. `num_ctx` / `num_predict` were **hardcoded to 65536 / 12288**. A 64K KV cache
   pushes a large model off the GPU entirely (gemma4:31b spills ~89% to CPU at
   32K on a 24GB card) with no error explaining the slowdown. Now overridable
   via `OLLAMA_NUM_CTX` / `OLLAMA_NUM_PREDICT`; defaults unchanged. Sizing
   guide: unextracted transcripts average 76 words, p99 826 words, so 16384 is
   ample. `num_predict` must stay below `num_ctx`.
4. `supports_native_thinking` added for models using Ollama's top-level `think`
   field (gemma4, gpt-oss, glm-4.7) rather than Qwen3's `/think` prefixes.
   Note that suppressing thought is a **throughput** lever only — the cost is
   generating the tokens, so discarding them afterwards saves nothing.

**MiniMax provider support added** (`config.py`, `extractor.py`,
`pipeline.py`). It is OpenAI-compatible, so it shares DeepSeek's code path.

---

## 2026-07-29 — The song filter is what keeps the symptom set clean

| band | transcripts | symptoms | diagnoses |
|---|---|---|---|
| speech (<0.2) | 44,880 | **93,972** | **23,662** |
| unscored | 22,847 | 0 | 0 |
| music_heavy (>=0.5) | 10,895 | 0 | 0 |
| music_some (0.2-0.5) | 198 | 0 | 0 |

**Every symptom and diagnosis in the database came from a scored clean-speech
transcript.** `--max-song-ratio` quarantined the contaminated ones before
extraction, so the hallucinated song lyrics documented above never polluted a
single result.

**This property is fragile.** The extraction filter is
`(song_lyrics_ratio IS NULL OR song_lyrics_ratio < %s)` — **NULL passes**. So
scoring is not a prerequisite for extraction to *run*; it is the quality gate
that stops unscored music being extracted as if it were speech. Of the 6,923
currently eligible transcripts, ~6,661 are unscored. Always run
`scripts/detect_song_lyrics.py` before an extraction pass.

Also worth revisiting: **18,112 transcripts are excluded as "too short"
(<20 words)**. Given what VAD revealed about hallucination, some of those are
likely near-empty garbage rather than genuinely short videos.

---

## 2026-07-29 — Two more Windows-era artifacts

1. **`.env` had CRLF line terminators.** `python-dotenv` strips them so the
   pipeline was unaffected, but any shell-based read (`grep | cut`) yields a
   trailing `\r` that corrupts HTTP headers. Normalized to LF.
2. **`tiktok_research/.venv` was a Windows venv** (`Lib/`, `Scripts/`), like
   the one in this repo. Rebuilt with `uv sync --no-install-project` — the
   plain `uv sync` fails because setuptools cannot auto-discover packages in
   that flat layout.

**Credential exposure:** the DeepSeek API key was printed in full in a terminal
error during this session and should be rotated, alongside the still-unrotated
Postgres password.

---

## 2026-07-29 — Environment additions

- **pgAdmin 4 v9.16** installed from the official pgAdmin APT repo (Ubuntu
  26.04 "resolute" is supported upstream). Launch: `/usr/pgadmin4/bin/pgadmin4`.
  The snap was deliberately avoided — it is published by an individual, not the
  pgAdmin project. Connect on **port 5432**.
- **Transcription backlog cleared**: all 765 previously untranscribed videos are
  done. These are the first VAD-processed transcripts, so the corpus now
  contains two preprocessing regimes — distinguishable via `transcribed_at`.

---

## 2026-07-29 — VAD enabled; measured effect on transcription quality

`vad_filter` (Silero) is now **on by default** for the faster-whisper path,
configurable via `WHISPER_VAD*` in `config.py`. Set `WHISPER_VAD=false` to
reproduce pre-2026-07 behaviour.

Justified by an A/B run (`scripts/audio_experiment.py`, label `exp1`, n=8 per
stratum, large-v3), not assumption:

| stratum | arm | avg words | avg song ratio | empty | avg s |
|---|---|---|---|---|---|
| music_heavy | baseline | 40.5 | 0.269 | 0 | 2.68 |
| music_heavy | **vad** | 9.1 | **0.180** | **3** | **0.95** |
| music_some | baseline | 214.9 | 0.156 | 0 | 7.12 |
| music_some | **vad** | 195.5 | **0.150** | 3 | **3.93** |
| speech_clean | baseline | 441.5 | 0.013 | 0 | 11.90 |
| speech_clean | **vad** | **444.5** | **0.006** | **0** | 10.83 |

**Interpretation.** On clean speech VAD is harmless — marginally *more* words
recovered and no empties, so it does not clip real content. On music-heavy
content it suppresses hallucinated lyrics and runs ~3x faster. Every one of the
six clips VAD reduced to zero was verified to be song lyrics in the baseline
arm (e.g. "I'm so ATL, I'm so ATL, I'm so ATL", "Craving me was useless").

A single 90%-music clip demonstrates the failure mode VAD fixes: baseline
produced 95 words of Taylor Swift lyrics; the *stored production transcript*
for that same clip is `"Thank you for watching this video! If you liked it,
please subscribe to my channel..."` — a canonical Whisper hallucination. There
are 11,093 transcripts with `song_lyrics_ratio >= 0.2` and 8,942 with
`word_count < 10`; a share of both is likely this artifact.

**Open follow-up:** the `demucs` arm (vocal isolation before VAD) is
implemented and its dependencies installed, but not yet run. Worth testing on
`music_heavy` specifically — VAD rejects music wholesale, whereas separation
could recover speech that is *mixed with* music rather than absent.

**Consequence for existing data:** the 78,604 stored transcripts were produced
without VAD. Any analysis sensitive to transcript noise should either filter on
`song_lyrics_ratio` or re-transcribe the affected strata via
`scripts/retranscribe.py`.

---

## 2026-07-29 — Two blocking Windows-era artifacts fixed

Both would have silently broken any audio work:

1. **`videos.audio_path` used Windows backslashes** in all 79,369 rows
   (`data\audio\...`). Zero resolved on Linux; all resolved after `\` -> `/`.
   Normalized in place with an `UPDATE`. Re-transcription, the audio
   experiment, and anything touching audio files were all dead before this.
2. **`.mcp.json` pointed the `db` MCP server at port 5433** — the same stale
   port as the `.env` files. That is why the database MCP never worked.

See also the port/collation entries under Environment notes.

---

## 2026-07-29 — SECURITY: Postgres password was published to a public repo

`.mcp.json` was git-tracked and pushed to `github.com/leakydata/tiktok_disorders`
(public) containing a plaintext DSN with the `postgres` password. Verified
present in the `origin/main` tree, not merely local.

`.env` is **not** tracked, so the Anthropic / DeepSeek / HuggingFace keys were
not exposed.

Mitigated so far: `.mcp.json` now reads `${DATABASE_URL}`, has been
`git rm --cached`ed, and is in `.gitignore`. **Not yet done, and required:**
the password remains recoverable from git history, so it must be **rotated**
(then updated in both projects' `.env`). Removing the file does not remove
history; scrubbing history would need a force-push over the public repo.

---

## 2026-07-29 — CUDA runtime was missing for GPU transcription

CTranslate2 `dlopen()`s `libcublas.so.12` / `libcudnn` by soname, and neither
existed on this machine — GPU transcription failed outright with
`Library libcublas.so.12 is not found`. Installed `nvidia-cublas-cu12` and
`nvidia-cudnn-cu12`, which place the libraries under
`site-packages/nvidia/*/lib` — a path the dynamic loader does not search.

Rather than requiring `LD_LIBRARY_PATH` at every invocation,
`transcriber._preload_cuda_libraries()` loads them with `RTLD_GLOBAL` at import
time, so the later `dlopen` resolves against the already-loaded copies. GPU
transcription now works from a bare `uv run` with no environment setup.

---

## Reference: project layout

Two repositories share one PostgreSQL database (`tiktok_disorders`, port 5432).
Table ownership is disjoint — verified by grep over both codebases:

| | writes `videos` / `transcripts` / `symptoms` | writes `annotation_chunks` / `llm_annotation_runs` / `*_stability_metrics` |
|---|---|---|
| `tiktok_disorders` (this repo) | yes | no |
| `tiktok_research` (`../tiktok_research`) | no | yes |

- **`tiktok_disorders`** collects the corpus: discovery, download, Whisper
  transcription, symptom/diagnosis extraction. Home of the substantive
  EDS/MCAS/POTS study ([OSF preregistration](https://osf.io/5y46c), STRAIN
  framework). **That paper is unwritten.**
- **`tiktok_research`** studies the *annotation method*. Home of the
  multi-run stability paper (`PAPER_DRAFT.md`), which its own §5.8 designates
  as coming before the substantive study ("Paper 2").

The methods manuscript, its figures, and its build script live in
`tiktok_research` and depend on `outputs/figures/` there. Moving the draft into
this repo would break all ten figure references and misdirect the repository
URL in Appendix C.

---

## 2026-07-29 — Follower counts are unobtainable via yt-dlp

`yt-dlp` returns `None` for `channel_follower_count` and
`uploader_follower_count` on TikTok, from **both** video URLs and profile
URLs. Verified directly against a live profile and video.

**Consequences:**

- `videos.creator_tier` is NULL for all 79,369 rows and cannot be backfilled.
  `VideoDownloader._calculate_creator_tier()` exists and is correct, but never
  receives a non-null input.
- `user_profiles.follower_count` is empty for the same reason.
- Any STRAIN analysis of **creator influence tier** (nano/micro/mid/macro/mega)
  — social contagion weighted by reach — is blocked until an alternate
  follower source is found. Candidates not yet tested: `tiktokapipy` (already a
  dependency), the TikTok Research API, or scraping the profile page directly.

Do not assume reach data is available when designing analyses. Engagement
counts (views/likes/comments/shares) **are** available and are the usable
proxy — see the snapshot tooling below.

---

## 2026-07-29 — Longitudinal tracking infrastructure added

New: `tracked_creators` table + `creators_due_for_check` view
(`scripts/migrations/001_tracked_creators.sql`), driven by `scripts/track.py`.

Two capabilities, both previously designed-but-dormant in the schema:

1. **Creator watchlist** — a durable list of creators to re-check on a
   per-creator cadence, with bookkeeping (`last_checked_at`,
   `consecutive_empty`, `new_videos_found`) so repeat runs only touch what is
   due. Answers "what has this cohort posted since we last looked?"
2. **Engagement snapshots** — `track.py snapshot` re-polls view/like/comment/
   share counts into `engagement_snapshots` (which existed but was empty),
   appending a row per run to build a time series. `track.py trend` reports
   growth.

**Why this matters for future papers:** creator tracking captures *what was
posted*; engagement snapshots capture *what spread*. The latter is the
diffusion measure that the methods paper's §5.8 lists as future work
("community-driven diffusion of self-diagnosis narratives"). Neither is
recoverable retroactively — a snapshot not taken is data permanently lost, so
the series only starts accumulating once `snapshot` runs regularly.

First live run showed a single video gaining **+32,600 views** since its
original collection, confirming that counts move enough to be worth sampling.

**Cohort selection caution.** Seeding a watchlist by raw post volume produces a
biased cohort — `chronic.kaleigh` alone has 2,155 videos and would dominate any
aggregate time series. `track.py seed` therefore supports `--stratified N`
(N per `primary_condition`) plus `--min-videos` / `--active-days` / `--condition`
filters, and records the exact rule in `tracked_creators.added_reason` so a
cohort remains reconstructible later. For preregistered work, state the rule
before seeding.

---

## 2026-07-29 — Ollama inference-engine version drift

All local-model runs behind the methods paper were produced on **Ollama
0.15.2** (logged per-inference in `llm_annotation_runs.ollama_version` —
this provenance logging paid off). The workstation now runs **0.20.3**.

**Consequence:** experiment 15 (`main_study_v1`, 500 chunks) must **not** be
resumed. It is ~2% complete (95 chunks, `deepseek-chat` only, stalled
2026-02-19), and finishing it on 0.20.3 would mix inference engines inside one
`experiment_id`. Because the paper's central claim is about run-to-run
*stability*, an engine change is precisely the confound that would invalidate
it. Any 500-chunk work must be a clean re-run of all models on one pinned
version, logged as a new experiment.

Keep logging `ollama_version` and `quantization`. It is what made this
detectable.

---

## 2026-07-29 — Unreported model in experiment 14 (RESOLVED: included)

`experiment_14` contains **four** complete cloud models, all on prompt version
v1, each with a full 6,000-run grid (100 chunks x 6 constructs x 2 temps x 5 runs):

| model | stability | coverage | clarity | in paper? |
|---|---|---|---|---|
| deepseek-chat | 93.5% | 99.2% | 96.7% | yes |
| **claude-haiku-4.5** | **85.7%** | **95.1%** | **91.2%** | **no** |
| gpt-5-nano | 70.8% | 98.1% | 92.3% | yes |
| minimax-m2.5 | 67.5% | 95.8% | 97.4% | yes |

`PAPER_DRAFT.md` reports N=3 and does not mention Haiku anywhere. Total 8,000
Haiku inferences exist across experiments 14, 17, 18.

Stated reason for exclusion: cost. Note that the cost is sunk — the inferences
are already collected and paid for, so including them requires no new spend.
Including Haiku does not threaten the paper's thesis: cloud mean stability
moves 77.3% → 79.4%, still below the local mean of 85.8%, so "local models
outperform cloud APIs" holds with a fourth architecture behind it.

The one genuine incompatibility: Haiku ran via the Anthropic **Batch API**, so
`processing_time_ms` logged as 0 and it cannot appear in the latency/cost table
(Table 5). That justifies its absence from Table 5 only, not from the stability
and cross-model consensus tables.

**RESOLVED 2026-07-29: included.** The manuscript now reports nine LLMs / four
cloud models / 54,282 inferences. Tables 1 and 4 recomputed, Figures 4/5/6
regenerated, prose updated throughout.

The analysis pipeline had in fact *always* included Haiku — it appears in
`outputs/experiment_14/cross_model_agreement_matrix.csv`. Only the manuscript's
hand-written tables omitted it, so no re-inference was needed.

Including it improved the paper's weakest result. Agency/control cross-model
alpha moved from 0.08 [-0.01, 0.16] (interval spanning zero — no reliable
boundary alignment claimable) to **0.20 [0.11, 0.29]**, bounded away from zero.
Certainty/hedging rose 0.69 -> 0.73 and symptom concreteness 0.60 -> 0.63. The
"local models outperform cloud APIs" claim survives (0.87 vs 0.81) with a
narrower margin. Unanimity rates fell as expected — 4/4 agreement is a stricter
bar than 3/3.

Haiku is excluded from Table 5 (timing/cost) only, because Batch API runs
logged `processing_time_ms` as 0. This is stated in §3.3.

---

## 2026-07-29 — Methods paper status

`../tiktok_research/PAPER_DRAFT.md`, 26pp built. Complete: abstract, all
sections, 5 tables with real numbers, 10 figures wired with captions,
references verified (spot-checked arXiv:2412.03796 — real).

Build: `cd ../tiktok_research && ./build_paper.sh` produces `build/paper.pdf`,
`build/paper.tex`, and `build/arxiv-submission.tar.gz` (vector PDF figures
substituted for PNGs; test-compiles standalone in two `pdflatex` passes).

**Blocking submission** (all require author input, not code):

1. Title page reads `AUTHOR NAME REQUIRED`.
2. Six `[VERIFY]` markers in the Ethics Statement / Data Availability sections,
   under a do-not-submit banner. The two substantive ones: the **IRB
   determination** (must not be asserted without knowing the real status) and
   the **data release scope** — `outputs/experiment_14/qualitative_samples.csv`
   contains raw chunk text, and releasing verbatim transcripts of identifiable
   people discussing their health carries re-identification risk and likely
   conflicts with TikTok's ToS. Safer default: release derived annotations and
   code, withhold raw transcript text, ship the collection scripts.
3. No archival DOI. Link the repo to Zenodo and cite the DOI rather than a bare
   GitHub URL.

**Known limitation, disclosed in §5.7:** all reported results come from a
100-chunk development sample (20% split). The 60% reliability split (500
chunks) remains un-run. Acceptable for an arXiv preprint given the honest
disclosure; a journal reviewer will require it.

---

## Reference: environment fixes

Things that broke and how they were fixed, so they are not re-diagnosed:

- **Postgres port.** Both projects' `.env` pointed at 5433; the cluster listens
  on **5432**. Corrected in both.
- **Collation mismatch.** Database built under glibc 2.42, OS now provides
  2.43, producing a warning on every query and risking silently wrong text
  index ordering. Fixed with `REINDEX DATABASE` +
  `ALTER DATABASE ... REFRESH COLLATION VERSION` (~7s on 550 MB). Expect this
  again after future OS upgrades.
- **`.venv` was a Windows venv** (`Lib/`, `Scripts/`) and unusable on Linux.
  Rebuilt with `uv sync`.
- **Ollama did not persist.** The systemd unit was disabled and the stock unit
  runs as user `ollama` against `/usr/share/ollama/.ollama/models`, while the
  88 GB of actual models live in `/home/scholyx/.ollama/models`. Fixed with a
  drop-in at `/etc/systemd/system/ollama.service.d/override.conf` setting
  `User=scholyx`, `OLLAMA_MODELS`, `OLLAMA_KEEP_ALIVE=30m`, `Restart=always`.
- **gemma4:31b is a poor fit** for the annotation protocol despite being newer:
  it emits 122–200 tokens to answer a one-word prompt (~10 s/task vs
  gemma3:27b's 571 ms), and needs `num_ctx` 8192 to stay on GPU — at 32k it
  loads as 33 GB and offloads 89% to CPU. A 30,000-task grid would take ~3.5
  days for this model alone. It is also a behavioral outlier against a prompt
  design that targets single-token responses.
