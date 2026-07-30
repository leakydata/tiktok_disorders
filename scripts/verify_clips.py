#!/usr/bin/env python3
"""
Verify that each candidate clip is really someone describing illness hardship.

Embedding distance finds topically nearby speech, but "nearby" includes a lot
that is not on theme: product plugs, clinicians explaining anatomy, recovery
updates, unrelated chatter that happens to share vocabulary. This reads each
clip's transcript segment and asks an LLM a single yes/no question, keeping only
the ones that pass.

Runs against MiniMax (fast, no reasoning-token overhead) or any provider the
extractor supports. Verification is cached in the database so re-runs are free.

Usage:
    python scripts/verify_clips.py --csv data/lyrics/montage_suffering.csv \
        --out data/lyrics/montage_verified.csv
    python scripts/verify_clips.py --csv ... --provider ollama --model gemma3:27b
"""
import sys
import csv
import json
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from database import get_connection

SCHEMA = """
CREATE TABLE IF NOT EXISTS clip_verification (
    id          BIGSERIAL PRIMARY KEY,
    text_hash   TEXT UNIQUE NOT NULL,
    spoken_text TEXT NOT NULL,
    on_theme    BOOLEAN NOT NULL,
    reason      TEXT,
    model       TEXT,
    checked_at  TIMESTAMP NOT NULL DEFAULT now()
);
"""

PROMPT = """You judge whether a short transcript excerpt is a person describing \
the hardship of living with their own chronic illness.

Answer YES only if the speaker is describing their own experience of being ill \
-- pain, exhaustion, being bedbound, being disbelieved or dismissed, losing \
their old life, isolation, or despair caused by illness.

Answer NO for: product or supplement promotion, a clinician or coach explaining \
something, advice or tips, recovery/success updates, someone discussing another \
person's illness, general chatter, or anything where illness hardship is not \
the substance of what is said.

Respond with JSON only: {"on_theme": true|false, "reason": "<8 words max>"}

EXCERPT:
"""


def sha(text):
    import hashlib
    return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:32]


def load_cache(hashes):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT text_hash, on_theme, reason FROM clip_verification "
                    "WHERE text_hash = ANY(%s)", (list(hashes),))
        return {h: (ok, why) for h, ok, why in cur.fetchall()}


def save(h, text, ok, why, model):
    with get_connection() as conn:
        conn.cursor().execute("""
            INSERT INTO clip_verification
                (text_hash, spoken_text, on_theme, reason, model)
            VALUES (%s,%s,%s,%s,%s)
            ON CONFLICT (text_hash) DO UPDATE
              SET on_theme=EXCLUDED.on_theme, reason=EXCLUDED.reason,
                  model=EXCLUDED.model, checked_at=now()
        """, (h, text[:2000], ok, (why or '')[:200], model))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--provider', default='minimax')
    p.add_argument('--model', default=None)
    p.add_argument('--workers', type=int, default=12)
    args = p.parse_args()

    with get_connection() as conn:
        conn.cursor().execute(SCHEMA)

    rows = list(csv.DictReader(open(args.csv, encoding='utf-8')))
    print(f"{len(rows)} clip(s) to verify\n")

    texts = {sha(r['spoken_text']): r['spoken_text'] for r in rows
             if r.get('spoken_text')}
    cache = load_cache(texts.keys())
    todo = {h: t for h, t in texts.items() if h not in cache}
    print(f"{len(cache)} cached, {len(todo)} to check via {args.provider}")

    if todo:
        from extractor import SymptomExtractor
        ex = SymptomExtractor(provider=args.provider, model=args.model)

        def judge(item):
            h, text = item
            try:
                out = ex._call_model(PROMPT + text)
                d = json.loads(out[out.find('{'):out.rfind('}') + 1])
                return h, bool(d.get('on_theme')), d.get('reason', '')
            except Exception as exc:
                # A failed judgement must not silently admit an off-theme clip.
                return h, False, f"verify failed: {type(exc).__name__}"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (h, ok, why) in enumerate(pool.map(judge, todo.items()), 1):
                save(h, todo[h], ok, why, ex.model)
                cache[h] = (ok, why)
                print(f"  [{i}/{len(todo)}] {'KEEP' if ok else 'drop'}  "
                      f"{why[:34]:34} \"{todo[h][:44]}\"")

    kept, dropped = [], []
    for r in rows:
        ok, why = cache.get(sha(r.get('spoken_text', '')), (False, 'no text'))
        r['verify_reason'] = why
        (kept if ok else dropped).append(r)

    if kept:
        fields = list(kept[0].keys())
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(kept)

    print(f"\nkept {len(kept)}  dropped {len(dropped)}")
    if dropped:
        print("\nsample of dropped:")
        for r in dropped[:10]:
            print(f"    @{r['creator']:20} {r['verify_reason'][:30]:30} "
                  f"\"{r['spoken_text'][:44]}\"")
    print(f"\nWrote {args.out}")
    print(f"{len({r['creator'] for r in kept})} creator(s) survive")
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
