# Custom 5-Layer MVP in 1 Day

This plan is designed for the current repo state. The goal is to produce a
small, reviewable movie-understanding subset in one day, not a full dataset.

## Target outcome

By the end of the day, the repo should contain:

- `data/custom_5layer_mvp/all_chunks.json`
- `data/custom_5layer_mvp/coverage_report.json`
- `data/custom_5layer_mvp/movie_summary.json`
- `data/custom_5layer_mvp/review_queue.json`

The subset should be built from the existing `videorag_chunks` store because it
is the closest available source to the repo's 5-layer chunk schema.

## Command

Run from the repo `src` directory:

```bash
python -m movierag.scripts.bootstrap_custom_5layer --max-movies 3 --max-chunks 50
```

Optional explicit movie selection:

```bash
python -m movierag.scripts.bootstrap_custom_5layer \
  --movies tt0120338 tt0167260 \
  --max-chunks 40
```

## One-day workflow

### 1. Bootstrap the MVP subset

Use the script above to export the strongest chunks from
`data/pipeline_output/videorag_chunks/all_chunks.json`.

The script adds these operating fields:

- `layer_status`
- `quality_score`
- `evidence_source`
- `review_status`
- `review_priority`
- `missing_layers`
- `missing_fields`

### 2. Review the queue

Use `review_queue.json` to focus manual work on the chunks that still need:

- character cleanup
- narrative enrichment
- script evidence enrichment

Prioritize `high` first, then `medium`.

### 3. Manual enrichment rules

For each reviewed chunk, fix these fields in order:

1. `dialogue_text`
2. `characters`
3. `cast_in_scene`
4. `narrative_arc`
5. `causal_relations`
6. `script_primary_heading`
7. `screenplay_context_excerpt`

This order keeps temporal, semantic, and dialogue layers stable before adding
harder movie-level reasoning fields.

## Acceptance criteria

The one-day MVP is good enough if it reaches:

- `layer_1_temporal`: full coverage = 1.00
- `layer_2_semantic`: full coverage >= 0.85
- `layer_3_dialogue`: full coverage >= 0.85
- `layer_4_character`: full coverage >= 0.80
- `layer_5_narrative_script`: partial coverage >= 0.50

The MVP does not need perfect screenplay alignment on day 1. It needs enough
coverage to validate the pipeline, retrieval, and review loop.

## Why this works

The repo currently has stronger support for layers 1 to 4 than for full
screenplay grounding. A one-day MVP should therefore:

- reuse existing movie-centric chunks
- standardize them into a clean reviewable package
- expose the remaining gaps as a finite manual review queue

This is a practical path to a real 5-layer dataset without waiting for a full
MovieNet or screenplay-aligned corpus.
