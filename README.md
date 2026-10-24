# mrcr-ab

A harness that measures long-context retrieval on local serving stacks with OpenAI MRCR
(`openai/mrcr`), and compares stacks on the same prompts.

The method follows NDF 381350 #921 (technobuddha, 2026-10-24). That post compared TensorFold and
exlr8-k3 for GLM-5.3-Flash at about 400K tokens. Its `mrcr_ab.py` was not attached to the post, so
this is a rewrite from the published method. The statistics reproduce the p-values in that post.

Results are in [SCORES.md](SCORES.md).

## What the test measures

Each sample is a long synthetic chat with N identical requests, for example "write a poem about
tapirs". The last turn asks for the k-th answer, verbatim and after a random prefix. For a full
grade, the model must find the right request type, count the copies in order, and reproduce the text
exactly.

The test measures precise retrieval at depth. It does not measure reasoning, coding, tool use or
language drift.

## Setup

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
./fetch-data.sh
```

`fetch-data.sh` downloads the 8-needle and 4-needle parquet files (about 900 MB) into `data/`. The
dataset revision is pinned and each file is checked against its SHA-256. The files are not in git,
because each one is larger than 100 MB.

## Run a test

Give each run a unique label and a `--stack` description:

```sh
.venv/bin/python -I mrcr_ab.py run --base http://HOST:PORT --model MODEL_ID \
    --label glm-b12x-8n-400k \
    --stack "GLM-5.3-Flash NVFP4, vLLM b12x, TP2 on 2x GX10, MAXLEN 950000"
```

The defaults reproduce the #921 set: 8 needles, the 34 samples closest to 400K tokens, effort
`high`, temperature 1.0, top_p 0.95, and `max_tokens` 16384. Requests run one at a time, with one
draw per sample.

- If a run stops, run the same command again. Samples that are already in the results are skipped.
- To see the token counts of the selection without a model call, add `--dry-run`.
- If the engine has no `/tokenize` endpoint, add `--tokenizer-dir` with the Strata pack's
  `tokenizer/` directory. The counts are then exact and offline.
- To give two configs the same samples at one depth, give both the same `--max-prompt-tokens`.

The selection is deterministic. Two stacks with the same selection arguments get the same samples.
`scoreboard` then pairs them.

The first run of a label writes `results-mrcr/<label>.meta.json` with the stack and the settings.
If the stack description is wrong, edit that file.

## Read the results

```sh
.venv/bin/python -I mrcr_ab.py summary results-mrcr/A.jsonl
.venv/bin/python -I mrcr_ab.py compare results-mrcr/A.jsonl results-mrcr/B.jsonl
.venv/bin/python -I mrcr_ab.py analyze --needles 8 results-mrcr/A.jsonl
.venv/bin/python -I mrcr_ab.py scoreboard
```

- `summary`: the mean grade, correct answers, prefix hits, cuts, reasoning length and request time.
- `compare`: both summaries, then a paired bootstrap interval, McNemar and a sign test.
- `analyze`: each answer is a right needle, a wrong needle (with its offset) or no needle.
- `scoreboard`: regenerates `SCORES.md` from all runs, `references.json` and every paired set.

After each run, regenerate `SCORES.md` with `scoreboard` and commit it with the results. Do not edit
`SCORES.md` by hand. To add a published result from somewhere else, add it to `references.json`.

## Grading

The headline number is the official OpenAI grade: 0 without the prefix, else the difflib
`SequenceMatcher` ratio. That ratio uses the default `autojunk` heuristic, which marks down some verbatim answers. For
example, a correct copy with curly quotes written as ASCII got 0.615. `analyze` and the Right needle
column use a strict comparison to show these cases. The official grade stays the headline number, so
the results stay comparable with #921 and other published runs.

## Strata context configs

`run-strata.sh a262k|b524k [N]` runs the 4-needle depth cells against a Strata server. Config A is
the default 262144 context. Config B is 524288 with static YaRN factor 2. Both configs are set in
the Strata install, outside this repo. The script checks the served context before it sends a
request.

- To use another server, set `STRATA_BASE`. The default is `http://127.0.0.1:8080`.
- To use another pack, set `STRATA_TOKENIZER` to the `tokenizer/` directory of that pack.

## Files

| Path | Content |
|---|---|
| `mrcr_ab.py` | The harness: `run`, `summary`, `compare`, `analyze` and `scoreboard` |
| `run-strata.sh` | The Strata depth cells for config A and B |
| `fetch-data.sh` | Download and verify the dataset files |
| `references.json` | Published results that are shown in `SCORES.md` |
| `SCORES.md` | Generated scoreboard |
| `results-mrcr/` | One `.jsonl` (answers and grades), `.log` and `.meta.json` per run |
| `data/` | The dataset parquet files (not in git) |

## License

MIT, see [LICENSE](LICENSE). The `openai/mrcr` dataset has its own MIT license and is not part of
this repo.
