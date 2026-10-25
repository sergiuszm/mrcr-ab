# MRCR scores

`mrcr_ab.py scoreboard` generates this file from `results-mrcr/` and `references.json`. Do not edit it by hand.
The next `scoreboard` run overwrites it.

The test is OpenAI MRCR (`openai/mrcr`). A long synthetic chat holds N identical requests, and the last
turn asks for the k-th answer, verbatim and after a random prefix. The columns are:

- **Mean**: the official OpenAI grade, averaged. The grade is 0 without the prefix, else the difflib ratio.
- **Correct**: answers with an official grade of 0.98 or more.
- **Right needle**: answers that copy the asked instance (strict difflib ratio of 0.9 or more). This column
  also counts verbatim copies that the official grade marks down, for example curly quotes written as ASCII.
- **Cuts**: answers that stopped at `max_tokens`.
- **Settings**: reasoning effort, temperature, top_p and `max_tokens`. Each sample gets one draw.

To add a run, give `mrcr_ab.py run` a `--stack` description. Then run `mrcr_ab.py scoreboard`.

## Runs

| Date | Label | Stack | Needles | Prompt tokens (median) | n | Mean | Correct | Right needle | Prefix | Cuts | Reasoning chars (median) | s/request (median) | Settings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | `strata-a262k-4n-128k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 262144 context, no rope scaling (config A) | 4 | 121K-136K (131K) | 20 | 0.950 | 19/20 | 19/20 | 20/20 | 0 | 1044 | 43 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `strata-b524k-4n-128k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 524288 context, static YaRN factor 2 (config B) | 4 | 121K-136K (131K) | 20 | 0.948 | 18/20 | 19/20 | 20/20 | 0 | 796 | 43 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `strata-a262k-4n-224k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 262144 context, no rope scaling (config A) | 4 | 212K-235K (222K) | 20 | 0.922 | 17/20 | 19/20 | 20/20 | 0 | 810 | 68 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `strata-b524k-4n-224k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 524288 context, static YaRN factor 2 (config B) | 4 | 212K-235K (222K) | 20 | 0.964 | 18/20 | 20/20 | 20/20 | 0 | 928 | 69 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `strata-b524k-4n-350k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 524288 context, static YaRN factor 2 (config B) | 4 | 319K-384K (358K) | 20 | 0.864 | 17/20 | 17/20 | 20/20 | 0 | 1042 | 108 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `strata-b524k-4n-480k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 524288 context, static YaRN factor 2 (config B) | 4 | 443K-495K (467K) | 20 | 0.762 | 15/20 | 15/20 | 20/20 | 0 | 1049 | 144 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-25 | `ds41-maya-8n-400k` | DeepSeek-V4.1-Flash-1017 Maya-S (experts IQ2_XXS/IQ3_XXS, dense Q8_0, Engram Q6_K), project-maya v1.1.27 maya-ds4, RTX 5090 + 184 GiB RAM (CPU-lane experts read in place, --mmap-experts on), 524288 context (model native 1M), DSpark off, effort high = DeepSeek Think High (Max exists) | 8 | 355K-447K (392K) | 34 | 0.413 | 13/34 | 16/34 | 27/34 | 1 | 10323 | 254 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `glm-b12x-8n-400k` | GLM-5.3-Flash NVFP4 (local-inference-lab, snapshot 520de24e), vLLM b12x nightly-20260907 with MTP, TP2 on 2x GX10, MAXLEN 950000 | 8 | 355K-448K (392K) | 34 | 0.389 | 12/34 | 12/34 | 28/34 | 6 | 4338 | 310 | high, T 1.0, top_p 0.95, max 16384 |
| 2026-10-24 | `strata-b524k-8n-400k` | Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090, 524288 context, static YaRN factor 2 (config B) | 8 | 371K-467K (410K) | 34 | 0.522 | 16/34 | 16/34 | 34/34 | 0 | 1539 | 130 | high, T 1.0, top_p 0.95, max 16384 |

## Paired comparisons

Every pair of runs that used the same sample ids. Mean diff is A minus B, with a 95% bootstrap interval. McNemar compares the Correct counts. The sign test compares the raw grades and counts differences under 0.02 as ties.

| A | B | n | Mean diff A-B (95% CI) | Correct only A / only B (McNemar p) | Better A / B / ties (sign test p) |
|---|---|---|---|---|---|
| `strata-a262k-4n-128k` | `strata-b524k-4n-128k` | 20 | +0.002 (-0.137, +0.144) | 1 / 0 (1.000) | 1 / 1 / 18 (1.000) |
| `strata-a262k-4n-224k` | `strata-b524k-4n-224k` | 20 | -0.043 (-0.128, +0.000) | 0 / 1 (1.000) | 0 / 1 / 19 (1.000) |
| `ds41-maya-8n-400k` | `glm-b12x-8n-400k` | 34 | +0.023 (-0.143, +0.187) | 5 / 4 (1.000) | 12 / 5 / 17 (0.143) |
| `ds41-maya-8n-400k` | `strata-b524k-8n-400k` | 34 | -0.110 (-0.284, +0.065) | 4 / 7 (0.549) | 5 / 12 / 17 (0.143) |
| `glm-b12x-8n-400k` | `strata-b524k-8n-400k` | 34 | -0.133 (-0.295, +0.033) | 3 / 7 (0.344) | 3 / 16 / 15 (0.004) |

## External references

Published results that use the same method. They were not run here, so they are not paired with the runs above.

| Source | Stack | Needles | Prompt tokens | n | Mean | Correct | Prefix | Cuts | Reasoning chars (median) | s/request (median) | Note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NDF 381350 #921 (technobuddha, 2026-10-24) | GLM-5.3-Flash, TensorFold (Mia AI Lab recipe v0.6.0), DENSE=fp8, KV fp8, DFlash2 k=7, SPILL_GIB=64, TP2 on 2x GX10 | 8 | 355K-448K (~392K) | 34 | 0.548 | 18/34 | 29/34 | 4 | 3.6K | 295 | effort high, T 1.0, top_p 0.95, max 16384; vs exlr8-k3 paired: +0.276 (+0.099, +0.462), McNemar 12/2 p=0.013 |
| NDF 381350 #921 (technobuddha, 2026-10-24) | GLM-5.3-Flash exlr8 k3 (kindlingai, mmastrac/spark-glm53:v29 = vLLM 0.30.1rc1 + mentat 0.17.1), MAX_MODEL_LEN 500000, TP2 on 2x GX10 | 8 | 355K-448K (~392K) | 34 | 0.273 | 8/34 | 28/34 | 5 | 5.9K | 214 | effort high, T 1.0, top_p 0.95, max 16384; very likely the same 34 prompts as our 8n runs, ids unproven |
