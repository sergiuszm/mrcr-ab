#!/usr/bin/env bash
# MRCR 4-needle cells against the Strata lane on this desktop (127.0.0.1:8080).
#
#   run-strata.sh a262k [N]   config A (262144, no YaRN): 128K and 224K cells
#   run-strata.sh b524k [N]   config B (524288, YaRN 2): 128K, 224K, 350K, 480K
#
# N = samples per cell (default 20). The 128K/224K cells share one prompt cap
# (245760 = 262144 - 16384 output) in both configs, so A and B get the same
# samples there and `mrcr_ab.py compare` pairs them. Resumable: rerun the same
# command after an interruption.
set -euo pipefail
cd "$(dirname "$0")"

config="${1:?usage: run-strata.sh a262k|b524k [N]}"
n="${2:-20}"
base=http://127.0.0.1:8080
tok=/home/serchio/Models/strata/data/packs/unsloth-ud-q6_k_xl/tokenizer

stack_base="Qwen3.8-Flash-Next UD-Q6_K_XL, Strata 0.1.45, RTX 5090"
case "$config" in
    a262k) want=262144; cells="128000:245760 224000:245760"
           stack="$stack_base, 262144 context, no rope scaling (config A)" ;;
    b524k) want=524288; cells="128000:245760 224000:245760 350000:507904 480000:507904"
           stack="$stack_base, 524288 context, static YaRN factor 2 (config B)" ;;
    *) echo "unknown config: $config" >&2; exit 1 ;;
esac

read -r model ctx < <(curl -sf -m 10 "$base/v1/models" |
    python3 -I -c 'import json,sys; m=json.load(sys.stdin)["data"][0]; print(m["id"], m["meta"]["n_ctx"])')
if [[ "$ctx" != "$want" ]]; then
    echo "Strata serves n_ctx $ctx, config $config needs $want: start the right config first" >&2
    exit 1
fi

for cell in $cells; do
    target="${cell%%:*}" cap="${cell##*:}"
    label="strata-$config-4n-$((target / 1000))k"
    echo "== $label ($model, n=$n)"
    .venv/bin/python -I -u mrcr_ab.py run --base "$base" --model "$model" --label "$label" --stack "$stack" \
        --needles 4 --target-tokens "$target" --chars-per-token 4.7 --n "$n" \
        --tokenizer-dir "$tok" --max-prompt-tokens "$cap" 2>&1 | tee -a "results-mrcr/$label.log"
done
