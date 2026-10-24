#!/usr/bin/env bash
# Download the openai/mrcr parquet files into data/<N>needle/ and verify
# their SHA-256. The files are 190-270 MB each, too large for git.
#
#   fetch-data.sh          8-needle and 4-needle sets (the ones the results use)
#   fetch-data.sh 8        one set only
#
# The revision is pinned, so a later dataset update cannot change the
# prompts under existing results. Downloads are capped at 80 MB/s.
set -euo pipefail
cd "$(dirname "$0")"

rev=f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d
declare -A sha=(
    [8needle/8needle_0.parquet]=65df601a2e0ae4a3cfb56920a6ef99f26c0de37c6b1018695e8aed684e6a94c1
    [8needle/8needle_1.parquet]=c80b19573bff1d38e1c157d6a0bdf9cfd1a8ab6372296174c9a7015e164189e3
    [4needle/4needle_0.parquet]=4d4fa3d11ce064749de3cd039eef1a621e30a81c2c9b3e64f1df37f8afeaf312
    [4needle/4needle_1.parquet]=8dfdb94a208cf3eee73c4e7ac6ee8a5ccb7236c6934c13c6c5f67c0a9928cdf3
)

sets=("$@")
[[ ${#sets[@]} -gt 0 ]] || sets=(8 4)

for n in "${sets[@]}"; do
    for part in 0 1; do
        f="${n}needle/${n}needle_${part}.parquet"
        [[ -n "${sha[$f]:-}" ]] || { echo "no pinned SHA-256 for $f" >&2; exit 1; }
        mkdir -p "data/${n}needle"
        if [[ -f "data/$f" ]] && echo "${sha[$f]}  data/$f" | sha256sum -c --quiet 2>/dev/null; then
            echo "ok (present): $f"
            continue
        fi
        curl -fSL --limit-rate 80M -o "data/$f.part" \
            "https://huggingface.co/datasets/openai/mrcr/resolve/$rev/$f"
        echo "${sha[$f]}  data/$f.part" | sha256sum -c --quiet
        mv "data/$f.part" "data/$f"
        echo "ok (downloaded): $f"
    done
done
