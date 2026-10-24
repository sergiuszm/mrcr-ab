#!/usr/bin/env python3
"""MRCR A/B runner for OpenAI-compatible chat endpoints.

Reimplementation of the method in NDF 381350 #921 (technobuddha, 2026-10-24).
The original mrcr_ab.py was not attached to the forum post, so this follows
the published method:

- openai/mrcr, N-needle parquet files
- the n samples whose character count is closest to target-tokens *
  chars-per-token (deterministic, so every stack gets the same prompts)
- thinking on, reasoning_effort sent as request field and chat_template_kwargs
- temperature 1.0, top_p 0.95, max_tokens 16384, one draw per sample,
  sequential
- OpenAI grading: 0 without the prefix, else difflib SequenceMatcher ratio

Only dependency: pyarrow. Runs resume: samples already in the output jsonl
are skipped.

Usage:
  mrcr_ab.py run --base http://host:port --model ID --label NAME --stack TEXT [--dry-run]
  mrcr_ab.py summary results-mrcr/A.jsonl [B.jsonl ...]
  mrcr_ab.py compare results-mrcr/A.jsonl results-mrcr/B.jsonl
  mrcr_ab.py analyze [--needles N] results-mrcr/A.jsonl [B.jsonl ...]
  mrcr_ab.py scoreboard      regenerate SCORES.md from every run
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pyarrow.parquet as pq

CORRECT = 0.98
TIE = 0.02


def load_samples(data_dir, needles):
    rows = []
    files = sorted(Path(data_dir, f"{needles}needle").glob(f"{needles}needle_*.parquet"))
    if not files:
        sys.exit(f"no parquet files under {data_dir}/{needles}needle")
    for f in files:
        part = f.stem.rsplit("_", 1)[1]
        for i, r in enumerate(pq.read_table(f).to_pylist()):
            r["id"] = f"{needles}n{part}-{i}"
            rows.append(r)
    return rows


def offline_counter(tok_dir):
    """Exact prompt token counts from a Strata pack's tokenizer/ dir (GPT-2 BPE files + Jinja chat template).

    For engines without a /tokenize endpoint. Needs the tokenizers and jinja2 packages."""
    import jinja2.sandbox
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers

    d = Path(tok_dir)
    meta = json.loads((d / "tokenizer.json").read_text())
    vocab = json.loads((d / "vocab.json").read_text())
    merges = [tuple(l.split(" ")) for l in (d / "merges.txt").read_text(encoding="utf-8").split("\n") if l]
    tok = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(meta["pre_pattern"]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    # GGUF token types 3 (CONTROL) and 4 (USER_DEFINED) are matched whole, not run through BPE
    inv = {i: t for t, i in vocab.items()}
    types = json.loads((d / "token_type.json").read_text())
    tok.add_special_tokens([inv[i] for i, t in enumerate(types) if t in (3, 4) and i in inv])

    def raise_exception(message):
        raise ValueError(message)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                       extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string((d / "chat_template.jinja").read_text(encoding="utf-8"))

    def count(messages, effort):
        kw = {"enable_thinking": False} if effort in ("none", "off") else {"reasoning_effort": effort}
        text = tpl.render(messages=messages, add_generation_prompt=True, **kw)
        return len(tok.encode(text, add_special_tokens=False).ids)

    return count


def select(rows, target_tokens, cpt, n, counter=None, effort=None, max_prompt=None):
    """The n samples closest to target_tokens: by exact count with a counter, else by n_chars / cpt.

    max_prompt drops samples that would not leave room for the output, before picking; give two configs the
    same value at one depth and they get the same samples."""
    target_chars = target_tokens * cpt
    if counter is None:
        fit = [r for r in rows if max_prompt is None or r["n_chars"] / cpt <= max_prompt]
        return sorted(fit, key=lambda r: (abs(r["n_chars"] - target_chars), r["id"]))[:n]
    near = [r for r in rows if abs(r["n_chars"] - target_chars) <= 0.35 * target_chars]
    for r in near:
        r["tokens"] = counter(json.loads(r["prompt"]), effort)
    fit = [r for r in near if max_prompt is None or r["tokens"] <= max_prompt]
    return sorted(fit, key=lambda r: (abs(r["tokens"] - target_tokens), r["id"]))[:n]


def grade(response, answer, prefix):
    if not response.startswith(prefix):
        return 0.0
    response = response.removeprefix(prefix)
    answer = answer.removeprefix(prefix)
    return float(SequenceMatcher(None, response, answer).ratio())


def post(url, body, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(base, model, messages, effort, timeout):
    body = {"model": model, "messages": messages, "add_generation_prompt": True,
            "chat_template_kwargs": {"reasoning_effort": effort}}
    with post(f"{base}/tokenize", body, timeout) as r:
        return json.load(r)["count"]


def chat(base, model, messages, args):
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.effort,
        "chat_template_kwargs": {"reasoning_effort": args.effort},
    }
    content, reasoning = [], []
    usage, timings, finish, ttft = {}, None, None, None
    t0 = time.monotonic()
    with post(f"{base}/v1/chat/completions", body, args.timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            if chunk.get("timings"):
                timings = chunk["timings"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    reasoning.append(rc)
                if d.get("content"):
                    content.append(d["content"])
                if (rc or d.get("content")) and ttft is None:
                    ttft = time.monotonic() - t0
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    return {
        "content": "".join(content),
        "reasoning": "".join(reasoning),
        "usage": usage,
        "timings": timings,
        "finish_reason": finish,
        "ttft_s": round(ttft, 2) if ttft is not None else None,
        "wall_s": round(time.monotonic() - t0, 2),
    }


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}


def cmd_run(args):
    base = args.base.rstrip("/")
    counter = offline_counter(args.tokenizer_dir) if args.tokenizer_dir else None
    picked = select(load_samples(args.data_dir, args.needles), args.target_tokens, args.chars_per_token,
                    args.n, counter, args.effort, args.max_prompt_tokens)
    out = Path(args.out_dir, f"{args.label}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    have = done_ids(out)
    print(f"{len(picked)} samples, {len(have)} already done -> {out}", flush=True)
    meta = out.with_suffix(".meta.json")
    if not args.dry_run and not meta.exists():
        meta.write_text(json.dumps({
            "label": args.label, "date": time.strftime("%Y-%m-%d"), "stack": args.stack,
            "base": base, "model": args.model, "needles": args.needles,
            "target_tokens": args.target_tokens, "n": args.n,
            "selection": "tokens" if counter else f"chars/{args.chars_per_token}",
            "max_prompt_tokens": args.max_prompt_tokens, "effort": args.effort,
            "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
        }, indent=2) + "\n")

    if args.dry_run:
        counts = []
        for r in picked:
            c = r["tokens"] if counter else \
                tokenize(base, args.model, json.loads(r["prompt"]), args.effort, args.timeout)
            counts.append(c)
            print(f"{r['id']}\tchars={r['n_chars']}\ttokens={c}", flush=True)
        print(f"tokens min {min(counts)} median {statistics.median(counts):.0f} max {max(counts)}")
        return

    for k, r in enumerate(picked, 1):
        if r["id"] in have:
            continue
        messages = json.loads(r["prompt"])
        for attempt in (1, 2):
            try:
                res = chat(base, args.model, messages, args)
                break
            except (urllib.error.URLError, OSError, ValueError) as e:
                print(f"{r['id']}: attempt {attempt} failed: {e}", flush=True)
                if attempt == 2:
                    sys.exit("two failures in a row, stopping (rerun to resume)")
                time.sleep(60)
        prefix = r["random_string_to_prepend"]
        rec = {
            "id": r["id"],
            "n_chars": r["n_chars"],
            "total_messages": r["total_messages"],
            "desired_msg_index": r["desired_msg_index"],
            "date_added": r["date_added"],
            "prompt_tokens": res["usage"].get("prompt_tokens"),
            "selection_tokens": r.get("tokens"),
            "cached_tokens": (res["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
            "completion_tokens": res["usage"].get("completion_tokens"),
            "timings": res["timings"],
            "finish_reason": res["finish_reason"],
            "cut": res["finish_reason"] == "length",
            "score": grade(res["content"], r["answer"], prefix),
            "score_lstrip": grade(res["content"].lstrip(), r["answer"], prefix),
            "prefix_ok": res["content"].startswith(prefix),
            "reasoning_chars": len(res["reasoning"]),
            "content_chars": len(res["content"]),
            "ttft_s": res["ttft_s"],
            "wall_s": res["wall_s"],
            "content": res["content"],
            "reasoning": res["reasoning"],
        }
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{k}/{len(picked)}] {r['id']} tok={rec['prompt_tokens']} "
              f"score={rec['score']:.3f} prefix={rec['prefix_ok']} "
              f"finish={rec['finish_reason']} out={rec['completion_tokens']} "
              f"ttft={rec['ttft_s']}s wall={rec['wall_s']}s", flush=True)


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def stats(recs):
    scores = [r["score"] for r in recs]
    uncut = [r["score"] for r in recs if not r["cut"]]
    toks = [r["prompt_tokens"] for r in recs if r["prompt_tokens"]]
    return {
        "n": len(recs),
        "mean": statistics.mean(scores),
        "correct": sum(s >= CORRECT for s in scores),
        "prefix": sum(r["prefix_ok"] for r in recs),
        "cuts": sum(r["cut"] for r in recs),
        "mean_uncut": statistics.mean(uncut) if uncut else None,
        "n_uncut": len(uncut),
        "reasoning": statistics.median(r["reasoning_chars"] for r in recs),
        "wall": statistics.median(r["wall_s"] for r in recs),
        "tokens": (min(toks), statistics.median(toks), max(toks)) if toks else None,
        "mean_lstrip": statistics.mean(r["score_lstrip"] for r in recs),
    }


def summarize(name, recs):
    s = stats(recs)
    print(f"== {name} (n={s['n']})")
    print(f"  mean score              {s['mean']:.3f}")
    print(f"  fully correct (>=0.98)  {s['correct']}/{s['n']}")
    print(f"  correct prefix          {s['prefix']}/{s['n']}")
    print(f"  hit max_tokens (cut)    {s['cuts']}")
    if s["mean_uncut"] is not None:
        print(f"  mean score excl. cuts   {s['mean_uncut']:.3f} (n={s['n_uncut']})")
    print(f"  median reasoning chars  {s['reasoning']:.0f}")
    print(f"  median wall per request {s['wall']:.0f} s")
    if s["tokens"]:
        lo, med, hi = s["tokens"]
        print(f"  prompt tokens           {lo}..{hi} (median {med:.0f})")
    if abs(s["mean_lstrip"] - s["mean"]) > 1e-9:
        print(f"  mean score (lstrip)     {s['mean_lstrip']:.3f}")


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cmd_summary(args):
    for p in args.files:
        summarize(Path(p).stem, load(p))


def needle_match(sample, content):
    """Which needle the response reproduces: (asked ordinal, best ordinal, best ratio)."""
    msgs = json.loads(sample["prompt"])
    prefix = sample["random_string_to_prepend"]
    truth = sample["answer"].removeprefix(prefix)
    k = next(i for i, m in enumerate(msgs) if m["role"] == "assistant" and m["content"] == truth)
    ask = msgs[k - 1]["content"]
    needles = [i + 1 for i, m in enumerate(msgs[:-1])
               if m["role"] == "user" and m["content"] == ask and i + 1 < len(msgs)]
    resp = content.removeprefix(prefix)
    # autojunk off: the official grade's default autojunk drops a verbatim copy with a few curly quotes
    # normalized (’ -> ') to ~0.6, which would hide that the right needle was found
    ratios = [SequenceMatcher(None, resp, msgs[i]["content"], autojunk=False).ratio() for i in needles]
    best = max(range(len(ratios)), key=ratios.__getitem__)
    return needles.index(k) + 1, best + 1, ratios[best]


def classify(samples, recs, match):
    """(right needle, {offset: count} for wrong needles, ids that matched no needle)."""
    exact, offsets, other = 0, {}, []
    for rec in recs:
        asked, got, ratio = needle_match(samples[rec["id"]], rec["content"])
        if ratio < match:
            other.append(rec["id"])
        elif got == asked:
            exact += 1
        else:
            offsets[got - asked] = offsets.get(got - asked, 0) + 1
    return exact, offsets, other


def cmd_analyze(args):
    samples = {r["id"]: r for r in load_samples(args.data_dir, args.needles)}
    for p in args.files:
        exact, offsets, other = classify(samples, load(p), args.match)
        n = exact + sum(offsets.values()) + len(other)
        wrong = ", ".join(f"{d:+d}: {c}" for d, c in sorted(offsets.items()))
        print(f"== {Path(p).stem} (n={n}, match ratio >= {args.match})")
        print(f"  right needle            {exact}")
        print(f"  wrong needle            {sum(offsets.values())} (offset {wrong or '-'})")
        print(f"  no needle matched       {len(other)} {' '.join(other)}")


def paired(a, b):
    bmap = {r["id"]: r for r in b}
    pairs = [(r, bmap[r["id"]]) for r in a if r["id"] in bmap]
    if not pairs:
        return {"n": 0}
    diffs = [x["score"] - y["score"] for x, y in pairs]
    rng = random.Random(0)
    boots = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
    ca = [x["score"] >= CORRECT for x, _ in pairs]
    cb = [y["score"] >= CORRECT for _, y in pairs]
    only_a = sum(p and not q for p, q in zip(ca, cb))
    only_b = sum(q and not p for p, q in zip(ca, cb))
    better = sum(d >= TIE for d in diffs)
    worse = sum(d <= -TIE for d in diffs)
    return {
        "n": len(pairs), "diff": statistics.mean(diffs), "ci": (boots[249], boots[9749]),
        "both": sum(p and q for p, q in zip(ca, cb)), "only_a": only_a, "only_b": only_b,
        "mcnemar": binom_two_sided(only_a, only_a + only_b),
        "better": better, "worse": worse, "ties": len(diffs) - better - worse,
        "sign": binom_two_sided(better, better + worse),
    }


def cmd_compare(args):
    a, b = load(args.a), load(args.b)
    na, nb = Path(args.a).stem, Path(args.b).stem
    summarize(na, a)
    summarize(nb, b)
    p = paired(a, b)
    print(f"== paired {na} - {nb} (n={p['n']})")
    if not p["n"]:
        return
    print(f"  mean difference         {p['diff']:+.3f} (95% bootstrap CI [{p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}])")
    print(f"  correct: both {p['both']}, only {na} {p['only_a']}, only {nb} {p['only_b']}")
    print(f"  exact McNemar           p = {p['mcnemar']:.3f}")
    print(f"  sign test (ties <{TIE})  {na} better {p['better']}, {nb} better {p['worse']}, "
          f"ties {p['ties']}, p = {p['sign']:.3f}")


SCORES_HEAD = """# MRCR scores

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
"""


def fmt_k(x):
    return f"{round(x / 1000)}K"


def cmd_scoreboard(args):
    out_dir = Path(args.out_dir)
    runs = []
    for p in sorted(out_dir.glob("*.jsonl")):
        recs = load(p)
        if not recs:
            continue
        meta_path = p.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        needles = meta.get("needles") or int(recs[0]["id"].split("n", 1)[0])
        runs.append({"label": p.stem, "recs": recs, "meta": meta, "needles": needles, "stats": stats(recs)})
    runs.sort(key=lambda r: (r["needles"], r["stats"]["tokens"][1] if r["stats"]["tokens"] else 0, r["label"]))

    samples = {}
    lines = [SCORES_HEAD, "## Runs", "",
             "| Date | Label | Stack | Needles | Prompt tokens (median) | n | Mean | Correct | Right needle "
             "| Prefix | Cuts | Reasoning chars (median) | s/request (median) | Settings |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if r["needles"] not in samples:
            samples[r["needles"]] = {x["id"]: x for x in load_samples(args.data_dir, r["needles"])}
        exact, _, _ = classify(samples[r["needles"]], r["recs"], args.match)
        s, m = r["stats"], r["meta"]
        lo, med, hi = s["tokens"] if s["tokens"] else (0, 0, 0)
        settings = (f"{m['effort']}, T {m['temperature']}, top_p {m['top_p']}, max {m['max_tokens']}"
                    if m else "-")
        lines.append(f"| {m.get('date', '-')} | `{r['label']}` | {m.get('stack', '-')} | {r['needles']} "
                     f"| {fmt_k(lo)}-{fmt_k(hi)} ({fmt_k(med)}) | {s['n']} | {s['mean']:.3f} "
                     f"| {s['correct']}/{s['n']} | {exact}/{s['n']} | {s['prefix']}/{s['n']} | {s['cuts']} "
                     f"| {s['reasoning']:.0f} | {s['wall']:.0f} | {settings} |")

    lines += ["", "## Paired comparisons", "",
              "Every pair of runs that used the same sample ids. Mean diff is A minus B, with a 95% bootstrap "
              "interval. McNemar compares the Correct counts. The sign test compares the raw grades and counts "
              f"differences under {TIE} as ties.", "",
              "| A | B | n | Mean diff A-B (95% CI) | Correct only A / only B (McNemar p) "
              "| Better A / B / ties (sign test p) |",
              "|---|---|---|---|---|---|"]
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if {x["id"] for x in a["recs"]} != {x["id"] for x in b["recs"]}:
                continue
            p = paired(a["recs"], b["recs"])
            lines.append(f"| `{a['label']}` | `{b['label']}` | {p['n']} "
                         f"| {p['diff']:+.3f} ({p['ci'][0]:+.3f}, {p['ci'][1]:+.3f}) "
                         f"| {p['only_a']} / {p['only_b']} ({p['mcnemar']:.3f}) "
                         f"| {p['better']} / {p['worse']} / {p['ties']} ({p['sign']:.3f}) |")

    refs_path = Path(args.references)
    if refs_path.exists():
        lines += ["", "## External references", "",
                  "Published results that use the same method. They were not run here, so they are not paired "
                  "with the runs above.", "",
                  "| Source | Stack | Needles | Prompt tokens | n | Mean | Correct | Prefix | Cuts "
                  "| Reasoning chars (median) | s/request (median) | Note |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in json.loads(refs_path.read_text()):
            lines.append(f"| {x['source']} | {x['stack']} | {x['needles']} | {x['tokens']} | {x['n']} "
                         f"| {x['mean']:.3f} | {x['correct']}/{x['n']} | {x['prefix']}/{x['n']} | {x['cuts']} "
                         f"| {x['reasoning_chars']} | {x['s_per_request']} | {x.get('note', '')} |")

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"{len(runs)} runs -> {args.output}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--stack", default="-",
                   help="what is measured (model, quant, engine, hardware, context config), for SCORES.md")
    r.add_argument("--needles", type=int, default=8)
    r.add_argument("--target-tokens", type=int, default=400000)
    r.add_argument("--chars-per-token", type=float, default=4.8)
    r.add_argument("--n", type=int, default=34)
    r.add_argument("--effort", default="high")
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--max-tokens", type=int, default=16384)
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--data-dir", default="data")
    r.add_argument("--out-dir", default="results-mrcr")
    r.add_argument("--tokenizer-dir", help="Strata pack tokenizer/ dir: exact offline counts, select by tokens")
    r.add_argument("--max-prompt-tokens", type=int, help="drop samples above this prompt size before picking")
    r.add_argument("--dry-run", action="store_true", help="tokenize the selection only")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("summary")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_summary)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.set_defaults(func=cmd_compare)
    a = sub.add_parser("analyze", help="classify answers: right needle, wrong needle (offset), none")
    a.add_argument("files", nargs="+")
    a.add_argument("--needles", type=int, default=8)
    a.add_argument("--data-dir", default="data")
    a.add_argument("--match", type=float, default=0.9)
    a.set_defaults(func=cmd_analyze)
    b = sub.add_parser("scoreboard", help="regenerate SCORES.md from every run in results-mrcr/")
    b.add_argument("--out-dir", default="results-mrcr")
    b.add_argument("--data-dir", default="data")
    b.add_argument("--references", default="references.json")
    b.add_argument("--output", default="SCORES.md")
    b.add_argument("--match", type=float, default=0.9)
    b.set_defaults(func=cmd_scoreboard)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
