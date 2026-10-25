# jev-structured-bench

**A reproducible benchmark for "System One"-style decision tasks on local hardware — and a case study in two evaluation-methodology bugs that inflated a 23-point gap into noise.**

[中文摘要](#中文摘要) | Hardware: 4× NVIDIA V100 SXM2 (16GB) | Model: Qwen3.8-27B W8A16 (1Cat-vLLM, TP4) | Date: 2026-10-25

---

## What this repo contains

1. **A benchmark harness** (`harness/`) that scores a 27B open-weights model on 5 standard classification datasets (MMLU, ARC-Challenge, SST-2, AG News, Banking77) under **three output-constraint mechanisms** — forced tool calling (Anthropic `/v1/messages`), `response_format=json_schema` (OpenAI `/v1/chat/completions`, the Instructor-style route), and unconstrained free text — with identical frozen prompts per dataset.
2. **Complete raw outputs** (`data/`): 1,170 per-request records with predictions, raw model text, latency, and token counts.
3. **Diagnostic probes** (`probes/`) that uncovered **two evaluation bugs** in the first run, plus a corrected re-run.
4. **The corrected results** and a full erratum.

## Why this exists

TypeSafe AI's **Jev** ("System One" model, arXiv:2609.37647) answers typed multiple-choice questions without generating text. The paper benchmarks it zero-shot on 37 datasets (346,009 requests) and scores **Qwen3.8-27B** — the same base model family we serve locally — via exact next-token probabilities as the open-model reference.

This repo asks a practical engineering question: *if you can't call Jev's API, how close can a locally-served open model get under structured-output constraints — and is the gap real?*

## Headline result (corrected, temperature-locked, case-sensitive scoring)

| Dataset | n | A: forced tool calling | B: json_schema | C: free text |
|---|---|---|---|---|
| MMLU | 150 | 68.7% | 68.0% | 74.7% |
| ARC-Challenge | 60 | 95.0% | 91.7% | 96.7% |
| SST-2 | 60 | 98.3% | 98.3% | 98.3% |
| AG News | 60 | 78.3% | 73.3% | 78.3% |
| Banking77 (77-way) | 60 | **80.0%** | 76.7%† | **80.0%** |
| **Weighted** | 390 | **80.0%** | 74.9% | **83.1%** |

† 3 of B's Banking77 "hits" were case-coercion artifacts (see Erratum 2); exact-scoring ≈ 71.7%.

**Conclusion: structured output costs ~3pp accuracy vs free text, and the two structured mechanisms are statistically indistinguishable (±1σ) once temperature is locked. The initially observed 23.4pp Banking77 gap was a methodology artifact.**

## Erratum — six bugs found after the first runs

This is the part worth reading if you build eval harnesses.

### Bug 1: temperature was never set (largest confound)

The first harness sent no `temperature` on any path. A probe (repeat the same question 3×, count output changes) showed **both endpoints default to sampling**: 2/4 questions changed under A, 3/4 under B. Path B — where the model must hand-write JSON — is hurt more by sampling noise than path A, whose output is framework-serialized. Re-running Banking77 with `temperature=0`:

| | default (sampling) | temp=0 |
|---|---|---|
| A | 70.0% | **80.0%** |
| B | 60.0% | **73.3%** |

The entire original "Banking77 is a watershed" finding was noise.

### Bug 2: case-coercion in scoring inflated path B

Banking77 has 77 options; labels were generated with `chr(65+i)` → after `Z` come punctuation and then **lowercase letters** (options 33–58 are `a`–`z`). The first harness normalized predictions with `.upper()` before comparison — so a prediction of `"j"` (a legal label) was coerced to `"J"` and could **match the wrong option's gold label**. Exact re-scoring from raw outputs: B had 8 such free hits on Banking77 (true score 40.0%, not 53.3%); A and C had zero.

The fix in `harness/jev_bench_run.py`:
- explicit `temperature=0` on every request
- `labels_for(n)`: bijective base-26 labels (A..Z, AA..AZ, BA..BY for n=77 — no collisions)
- case-sensitive scoring; parse with an exact-match extractor

### A self-correction worth recording

Mid-investigation we concluded "path B's json_schema doesn't lock its enum — it emitted lowercase letters outside the schema." That was **wrong**: the lowercase letters *are* the legal labels (Bug 2's root cause). Two controls — a single-value enum `["X"]` and a 77-value enum of full label texts — both showed B's constraint holding perfectly. The harness had lied to us twice before the mechanism did anything wrong.

### Bug 3: transport failures impersonated "parse failures"

The first filler-token probe reported a dramatic collapse — F0 75.0% → F1 65.0% → F2 65.0% → F3 57.5% — apparently proving that filler tokens destroy accuracy. Per-question audit: **every single "parse failure" was an HTTP 502 `upstream connect failed: connection timed out`** — transport-layer failures that occurred while two benchmarks were hitting the server concurrently and the harness's `urllib` was routing through the Mac system proxy. Restricting to questions that actually returned an answer: **F0 83.3% (30/36) → F1 81.3% (26/32) → F2 81.3% → F3 79.3% (23/29)** — all within noise, no collapse, no filler benefit.

Fix: retry with backoff on 5xx/timeout (never on 4xx), bypass the system proxy (`ProxyHandler({})` — which also saved ~2.5s per request), and report `transport_err` and `parse_fail` as separate columns. **Lesson: before believing a harness's "parse failure" rate, read the error strings — infrastructure failures wear a model-behavior costume.**

### Bug 5: the results file was never written (crash after the run)

A companion run of the 4 additional datasets completed all 240 questions (952s) and then **crashed while saving**: `json.dump({...}, ensure_ascii=False)` was missing the file handle. The per-question raw records were lost; only the console summary survived. The fixed harness writes the file correctly, and the datasets were re-run clean (direct connection, 0 transport errors, 0 parse failures — `data/results_1792943210.json`). **Lesson: smoke-test the save path before the long run, not after.**

### Bug 6: a refactor silently broke every request (60/60 "model errors" that were ours)

While adding retry logic to the thinking probe's `curl_post`, the helper's return type changed from *raw response string* to *parsed dict* — but the six call sites still did `json.loads(body)` on it. Every request "failed" with `TypeError: the JSON object must be str, bytes or bytearray, not dict` and was dutifully recorded as a model error: **acc 0/60, err 60** after a 45-minute run. The summary then crashed on a `ZeroDivisionError` (0 present instincts), losing the results file — again. Fixes: call sites use the dict directly; the summary guards division by zero; **results are now saved before the summary is printed** (the Bug 5 lesson, applied). **Lesson: when a run reports 100% failure, suspect the harness first — and make the save path crash-proof, because reporting code fails too.**

### Bug 4: the "dose" never doubled (tokenizer ate the fillers)

The same probe intended F1 = 16 dots vs F2 = 64 dots (4× dose). Qwen's BPE merges dot runs: both configurations added **exactly +2 prompt tokens** (`input_tokens` 1067 vs 1067, identical). F1 ≡ F2 by construction — the dose-response sweep never happened. The v2 probe uses `"hmm "` repetitions (≈1 token each: 8 / 64) plus a 512-dot arm and a rephrase arm, so the sweep is real. **Lesson: verify your manipulation changed the thing you think it changed — diff the token counts, don't trust the string length.**

## Method

- **One model, zero fine-tuning**: Qwen3.8-27B W8A16 Marlin on 1Cat-vLLM (V100/SM70 fork), TP4 NVLink, 180W power cap, CUDA 12.8 forward-compat. The only variable is the output-constraint mechanism.
- **Frozen templates**: one prompt template per dataset, options rendered as `A) … B) …`, identical across paths.
- **Path A** (this repo's home mechanism, from the [dsh-structured-output](https://github.com/dd2003-460/dsh-structured-output) plugin): Anthropic `/v1/messages` + forced tool calling (`tool_choice` pins a single function whose `input_schema` carries the dynamic enum), then the `tool_use` block is unwrapped to plain JSON.
- **Path B**: OpenAI `/v1/chat/completions` + `response_format=json_schema` — the generic open-source route (Instructor's default, vLLM native structured outputs).
- **Path C**: unconstrained generation + regex extraction of the answer letter (the paper's §3.6 "indirect implementation" baseline).
- **Thinking**: off on all paths. Verified two ways: 0/1,170 responses contained thinking artifacts, and this model/fork has **no separate reasoning channel at all** — `reasoning_content` is always empty and `reasoning_tokens` is never reported, under every toggle we tried (`chat_template_kwargs.enable_thinking`, payload `thinking:{type:enabled}`, default). Unconstrained, the model writes its reasoning *inline* in `content` (312 tokens of "Step 1… Step 2…" for 17+25). Consequence for the thinking probe: a "schema + enable_thinking" config silently degenerates to the no-thinking config, because the schema squeezes the inline reasoning out (that's why C2 was *faster* than C0).

## Token economics (the practical argument for structured output)

| Mechanism | output tokens/request | wall-clock/request |
|---|---|---|
| C free text | ~2 (letter only, after parse) | ~1.5s |
| A forced tool call | ~7–13 | 4.7–17.2s |
| B json_schema | ~7–13 | ~1.5–4s |

On this KV-bandwidth-bound 4×V100 box, decode step cost dominates; A's 3–7× latency penalty in run v1 came from the Anthropic-endpoint forwarding chain (SSE assembly in the local proxy) — **not** from the constraint. Hitting vLLM's `/v1/messages` directly (verified while building this repo's reproduce path) removes most of that gap.

## Public scores for context (mind the caliber!)

Scores in the wild use **three incompatible calibers** — do not mix them:

| Caliber | What it measures | Example |
|---|---|---|
| **Generation** (chat/completions, temp 0, parse letter) | what a deployed agent would get | this repo; Cerebellum Q2: ARC-C 95.48% (1,172 q), HellaSwag 91.15% (10,042 q), MMLU-Redux 71.21% (2,400 q) |
| **Log-likelihood** (per-option next-token prob, pick argmax) | an upper bound not requiring parse | Jev paper's Qwen3.8-27B reference; llama.cpp boards: BF16 MMLU 85.3 / ARC-C 96.8 / HellaSwag 94.3 |
| **Self-built held-out** (private test sets) | vendor/author-chosen slices | Bespoke-Nimble 324 q: Qwen3.5-9B 66.36% / Qwen3.8-27B 84.88% / Nimble-9B 90.12% / Jev 93.21%; Kev OOD 0.79–0.80 vs Jev 0.86 |

Jev itself (log-likelihood caliber): **95–99%** on IMDB/SST-2/HellaSwag/ARC, 86.7% Belebele, MMLU 91% (calculation items 94%). It beat Qwen3.8-27B on 27/37 datasets; Qwen's 9 leads all fell inside bootstrap intervals.

**The honest gap statement**: on generation-caliber ARC we reach 95.0–96.7% vs Jev's ~97.6%; on knowledge-heavy MMLU we're at ~69% vs Jev's 91% — but that MMLU comparison crosses calibers (generation vs log-likelihood) and is **not** apples-to-apples. The one real, structural deficit: our `confidence` values are self-reported, not calibrated — Jev's are calibrated and support selective prediction.

## Reproduce (10 minutes on any GPU)

The benchmark needs **only a vLLM server** — no proxy, no plugin, no special fork. We verified Path A (forced tool calling) works by hitting vLLM's Anthropic-compatible `/v1/messages` **directly**: HTTP 200, clean `tool_use` block, correct answer. The original runs went through a local proxy (`:8787`); the proxy added the 3–7× latency, not the constraint.

```bash
# 0. prerequisites: docker + an NVIDIA GPU + python3 (>=3.9, stdlib only)

# 1. configure and start vLLM (any tool-calling instruct model works)
cp .env.example .env && $EDITOR .env     # set MODEL / VLLM_API_KEY
docker compose up -d                     # waits for /health (big models take minutes)

# 2. export the same env for the harness
set -a && . ./.env && set +a

# 3. one command: download datasets -> run 3 paths -> print summary
./scripts/reproduce.sh                   # full run: 390 questions x 3 paths
./scripts/reproduce.sh --limit 30        # 5-minute smoke test

# 4. (optional) thinking/prefill experiments
python3 harness/thinking_probe.py        # C0 pure / C1 +reason / C2 full thinking / C3 two-phase / C4 logprobs
python3 harness/prefill_thinking_probe.py  # filler-token "thinking during prefill" (F0-F3)
```

**Fully containerized** (no host python needed):

```bash
docker compose up -d                          # vLLM server
docker build -t jev-bench .                   # benchmark runner image
docker run --rm --env-file .env jev-bench     # full run
docker run --rm --env-file .env jev-bench --limit 30   # smoke test
```

Environment variables (all optional, sane defaults): `VLLM_BASE` (default `http://127.0.0.1:8787`), `VLLM_API_KEY`, `VLLM_MODEL`, `BENCH_DIR`, `CONCURRENCY`.

**Honest scope note**: the *numbers* in this repo were produced on 4×V100 + the 1Cat-vLLM fork + Qwen3.8-27B W8A16 — a setup you may not have. What reproduces anywhere is the **methodology** (three constraint mechanisms, frozen templates, locked temperature, bijective labels, case-sensitive scoring) and the **bug findings** (any harness that forgets `temperature` or coerces label case will manufacture gaps that aren't there). Swap in any instruct model via `.env` to re-run the comparison on your own hardware.

## Files

```
docker-compose.yml                # one-command vLLM server (standard image, any GPU)
Dockerfile                        # benchmark runner image (fully containerized path)
.env.example                      # copy to .env; MODEL / VLLM_BASE / VLLM_API_KEY
scripts/reproduce.sh              # one-command pipeline: download -> bench -> summary
requirements.txt                  # empty: stdlib only
harness/jev_bench_download.py     # pull 5 datasets via HF datasets-server API (no datasets lib)
harness/jev_bench_download2.py    # pull 4 more datasets (hellaswag/mathqa/truthfulqa/imdb)
harness/jev_bench_run.py          # 3-path benchmark, temp-locked, bijective labels, case-sensitive,
                                  #   retry-on-transport-error, proxy-bypass, err/parse_fail split
harness/thinking_probe.py         # C0-C5 thinking-mode experiment (incl. prompt_logprobs "first instinct"
                                  #   and the JEV-paper logprob protocol via /v1/completions)
harness/prefill_thinking_probe.py # F0-F4 filler-token experiment ("think during prefill")
probes/ab_probe.py                # determinism probe + enum strictness controls
probes/ab_rerun60.py              # 60-question temperature-locked re-run
data/*.jsonl                      # evaluation sets (9 datasets, 840 questions, frozen splits)
data/results_1792928998.json      # ALL 1,170 raw records from run v1 (contains the two bugs — kept for transparency)
data/results_1792943210.json      # clean re-run of the 4 new datasets (direct connection, 0 transport errors)
data/prefill_probe_1792941546.json # filler probe v1 raw records (contains the 502 artifact — kept for transparency)
data/run2_pathA.log               # first run of the 4 new datasets via proxy (summary survives, raw lost to Bug 5)
data/full_run.log                 # run v1 console log
reports/                          # full write-up (Chinese) + interactive per-question viewer + Path A 9-dataset table
```

## Thinking during prefill (filler tokens) — papers and probe

The token-economics argument above says decode tokens are the expensive currency. Can a model "think" without spending them? A 2024–2026 line of work says yes:

| Paper | Finding |
|---|---|
| Goyal et al. 2023, *Think before you speak* (pause tokens) | models **trained** with pause tokens use those positions for extra internal computation |
| Merrill & Sabharwal 2024, *Let's Think Dot by Dot* (arXiv:2404.15758) | on synthetic tasks, filler `…` lifts a trained Llama-34M from 66% → 100%; but Claude 2 / GPT-3.5 showed **no gain** on standard QA, and exploiting fillers needs dense supervision training |
| 2026, *Reading Between the Dots* (arXiv:2607.03502) | frontier open models (DeepSeek V3, Kimi K2) **do** compute over filler regions — attention routes the question through the filler block to the answer, and KV-cache transplants at filler positions causally swap outputs |

The disagreement is about *which* models learned to use filler positions. `harness/prefill_thinking_probe.py` tests it on our stack with zero training: F0 baseline / F1 +`"hmm "`×8 / F2 +`"hmm "`×64 / F3 +512 dots (the dot-by-dot arm) / F4 question-restatement, all appended to the prompt (pure **prefill** cost, zero decode tokens), structured output forced. Watch `input_tokens` vs `output_tokens` in the results — that's the whole trade in one line. (v1 of this probe used 16 vs 64 dots — see Bug 4: BPE merged them into the same +2 tokens, so the sweep was fake.)

**The C4 result predicts the outcome**: the label-restricted argmax of the prefill distribution already agrees with the final answer 30/30 (see Thinking-mode experiment). The model decides *while reading the question* — so anything appended after the question cannot change the decision, and filler tokens have nothing to compute with. Filler "thinking" needs a model that was *trained* to use pause positions (Goyal et al.); ours wasn't.

## Thinking-mode experiment (C0–C4)

`harness/thinking_probe.py`, 60 questions (banking77 30 + arc 30), temperature locked, structured output forced except where noted:

| Config | Mechanism | Accuracy | Output tokens | Latency |
|---|---|---|---|---|
| C0 | pure structured (json_schema, single enum) | **85.0%** (51/60) | 13 | 14.8s |
| C1 | + one-line reason (low-budget CoT) | 81.7% (49/60) | 86 (×6.6) | 17.5s |
| C2 | schema + enable_thinking | 83.3% (50/60) | 10 | 10.8s |
| C3 | two-phase (unconstrained analysis → digest under schema) | 73.3% (44/60) | 315 (×24) | 60.4s |
| C4 | C0 + prompt_logprobs "first instinct" (fixed version, arc 30) | 96.7% (29/30) | 13 | 14.8s |

**Low-budget CoT loses on every axis for choice tasks**: 6.6× the tokens, 18% more latency, −3.3pp accuracy (including 2 JSON parse failures). Full two-phase thinking is worse still: −11.7pp vs C0 at 24× the tokens. C2 is a control that degenerated to C0 (no reasoning channel on this stack — see Method). The structured constraint already does the "thinking" for a decision task; making the model write it down costs tokens and buys nothing.

**The C4 finding is the deepest one**: the label-restricted argmax of the *prefill* distribution (free — the forward pass already computed it) agrees with the structured answer **30/30** on arc and scores identically (96.7%). *The model decides while reading the question; the constrained decode just voices the decision.* Consequences: (1) filler-token "thinking during prefill" cannot help on this stack, because the decision is already made by the time any appended filler is read; (2) `prompt_logprobs` gives you a **free calibrated-ish confidence signal** for selective prediction — abstain or escalate when the option distribution is flat or disagrees with the generated answer; (3) the JEV paper's log-likelihood protocol reproduces at parity with the generation protocol on this model — Jev's edge over Qwen is calibration and knowledge depth, not "choosing better".

**On hard 77-way items the logprob protocol actually beats the JSON route** (banking77, 30-question subset): first instinct 86.7% vs structured 73.3%, with 5 of 6 disagreements favoring the instinct — all five were the same near-tie pair (`card_arrival` vs `card_delivery_estimate`, semantically adjacent intents). Mechanism: the json_schema route must first emit the `{"choice": "` scaffolding, so the answer letter is generated in a *JSON-mode* context, which shifts the distribution on near-ties; the first instinct is the raw preference at the answer position, conditioned only on the question. This is exactly why Jev generates no text at all — the generation scaffolding is itself a noise source. (Honest caveats: n=30, the 5:1 split is p≈0.11, and these particular gold labels are themselves debatable; on arc — easy items, no near-ties — the two routes agree 30/30.)

## Hardware note

All runs on 4× V100 SXM2 at a 180W power cap — prefill is compute-bound (MFU 12–15%), decode is KV-bandwidth-bound (~41ms/step short-stream, ~71ms at 50K context). This shapes the token-economics argument above: on this box, **every decode token is expensive**, so shrinking outputs from ~300–800 tokens (free-text "reasoning + answer") to ~7–15 tokens (structured) is the single highest-leverage latency intervention for decision tasks.

## License

MIT. Benchmark data derived from public datasets (their respective licenses apply).

---

## 中文摘要

本仓库是一次**本地 4×V100 上"结构化输出 vs 自由文本"决策任务基准测试**的完整记录，核心价值不是跑分，而是**跑分之后抓虫的过程**：

- **被测**：Qwen3.8-27B W8A16（1Cat-vLLM，TP4，180W 功耗墙），零微调。三种输出约束机制同题对比：A=强制工具调用（`tool_choice` 锁单 function，Anthropic 端点）、B=`response_format=json_schema`（OpenAI 端点，Instructor/vLLM 原生路线）、C=自由文本+正则解析。
- **结论（修正后）**：结构化几乎不损准确率（A vs C −3.1pp）；A/B 两种结构化机制锁温后差距 ~7pp ≈ 1σ，**不显著**。Banking77 大 enum 上 A 80.0% / B 76.7%（精确口径约 71.7%）。
- **两个方法论 bug**（详见 Erratum）：① 忘了锁 temperature——两端点默认都在采样，旧报告 23.4pp 的"分水岭"几乎全是噪声；② 77 选项标签 `chr(65+i)` 超过 Z 后含小写字母，`.upper()` 评分把合法小写标签 coerced 成大写、冒充别的选项，**虚高 B 8 题**。
- **一次自我更正**：中途误判"B 的 enum 约束泄漏"，用单值 enum + 77 文本 enum 两个对照实验当场推翻——小写本就是合法标签，B 的约束完全正常。
- **公开成绩对比的三个口径陷阱**：生成口径（chat 解析字母，我们和 Cerebellum）、对数概率口径（Jev 论文的 Qwen 参照、llama.cpp 榜）、自建 held-out（Nimble 324 题）——**三者不可互比**。我们 ARC 95.0–96.7% 已逼近 Jev 的 97.6%（生成口径）；MMLU 跨口径不可比。真正结构性差距：我们的 confidence 是自称、未校准，Jev 的校准过、支持选择性拒答。
