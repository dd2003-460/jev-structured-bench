#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prefill_thinking_probe.py — "用 prefill 过程思考"实验：filler tokens / 问题复述
================================================================================
背景（论文依据，2024-2026 一条线）：
  - Goyal et al. 2023 "Think before you speak: Training Language Models With
    Pause Tokens"：训练时插入 pause token，模型学会用这些位置做额外内部计算。
  - Merrill & Sabharwal 2024 "Let's Think Dot by Dot"（arXiv 2404.15758）：
    合成任务上 filler token（无意义 "..."）让 Llama-34M 从 66% → 100%；
    但同年证据显示 Claude2/GPT-3.5 在常规 QA 上不受益，且需要专门密集监督训练。
  - 2026 "Reading Between the Dots"（arXiv 2607.03502）：前沿开源模型
    （DeepSeek V3 / Kimi K2）确实在 filler 区做隐藏计算——注意力把问题经
    filler 区路由到答案；filler 位置的 KV-cache 移植可因果性改变输出。
    说明现代大模型"真的在用" prefill 里的 filler 位置思考。

本探针在本仓库的同一模型/同一模板上零训练实测四种配置：
  F0  无 filler（基线）
  F1  + 16 个 "."           （prompt 尾部追加，纯 prefill 成本，零 decode 成本）
  F2  + 64 个 "."
  F3  问题复述一遍再答        （同 dot-by-dot 论文的 rephrase 技巧）

关键度量：input_tokens（prefill 成本）vs output_tokens（decode 成本）分开看——
filler token 全进 input，不产生任何 output，这就是"免费思考"的账面。

用法：
  VLLM_BASE=http://127.0.0.1:8000 VLLM_API_KEY=... python3 prefill_thinking_probe.py
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8787").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b-w8a16")
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.environ.get(
    "BENCH_DIR", os.path.normpath(os.path.join(_HERE, "..", "data", "jev-bench")))
TEMPERATURE = 0

TASKS = {
    "banking77": "Classify the banking intent of the following customer message. Answer with the letter of the correct option.",
    "arc": "The following is a multiple-choice science question. Answer with the letter of the correct option.",
}


def labels_for(n: int):
    out = []
    for i in range(n):
        s, k = "", i
        while True:
            s = chr(65 + k % 26) + s
            k = k // 26 - 1
            if k < 0:
                break
        out.append(s)
    return out


def post(url: str, payload: dict, timeout: int = 180):
    data = json.dumps(payload).encode()
    headers = {"content-type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = "Bearer " + VLLM_API_KEY
        headers["x-api-key"] = VLLM_API_KEY
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def build_prompt(ds: str, item: dict, filler: str = "") -> str:
    letters = labels_for(len(item["options"]))
    opts = "\n".join(f"{l}) {o}" for l, o in zip(letters, item["options"]))
    base = (f"{TASKS[ds]}\n\n{item['question']}\n\nOptions:\n{opts}\n\n"
            f"Answer with only the letter.")
    if filler:
        base = base + "\n\n" + filler
    return base


def run_config(name: str, ds: str, item: dict, letters):
    opts_prompt = build_prompt(ds, item)
    if name == "F0":
        prompt = opts_prompt
    elif name == "F1":
        prompt = opts_prompt + "\n\n" + "." * 16
    elif name == "F2":
        prompt = opts_prompt + "\n\n" + "." * 64
    elif name == "F3":
        prompt = opts_prompt + f"\n\nRestate the question briefly, then answer.\nQuestion restated: {item['question']}"
    else:
        raise ValueError(name)
    payload = {
        "model": MODEL, "max_tokens": 128, "temperature": TEMPERATURE,
        "stream": False, "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"name": "answer", "input_schema": {
            "type": "object",
            "properties": {"choice": {"type": "string", "enum": letters}},
            "required": ["choice"]}}],
        "tool_choice": {"type": "tool", "name": "answer"},
    }
    body, dt, err = post(f"{VLLM_BASE}/v1/messages", payload)
    pred, in_tok, out_tok, raw = None, None, None, ""
    if body:
        usage = body.get("usage") or {}
        in_tok, out_tok = usage.get("input_tokens"), usage.get("output_tokens")
        for c in body.get("content", []):
            if c.get("type") == "tool_use":
                raw = json.dumps(c.get("input", {}), ensure_ascii=False)
                pred = (c.get("input") or {}).get("choice")
    return {"pred": pred, "lat": dt, "in_tok": in_tok, "out_tok": out_tok,
            "raw": raw, "err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--configs", nargs="*", default=["F0", "F1", "F2", "F3"])
    args = ap.parse_args()

    items = {}
    for ds in TASKS:
        path = os.path.join(BENCH_DIR, f"{ds}.jsonl")
        if not os.path.exists(path):
            print(f"缺少评测集 {path}，先跑 jev_bench_download.py")
            sys.exit(1)
        rows = [json.loads(l) for l in open(path) if l.strip()][:args.limit]
        for r in rows:
            r["_ds"] = ds
            items[r["id"]] = r
    qlist = list(items.values())
    print(f"共 {len(qlist)} 题 × {len(args.configs)} 配置 = {len(qlist)*len(args.configs)} 请求")

    results = {}
    for cfg in args.configs:
        def work(it):
            letters = labels_for(len(it["options"]))
            r = run_config(cfg, it["_ds"], it, letters)
            r.update({"id": it["id"], "ds": it["_ds"],
                      "gold": labels_for(len(it["options"]))[it["answer"]]})
            return r
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            rs = list(ex.map(work, qlist))
        results[cfg] = rs
        ok = sum(1 for r in rs if r["pred"] == r["gold"])
        pf = sum(1 for r in rs if not r["pred"])
        lats = [r["lat"] for r in rs if r["lat"]]
        it_ = [r["in_tok"] for r in rs if r["in_tok"]]
        ot_ = [r["out_tok"] for r in rs if r["out_tok"]]
        print(f"[{cfg}] acc={ok/len(rs)*100:.1f}% ({ok}/{len(rs)}) parse_fail={pf} "
              f"wall_avg={sum(lats)/len(lats):.2f}s in_tok={sum(it_)/len(it_):.0f} "
              f"out_tok={sum(ot_)/len(ot_):.1f}")

    out = os.path.join(BENCH_DIR, f"prefill_probe_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump({"configs": args.configs, "results": results}, f)
    print("saved:", out)


if __name__ == "__main__":
    main()
