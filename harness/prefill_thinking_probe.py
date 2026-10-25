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

本探针在本仓库的同一模型/同一模板上零训练实测五种配置（2026-10-25 第二版）：
  F0  无 filler（基线）
  F1  + "hmm" × 8        （约 8 个 filler token）
  F2  + "hmm" × 64       （约 64 token，剂量 ×8）
  F3  + "." × 512        （忠实复刻 dot-by-dot 论文的点串 filler）
  F4  问题复述一遍再答     （同论文的 rephrase 技巧）

第一版作废的两个教训（已写进报告 §0 勘误）：
  1. F1/F2 原用 "."*16 / "."*64 —— Qwen BPE 把点串合并，两者都只 +2 token
     （in_tok 1067 vs 1067 完全相同），"剂量翻 4 倍"从未发生，F1=F2 是必然结果。
     改用量词化 1:1 的 "hmm" 重复，才有真实剂量扫描。
  2. 原版把 HTTP 502（代理超时）记成 parse_fail —— 两个 benchmark 并发打服务器时
     8787 狂返 502，准确率出现"75%→57.5% 崩盘"假象。实为传输错误，与模型无关。
     现在 post() 带退避重试，汇总把 transport_err 与 parse_fail 分开报。

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

# 绕过 Mac 系统代理（http_proxy=127.0.0.1:51647）：该代理对 tailscale 地址时通时断，
# 失败时报 "upstream connect failed: connection timed out"，曾把传输错误误记成
# 模型解析失败（2026-10-25 第一版作废的直接原因）。直连还每请求省 ~2.5s。
_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

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


def post(url: str, payload: dict, timeout: int = 180, retries: int = 3):
    data = json.dumps(payload).encode()
    headers = {"content-type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = "Bearer " + VLLM_API_KEY
        headers["x-api-key"] = VLLM_API_KEY
    last, t0 = None, time.time()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers=headers)
        t0 = time.time()
        try:
            with _NO_PROXY_OPENER.open(req, timeout=timeout) as r:
                return json.load(r), time.time() - t0, None
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200]!r}"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        # 4xx 是请求问题重试无意义；5xx/超时/连接失败退避重试
        # （2026-10-25 教训：两个 benchmark 并发打服务器时 8787 代理狂返 502，
        #  不重试就会把"传输错误"误记成"模型解析失败"，准确率全是假崩）
        if last.startswith("HTTP 4"):
            break
        time.sleep(2 * (attempt + 1))
    return None, time.time() - t0, "transport: " + str(last)


def is_transport(err):
    return bool(err) and str(err).startswith("transport:")


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
    # 剂量扫描（2026-10-25 第一版作废重跑）：
    # 第一版 F1/F2 用 "."*16 / "."*64，Qwen BPE 把点串合并成 ~2 token，
    # 两个配置 in_tok 完全相同（1067 vs 1067）→ "剂量翻 4 倍"根本没发生，F1=F2 是必然。
    # 这一版用 "hmm " 重复（每词约 1 token）做真实线性剂量：8 / 64 / 512 token。
    if name == "F0":
        prompt = opts_prompt
    elif name == "F1":
        prompt = opts_prompt + "\n\n" + "hmm " * 8
    elif name == "F2":
        prompt = opts_prompt + "\n\n" + "hmm " * 64
    elif name == "F3":
        prompt = opts_prompt + "\n\n" + "." * 512      # 忠实复刻 dot-by-dot 论文的 filler
    elif name == "F4":
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
    ap.add_argument("--configs", nargs="*", default=["F0", "F1", "F2", "F3", "F4"])
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
        n_tr = sum(1 for r in rs if is_transport(r.get("err")))
        ans = [r for r in rs if not is_transport(r.get("err"))]
        ok_all = sum(1 for r in rs if r["pred"] == r["gold"])
        ok_ans = sum(1 for r in ans if r["pred"] == r["gold"])
        pf = sum(1 for r in ans if not r["pred"])
        lats = [r["lat"] for r in rs if r["lat"]]
        it_ = [r["in_tok"] for r in rs if r["in_tok"]]
        ot_ = [r["out_tok"] for r in rs if r["out_tok"]]
        print(f"[{cfg}] acc={ok_all/len(rs)*100:.1f}% ({ok_all}/{len(rs)}) "
              f"answered_acc={ok_ans/len(ans)*100:.1f}% ({ok_ans}/{len(ans)}) "
              f"transport_err={n_tr} parse_fail={pf} "
              f"wall_avg={sum(lats)/len(lats):.2f}s in_tok={sum(it_)/len(it_):.0f} "
              f"out_tok={sum(ot_)/len(ot_):.1f}", flush=True)

    out = os.path.join(BENCH_DIR, f"prefill_probe_{int(time.time())}.json")
    with open(out, "w") as f:
        json.dump({"configs": args.configs, "results": results}, f)
    print("saved:", out)


if __name__ == "__main__":
    main()
