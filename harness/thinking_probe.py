#!/usr/bin/env python3
"""Thinking-mode controlled experiment for structured decision tasks.

Six configurations, same questions, same frozen template, temperature=0:

  C0  structured only          — response_format json_schema, choice enum, thinking off
  C1  + one-line reason        — same + {choice, confidence, reason} (low-budget CoT)
  C2  full thinking            — same schema but thinking enabled (Qwen enable_thinking=true)
  C3  two-phase                — phase 1: unconstrained generation (model thinks freely);
                                 phase 2: original question + phase-1 analysis re-prefilled,
                                 structured output forced (thinking absorbed in prefill)
  C4  C0 + prompt_logprobs     — same request as C0, plus top-k logprobs at the last prompt
                                 position ("first instinct" after reading the question).
                                 Zero extra decode cost: the prefill forward pass already
                                 computed that distribution. Tests free selective-prediction.
  C5  paper protocol           — JEV arXiv 2609.37647 口径的本地复刻：/v1/completions
                                 （无 chat 模板）raw prompt 正好结束在答题位，
                                 prompt_logprobs top-20，只在选项字母 token 内重归一化取
                                 argmax。零额外 decode 成本。

Metrics per config: accuracy, wall-clock, output tokens, thinking tokens (C2/C3 phase1),
and (C4/C5) agreement between first-instinct argmax and the structured answer.

C4 bug found 2026-10-25（第一版 C4 输出作废，已修）：
  1. vLLM 的 prompt_logprobs dict 的 **key 是 token ID 字符串**（"271"），不是 token 文本；
     文本在 value 的 `decoded_token` 字段里。旧代码 best[0] 取到 "271"，永远对不上字母。
  2. 更根本的：chat 模板让最后 prompt 位置的 top1 恒为 '\n\n'（logprob=0.0，确定性），
     答案字母 ' A' 在 rank 4——"全局 argmax"根本不是答案。论文协议的正确做法是
     **只在选项字母 token 集合内重归一化**再取 argmax。
  3. top_logprobs=5 覆盖不全（ banking77 的 "BY" 等标签 token 排在很后面），改 top-20。
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import os
import argparse
BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8787").rstrip("/")
API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b-w8a16")
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get("BENCH_DIR", os.path.normpath(os.path.join(_HERE, "..", "data", "jev-bench")))
OUT = f"{BENCH}/thinking_probe_results.json"
# prompt_logprobs 请求在这个 KV-bound 盒子上极慢（对整个 prompt 多跑一次前向），
# C4/C5 可用这两个环境变量缩题量（如 N_ARC=30 N_BANK=0 只跑 arc）。
N_BANK = int(os.environ.get("N_BANK", "30"))
N_ARC = int(os.environ.get("N_ARC", "30"))
CONCURRENCY = int(os.environ.get("CONCURRENCY", "2"))


def curl_post(url, payload, timeout=300):
    r = subprocess.run(
        ["curl", "-s", "--noproxy", "*", "--max-time", str(timeout),
         "-X", "POST", url, "-H", "content-type: application/json",
         *(["-H", f"Authorization: Bearer {API_KEY}"] if API_KEY else []),
         "-d", json.dumps(payload)],
        capture_output=True, text=True)
    return r.stdout


def labels_for(n):
    out, i = [], 0
    while len(out) < n:
        x, m = i, ""
        while True:
            m = chr(65 + x % 26) + m
            x = x // 26 - 1
            if x < 0:
                break
        out.append(m)
        i += 1
    return out


def render_prompt(item):
    letters = labels_for(len(item["options"]))
    opts = "\n".join(f"{letters[i]}) {t}" for i, t in enumerate(item["options"]))
    return (f"The following is a multiple-choice question. Answer with the letter "
            f"of the correct option.\n\n{item['question']}\n\nOptions:\n{opts}\n\n"
            f"Answer with only the letter."), letters


def schema_choice(letters):
    return {"type": "json_schema", "json_schema": {"name": "answer", "schema": {
        "type": "object",
        "properties": {"choice": {"type": "string", "enum": letters}},
        "required": ["choice"]}}}


def schema_choice_reason(letters):
    return {"type": "json_schema", "json_schema": {"name": "answer", "schema": {
        "type": "object",
        "properties": {
            "choice": {"type": "string", "enum": letters},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}},
        "required": ["choice", "confidence", "reason"]}}}


def extract_choice(raw, letters):
    m = re.search(r'"choice"\s*:\s*"([^"]{1,4})"', str(raw))
    if m and m.group(1) in letters:
        return m.group(1)
    m = re.search(r"\b([A-Za-z]{1,3})\b", str(raw))
    if m and m.group(1) in letters:
        return m.group(1)
    return None


def usage_of(body):
    u = (body or {}).get("usage") or {}
    comp = u.get("completion_tokens_details") or {}
    return u.get("completion_tokens"), u.get("prompt_tokens"), comp.get("reasoning_tokens")


def run_one_cfg(cfg, item):
    prompt, letters = render_prompt(item)
    t0 = time.time()
    rec = {"id": item["id"], "cfg": cfg}
    try:
        if cfg == "C0":
            body = curl_post(f"{BASE}/v1/chat/completions", {
                "model": MODEL, "max_tokens": 64, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": schema_choice(letters)})
            j = json.loads(body)
            raw = j["choices"][0]["message"].get("content", "")
            rec.update(raw=raw, pred=extract_choice(raw, letters),
                       out_tok=usage_of(j)[0], think_tok=usage_of(j)[2])
        elif cfg == "C1":
            body = curl_post(f"{BASE}/v1/chat/completions", {
                "model": MODEL, "max_tokens": 256, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": schema_choice_reason(letters)})
            j = json.loads(body)
            raw = j["choices"][0]["message"].get("content", "")
            rec.update(raw=raw, pred=extract_choice(raw, letters),
                       out_tok=usage_of(j)[0], think_tok=usage_of(j)[2])
        elif cfg == "C2":
            # thinking on + same hard schema (schema governs final JSON only)
            body = curl_post(f"{BASE}/v1/chat/completions", {
                "model": MODEL, "max_tokens": 3072, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": schema_choice(letters),
                "chat_template_kwargs": {"enable_thinking": True}})
            j = json.loads(body)
            msg = j["choices"][0]["message"]
            raw = msg.get("content", "")
            rec.update(raw=raw[:600], think_raw=str(msg.get("reasoning_content") or "")[:600],
                       pred=extract_choice(raw, letters),
                       out_tok=usage_of(j)[0], think_tok=usage_of(j)[2] or
                       (len(str(msg.get("reasoning_content") or "")) // 4))
        elif cfg == "C3":
            # phase 1: free thinking (no constraint)
            p1 = (prompt + "\n\nThink step by step. After your analysis, write the final "
                  "answer on its own line as: LETTER: X")
            b1 = curl_post(f"{BASE}/v1/chat/completions", {
                "model": MODEL, "max_tokens": 1536, "temperature": 0,
                "messages": [{"role": "user", "content": p1}]}, timeout=400)
            j1 = json.loads(b1)
            t1 = j1["choices"][0]["message"].get("content", "")
            m = re.search(r"LETTER:\s*([A-Za-z]{1,3})", t1)
            letter1 = m.group(1) if m else None
            # phase 2: digest analysis under hard schema (analysis rides in prefill)
            p2 = (prompt + "\n\nYour previous analysis of this exact question:\n"
                  + t1[-2600:] + "\n\nBased on the analysis above, give the final answer.")
            b2 = curl_post(f"{BASE}/v1/chat/completions", {
                "model": MODEL, "max_tokens": 64, "temperature": 0,
                "messages": [{"role": "user", "content": p2}],
                "response_format": schema_choice(letters)})
            j2 = json.loads(b2)
            raw = j2["choices"][0]["message"].get("content", "")
            rec.update(raw=raw, think_raw=t1[:600], phase1_letter=letter1,
                       pred=extract_choice(raw, letters),
                       out_tok=(usage_of(j1)[0] or 0) + (usage_of(j2)[0] or 0),
                       think_tok=usage_of(j1)[0], two_phase=True)
        elif cfg == "C4":
            body = curl_post(f"{BASE}/v1/chat/completions", {
                "model": MODEL, "max_tokens": 64, "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": schema_choice(letters),
                "prompt_logprobs": 20})
            j = json.loads(body)
            raw = j["choices"][0]["message"].get("content", "")
            lp = j.get("prompt_logprobs") or []
            instinct, top1_tok = None, None
            if lp:
                last = lp[-1] or {}
                # 全局 top1（记录用：预期是 '\n\n' 之类模板前缀 token，不是答案）
                best = max(last.items(), key=lambda kv: kv[1].get("logprob", -99)
                           if isinstance(kv[1], dict) else -99)
                top1_tok = (best[1] or {}).get("decoded_token") if isinstance(best[1], dict) else best[0]
                # 论文协议：只在选项字母 token 内找 argmax（key 是 token ID，文本在 decoded_token）
                cands = [(v.get("logprob", -99), (v.get("decoded_token") or "").strip())
                         for v in last.values()
                         if isinstance(v, dict) and (v.get("decoded_token") or "").strip() in letters]
                if cands:
                    instinct = max(cands)[1]
            rec.update(raw=raw, pred=extract_choice(raw, letters), instinct=instinct,
                       top1_tok=repr(top1_tok),
                       agree=(instinct is not None and instinct == extract_choice(raw, letters)),
                       out_tok=usage_of(j)[0], think_tok=usage_of(j)[2])
        elif cfg == "C5":
            # JEV 论文口径本地复刻：raw completion（无 chat 模板），prompt 结束在答题位，
            # 最后位置 top-20 logprobs，选项字母 token 内重归一化取 argmax。
            raw_prompt = (prompt + "\n\nAnswer: ")
            body = curl_post(f"{BASE}/v1/completions", {
                "model": MODEL, "max_tokens": 1, "temperature": 0,
                "prompt": raw_prompt, "prompt_logprobs": 20})
            j = json.loads(body)
            lp = j.get("prompt_logprobs") or []
            instinct, covered, top1_tok = None, 0, None
            if lp:
                last = lp[-1] or {}
                best = max(last.items(), key=lambda kv: kv[1].get("logprob", -99)
                           if isinstance(kv[1], dict) else -99)
                top1_tok = (best[1] or {}).get("decoded_token") if isinstance(best[1], dict) else best[0]
                cands = [(v.get("logprob", -99), (v.get("decoded_token") or "").strip())
                         for v in last.values()
                         if isinstance(v, dict) and (v.get("decoded_token") or "").strip() in letters]
                covered = len(cands)
                if cands:
                    instinct = max(cands)[1]
            rec.update(instinct=instinct, labels_in_top20=covered,
                       top1_tok=repr(top1_tok), out_tok=1, think_tok=0)
        rec["lat"] = round(time.time() - t0, 2)
        rec["gold"] = letters[item["answer"]]
        rec["ok"] = (rec.get("pred") == rec["gold"])
    except Exception as e:
        rec.update(error=str(e)[:200], ok=False, lat=round(time.time() - t0, 2))
    return rec


def rec_pred_check(raw, letters):
    return extract_choice(raw, letters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=["C0", "C1", "C2", "C3", "C4", "C5"])
    args = ap.parse_args()
    items = []
    for ds, n in (("banking77", N_BANK), ("arc", N_ARC)):
        rows = [json.loads(l) for l in open(f"{BENCH}/{ds}.jsonl")][:n]
        for r in rows:
            r["ds"] = ds
        items += rows
    cfgs = args.configs
    # 部分配置跑时单独存文件，避免覆盖全量结果（C4 第一版有 bug，重跑不能抹掉 C0-C3）
    global OUT
    if sorted(cfgs) != sorted(["C0", "C1", "C2", "C3", "C4", "C5"]):
        OUT = f"{BENCH}/thinking_probe_{'_'.join(cfgs)}.json"
    results = {}
    for cfg in cfgs:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            rs = list(ex.map(lambda it: run_one_cfg(cfg, it), items))
        for r in rs:
            r["ds"] = next(it["ds"] for it in items if it["id"] == r["id"])
        results[cfg] = rs
        ok = sum(1 for r in rs if r.get("ok"))
        lats = [r["lat"] for r in rs]
        toks = [r.get("out_tok") or 0 for r in rs]
        thk = [r.get("think_tok") or 0 for r in rs]
        err = sum(1 for r in rs if r.get("error"))
        print(f"{cfg}: acc={ok}/{len(rs)} ({ok/len(rs)*100:.1f}%) "
              f"err={err} wall={time.time()-t0:.0f}s "
              f"avg_lat={sum(lats)/len(lats):.1f}s avg_out_tok={sum(toks)/len(toks):.0f} "
              f"avg_think_tok={sum(thk)/len(thk):.0f}", flush=True)
        if cfg in ("C4", "C5"):
            ag = sum(1 for r in rs if r.get("agree"))
            has = sum(1 for r in rs if r.get("instinct"))
            ok_inst = sum(1 for r in rs if r.get("instinct") == r.get("gold"))
            cov = [r.get("labels_in_top20") or 0 for r in rs]
            print(f"  {cfg} instinct: present {has}/{len(rs)}, agree-with-structured {ag}, "
                  f"instinct-alone acc {ok_inst}/{len(rs)} "
                  f"({ok_inst/len(rs)*100:.1f}%), labels_in_top20 avg {sum(cov)/len(cov):.1f}",
                  flush=True)
        if cfg == "C3":
            agree13 = sum(1 for r in rs if r.get("phase1_letter") and
                          r.get("phase1_letter") == r.get("pred"))
            print(f"  C3 phase1-letter == final: {agree13}/{len(rs)}", flush=True)
    json.dump(results, open(OUT, "w"), ensure_ascii=False)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
