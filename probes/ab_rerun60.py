#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ab_rerun60.py — 60 题 banking77 全量受控重跑：A+t0 / B+t0 / C+t0"""
import json, subprocess, time, re
from concurrent.futures import ThreadPoolExecutor

PROXY = "http://127.0.0.1:8787"
MODEL = "qwen3.8-27b-w8a16"
BENCH = "/Users/a1-6/WorkBuddy/2026-10-16-20-29-01/data/jev-bench"

def curl_post(url, payload, timeout=120):
    r = subprocess.run(["curl", "-s", "--noproxy", "*", "--max-time", str(timeout),
                        "-X", "POST", url, "-H", "content-type: application/json",
                        "-d", json.dumps(payload)], capture_output=True, text=True)
    return r.stdout

def build_prompt(item):
    letters = [chr(ord('A') + i) for i in range(len(item["options"]))]
    opts = "\n".join(f"{l}) {o}" for l, o in zip(letters, item["options"]))
    return ("Classify the banking intent of the following customer message. "
            "Answer with the letter of the correct option.\n\n"
            f"{item['question']}\n\nOptions:\n{opts}\n\nAnswer with only the letter.")

def call(path, item):
    letters = [chr(ord('A') + i) for i in range(len(item["options"]))]
    if path in ("A", "B"):
        if path == "A":
            payload = {"model": MODEL, "max_tokens": 128, "temperature": 0,
                       "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": build_prompt(item)}],
                       "tools": [{"name": "answer", "input_schema": {"type": "object",
                                  "properties": {"choice": {"type": "string", "enum": letters}},
                                  "required": ["choice"]}}],
                       "tool_choice": {"type": "tool", "name": "answer"}}
            url, parse = f"{PROXY}/v1/messages", "tool"
        else:
            payload = {"model": MODEL, "max_tokens": 128, "temperature": 0,
                       "messages": [{"role": "user", "content": build_prompt(item)}],
                       "response_format": {"type": "json_schema", "json_schema": {"name": "answer",
                                  "schema": {"type": "object",
                                             "properties": {"choice": {"type": "string", "enum": letters}},
                                             "required": ["choice"]}}}}
            url, parse = f"{PROXY}/v1/chat/completions", "content"
        out = curl_post(url, payload)
        try:
            body = json.loads(out)
            if parse == "tool":
                raw = next((json.dumps(c.get("input", {}), ensure_ascii=False)
                            for c in body.get("content", []) if c.get("type") == "tool_use"), None)
            else:
                raw = body["choices"][0]["message"].get("content", "")
        except Exception:
            raw = "ERR:" + out[:100]
        m = re.search(r'"choice"\s*:\s*"([^"]{1,3})"', str(raw))
        pred = m.group(1)[:1].upper() if m else None
        if pred not in letters: pred = None
        return pred, str(raw)[:60]
    else:  # C 自由文本
        payload = {"model": MODEL, "max_tokens": 24, "temperature": 0,
                   "messages": [{"role": "user", "content": build_prompt(item)}]}
        out = curl_post(f"{PROXY}/v1/chat/completions", payload)
        try:
            txt = json.loads(out)["choices"][0]["message"].get("content", "").strip()
        except Exception:
            txt = "ERR:" + out[:100]
        m = re.search(r"\b([A-Za-z])\b", txt)
        pred = m.group(1).upper() if m else None
        if pred not in letters: pred = None
        return pred, txt[:60]

if __name__ == "__main__":
    items = [json.loads(l) for l in open(f"{BENCH}/banking77.jsonl")][:60]
    print(f"=== 60 题 banking77 受控重跑（temperature=0）===", flush=True)
    for path in ["A", "B", "C"]:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(lambda it: (call(path, it), it), items))
        ok = sum(1 for (pred, _), it in res if pred == chr(ord('A') + it["answer"]))
        wrong = [(it, pred) for (pred, _), it in res if pred != chr(ord('A') + it["answer"])]
        print(f"  {path}+t0: acc={ok/len(items)*100:.1f}% ({ok}/{len(items)}) wall={time.time()-t0:.0f}s", flush=True)
        if path == "B":
            # B 的越界/大小写统计
            bad = 0
            for (pred, raw), it in res:
                m = re.search(r'"choice"\s*:\s*"([^"]{1,3})"', raw)
                if m and m.group(1) != m.group(1).upper(): bad += 1
            print(f"    B 小写越界次数: {bad}")
    print("完成", flush=True)
