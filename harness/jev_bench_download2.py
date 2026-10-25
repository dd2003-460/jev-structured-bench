#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jev_bench_download2.py — 第二批 4 个数据集（hellaswag/mathqa/truthfulqa/imdb）
-> 统一 schema {id, source, question, options, answer, dataset}，落 data/jev-bench/。

来源：
  hellaswag  Rowan/hellaswag default/validation（HF datasets-server rows API，官方原版 3953 题）
  mathqa     官方原始 MathQA.zip -> test.json（datasets-server 不支持该 loading script，
             直接拉 math-qa.github.io 原始数据；options 字段 "a ) 38 , b ) ..." 解析成选项列表）
  truthfulqa truthfulqa/truthful_qa config=multiple_choice split=validation（mc1_targets，5 选项）
  imdb       SetFit/imdb default/test（与既有 SetFit/sst2 同口径）

统一 schema 与既有 5 集完全一致：id / question / options / answer(0-based int) / dataset。
"""
import json, os, re, sys, time, random, urllib.request, urllib.parse

BENCH = "/Users/a1-6/WorkBuddy/2026-10-16-20-29-01/data/jev-bench"
N = 60  # 每集 60 题（题源都 >= 60，够用）
SEED = 20261025
os.makedirs(BENCH, exist_ok=True)


def fetch_rows(dataset, config, split, offset, length, retries=4):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(dataset, safe='')}"
           f"&config={urllib.parse.quote(config, safe='')}&split={split}&offset={offset}&length={length}")
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            wait = 3 * (i + 1)
            print(f"    retry {i+1}/{retries} after {type(e).__name__}: {e} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    return None


def save(key, items):
    items = items[:N]
    for i, it in enumerate(items):
        assert isinstance(it["answer"], int) and 0 <= it["answer"] < len(it["options"]), \
            f"{key} row {i} bad answer: {it['answer']} / {len(it['options'])} opts"
        it["dataset"] = key
    out = f"{BENCH}/{key}.jsonl"
    with open(out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  saved {len(items)} -> {out}", flush=True)


# ---------- hellaswag ----------
def dl_hellaswag():
    print("=== hellaswag (Rowan/hellaswag validation) ===", flush=True)
    items, got, offset = [], 0, 0
    while got < N:
        d = fetch_rows("Rowan/hellaswag", "default", "validation", offset, min(100, N - got))
        if not d:
            print("  FAILED"); break
        for row in d.get("rows", []):
            r = row["row"]
            items.append({
                "id": f"hellaswag-{got}",
                "question": f"{r['ctx']} {r['endings'][0]}".strip(),  # 占位，下面整体重写
                "options": r["endings"],
                "answer": int(r["label"]),
            })
            # question 用完整 ctx 语境（不含任何 ending），选项是 4 个 ending
            items[-1]["question"] = r["ctx"].strip() + "\nWhich ending is most likely?"
            got += 1
            if got >= N: break
        offset += len(d.get("rows", []))
        if not d.get("rows"): break
    save("hellaswag", items)


# ---------- mathqa（官方原始 zip 已解到 /tmp/MathQA_raw/test.json） ----------
def dl_mathqa():
    print("=== mathqa (official MathQA.zip test.json) ===", flush=True)
    raw = json.load(open("/tmp/MathQA_raw/test.json"))
    items = []
    for i, r in enumerate(raw):
        if len(items) >= N: break
        # options 形如 "a ) 38 , b ) 27.675 , c ) 30 , d ) data inadequate , e ) none of these"
        pairs = re.findall(r"([a-e])\s*\)\s*(.*?)(?=\s*,\s*[a-e]\s*\)|$)", r["options"])
        letters, texts = [p[0] for p in pairs], [p[1].strip() for p in pairs]
        if len(letters) != len(set(letters)) or not letters:
            print(f"  WARN row {i} skip (option parse): {r['options']!r}")
            continue
        letter2idx = {l: i2 for i2, l in enumerate(letters)}
        ans_letter = r["correct"].strip().lower()
        if ans_letter not in letter2idx:
            print(f"  WARN row {i} skip (correct={ans_letter!r} not in {letters})")
            continue
        items.append({
            "id": f"mathqa-{len(items)}",
            "question": r["Problem"].strip(),
            "options": texts,
            "answer": letter2idx[ans_letter],
        })
    save("mathqa", items)


# ---------- truthfulqa (multiple_choice, mc1) ----------
# 坑：mc1_targets 的正确答案恒在位置 0（HF 数据集设计如此），不打破位置偏置
# 则模型蒙 A 即满分、评测失真。对策：固定种子打乱选项顺序（题目内容不变，
# 只破位置偏置），SEED 固定可复现。
def dl_truthfulqa():
    print("=== truthfulqa (truthfulqa/truthful_qa multiple_choice/validation, 选项已按 SEED 打乱) ===", flush=True)
    items, got, offset = [], 0, 0
    rng = random.Random(SEED)
    while got < N:
        d = fetch_rows("truthfulqa/truthful_qa", "multiple_choice", "validation", offset, min(100, N - got))
        if not d:
            print("  FAILED"); break
        for row in d.get("rows", []):
            r = row["row"]
            m = r["mc1_targets"]
            true_idx = m["labels"].index(1)
            order = list(range(len(m["choices"])))
            rng.shuffle(order)
            items.append({
                "id": f"truthfulqa-{got}",
                "question": r["question"].strip(),
                "options": [m["choices"][i] for i in order],
                "answer": order.index(true_idx),
            })
            got += 1
            if got >= N: break
        offset += len(d.get("rows", []))
        if not d.get("rows"): break
    save("truthfulqa", items)


# ---------- imdb (SetFit/imdb test) ----------
# 坑：SetFit/imdb test 前 12500 条全 negative、offset>=12500 全 positive（各抽 50 行验证过），
# 前 60 题全 A 会使评测失真（蒙 A 满分）。对策：均衡采样 —— 前半 30 题取 negative
# (offset 0..29)、后半 30 题取 positive (offset 12500..12529)。
def dl_imdb():
    print("=== imdb (SetFit/imdb test, 30 negative + 30 positive 均衡) ===", flush=True)
    half = N // 2
    items = []
    for off in (0, 12500):
        d = fetch_rows("SetFit/imdb", "default", "test", off, half)
        if not d or not d.get("rows"):
            print("  FAILED at offset", off); return
        for row in d["rows"]:
            r = row["row"]
            items.append({
                "id": f"imdb-{len(items)}",
                "question": r["text"].strip(),
                "options": ["negative", "positive"],
                "answer": int(r["label"]),
            })
    save("imdb", items)


if __name__ == "__main__":
    for fn in [dl_hellaswag, dl_mathqa, dl_truthfulqa, dl_imdb]:
        fn()
    print("DONE", flush=True)
