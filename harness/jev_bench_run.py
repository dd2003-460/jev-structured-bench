#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jev_bench_run.py — JEV 风格 benchmark：本地 skill 机制 vs 公开同类机制 vs 自由文本
================================================================================
背景：JEV（arXiv 2609.37647, TypeSafe AI "System One" 决策模型）= 不生成文本、
给定选项只做选择/打分的专用决策模型；论文在 37 个公开数据集零样本评测，
对照组含 Qwen3.8-27B。本脚本复刻其"多选决策"协议，在本地 1Cat-vLLM
(qwen3.8-27b-w8a16, TP4 V100) 上对比三条输出路径：

  Path A  本地 skill 机制（structured-dispatch / DSH 插件）：
          Anthropic /v1/messages + 强制工具调用（tool_choice 锁单 function，
          input_schema = 动态 enum）。与 lib/structured.mjs 的 injectStructured
          同构，仅 schema 按题目动态生成（agents.json 注册表是静态 enum）。

  Path B  公开同类机制（Instructor / vLLM 原生 structured outputs）：
          OpenAI /v1/chat/completions + response_format(json_schema, 单 enum)。

  Path C  自由文本基线（论文 §3.6 的"间接实现"）：
          无约束生成（greedy），正则扒标签。

数据集（JEVEval 风格子集，冻结模板、标准测试 split）：
  mmlu(150) arc(60) sst2(60) agnews(60) banking77(60)

两个已修的方法论 bug（2026-10-25 深夜，原始成绩作废重跑）：
  1. 温度：原版没传 temperature，两端点默认都在采样（实测同题重复 3 次输出会变），
     B 的 JSON 自写格式在采样下漂移尤其严重。现在三条路径全部显式 temperature=0。
  2. 标签越界：原版标签用 chr(65+i)，超过 Z 之后是标点、然后是小写字母，
     harness 把预测 .upper() 后和金标比 → 小写答案能"冒充"大写选项（banking77
     白送过 B 8 题）。现在标签用 A..Z,AA..AZ,BA.. 的 bijective base-26，
     评分大小写敏感精确比较。

用法：
  python3 jev_bench_run.py --limit 60 --concurrency 6          # 跑全部三条路径
  python3 jev_bench_run.py --paths A B --limit 30              # 只跑指定路径
  python3 jev_bench_run.py --paths A --datasets mmlu sst2      # 只跑指定数据集

环境变量（复现时按部署改这三个就够）：
  VLLM_BASE     vLLM 服务基址，默认 http://127.0.0.1:8787
                （实测可直连服务器 http://100.112.53.47:8000，无需任何代理）
  VLLM_API_KEY 服务器开 --api-key 时必填（Authorization Bearer + x-api-key 双头发）
  VLLM_MODEL   模型名，默认 qwen3.8-27b-w8a16（换成你自己的部署模型即可）
  BENCH_DIR    评测集目录，默认脚本同级 ../data/jev-bench/
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

# ---- 环境变量配置（复现友好：一条 vLLM 基址即可，无需任何代理）----
# 实测：1Cat-vLLM fork 的 /v1/messages 原生支持 tools + tool_choice 强制工具调用，
# Path A 直连服务器即可跑通（HTTP 200, tool_use 干净返回），不需要 8787 代理。
# 标准 vLLM（含 Anthropic API 支持的版本）同样适用。
VLLM_BASE = os.environ.get("VLLM_BASE", "http://127.0.0.1:8787").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")  # 服务器开 --api-key 时必填
MODEL = os.environ.get("VLLM_MODEL", "qwen3.8-27b-w8a16")
_HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.environ.get(
    "BENCH_DIR", os.path.normpath(os.path.join(_HERE, "..", "data", "jev-bench")))
TEMPERATURE = 0  # benchmark 一律 greedy；两端点默认都在采样，必须显式锁 0

# ---------------- 冻结模板（每数据集一个，对齐论文协议） ----------------
TASKS = {
    "mmlu":     "The following is a multiple-choice question. Answer with the letter of the correct option.",
    "arc":      "The following is a multiple-choice science question. Answer with the letter of the correct option.",
    "sst2":     "Classify the sentiment of the following text. Answer with the letter of the correct option.",
    "agnews":   "Classify the topic of the following news article. Answer with the letter of the correct option.",
    "banking77": "Classify the banking intent of the following customer message. Answer with the letter of the correct option.",
}
LIMITS = {"mmlu": 150, "arc": 60, "sst2": 60, "agnews": 60, "banking77": 60}


def labels_for(n: int):
    """bijective base-26 标签：1..26 -> A..Z，27..702 -> AA..ZZ，依此类推。
    绝不能用 chr(65+i)（超过 Z 会窜到标点和小写字母，造成大小写混淆评分 bug）。"""
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


def build_prompt(ds: str, item: dict) -> str:
    letters = labels_for(len(item["options"]))
    opts = "\n".join(f"{l}) {o}" for l, o in zip(letters, item["options"]))
    return f"{TASKS[ds]}\n\n{item['question']}\n\nOptions:\n{opts}\n\nAnswer with only the letter."


def post(url: str, payload: dict, timeout: int = 90):
    data = json.dumps(payload).encode()
    headers = {"content-type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = "Bearer " + VLLM_API_KEY
        headers["x-api-key"] = VLLM_API_KEY  # 双头发，兼容两种鉴权风格
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
        return body, time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, f"HTTP {e.code}: {e.read()[:200]!r}"
    except Exception as e:
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


def extract_choice(raw: str, labels):
    """从原始返回精确抽取标签值（大小写敏感）。返回 (pred or None, 发出的原值)。"""
    s = str(raw or "")
    m = re.search(r'"choice"\s*:\s*"([^"]{1,4})"', s)
    if m:
        v = m.group(1).strip()
        return (v if v in labels else None), v
    # 自由文本兜底：抓所有独立字母token，取第一个命中标签集的
    for tok in re.findall(r"\b([A-Za-z]{1,3})\b", s):
        if tok in labels:
            return tok, tok
    return None, None


# ---------------- Path A：本地 skill 机制（Anthropic 端点强制工具调用） ----------------
def run_pathA(ds, item):
    labels = labels_for(len(item["options"]))
    payload = {
        "model": MODEL,
        "max_tokens": 128,
        "temperature": TEMPERATURE,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": build_prompt(ds, item)}],
        "tools": [{
            "name": "answer",
            "description": "Select the correct option.",
            "input_schema": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": labels}},
                "required": ["choice"],
            },
        }],
        "tool_choice": {"type": "tool", "name": "answer"},
    }
    body, dt, err = post(f"{VLLM_BASE}/v1/messages", payload)
    if err:
        return {"pred": None, "err": err, "lat": dt, "out_tok": None, "raw": None}
    raw = None
    for c in body.get("content", []):
        if c.get("type") == "tool_use":
            raw = json.dumps(c.get("input", {}), ensure_ascii=False)
    pred, _ = extract_choice(raw, labels)
    out_tok = body.get("usage", {}).get("output_tokens")
    return {"pred": pred, "err": None if pred else "no tool_use/parse", "lat": dt,
            "out_tok": out_tok, "raw": (raw or json.dumps(body.get("content"))[:200])}


# ---------------- Path B：公开同类机制（OpenAI 端点 response_format） ----------------
def run_pathB(ds, item):
    labels = labels_for(len(item["options"]))
    payload = {
        "model": MODEL,
        "max_tokens": 128,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": build_prompt(ds, item)}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"choice": {"type": "string", "enum": labels}},
                    "required": ["choice"],
                },
            },
        },
    }
    body, dt, err = post(f"{VLLM_BASE}/v1/chat/completions", payload)
    if err:
        return {"pred": None, "err": err, "lat": dt, "out_tok": None, "raw": None}
    txt = body["choices"][0]["message"].get("content", "")
    pred, _ = extract_choice(txt, labels)
    out_tok = body.get("usage", {}).get("completion_tokens")
    return {"pred": pred, "err": None if pred else "no parse", "lat": dt,
            "out_tok": out_tok, "raw": txt[:200]}


# ---------------- Path C：自由文本基线 ----------------
def run_pathC(ds, item):
    labels = labels_for(len(item["options"]))
    payload = {
        "model": MODEL,
        "max_tokens": 24,
        "temperature": TEMPERATURE,
        "messages": [{"role": "user", "content": build_prompt(ds, item)}],
    }
    body, dt, err = post(f"{VLLM_BASE}/v1/chat/completions", payload)
    if err:
        return {"pred": None, "err": err, "lat": dt, "out_tok": None, "raw": None}
    txt = body["choices"][0]["message"].get("content", "").strip()
    pred, _ = extract_choice(txt, labels)
    out_tok = body.get("usage", {}).get("completion_tokens")
    return {"pred": pred, "err": None if pred else "no parse", "lat": dt,
            "out_tok": out_tok, "raw": txt[:200]}


RUNNERS = {"A": run_pathA, "B": run_pathB, "C": run_pathC}
PATH_NAMES = {
    "A": "本地 skill 机制（structured-dispatch：Anthropic 端点+强制工具调用）",
    "B": "公开同类机制（Instructor/vLLM：response_format json_schema）",
    "C": "自由文本基线（无约束+正则解析）",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="+", default=["A", "B", "C"])
    ap.add_argument("--datasets", nargs="+", default=list(TASKS.keys()))
    ap.add_argument("--limit", type=int, default=0, help="覆盖每集题数（0=用默认）")
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    # 载入题目
    items = []
    for ds in args.datasets:
        n = args.limit or LIMITS[ds]
        rows = [json.loads(l) for l in open(f"{BENCH_DIR}/{ds}.jsonl")][:n]
        for it in rows:
            items.append((ds, it))
    print(f"题目总数: {len(items)}/路径 × {len(args.paths)} 路径, 并发 {args.concurrency}, temp={TEMPERATURE}", flush=True)

    all_results = {}
    for path in args.paths:
        runner = RUNNERS[path]
        print(f"\n===== Path {path}: {PATH_NAMES[path]} =====", flush=True)
        results = [None] * len(items)

        def work(i_ds_it):
            i, (ds, it) = i_ds_it
            gold = labels_for(len(it["options"]))[it["answer"]]
            r = runner(ds, it)
            r.update({"i": i, "ds": ds, "gold": gold, "id": it["id"]})
            return r

        t0 = time.time()
        done = 0
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for r in ex.map(work, list(enumerate(items))):
                results[r["i"]] = r
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{len(items)}  ({time.time()-t0:.0f}s)", flush=True)
        wall = time.time() - t0
        all_results[path] = results

        # 汇总
        print(f"  --- Path {path} 汇总 (wall {wall:.0f}s) ---")
        for ds in args.datasets:
            rs = [r for r in results if r["ds"] == ds]
            if not rs:
                continue
            acc = sum(1 for r in rs if r["pred"] == r["gold"]) / len(rs)
            parse_fail = sum(1 for r in rs if not r["pred"])
            lats = [r["lat"] for r in rs if r["lat"]]
            toks = [r["out_tok"] for r in rs if r["out_tok"]]
            print(f"    {ds:10s} n={len(rs):3d} acc={acc*100:5.1f}% parse_fail={parse_fail:3d} "
                  f"lat_avg={sum(lats)/len(lats):5.2f}s tok_avg={sum(toks)/len(toks) if toks else 0:5.1f}")

    out = f"{BENCH_DIR}/results_{int(time.time())}.json"
    with open(out, "w") as f:
        json.dump({"model": MODEL, "temperature": TEMPERATURE, "paths": args.paths,
                   "datasets": args.datasets, "results": all_results}, ensure_ascii=False)
    print(f"\n结果已存: {out}", flush=True)


if __name__ == "__main__":
    main()
