#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ab_probe.py — 诊断 Path A vs Path B 分数差：温度默认值 / enum 严格性 / 思考默认态"""
import json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

PROXY = "http://127.0.0.1:8787"
MODEL = "qwen3.8-27b-w8a16"
BENCH = "/Users/a1-6/WorkBuddy/2026-10-16-20-29-01/data/jev-bench"

def curl_post(url, payload, timeout=120):
    cmd = ["curl", "-s", "--noproxy", "*", "--max-time", str(timeout),
           "-X", "POST", url, "-H", "content-type: application/json",
           "-d", json.dumps(payload)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.stdout

def build_prompt(item):
    letters = [chr(ord('A') + i) for i in range(len(item["options"]))]
    opts = "\n".join(f"{l}) {o}" for l, o in zip(letters, item["options"]))
    return (f"Classify the banking intent of the following customer message. "
            f"Answer with the letter of the correct option.\n\n{item['question']}\n\n"
            f"Options:\n{opts}\n\nAnswer with only the letter.")

def call_A(item, temperature=None):
    letters = [chr(ord('A') + i) for i in range(len(item["options"]))]
    payload = {
        "model": MODEL, "max_tokens": 128, "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": build_prompt(item)}],
        "tools": [{"name": "answer", "description": "Select the correct option.",
                   "input_schema": {"type": "object",
                                    "properties": {"choice": {"type": "string", "enum": letters}},
                                    "required": ["choice"]}}],
        "tool_choice": {"type": "tool", "name": "answer"},
    }
    if temperature is not None: payload["temperature"] = temperature
    out = curl_post(f"{PROXY}/v1/messages", payload)
    try:
        body = json.loads(out)
        for c in body.get("content", []):
            if c.get("type") == "tool_use":
                return json.dumps(c.get("input", {}), ensure_ascii=False)
        return "NO_TOOL_USE:" + out[:120]
    except Exception:
        return "ERR:" + out[:120]

def call_B(item, temperature=None):
    letters = [chr(ord('A') + i) for i in range(len(item["options"]))]
    payload = {
        "model": MODEL, "max_tokens": 128,
        "messages": [{"role": "user", "content": build_prompt(item)}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "answer",
            "schema": {"type": "object",
                       "properties": {"choice": {"type": "string", "enum": letters}},
                       "required": ["choice"]}}},
    }
    if temperature is not None: payload["temperature"] = temperature
    out = curl_post(f"{PROXY}/v1/chat/completions", payload)
    try:
        body = json.loads(out)
        return body["choices"][0]["message"].get("content", "")
    except Exception:
        return "ERR:" + out[:120]

def extract_letter(raw, letters):
    """从原始返回扒字母；返回 (letter or None, in_enum_strict)"""
    s = str(raw)
    import re
    m = re.search(r'"choice"\s*:\s*"([^"]{1,3})"', s)
    if not m: return None, None
    v = m.group(1)
    up = v.strip()[:1].upper()
    return (up if up in letters else None), v

if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "probe"
    items = [json.loads(l) for l in open(f"{BENCH}/banking77.jsonl")][:60]

    if phase == "probe":
        # 阶段1：同一题 ×3 重复 × {A默认, B默认} —— 测确定性（反推温度）
        print("=== 阶段1：确定性探针（同题重复3次，不带 temperature）===", flush=True)
        sample = items[:4]
        jobs = []
        for it in sample:
            for rep in range(3):
                jobs.append(("A", rep, it))
                jobs.append(("B", rep, it))
        def work(j):
            path, rep, it = j
            raw = call_A(it) if path == "A" else call_B(it)
            return path, rep, it["id"], raw
        with ThreadPoolExecutor(max_workers=4) as ex:
            results = list(ex.map(work, jobs))
        bykey = {}
        for path, rep, qid, raw in results:
            bykey.setdefault((path, qid), []).append(raw)
        for (path, qid), raws in sorted(bykey.items()):
            uniq = len(set(raws))
            print(f"  {path} q={qid[:16]} 重复3次不同输出数={uniq}")
            if uniq > 1:
                for r in raws: print(f"      {str(r)[:100]!r}")

        # 阶段2：enum 严格性 —— 用小 enum 诱导越界
        print("\n=== 阶段2：enum 严格性（3选项 enum，重复6次）===", flush=True)
        tiny = {"id": "tiny", "question": "I was charged twice for one order. What does the customer want?",
                "options": ["refund", "cancel order", "change address"], "answer": 0}
        letters3 = ["A", "B", "C"]
        # 手工发：覆盖 A/B，看有没有越界值
        def call_A_enum(it, letters):
            payload = {"model": MODEL, "max_tokens": 128, "thinking": {"type": "disabled"},
                       "messages": [{"role": "user", "content": build_prompt(it)}],
                       "tools": [{"name": "answer", "input_schema": {"type": "object",
                                  "properties": {"choice": {"type": "string", "enum": letters}},
                                  "required": ["choice"]}}],
                       "tool_choice": {"type": "tool", "name": "answer"}}
            out = curl_post(f"{PROXY}/v1/messages", payload)
            try:
                body = json.loads(out)
                for c in body.get("content", []):
                    if c.get("type") == "tool_use":
                        return json.dumps(c.get("input", {}), ensure_ascii=False)
                return "NO_TOOL_USE"
            except Exception: return "ERR:" + out[:100]
        def call_B_enum(it, letters):
            payload = {"model": MODEL, "max_tokens": 128,
                       "messages": [{"role": "user", "content": build_prompt(it)}],
                       "response_format": {"type": "json_schema", "json_schema": {"name": "answer",
                                  "schema": {"type": "object",
                                             "properties": {"choice": {"type": "string", "enum": letters}},
                                             "required": ["choice"]}}}}
            out = curl_post(f"{PROXY}/v1/chat/completions", payload)
            try:
                return json.loads(out)["choices"][0]["message"].get("content", "")
            except Exception: return "ERR:" + out[:100]
        jobs2 = [("A", i) for i in range(6)] + [("B", i) for i in range(6)]
        def work2(j):
            path, i = j
            raw = call_A_enum(tiny, letters3) if path == "A" else call_B_enum(tiny, letters3)
            return path, i, raw
        with ThreadPoolExecutor(max_workers=6) as ex:
            for path, i, raw in ex.map(work2, jobs2):
                v = None
                import re
                m = re.search(r'"choice"\s*:\s*"([^"]{1,3})"', str(raw))
                if m: v = m.group(1)
                strict = "OK" if (v in letters3) else ">>> 越界!"
                print(f"  {path} #{i}: {str(raw)[:60]!r}  解析={v} {strict}")

    elif phase == "rerun":
        # 阶段3：30 题受控重跑 A / A+temp0 / B / B+temp0
        print("=== 阶段3：受控重跑 30 题（banking77）===", flush=True)
        sample = items[:30]
        configs = [("A", None), ("A", 0.0), ("B", None), ("B", 0.0)]
        accs = {}
        for path, temp in configs:
            letters = None
            def work(it):
                raw = call_A(it, temp) if path == "A" else call_B(it, temp)
                letters_i = [chr(ord('A') + k) for k in range(len(it["options"]))]
                pred, _ = extract_letter(raw, letters_i)
                gold = letters_i[it["answer"]]
                return pred == gold, pred, gold, str(raw)[:60]
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=6) as ex:
                res = list(ex.map(work, sample))
            ok = sum(1 for r in res if r[0])
            tag = f"{path}{'+t0' if temp == 0.0 else '+默认'}"
            accs[tag] = ok / len(res)
            print(f"  {tag}: acc={ok/len(res)*100:.1f}% ({ok}/{len(res)}) wall={time.time()-t0:.0f}s", flush=True)
            if path == "B" and temp == 0.0:
                wrong = [r for r in res if not r[0]][:5]
                for w in wrong: print(f"      错题 gold={w[2]} pred={w[1]} raw={w[3]!r}")
        print("\n汇总:", json.dumps({k: f"{v*100:.1f}%" for k, v in accs.items()}, ensure_ascii=False))

    elif phase == "thinking":
        # 阶段4：两条端点的"默认思考态"——不加任何约束，看模型自己会不会思考
        print("=== 阶段4：默认思考态探测（无约束自由生成）===", flush=True)
        q = "What is 17 + 25? Reply with just the number."
        # OpenAI 路径，无 response_format
        p1 = {"model": MODEL, "max_tokens": 400, "temperature": 0,
              "messages": [{"role": "user", "content": q}]}
        out1 = curl_post(f"{PROXY}/v1/chat/completions", p1)
        try: c1 = json.loads(out1)["choices"][0]["message"].get("content", "")
        except Exception: c1 = "ERR:" + out1[:200]
        print(f"  [chat/completions 默认] {c1[:300]!r}")
        # Anthropic 路径，无 thinking 字段
        p2 = {"model": MODEL, "max_tokens": 400,
              "messages": [{"role": "user", "content": q}]}
        out2 = curl_post(f"{PROXY}/v1/messages", p2)
        try:
            b2 = json.loads(out2)
            txt = " | ".join(c.get("text", c.get("thinking", "")) if isinstance(c, dict) else str(c)
                             for c in b2.get("content", []))
            print(f"  [/v1/messages 默认] blocks={[c.get('type') for c in b2.get('content',[])]} {txt[:300]!r}")
        except Exception:
            print(f"  [/v1/messages 默认] ERR: {out2[:200]!r}")
        # Anthropic 路径，显式关思考
        p3 = dict(p2); p3["thinking"] = {"type": "disabled"}
        out3 = curl_post(f"{PROXY}/v1/messages", p3)
        try:
            b3 = json.loads(out3)
            txt3 = " | ".join(c.get("text", "") for c in b3.get("content", []) if isinstance(c, dict))
            print(f"  [/v1/messages 关思考] {txt3[:200]!r}")
        except Exception:
            print(f"  [/v1/messages 关思考] ERR: {out3[:200]!r}")
