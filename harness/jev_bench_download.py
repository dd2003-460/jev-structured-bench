#!/usr/bin/env python3
"""Download JEV-style benchmark subsets from HF datasets-server API -> unified JSONL."""
import json, time, urllib.request, urllib.error, sys, os

OUT = "/tmp/jev_bench"
os.makedirs(OUT, exist_ok=True)

# (key, dataset, config, split, n, kind)
SPECS = [
    ("mmlu",       "cais/mmlu",        "all",    "test", 200, "mc4"),
    ("arc",        "allenai/ai2_arc",  "ARC-Challenge", "test", 100, "mc4"),
    ("sst2",       "SetFit/sst2",      "default", "validation", 100, "binary"),
    ("agnews",     "fancyzhx/ag_news", "default", "test", 100, "mc4"),
    ("banking77",  "legacy-datasets/banking77", "default", "test", 100, "mc77"),
]

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

def to_letter(i):
    return chr(ord('A') + i)

def normalize(key, kind, row):
    """-> dict(question, options, answer_idx) or None"""
    if key == "mmlu":
        return {"question": row["question"], "options": row["choices"], "answer": int(row["answer"])}
    if key == "arc":
        ch = row["choices"]["text"]
        ans = row["answerKey"]
        if ans in [str(i + 1) for i in range(len(ch))]:
            idx = int(ans) - 1
        else:
            idx = ord(ans.upper()) - ord('A')
        return {"question": row["question"], "options": ch, "answer": idx}
    if key == "sst2":
        return {"question": row["text"],
                "options": ["negative", "positive"], "answer": int(row["label"])}
    if key == "agnews":
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        return {"question": row["text"], "options": labels, "answer": int(row["label"])}
    if key == "banking77":
        return {"question": row["text"], "options": row["label_candidates"] if "label_candidates" in row else None,
                "answer": int(row["label"])}
    return None

# banking77 needs the full label list; fetch from dataset features
def banking77_labels():
    url = ("https://datasets-server.huggingface.co/rows?dataset=legacy-datasets%2Fbanking77"
           "&config=default&split=test&offset=0&length=1")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            d = json.load(r)
        for f in d.get("features", []):
            if f["name"] == "label":
                return f["type"]["names"]
    except Exception as e:
        print("  rows endpoint failed:", e)
    return None

def main():
    b77_labels = None
    for key, dataset, config, split, n, kind in SPECS:
        print(f"=== {key} ({dataset} {config}/{split}) n={n} ===", flush=True)
        if key == "banking77":
            b77_labels = banking77_labels()
            if not b77_labels:
                print("  SKIP: no label names"); continue
        items = []
        got = 0
        offset = 0
        while got < n:
            length = min(100, n - got)
            d = fetch_rows(dataset, config, split, offset, length)
            if not d:
                print("  FAILED to fetch"); break
            for row in d.get("rows", []):
                r = row["row"]
                if key == "banking77":
                    r = dict(r); r["label_candidates"] = b77_labels
                it = normalize(key, kind, r)
                if it is None or it["options"] is None:
                    continue
                # sanity
                if not (0 <= it["answer"] < len(it["options"])):
                    continue
                items.append({"id": f"{key}-{got}", "source": key, **it})
                got += 1
                if got >= n: break
            offset += len(d.get("rows", []))
            if not d.get("rows"): break
        out = f"{OUT}/{key}.jsonl"
        with open(out, "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"  saved {len(items)} -> {out}", flush=True)

if __name__ == "__main__":
    import urllib.parse  # noqa
    main()
