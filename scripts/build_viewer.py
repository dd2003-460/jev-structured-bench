#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_viewer.py — 从 results_*.json 生成交互式答题结果查看器 HTML。

用法：
  python3 scripts/build_viewer.py [--out reports/JEV-benchmark答题结果查看器.html]

数据源（按数据集合并，路径缺失显示"未跑"）：
  data/results_1792928998.json   旧 5 集 × A/B/C 三路径（含两个方法论 bug，保留作透明记录）
  data/results_1792943210.json   新 4 集 × Path A（干净重跑，直连+重试）
题目与选项来自 data/*.jsonl（id 对齐）。

比第一版查看器修的：
  - 选项字母用 bijective base-26（A..Z, AA..BY），banking77 77 选项不再越界成乱码
  - 新 4 集只有 Path A，B/C 列显示"未跑"而不是崩页面
  - "只看有路径答错"过滤兼容缺失路径
"""
from __future__ import annotations
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(_HERE, ".."))

# (结果文件, 该文件覆盖的数据集)
RESULT_FILES = [
    ("results_1792928998.json", None),   # None = 文件内 datasets 字段
    ("results_1792943210.json", None),
]


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


def load_questions(bench_dir: str):
    qs = {}
    for fn in sorted(os.listdir(bench_dir)):
        if not fn.endswith(".jsonl"):
            continue
        with open(os.path.join(bench_dir, fn)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                qs[it["id"]] = it
    return qs


def cell(rec):
    """一条路径记录 → [pred, ok, lat, raw, out_tok]；未跑/失败 → None 字段。"""
    if rec is None:
        return None
    pred = rec.get("pred")
    ok = (pred is not None and pred == rec.get("gold"))
    raw = rec.get("raw")
    if raw is None and rec.get("err"):
        raw = f"[err] {rec['err']}"
    return [pred, ok, round(rec.get("lat") or 0, 2), raw, rec.get("out_tok")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "JEV-benchmark答题结果查看器.html"))
    args = ap.parse_args()

    questions = load_questions(args.bench_dir)
    rows = {}
    ds_of = {}
    for fn, _ in RESULT_FILES:
        path = os.path.join(args.bench_dir, fn)
        if not os.path.exists(path):
            print(f"跳过（不存在）: {path}", file=sys.stderr)
            continue
        d = json.load(open(path))
        for p, recs in d["results"].items():
            for rec in recs:
                qid = rec["id"]
                if qid not in questions:
                    continue
                row = rows.setdefault(qid, {"id": qid})
                row[p] = cell(rec)
                ds_of[qid] = rec.get("ds") or ds_of.get(qid)

    out_rows = []
    for qid, row in rows.items():
        it = questions[qid]
        letters = labels_for(len(it["options"]))
        r = {
            "id": qid,
            "ds": it.get("dataset") or ds_of.get(qid) or "",
            "q": it["question"],
            "opts": [f"{l}) {o}" for l, o in zip(letters, it["options"])],
            "gold": letters[it["answer"]],
        }
        for p in ("A", "B", "C"):
            r[p] = row.get(p)
        out_rows.append(r)
    out_rows.sort(key=lambda r: (r["ds"], r["id"]))

    n_total = len(out_rows)
    n_paths = {"A": sum(1 for r in out_rows if r["A"]),
               "B": sum(1 for r in out_rows if r["B"]),
               "C": sum(1 for r in out_rows if r["C"])}
    print(f"查看器题目 {n_total} 题；路径覆盖 A={n_paths['A']} B={n_paths['B']} C={n_paths['C']}")

    html = TEMPLATE.replace("__DATA__", json.dumps({"rows": out_rows}, ensure_ascii=False))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"已生成: {args.out} ({os.path.getsize(args.out)//1024} KB)")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JEV Benchmark 答题结果 · 路径对比</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
         background: #f5f6f8; color: #1f2329; padding: 24px; font-size: 14px; }
  .wrap { max-width: 1500px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  .sub { color: #646a73; font-size: 13px; margin-bottom: 20px; }
  .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 16px; }
  .card { background: #fff; border: 1px solid #e5e6eb; border-radius: 10px; padding: 14px 16px; }
  .card h3 { font-size: 14px; margin-bottom: 6px; }
  .card .tag { display: inline-block; font-size: 11px; padding: 1px 8px; border-radius: 10px; margin-left: 6px; vertical-align: middle; }
  .tagA { background: #e8f3ff; color: #1456f0; } .tagB { background: #fdf0e5; color: #c26209; } .tagC { background: #eef0f3; color: #646a73; }
  .card p { font-size: 12px; color: #646a73; line-height: 1.6; }
  .card .big { font-size: 26px; font-weight: 700; margin: 4px 0; }
  table.sum { width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden; margin-bottom: 16px; }
  table.sum th, table.sum td { border: 1px solid #e5e6eb; padding: 7px 10px; text-align: center; font-size: 13px; }
  table.sum th { background: #f0f1f3; font-weight: 600; }
  table.sum td.l { text-align: left; }
  .bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; background: #fff;
         border: 1px solid #e5e6eb; border-radius: 10px; padding: 10px 12px; position: sticky; top: 0; z-index: 10; }
  .bar select, .bar input[type=text] { padding: 6px 10px; border: 1px solid #d0d3d9; border-radius: 6px; font-size: 13px; background: #fff; }
  .bar input[type=text] { width: 260px; }
  .bar label { font-size: 13px; color: #1f2329; display: flex; align-items: center; gap: 5px; cursor: pointer; }
  .bar .cnt { margin-left: auto; font-size: 12px; color: #646a73; }
  table.res { width: 100%; border-collapse: collapse; background: #fff; border-radius: 10px; overflow: hidden; }
  table.res th, table.res td { border: 1px solid #e5e6eb; padding: 6px 8px; font-size: 12px; vertical-align: top; }
  table.res th { background: #f0f1f3; font-weight: 600; position: sticky; top: 53px; z-index: 5; }
  .qtext { max-height: 34px; overflow: hidden; cursor: pointer; line-height: 1.5; }
  .qtext.open { max-height: none; }
  .opts { color: #646a73; font-size: 11px; margin-top: 3px; line-height: 1.5; }
  .gold { font-weight: 700; color: #1f2329; }
  .pill { display: inline-block; padding: 1px 9px; border-radius: 10px; font-size: 12px; font-weight: 600; }
  .pill.ok { background: #e8f5e9; color: #1a7f37; }
  .pill.no { background: #fdecea; color: #c0392b; }
  .pill.na { background: #f0f1f3; color: #9a9ea6; }
  .meta { font-size: 10px; color: #9a9ea6; margin-top: 2px; }
  .raw { font-size: 10px; color: #646a73; font-family: ui-monospace, Menlo, monospace;
         max-width: 150px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-top: 2px; }
  .dsname { font-size: 11px; color: #646a73; }
  .note { font-size: 12px; color: #646a73; margin-top: 14px; line-height: 1.7; }
  .note code { background: #eef0f3; padding: 1px 5px; border-radius: 4px; font-size: 11px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>JEV 风格 Benchmark · 答题结果查看器</h1>
  <div class="sub">模型 qwen3.8-27b-w8a16（1Cat-vLLM TP4 V100，零微调零样本）· 9 数据集 630 题 · 旧 5 集三路径对比，新 4 集（hellaswag/mathqa/truthfulqa/imdb）仅 Path A · 2026-10-25</div>

  <div class="cards">
    <div class="card"><h3><span class="tag tagA">A</span>本地 skill 机制</h3>
      <p>structured-dispatch 插件机制：Anthropic <code>/v1/messages</code> 端点 + 强制工具调用（tool_choice 锁死单 function，input_schema=动态 enum），tool_use 剥壳成 JSON。</p></div>
    <div class="card"><h3><span class="tag tagB">B</span>公开同类机制</h3>
      <p>开源社区通用路线（Instructor 默认机制 / vLLM 原生 structured outputs）：OpenAI <code>/v1/chat/completions</code> + <code>response_format=json_schema</code>。</p></div>
    <div class="card"><h3><span class="tag tagC">C</span>自由文本基线</h3>
      <p>论文 §3.6 的"间接实现"：不约束，让模型自由生成再正则扒字母。</p></div>
  </div>

  <table class="sum" id="sumTable"></table>

  <div class="bar">
    <select id="dsSel"><option value="">全部数据集</option></select>
    <select id="pathSel">
      <option value="">三路径并列</option>
      <option value="A">只看 A 本地 skill</option>
      <option value="B">只看 B 公开同类</option>
      <option value="C">只看 C 自由文本</option>
    </select>
    <label><input type="checkbox" id="onlyWrong"> 只看有路径答错的题</label>
    <label><input type="checkbox" id="onlyA"> 只看 A 答错的题</label>
    <input type="text" id="search" placeholder="搜题目内容…">
    <span class="cnt" id="cnt"></span>
  </div>

  <table class="res">
    <thead><tr>
      <th style="width:34px">#</th><th style="width:90px">数据集</th><th>题目</th><th style="width:52px">金标</th>
      <th class="pathcol" style="width:120px">A 本地 skill</th>
      <th class="pathcol" style="width:120px">B 公开同类</th>
      <th class="pathcol" style="width:120px">C 自由文本</th>
    </tr></thead>
    <tbody id="tbody"></tbody>
  </table>

  <div class="note">
    说明：绿=答对，红=答错，灰=该路径未跑；每格第二行是延迟与输出 token，第三行（截断）是原始返回。<br>
    旧 5 集原始数据：<code>data/jev-bench/results_1792928998.json</code>（含两个方法论 bug，保留作透明记录，见报告 §0 勘误）；
    新 4 集干净重跑：<code>data/jev-bench/results_1792943210.json</code>（直连+重试，零传输错误）。
    复现脚本：<code>harness/jev_bench_run.py</code>；本查看器由 <code>scripts/build_viewer.py</code> 生成。
  </div>
</div>

<script>
const DATA = __DATA__;
const rows = DATA.rows;
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function buildSummary(){
  const byDs = {};
  rows.forEach(r => { (byDs[r.ds] = byDs[r.ds] || []).push(r); });
  let h = '<tr><th>数据集</th><th>题数</th><th>A 本地 skill</th><th>B 公开同类</th><th>C 自由文本</th></tr>';
  const acc = (rs, p) => { const v = rs.filter(r => r[p]); return v.length ? (v.filter(r => r[p][1]).length / v.length * 100).toFixed(1) + '% (' + v.filter(r => r[p][1]).length + '/' + v.length + ')' : '—'; };
  const nRun = (rs, p) => rs.filter(r => r[p]).length || '—';
  Object.keys(byDs).sort().forEach(ds => {
    const rs = byDs[ds];
    h += '<tr><td class="l">' + esc(ds) + '</td><td>' + rs.length + '</td><td>' + acc(rs,'A') + '</td><td>' + acc(rs,'B') + '</td><td>' + acc(rs,'C') + '</td></tr>';
  });
  h += '<tr><td class="l"><b>合计</b></td><td><b>' + rows.length + '</b></td><td><b>' + acc(rows,'A') + '</b></td><td><b>' + acc(rows,'B') + '</b></td><td><b>' + acc(rows,'C') + '</b></td></tr>';
  document.getElementById('sumTable').innerHTML = h;
  const sel = document.getElementById('dsSel');
  Object.keys(byDs).sort().forEach(ds => { const o = document.createElement('option'); o.value = ds; o.textContent = ds + '（' + byDs[ds].length + '）'; sel.appendChild(o); });
}

function render(){
  const ds = document.getElementById('dsSel').value;
  const path = document.getElementById('pathSel').value;
  const onlyWrong = document.getElementById('onlyWrong').checked;
  const onlyA = document.getElementById('onlyA').checked;
  const kw = document.getElementById('search').value.trim().toLowerCase();
  const tb = document.getElementById('tbody');
  let html = '', n = 0;
  rows.forEach((r, idx) => {
    if (ds && r.ds !== ds) return;
    if (kw && r.q.toLowerCase().indexOf(kw) < 0) return;
    const anyWrong = ['A','B','C'].some(p => r[p] && !r[p][1]);
    if (onlyWrong && !anyWrong) return;
    if (onlyA && r.A && r.A[1]) return;
    n++;
    const showP = p => !path || path === p;
    let h = '<tr><td>' + (idx+1) + '</td><td><span class="dsname">' + esc(r.ds) + '</span></td>';
    h += '<td class="q"><div class="qtext" onclick="this.classList.toggle(\'open\')">' + esc(r.q) + '</div><div class="opts">' + esc(r.opts.join('  ')).slice(0,400) + '</div></td>';
    h += '<td style="text-align:center"><span class="gold">' + esc(r.gold) + '</span></td>';
    for (const p of ['A','B','C']) {
      if (!showP(p)) { h += '<td style="text-align:center;color:#c8ccd2">—</td>'; continue; }
      const v = r[p];
      if (!v) { h += '<td style="text-align:center"><span class="pill na">未跑</span></td>'; continue; }
      const pred = v[0] == null ? '×' : v[0];
      h += '<td style="text-align:center"><span class="pill ' + (v[1] ? 'ok' : 'no') + '">' + esc(pred) + '</span>'
         + '<div class="meta">' + (v[2]||0).toFixed(1) + 's · ' + (v[4]==null?'?':v[4]) + ' tok</div>'
         + '<div class="raw" title="' + esc(v[3]) + '">' + esc(v[3]) + '</div></td>';
    }
    h += '</tr>';
    html += h;
  });
  tb.innerHTML = html;
  document.getElementById('cnt').textContent = '显示 ' + n + ' / ' + rows.length + ' 题';
}
buildSummary();
['dsSel','pathSel','onlyWrong','onlyA'].forEach(id => document.getElementById(id).addEventListener('change', render));
document.getElementById('search').addEventListener('input', render);
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
