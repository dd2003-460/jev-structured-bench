#!/usr/bin/env bash
# reproduce.sh — 一键复现：下载评测集 → 跑三路径 benchmark → 输出汇总
#
# 前置条件：
#   1. 一个 vLLM 服务（docker compose up -d 或任何现有部署）
#   2. python3（>=3.9，纯标准库，零 pip 依赖）+ curl
#
# 用法：
#   cp .env.example .env && vim .env      # 按你的部署改 VLLM_BASE 等
#   set -a && . ./.env && set +a          # 导出环境变量
#   ./scripts/reproduce.sh                # 全量（390 题 × 3 路径 ≈ 15-40 分钟）
#   ./scripts/reproduce.sh --limit 30     # 快速冒烟（30 题 × 3 路径）
set -euo pipefail
cd "$(dirname "$0")/.."

LIMIT_ARG=""
[[ "${1:-}" == "--limit" ]] && LIMIT_ARG="--limit ${2:-30}"

echo "== 0/3 环境检查 ================================================"
python3 --version
curl -s --noproxy '*' --max-time 5 "${VLLM_BASE:-http://127.0.0.1:8000}/v1/models" \
  ${VLLM_API_KEY:+-H "Authorization: Bearer $VLLM_API_KEY"} >/dev/null \
  && echo "vLLM 可达: ${VLLM_BASE}" || { echo "vLLM 不可达，先起服务（docker compose up -d）"; exit 1; }

echo "== 1/3 下载评测集（HF datasets-server，600 题）=================="
python3 harness/jev_bench_download.py

echo "== 2/3 三路径 benchmark（A=强制工具调用 / B=json_schema / C=自由文本）=="
# 并发默认 6；显存小或服务远（跨网 300ms+）建议降到 2-3
python3 harness/jev_bench_run.py --concurrency "${CONCURRENCY:-6}" $LIMIT_ARG

echo "== 3/3 完成 ======================================================"
echo "结果 JSON（含每题预测/原文/延迟/token）在 data/jev-bench/results_*.json"
echo "如装了 jq，快速看汇总："
echo "  ls -t data/jev-bench/results_*.json | head -1 | xargs -I{} python3 -c \\"
echo "    'import json,sys; d=json.load(open(sys.argv[1])); [print(p, sum(1 for r in d[\"results\"][p] if r[\"pred\"]==r[\"gold\"]), \"/\", len(d[\"results\"][p])) for p in d[\"paths\"]]' {}"
