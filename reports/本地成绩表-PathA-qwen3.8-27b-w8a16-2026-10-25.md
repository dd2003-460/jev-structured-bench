# 本地成绩表 — JEV 风格 benchmark · Path A（structured-dispatch）· qwen3.8-27b-w8a16

模型：`qwen3.8-27b-w8a16`（1Cat-vLLM fork，零训练零微调，零样本）
机制：Path A = Anthropic `/v1/messages` + 强制工具调用（`tool_choice` 锁 `answer`，`input_schema` 请求内**动态 enum** 选项字母）+ `thinking:{type:disabled}` + `max_tokens 128` + `temperature=0`
评测链路：Mac → **直连** 1Cat-vLLM（100.112.53.47:8000，无代理；旧 5 集经 8787）

## 一、新 4 集（2026-10-25 干净重跑，直连+重试，`results_1792943210.json`）

| 数据集 | 题数 | 准确率 | 解析失败数 | 传输错误 | 平均延迟(s) | 平均输出 token | 选项数 |
|---|---|---|---|---|---|---|---|
| hellaswag | 60 | **86.7%** | 0 | 0 | 19.66 | 7.2 | 4 |
| mathqa | 60 | **50.0%** | 0 | 0 | 14.81 | 11.1 | 5 |
| truthfulqa | 60 | **78.3%** | 0 | 0 | 16.24 | 11.2 | 2–11 |
| imdb | 60 | **91.7%** | 0 | 0 | 26.00 | 12.0 | 2 |
| **新4集合计** | **240** | **76.7%** | **0** | **0** | — | — | |

> **首跑作废说明**：DSH 侧首跑（经 8787，`run2_pathA.log`）报 hellaswag 85.0 / mathqa 48.3 /
> truthfulqa 71.7 / imdb 78.3，且"解析失败" 1/2/6/9 共 18 题——逐类核查证明那 18 题全是
> HTTP 502 传输错误（代理超时），不是模型不会答。干净重跑（直连+退避重试+分列统计）后
> 四集全涨、解析失败归零。另：首跑因 `json.dump()` 漏传文件句柄在保存时崩溃，逐题原始
> 数据丢失，本表数字来自重跑的 `results_1792943210.json`（含每题预测/原文/延迟）。

## 二、旧 5 集（results_1792928998.json 复算，同 Path A）

| 数据集 | 题数 | 准确率 | 解析失败数 | 平均延迟(s) | 平均输出 token | 选项数 |
|---|---|---|---|---|---|---|
| mmlu | 150 | 68.7% | 2 | 17.15 | 9.8 | 4 |
| arc | 60 | 95.0% | 0 | 6.23 | 11.2 | 4 |
| sst2 | 60 | 98.3% | 0 | 4.72 | 11.2 | 2 |
| agnews | 60 | 78.3% | 0 | 3.57 | 11.6 | 4 |
| banking77 | 60 | 76.7% | 0 | 3.96 | 7.1 | 77 |
| **旧5集合计** | **390** | **80.0%** | **2** | — | — | |

## 三、横向对比

**9 集合计 630 题，Path A 总准确率 78.7%**。新 4 集（76.7%）低于旧 5 集（80.0%）的原因清晰：
mathqa（50.0%）是多步计算软肋——与 JEV 论文"决策模型计算项比前沿中位数低 17.7pp"同向；
hellaswag/truthfulqa/imdb 都与旧集同档（78-92%）。**旧 5 集的 2 个"解析失败"经核查同为
传输层问题**（harness 已修：重试+绕代理+分列 transport_err/parse_fail）。

## 四、本地复现口径说明

- **机制**：Path A 动态 enum —— 每题把选项字母集合动态写进请求的
  `tools[0].input_schema.properties.choice.enum`，`tool_choice={type:"tool",name:"answer"}`
  强制调用；从返回 `content[]` 的 `tool_use` block 取 `input.choice` 作答。
  **不用** agents.json 注册表（8787 当前实例是旧进程，新形状 bench_mc 会透传成自由文本）。
- **模板**：与已跑 5 集同一冻结模板口吻（`scripts/jev_bench_run.py` TASKS，
  全部以 "… Answer with the letter of the correct option." 结尾；
  hellaswag/mathqa/truthfulqa/imdb 四条为本次按同口吻新增）。
- **模型**：qwen3.8-27b-w8a16，零训练零微调，零样本。
- **采样**：temperature=0（greedy），max_tokens=128，thinking disabled。
- **并发**：ThreadPoolExecutor max_workers=6（与旧 5 集同）。
- **端点**：http://127.0.0.1:8787/v1/messages（旧 5 集同链路；8787 代理进程未动）。
- **跑前负载检查**：Prometheus :9091 采样 `vllm:num_requests_running`/`num_requests_waiting`，
  21:56–21:57 与 22:50 两轮均无排队（waiting=0），running ≤5 轻载，撞车风险低。
- **时间戳**：干净重跑 2026-10-25 23:32–23:47（UTC+8，wall 871s，并发 6）。
- **数据源**（`scripts/jev_bench_download2.py`，HF datasets-server REST）：
  - hellaswag：Rowan/hellaswag default/validation 前 60 题（官方 3953 题 split）；
  - mathqa：datasets-server 不支持该 loading script，改拉官方原始 MathQA.zip → test.json 前 60 题；
  - truthfulqa：truthfulqa/truthful_qa config=multiple_choice validation，mc1_targets；
    **HF 设计上 mc1 正确答案恒在位置 0**，本表选项已按 SEED=20261025 固定打乱破位置偏置（内容不变）；
  - imdb：SetFit/imdb default/test（与既有 SetFit/sst2 同口径）；
    **test 前 12500 条全 negative、offset≥12500 全 positive**（各抽 50 行验证），
    本表 30 negative + 30 positive 均衡采样。
- **schema**：与既有 5 集一致 `{id, source/question/options/answer/dataset}`（answer 为 0-based int）。
- **原始结果**：干净重跑 `data/jev-bench/results_1792943210.json`（口径照 results_1792928998.json）；
  DSH 首跑逐题数据因保存 bug 丢失，汇总数字存 `run2_pathA.log`。
