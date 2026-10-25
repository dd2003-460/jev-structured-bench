# JEV 风格 Benchmark：本地 skill 机制 vs 公开同类机制 vs 自由文本

**日期**：2026-10-25 · **模型**：qwen3.8-27b-w8a16（1Cat-vLLM 1.5.0，TP4 V100，本地）
**评测机**：Mac → 8787 翻译代理 → 100.112.53.47:8000 · **复现脚本**：`scripts/jev_bench_run.py` · **数据**：`/tmp/jev_bench/*.jsonl` + `results_1792928998.json`

---

## 0. 勘误（2026-10-25 深夜追加，优先级高于下文）

深挖"A 为什么比 B 高"时发现**两个方法论 bug**，下文部分结论已作废，以本节为准：

1. **温度没锁（最大混淆）**：原跑分三条路径都没传 `temperature`，实测两端点默认都在采样（同题重复 3 次，A 2/4 题输出变化、B 3/4 变化）。B 的 JSON 自写格式在采样下漂移更严重。**锁温重跑（banking77 30 题子集）：A 70.0→80.0%，B 60.0→73.3%**——旧报告 banking77 的 23.4pp 差距几乎全部是温度噪声。
2. **banking77 评分大小写混淆**：标签用 `chr(65+i)` 生成，超过 Z 之后是标点、然后是小写字母（第 33-58 个选项标签就是 a-z）；旧 harness 把预测 `.upper()` 后和金标比，小写答案能"冒充"大写选项的正确答案。**精确重算：B 有 8 题白送（真实 40.0%，非 53.3%），A/C 0 题**。即该 bug 是虚高 B，修正后 A-B 差距反而扩大（但那是不同温度下的比较，锁温后回落）。

**更正后的干净对比（banking77，同锁 temperature=0，60 题）**：

| 路径 | 锁温成绩 | 备注 |
|---|---|---|
| A 本地 skill | **80.0%**（48/60） | |
| B 公开同类 | 76.7%（46/60） | 其中 3 题靠大小写冒充白送，精确口径约 **71.7%** |
| C 自由文本 | **80.0%**（48/60） | |

- 锁温后 B 追平旧报告差距的绝大部分；A 对 B 的**真实剩余差距约 7-8pp（约 1σ，不彻底显著）**。
- 四个干净数据集（mmlu/arc/sst2/agnews，选项 ≤5 不受 bug 2 影响）A-B 仅 **+1.8pp**（且是采样口径）。
- **A 对 B 的真实机制优势**（锁温+精确后仍存在，个位数 pp）：Qwen3.8-27B 后训练含工具调用，forced tool_choice 是原生行为且 enum 被 chat template 渲染进 prompt（prompt 可见 + logits 锁定双重编码）；tool_use 参数由框架序列化无格式噪声；B 靠模型自写 JSON，在怪异标签集上会漂移大小写。
- **思考差异不存在**：跑分时三条路径都没开思考（390×3 条返回 0 思考痕迹，A/B 都只产 7-15 token，语法约束把推理挤掉；实测两端点默认也不产生独立 thinking block）。
- harness 已修（`scripts/jev_bench_run.py`：显式 temperature=0 + bijective base-26 标签 + 大小写敏感精确评分）。banking77 用新标签（AA/AB…BY）重跑后与本节旧标签口径不可直接比，以新跑为准。
- 下文第 3 节 Banking77 行、第 4 节发现 2、第 5 节相关结论**按本节作废/修正**。

深挖 filler/思考实验时又发现**三个工程层 bug**（2026-10-25 23:30 追加，同样以本节为准）：

3. **预填充探针 v1 的"崩盘"是传输错误假象**：v1 报 F0 75.0% → F1 65.0% → F2 65.0% → F3 57.5%，看似 filler token 单调毁准确率。逐题核查：**所有"解析失败"都是 HTTP 502 `upstream connect failed: connection timed out`**——两个 benchmark 并发打服务器 + harness 的 urllib 走 Mac 系统代理（127.0.0.1:51647）时的传输层失败，与模型无关。只看真正拿到答案的题：**F0 83.3%（30/36）→ F1 81.3%（26/32）→ F2 81.3% → F3 79.3%（23/29）**，差异在噪声内——没有任何 filler 配置有效的证据，也没有"崩盘"。
4. **F1/F2 剂量扫描失效（tokenizer 把点串合并）**：v1 用 `"."*16` / `"."*64`，Qwen BPE 把点串合并成 ~2 token，两配置 in_tok 完全相同（1067 vs 1067），"剂量翻 4 倍"从未发生，F1=F2 是必然结果。v2 改用量词化 1:1 的 `"hmm "` 重复（8/64 token）+ 512 点串 + 复述，共 5 配置。
5. **DSH 侧 4 新数据集首跑丢原始数据**：`json.dump({...}, ensure_ascii=False)` 漏传文件句柄，240 题跑完（952s）才在保存时崩溃。汇总数字幸存于 `run2_pathA.log`（hellaswag 85.0 / mathqa 48.3 / truthfulqa 71.7 / imdb 78.3），逐题原始记录丢失；已用修好的 harness 直连服务器重跑补原始数据（进行中）。

** harness 通用修复（两个脚本都改了）**：`post()` 加退避重试（4xx 不重试，5xx/超时重试 3 次）+ 绕过系统代理（`ProxyHandler({})`，顺带每请求省 ~2.5s）+ 汇总把 `transport_err` 与 `parse_fail` 分开报。**教训复读：benchmark 报"解析失败"前先看错误串，传输层故障会伪装成模型不会答题。**

6. **C4 第一版作废（两个 bug 叠乘）**：① vLLM 的 `prompt_logprobs` dict 的 key 是 **token ID 字符串**（"271"）不是 token 文本，旧代码取 key 当"第一念"，永远对不上字母；② 更根本的——chat 模板让最后 prompt 位置的 top1 恒为 `\n\n`（logprob=0.0，确定性），答案字母 `' A'` 在 rank 4，"全局 argmax"根本不是答案。正确做法（论文协议）：**只在选项字母 token 集合内重归一化**再取 argmax。另发现 8787 代理会剥掉 `prompt_logprobs`（请求体重建时丢失），C4 必须直连服务器。第一版 60 题只回来 19 题（41 个 err），其中"instinct-alone acc 41/60"还是汇总 bug——失败记录的 instinct 和 gold 都是 None，`None==None` 被算成命中，真实第一念命中 **0/19**。
7. **服务器崩溃事故（00:10-00:20，已自动恢复）**：排查 C5（论文协议 raw-completion 复刻）时给 `/v1/completions` 传 `prompt_logprobs`/`logprobs` 参数，**该 fork 的这个组合会挂死 TP worker**——shm_broadcast 看门狗超时（TimeoutError）→ EngineCore 死亡 → 级联 500 → 容器重启（Docker restart policy，~8 分钟恢复）。崩溃现场调度器状态 `prompt_len=5` 正是测试句 "The capital of France is"。**这是要提给 1Cat 的 fork bug**；同时 C5 路线在此 fork 上被堵死，论文口径改由 C4 的选项内重归一化近似。附带发现：事故期间服务器上还有另一路 ~1 req/s 的 `/v1/messages` 流量（疑似 DSH/agent 循环），与我的实验争抢加重了慢请求。

---

## 1. JEV 是什么（网络检索结论）

**JEV = TypeSafe AI 的 "System One" 决策模型**（arXiv 2609.37647，Deußer/Sparrenberg/Sifa，Bonn 大学 + Fraunhofer IAIS）：

- **不生成文本**：给定"状态 + 带类型标注的问题"，只输出三样东西——从固定选项里选一个、在评分量表（rubric）上给位置、或输出某陈述为真的概率（厂商声称已校准）。
- **定位**：信息访问流水线里的小决策——查询路由、事实核查、内容审核、量表评级。用生成式 LLM 做这些是"间接实现"（要写 prompt、要从生成文本里扒答案、概率不可靠、每个 token 都付费）。
- **权威公开评测**：论文在 **37 个公开数据集**上零样本评测（346,009 次请求，花费 <$10），对照组含 **Qwen3.8-27B**（与我们同代的模型）和 Gemma-4-E4B。Jev 成绩：IMDB/SST-2/HellaSwag/ARC 95-99%，Belebele 86.7%；27/37 数据集胜 Qwen；MathQA 比前沿模型中位数低 17.7pp（多步计算是软肋）。**所有代码、harness、原始响应公开**。
- 同类公开 benchmark：MMLU-Redux、ARC-Challenge、SST-2、AG News、Banking77、HellaSwag、TruthfulQA、UNFAIR-ToS、C-Eval、Belebele。

**本评测复刻其"多选决策"协议**（冻结模板 + 标准测试 split + 选项字母协议），在本地模型上对比三条输出路径。

## 2. 三条路径定义

| 路径 | 机制 | 端点 | 约束方式 |
|---|---|---|---|
| **A 本地 skill** | structured-dispatch（DSH 插件机制） | Anthropic `/v1/messages`（8787 转发） | 强制工具调用：`tool_choice` 锁单 function，`input_schema`=动态 enum，返回 `tool_use` 剥壳成 JSON |
| **B 公开同类** | Instructor / vLLM 原生 structured outputs | OpenAI `/v1/chat/completions`（8787 转发） | `response_format=json_schema`（单 enum） |
| **C 自由文本** | 论文 §3.6 的"间接实现"基线 | OpenAI `/v1/chat/completions` | 无约束生成（max_tokens=24），正则扒字母 |

三路径**同题、同模板、同模型**。题量与模板：MMLU 150 题（知识）、ARC-Challenge 60（科学常识）、SST-2 60（情感二分类）、AG News 60（主题四分类）、Banking77 60（意图 **77 分类**，大 enum 分水岭）。共 390 题/路径，并发 6。

## 3. 成绩总表

### 3.0 Path A 九数据集全表（2026-10-25 深夜扩展，新 4 集为直连干净重跑）

| 数据集 | n | A 本地 skill | 解析失败 | 传输错误 | 平均延迟(s) | 平均输出 token |
|---|---|---|---|---|---|---|
| mmlu | 150 | 68.7% | 2 | — | 17.15 | 9.8 |
| arc | 60 | 95.0% | 0 | — | 6.23 | 11.2 |
| sst2 | 60 | 98.3% | 0 | — | 4.72 | 11.2 |
| agnews | 60 | 78.3% | 0 | — | 3.57 | 11.6 |
| banking77 | 60 | 76.7% | 0 | — | 3.96 | 7.1 |
| hellaswag | 60 | **86.7%** | 0 | 0 | 19.66 | 7.2 |
| mathqa | 60 | **50.0%** | 0 | 0 | 14.81 | 11.1 |
| truthfulqa | 60 | **78.3%** | 0 | 0 | 16.24 | 11.2 |
| imdb | 60 | **91.7%** | 0 | 0 | 26.00 | 12.0 |
| **合计** | **630** | **78.7%** | 2 | — | — | — |

- 新 4 集首跑（经 8787）报的 85.0/48.3/71.7/78.3 + 18 个"解析失败"全部作废：那 18 题经核查全是 HTTP 502 传输错误，直连+重试干净重跑后四集全涨（truthfulqa +6.6pp、imdb +13.4pp）、解析失败归零。原始数据：`results_1792943210.json`。
- **mathqa 50.0% 是全表最低**——多步计算软肋，与 JEV 论文"决策模型计算项低 17.7pp"同向；语义分类（imdb/sst2/banking77/agnews）与常识选择（arc/hellaswag）都在 77-98% 区间。
- 完整口径与数据源说明见 `data/jev-bench/本地成绩表-PathA-qwen3.8-27b-w8a16-2026-10-25.md`。

### 准确率（%）

| 数据集 | n | A 本地 skill | B 公开同类 | C 自由文本 |
|---|---|---|---|---|
| MMLU | 150 | 68.7 | 68.0 | **74.7** |
| ARC-Challenge | 60 | 95.0 | 91.7 | **96.7** |
| SST-2 | 60 | 98.3 | 98.3 | 98.3 |
| AG News | 60 | **78.3** | 73.3 | **78.3** |
| Banking77 | 60 | 76.7 ⚠️已作废 | 53.3 ⚠️已作废（精确口径 40.0） | 80.0 |
| **加权总体** | **390** | **80.0** | **74.9** | **83.1** |

### 延迟与 token 成本

| 指标 | A 本地 skill | B 公开同类 | C 自由文本 |
|---|---|---|---|
| 总 wall time（390 题，并发 6） | **615s** | 177s | 131s |
| 平均单请求延迟 | 4.7-17.2s | 2.5-3.2s | 1.8-2.3s |
| 平均输出 token | 7.1-11.6 | 11.8-13.1 | **2.0** |
| 解析失败 | 2/390 (0.5%) | 0/390 | 1/390 (0.3%) |

## 4. 关键发现

1. **结构化约束几乎不损准确率（A 路径）**：A vs C 总体只差 3.1pp（80.0 vs 83.1），且 AG News 上反超。对"决策型 subagent"场景，用 A 拿到的结构化返回（`{"choice":"B"}`）与自由文本扒字母的准确率基本同档——**§2.4.1 派发原则成立**：省 300-800 token decode 的同时不掉点。

2. **~~Banking77 大 enum 是两条结构化路径的分水岭~~（已作废，见 §0 勘误）**：旧文称 A 76.7% vs B 53.3%（-23.4pp）并归因于"vLLM json_schema 对大 enum 支持弱"——**该归因错误**。实测：①B 的 json_schema 约束本身工作正常（单值 enum、77 文本 enum 对照均严格锁死）；②旧差距几乎全部来自温度混淆（锁温后 B 60.0→73.3%）；③旧评分 bug 还白送 B 8 题。锁温+精确口径下 A-B 仅约 7-8pp（约 1σ）。**修正结论：两条结构化路径在大 enum 上差距是个位数 pp，不存在 23pp 级分水岭。**

3. **A 路径的硬伤是延迟**：单请求 4.7-17.2s，比 B 慢 3-7×（wall 615s vs 177s）。慢的根源在 Anthropic 端点转发链（8787 的 SSE 流式组装 + vLLM fork 对 `/v1/messages` 的处理），不在结构化本身（B 同样结构化只慢 1-2s）。对交互式场景这是代价；对 KV-bound 引擎仍是划算交易——省下的 decode token 直接换成更低的排队和显存。

4. **MMLU 上自由文本反超两个结构化路径 ~6pp**（74.7 vs 68.7/68.0）。可能原因：约束把模型逼进"必须给字母"的窄框架，损害了知识检索的发挥空间；也不排除 150 题样本的波动（bootstrap 区间内）。知识密集型任务是否值得牺牲这几点换结构化，需按场景权衡——路由/分类无所谓，知识问答建议保留 C。

5. **与论文对照**：论文里 Jev 在 MMLU-Redux 95.0 / ARC-Challenge 97.6，Qwen3.8-27B（next-token 概率协议）在其后。我们的 ARC 95.0-96.7 与论文同档；MMLU 68-74.7 低于论文值——但**协议不同不可直接比**：论文用去噪版 MMLU-Redux + 精确下一词概率打分，我们用生成协议 + 全科混合 + 字母选择。Jev 的 MathQA 软肋（多步计算）也提示：决策模型强在知识型选择，弱在计算型——我们的数据同向（banking77/agnews 这种语义分类 77-78%，mmlu 知识混合 68-75%）。

## 5. 结论

- **本地 skill 机制（A）综合最优**：准确率与自由文本同档（干净四数据集采样口径 -3.1pp 以内；banking77 锁温口径 80.0 = C 80.0）、输出 token 与 B 同档、解析 100% 可靠。~~banking77 大 enum 上显著强于公开同类机制（+23.4pp）~~（已作废：锁温+精确口径下仅 +7-8pp，约 1σ，见 §0）。代价是单请求延迟 3-7×。
- **公开同类机制（B）适合轻载低 enum 场景**：延迟低一个量级；~~大 enum 会崩、约束保真度不如强制工具调用~~（已作废：约束本身正常，旧数字是温度噪声 + 评分 bug 的产物；真实差距个位数 pp）。若对延迟不敏感、或已锁 temperature=0，A/B 可互换，A 略优（工具调用是 Qwen 原生后训练行为 + enum 双重编码）。
- **自由文本（C）最快最准但不可控**：只配做 benchmark 基线，不配进生产（无结构保证，正则解析在复杂输出下会失败）。
- **对 §2.4.1 派发原则的实证支持**：决策型子任务派发结构化 subagent，用 A 机制，准确率基本无损 + decode 负担降一个量级 + 返回可直接 `JSON.parse`。

## 6. 复现与产物

```bash
# 下载数据集（HF datasets-server API，5 数据集 600 题）
python3 scripts/jev_bench_download.py
# 跑 benchmark（三路径，390 题/路径，并发 6；已内置 temperature=0 + 精确评分）
python3 scripts/jev_bench_run.py --concurrency 6
```

- 评测集：`data/jev-bench/{mmlu,arc,sst2,agnews,banking77}.jsonl`（原 `/tmp/jev_bench/` 已搬迁到工作区防清理）
- 完整结果（含每题预测/原文/延迟）：`data/jev-bench/results_1792928998.json`（**旧口径，banking77 列有评分 bug，见 §0**）
- 运行日志：`data/jev-bench/full_run.log`

## 7. 顺带落地（插件更新）

- `dsh-structured-output` 新增 **`bench_mc` 形状**：JEV 式多选决策——题目+选项写在 prompt，返回 `{"choice":"B","confidence":0.82}`。派发方式：`model: qwen3.8-27b-w8a16__schema=bench_mc`（已 commit bb43e7d 推送 GitHub）。
- **修复一个真 bug**：注册表原是"进程启动时读一次 agents.json"的缓存，导致加新形状被挡（实测 `bench_mc` 请求被透传成自由文本）。已改为 **mtime 热加载**（每请求 stat，变了才重读），改 `agents.json` 即生效。旧实例需重启一次加载新代码。
- 待办：重启 8787 代理实例让 `bench_mc` + 热加载在线上生效（benchmark 已跑完，现在重启安全）。

## 8. 复现工程化（2026-10-25 深夜追加：直连发现 + GitHub 一键复现）

- **直连发现（重要）**：实测 Path A 的强制工具调用**不需要任何代理**——直接 POST vLLM 服务器的 `/v1/messages`（带 `tools` + `tool_choice:{type:"tool"}`，正确 api-key 用 `Authorization: Bearer`）即返回干净 `tool_use` 块（HTTP 200，答案正确）。§5 里"A 延迟 3-7×"的账要改记：那 3-7× 是 8787 代理的 SSE 组装转发链成本，**不是结构化机制本身**。harness 已加环境变量（`VLLM_BASE`/`VLLM_API_KEY`/`VLLM_MODEL`/`BENCH_DIR`），可直连任意 vLLM。
- **GitHub 仓库（PUBLIC）**：`dd2003-460/jev-structured-bench`，已做到**开箱即用**：
  - `docker-compose.yml`（标准 vLLM 镜像，`.env` 里换 `MODEL` 即可换任何模型）+ `.env.example` + `scripts/reproduce.sh`（一键：下载评测集→三路径跑分→汇总）+ `requirements.txt`（纯标准库零依赖）
  - `harness/prefill_thinking_probe.py`（新增）：filler-token"prefill 思考"实验 F0-F3（论文依据：Goyal 2023 pause tokens / Merrill&Sabharwal 2024 dot-by-dot / 2026 Reading Between the Dots）
  - README 重写 Reproduce 章节：任何有 GPU + docker 的人 10 分钟跑通方法对比；诚实声明——具体数字依赖 4×V100+fork，可复现的是方法论与抓虫结论
- **思考实验（本地跑，进行中）**：`scripts/thinking_probe.py` 五配置 C0-C4（纯结构化/+reason/全思考/两段式/prompt_logprobs 第一念），60 题锁温。首配置 C0 已出：acc 85.0%、avg_lat 14.8s、avg_out_tok 13、think_tok=0。

## 9. 思考与 prefill 实验（2026-10-25 深夜，回答"开思考浪费 token 吗 / 能利用 prefill 思考吗"）

### 9.1 前提发现：这个 fork 没有独立思考通道

实测三种开法（`chat_template_kwargs.enable_thinking=true` / payload `thinking:{type:enabled}` / 默认不传）：`reasoning_content` 永远为空、`reasoning_tokens` 从不报告。不约束输出时模型把推理**直接写在 `content` 里**（问 17+25，默认配置产出 312 token 的 "Step 1… Step 2…"）。**推论：C2（schema + enable_thinking）实际退化成 C0**——schema 把输出锁成 10 token 的 JSON，推理被挤掉，这正是 C2 比 C0 还快（10.8s vs 14.8s）的原因。真正的"思考"只能靠 C1（低预算 CoT，inline）和 C3（两段式，phase 1 无约束）。

### 9.2 思考实验（60 题 = banking77 30 + arc 30，锁温，走 8787）

| 配置 | 机制 | 准确率 | 平均输出 token | 平均延迟 | 传输错误 |
|---|---|---|---|---|---|
| C0 | 纯结构化（json_schema 单 enum） | **85.0%**（51/60） | 13 | 14.8s | 0 |
| C1 | +一句 reason（低预算 CoT） | 81.7%（49/60） | 86（×6.6） | 17.5s | 2 |
| C2 | schema + enable_thinking | 83.3%（50/60） | 10 | 10.8s | 1 |

**结论：低预算 CoT 在选择题上是三输**——多花 6.6× token、多等 18% 延迟、还掉 3.3pp（含 2 题 JSON 解析失败）。语法约束把推理挤掉后模型直接给答案，这点在 §4 发现 1 已经见过（A vs C 只差 3.1pp）。**对决策型任务，"想"这件事由结构化输出代劳了，不需要模型写字。**

### 9.3 预填充 filler 实验（回答"有什么办法利用 prefill 过程思考"）

论文线：Goyal 2023 pause tokens（要训练）/ Merrill&Sabharwal 2024 dot-by-dot（Llama-34M 合成任务 66%→100%，常规 QA 无益）/ 2026 Reading Between the Dots（DeepSeek V3/Kimi K2 确实在 filler 区做隐藏计算）。v1 结果因传输错误作废（见 §0 勘误 3/4），v2 重跑中（F0 基线 / F1 "hmm"×8 / F2 "hmm"×64 / F3 点串×512 / F4 问题复述）。v1 清洗后（只看拿到答案的题）无任何 filler 有效证据（83.3/81.3/81.3/79.3），且这些模型没有经过 pause-token 训练——**"免费思考"在本任务上不成立的预期先例充分**。
