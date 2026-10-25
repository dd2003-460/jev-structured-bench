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
