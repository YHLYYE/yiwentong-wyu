# 邑问通 · 五邑大学学业与资助答疑智能体

基于自研 RAG 全链路的校园政策问答助手：

**混合召回（BM25 + bge-m3 + RRF）→ 三态质检 → 引用来源 → 无据拒答。**

> 定位：只答“学业（含学籍/教务）+ 资助”；就业、校园卡、宿舍、食堂、校车等超范围问题明确拒答并指路。
> 原则：有据才答，无据即拒；宁可少答，不可答错。

---

## 目录

```
05-项目源码/
├── src/
│   ├── models.py              # Chunk / RetrievedChunk
│   ├── rag/
│   │   ├── bm25.py            # 纯 Python BM25（中文 bigram）
│   │   └── hybrid.py          # RRF 融合
│   ├── build_wyu_index.py     # 建 BM25 索引 / chunks.pkl
│   ├── build_wyu_dense.py     # 建 bge-m3 + Faiss 索引
│   ├── wyu_qa.py              # 混合召回 + 三态判定 + 生成/降级
│   ├── interaction_log.py     # 问答日志 / 反馈 / 工作台统计
│   ├── eval_wyu.py            # 20 题快速回归
│   ├── eval_wyu_60.py         # 60 题测试（可切换混合召回/BM25-only）
│   └── eval_scenarios.py      # 5 场景 × 5 题走查
├── ui/
│   └── wyu_app.py             # 学生问答 + 辅导员工作台
├── tests/
│   ├── test_retrieval.py
│   ├── test_refusal_behavior.py
│   └── test_interaction_log.py
├── data/
│   ├── wyu/                   # 公开资料 txt（29 份）
│   ├── wyu_archive/           # 归档的教师/管理员材料（不参与索引）
│   ├── index/wyu/             # chunks.pkl + faiss.index
│   └── logs/                  # 问答日志（gitignore）
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 环境

- Python 3.10+
- GPU 可选：有 CUDA 自动用 GPU，没有则用 CPU（`WYU_DEVICE=auto`）
- 依赖：

```bash
cd 05-项目源码
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements-local.txt
# macOS / Linux
# .venv/bin/python -m pip install -r requirements-local.txt
```

- 模型：`BAAI/bge-m3`（首次运行自动下载，约 2.2GB；国内可设 `HF_ENDPOINT=https://hf-mirror.com`）
- 兼容性：本项目固定 `transformers 4.x`。原因是 bge-m3 的常见 `.bin` 权重在
  `transformers 5.x + torch 2.5` 下会被安全检查拒绝（要求 torch≥2.6）；
  代码会优先读本地缓存，缓存不完整时才联网下载。

## 部署
公开演示默认使用 BM25-only / 离线原文摘录，依赖见 `requirements.txt`；
本地完整模式安装 `requirements-local.txt`。具体步骤见 `DEPLOY.md`。

## 配置

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

`.env` 里只需要填 `DEEPSEEK_API_KEY`；不填也能跑，系统会自动进入**离线原文摘录模式**。

```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-chat
WYU_USE_DENSE=1
WYU_OFFLINE=0
WYU_DEVICE=auto
WYU_COVERAGE_MIN=0.5
# 可选：本地 bge-m3 目录
# WYU_MODEL_PATH=C:/models/bge-m3
# 可选：问答日志路径
# WYU_LOG_PATH=data/logs/qa_log.jsonl
```

> 不要把 `.env` 提交到仓库；`.env.example` 里的 key 留空。

---

## 运行

```bash
cd 05-项目源码

# 1. 建 BM25 索引（先 dry-run 看切片数）
python src/build_wyu_index.py --dry-run
python src/build_wyu_index.py

# 2. 建稠密索引（需要 bge-m3；没有 GPU 会自动 CPU）
python src/build_wyu_dense.py

# 3. 命令行问答
python src/wyu_qa.py

# 4. 60 题测试（建议先输出到临时文件）
python src/eval_wyu_60.py --output C:/tmp/wyu_eval_60.csv

# 5. 网页演示（左侧切换“学生问答 / 辅导员工作台”）
streamlit run ui/wyu_app.py
```

### 降级模式（评委环境友好）

| 场景 | 设置 | 行为 |
|---|---|---|
| 没有 GPU | `WYU_DEVICE=cpu` 或保持 `auto` | 自动用 CPU |
| 没有模型/不想下载 bge-m3 | `WYU_USE_DENSE=0` | 只用 BM25，仍可回答/拒答 |
| 没有 DeepSeek key | 不填 `DEEPSEEK_API_KEY` 或 `WYU_OFFLINE=1` | 不调用 LLM，返回检索原文摘录 + 来源 |
| 演示需要密码 | 设置 `WYU_DEMO_PASSWORD` | Streamlit 才要求密码；不设置直接进入 |

> 降级模式是为了“能复现、能演示”，不代表完整链路的效果；PPT 里的指标应标注模式。

---

## 当前数据状态（务必如实写）

当前索引经实际检查为：

- **29 份** `data/wyu/*.txt`：学业与教务 16 份、学籍 8 份、国际交流 1 份、资助 4 份；
- **135 个** `chunks.pkl` 切片；
- **135 条** Faiss 向量（1024 维）；
- 新增 2024 年课程考核、学分与课程修读、考试纪律文件，以及学籍/学位表格入口；
- 6 份教师/管理员材料保留在 `data/wyu_archive/`，不参与索引。

已完成导航/页脚清洗，并重建 BM25 + Faiss。旧版 50 题离线结果和旧版截图已清理。

注意事项：

- 不要写“答案 100% 正确”；检索命中 ≠ 答案正确，答案正确率需要人工核对；
- 当前 60 题测试中，混合召回与 BM25-only 的来源命中均为 52/52，不能据此宣称混合召回更优；
- 旧截图中如出现 19/94 等数字，不要继续使用。

### 当前实测（60 题，离线模式）

| 指标 | 混合召回 | BM25-only |
|---|---|---|
| 可回答题来源命中 | 52/52 | 52/52 |
| 范围内无资料兜底 | 3/3 | 3/3 |
| 范围外拒答 | 5/5 | 5/5 |

> 这组数字检查“来源是否命中、是否兜底、是否拒答”，不是答案正确率。
> 逐题结果见 `04-测试数据/60题测试结果-混合召回.csv` 和
> `04-测试数据/60题测试结果-BM25-only.csv`。
>
> **真实 LLM 50 题评测**对应扩展前的 19 份资料版本：可回答题进入回答 40/40，
> 无资料兜底 5/5，范围外拒答 5/5；其中 14 题附时效提醒。
> 结果见 `04-测试数据/真实LLM测试结果-50题-扩展前19份资料.csv`，不能与当前 60 题混写。
>
> 当前 60 题测试集：`04-测试数据/60题测试集.csv`。
> 场景走查：`04-测试数据/场景走查-5场景25题.csv`（5 个场景 × 5 题）。

---

## 辅导员工作台与测试日志

- 学生端每次提问自动写入 `data/logs/qa_log.jsonl`；
- 回答下的“有用 / 没用 / 需要人工”按钮会写入反馈；
- 辅导员工作台展示：总提问、已回答、拒答/兜底、需人工、高频问题、
  知识缺口、最近提问，并支持下载 CSV；
- 测试数据索引：`../04-测试数据/测试数据汇总.md`。

> 自己问问题、自己记录可以，但请标注“项目组测试/内部试用”，
> 不要写成“真实用户数据”。外部 5–10 名同学试用会更有说服力。

---

## 检索与拒答机制

1. **BM25**：中文 bigram + 英文/数字词，纯 Python 实现，便于复现；
2. **稠密检索**：bge-m3 + Faiss（IndexFlatIP）；
3. **融合**：RRF 按名次融合，避免两路分数不可比；
4. **阈值**：BM25 < 1.0 且向量 < 0.65 → 直接拒答；
5. **三态判定**：correct / ambiguous / incorrect，非 correct 不生成；
6. **生成**：只根据资料回答，答案标注来源文件名；
7. **降级**：无 key 时返回原文摘录；无 GPU/模型时 BM25-only。

> 阈值和三态判定是当前策略；需要消融实验证明它们比“直接生成”更好，PPT 里要放对比表。

---

## 评测说明

`eval_wyu.py` 把题目分成三类：

| 类型 | 例子 | 期望行为 |
|---|---|---|
| 可回答 | 补考怎么报名 | 检索命中 + 回答 + 来源 |
| 范围内无资料 | 研究生支教团保研政策 | 说明“暂未收录/资料不足”并指路 |
| 范围外 | 食堂几点开门、三方协议怎么签 | 明确拒答 + 官方入口 |

输出 CSV 列：类型、问题、期望来源、检索 Top3、BM25 最高分、向量最高分、
覆盖度、LLM 判定、结果、答案摘要、引用、模式。

**注意：** `检索 top-3` 只说明“期望文档在不在前 3”，不说明答案正确；答案正确率需要人工核对“答案摘要”列。

---

## 数据来源与安全

- 资料来自五邑大学教务处、学生工作部等公开页面，仅用于非商业校园问答演示；
- 非官方答复，以学校最新通知为准；
- 不抓取登录后/内部系统内容；
- 不提交 `.env`、真实 key、个人数据；
- 回答涉及处分/退学/学籍取消等，只讲规则流程，不做结果承诺。

---

## 已知限制

- 已做基础清洗，但仍需人工抽查切片质量；
- 当前 29 份资料、135 个切片；复杂政策仍可能缺少细则；
- 60 题测试仍以来源/兜底/拒答检查为主，答案正确率需人工核对；
- 阈值 1.0/0.65/0.6 尚未做敏感性分析；
- 已有项目组测试/日志/导出机制，尚无外部用户数据；
- 真实 LLM 评测来自扩展前版本，当前 29 份资料尚未重新跑 LLM 50 题；
- 旧截图已删除，PPT 使用当前 04–08。

