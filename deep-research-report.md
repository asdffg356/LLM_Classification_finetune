# 适合本科课设的 LLM Classification Finetuning 实验方案

## 比赛任务与这门课设真正要做的事

你研究的这场 Kaggle 比赛，官方任务非常明确：给定同一个用户 prompt 下两个大模型回答 `response_a` 和 `response_b`，预测人类评委更偏好 A、B，还是认为两者平局；提交文件要为 `winner_model_a`、`winner_model_b`、`winner_tie` 三个类别输出概率，官方评价指标是 **log loss**。Kaggle 的数据页还说明，`model_a` 和 `model_b` 只在 `train.csv` 里出现，在 `test.csv` 中不提供，因此模型不能把“某个模型名字通常更强”当成核心特征，必须主要根据 prompt 与两段回答内容来判断偏好。citeturn2search0turn13search1turn1view7

LMSYS 为这类比赛提供的背景也很关键。官方博客写明，竞赛数据来自 Chatbot Arena，训练集包含 **5.5 万+** 条真实用户对话与偏好标注，隐藏测试集约 **2.5 万** 条，覆盖 GPT-4、Claude、Llama、Gemini、Mistral 等 **70 多个** 模型。也就是说，这不是普通的情感分类题，而是“基于真实人类投票的偏好预测”题。citeturn1view3

把它翻译成适合本科机器学习课设的话，其实可以理解成一句话：**训练一个三分类模型，模拟人类在两个聊天机器人回答之间的选择。** 这和 RLHF 里的“奖励模型”很接近，但又没有 RLHF 那么重。Chatbot Arena 论文说明，平台采用匿名、随机的两两对战，让用户在两个回答中投票，并已经收集了大规模人类偏好数据；论文还报告，Arena 的众包偏好与专家评分有较好一致性，因此这类数据可以被视为学习“人类偏好”的可靠监督信号。citeturn1view4turn1view5

不过，**这门课设最合适的做法不是把它包装成完整 RLHF 系统，而是老老实实当成三分类问题来做。** 这是因为 Hugging Face TRL 的 `RewardTrainer` 本质上要求 `num_labels=1`，输出单个标量 reward，更适合 Bradley–Terry 风格的“二选一偏好建模”；而这场比赛把 **tie** 当成一等公民类别，4th place 公开方案 `pairjudge` 也明确指出，标量 Bradley–Terry reward model 不能把 tie 作为第一类标签自然表示出来，因此对本科课设来说，**直接做 A/B/tie 三分类，比硬上 reward model 更稳、更直观、更容易解释。** citeturn16search2turn10view0turn10view1

## 路线可行性评估

### 传统机器学习基线最适合拿来打底

**TF-IDF + Logistic Regression** 是你最该做的 baseline。理由很简单：它对文本分类非常成熟，`LogisticRegression` 原生支持三分类的 multinomial loss，并且天然输出概率，和比赛的 log loss 指标高度匹配；而这个具体比赛已经有公开的 TF-IDF + Logistic Regression baseline notebook，说明这条路线不仅理论上可行，而且在该竞赛语境下已经被很多人复现过。citeturn8search2turn18search2turn18search7

从课程设计角度看，这条路线的优点不是“分数高”，而是“**必能跑通**”。它难度低，几乎不依赖 GPU，调试链条短，非常适合本科生用来搭建完整实验框架：数据读取、文本拼接、标签转换、训练验证划分、log loss/accuracy/F1 评价、错误样本分析，都能先在这个 baseline 上完成。它的弱点也很明确：只能吃到词面模式，对长文本细粒度语义判断较弱，所以它更适合做对照组，而不适合做主模型。citeturn18search2turn8search2

**Linear SVM** 也能做文本分类，但这题官方要的是概率而不是硬标签。`LinearSVC` 虽然对稀疏文本特征很友好，但官方文档强调它走的是 one-vs-rest 方案，而概率输出通常要借助 `CalibratedClassifierCV` 额外校准，这会提升实现复杂度。对一门“要稳”的课设来说，它不是最优选择。citeturn14search0turn14search1turn14search3

**LightGBM** 可以作为特征工程路线的补充，但不建议当主线。没有高质量手工特征或 dense embedding 时，LightGBM 对纯文本原始输入并不如 TF-IDF + LR 那样“拿来就用”；如果只是为了展示“我还试了一个树模型”，可以做一个轻量版本，但它不应该挤占你主实验的时间。这个判断也和公开 notebooks 的生态一致：这个题更常见的是 TF-IDF 基线、encoder 微调、或更重的 LoRA/QLoRA，而不是把 LightGBM 当核心路线。citeturn18search2turn7search10turn7search14

### Encoder 微调是最稳的主线

如果只选一条**真正适合本科课设、且能明显强于 baseline** 的路线，我会选 **DeBERTa-v3-base 的三分类微调**。原因有三层。第一，DeBERTa-v3-base 的模型规模还在可控范围内：模型卡给出的 backbone 仅 **86M** 参数，远没有 7B 那么重。第二，这个比赛已经有公开的 DeBERTa baseline notebook，说明在实际竞赛环境里它就是一条成熟路线。第三，Hugging Face 官方 sequence classification、padding/truncation、Trainer 与 gradient checkpointing 文档都很完善，工程风险低。citeturn5search0turn0search7turn8search0turn8search1turn5search2

**RoBERTa-base** 也可行，它是经典 English encoder，生态成熟、资料多、代码模板丰富。官方模型卡说明它是标准的 masked language model 预训练编码器，因此改成 sequence classification 路线很自然。只是从你这个题目的“稳妥完成”目标出发，RoBERTa-base 更像 DeBERTa 的备选，而不是必须首选。citeturn12search0turn8search0

**DistilBERT** 的价值在于救场。官方文档与模型卡都强调，它比 BERT base 更小、更快，训练与推理成本更低；如果你的本地显存、时间或驱动环境不稳定，它是非常好的降级方案。问题是，它通常也意味着更弱的表征能力，所以更适合“显存不够时的 Plan B”，而不是最优主模型。citeturn5search1turn5search9turn5search13

**ModernBERT** 的卖点是原生 **8192** 上下文，理论上很适合这道“prompt + 两个长回答”的任务；但它较新，工程生态和比赛现成代码不如 DeBERTa 稳，且你这次目标不是冲榜，而是顺利交付课程设计。因此，我会把它放在“可了解、不建议首发”的位置。citeturn11search1turn11search6

综合来看，encoder 微调这条路线的结论非常清楚：**难度中等、算力要求中等、失败风险可控、最适合作为主方案。** 对你这种“有 Python 基础、能借助 AI 写代码、但不想被大模型训练细节拖死”的同学来说，这是最合身的路线。公开 notebook 已经表明，DeBERTa 和 TF-IDF 都是这个比赛的常见可复现做法。citeturn0search7turn18search2

### LoRA 与 QLoRA 适合当扩展，不适合当主线

从研究角度说，LoRA 和 QLoRA 很重要。LoRA 论文说明，它通过在原模型中注入低秩适配矩阵，大幅减少可训练参数与显存消耗；QLoRA 论文进一步说明，4-bit 量化加 LoRA 甚至能把超大模型的微调压到单卡可做的范围。Hugging Face 的 PEFT 与 bitsandbytes 文档也说明，这类方法确实是“有限资源微调大模型”的主流工具。citeturn6search1turn6search0turn5search3turn6search2turn6search5

但**能做**不等于**适合作为你的主线**。Kaggle 官方总结 winning solutions 时提到，Gemma-2-9b 在这类偏好数据上表现很强，优秀方案还大量使用了 reward/ranking models、知识蒸馏、伪标签与多个 LoRA 平均等技巧；公开获奖方案摘要也显示，顶级选手往往会“生成更多训练数据、尽可能微调更大的模型、并做更复杂的高效推理”。这说明 LoRA/QLoRA 的真实竞赛上限很高，但同时也意味着它往往伴随更复杂的数据管线、更长的调参时间和更高的失败概率。citeturn4view3turn17search22turn17search13turn10view1

对本科课设来说，这条路线最大的风险不在“代码不会写”，而在“**你会被工程问题拖死**”：显存溢出、量化兼容、pad token、长上下文截断、训练不稳定、inference 太慢、成绩不稳定，这些都是真实问题。尤其是公开的 4th place 方案明确强调，长输入上的粗糙截断会直接破坏比较信息，甚至让模型学到位置伪迹；他们还专门做了位置偏差修正和伪标签蒸馏，这远超一门普通课程设计所需的复杂度。citeturn10view3turn10view0turn10view1

所以我的建议很明确：**LoRA/QLoRA 可以作为“有余力再试”的加分实验，但绝不应该成为你的主方案。** 你的课设目标是“做成一篇完整、可信、可演示的实验报告”，不是复刻金牌方案。citeturn6search0turn17search22turn10view1

### 特征工程、A/B 交换增强、简单融合都值得做

这题非常值得做一些**小而稳的增强**。其中最值得做的是 **A/B 交换增强**。4th place 公开方案给出了一个很强的经验事实：在真实偏好数据上，一个小型 LoRA judge 把同一对回答顺序对调后，**有 29.2%** 的样本会翻转判断；该方案也明确采用了 swap-debias averaging。对于你的课设，这几乎就是“必须做的实验证据”：说明模型确实可能存在位置偏差，而 A/B 交换增强可以作为一个非常合理、非常好写进报告的改进点。citeturn10view0

**长度特征、结构化特征、是否包含代码、是否拒答、标点数量** 这些特征也值得尝试，但建议把它们放在“轻量级附加分析”而不是“大型手工特征工程”位置。原因是公开 notebooks 已经有人在这题上加入长度、sentiment、结构化清洗等特征；这说明路子存在，但它更适合做附加解释，而不是和 encoder 主模型抢主线。citeturn15search4turn15search0

**概率平均融合** 也值得做，因为它实现最便宜：把 TF-IDF + LR 的三分类概率与 DeBERTa 的 softmax 概率按 0.2/0.8、0.3/0.7 等简单权重平均，然后在验证集上选最优权重即可。顶级竞赛实践也表明，蒸馏、伪标签、adapter 平均和集成确实有效；但在你的课设里，完全没必要做复杂 stacking，做一个“简单均值融合”就够了。citeturn4view3turn17search10

## 推荐的最终方案

### 稳妥主方案

你的主方案，我建议就定成下面这个版本：

**主线模型组合：**
- **Baseline**：TF-IDF + Logistic Regression  
- **主模型**：DeBERTa-v3-base 三分类微调  
- **固定分析项**：log loss、accuracy、macro F1、混淆矩阵、错误案例分析  
- **固定增强项**：A/B 交换增强  
- **可选增强项**：长度特征、简单概率融合  

这是最适合你的原因并不是“学术上最先进”，而是它同时满足四件事。其一，官方任务本身就是三分类概率输出，Logistic Regression 和 sequence classification 都天然匹配；其二，公开 notebooks 已经证明 TF-IDF baseline 与 DeBERTa baseline 都能在这个比赛里工作；其三，DeBERTa-v3-base 的规模对普通显卡更友好；其四，A/B 交换、长度偏差、tie 样本分析都非常适合写成课程报告里的“实验洞察”。citeturn2search0turn18search2turn0search7turn5search0turn10view0

我不建议你把主模型定成 RoBERTa、ModernBERT 或 DistilBERT，原因分别是：RoBERTa相对可行但不如 DeBERTa 贴近该题公共基线生态；ModernBERT 虽然长上下文强，但生态较新；DistilBERT 则更像性能换稳定的备用方案。综合“稳定交付”优先级之后，**DeBERTa-v3-base 是主模型，DistilBERT 是兜底模型**，这是最现实的搭配。citeturn12search0turn11search1turn5search1turn5search0

### 加分扩展方案

如果主方案先跑通，再加下面三项就足够“看起来像认真做过研究”：

第一，**A/B 交换增强**：对训练集复制一份，把 `response_a` 与 `response_b` 互换，同时把 A 胜/B 胜标签对换，tie 保持不变。这个增强有清晰的竞赛动机，因为公开金牌方案明确测到了显著位置偏差。citeturn10view0

第二，**长度与结构化特征拼接或后融合**：例如加入 prompt 长度、A/B 回答长度、长度差、是否出现 markdown 列表、是否出现代码块、是否为空回答、是否完全相同。公开方案与 notebook 生态都表明，这类轻量元特征可以作为辅助信息。citeturn15search4turn17search17

第三，**简单融合**：把 TF-IDF baseline 的概率与 DeBERTa 的概率平均。因为这两类模型擅长的东西不同：前者偏词面统计，后者偏语义表征，简单平均往往能带来一点稳健性。顶级竞赛复盘也强调了集成、蒸馏、伪标签的重要性；你的版本可以做最轻量的“概率均值融合”。citeturn4view3turn17search10

至于 **LoRA/QLoRA 微调 7B 左右模型**，我建议你把它写进“扩展工作设想”，不要写进“必须完成的实验设计”。如果你后面确实有余力，最多尝试一个很小的 LoRA 分类实验作为附录，不要让它影响主线按时完成。citeturn6search0turn17search22

## 详细实验流程

### 数据下载、清洗与 EDA

数据阶段建议分成三小步。第一步，读取 `train.csv`、`test.csv`、`sample_submission.csv`，把三列目标 `winner_model_a`、`winner_model_b`、`winner_tie` 转成单标签类别，例如 A→0、B→1、tie→2。训练集里的 `model_a`、`model_b` 可以保留做 EDA，但不要把它们放进最终主模型特征，因为测试集没有这两列。citeturn13search1turn1view7

第二步，做最基本的数据清理：空字符串填充、去掉异常 `NaN`、统一换行符，并统计 prompt、response_a、response_b 的字符数或词数。特别要注意长文本问题。Hugging Face 文档说明，padding 和 truncation 是批处理可训练输入的必要步骤；而 4th place 方案进一步提醒，粗暴截断会使比较信息被破坏，甚至可能把 prompt 或 response B 截没。对这题来说，这不是细节，而是成败点。citeturn8search1turn10view3

第三步，做 EDA。你至少应该产出四张图：标签分布图、prompt 长度分布图、response 长度分布图、长度差与胜负关系图。这里最值得写进报告的不是“图画得多漂亮”，而是你要明确提出几个可验证问题：**平局是否明显更难学？更长回答是否更容易赢？A/B 是否存在位置偏差？** 这些分析会直接服务后续实验设计。Chatbot Arena 的背景本来就是用户在真实场景中对两个回答进行偏好选择，因此围绕“回答风格与偏好”的分析非常自然。citeturn1view4turn1view3

### Baseline 方案

TF-IDF baseline 我建议用最容易复现的输入拼接方式：

```text
[Prompt]
{prompt}

[Response A]
{response_a}

[Response B]
{response_b}
```

对这个长文本做词级或词+字符级 TF-IDF，然后接 `LogisticRegression` 的 multinomial 三分类。官方文档说明 `LogisticRegression` 在多分类情形下可以直接优化 multinomial loss；这和比赛的三分类概率输出非常匹配。公开 baseline notebook 也正是沿着“TF-IDF + Logistic Regression”这条路做的。citeturn8search2turn18search2turn18search7

训练验证划分建议用 **stratified 80/20 holdout**；如果时间允许，再做 **5-fold stratified CV**。本科课设里，holdout 已经足够支持报告撰写，因为你更需要的是一套稳定的分析流程，而不是极致分数。评价指标固定成三项：**log loss** 作为主指标，**accuracy** 和 **macro F1** 作为辅助指标。这样很好解释：log loss 对应官方竞赛；accuracy 方便直观比较；macro F1 能提醒你别忽视平局类。citeturn2search0

作为补充模型，如果你想多做一个传统 ML 对照，优先级应该是 `LinearSVC + CalibratedClassifierCV`，而不是直接裸用 `LinearSVC`，因为后者没有现成概率输出。只是对你的课设来说，我会把这条线放到“有空再做”，不建议它替代 Logistic Regression。citeturn14search0turn14search1turn14search3

### 预训练模型微调方案

我建议你的主模型直接定为 **`microsoft/deberta-v3-base` + `AutoModelForSequenceClassification(num_labels=3)`**。原因已经说过：模型规模适中、文档成熟、与公开比赛基线贴近。Hugging Face 的 text classification 文档就是一条标准 sequence classification 微调流水线；padding/truncation 文档和 Trainer 文档也给了足够稳定的训练接口。citeturn5search0turn8search0turn8search1turn5search2

输入模板仍建议保留结构标签，并且尽量显式地区分三个部分：

```text
Prompt:
{prompt}

Response A:
{response_a}

Response B:
{response_b}
```

但**不要把原始超长文本一股脑塞进去**。结合官方 truncation 机制与 4th place 的经验，我更推荐“**分段裁剪后再拼接**”而不是“对总串做一次性截断”。一个务实配置是：`prompt` 预裁到 96–128 tokens，`response_a` 和 `response_b` 各裁到 128–192 tokens，再拼接成总长 **384 或 512**。如果你在 Kaggle GPU 上跑，优先尝试 512；如果在本地 4060 级别笔记本上跑，优先从 384 起步。这样做的目的不是追求最全信息，而是尽量避免某一边回答被截断得过于严重。citeturn8search1turn10view3

一个非常稳的训练超参数起点如下：
- `max_length=384` 本地优先，`512` 作为 Kaggle GPU 版本  
- `per_device_train_batch_size=2` 或 `4`  
- `gradient_accumulation_steps=4`  
- `learning_rate=2e-5` 或 `1.5e-5`  
- `num_train_epochs=2` 到 `3`  
- `fp16=True`  
- `gradient_checkpointing=True`  
- 动态 padding，按 batch 中最长样本补齐  

这里面最关键的是 `fp16` 和 `gradient_checkpointing`。Hugging Face 官方 Trainer 文档明确说明，gradient checkpointing 用计算时间换显存，通常能显著减小内存占用，代价是训练大约慢 **20%**；这对你这种单卡资源是很值得的。citeturn5search2turn5search14

如果你真的遇到显存问题，降级顺序不要乱试，直接按下面顺序来：**先把 batch size 降到 2，再把 max_length 从 512 降到 384，再开/保留 gradient checkpointing，再把主模型从 DeBERTa 换成 DistilBERT。** DistilBERT 官方文档就强调了它更小、更快、更省资源，所以它是最合理的“兜底主模型”。citeturn5search1turn5search9turn5search13

### 可选增强、错误分析与提交

**A/B 交换增强** 建议一定做。你可以先训练一个不增强版本，再训练一个增强版本，用验证集比较 log loss 是否下降。这样报告里就能清楚回答“位置偏差校正有没有帮助”。考虑到公开方案已经测到明显 flip-rate，这个实验很有说服力。citeturn10view0

**长度特征** 可以作为独立分析，也可以作为融合特征。例如先训练一个小的 tabular 模型，输入 `len_prompt`、`len_a`、`len_b`、`len_diff`、`is_identical`、`has_code_a`、`has_code_b` 等，然后把它的输出概率跟主模型做平均。你甚至不需要复杂建模，先把这些特征用于数据分析和错误案例归因，就已经足够为报告增色。公开 notebooks 确实有人在这题上加入长度、情绪或结构类特征。citeturn15search4turn15search0

错误案例分析建议固定四类：**平局误判**、**长回答偏好误判**、**位置翻转误判**、**空回答或近似重复回答**。4th place 方案还专门提到，空样本和完全相同回答这类退化情形做规则处理能带来可测量的 log loss 收益；你不一定要做规则后处理，但至少应该在报告里分析这些特殊样本。citeturn10view1

如果你最后真的向 Kaggle 提交，生成 `submission.csv` 的方法非常简单：按官方格式保留 `id` 与三列概率，确保每行三项之和为 1。还要知道 Kaggle 榜单的分法：该比赛榜单页面说明，**public leaderboard 约基于 55% 测试集**，最终名次基于剩余 **45%**。这意味着你在课程项目里不应该把一次 public score 当成唯一结论，验证集结果与误差分析同样重要。citeturn2search0turn0search22

为了方便你实际记结果，建议从第一天就固定一个结果表，字段包括：模型名、是否 A/B 交换增强、max_length、batch size、learning rate、epoch、验证集 log loss、accuracy、macro F1、备注。你后面写报告时，这张表会非常省事。citeturn2search0turn5search2

## 项目结构与代码实现建议

### 推荐项目结构

我建议你把项目拆成下面这个结构：

```text
project/
  data/
    raw/
    processed/
  notebooks/
    01_eda.ipynb
    02_tfidf_baseline.ipynb
    03_deberta_finetune.ipynb
    04_error_analysis_and_fusion.ipynb
  src/
    config.py
    data_utils.py
    features.py
    metrics.py
    train_tfidf.py
    train_transformer.py
    inference.py
    postprocess.py
  outputs/
    figures/
    models/
    oof/
    submissions/
    logs/
  report/
    outline.md
    figures/
    references.md
```

这个拆法的好处是非常符合课程作业节奏：`01` 先做 EDA 并出图；`02` 跑 baseline；`03` 跑主模型；`04` 统一做错误分析与融合；`src/` 里保留可复用函数，后期从 notebook 迁移成 `.py` 也不痛苦。公开项目与 notebooks 生态基本也都是按“数据理解 → 清洗 → 特征工程 → 训练 → 评估 → 提交”这样的顺序展开，所以这套结构很顺手。citeturn1view7turn18search2turn0search7

### 关键库与关键函数

关键库建议只用一套大众组合，不要引入太多花活：  
`pandas`、`numpy`、`scikit-learn`、`matplotlib`、`transformers`、`datasets`、`torch`、`accelerate`。如果你后面真的做 LoRA 扩展，再额外加 `peft` 和 `bitsandbytes`。这样你的环境最稳，也最容易让 AI 帮你补代码。Hugging Face 文档已经覆盖了 sequence classification、Trainer、padding/truncation、PEFT 和 bitsandbytes 等核心部分。citeturn8search0turn5search2turn5search3turn6search2

你应该尽快固定以下函数接口：
- `load_data()`：读入 train/test  
- `build_label()`：把三列 winner 转成单标签  
- `make_text_input()`：把 prompt/A/B 拼接成输入文本  
- `compute_basic_features()`：长度、差值、结构化特征  
- `train_tfidf_lr()`：baseline 训练  
- `tokenize_batch()`：transformer 分词  
- `compute_metrics()`：log loss / accuracy / macro F1  
- `predict_proba_test()`：输出测试集三类概率  
- `swap_augment()`：生成 A/B 交换增强集  
- `analyze_errors()`：导出最难样本与混淆矩阵  

这些都是“高频、低风险”的函数，不需要你一次写得多漂亮，但必须早早定住。这种模块化做法既符合 scikit-learn/Hugging Face 的常见工作流，也能降低你依赖 AI 生成代码时的混乱度。citeturn8search0turn8search2turn5search2

### 最容易出错的地方

这题最容易出错的地方其实不是模型，而是**输入处理**。第一类错误是标签处理错，把三列 one-hot 弄成多标签而不是单标签三分类。第二类错误是把 `model_a/model_b` 无意识喂进模型，导致本地验证“看起来很强”，但测试集没法用。第三类错误是长文本粗暴截断，使 response B 经常被截没，或者 prompt 信息丢失。公开 4th place 方案已经明确把这件事点出来了。citeturn13search1turn10view3

第四类错误是**指标不一致**。官方主指标是 log loss，所以你不能只盯 accuracy；模型即便分类对了，如果概率过度自信，也可能 log loss 很差。第五类错误是**没有保存 OOF 或验证集预测概率**，后面想做融合与错误分析时只能重跑。第六类错误是**显存策略乱试**，应该优先通过动态 padding、较小 batch、梯度累积、fp16、gradient checkpointing 去控内存，而不是一上来就切到复杂量化。citeturn2search0turn5search2turn8search1

## 报告与视频应该怎么组织

### 论文式报告框架

你的报告题目《基于 Chatbot Arena 对话数据的人类偏好预测模型研究》是合适的。整篇报告最重要的是让老师看到：你不是“跑了一个 notebook”，而是**完成了一次完整的机器学习实验**。下面这套写法很稳。

**摘要** 要交代任务、方法、结果与结论四件事：你做的是基于 Chatbot Arena 数据的人类偏好三分类；方法包括 TF-IDF + Logistic Regression 和 DeBERTa 微调；结果上主模型优于 baseline；并分析了 tie 样本、长度偏差与 A/B 位置偏差。摘要不需要大段背景，只要把“做了什么、得到什么”写清楚。比赛任务与背景可直接依据 Kaggle 官方描述和 LMSYS 介绍来写。citeturn0search0turn1view3turn1view4

**引言** 重点写“为什么这个题有意义”。你可以从传统 benchmark 与真实用户满意度存在差距切入，再引出 Chatbot Arena 通过真实用户 pairwise voting 提供了新的偏好监督信号，最后说明本研究将该问题建模为 A/B/tie 三分类，以探索人类偏好预测是否能通过中小模型稳定完成。这个逻辑和 LMSYS 博客与 Chatbot Arena 论文是完全一致的。citeturn1view3turn1view4

**相关工作** 不要写太多。三块就够：Chatbot Arena 的人类偏好评测；RLHF/Reward Model 的偏好建模思想；预训练编码器在文本分类中的常规微调。这里尤其适合加一句：虽然 reward model 与本题相关，但本课设采用显式三分类，是因为官方任务包含 tie，而常见标量 reward modeling 不自然支持这一点。citeturn1view4turn16search2turn10view1

**方法** 这一节要放三张最重要的图/表：数据字段说明表、整体流程图、模型结构/输入模板示意。内容按“数据预处理 → baseline → DeBERTa 微调 → A/B 交换增强 → 可选融合”顺序写。这里最好明确写出：训练集有 `model_a/model_b` 但测试集没有，因此正文不使用模型身份作为主特征；同时说明长文本采用分段裁剪以减轻比较信息丢失。citeturn13search1turn10view3

**实验** 这一节至少要有六个东西：实验环境、数据划分方式、评价指标、主对比表、增强实验表、错误案例图表。对本科课设来说，不需要做过多 ablation，但必须有“baseline vs 主模型”的清晰对比，而且最好再有 “加不加 A/B 交换增强” 的对比。citeturn2search0turn10view0

**结果分析** 是最容易拉开你和普通作业差距的一节。建议围绕四个问题展开：为什么 DeBERTa 比 TF-IDF 更强；为什么 tie 样本更难；回答长度是否影响偏好；为什么顺序交换增强可能有效。这个部分可以插入几条具体错误样本，做人工解释。因为这道题本来就是“人类偏好”任务，所以错误案例分析会比纯数值更有说服力。citeturn1view4turn10view0

**结论** 要克制：不要写“显著提升大模型对齐能力”这种大话。更合适的表述是：在 Chatbot Arena 偏好预测任务上，传统文本分类 baseline 可以建立可复现下界，预训练 encoder 微调能进一步提升预测质量，而位置偏差与 tie 样本仍是主要难点。这样的结论既真实，也适合课程设计。citeturn1view3turn10view0

**成员贡献** 就按实际分工写：数据与 EDA、baseline、主模型训练、报告撰写、视频制作。不要空泛。citeturn1view7

### 十分钟视频的稳妥脚本

视频非常建议按“问题—方法—结果—分析—结论”五段式展开。开头 1 分钟讲比赛和任务；中间 2 分钟讲数据与 EDA；再用 2 分钟讲 TF-IDF baseline；再用 2 分钟讲 DeBERTa 主模型与 A/B 交换增强；接着用 2 分钟讲结果表和错误案例；最后 1 分钟总结。这样结构最像正规的课程答辩，也最不容易超时。任务背景和比赛设置本身就很适合口头解释，因为它和“让 AI 更符合人类偏好”高度相关。citeturn0search0turn1view3turn1view4

## 明确结论

最终我给你的结论非常明确：

**你应该采用这条路线：**
**TF-IDF + Logistic Regression 作为 baseline，DeBERTa-v3-base 三分类微调作为主模型，A/B 交换增强与错误案例分析作为核心亮点，长度/结构化特征与简单概率融合作为可选加分项。** citeturn18search2turn0search7turn5search0turn10view0

之所以是这条路线，而不是 7B + QLoRA 主线，原因有四个。第一，官方任务本质就是三分类概率预测，Logistic Regression 与 sequence classification 天然匹配。第二，这个具体比赛已经有公开的 TF-IDF 与 DeBERTa baseline，复现风险低。第三，DeBERTa-v3-base 的模型规模对普通单卡更友好，配合 fp16、动态 padding 与 gradient checkpointing，完成训练是现实的。第四，A/B 交换增强、长度偏差、tie 分析这些实验点都非常适合写成一篇完整的本科机器学习课设报告。citeturn2search0turn18search2turn0search7turn5search0turn5search2turn10view0

换句话说，**这不是“最强冲榜方案”，但它是“最可能被你结合 AI 辅助真正做成”的方案。** 如果你照这个结构执行，代码规模可控，实验结果有对比，报告有分析，视频也能讲清楚；而这正是本科课程设计最需要的完成度。citeturn1view3turn4view3turn17search13