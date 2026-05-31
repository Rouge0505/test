# Mini-LLM 设计文档

本文档记录架构选型理由、原理推导、方案对比。参数值以各 run 的 `config.json` 为准，此处不硬编码。

---

## 一、模型架构

### 1.1 架构全景

```
输入: token_ids (B, S)     B=batch_size, S=seq_len
         │
         ▼
┌─────────────────┐
│   Embedding     │  (B, S) → (B, S, d_model)
│                 │  权重与 LM Head 共享 (weight tying)
└────────┬────────┘
         │
         ▼ ×N 层 (每层权重独立，仅 RotaryEmbedding 共享)
┌─────────────────────────────────────────────────┐
│  TransformerBlock                               │
│  ┌───────────────────────────────────────────┐  │
│  │  RMSNorm → Attention → Residual Add       │  │
│  │           ┌──────────────────────┐        │  │
│  │           │ W_QKV → Q, K, V      │        │  │
│  │           │ RoPE(Q, K)           │        │  │
│  │           │ Q@K.T → mask → softmax│        │  │
│  │           │ attn@V → W_O         │        │  │
│  │           └──────────────────────┘        │  │
│  └───────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────┐  │
│  │  RMSNorm → SwiGLU FFN → Residual Add     │  │
│  │           ┌──────────────────────┐        │  │
│  │           │ W_gate(x) → SiLU     │        │  │
│  │           │ W_up(x) → 门控乘法    │        │  │
│  │           │ W_down → 输出         │        │  │
│  │           └──────────────────────┘        │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│   RMSNorm       │  final_norm
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LM Head       │  F.linear(x, embedding.weight)
│                 │  → logits，对 vocab 的概率分布
└─────────────────┘
```

### 1.2 技术路线：LLaMA 而非 GPT-2

采用 LLaMA 风格架构，因为 LLaMA 是当前所有主流开源 LLM 的基础（LLaMA 2/3、Mistral、Qwen、DeepSeek），学到的知识直接可迁移。

### 1.3 RoPE 而非可学习位置嵌入/正弦编码

理由：
1. **零额外参数** -- 可学习位置嵌入需 `max_seq_len × d_model` 个参数，RoPE 为零
2. **相对位置感知** -- 在注意力分数中编码 token 间的相对距离，语言学上更合理
3. **长度泛化** -- 推理时可通过 NTK-aware scaling 外推到训练长度之外
4. **行业标准** -- LLaMA 1/2/3、Mistral、Qwen、DeepSeek、Gemma 均使用
5. **数值稳定** -- 经典加法 `x = emb + pos_enc`，pos_enc 固定在 [-1,1]，若 emb 幅值远大于此（训练中可能增长到 10+），位置信号被淹没成噪声。RoPE 是正交旋转，不改变向量模长，位置编码在角度中，不和幅值竞争

**核心数学性质**：两个位置 $m$, $n$ 的内积仅依赖相对距离 $(m - n)$：

$$\langle \text{RoPE}(\mathbf{q}, m),\; \text{RoPE}(\mathbf{k}, n) \rangle = f(\mathbf{q}, \mathbf{k}, m - n)$$

这使得注意力分数天然编码相对位置信息，无需额外参数。

**相邻配对 vs 前后半段配对**：

原始论文用相邻配对 $(x_{2k}, x_{2k+1})$，HuggingFace 实现用前后半段 $(x_k, x_{k+d\_head/2})$。两者数学等价——只是维度排列不同，对注意力计算结果无影响。我们选择相邻配对，与原始论文一致，更直观。

**为什么不缓存 cos/sin 表**：

- 训练时三角函数计算占比 < 0.01%，缓存无可测量提升
- 推理时每步只需 1 个位置 × d_head/2 次三角函数（~1 微秒），对比矩阵乘法可忽略
- 缓存方案需要预设表大小，推理时 pos 随滑动窗口持续递增，没有合理上限
- 取模方案不可行——会导致 RoPE 相对距离符号反转和位置重叠
- 实时计算代码最简洁，训练推理统一逻辑

**inv_freq 存储**：算一次存 buffer（`register_buffer(persistent=False)`），cos/sin 每次 forward 实时算。不存入 checkpoint，加载时由 RotaryEmbedding 构造函数重算。

**API 设计选择：共享实例 vs 顶层分发**

HuggingFace 做法：Model 顶层算 cos/sin → 传给每层 Attention → 各层调 `apply_rotary_pos_emb(q, k, cos, sin)`。原因是它要支持各种 RoPE 变体（NTK scaling、YaRN 等），需要解耦。

我们的做法：Model 持有单个 RotaryEmbedding 实例，构造时注入各层 Attention 共享引用。调用方只需 `self.rope(q, k)` 或 `self.rope(q, k, offset=pos)`。好处：
1. 逻辑上它就是一张共享常量表，单实例避免读者疑惑"各层参数不同吗？"
2. 训练时零参数（offset 默认 0，position 内部 arange 生成）
3. 推理时一个 offset 参数，含义清晰
4. 不需要在 Model → Block → Attention 间穿透 cos/sin
5. PyTorch 对共享子模块处理正确：buffers() / parameters() 自动去重

**实现风格选择：全宽运算（方案 A）vs 半宽拆分（方案 B）**

方案 A（全宽运算，项目采用）：
```python
cos = repeat_interleave(2)  # (d_head/2,) → (d_head,)
q_rot = q * cos + rotate_half(q) * sin
```

方案 B（半宽拆分）：
```python
q_even, q_odd = q[..., 0::2], q[..., 1::2]
q_even_rot = q_even * cos - q_odd * sin
q_odd_rot  = q_even * sin + q_odd * cos
q_rot = torch.stack([q_even_rot, q_odd_rot], dim=-1).flatten(-2)
```

选择方案 A 的理由：
1. **行业通用** -- HuggingFace / Meta LLaMA / Mistral / Qwen 全部使用此模式
2. **单表达式** -- 主逻辑只有一行，不需管理 4 个中间变量
3. **无 strided view** -- `q[..., 0::2]` 是非连续内存视图，后续运算可能隐式触发 `.contiguous()` 拷贝

方案 B 的优势（教学角度）：每行直接映射 2D 旋转矩阵，cos/sin 保持自然尺寸，不需要 `rotate_half`。但行业统一用方案 A 主要是路径依赖，性能差异不可测量。

### 1.4 MHA 而非 GQA/MQA

小参数模型的 KV-cache 极小，GQA/MQA 的推理优化在此规模无意义。MHA 概念最清晰，是 GQA/MQA 的基础。GQA 留到更大模型。

**QKV 融合投影**：W_Q/W_K/W_V 合为单个 W_QKV，一次 GEMM 代替三次。参数量不变（3 × d_model² = d_model×3 × d_model），MPS 上减少 kernel launch 开销。这是 LLaMA/Mistral/Qwen 的标准做法。

### 1.5 masked_fill_ 而非加法 float 掩码

两种方案对比：
- **方案 A：加法 float 矩阵** -- 预计算上三角为 $-\infty$、下三角为 0 的 float32 矩阵，直接加到 attention scores 上
- **方案 B（选择）：`masked_fill_`** -- 预计算上三角 bool 矩阵，用 `scores.masked_fill_(mask[:S,:S], float('-inf'))` 原地写入

选择理由：
1. 语义清晰——代码意图一目了然"把这些位置填为 -inf"
2. 无临时张量——加法方案需广播产生临时 tensor；`masked_fill_` 是 in-place，零额外分配
3. 融合算子——PyTorch 底层走 fused kernel

**Decode 时传 None 跳过掩码**：decode 阶段 seq_len=1，单个 query 对所有 cached K 做注意力，所有位置都应被看到，无需掩码。

**为什么保留 `[:S, :S]` 切片**：训练时 S 恒等于 block_size，切片是 no-op。但 inference prefill 时 prompt 可能短于 block_size，需要 `[:S, :S]` 切出正确大小的掩码。

### 1.6 Pre-Norm + RMSNorm

**Pre-Norm vs Post-Norm**：
- Pre-Norm: `output = x + SubLayer(Norm(x))` -- 训练更稳定，不需小心 warmup
- Post-Norm: `output = Norm(x + SubLayer(x))` -- 原始 Transformer，对超参更敏感
- 结论：Pre-Norm，现代 LLM 无争议的选择

**RMSNorm vs LayerNorm**：
- LayerNorm: `y = (x - mean) / sqrt(var + eps) * gamma + beta` -- 两组参数
- RMSNorm: 去掉均值中心化，去掉 beta，只保留缩放

选择 RMSNorm 的理由：
1. 保留更多信息 -- LayerNorm 减去均值会丢弃"各维度整体偏移"这一信号
2. 参数更少 -- 无 beta
3. 计算量少 ~15% -- 省去求均值、减均值两步
4. 稳定性相当 -- 除以 RMS 已足够防止幅值爆炸
5. 行业验证 -- LLaMA/Mistral/Gemma 均使用

### 1.7 SwiGLU 而非标准 FFN

标准 FFN（2 个矩阵）：$\text{FFN}(\mathbf{x}) = W_2 \cdot \text{activation}(W_1 \cdot \mathbf{x})$

SwiGLU（3 个矩阵）多一个门控，实验效果更好。

`d_ff` 的由来：`≈ (2/3) × 4 × d_model`，使 SwiGLU（3 个矩阵）的总参数量与标准 FFN（2 个矩阵，d_ff = 4 × d_model）大致相等。

SiLU vs GELU：形状几乎一致，SiLU 实现更简单（`x * torch.sigmoid(x)`），LLaMA 使用 SiLU。

### 1.8 全部不用 Bias

LLaMA/Mistral 均省略 bias；参数量节省微小但避免了 bias 与 weight decay 交互的微妙 bug（bias 不应被 weight decay，分组管理繁琐）。

### 1.9 Weight Tying

输入 embedding 与输出 LM Head 共享权重。用 `F.linear(x, embedding.weight)` 实现，不额外分配参数。

### 1.10 初始化 scaled init 推导

W_O 和 W_down 是残差路径的输出矩阵。每个 block 有 2 个残差加法（注意力+FFN），N 层共 2N 次。若不缩放，残差流的方差会线性增长 2N 倍。除以 √(2N) 保持方差恒定。即 `Normal(0, 0.02/√(2N))`。

### 1.11 残差连接的含义

每个子层的输出不是替换 x，而是加到 x 上。x 本身直接"穿过"子层不受影响（即 residual），子层只需学习"增量修正"。好处：梯度可沿残差路径直达底层，训练更稳定。

---

## 二、分词器

### 2.1 BPE 原理

Byte-level BPE 是 GPT-2/LLaMA 等主流 LLM 使用的分词方式。

**基础词表**：256 个字节（0x00-0xFF），覆盖所有可能的输入，**永远不会有 UNK**。

**训练过程**（在语料上执行一次，结果落盘）：
1. 将文本编码为 UTF-8 字节序列
2. 统计所有相邻字节对（byte pair）的频次
3. 将最高频的 pair 合并为新 token，加入词表
4. 重复步骤 2-3，直到达到目标词表大小

**编码过程**（对任意新文本使用已训练的分词器）：
1. 将输入文本转为 UTF-8 字节序列
2. 按 merge 规则的学习顺序（优先级），依次合并可合并的 pair
3. 直到无法继续合并，得到 token ID 序列

**为什么编码需要顺序**：训练时每轮合并的 pair 按顺序记录（第 N 次合并 = 第 N 条规则）。编码新文本时，按这个顺序重放合并操作，保证同样的文本总是得到同样的 token 序列。merge rules 本质是训练过程的日志，不是人写的规则。

**中文示例**：
- "你" 的 UTF-8 = `[0xe4, 0xbd, 0xa0]`（3 字节）
- BPE 训练时高频字逐步合并：先合并前两字节为双字节 token，再合并该 token 与第三字节
- 极罕见的字可能仍是 2-3 个字节 token（fallback）

### 2.2 SentencePiece vs HuggingFace：选择 SPM

项目曾同时训练 HuggingFace ByteLevel BPE 和 SentencePiece BPE (byte_fallback) 进行对比，最终选择 SentencePiece。

**核心结论**：SPM 在同等 context window 下多装 ~4.3% 的内容。对小模型，训练时间差可忽略，token 效率直接决定模型质量。

#### 机制差异

| | HF ByteLevel BPE | SPM BPE (byte_fallback) |
|---|---|---|
| 起点 | 256 个原始字节 | 语料中高频字符（character_coverage=0.9995）+ 256 字节回退 |
| 中文字符 | 每个 UTF-8 中文 = 3 字节，需 merge 才能成字 | 高频字符直接作为 base token，无需 merge |
| Merge 预算 | 用于字级+词级合并 | 仅用于词级合并（字级免费） |
| 优先级 | 高频字节对优先 → 倾向合出长词 | 高频字符对优先 → 直接合出多字词 |

#### 关键差异：单字覆盖率

- **SPM**：~3,650 个中文单字 token，来自语料中覆盖 99.95% 字符频率的高频字符集（由 `character_coverage=0.9995` 决定），**不消耗 BPE merge 预算**
- **HF**：仅 ~2,022 个中文单字 token。每个单字需要 2~3 次 BPE merge（3 字节 UTF-8），消耗大量 merge 预算

缺失的单字意味着 SPM 能用 1 token 表示的字符，HF 需回退到 3 字节 UTF-8 编码（3 个 token），长尾内容尤其吃亏。

#### HF 的"跨字合并"现象

由于 merge 预算有限，HF 会跳过某些低频单字，直接合并出多字 token（如 "需要" 是 1 token，但 "需" 不在词表中 → "需求" 可能需 4 tokens）。这不是优化，而是预算不足时的妥协——同一字在不同组合中编码不一致，增加模型学习负担。

#### 选 SPM 的理由

1. **Token 效率**：同等语料少 ~4.3% tokens，context window 多装 ~5% 内容
2. **中文单字覆盖率**：3,650 vs 2,022，长尾内容编码更稳定
3. **表示一致性**：绝大多数汉字（覆盖 99.95% 出现频率）都有独立 token，同一字在不同上下文中 token 表示一致；极低频字走 byte fallback
4. **Byte fallback 更优雅**：SPM 的 `<0xXX>` 与字符级编码自然衔接

详细对比数据（BELLE 1M 语料实测）：SPM 训练 ~14min / HF 训练 ~7min；编码均 ~2min；SPM tokens 少 4.34%。

### 2.3 EOS 设计推理

EOS 不参与 BPE 训练/编码。它在 BPE 编码完成后以裸 token ID 形式插入，因此：
- EOS 不会出现在 merge rules 里
- EOS 不会与相邻 token 合并为更大的 token
- 推理时用户输入经 BPE 编码永远不会产生 EOS — 它只能由模型从 logits 中主动"选择输出"

### 2.4 vocab_size 8192 合并预算分析（中文）

以 HF ByteLevel BPE 为例（SPM 的字级 token 不消耗 merge 预算，预算更充裕）：

常用汉字 UTF-8 为 3 字节（byte1: 0xE4-0xE9，byte2/byte3: 0x80-0xBF）。BPE 组装单字需两轮合并：
- 第一轮（byte1+byte2 → 双字节 token）：**共享的**，~200 次合并覆盖 2500 常用字的所有前缀
- 第二轮（双字节 token+byte3 → 完整汉字 token）：每字独占一次，~2500 次

总计：字符组装 ~2700 次，剩余 ~5200 次用于词级合并。BPE 按频率驱动，高频字先组装，低频字可能保持 subword 拆分状态——这是正常行为。

英文 8192 偏富余（Shakespeare 4096 即可），中文合适。

### 2.5 标点符号

标点和普通文字一视同仁，无需特殊处理。ASCII 标点本身就是单字节 base token；中文标点是 3 字节 UTF-8，因出现频率极高，BPE 训练前几轮即合并为单 token。模型自然学会在恰当位置生成标点。

### 2.6 为什么存 .bin

BPE 训练（BELLE 1M 条语料，SPM ~14min）+ 编码（~2min）虽不是瞬间完成，但只需跑一次；缓存到 .bin 的好处：
- 训练脚本只需 `np.fromfile`，不依赖 tokenizer 库，职责解耦
- resume 训练时一致性保证（库版本变化不影响已编码数据）
- 代码最简——训练脚本不需要 import tokenizers、不需要知道 EOS 插入逻辑

---

## 三、工程设计

### 3.1 模块依赖

| 模块 | 依赖 | 层级 |
|------|------|------|
| prepare.py | corpus, sampler, dirs, tokenizer, config | CLI |
| train.py | config, sampler, model, dirs | CLI |
| chat.py | generate, model, dirs, tokenizer, config | CLI |
| corpus.py | tokenizer | 业务 |
| dirs.py | — | 业务 |
| generate.py | — | 业务 |
| sampler.py | — | 业务 |
| config.py | — | 基础 |
| tokenizer.py | — | 基础 |
| model.py | — | 基础 |

**依赖方向**：严格自上而下，下层不依赖上层，无循环依赖。

**vocab_size 数据流**：`tokenizer.model` 是 vocab_size 的唯一信息源。prepare 阶段通过
`--vocab-size` CLI 参数决定词表大小并训练 tokenizer；train.py 从 `vocab_config.json` 读
vocab_size（与 tokenizer 模块解耦），chat.py 从 `tokenizer.model` 加载后取 vocab_size。
`config.py` 不含 `vocab_size` 字段。


### 3.2 CLI 入口

三个独立脚本，各管一件事：

| 脚本 | 用途 |
|------|------|
| `prepare.py` | 数据准备：PT 语料编码（`prepare`）、SFT 语料编码（`prepare-sft`）、run 目录初始化（`init`）、数据预览（`preview`） |
| `train.py` | 训练循环：支持断点续训、SFT 热启动（仅加载权重）、sanity check |
| `chat.py` | 交互式对话：自动判定 PT/SFT 模式，SFT 套用 chat 模板 |

参数细节见 README。

### 3.3 数据流

```
原始数据 (.jsonl)
    │ prepare.py prepare-pt / prepare-sft
    ▼
<prepared_dir>/ (tokenizer.model + train.bin/val.bin + vocab_config.json [+ chat_format.json + *_mask.bin])
    │ prepare.py init --run <run_dir> --prepared <prepared_dir>
    ▼
<run_dir>/ (config.json + tokenizer.model + prepared → symlink + checkpoints/ [+ chat_format.json])
    │ train.py --run <run_dir>
    ▼
<run_dir>/best.pt
    │ chat.py --run <run_dir>
    ▼
交互式输出
```

所有路径参数接受相对路径或绝对路径，代码不做前缀推导。`prepared/` 和 `run/` 是惯例目录名，非强制。

**vocab_size 唯一信息源**：`tokenizer.model` 文件。prepare 阶段写入，train / chat 阶段读取。
`config.json` 不含 vocab_size，消除了配置与 tokenizer 不一致的可能。

### 3.4 元信息文件分工

| 文件 | 写入阶段 | 消费者 | 内容 |
|------|----------|--------|------|
| `vocab_config.json` | prepare | sampler（pad_id）、train.py（vocab_size） | 训练对 tokenizer 的最小依赖，解耦后无需加载 SPM |
| `chat_format.json` | prepare-sft | chat.py | 推理时 chat 模板契约（user_prefix / assistant_prefix / prepend_bos） |
| `corpus_manifest.json` | prepare | 人 | 数据来源、样本数、token 数等统计（纯记录，程序不消费） |

`vocab_config.json` 只含两个字段：`vocab_size`（建 embedding 层）和 `pad_token_id`（loss 时 ignore_index）。
bos/eos token ID 不在此文件中——它们仅在 prepare 阶段被消费（编码进 .bin），训练循环不需要知道。

### 3.5 Run 目录约定

- 每次训练使用一个独立 `run/<实验名>/` 目录
- `config.json` 是该训练目录的唯一配置真相（不是 config.py 的默认值）
- `current_run` 软链接指向当前活跃的训练目录，chat.py 默认使用它
- `run/`、`prepared/`、`raw/` 整体 gitignore，所有实验数据不进 Git
- config.py 只定义 schema、默认值与校验；schema 新增字段时，旧 run 的 config.json 需要补齐

### 3.6 Checkpoint 命名

| 文件 | 含义 |
|------|------|
| `best.pt` | 最佳 val_loss 模型（在 run 根目录） |
| `checkpoints/step_N.pt` | 定期保存 |
| `checkpoints/interrupted.pt` | Ctrl+C 中断保存 |
| `checkpoints/early_stop.pt` | early stopping 保存 |

Checkpoint 内容包括 model_state_dict、optimizer_state_dict、step、best_val_loss、rng_states。inv_freq 和因果掩码不存入 checkpoint（确定性生成，加载时重建）。

### 3.7 训练

**Pre-training (PT)**：全 token 计算 loss（PAD 除外）。采样方式为 random_slice。数据格式 jsonl_stream，每条 QA 后追加 EOS。数据使用 no-prefix 格式（instruction\nanswer），模型直接学习问答对应关系，无需格式标记。

### 3.8 推理设计

#### KV-Cache 原理

- 无 KV-Cache：每生成一个新 token 都要把整个已有序列重新过所有层。生成 200 token 需要 ~30 秒。
- 有 KV-Cache：缓存每层已计算的 K/V 向量，新 token 只需算自己的 Q/K/V，然后与缓存做注意力。生成 200 token 仅需 ~3 秒。

#### 滑动窗口 + RoPE

窗口大小 = max_seq_len，任意 query 到 cache 中最远 key 的相对距离始终在训练范围内。注意力计算数学上正确。

代价：模型"忘记"窗口外的早期内容，长文本会丢失开头信息。

RoPE position 持续递增（不取模），滑动窗口只影响 cache 中保留哪些 K/V，不影响位置编码的正确性。

有了滑动窗口，生成可以无限持续（直到 EOS 或用户设定上限），不受 max_seq_len 限制。

#### 流式输出

`generate()` 是 generator，支持 `stream=True` 逐 token 输出。chat.py 默认使用流式模式。

#### 解码策略

temperature + top-k + top-p + repetition penalty + n-gram blocking。SFT 模型由 chat.py 根据 `chat_format.json` 包装 prompt（`用户：{input}\n助手：\n`）；PT 模型不包装。

### 3.9 训练分析

#### FLOPs 计算

每个训练步骤（含前向+反向）：

$$\text{FLOPs/step} \approx 6 \times N \times B \times T$$

- 6 = 2（前向）+ 2（反向）+ 2（优化器），每个参数每个 token 的操作次数
- N = 参数量，B = batch_size，T = block_size

#### 过拟合

小模型 + 小语料 = 严重过参数化，过拟合是必然的。best.pt 保存 val_loss 最优时刻的权重即可。

#### 预期 Loss 轨迹

| Step | 预期 Loss | 模型学会了什么 |
|---|---|---|
| 0 | ~ln(vocab_size) | 随机预测 |
| 50 | ~7.0 | token 频率分布 |
| 200 | ~5.5 | 常见 bigram/trigram |
| 500 | ~4.5 | 短语和句式模式 |
| 1000 | ~3.8 | 语法结构和常见表达 |
| 2000 | ~3.2 | 较长距离连贯性 |
| 5000+ | ~2.5-3.0 | 收敛，能生成通顺文本 |

#### 优化器

AdamW，参数分组：衰减组（2D 权重含 Embedding，weight_decay=0.1）和不衰减组（RMSNorm gamma，weight_decay=0.0）。学习率调度：线性 warmup → 余弦退火至 min_lr。

#### 混合精度

CPU / MPS：float32。未来 CUDA：bfloat16 + autocast。
