---
title: "LocateAnything 深度解读：Parallel Box Decoding 如何让 VLM 检测速度翻倍、精度封顶"
published: 2026-06-25
description: "NVIDIA 提出的 LocateAnything 用 Parallel Box Decoding 将 VLM 检测从逐 token 串行解放为每步一个完整框的并行生成，吞吐量提升 2.5-10 倍的同时，高 IoU 定位精度全面领先 SOTA。本文从问题建模、注意力掩码数学、双序列联合训练、Stream Packing + MagiAttention 训练基础设施、KV Cache 管理、Hybrid 自适应推理、138M 数据引擎构造、四阶段训练策略到完整消融实验，逐层拆解全部技术细节。"
tags: [论文笔记, VLM, 目标检测, 并行解码, 多模态, 视觉定位, MTP, 注意力机制, 数据工程]
category: AI
draft: false
---

> **论文**: LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
>
> **作者**: Shihao Wang (PolyU), Shilong Liu (Princeton), Yuanguo Kuang, Xinyu Wei, Yangzhou Liu, Zhiqi Li, Yunze Man, Guo Chen, Andrew Tao, Guilin Liu, Jan Kautz, Lei Zhang, Zhiding Yu (NVIDIA)
>
> **链接**: [arXiv 2605.27365v2](https://arxiv.org/abs/2605.27365v2) | [GitHub](https://github.com/NVlabs/LocateAnything) | [HuggingFace](https://huggingface.co/nvidia/LocateAnything-3B) | [Demo](https://huggingface.co/spaces/nvidia/LocateAnything)

---

## 1. 问题的数学本质：为什么逐 token 生成坐标注定低效

### 1.1 VLM 做 Grounding 的两条路

VLM 的视觉定位可以统一为一个条件生成问题：给定图像 $\mathcal{I}$ 和文本查询 $\mathcal{E}$，输出一个定位序列。当前 SOTA 方法都沿袭 next-token prediction (NTP) 范式，但在坐标表示上走了两条不同的路。

**文本数字形式（Textual Digits）** — Qwen3-VL、DeepSeek-VL2、Cosmos-Reason2 等主流通用 VLM 的默认表示：

```
<ref>cat</ref> <box> <238> <419> <501> <703> </box>
```

每个数字字符独立成一个 token。一个 `238` 拆成 `"2"`、`"3"`、`"8"` 三个 token，模型需要逐一预测。全部 12+ 步才能出一个框。

**量化坐标形式（Quantized Coordinates）** — Rex-Omni 的表示：

```
<ref>cat</ref> <box> <x238> <y419> <x501> <y703> </box>
```

连续坐标 $[0, W] \times [0, H]$ 归一化到 $[0, 1000]$ 后离散化为 1000 个专用 token。每个坐标维度 1 步完成，但四个坐标仍需 4 步。

二者的共同本质是：**把二维几何量 $(x_1, y_1, x_2, y_2)$ 序列化为无结构的 1D token 流**。联合概率被因式分解为：

$$P(x_1, y_1, x_2, y_2 \mid \text{context}) = p(x_1) \cdot p(y_1 \mid x_1) \cdot p(x_2 \mid x_1,y_1) \cdot p(y_2 \mid x_1,y_1,x_2)$$

这种条件链条有三个深刻缺陷：

**延迟**：$N$ 个目标的场景需要 $4N$ 到 $12N$ 个解码步骤。每一步都依赖前一步的 KV cache 更新——GPU 的并行计算能力在自回归生成中完全被浪费。从论文 Figure 7 右侧可以看到：`Textual` 和 `Quantized` NTP 方法的生成时间随目标数量线性增长，300 个框时需要数秒。

**结构**：因果注意力天然断裂后续 token 对前序 token 的回顾路径。当模型已生成 `x1=238` 和 `y1=419` 后生成 `x2=501` 时，无法建立 `x2` 必须大于 `x1` 的约束——这个约束信息分散在前几步的输出中，无法被集中编码。论文 Table 1 的 IoU=0.95 指标暴露了这个问题：NTP 方法的高精度 F1 显著低于低 IoU 阈值下的表现，说明框的**四个角的相对位置关系不够精确**。

**监督信号扭曲**：交叉熵损失建立在 token 级别，而检测质量的度量建立在 IoU 级别——这是一个**度量错配**。当 `x1=238` 恰好预测正确但 `x2` 偏移 50 像素时，损失可能很小（只有一个 token 不对），但 IoU 会大幅下降。逐 token 损失无法反映框的整体几何质量。

### 1.2 标准 MTP 为什么没有解决问题

Multi-Token Prediction (MTP) 是并行解码领域最自然的思路：一次预测多个未来 token，从而减少解码步数。LLM 领域的典型做法包括：

- **固定块切分**（SDLM, Dream）：把序列按固定长度 B (如 B=6) 切成块，每步预测一个块内的所有 token。块内双向注意，块间因果。但块边界是**随机的**——可能恰好把一个框的 `x2` 和下一个框的 `<ref>` 切到同一个块中。论文 Figure 2 的中间一列展示了这种切分导致的"非相干"模式：坐标 token 的分布错乱，语义边界被打破。

- **随机掩码重建**（Block Diffusion, LLaDA）：在任意位置 mask 掉若干 token，训练模型从剩余 token 重建缺失部分。这种方式天然学习各种不规则的条件分布——一个框的四个坐标可能在 mask 中被分散到完全不同的位置。

这些方法的共性问题是 **structure-agnostic**：MTP/Diffusion 框架将所有 token 视为无差别的符号流。对于 "the" / "a" / "cat" 这类自然语言 token，相邻 token 之间的关系主要是共现性的；但对于 `(x1, y1, x2, y2)`，它们之间存在**几何必然性**——$x_2 > x_1$、$y_2 > y_1$，且它们共同定义了一个矩形区域的形状和位置。松散的结构化已经足以约束自然语言（"the cat" 是一个有效的 bigram），但对坐标而言不够：模型必须学习到不同级别的一致程度。

论文在这个点上打了一个关键论证：**标准 MTP 的监督信号是错误的**。固定大小块迫使模型学习"将 `<box>` 和 `</ref>` 放在同一个预测组"这种无意义的模式。由于块边界与框边界不对齐，MTP 学习到的是一组**跨几何边界的虚假关联**，等效于向训练中注入了噪声。消融实验在 Table 6(b) 中给出了最直接的证据：SDLM-B6（块大小 6）在 COCO 上只拿到 46.1 F1，吞吐量 5.5 BPS；而 LocateAnything 的 PBD (Fast) 在同样的块大小下拿到 49.6 F1，吞吐量 16.9 BPS——**对齐框边界带来了 +3.5 F1 的提升和 3 倍的加速**。

---

## 2. Parallel Box Decoding：核心思路与格式定义

### 2.1 基本洞察

LocateAnything 的核心洞察是一种**几何本体的回归**：一个边界框 $(x_1, y_1, x_2, y_2)$ 不是四个独立的标量，而是一个**不可分割的原子几何单元**。四个坐标共同定义了一个二维矩形——分开来预测，就失去了"矩形"应当承载的约束力。

PBD 的策略：**让 MTP 预测块的边界严格与几何单元边界对齐**。每个预测块恰好是一个完整的框（及其语义前缀），块内 6 个 token 作为一个不可分割的原子单位被同时生成。

### 2.2 块格式定义

输出序列被重组为固定长度 $L=6$ 的原子块序列 $B = (b_1, b_2, \ldots, b_N)$。连续坐标归一化到 $[0, 1000]$ 后离散化为整数量化 token。条件概率重新因式分解为块级形式：

$$P(B \mid Z, \mathcal{E}) = \prod_{i=1}^{N} P(b_i \mid b_{<i}, Z, \mathcal{E})$$

四种功能块类型：

| 类型 | Token 组成 (L=6) | 职责 |
|------|------------------|------|
| **Semantic Block** | `<ref> [word_1] [word_2] [word_3] </ref> <null>` | 编码目标语言描述。若描述超出单块容量，切分到多个连续 Semantic Block |
| **Box Block** | `<box> <x1> <y1> <x2> <y2> </box>` | 编码一个完整检测框。四个坐标在一个块内，块内全连接注意力捕获几何约束 |
| **Negative Block** | `<ref> [query] </ref> <neg> <null> <null> <null>` | 显式声明"此图中不存在该类别"，抑制幻觉检测 |
| **End Block** | `<eos> <null> <null> <null> <null> <null>` | 终止生成过程，作为结束标记 |

几个关键的格式设计决策：

**(1)** Box Block 的四个坐标**一定在一个完整的块内**，永不被跨块切割——这是 PBD 与 SDLM/Block Diffusion 最本质的区别。论文在 Figure 7 (left) 的消融实验验证了四种排序策略的影响：X-Y Corner Order（先左上角 x 再 y → 先右下角 x 再 y）最优，优于按面积排序、按中心距离排序、随机排序。X-Y Corner Order 编码了自然的**水平-垂直扫描顺序**，方便模型建立方向性几何先验。

**(2)** 未占满的位置用 `<null>` token 填充。这保证了所有块的维度恒定 ($L=6$)，方便做高效的批量并行张量运算，无需动态处理变长块。

**(3)** Negative Block 是一个**被大幅低估的设计选择**。论文在 Table 10 中报告了数据集中约 22M 个负样本，分布在检测和指代理解两个域中。没有这些负样本，模型在"图中不存在该类"时倾向于幻觉出框——VLM 做检测的典型问题之一。

**(4)** End Block 的 `<eos>` token 承担了传统 NTP 中 `<eos>` 的角色。区别是：在 PBD 中，`<eos>` 和后面的 5 个 `<null>` 在同一个**并行预测步骤**中一起产生——模型一次性生成 6 个 token 并宣告结束。

---

## 3. 双序列联合训练：为什么「一份标注，两种格式，一个模型」是关键

### 3.1 不能只训练 MTP 的原因

最直接的训练策略是：把训练数据全量转成 PBD 块格式，块内 mask 掉后 5 个 token（只保留第一个 token 作为锚点），只算 MTP 损失。这看似简单直接，但消融实验暴露了严重问题：**只训练 MTP 损失时，Fast 模式 F1 从 49.6 跌到 47.2**（Table 6c 第三行）。原因在于：

模型在预训练阶段建立的是严格的因果推理链——每个 token 只参与预测下一个 token。纯 MTP 训练打破了这一逻辑，要求模型"跳过中间环节直接看到 6 步之后的未来"，而这种**跳跃式因果**与原始因果注意力不兼容。这同时损害了 NTP（需要因果自信）和 MTP（需要跳跃式自信）的推理能力。

### 3.2 双序列构造

LocateAnything 的训练方案是**在同一前向传播中同时送入两份格式的 ground truth**：

```
x_all = [ x_vis | x_q | x_ntp | x_blk ]
```

其中：
- $x_{vis} \oplus x_q$：共享上下文（视觉编码器输出 + 文本查询）
- $x_{ntp}$：标准 NTP 序列——完整的 token 级 ground truth
- $x_{blk}$：MTP 块序列——与 $x_{ntp}$ 表达同一份内容，但按 PBD 规则重组为块格式

以一个具体例子说明。假设 ground truth 是检测到一个 "hot dog"，框为 `(663, 638, 313, 345)`：

```
x_ntp:
<ref> hot dog </ref> <box> <663> <638> <313> <345> </box> <eos>

x_blk:
Block 1: <ref>  [mask] [mask] [mask] [mask] [mask]
Block 2: <box>  [mask] [mask] [mask] [mask] [mask]
Block 3: <eos>  [mask] [mask] [mask] [mask] [mask]
```

$x_{blk}$ 的构造过程：从左到右遍历 $x_{ntp}$，按块规则切分并填充。每个块内**保留第一个 token 作为锚点**（告诉模型"这个块是什么类型——Semantic/Box/End"），其余 5 个位置替换为 `[mask]`。锚点 token 决定了模型预测后续 token 时的目标语义——当模型看到 `Block 2` 的锚点是 `<box>` 时，它知道接下来的 5 个 token 应该是四个坐标加 `</box>`。

### 3.3 三种注意力规则的统一掩码

这是 LocateAnything 训练设计中最精妙的部分。一个统一的注意力掩码同时实现了三种完全不同语义的注意力行为：

**(A) 因果注意力（NTP 区域）**

共享上下文 $(x_{vis}, x_q)$ 和 NTP 序列 $x_{ntp}$ 整体使用标准的下三角因果掩码：

$$\text{Attn}(i, j) = \begin{cases} 1 & j \leq i \\ 0 & j > i \end{cases}, \quad i \in (x_{vis}, x_q, x_{ntp})$$

与原始 VLM 训练中的前向行为完全一致，保证了 NTP 流的因果推理链不受干扰。NTP 区域是**禁止 attend 到 MTP 区域**的——这是为了防止数据泄漏：如果 NTP 流能看到未来 token（即使在 MTP 掩码中），因果性就被破坏了。

**(B) 块间因果注意力（MTP 块与块之间）**

不同 MTP 块之间执行严格的因果可见性：

$$\text{Attn}(b_k, b_j) = 0 \quad \text{for} \quad k < j$$

Block $k$ 只能 attend 到 Block $1 \ldots k$、共享上下文、以及整个 NTP 序列。这与实际推理时的生成顺序完全一致——在推理时，模型基于之前所有已提交的块来预测当前块。

**(C) 块内全连接双向注意力（MTP 块内部）**

同一个块内的 6 个 token 之间**完全双向可见**：

$$\text{Attn}(i, j) = 1 \quad \forall i, j \in b_k$$

为什么这是 PBD 的灵魂？因为在预测块 $b_k$ 的第一个 `[mask]` token（比如一个 x1 坐标）时，模型可以同时参考这个块内的其他 `[mask]` token。这允许模型学习到联合推理空间关系的能力：x2 应该大于 x1，y2 也应该大于 y1——这些约束在双向注意力中被**同时编码**。具体来说，假设真实的框是 `<box> <200> <300> <500> <600> </box>`。在块内双向注意力下，模型学习的是：

$$P(x_2=500 \mid \ldots, x_1=200, y_1=300, \text{双向上下文})$$

而不是：

$$P(x_2=500 \mid \ldots, x_1=200, y_1=300, \text{仅有前向上下文})$$

这个差异至关重要——双向注意力的联合分布可以编码 $(x_1, y_1, x_2, y_2)$ 之间的全部空间约束，而因果注意力的条件分布每步只能编码已生成 token 对未来的约束。论文 Table 6(a) 的消融实验间接验证了这一点：PBD (Slow Mode, 纯 NTP) 的 F1=52.1 优于量化坐标 (50.1)，而 PBD (Fast, 纯 MTP) 就达到了 49.6——这说明即使是 NTP 范式下，块对齐的**训练监督**（而不是块对齐的推理）就已经带来了 +2.0 F1 的增益。这是训练和推理的叠加受益：块对齐的训练监督提升了 NTP 的结构化理解；而 MTP 推理则让这块能力以并行的速度释放。

**(D) 隔离约束的严谨性**

MTP 与 NTP 区域之间是**单向隔离**的：NTP 区域完全禁止 attend 到 MTP 区域 —— "已经看到标准答案" 的因果流不能知道 "掩码预测练习" 的内容。MTP 区域可以 attend 到 NTP 区域 —— 在预测掩码 token 时可以参考完整的因果答案链作为上下文辅助。NTP 流服务于因果推理保护，MTP 流服务于块级并行能力构建，二者通过掩码设计既互不污染又互相增益。

### 3.4 损失函数

$$\mathcal{L} = \mathcal{L}_{ntp} + \mathcal{L}_{mtp}$$

两个损失都是标准的逐 token 交叉熵，权重相等（1:1），直接相加。$\mathcal{L}_{ntp}$ 只在 NTP 序列上计算；$\mathcal{L}_{mtp}$ 只在 MTP 块序列的 `[mask]` token 上计算。

消融实验 (Table 6c) 对比了联合损失和单独损失的差异：

```
NTP only:           Slow F1=50.1  (基线)
MTP only:           Fast F1=47.2  (受损)
NTP + MTP (joint):  Slow F1=52.1, Fast F1=49.6, Hybrid F1=51.6
```

联合训练的结果揭示了一个反直觉现象：**加 MTP 损失不仅不损害 NTP，反而显著提升了 NTP 的定位精度**（Slow F1 从 50.1 到 52.1）。论文的解释是：MTP 流中学到的块内空间联合分布（通过双向注意力）反向传递到 NTP 流的因果推理中，强化了模型对坐标结构的整体理解。

---

## 4. 训练基础设施：Stream Packing + MagiAttention

### 4.1 挑战：变长序列与异构掩码

双序列联合训练带来了两个工程难题：

1. **序列长度不均**：不同样本经块格式化后序列长度差异极大。纯填充式 batching 浪费大量 GPU 显存。
2. **异构注意力掩码**：同一批次内需要混合因果、块因果、全连接三种掩码模式。标准的 Flash-Attention 内核假设统一掩码模式，无法直接处理。

### 4.2 Stream Packing：让 H100 吃饱

Nova 团队实现了一个**在线流式打包器**，将多条变长样本动态拼接成一条紧致的目标长度序列（如 36,864 token）：

**加权采样**：一个无限迭代器按各数据集预先设定的混合权重抽取样本。这一步保证了每个训练 step 的数据分布符合预设比例（例如 OpenImages 比 RefCOCO 需要更高的采样率）。

**最优拟合缓冲区**：固定大小缓冲区（默认 32）暂存待打包样本。打包一个 batch 时，先在缓冲区中扫描出**当前能填入剩余 token 预算的最大样本**（best-fit decreasing 启发式），实现 > 95% 的打包效率。如果缓冲区中所有样本都装不下，从迭代器取新样本——装得下则填入，装不下则放回缓冲区。

**大石块优先播种**：batch 完成后，下一个 batch 用缓冲区中最大的样本"播种"，确保超大样本不会被长期"饿死"。

每条打包序列附带一个 `sub_sample_lengths` 张量记录各组成样本的边界。这个元数据被下游的 MagiAttention 消费，用于构造**每个样本独立的、不互相干扰**的注意力掩码——即使它们在打包序列中物理连续，逻辑上完全隔离。

### 4.3 MagiAttention：训练规模的秘密武器

[MagiAttention](https://github.com/SandAI-org/MagiAttention/) 是一个专为超长上下文 + 异构掩码训练的分布式注意力框架。在 LocateAnything 的场景中，它的核心价值是：

- **原生支持异构掩码**：可以为每个样本指定不同的掩码模式，同时在一批内处理
- **线性扩展**：在多 GPU 分布式训练中，注意力计算的开销与 GPU 数量线性缩放
- **与 Stream Packing 协同**：Packing 最大化 token 利用率，MagiAttention 保证异构掩码处理正确高效

这两个基础设施的组合是 LocateAnything 能训练 138M 样本的工程基础。

---

## 5. 推理机制：三种模式与 KV Cache 管理

### 5.1 三种解码模式

| 模式 | 解码方式 | 吞吐量 (BPS) | F1 (COCO) | 原理 |
|------|---------|-------------|-----------|------|
| **Slow** | 逐 token 自回归 (NTP) | 3.9 | **52.1** | 保留因果推理最强能力 |
| **Fast** | 每步并行预测完整 6-token 块 (MTP) | **16.9** | 49.6 | 块并行最大化吞吐 |
| **Hybrid** | 默认 Fast，异常时回退 Slow | 12.7 | **51.6** | 平衡速度与精度 |

### 5.2 Fast 模式的逐步推演

```
Step 0 (prefill): [ x_vis | x_q ]
    → 视觉编码 → 共享上下文 KV cache 建立

Step 1 (decode): [ x_vis | x_q + KV_0 ]
    → 并行预测 6-token →
        预测: <ref> hot  dog  </ref> <null> <null>
        提交: <ref> hot  dog  </ref> (4 个真实 token)
        丢置: <null> <null> (padding)

Step 2 (decode): [ KV_0 | committed tokens ]
    → 并行预测 6-token →
        预测: <box> <663> <638> <313> <345> </box>
        提交: 全部 6 个 token (Box Block 恰好占满 L=6)
```

每步并行预测一个完整的 6-token 块，然后**只把真实 token**（非 `<null>` padding）提交给 KV cache。论文 Table 12 给出了三种模式在全部评测任务上的完整对比，这里有几个关键数字——Fast 在 GUI grounding 上 59.7 → Hybrid 60.3 → Slow 60.5（差距仅 0.8 分），但在 Dense200 上 Fast 46.8 → Hybrid 61.3 → Slow 61.5（差距 14.7 分），VisDrone 上 Fast 34.4 → Hybrid 39.8 → Slow 40.2。这说明**密集小物体场景是并行解码的主要弱点**——多个物体密集排布时，边界模糊，MTP 更容易产生空间错位。

### 5.3 KV Cache 管理的关键细节

每个 MTP 步骤完成后，KV cache 必须做一次**截断清理**：

```
保留: 共享上下文 (x_vis, x_q) + NTP 前缀 + 已提交的真实 token
删除: [mask] token + 重复的锚点 token
```

这步操作看似常规，但有关键语义：[mask] token 在训练阶段只存在于 MTP 块内，推理阶段如果不删除它们的历史 KV 条目，后续步骤的注意力分布就会出现**训练-推理不一致**——模型会尝试 attend 到这些从未在训练中作为"已提交历史"出现过的 token。

### 5.4 Hybrid 模式的故障检测与局部回退

并行解码并非完美无缺。论文识别了两种典型失败模式：

**格式不规则（Format Irregularity）**

当场景中包含大量跨类别的实例时，模型在类别边界处会产生犹豫——它不确定应该继续预测当前类别还是切换到新类别。这表现为单块内的 token 组合违反语法：如 `<box> <211> </ref> <911> <887> </box>`——`</ref>` 不应该出现在 Box Block 中。

检测方法：验证块内 token 是否符合四类块的语法模式（在论文中是程序化检查块结构，如 Box Block 必须满足 `token[0]=<box>` 且 `token[5]=</box>`）。

**空间模糊（Spatial Ambiguity）**

当物体在规则网格中紧密排列时（如货架商品、钢琴键），MTP 可能输出空间上"卡在中间"的坐标——处于两个物体的交界处，导致低 IoU 预测。

检测条件：
1. **Top-1 坐标 token 概率 < 0.7** — 模型对坐标位置本身不自信
2. **Top-5 坐标 token 在 [0, 1000] 归一化空间内 max-min 差 > 80** — 即 top-5 备选项分布跨度超过半个物体的宽度，意味着确实存在多个"接近但不同的"候选位置

两个条件**必须同时满足**才触发回退：只有低概率没有空间发散可能是单一位置的不确定性（用阈值即可），只有空间发散但没有低概率说明 top-5 确实宽幅但模型自己很确定（通常是因为场景本身高度模糊）。

**回退流程**：

1. 丢弃当前不可靠块
2. 回退 KV cache 到上一已验证前缀
3. 在该前缀基础上用 NTP 逐 token 自回归补全该块（最多 6 个 token）
4. 补完后**无缝切回 MTP** 继续后续块的并行预测

回退比例不高（论文指出"Hybrid 保存了大部分快速模式的速度增益"），因此 Hybrid 能以接近 Fast 的速度 (12.7 vs 16.9 BPS) 运行，而精度接近 Slow（51.6 vs 52.1 F1）。

### 5.5 Backbone 泛化实验

论文在 Table 13 中做了**跨 Backbone 验证**：将 PBD 解码设计实例化在 Qwen3-VL-4B 上，只在 COCO 上训练（隔离大规模数据的影响）：

| Backbone | Decoding | F1 | BPS |
|----------|----------|-----|-----|
| Qwen3-VL-4B | NTP (baseline) | 50.8 | 2.8 |
| Qwen3-VL-4B + PBD | Slow | 52.2 | 2.8 |
| Qwen3-VL-4B + PBD | Fast | 49.6 | 11.4 |
| Qwen3-VL-4B + PBD | Hybrid | 52.0 | 9.4 |

这个实验的重要性在于它**排除了"好成绩来自大模型"的怀疑**：PBD 在 Qwen3-VL-4B 上同样有效（Hybrid +1.2 F1, 9.4 BPS vs NTP 的 2.8 BPS）。这不是"某一个特定预训练模型的特性"，而是 PBD 方法本身的泛化能力。

---

## 6. 数据引擎：138M 查询的构造流水线

### 6.1 数据全景

```
12,000,000+  独立图片
138,000,000+ 自然语言查询
785,000,000+ 标注框
```

这组数字的分布如下（来自 Table 10）：

| 任务域 | 查询数 | 负样本 | 目标数/Q (Max) | 平均字数/Q |
|--------|--------|--------|----------------|-----------|
| Detection | 93.4M | 21.0M | 745 | 24.2 |
| GUI | 23.0M | 0 | 14 | 4.1 |
| Referring | 10.1M | 93K | 818 | 5.5 |
| OCR | 5.1M | 0 | 2,337 | 1.2 |
| Layout | 4.9M | 1.4M | 176 | 2.2 |
| Pointing | 3.1M | 353K | 675 | 2.6 |

两个关键观察：
- 检测数据 (93M) 主导了训练（66.9%），提供了最大量的框级监督信号（83.1% 的标注框来自检测域）
- 负样本分布不均：检测有 21M 负样本，指代有 93K，GUI 完全无负样本。这反映了论文对不同域的"幻觉风险"有差异化的认知——开放域 detection 最易产生幻觉，closed GUI 最少。

### 6.2 多目标接地数据引擎

这是论文**最具工程价值**的部分。对于占 66.9% 的检测数据，直接使用原始类别标签（如 "person"、"car"）作为查询过于单一——无法覆盖属性描述（"红色的车"）、空间描述（"左边的狗"）、推理描述（"正在跑步的人"）等复杂语义。

论文设计了一个三级数据流水线（Figure 9）：

**从检测标注框出发**：

```
Step 1: Qwen3-VL 读取 (全图 + GT框 + 类别标签)
         → 生成多角度自然语言查询
           角度1: 外观属性 ("the red sports car")
           角度2: 空间关系 ("the car to the left of the traffic light")
           角度3: 推理线索 ("the vehicle that appears to be turning")
           角度4: 部件描述 ("the left front wheel of the car")

Step 2: 用生成的查询 prompt Molmo
         → 预测候选定位点 (points)

Step 3: 过滤: 保留落在对应 GT 框内的点
         → 可靠的定位监督信号
```

**从无标注图片出发**（进一步扩展多样性）：

```
Step 1: Qwen3-VL 直接从图片生成查询
         → 描述图片中可能存在的物体/区域

Step 2a: Molmo 预测点 → SAM 3 将点转为框
Step 2b: Rex-Omni 直接预测框

Step 3: Qwen3-VL 后验证
         → 检查框内容是否与查询一致
         → 滤掉不一致的预测
```

这个流水线的关键创新在于**两步验证**：Molmo/SAM 3/Rex-Omni 生成初始定位后，Qwen3-VL 作为验证器做最终筛选。这种"生成-验证"范式让非人类标注也能达到可用的质量。

### 6.3 GroundCUA 的数据增强

GroundCUA 数据集（UI 元素接地）有一个特殊处理环节。因为其原始标注是简短的 UI 元素描述（"submit button"、"search field"），论文使用了专门的**VLM 描述生成流程**：

- 在截图上渲染目标框 → 裁剪局部区域
- 将全图 + 裁剪区域 + 标签 + 类别 + 平台（Web/Desktop/Mobile）喂给 Qwen3-VL
- Qwen3-VL 从视觉外观、空间位置、功能语义三个互补角度生成描述

例如，原本的 `"submit button"` 被丰富为三条查询：
- 外观：`"the blue rectangular button with rounded corners and white text 'Submit'"`
- 空间：`"the button at the bottom right corner of the form"`
- 功能：`"the button you click to submit the form"`

这种增强将离散标签空间转换为连续的自然语言分布，同时保留了原始的 bbox ground truth 作为可靠监督。

### 6.4 框排序策略

论文在 Figure 7 (left) 中消融了四种框排序策略对 F1-score 的影响：

- **X-Y Corner Order**（默认）：先按左上角 x 坐标升序，同值按 y 坐标升序 → **最优 F1**
- Center Distance：按框中心到原点 (0,0) 的距离排序
- Area：按框面积从大到小排序
- Random：随机打乱

X-Y Corner Order 的优胜是**符合直觉的**：人类从左到右、从上到下的扫描顺序与空间布局的物理结构一致。模型学习具有固定顺序的框之间的空间转移规律（"前一个框在左上方 → 下一个框在其右侧"），形成连贯的空间预测链。

---

## 7. 训练策略：四阶段逐步注入

论文设计了四阶段渐进训练管线（Table 7）：

| 阶段 | 目标 | 数据 | 学习率 | Max Seq Len | 可训练组件 |
|------|------|------|--------|-------------|-----------|
| Stage 1 | 视觉概念初始化 | Caption 数据 | 2×10⁻⁴ | 32,768 | MLP only |
| Stage 2 | 全面多模态学习 | 全量通用数据 (不含检测/定位) | 4×10⁻⁵ | 32,768 | All |
| Stage 3 | 检测 & 定位注入 | 全量 138M 查询 | 4×10⁻⁵ | 25,600 | All |
| Stage 4 | 密集场景精锐化 | 20% 通用 + 高密度数据 (MOT20Det, SKU110K) | 1×10⁻⁵ | 25,600 | All |

**为什么 Stage 1 只训练 MLP？** 因为视觉编码器 (Moon-ViT) 和语言模型 (Qwen2.5) 都已经在各自的大规模预训练中获得了强特征表示能力。Stage 1 只训练 MLP Projector，目的是**先让视觉和语言表示对齐**，再在 Stage 2 打开所有参数做全面微调。这避免了在不对齐的早期状态下，大模型参数被无意义的梯度扰动。

**Stage 4 的"20% 通用 + 高密度"数据配比**值得特别关注。论文在 Stage 3 注入全量 138M 查询后，发现模型在**密集场景中的召回率仍有提升空间**——因为 138M 数据中以中等密度场景为主（大多数图片中物体数 < 30）。Stage 4 大幅增加高密度数据（MOT20Det 的密集人群、SKU110K 的货架商品）比例，同时保留 20% 通用数据防止灾难性遗忘。

### 7.1 基础 VLM 的预训练数据

Stage 1-2 的基础 VLM 预训练数据（Table 8）覆盖了**远超检测/定位领域**的范围：

- Captioning & Knowledge（15+ 数据集）
- Mathematics & Code（14+ 数据集）
- Science（8+ 数据集）
- Chart & Table（17+ 数据集）
- Naive OCR（34+ 数据集）
- OCR QA（24+ 数据集）
- General VQA（32+ 数据集）
- Text-only（14+ 数据集）
- Counting（2 数据集）

这个广度解释了 LocateAnything 在 NLP 理解和多模态推理任务上的泛化能力——模型在学检测之前，已经建立了坚实的对话、知识、推理基础。

---

## 8. 实验结果深度分析

### 8.1 通用检测 (COCO/LVIS) — Table 1

| 方法 | 参数量 | LVIS F1@0.5 | LVIS F1@0.95 | LVIS F1 Mean | COCO F1 Mean |
|------|--------|------------|-------------|-------------|-------------|
| Qwen3-VL-8B | 8B | 61.5 | 20.2 | 44.8 | 45.7 |
| SEED1.5-VL | - | 65.6 | 19.5 | 46.7 | 51.4 |
| Rex-Omni-3B | 3B | 64.3 | 20.7 | 46.9 | 52.9 |
| **LocateAnything-3B** | 3B | 62.3 | **31.1** | **50.7** | **54.7** |

最重要的一列是 **LVIS F1@0.95 = 31.1**（Rex-Omni=20.7，Qwen3-VL-8B=20.2）。为什么这比 F1@0.5 更具决定性？因为：

- F1@0.5 衡量"大致能定位"——框与 GT 有多半重合就行
- F1@0.95 衡量"精确框边界"——框的边缘必须在 GT 的 5% 容差内
- 逐 token NTP 因坐标间的结构耦合被破坏，高 IoU 精度天然受损。PBD 的单步同步预测修复了这种结构耦合，四个坐标之间的相对位置关系被联合优化

这种精度提升不是参数量的胜利——LocateAnything-3B 在 LVIS F1@0.95 上**碾压了 Qwen3-VL-8B (31.1 vs 20.2)**——说到底是方法论的胜利，而不是规模的胜利。

### 8.2 密集检测 (Dense200/VisDrone) — Table 2

两个密集检测基准呈现出有趣的差异：

- Dense200：LocateAnything 58.7 vs Rex-Omni 58.3（差距仅 +0.4）
- VisDrone：LocateAnything 39.9 vs Rex-Omni 35.8（差距 +4.1）

这两个数据集的本质区别是什么？Dense200 是通用密集场景（多为 COCO 类别的密集变体），而 VisDrone 是**无人机俯拍视角**的极小物体密集检测。LocateAnything 在 VisDrone 的 F1@0.95 上达到 3.2（Rex-Omni 仅 1.5），说明 PBD 在面对**极小 (< 20×20 像素)** 密集物体的框边界划分上确实比 NTP 有优势——即使框很小，一次性同步预测四角的相对约束仍然有效。

### 8.3 GUI 接地 (ScreenSpot-Pro) — Table 3

这是 LocateAnything 最令人印象深刻的实验：**3B 的模型在 ScreenSpot-Pro 上达到 SOTA（60.3），超越了包括 32B GUI-Owl 和 7B UI 专门模型在内的所有系统**。

按元素类型分解（Text vs Icon）尤其揭示问题：

```
LocateAnything-3B:  Text Avg=65.4  Icon Avg=43.8
GUI-Owl-32B:        Text Avg=70.1  Icon Avg=36.0
UI-Venus-1.5-2B:    Text Avg=73.8  Icon Avg=51.7
GTA1-7B:            Text Avg=48.6  Icon Avg=25.9
```

在 Icon 元素的接地准确率上，3B 的 LocateAnything (43.8) 超过了 7B 的 GTA1 (25.9) 和 Rex-Omni-3B（Text 61.7 但 Icon 仅 9.7）。GTA1 有专门的 UI 理解模块，但它的 Text 48.6 表现明显弱于很多开放域 VLM，而 Icon 也不如 LocateAnything——说明它的训练规模和多样性不如 138M 数据引擎的贡献。

Icon 接地难的根源在于：按钮、图标通常很小（< 30×30 像素），且通常密集排列（工具栏一行 10+ 图标），对空间精确度要求极高。PBD 的块内全连接允许多个相邻 Icon 之间的空间关系被同步编码，减少了相邻物体边界混淆。

按场景领域分解：

| 领域 | LocateAnything | GUI-Owl-32B | 优势 |
|------|---------------|-------------|------|
| Dev | 70.8 / 50.3 (T/I) | 84.4 / 39.3 | Text 较低但 Icon 大幅领先 |
| Creative | 60.1 / 46.9 | 65.2 / 18.2 | Icon 是 2.6 倍 |
| CAD | 57.9 / 40.6 | 62.4 / 28.1 | CAD 密集元素中 Icon 一直占优 |
| Sci. | 69.4 / 58.2 | 82.6 / 39.1 | Science 中含公式/图表元素 |
| Office | 77.2 / 69.8 | 81.4 / 39.6 | Office 中 Icon 识别几乎是 1.8 倍 |
| OS | 65.4 / 43.8 | 70.1 / 36.0 | 跨 OS 一致性更好 |

### 8.4 吞吐量全量对比

直观的加速对比 (Table 1, Figure 7)：

```
Qwen3-VL-4B  (Textual):    1.1 BPS
Qwen3-VL-8B  (Textual):    1.0 BPS
Rex-Omni-3B  (Quantized):  5.0 BPS
LocateAnything (Hybrid):  12.7 BPS  — 2.5× vs Rex-Omni, 12.7× vs Qwen3-VL
LocateAnything (Fast):    16.9 BPS  — 3.4× vs Rex-Omni, 16.9× vs Qwen3-VL
```

Figure 7 (right) 展示了**不同目标数量下的吞吐量变化**：
- NTP 方法（Textual, Quantized）：随着目标数从 20 增长到 300，生成时间**线性增长**（与目标数成正比），吞吐量基本不变（约 1-5 BPS）
- PBD (Parallel)：生成时间增长极微，在 300 个目标场景中吞吐量飙升至约 **25 BPS**

这是因为 NTP 的解码步数 = 框数 × 每框步数，而 PBD 的解码步数 = 框数 × 1（每框一块）。在多目标密集场景中节省的不是常数倍，而是**阶数差**——NTP 复杂度 $O(N \times S)$（$S$ 为每框 token 数，4-12），PBD 复杂度 $O(N)$。

### 8.5 消融实验完整解读 — Table 6

**(a) 坐标表示消融**

| 表示方式 | 吞吐量 (BPS) | F1 |
|----------|------------|-----|
| Textual | 1.3 | 49.1 |
| Quantized | 3.9 | 50.1 |
| PBD (Slow) | 3.9 | **52.1** |
| PBD (Fast) | 16.9 | 49.6 |
| PBD (Hybrid) | 13.2 | 51.6 |

关键逻辑链：
1. Quantized 优于 Textual（+1.0, +2.6× BPS）— 减少 token 数降低了累积误差
2. PBD Slow 优于 Quantized（+2.0, 同 BPS）— **块对齐训练监督的纯粹效果**（推理方式相同，都是 NTP）
3. PBD Fast (MTP) 的吞吐量是 Slow (NTP) 的 4.3 倍 — **块对齐推理的纯粹加速效果**
4. PBD Hybrid = 98.8% 的 Slow 精度 + 3.4 倍的速度 — **生产级平衡点**

**(b) MTP 块对齐消融**

| MTP 方案 | BPS | F1 |
|----------|-----|-----|
| SDLM-B4 (随机切分, L=4) | 5.2 | 46.5 |
| SDLM-B6 (随机切分, L=6) | 5.5 | 46.1 |
| SDLM-B8 (随机切分, L=8) | 6.7 | 45.8 |
| Block Diffusion-B6 | 4.7 | 44.8 |
| PBD (Fast, 对齐框边界, L=6) | **16.9** | **49.6** |

这里有一个耐人寻味的模式：SDLM 的块大小从 B4→B6→B8 时，吞吐量从 5.2→5.5→6.7（线性增长），**但 F1 却在持续下降**（46.5→46.1→45.8）。为什么块越大，精度反而越差？

因为更大的块意味着每个块内包含**更高概率的跨几何边界 token 组合**。B=8 的随机切分可能在一个块中同时包含 `</box> <ref> person </ref> <box> <x`——这是三个不同几何单元的混合。模型被要求学习"预测 `</ref>` 之后跟 `<box>` 之后又跟 `<x`"，这种跨边界关联实际上是不稳定的——顺序受框排序策略影响，缺乏一致的几何动机。**更大的块 = 更差的边界对齐 = 更多的训练噪声**。

PBD 用**框边界对齐**打破了这个衰减趋势：L=6 恰好是一个标准检测框的 token 数，块边界与几何单元边界完美重合。模型不再需要学习跨边界的虚假关联——每个块内的 6 个 token 永远属于同一个语义/几何单元。

**(c) 联合损失与解码模式消融**

```
配置                     | Slow F1 | Fast BPS | Fast F1 | Hybrid F1
NTP only                  | 50.1   | -       | -       | -
NTP + MTP (joint)         | 52.1   | 16.9    | 49.6    | 51.6
MTP only                  | -       | 16.7    | 47.2    | -
```

MTP only 不仅损害了 MTP 自身（47.2 vs joint 49.6），也无法提供 Slow 模式——训练时如果只学 MTP，模型无法逐 token 自回归生成。这就是为什么"双序列联合训练"不是两种损失的简单相加，而是一个**完整的训练哲学**：MTP 需要 NTP 提供因果基础，NTP 需要 MTP 提供结构化强化。

### 8.6 Point 任务

论文在 Appendix Table 11 中展示了 LocateAnything 在纯点定位（Pointing）任务上的表现：

| 方法 | COCO F1@Point | Dense200 F1@Point | RefCOCOg test F1@Point |
|------|--------------|-------------------|----------------------|
| Molmo-7B-D | 77.3 | 33.1 | 83.6 |
| Rex-Omni-3B | 80.5 | 82.5 | 85.1 |
| **LocateAnything-3B** | **83.9** | **87.6** | **91.0** |

RefCOCOg test 的 91.0 是在指代理解（REC）任务上的点定位分数，它匹配的是"给定一段自然语言描述，定位目标物体"——这是 VLM 最核心的跨模态语义对齐能力。PBD 在这个任务上的优势可能是由于 138M 数据中大量自然语言查询（尤其是 GroundCUA 增强的查询）带来的语义多样性。

---

## 9. 局限性分析

### 9.1 当前局限

**RL 优化空间**：Hybrid 模式的回退阈值（概率 < 0.7，max-min 差 > 80）是启发式设计的。如果训练信号能直接优化"何时使用 MTP vs NTP"，回退频率可能进一步降低。论文明确提到 RL 是"重要的下一步"，目标是优化块级解码策略以降低 fallback 频率。

**极小目标的离散化精度**：1000 级量化对 `< 10×10` 像素目标（如 VisDrone 中的无人机俯拍行人）存在精度天花板。$1000/N$ 的量化步长在 $N<10$ 时可能退化到 < 100 个可区分的坐标值，无法精确表达极小的位置差异。

**格式验证的鲁棒性**：Hybrid 模式依赖对块内 token 的语法格式验证（检测 `<box> <211> </ref>` 这类错误）。这需要程序化的规则检查——但有些格式错误可能以更微妙的方式出现（如 `<ref> person </ref> <eos>` 在应该输出框的时候输出了结束块），此时无法通过 token-level 语法检测发现。

**数据偏差分布**：138M 查询中检测域占 93M (67%)，GUI 域 23M (17%)。指代/OCR/Layout/Point 四个域加起来只有 21%——这种分布可能在极端指代任务中对复杂的语言推理不够支持。

### 9.2 未来方向

**RL for Block-Level Policy**：训练模型自主判断"当前块适合 MTP 还是 NTP"，形成基于置信度的自适应解码策略。

**结构化视觉编码器**：PBD 目前只改革了**输出侧**的结构。如果视觉编码器内部也引入明确的空间结构表示（如带有空间坐标意识的 visual token），可能进一步强化视觉-坐标对齐。

**跨模态扩展**：同一个并行块解码思路可以扩展到 3D 检测（框 = 3D cuboid 的 8 个角点 + 朝向）和视频（时间维度上的 box tube 作为一个结构化单元），这些任务的结构耦合同样被当前的逐 token 范式所压制。

---

## 10. 代码实践

模型已全量开源在 [HuggingFace](https://huggingface.co/nvidia/LocateAnything-3B)。最小可运行示例：

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnything:
    """封装三种推理模式，支持 detect / point / phrase_grounding"""

    def __init__(self, model_path: str = "nvidia/LocateAnything-3B",
                 device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device).eval()

    def predict(self, image, question, generation_mode="hybrid",
                max_new_tokens=2048, temperature=0.7, verbose=False):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )
        result = response[0] if isinstance(response, tuple) else response
        return {"answer": result}

    def detect(self, image, categories, **kwargs):
        prompt = f"Locate all the instances that matches the following description: {'</c>'.join(categories)}."
        return self.predict(image, prompt, **kwargs)

    def phrase_grounding(self, image, phrase, **kwargs):
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image, phrase, **kwargs):
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def single_box(self, image, phrase, **kwargs):
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)


if __name__ == "__main__":
    worker = LocateAnything("nvidia/LocateAnything-3B")
    img = Image.open("example.jpg").convert("RGB")

    # 通用检测 — 输出多个框
    result = worker.detect(img, ["person", "car", "bicycle"])
    print("Detection:", result["answer"])

    # GUI 元素定位 — 点输出
    result = worker.point(img, "the submit button at the bottom of the form")
    print("Point:", result["answer"])

    # 短语定位 — 支持属性/空间/推理查询
    result = worker.phrase_grounding(img, "the red car next to the traffic light")
    print("Grounding:", result["answer"])
```

**输出格式**：
```
<ref> person </ref><box><x120><y200><x350><y580></box>
<ref> car </ref><box><x400><y300><x650><y520></box>
<eos>
```

**硬件要求**：
- 参数：3B | 显存：~6.3 GB (bf16) | GPU：H100/A100/RTX 4090 (CUDA 11.8+)
- 推理速度：Hybrid 模式 ~12.7 BPS（单张 H100, batch=1）


## 11. 总结

LocateAnything 在四个维度上产生了实质的贡献：

**理论层面**：首次系统论证了**块边界对齐对结构化生成的重要性**。固定大小随机切分块虽然在 LLM 领域有效（因为在自然语言中 token 的相邻关系已经是够好的先验），但对几何坐标这种"高度结构化耦合"的数据，随机切分是负优化（SDLM-B8 的 45.8 vs PBD 的 49.6）。"什么时候对齐，什么时候不需要"——这个判定标准会影响所有把结构化输出引入 VLM 的后续工作。

**工程层面**：138M 样本的数据引擎（Qwen3-VL → Molmo → SAM 3 → Rex-Omni → Qwen3-VL 后验证）和双序列联合训练 + Stream Packing + MagiAttention 的训练基础设施构成了一个可重用的方法论。"用 VLM 训练 VLM"的范式从这个工作获得了系统的验证——不是零散地用一个 VLM 标注一个数据集，而是用三个 VLM 编排成一条生产级流水线。

**系统层面**：三种解码模式按需切换（Fast/Slow/Hybrid）为不同应用场景提供了细粒度的控制。这个设计与传统检测系统的单一模式输出形成了鲜明对比——它承认了"不同场景对速度/精度/鲁棒性的需求是不同的"，并把选择权交给了使用方。Hybrid 模式是理论最优：12.7 BPS 配合 98.8% 的 Slow 模式精度，没有硬性代价。

**基准层面**：LocateAnything 在 COCO、LVIS、VisDrone、Dense200、ScreenSpot-Pro、DocLayNet、M6Doc、TotalText、HumanRef、RefCOCOg 等 10+ 基准上全部取得了 SOTA 或接近 SOTA 的结果——这不是"在某个任务上表现好"，而是**跨域的泛化能力**。数据引擎的六个域覆盖（检测 67%、GUI 17%、指代 7%、OCR 4%、Layout 3%、Point 2%）很可能就是这种泛化的数据侧来源。

把空间归纳偏置重新注入语言模型的输出格式——这个看似简单的操作，可能是 VLM 感知层演进中的一个重要转折点。

---

## 参考

- [LocateAnything 论文](https://arxiv.org/abs/2605.27365v2)
- [GitHub](https://github.com/NVlabs/LocateAnything)
- [HuggingFace 模型](https://huggingface.co/nvidia/LocateAnything-3B)
- [在线 Demo](https://huggingface.co/spaces/nvidia/LocateAnything)
- [MagiAttention](https://github.com/SandAI-org/MagiAttention/)
- [Rex-Omni](https://arxiv.org/abs/2504.18089)
- [SDLM (Sequential Diffusion Language Models)](https://arxiv.org/abs/2509.24007)
- [Block Diffusion](https://arxiv.org/abs/2503.09573)
