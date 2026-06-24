---
title: "LocateAnything：当 VLM 学会一次性画出整个框 — 并行框解码深度解读"
published: 2026-06-24
description: "NVIDIA 提出的 LocateAnything 用 Parallel Box Decoding 颠覆了 VLM 逐 token 生成坐标的范式，检测速度提升 2.5 倍以上，同时准确率全面 SOTA。本文深入拆解完整技术栈、训练策略、推理机制，并附可运行的代码实践。"
tags: [论文笔记, VLM, 目标检测, 多模态, 视觉定位, 模型部署]
category: AI
draft: false
---

> **论文**: LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
>
> **作者**: Shihao Wang, Shilong Liu, Yuanguo Kuang 等（NVIDIA）
>
> **链接**: [arXiv 2605.27365v2](https://arxiv.org/abs/2605.27365v2)
>
> **代码 & 模型**: [GitHub](https://github.com/NVlabs/LocateAnything) | [HuggingFace](https://huggingface.co/nvidia/LocateAnything-3B) | [在线 Demo](https://huggingface.co/spaces/nvidia/LocateAnything)

---

## 1. 问题的根源：当语言模型被逼着做检测

### 1.1 VLM 做 Grounding 的基本范式

从 Qwen-VL、InternVL 到 GPT-4V，主流 VLM 已经把视觉定位统一成了生成问题：输入一张图加一段文字指令，输出一段包含坐标的文本。

编码器从图中提取视觉 token，语言模型在这两组 token 之上做因果自回归生成。想让它输出一个框 `(x1, y1, x2, y2)`，当前主流方案有两种：

**方案 A：文本数字形式（Textual Digits）**

```
<ref>cat</ref><box><238><419><501><703></box>
```

每个数字是一个独立 token。"238" 被拆成 "2"、"3"、"8" 三个 token，模型逐一预测。Qwen3-VL、DeepSeek-VL2 走这条路。

**方案 B：量化坐标形式（Quantized Tokens）**

```
<ref>cat</ref><box><x238><y419><x501><y703></box>
```

坐标映射到专用 token（x0~x999），每个坐标一步完成。Rex-Omni 是典型。

无论哪种方案，本质都是把**一个二维空间量**（四个角点定义的矩形）强行序列化成**一维 token 流**。把空间结构压扁成线性序列，是后续一切问题的根源。

### 1.2 三个层面的困境

**延迟**：每预测一个框需要至少 4 个甚至 12+ 个解码步骤。一张图里 50 个物体，解码步数动辄几百步，每一步都依赖前一步——纯串行瓶颈，GPU 并行能力完全浪费。

**结构**：逐 token 预测无法建模坐标之间的联合依赖。当模型已经生成 "x1=238"，生成 "y1" 时它知道左边界，但到 "x2=501" 时已无法回顾 "x1" 曾赋予什么值——因果注意力天然断了后路。框的边界容易不精准。

**监督**：训练损失是逐 token 交叉熵，对框的"好""坏"没有整体空间衡量。差之毫厘的框和谬以千里的框在 token 级损失上可能差异不大——因为 x1 和 y1 碰巧一样。

### 1.3 此前的 MTP 方案为什么不够

Multi-Token Prediction 是并行解码领域的自然思路：不再逐个预测，而是一次预测多个。典型范式：

- **SDLM**：序列切分成固定大小块（B=6），块内并行预测，块间因果。但块边界完全随机——可能把一个框的 x2 和下一个框的 `<ref>` 切到同一个块
- **Block Diffusion**：随机 mask 部分 token，从任意位置重建缺失内容。块内全连接，块间因果，但同样不关心几何结构边界
- **LLaDA / Dream**：纯扩散范式，从完全 mask 的噪声状态逐步去噪，实现双向并行生成

共性问题是 **structure-agnostic**：把任何 token 当成无差别符号流。对弱结构化的自然语言还行，但对高度耦合的 `(x1,y1,x2,y2)`，随机切分意味着模型学习各种跨边界、跨类别的虚假关联。论文 Figure 2 直观展示了这一现象：标准 MTP 产生各种"错位"，坐标 token 和语义 token 混在一起，等效于注入大量噪声监督信号。

---

## 2. Parallel Box Decoding：让框重新成为一个整体

### 2.1 核心洞察

LocateAnything 的核心洞察是一种对几何本体的回归：**`(x1,y1,x2,y2)` 不是四个独立标量，而是一个不可分割的原子单元**。它们共同定义了一个二维矩形，分开预测就失去"矩形"应当承载的约束力。

Parallel Box Decoding（PBD）的策略是：**让 MTP 的预测块边界严格与几何单元对齐，每个块就是一个完整的框（或其语义前缀）。**

### 2.2 输出格式设计

输出重构成固定长度 **L=6** 的原子块序列。坐标归一化到 [0, 1000] 后离散化：

```
<box> <x1> <y1> <x2> <y2> </box>
token0 token1 token2 token3 token4 token5
```

四种功能块类型：

| 块类型 | 作用 | 示例 |
|--------|------|------|
| Semantic Block | 编码目标语言描述 | `<ref> hot dog </ref> <null> <null>` |
| Box Block | 编码四个量化坐标 | `<box> <663> <638> <313> <345> </box>` |
| Negative Block | 明确表示未找到目标 | `<ref> cat </ref> <neg> <null> <null> <null>` |
| End Block | 标记生成结束 | `<eos> <null> <null> <null> <null> <null>` |

语义描述若超出单块容量，被切分到多个连续 Semantic Block；但 Box Block 一定完整地属于一个块。未占满位置用 `<null>` 填充。

---

## 3. 训练设计：一条序列，两种格式，一个模型

### 3.1 为什么不能只训练 MTP

直接把 NTP 数据按 PBD 规则填充成块格式，块内 mask 掉后 5 个 token，只算 MTP 损失——看似一步到位，但会破坏模型预训练中建立的因果推理链。消融实验验证：只用 MTP 损失训练，Fast 模式 F1 从 49.6 跌到 47.2。

### 3.2 双序列联合训练

LocateAnything 在同一次前向传播中送入两个视角的输入：

```
x_all = [ 视觉token | 文本查询 | NTP序列 | MTP块序列 ]
```

同一份 ground truth，两种格式：

**NTP 序列**（标准 token 级格式）：

```
<ref> hot dog </ref> <box> <663> <638> <313> <345> </box> <eos>
```

**MTP 块序列**（每 6 个 token 一组，保留每组的第一个 token 作为锚点，其余替换为 `[mask]`）：

```
Block 1:  <ref>  [mask] [mask] [mask] [mask] [mask]
Block 2:  <box>  [mask] [mask] [mask] [mask] [mask]
Block 3:  <eos>  [mask] [mask] [mask] [mask] [mask]
```

锚点 token（`<ref>`, `<box>`, `<eos>`）告诉模型"这一块是什么类型"，指导并行预测方向。

### 3.3 注意力掩码设计

这是训练框架中最精妙的部分。一个统一掩码同时管控三种 attention 行为：

**（1）共享上下文 + NTP 序列：标准因果注意力** —— 视觉 token 和文本查询构成共享上下文，加上 NTP 序列整体使用因果掩码，与原始 VLM 前向行为完全一致，保护因果推理能力。

**（2）MTP 与 NTP 之间：单向隔离** —— NTP 区域禁止 attend 到 MTP 区域，防止数据泄露。但 MTP 区域可以 attend 到 NTP 区域——MTP 预测时可"参考"完整因果答案链作为上下文。

**（3）MTP 块间：块级因果** —— 一个块只能 attend 到前面所有块（包括共享上下文、NTP 序列和已完成 MTP 块），不能看到后面。与实际推理时的生成顺序完全一致。

**（4）MTP 块内：全连接双向注意力** —— 同一块内的 6 个 token 完全双向可见。这是 PBD 的灵魂——块内 `[mask]` token 互相可见，学会"x1 很大时 y1 通常也很大"、"x2 必须大于 x1"等几何约束。6 个 token 同时预测，损失是 6 个位置交叉熵之和。

### 3.4 损失函数

```
L = L_ntp + L_mtp
```

两者权重相等，直接相加。NTP 流学到的因果推理提升了 MTP 块内预测质量，MTP 流学到的空间耦合反过来增强 NTP 定位精度。消融实验验证：加 MTP 损失后，Slow Mode（纯 NTP）F1 从 50.1 提升到 52.1。

---

## 4. 推理机制：三种模式与自适应回退

### 4.1 三种推理模式

| 模式 | 行为 | 吞吐量 (BPS) | 适用场景 |
|------|------|-------------|----------|
| **Slow** | 逐 token 自回归生成 | ~3.9 | 数据标注、离线评估 |
| **Fast** | 每次并行预测完整 6-token 块 | **~16.9** | 机器人、端侧推理 |
| **Hybrid** | 默认 Fast，异常时回退到 Slow | ~13.2 | 生产环境 |

**Fast 模式推理流程**：

```
Step 1: [vision_tokens | query] → 并行预测 → <ref> hot dog </ref>
Step 2: [history | Block 1]    → 并行预测 → <box> <663> <638> <313> <345> </box>
Step 3: [history | Block 2]    → 并行预测 → <eos>（结束）
```

每步一次性并行预测一个完整 6-token 块，`<null>` 位置丢弃。已提交 token 进入 KV cache 作为后续因果上下文。

### 4.2 KV Cache 管理

每步 MTP 完成后清理 cache：保留共享上下文 + NTP 前缀 + 已提交真实 token；删除 mask token + 重复锚点 token。不做截断的话，历史中残留的 mask token 会导致注意力偏离训练分布。

### 4.3 Hybrid 模式的故障检测与回退

并行解码并非万能。两种典型失败模式：

**格式不规则（Format Irregularity）**：类别边界处模型犹豫切换类别，导致单块中结构混合，如 `"<box> <211> </ref> <911> <887> </box>"`——`</ref>` 不应出现在 Box Block 内。检测方法：验证块内 token 语法结构。

**空间模糊（Spatial Ambiguity）**：物体规则网格排布时（如货架商品），MTP 可能输出"卡在中间"的坐标。检测条件：（1）top-1 坐标 token 概率 < 0.7；（2）top-5 坐标在 [0, 1000] 空间内 max-min 差 > 80。

触发后丢弃问题块，回到上一已验证前缀，NTP 逐 token 补全该块。回退比例不高，Hybrid 因此能以接近 Fast 的速度运行。

---

## 5. 数据工程：规模决定上限

### 5.1 数据规模

```
12,000,000+ 张独立图片
138,000,000+ 条自然语言查询
785,000,000+ 个标注框
```

| 任务域 | 占比 | 代表数据集 |
|--------|------|-----------|
| 通用目标检测 | 66.9% | COCO, Object365, OpenImages, V3Det, EgoObjects |
| UI 界面接地 | 16.5% | ScaleCUA, GroundCUA, GTAGrounding, OSAtlas |
| 自然语言指代 | 7.3% | RefCOCO/g/+, gRefCOCO, Flickr30k Entities |
| 文本定位 (OCR) | 3.6% | BLIP3OCR, TextOCR, LSVT, COCO-Text |
| 文档版面分析 | 3.5% | DocLayNet, PubLayNet, M6Doc, TableBank |
| 点定位 | 2.2% | PixmoPoints, RoboAfford |

### 5.2 数据构造引擎

整体是 **VLM-as-Annotator** 流水线：对有标注框的检测数据，用 Qwen3-VL 以每个框类别为提示，合成含属性、空间关系、推理线索的自然语言查询；将这些查询喂给 Molmo 预测候选点，保留落在对应 GT 框内的点；对纯语言查询（指代/短语定位），Molmo 从提示中预测候选定位点，IoU 匹配 GT 确定正负样本。大规模检测数据的框级监督 + 自然语言查询的语义多样性，两者结合是模型多域泛化能力的基础。

### 5.3 四阶段训练策略

| 阶段 | 目标 | 数据 | 学习率 | 最大长度 |
|------|------|------|--------|---------|
| Stage 1 | 世界知识注入 | 通用 caption 数据 | 2e-4 | 8,192 |
| Stage 2 | 通用 VQA 对齐 | 全量通用数据（不含检测/定位） | 4e-5 | 8,192 |
| Stage 3 | 检测 & 定位能力注入 | 全量 138M 查询 | 4e-5 | 25,600 |
| Stage 4 | 密集场景精锐化 | 20% 通用 + 高密度数据 | 1e-5 | 25,600 |

Stage 3/4 采用 **Stream Packing**：将多条样本拼接填满最大长度，通过特殊 attention mask 实现样本隔离，大幅提升训练效率。

---

## 6. 实验结果

### 6.1 吞吐量（单张 H100，batch_size=1）

| 方法 | 坐标格式 | 解码方式 | BPS |
|------|---------|---------|-----|
| Qwen3-VL-4B | 文本数字 | NTP | 1.1 |
| Qwen3-VL-8B | 文本数字 | NTP | 1.0 |
| OVIS2.5-2B | 文本数字 | NTP | 1.3 |
| MiMo-VL-7B | 文本数字 | NTP | 1.0 |
| Cosmos-Reason2-8B | 文本数字 | NTP | 1.0 |
| Rex-Omni-3B | 量化坐标 | NTP | 5.0 |
| LocateAnything-3B (Slow) | PBD | NTP | 3.9 |
| **LocateAnything-3B (Hybrid)** | PBD | MTP + fallback | **12.7** |
| **LocateAnything-3B (Fast)** | PBD | MTP | **16.9** |

Hybrid 模式 12.7 BPS 意味着每秒约 13 个精确定位框，实时视频流（30 FPS 每帧 5 物体）绰绰有余。Fast 模式在密集场景飙升至约 25 BPS。

### 6.2 准确率（全部 Hybrid 模式）

**通用目标检测（COCO / LVIS）**

| 方法 | LVIS F1 Mean | COCO F1 Mean |
|------|-------------|-------------|
| Grounding DINO-Swin-T | 38.8 | 56.6 |
| Qwen3-VL-8B | 44.8 | 45.7 |
| SEED1.5-VL | 46.7 | 51.4 |
| Rex-Omni-3B | 46.9 | 52.9 |
| **LocateAnything-3B** | **50.7** | **54.7** |

**密集检测（Dense200 / VisDrone）**

| 方法 | Dense200 F1 Mean | VisDrone F1 Mean |
|------|-----------------|-----------------|
| Qwen3-VL-8B | 9.6 | 25.8 |
| Rex-Omni-3B | 58.3 | 35.8 |
| **LocateAnything-3B** | **58.7** | **39.9** |

LocateAnything 在 IoU=0.95 严格阈值下 F1 显著更高。LVIS 的 0.95 IoU 拿到 31.1（Rex-Omni 20.7）。PBD 不是简单地快，而是真正提高了框的边界精度——模型一次性预测四个角，空间一致性天然保留。

**GUI 接地（ScreenSpot-Pro）**

| 方法 | 参数量 | Avg F1 |
|------|--------|--------|
| Rex-Omni-3B | 3B | 36.8 |
| Qwen3-VL-30B-A3B | 3B (activated) | 53.7 |
| GUI-Owl-32B | 32B | 58.0 |
| UI-Venus-1.5-2B | 2B | 57.7 |
| **LocateAnything-3B** | 3B | **60.3** |

3B 参数碾压 7B/32B 的 UI 专门模型。Icon 类元素表现尤为突出（50.3 vs 同级 20-40），归功于 PBD 在密集小尺度场景的精确边界定位。

**文档版面 / OCR**

| 方法 | DocLayNet Mean | M6Doc Mean | TotalText Mean |
|------|---------------|------------|---------------|
| Qwen3-VL-4B | 37.2 | 19.0 | 36.1 |
| Rex-Omni-3B | 70.7 | 55.6 | 40.6 |
| **LocateAnything-3B** | **76.8** | **70.1** | **43.3** |

### 6.3 消融关键结论

| 维度 | 发现 |
|------|------|
| 坐标表示 | PBD Slow（3.9 BPS, 52.1 F1）> 量化（3.9 BPS, 50.1 F1）> 文本（1.3 BPS, 49.1 F1） |
| MTP 块对齐 | PBD 对齐框边界（16.9 BPS, 49.6 F1）远优 SDLM-B6 随机切分（5.5 BPS, 46.1 F1） |
| 双损失联合 | NTP 单独 50.1 F1；加 MTP 联合 52.1 F1。MTP 流空间约束反哺 NTP |
| 框排序 | X-Y Corner Order 最优（先左上角 x 再 y），优于 Center Distance / Area / Random |

---

## 7. 快速上手：代码实践

模型已全量开源在 [HuggingFace nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)。下面是最小可运行样例：

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class LocateAnythingWorker:
    """LocateAnything 封装，支持 detect 和 point 任务"""

    def __init__(self, model_path: str, device: str = "cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True,
        ).to(device).eval()

    def predict(self, image, question, generation_mode="hybrid",
                max_new_tokens=2048, temperature=0.7, verbose=True):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
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
        result = {"answer": response[0] if isinstance(response, tuple) else response}
        return result

    def detect(self, image, categories, **kwargs):
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image, phrase, **kwargs):
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)


if __name__ == "__main__":
    worker = LocateAnythingWorker("nvidia/LocateAnything-3B")
    img = Image.open("example.jpg").convert("RGB")
    result = worker.detect(img, ["person", "car", "bicycle"])
    print("Detection:", result["answer"])
```

### 7.1 推理模式说明

- `generation_mode="hybrid"`（推荐）：默认并行解码，异常时局部回退到逐 token 生成
- `generation_mode="fast"`：纯并行，最快但复杂场景可能出错
- `generation_mode="slow"`：纯 NTP，最稳定

### 7.2 输出格式

```
<ref> person </ref><box><x120><y200><x350><y580></box>
<ref> car </ref><box><x400><y300><x650><y520></box>
<eos>
```

### 7.3 硬件要求

- 参数：3B / 显存：~6.3 GB（bf16） / GPU: A100/H100/RTX 4090, CUDA 11.8+

---

## 8. 分析

### 8.1 为什么有效

PBD 在三个环节同时获益：

- **训练效率**：直接在 MTP 块内用全连接注意力学习坐标联合分布，块对齐策略让模型学到强几何约束（x2 必须大于 x1 且 y2 大于 y1）
- **推理速度**：密集场景中 NTP 的 KV cache 随框数线性增长，PBD 每步一块增长更慢。目标框从 20 到 300 时，并行吞吐量从 ~12 升至 ~25 BPS
- **边界精度**：IoU=0.95 下 F1 提升是最有力证据——PBD 输出框整体质量更高，四个角相对位置更合理。更好的结构归纳偏置同时带来速度和精度

### 8.2 局限与展望

- **RL 优化**：当前 fallback 阈值（概率 < 0.7，max-min 差 > 80）是启发式设计。用 RL 让模型学判断"何时 MTP 何时 NTP"，可能进一步降低回退频率
- **长尾与极小物体**：< 10x10 像素时 1000 量级离散化有精度限制
- **结构化输入侧**：能否将结构化思路延伸到视觉编码器内部，让视觉 token 也带空间结构意识？

### 8.3 更宏观的意义

VLM 正从"智能对话"走向"具身行动"——机器人需实时定位操作目标，自动驾驶需瞬时检测道路物体，GUI Agent 需精确定位按钮。3B 参数做到 12.7 BPS 吞吐量 + 60.3 GUI grounding 分数，意味着高质量通用视觉定位**不再需要以规模换速度**。把空间归纳偏置重新注入语言模型输出格式——这个看似简单的思路，可能成为 VLM 感知能力演进中的重要范式转变。

---

## 参考

- [LocateAnything 论文](https://arxiv.org/abs/2605.27365v2)
- [GitHub](https://github.com/NVlabs/LocateAnything)
- [HuggingFace 模型](https://huggingface.co/nvidia/LocateAnything-3B)
- [在线 Demo](https://huggingface.co/spaces/nvidia/LocateAnything)
- [Rex-Omni](https://arxiv.org/abs/2504.18089)
