---
title: "ScriptCraft：从小说到分镜剧本，用 LangGraph 编排 AI 改编流水线"
published: 2026-06-24
description: "七牛云实训路演作品 —— 一个小说改编工作台，把整本小说通过 Agent 协作逐步转为结构化 YAML 剧本，支持分镜调度、画布预演与流式追踪。详析 LangGraph 编排、双 Schema 设计和工具流自修复机制。"
tags: [项目复盘, Agent, LangGraph, LangChain, 剧本生成, 流式处理]
category: 项目
draft: false
---

## 缘起：一个实训路演作品

ScriptCraft 是我提交给七牛云实训的路演作品，初筛过了但没去成路演。它做的事情一句话就能说清：**把一本小说从原始文本一步步变成可执行的分镜剧本 YAML，并在工作台里预览编辑、画布预演、导出**。背后的核心技术挑战包括长文本上下文管理、多步 Agent 协作、流式进度推送和结构化输出校验。

---

## 1. 整体流水线：六个步骤，四个 Agent

把一本几十万字的小说一次性丢给 LLM 输出结构化 YAML 剧本——在当前的模型能力边界之外。ScriptCraft 的解法是把改编流程拆成六个顺序步骤：

```
原始小说正文
  → (1) 章节解析（正则，纯工程侧）
  → (2) AI 章节分析（LangGraph 线性流水线）
  → (3) 故事元素抽取（Tool-based Agent，卡片库归并）
  → (4) YAML 剧本生成（三阶段流水线：规划→生成→记忆更新，逐章循环）
  → (5) 编辑 / 修复 / 版本保存 / 导出
  → (6) 画布预演（PixiJS 渲染 storyboard 时间线）
```

每一步严格依赖前一步。后端在项目级做了 AI 任务互斥锁：同一时刻一个项目只能有一个 AI 任务。服务器重启时自动将 running 状态标记为 failed，防止僵尸任务。

---

## 2. 长文本的答案：章节流式 + 短记忆

这是整个架构中最关键的决策。步骤 2、3、4 都采用按章节顺序流式处理的策略。

### 2.1 章节分析（步骤 2）

`ChapterAnalyzer` 是 LangGraph 三节点流水线（build_prompt → call_model → normalize）。每次只处理当前章节正文，加上一个极小的 short memory：`story_so_far`（前文摘要拼接）、`known_characters`（已出现角色名）、`continuity_notes`（模型标记的连续线索）。模型不需要记住十万字正文，只需知道"剧情走到哪"、"出现了谁"、"有什么伏笔"。

### 2.2 故事元素归并（步骤 3）

`StoryElementExtractor` 是 Tool-based LangGraph Agent，持有 8 个工具：读取当前章节、搜索任意章节、读取卡片库、upsert 角色/地点/事件/场景卡片、校验一致性。关键设计是 **upsert 而非 insert**——后文揭示角色身份变化时必须更新已有角色卡，不创建重复。这对小说中后段尤其重要，因为角色性格弧光和悬念铺设往往中途才开始显现。

### 2.3 剧本生成 + 记忆链（步骤 4）

`ScriptYamlGenerator` 每章执行三步：Scene Planner（拆解场景计划）→ Fragment Generator（生成本章 YAML 片段，自带校验-重试）→ Memory Updater（浓缩为结构化 script_memory）。这份 memory 送入下一章，形成 **逐章滚动的记忆链**。每章只看到精炼上下文而非整本小说包袱，对长篇文本（如网文几百章）至关重要。

---

## 3. YAML Schema 设计：正文 + 调度双层分离

ScriptCraft 的 YAML 输出是一个同时包含剧本文本和导演调度的双层结构。顶层包含 metadata、characters、locations、events、scenes、storyboard。scenes 是正文层（对白、动作、旁白、转场、声音），storyboard 是调度层（镜头、站位、姿态、道具、氛围、时长）。两层通过 source_beat_id 追溯，作者改台词不丢分镜，改分镜不覆盖台词。

定义约 15 条校验规则覆盖引用完整性、ID 唯一性、字段合法性。前后端各维护一套校验——前端 `useScriptYamlDocument` 提供即时反馈，后端 `validate_script_yaml_payload` 保障数据安全。

---

## 4. 四个 LangGraph Agent 的设计对比

| Agent | 图结构 | 工具 | 自修复 |
|-------|--------|------|--------|
| Chapter Analyzer | 线性 3 节点 | 无 | 无 |
| Story Element Merger | Agent → Tools → Validate | 8 个 | 校验-重试 |
| Script Fragment Generator | 线性 + 条件边 | 无（纯 prompt） | 校验-重试（2 轮） |
| YAML Repairer | Agent → Tools → Validate | 6 个专用 | 校验-重试（3 轮） |

### 4.1 YAML Repair Agent 的工具设计

这是工具最密集、设计最精巧的 Agent。6 个工具：read_yaml、read_yaml_context、write_yaml（支持 set_field/append_to_list/replace_node/remove_field）、append_shot_action、ensure_cast_member、set_shot_field。

关键约束：只做局部修改（禁止输出完整 YAML，确保追踪性和可回滚）；ID 稳定性（用 read_yaml_context 解析角色真实姓名来判断 ID 语义，理解用语义、写回用稳定 ID）；旁白检测（旁白/叙述者/画外音绝不进 cast）；操作追踪（三轮内修复成功就保存版本，超三轮报错要求人工介入）。

### 4.2 流式回调

所有 Agent 挂载 `on_stream` 回调作为独立观测通道，通过 `emit_tool_event` 推送实时文本到前端。后端写入 `AIRun.output_payload.stream` 数组，前端轮询渲染成事件时间线。用户看到的是"正在分析第 3/12 章"、"正在合并角色卡片"——而非死板的 loading。

---

## 5. 前端工程

### 5.1 技术栈

Vue 3 + Vite + TypeScript + shadcn-vue + Tailwind CSS v4。三个核心可视化库：D3 Force（故事图谱力导向布局）、PixiJS（画布预演渲染）、GSAP（动画缓动）。

### 5.2 故事图谱：D3 Force 驱动的关系可视化

故事元素抽取完成后，前端拿到的是一组孤立列表——读者能看到"林舟出现在第三章"，但看不出"林舟和谁在哪个地点发生了哪些事"。ScriptCraft 用 D3 Force 把四组故事元素转成可交互关系图谱。

`StoryGraphPanel.vue` 从 `StoryElementsSnapshot`（角色/地点/事件/场景卡片）中提取四类节点和四种连接：角色→事件（character-event）、事件→地点（event-location）、场景→角色/地点/事件（scene-character/location/event）。力导向布局用 `forceManyBody`（斥力）+ `forceLink`（引力）+ `forceCollide`（防重叠）+ `forceCenter`（居中）调度。

每个节点类型分配不同视觉权重：角色节点最大（r=6，蓝色，往往是最核心的枢纽）、事件节点较小（r=4.5，琥珀色）、地点节点中等（r=5，翠绿色）、场景卡片节点最大（r=6.5，紫色，连接最多的角色和地点）。

交互层面支持三件事：拖拽节点调整布局（d3-drag）、滚轮缩放（d3-zoom）、点击节点弹出详情面板。一个关键的交互设计：**图谱不是静态导出图，而是探索工具**。左侧选中角色时，节点高亮，相关事件和地点通过连接线形成星形辐射，用户可在角色→事件→地点→其他角色之间跳转，快速建立故事心智模型。

工程上，simulation tick 是 D3 的每帧事件循环，在 Vue `onMounted` 启动、`onBeforeUnmount` 中 `simulation.stop()` 清理。节点和连接通过 computed 响应式计算——数据更新时图谱自动重建。

### 5.3 YAML 的消费链

`useScriptYamlDocument(scriptYaml)` 解析 → computed 产出响应式数据 → 被 ScriptYamlPanel / ScriptResultPanel / StoryboardPlayerDialog / StoryGraphPanel 分别消费。前端还做了完整校验（scriptValidationIssues），列出问题清单并支持跳转定位。

### 5.4 TypeScript 类型建模

前端用 TypeScript 完整建模了 YAML Schema：ScriptCharacter / ScriptLocation / ScriptEvent / ScriptScene / ScriptBeat / ScriptStoryboardScene / ScriptStoryboardShot / ScriptStoryboardAction 等，IDE 自动补全和类型检查减少手误。

---

## 6. 数据库与状态管理

SQLAlchemy + PyMySQL，六张核心表：Project（含状态机）、Chapter、ChapterAnalysis（JSON payload）、StoryElement（JSON 快照）、ScriptVersion（每次编辑/修复/保存生成新版本）、AIRun（含 stream 数组和 input/output payload）。增量处理：只重新分析有变化的章节，已分析章节复用。

---

## 7. 创新总结

- **章节流式 + 短记忆链**：用循序渐进策略解决长文本上下文管理，每步只看精炼记忆，避免角色漂移和情节矛盾
- **YAML 双层 Schema**：正文层和调度层解耦，同一份数据同时服务文本编辑、分镜预览、画布预演、版本管理和导出
- **故事图谱关系可视化**：D3 Force 把角色/地点/事件/场景编织成可探索的关系网络，不是静态图而是交互式分析工具
- **Tool-based Agent 自修复**：故事元素归并和 YAML 修复都采用工具驱动局部修改。6 个专用工具形成分层工具栈，灵活不失精确
- **流式追踪**：LangGraph 节点流从内部状态机推送到前端实时展示，大幅改善长耗时 AI 任务体验
- **项目级 AI 任务互斥**：数据库行锁 + AIRun 状态追踪 + 服务重启自动恢复
- **双端 YAML 校验**：前端即时反馈 + 后端安全保障，同构逻辑

---

## 8. 遗憾与如果重来

缺少语音和口型同步（storyboard 已定义每个 shot 的 duration_ms 和 dialogue.text）；YAML 编辑仍是纯文本而非结构化块编辑器；未对接 PDF/Fountain 导出；规模化测试（>100 章）待补。如果重来，音频合成和结构化 YAML 编辑器是最高优先级的下一步。

---

**技术栈**：Vue 3 / TypeScript / Tailwind CSS v4 / shadcn-vue / D3 Force / PixiJS / GSAP / FastAPI / SQLAlchemy / LangGraph / LangChain / MySQL
