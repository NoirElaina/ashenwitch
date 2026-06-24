---
title: "ScriptCraft 深度拆解：用 LangGraph 编排小说→分镜剧本的全链路 Agent 流水线"
published: 2026-06-25
description: "深入 ScriptCraft 源码，逐层拆解：正则章节解析 + 中文数字转换、LangGraph StateGraph 三节点线性 Agent、Tool-based 卡片归并 Agent（8 工具 + 校验-重试回路）、三阶段逐章 YAML 生成（Planner → Fragment Generator → Memory Updater）的滚动记忆链、六工具 YAML 修复 Agent（支持 set_field / append_to_list / replace_node / remove_field + resolution 引擎）、15 条校验规则的完整 YAML 双层 Schema、Pipeline Service 的任务互斥与流式追踪、PixiJS Storyboard 渲染引擎、D3 Force 故事图谱的数据建模。全文以代码路径为主线，展示 LangGraph Agent 编排在非聊天场景中的实战模式。"
tags: [项目分析, LangGraph, LangChain, Agent, 剧本生成, YAML, 流式处理, FastAPI, Vue, 故事图谱]
category: 项目
draft: false
---

ScriptCraft 是一个小说改编工作台：把一本几十万字的小说从原始文本逐步转为结构化的分镜剧本 YAML，并在工作台里提供预览编辑、画布预演、故事图谱和流式追踪。后端用 Python + LangGraph 编排四类 Agent，前端用 Vue 3 + PixiJS + D3 Force 搭建交互式创作工具。

下面从源码出发，按数据流顺序逐层拆解。

---

## 1. 整体架构：六个流水线阶段，四个 Agent 类型

```
原始小说正文
  → (1) 正则章节解析（纯工程侧，不调 AI）
  → (2) AI 章节分析（LangGraph 三节点线性流水线，逐章 + 短记忆）
  → (3) 故事元素抽取（Tool-based Agent，8 个工具 + 校验-重试回路）
  → (4) YAML 剧本生成（三阶段逐章循环：规划→片段生成→记忆更新）
  → (5) 编辑 / 修复 / 版本保存 / 导出
  → (6) 画布预演（PixiJS 渲染 storyboard 时间线）
```

后端在 `pipeline_service.py` 中实现了完整的项目级任务编排：同一时刻一个项目只能有一个 AI 任务运行（数据库行锁 + `with_for_update()`），服务器重启时自动将所有 `running` 状态标记为 `failed` 并恢复项目状态。

项目状态机定义了有序的流转路径：

```
chapters_ready → chapter_analysis_running → chapter_analyses_ready
→ story_elements_running → story_elements_ready
→ script_yaml_running → script_ready
```

中间每个阶段都有对应的 `_failed` 状态作为错误路径，确保前端能精确知道项目卡在哪一步。

---

## 2. 章节解析：正则引擎 + 中文数字转换

`chapter_parser.py` 是整个流水线中唯一不调 AI 的阶段，但它的正确性决定了后续所有 Agent 的输入质量。

### 2.1 正则设计

核心正则是 `CHAPTER_HEADING_PATTERN`，支持中英混合匹配：

```python
CHAPTER_HEADING_PATTERN = re.compile(
    r"^\s*(?P<heading>("
    r"第(?P<cn_index>[0-9一二三四五六七八九十百千万]+)[章节回部卷集]"
    r"|Chapter\s+(?P<en_index>[0-9]+)"
    r"))\s*(?P<title>.*)$",
    re.IGNORECASE,
)
```

这个正则覆盖了从"第一章 开端"到"Chapter 12 The Confrontation"到"第三十七回"到"第一百二十三章"的全量格式。关键设计是捕获分组：`heading`（完整章节标题行）、`cn_index`（中文数字索引）、`en_index`（英文数字索引）、`title`（章名）。

### 2.2 中文数字解析

`_parse_chinese_number` 是一个紧凑的状态机：遍历字符串，遇到数字字符（一/二/.../九）累加到 `number`，遇到单位（十/百/千/万）做分段乘法。关键处理是"万"作为段分隔符——`section = (section + number) * unit` 然后加到 total，因为中文的"十三万五千"不是简单的从左到右累加。

```python
if unit == 10000:
    section = (section + number) * unit
    total += section
    section = 0  # 重置当前段
else:
    section += (number or 1) * unit  # 十/百/千 级别
```

### 2.3 增量同步逻辑

`_sync_project_chapters` 做增量更新而非全量替换：若章节内容变化（heading/title/content 任一不同），删除该章的 AI 分析记录（级联失效），然后更新章节行。这样追加新章节时不会触发已分析章节的重新处理。

---

## 3. LangGraph 的三种图模式

ScriptCraft 使用 LangGraph 编排了四种 Agent，但图结构只有三种模式：

### 3.1 模式 A：线性三节点（Chapter Analyzer）

```
START → build_prompt → call_model → normalize → END
```

最简图结构。三个节点都是纯函数，无工具调用，无分支。`build_prompt_node` 构建 System + Human 消息对，`call_model_node` 调模型并解析 JSON 响应，`normalize_node` 对模型输出做字段清洗和默认值填充。

状态定义为 `TypedDict`：

```python
class ChapterAnalysisState(TypedDict, total=False):
    title: str
    chapter: Mapping[str, Any]
    memory: Mapping[str, Any]
    messages: list[BaseMessage]
    on_stream: StreamCallback
    raw_payload: dict[str, Any]
    result: dict[str, Any]
```

状态在节点间传递，每个节点返回部分状态更新（`PartialState`），LangGraph 自动合并。`result` 字段只在最终节点写入，作为 graph.invoke() 的返回值。

这种模式适用于**单次 LLM 调用即可完成且不需要工具**的任务。Chapter Analyzer、Scene Planner、Fragment Generator、Memory Updater 都采用此模式。

### 3.2 模式 B：Tool-based Agent + 校验回路（Story Element Merger）

```
START → agent ⇄ tools → sync_tools → validate → finish → END
         ↓                        ↓ agent      ↑ (fail)
      missing_tool             (not full)   agent (retry)
```

这是整个项目中最复杂的图结构。核心路由逻辑：

```python
def _route_after_agent(state) -> str:
    # 有 tool_calls → 执行工具
    # 无 tool_calls 但有操作 → 进入校验
    # 无 tool_calls 且无操作 → missing_tool（报错）

def _route_after_tools(state) -> str:
    # 操作数 >= TOOL_CALL_LIMIT (24) → 强制校验
    # 否则 → 回到 agent 继续操作

def _route_after_validate(state) -> str:
    # result 存在 → finish
    # 否则 → agent 重新修复
```

关键的设计决策：

**(1) 24 次工具调用上限 + 校验触发点**：不是让 Agent 无限循环直到自己说"完成"，而是在 24 次工具调用后**强制进入校验**。这防止了 Agent 在复杂章节中无限 upsert，同时保证了最终一定会执行校验。即使校验失败，还有额外的 3 轮修复机会。

**(2) sync_tools 作为独立节点**：它在 tools 和 validate 之间同步 `operation_count` 到 state。这个设计避免了在 tools 节点中直接操作 state（LangGraph 中 ToolNode 只修改 messages），同时让后续路由有据可依。

**(3) 校验重试上限 (VALIDATION_REPAIR_LIMIT=3)**：每轮校验失败时，向 messages 追加一条 HumanMessage 描述错误列表，让 Agent 回到 agent 节点重新分析和修复。三轮后仍未通过则直接抛出 LLMResponseError。

### 3.3 模式 C：Tool-based Agent + 校验回路 + 写入追踪（YAML Repairer）

```
START → agent ⇄ tools → sync_tools → validate → finish → END
         ↓                        ↓ agent      ↑ (fail)
      missing_tool             (no writes)  agent (retry)
```

与模式 B 类似，但有关键差异：

```python
def _route_after_tools(state) -> str:
    operation_count = state["operation_count"]
    last_validated = state["last_validated_operation_count"]
    return "validate" if operation_count > last_validated else "agent"
```

这里的路由逻辑是"**有新写入才进校验**"——`last_validated_operation_count` 记录了上一次校验时的操作数。如果工具被调用但没有执行写入（比如 read_yaml 读取节点后 Agent 觉得不需要修改），就不会触发校验，而是让 Agent 继续工作。这避免了 Agent 在"读取→思考→再读取"的探索阶段被频繁的校验打断。

---

## 4. 故事元素卡片系统：upsert 语义 + 引用完整性

### 4.1 CardStore 的 upsert 设计

`StoryCardStore` 是故事元素抽取的核心数据结构。它维护四类卡片（角色/地点/事件/场景），每类卡片支持 **upsert 而非 insert** 语义——同一个 `id` 的卡片可以被后续章节更新。

为什么这很重要？小说中角色的性格弧光、身份揭示往往在中后段才显现。第一章的角色"林舟"可能只是一个"沉默寡言的年轻人"，到第三十章才揭示他是"潜伏多年的卧底"。不更新已有角色卡，就会产生两个"林舟"，后续剧本生成引用混乱。

upsert 的实现是在 `StoryCardStore` 的快照字典上直接修改：

```python
def upsert_character_card(self, ...):
    # 1. 查找是否已有同名/同 ID 的角色卡
    # 2. 若存在 → 原地更新字段（name, aliases, role, description, motivation）
    # 3. 若不存在 → 追加新卡片
    # 4. 记录操作日志 (self.operations.append)
```

### 4.2 引用完整性校验

`validate_snapshot` 检查四类约束：
- 事件卡片的 `involved_characters` 必须能解析到已有角色 ID
- 场景卡片的 `characters` / `location` / `source_events` 必须能解析到已有卡片
- 所有卡片的 `source_chapter` 必须指向项目中存在的章节
- 不允许创建旁白/叙述者/画外音为角色卡片

### 4.3 上下文工具

`ChapterContextTools` 提供两个关键工具：
- `read_current_chapter`：返回当前章节的完整正文 + 章节分析结果。Agent 必须在抽取卡片前调用它
- `search_project_chapters`：按关键词搜索项目内任意章节。用于"这个人物在前文出现过吗？在哪里？"的跨章节确认

这两个工具的联合使用保证了 Agent 在"抽取新卡片"和"更新已有卡片"之间有充分的证据支持。

---

## 5. YAML 剧本生成：逐章三阶段 + 滚动记忆链

`ScriptYamlGenerator.generate()` 是整个项目中最长的单函数（约 120 行），实现了 4 步逐章循环：

```
for each chapter:
    1. Scene Planner     → 拆解本章的场景计划 (plan)
    2. Fragment Generator → 生成该章的 YAML 片段 (fragment)
    3. Memory Updater     → 浓缩为 script_memory (memory)
    4. 下一章以 memory 作为上下文
```

### 5.1 Scene Planner 的约束注入

Scene Planner 的 prompt 中注入了严格的 ID 约束：

```python
"- scene.id 必须使用 {scene_prefix} 开头，按 001、002 递增。"
"- location_id 只能使用输入 locations 中已有 ID。"
"- characters 只能使用输入 characters 中已有 ID。"
"- source_chapters 只能包含当前章节 ID。"
"- 每章至少 1 个场景，不写完整对白，不写 storyboard。"
"- 优先复用 current_chapter_scene_cards 的场景标题、地点、出场角色、戏剧目的和关键节拍。"
```

这些约束在后端 `normalize_chapter_scene_plan` 中被程序化验证：检查 `chapter_id` 是否匹配、`source_chapters` 是否只含当前章节、所有 ID 引用是否存在。模型输出不符合约束**直接抛错**——不尝试修复，因为 Planner 是一个线性节点，没有工具回路。

### 5.2 Fragment Generator 的校验-重试

`ChapterScriptFragmentGenerator` 在内部实现了**2 轮校验-重试**：生成 YAML 片段后调用 `validate_script_yaml_fragment` 检查结构完整性，失败时重新 prompt 模型（附上错误信息），最多重试 2 次。

### 5.3 Memory Updater：连续性引擎

Script Memory 的结构定义：

```python
EMPTY_SCRIPT_MEMORY = {
    "story_so_far": "",           # 截至当前章的 120-240 字剧情摘要
    "character_state": [],        # [{character_id, current_goal, relationship_changes}]
    "open_threads": [],           # 尚未解决的伏笔或冲突
    "used_events": [],            # 已经改编进剧本的 event.id
    "last_scene_summary": "",     # 当前章最后一个场景的结果
}
```

Memory Updater 在每章末尾执行，把本章生成结果**浓缩为下一章可用的短记忆**。模型需要决定：
- "哪些信息下一章必须知道？"——存进 `story_so_far`
- "角色目标/关系发生了什么变化？"——更新 `character_state`
- "哪些伏笔还没解开？"——保留在 `open_threads`
- "哪些事件已经用于剧本了？"——标记 `used_events`，防止重复改编

这不是简单的信息累积，而是一个**逐章蒸馏**的过程：每章的 6000 字正文 + 场景计划 + YAML 片段被压缩为 ~500 字的结构化记忆，送入下一章。

---

## 6. YAML 修复 Agent：六工具 + 局部写入 + Resolution 引擎

`ScriptYamlToolRepairer` 是整个项目里代码最密集的模块（996 行），也是一个教科书级的 Tool-based Agent 实现。

### 6.1 六个专用工具

| 工具 | 函数 | 核心职责 |
|------|------|----------|
| `read_yaml` | 读取 YAML 局部节点 | target 可以是路径（`$.scenes[0]`）、对象 ID（`shot_001_001_03`）、带前缀的引用（`scene:scene_001`） |
| `read_yaml_context` | 读取目标节点的完整上下文 | 返回 target + scene + storyboard_scene + shot + beat + cast + entity_index，供 Agent 全面了解修复环境 |
| `write_yaml` | 局部写入 | 四种操作：`set_field`（赋字段值）、`append_to_list`（数组追加）、`replace_node`（替换整个节点）、`remove_field`（删除字段） |
| `append_shot_action` | 给指定 shot 添加动作 | 专门处理 action 类型 shot 缺少 actions 的问题 |
| `ensure_cast_member` | 确保 cast 包含指定角色 | 自动分配站位（left/center/right/back_left/back_right），已存在则更新 |
| `set_shot_field` | 修改 shot 的单个允许字段 | 白名单：type / source_beat_id / duration_ms / camera / dialogue / actions / effects / props |

### 6.2 Resolution 引擎

`_YamlDocumentTools._resolve` 是所有读写工具的底层基础设施。它支持两种目标定位方式：

**(A) 路径解析**：`$` → 根、`$.scenes` → scenes 列表、`$.scenes[0].beats[2]` → 第三个 beat。实现是按 `.` 和 `[N]` 分割路径 token，沿字典树逐层下行。

**(B) ID 解析**：`shot_001_001_03` → 在整棵 YAML 树中 DFS 搜索匹配的 `id` 字段。支持前缀 `storyboard_scene:scene_001` → 按 `scene_id` 字段搜索。`_find_mapping` 用栈实现 DFS，对每层字典检查 `predicate(value)` 条件。

### 6.3 局部写入操作

`write_yaml` 的四种操作是专门设计的**最小修改单位**：

- `set_field`：修改单个字段值（如 `write_yaml("shots[0]", "set_field", key="duration_ms", value=3000)`）
- `append_to_list`：向数组追加元素（如追加新的 beat）
- `replace_node`：替换整个节点（在确认整个 subtree 都不对时使用）
- `remove_field`：删除字段（如移除多余的 dialogue.speaker_id）

所有操作都记录到 `self.operations` 列表，每次操作后校验 YAML 是否通过了结构验证。这个设计的关键目的是**可追溯性**——每次修复都有明确的操作日志，用户可以回滚（通过 ScriptVersion 的版本管理）。

### 6.4 旁白检测

YAML 修复 Agent 的系统 prompt 中有这条关键规则：

> "如果错误是 dialogue/actions 角色没有出现在 storyboard scene.cast 中...如果该角色的真实姓名、别名或角色描述是旁白、叙述者、画外音或 narrator，不要把它加入 cast；应把对应 shot/beat 改为 narration，并移除 dialogue.speaker_id。"

这是在实际使用中发现的边缘情况：模型生成 dialogue 时可能错误地把旁白文本分配给一个角色，或者把一个只存在于旁白中的"叙述者"当作角色。修复 Agent 需要能识别这种语义级别的错误——不仅仅是引用完整性问题，更是类型错误。

---

## 7. YAML Schema 设计：正文 + 调度双层分离

### 7.1 双层结构

ScriptCraft 的 YAML 输出是一个**正文层**（文学内容）和**调度层**（导演指令）的二元结构：

```
顶层:
  schema_version: "2.0"
  title: "作品名"
  metadata: { source_title, chapters_count, adaptation_style, language, script_memory }

正文层 (scenes):
  scenes:
    - id: scene_001
      title: "初遇"
      location_id: loc_001
      characters: [char_001, char_002]
      source_events: [event_001]
      beats:
        - id: beat_001_01
          type: dialogue
          speaker_id: char_001
          content: "你好。"
        - id: beat_001_02
          type: action
          content: "林舟转过身"

调度层 (storyboard):
  storyboard:
    scenes:
      - scene_id: scene_001
        setting: { time_of_day: "dusk", atmosphere: "紧张" }
        cast:
          - character_id: char_001
            position: left
            pose: standing
          - character_id: char_002
            position: right
            pose: idle
        timeline:
          - id: shot_001_001_01
            type: dialogue
            source_beat_id: beat_001_01  ← 溯源到正文层
            duration_ms: 3000
            camera: { angle: "medium_close_up", movement: "static" }
          - id: shot_001_001_02
            type: action
            source_beat_id: beat_001_02
            duration_ms: 2000
            camera: { angle: "wide", movement: "pan_left" }
            actions:
              - actor: char_001
                motion: "转身"
                from: "面向窗外"
                to: "面向门口"
                emotion: "警惕"
```

正文层和调度层的关联通过 `source_beat_id` 完成——每个 storyboard shot 指向一个 beat。这意味着修改对白（改 `beat.content`）不会影响分镜调度，修改相机位置（改 `shot.camera`）不会影响台词内容。两个维度独立编辑，通过 ID 引用维持一致性。

### 7.2 15 条校验规则

`validation.py` 的 `validate_script_yaml_payload` 实现了 15 条校验规则，覆盖四个层面：

**结构完整性**（4 条）：
1. 必须包含 8 个顶层字段
2. `schema_version` 必须为 `"2.0"`
3. `scenes` 必须是非空数组
4. `storyboard` 必须是对象，`storyboard.scenes` 必须是非空数组

**ID 唯一性**（3 条）：
5. 角色/location/event ID 不重复
6. scene ID 不重复
7. storyboard scene_id 不重复

**引用完整性**（9 条）：
8. scene.location_id 必须在 location_ids 中存在
9. scene.characters 必须在 character_ids 中存在
10. scene.source_events 必须在 event_ids 中存在
11. beat.speaker_id（dialogue 类型）必须在 character_ids 中存在
12. storyboard.scene_id 必须在 scene_ids 中存在
13. cast.character_id 必须在 character_ids 中存在
14. shot.source_beat_id 必须在对应 scene 的 beat_ids 中存在
15. shot.action.actor 必须在 cast_ids 中存在（即出现在当前 storyboard scene 的 cast 中）

**语义正确性**（旁白检测，贯穿多条规则）：
- 旁白/叙述者/画外音不能作为角色出现在 `characters` 中
- 旁白不能出现在 `scene.characters` 中
- 旁白不能作为 `dialogue.speaker_id`
- 旁白不能出现在 `storyboard.cast` 中
- 旁白不能作为 `shot.action.actor`
- `cast.position` 不能以 `offscreen` 开头
- action 类型的 shot 必须包含 actions/effects/props

### 7.3 错误消息的可读性

校验错误消息不只是 "character_id not found"，而是携带可读标签：

```
"shot_001_001_02 的 action 角色未出现在 cast 中：char_003（林舟 / 别名：舟哥、小舟 / 主角）；
 当前 cast：char_001（苏瑶 / 女主）、char_002（老陈 / 配角）。
 请把该角色加入当前 storyboard scene.cast 并分配站位"
```

这是通过 `_id_labels` 字典在整个校验过程中维护的——每个 ID 都关联了 `character_label(id, name, aliases, role)`、`location_label(id, name)`、`event_label(id, summary)`，错误消息中用标签替代裸 ID。

---

## 8. Pipeline Service：进程内任务编排

`pipeline_service.py` 1060 行实现了完整的项目级任务编排，是后端最核心的"胶水代码"。

### 8.1 状态机与行锁

```python
def _get_project_for_ai_task(session, project_id, owner_id) -> Project:
    project = session.scalar(
        select(Project).where(Project.id == project_id).with_for_update()
    )
```

`with_for_update()` 是 MySQL 行级排他锁——两个并发请求同时尝试启动同一个项目的 AI 任务时，一个拿到锁，另一个等待或超时。

### 8.2 任务互斥检查

```python
running_ai_run = _running_ai_run_for_task_or_raise(session, project, "chapter_analysis")
if running_ai_run:
    return PreparedAITaskJob(..., should_start=False)
```

如果项目中已有 `status=running` 的 AI 任务，新的启动请求直接返回**当前任务的响应**（含 AI 运行 ID），前端可以继续轮询——而不是启动第二个任务。

### 8.3 增量处理

章节分析和故事元素抽取都支持增量处理——只处理"缺失结果"的章节。核心逻辑：

```python
analyses_by_chapter_id = _chapter_analysis_by_chapter_id(session, project.id)
pending_chapters = [ch for ch in chapters if ch.id not in analyses_by_chapter_id]
```

已经分析过的章节直接跳过，只对新增或内容变化的章节执行 AI 分析。章节内容变化时会级联删除对应的分析记录（`_sync_project_chapters` 中的 delete 逻辑），下次分析时自动重新处理。

### 8.4 流式事件推送

`_append_ai_run_stream_event` 是所有 Agent 的流式输出通道。关键设计是**相邻同类事件合并**（避免前端渲染一堆碎片化的小消息）：

```python
if (
    event.get("append")
    and stream
    and stream[-1].get("type") == entry["type"]
    and stream[-1].get("node") == entry["node"]
    and stream[-1].get("title") == entry["title"]
):
    stream[-1]["content"] += entry["content"]  # 合并内容
else:
    stream.append(entry)  # 新条目
```

这意味着单个 LLM token 的增量输出如果属于同一个 node 和 title，会被合并到同一个 stream 条目中，前端渲染为一条持续更新的消息而非每秒数十条的碎片。

### 8.5 启动恢复

```python
def mark_interrupted_ai_runs_on_startup() -> int:
    # 1. 查询所有 status=running 的 AIRun
    # 2. 标记为 failed
    # 3. 对应 project 状态恢复为 {task_type}_failed
```

这个函数在 FastAPI 的 `lifespan` 事件中调用——服务重启时自动清理任何因进程中断而遗留的僵尸任务。

---

## 9. 前端核心模块

### 9.1 故事图谱：D3 Force 的数据建模

`StoryGraphPanel.vue` 把故事元素快照转为 D3 力导向图。数据建模的核心是**四类节点 + 四种边**：

**节点**（按视觉权重排序）：
- 场景卡片（r=6.5, 紫色）——连接最多的角色和地点，是图谱中的"超级枢纽"
- 角色（r=6, 蓝色）——通常是最核心的实体，多数边汇聚于此
- 地点（r=5, 翠绿色）——中等语义密度
- 事件（r=4.5, 琥珀色）——较小但数量最多

**边**（从 `StoryElementsSnapshot` 中推断）：
- `character → event`：事件卡片的 `involved_characters` 引用了角色 ID
- `event → location`：通过场景卡片的间接关联
- `scene → character/location/event`：场景卡片的三类引用

四种力协同：`forceManyBody`（所有节点互相排斥，强度与节点度成正比）、`forceLink`（边连接产生引力）、`forceCollide`（防止节点重叠）、`forceCenter`（整体居中）。

交互层面支持三件事：拖拽节点调整布局（`d3-drag`）、滚轮缩放（`d3-zoom`）、点击节点弹出详情面板（角色全名/别名/描述/关联事件和场景）。图谱不是静态导出图，而是**探索工具**——用户可以在角色→事件→地点→其他角色之间跳转，快速建立对故事结构的直觉。

### 9.2 PixiJS Storyboard 渲染引擎

`pixi-storyboard-renderer.ts` 是画布预演的核心渲染器。它把 YAML 中 storyboard 的 timeline 数组转换为 PixiJS 时间线：

```
storyboard.scenes[0].timeline:
  shot_001: { type: "dialogue", duration_ms: 3000, ... }
  shot_002: { type: "action", duration_ms: 2000, ... }
  shot_003: { type: "narration", duration_ms: 5000, ... }
  ...
```

渲染器按 timeline 顺序绘制 color-coded 时间条（dialogue=蓝色, action=橙色, narration=灰色），长度与 `duration_ms` 成正比。画布底部的角色站位通过 cast 数组初始化——`ensure_cast_member` 工具保证角色不会在对话/动作中出现但不在 cast 中。

### 9.3 YAML 文档模型

`useScriptYamlDocument.ts` 是整个前端的核心 composable。它做四件事：
1. 解析 YAML → JavaScript 对象
2. 运行同构校验（与后端 shared 逻辑一致）→ `scriptValidationIssues`
3. 产生 computed 衍生数据（`scenes`, `storyboardScenes`, `characters`, `locations`, `events`）
4. 被五个消费组件共享：ScriptYamlPanel（编辑）、ScriptResultPanel（预览）、StoryboardPlayerDialog（预演）、StoryGraphPanel（图谱）、Export 按钮

前端校验和后端校验是**独立实现的同构逻辑**——前端提供即时反馈（边编辑边检查），后端在保存时做最终保障。校验规则的实现语言不同（TypeScript vs Python），但检查的约束条件一致。

---

## 10. Chapter 分析的内存量控制

### 10.1 逐章短记忆

Chapter Analyzer 每次只传当前章节正文 + 一个极小的 `memory` 对象：

```python
def _chapter_analysis_memory(previous_analyses):
    return {
        "story_so_far": " ".join(
            str(item.get("summary", "")).strip()
            for item in previous_analyses
            if item.get("summary")
        ),
        "known_characters": _unique_texts(
            character.get("name")
            for analysis in previous_analyses
            for character in analysis.get("characters", [])
            if isinstance(character, dict)
        ),
        "known_locations": _unique_texts(...),
        "continuity_notes": _unique_texts(...),
    }
```

每个维度的数据都是**去重后的短文本列表**，不需要向量检索或 embedding——因为规模够小（几十个角色名 + 几十个位置 + 十几个伏笔），模型完全可以一次性消费。

### 10.2 累计的边界

`story_so_far` 会随着章节增加而线性增长（每章 ~100 字的摘要拼接），当章节数 > 200 时这可能超过 20K 字。ScriptCraft 处理这个边界的方式是：**不在 Chapter Analyzer 层面处理，而依赖 ScriptMemory Updater 在剧本生成阶段做逐章蒸馏**。Chapter Analyzer 只负责单章分析（它不需要知道完整情节），真正需要跨章上下文的阶段是剧本生成——而那一步已经在用 ScriptMemory 的精炼版本。

---

## 11. 工程权衡与遗憾

**做得好的**：

- **LangGraph 图模式选择**：没有一刀切地用 ReAct Agent，而是根据任务复杂度——线性节点用于单次 LLM 调用场景，Tool-based Agent + 校验回路用于需要工具交互的场景。这个选择让简单任务不引入工具调用开销，复杂任务有足够的容错空间
- **局部写入 + 操作日志**：YAML Repair Agent 不输出完整 YAML，只做局部修改——这是确保大型结构化输出可维护性的关键设计
- **逐章流式 + 滚动记忆**：用时间换空间——按章节顺序处理，每步只看精炼上下文，避免了把整本小说塞进上下文窗口
- **15 条校验规则的可读错误消息**：校验不是简单的通过/不通过——错误消息是给用户看的，携带了可操作的修复建议

**可以更好的**：

- **Memory Updater 的蒸馏精度**：当前一章的 6000 字正文被压缩为 ~500 字的记忆，丢失了什么？精确的文件路径不会丢失（它不存在于小说中），但角色之间的**微妙张力**可能会在浓缩中丢失。这在短篇（< 30 章）中不明显，但在长篇（> 200 章）中可能会导致后期的角色行为与前期铺垫不一致
- **校验规则的 TypeScript 同构实现**：前端和后端的校验逻辑是独立实现的两套代码。如果规则更新（例如新增一条校验），需要同步修改两个语言的版本
- **Scene Planner 的不可恢复性**：Planner 的输出如果不合规，当前设计是直接抛错——没有工具回路来修复。这是因为 Planner 输出是其他 Agent 的输入，修复 Planner 错误的复杂度高于让它重新生成
- **规模化测试缺失**：当前只测试了章节数 < 50 的中短篇小说，> 100 章的超长篇（如网文）的连续性和性能特性未经充分验证

---

## 12. 总结

ScriptCraft 的核心技术贡献不在单一模块，而在**几个子系统的组合方式**：

1. **LangGraph 的三种图模式**（线性 / Tool+校验回路 / Tool+校验回路+写入追踪）按任务复杂度分级使用——不是"all tools or nothing"
2. **YAML 正文+调度双层 Schema** 通过 `source_beat_id` 实现两个维度的独立编辑和一致性保证
3. **逐章流式处理 + 记忆蒸馏** 将长文本的上下文管理从"一次性加载"转变为"逐步推进 + 动态浓缩"
4. **三种 Agent 都有校验回路**（Story Merger 3 轮、Fragment Generator 2 轮、YAML Repairer 3 轮）——结构化输出不是 trust-model-once，而是 verify-and-retry
5. **15 条校验规则覆盖结构完整性、ID 唯一性、引用完整性、语义正确性四个层面**，错误消息可操作而非裸 ID

在一个"把 LLM 当作全能黑盒"的时代，ScriptCraft 展示了 Agent 编排的另一个方向：**把 LLM 绑定到一组工具和校验规则中，让它在严格的约束空间内探索**。约束不是自由的敌人，而是可靠性的保证。

---

**技术栈**：Python / FastAPI / LangGraph / LangChain / SQLAlchemy / PyMySQL / Vue 3 / TypeScript / Tailwind CSS v4 / shadcn-vue / PixiJS / D3 Force / GSAP / MySQL
