---
title: "Nova 源码深度拆解：从 Tauri IPC 到 LLM Agent 全链路架构解析"
published: 2026-06-25
description: "深入 Nova 桌面 AI 编码助手的 Rust 后端源码,逐层拆解：工具注册与并发执行引擎、三层上下文压缩与熔断、8 类生命周期 Hook、文件级 MD 记忆系统、权限审批状态机、持久化 Shell 会话、浏览器自动化 IPC 桥接、流式 SSE 规范化与 Provider 适配、Turn Snapshot 恢复、Context Assembler 注入、Plan/Subagent 控制流、Stop Hook 限流等核心子系统的设计细节与代码级权衡。"
tags: [Nova, Tauri, Rust, AI Agent, 源码分析, 上下文压缩, 工具系统, Hook 架构, 记忆系统, 权限管理, SSE 流式处理]
category: 项目
draft: false
---

Nova 是一个基于 Tauri 2 (Rust) + Vue 3 的本地桌面 AI 编码助手。和 Claude Code、Cursor 这类云侧 Agent 不同，Nova 的一切执行都在本地完成——文件操作走 `apply_patch` 直接 diff-apply，Shell 命令通过持久化 Tokio Child 进程执行，浏览器自动化由 Tauri Webview 渲染并标注交互元素。后端 7 个核心 Rust 模块（`llm/tools`、`llm/query`、`llm/services`、`llm/providers`、`command`、`prompt`、`llm/utils`）组成了完整的 Agent 运行时。

本文从源码出发，按数据流顺序逐层拆解。

---

## 1. 工具系统：build.rs 编译期注册 + 读写分批并发 + 三级权限拦截

### 1.1 编译期自动注册

传统 Agent 框架的工具注册通常是手写清单。Nova 的做法是**编译期扫描**：在 `build.rs` 中遍历 `src/llm/tools/*/` 目录，对每个含 `pub(crate) fn registrations()` 的 `mod.rs`，生成一个 `builtin_tool_registry.rs`，最终通过 `include!` 宏嵌入运行时：

```rust
// tools/mod.rs（这段代码不直接写工具名）
include!(concat!(env!("OUT_DIR"), "/builtin_tool_registry.rs"));
```

每个工具模块（如 `BashTool/mod.rs`）暴露：

```rust
pub(crate) fn registrations() -> Vec<ToolRegistration>
```

返回的 `ToolRegistration` 结构体包含五个关键元数据：

```rust
struct ToolRegistration {
    tool: fn() -> Tool,              // name/description/schema 静态定义
    execute_with_app: AppExecuteFn,  // 唯一执行入口，注入 AppHandle
    permission: Option<PermissionFn>, // 自定义权限描述函数
    read_only: bool,                 // 只读 -> 批并并发
    role: ToolRegistrationRole,      // General / SubagentStart / SubagentStop
}
```

新增工具只需在 `tools/` 下新建目录 + 实现 `registrations()`，`build.rs` 在下次编译时自动发现——**不修改中心模块一行代码**。

### 1.2 26 个工具清单

| 类别 | 工具 |
|------|------|
| Shell | `execute_bash`, `execute_powershell`, `reset_shell_session` |
| 文件 | `apply_patch`（唯一写工具，diff-apply 而非覆盖写入） |
| 搜索 | `web_search`, `web_fetch`, `search_code`（内置 rg） |
| 浏览器 | `navigate`, `click`, `type_text`, `take_snapshot`, `reset_browser` |
| 桌面 | `computer_use`（截图+键鼠操作） |
| 规划 | `enter_plan_mode`, `exit_plan_mode`, `plan_for_approval`, `set_goal` |
| 知识 | `search_rag`, `remember_global_memory`, `load_skill` |
| MCP | `mcp_auth`, `list_mcp_resources`, `read_mcp_resource` |
| 任务 | `cron_create`, `cron_delete`, `cron_list` |
| 工具 | `tool_search`, `ask_user`, `config`, `sleep` |

### 1.3 读写分批并发执行引擎

核心函数 `execute_tool_calls_with_app` 的并发策略是整条链路里最精巧的部分：

```rust
pub async fn execute_tool_calls_with_app(app, conversation_id, calls) -> Vec<ToolCallResult> {
    let mut read_only_batch = Vec::new();
    for call in calls {
        if is_read_only_tool(&call.name) {
            read_only_batch.push(call);  // 只读工具先攒着
            continue;
        }
        flush_read_only_batch(&mut read_only_batch).await; // 遇到写工具，先 flush
        results.push(execute_single_tool_call(call).await); // 写工具单独顺序执行
    }
    flush_read_only_batch(&mut read_only_batch).await; // 尾部 flush
}
```

只读工具通过 `execute_read_only_batch` 汇入 Tokio `JoinSet`，默认最多 10 个并发（`NOVA_MAX_TOOL_USE_CONCURRENCY` 可调）。写工具出现时**先等待所有在读并发完成**，保证副作用执行顺序与模型输出顺序一致。取消信号贯穿全程——每个 `while` 循环迭代检查 `CancellationToken`，已取消的未执行工具全部标记为 `cancelled`。

### 1.4 单工具执行的全部拦截点

`execute_single_tool_call` 的流程是：SubagentStart Hook → PreToolUse Hook（deny-list / override error）→ JSON Schema 校验 → 权限检查 → 实际执行 → PostToolUse Hook（pattern match 阻断）→ PostToolUseFailure Hook → SubagentStop Hook。八个拦截点，任意一个都能注入上下文、拒绝执行或覆盖错误信息。

---

## 2. 上下文压缩：Micro/Full/Context Editing 三层 + 熔断器

`services/compact/` 约 1000 行的实现，是对 Claude Code AutoCompact 策略的独立重写。

### 2.1 Token 估算：字符分桶 + JSON 结构

Nova 不依赖 tiktoken-rs 做实时估算（只在关键路径使用了 `tiktoken-rs` crate 但 Compact 模块自己实现了一套），而是用一套经验规则：

```rust
fn estimate_text_tokens(text: &str) -> i64 {
    // CJK 字符: ~1 token/字
    // ASCII 字母数字: 4 字符 ≈ 1 token
    // 标点符号: 2 字符 ≈ 1 token
    // 空白: 16 字符 ≈ 1 token
}
fn estimate_json_tokens(value: &Value) -> i64 {
    // 递归展开 JSON 树，数组/对象加结构开销
}
fn estimate_image_tokens_from_dimensions(width, height) -> i64 {
    // ceil(width * height / 750).clamp(256, 8192)
}
```

关键不是绝对精度，而是**相对趋势追踪**。只要"true token ≈ estimated × C"成立，阈值就不会误触发。

### 2.2 策略决策：纯 Token 比例触发

```rust
fn decide_compact_strategy(messages, window_tokens) -> CompactDecision {
    let estimated = estimate_message_tokens(messages);
    let micro_threshold = (window_tokens * 80) / 100;
    let full_threshold  = window_tokens - 13_000; // 对标 CC 的 windowSize - 13k
    // 决策: Full > Micro > None
}
```

- **Micro**（token ≥ 80% 窗口）：本地 Truncate，不调模型。JSON 深度 > 3 截断，数组 > 12 元素截断，纯文本 1200 字符截断（头 60% + 尾 30%）- **不消耗 API 调用**
- **Full**（token ≥ 窗口 - 13k）：先做 Micro，再调模型生成摘要压缩。保留最近 10 条消息，更早的替换为一条 `[Auto Compact Summary]` User 消息
- **Reactive**（模型返回 `prompt_too_long`）：尝试 Full Compact 重试一次；若无效则降级截断到最近 8 条消息

### 2.3 Context Editing：主循环内的轻量清理

`apply_tool_result_context_editing` 在主循环每次 Provider 请求前执行。触发条件：已用 token > 50% 窗口。清理逻辑：保留最近 3 组 tool_use/tool_result，更早的大型工具结果替换为 `[tool_result removed by context editing to save prompt space]` 占位符。

### 2.4 熔断器

`state.rs` 使用 `Mutex<HashMap<conversation_id, u8>>` 记录每个会话的连续 Full Compact 失败次数。达到 3 次后打开熔断——**跳过所有后续 Full Compact**，同时向前端发送 `backend-error` 通知。成功后重置计数器。

---

## 3. Hook 系统：8 类生命周期拦截 + 环境变量驱动

Hook 系统的核心设计原则是**零代码侵入**——所有 Hook 配置通过环境变量注入，Hook 逻辑不嵌入任何工具代码。

### 3.1 全部 Hook 类型

| Hook | 触发点 | 可操作项 |
|------|--------|----------|
| `SessionStart` | 会话首轮 | inject context, override error |
| `UserPromptSubmit` | 用户发送消息后 | inject context |
| **`PreToolUse`** | 工具执行前 | inject context, **deny-list 拒绝工具**, override error |
| **`PostToolUse`** | 工具成功后 | inject context, **pattern match 阻断续跑** |
| **`PostToolUseFailure`** | 工具失败后 | inject context, **stop on error** |
| `PreCompact` | 压缩前 | inject context |
| `PostCompact` | 压缩后 | inject context, override error |
| `SubagentStart` / `SubagentStop` | 子代理启停 | inject context（含子代理名） |
| **`Stop`** | 回合停止前 | pattern match 阻断, inject context, **限制 assistant 消息数** |
| `SessionEnd` | 会话结束时 | 覆盖 stop_reason |
| `Error` | 异常时 | 覆盖错误信息 |

### 3.2 Hook 配置机制

Hook 配置从 `Settings.hook_env: HashMap<String, String>` 读取，本质是环境变量 key-value 对。支持三种读取模式：

```rust
impl HookConfig {
    fn value(key) -> Option<&str>      // 纯文本
    fn truthy(key) -> bool             // "1"/"true"/"yes"/"on"
    fn csv_lower_list(key) -> Vec<String> // 逗号分隔列表
}
```

**实际用法示例**：

```bash
# 阻止 Shell 执行
NOVA_PRE_TOOL_DENY_TOOLS=execute_bash,execute_powershell

# 匹配输出中的特定模式后停止续跑
NOVA_POST_TOOL_BLOCK_PATTERN=permission denied

# 工具失败时立即终止
NOVA_POST_TOOL_FAILURE_STOP=true

# Stop Hook: 最多 50 条 assistant 消息
NOVA_STOP_HOOK_MAX_ASSISTANT_MESSAGES=50

# Stop Hook: assistant 输出匹配模式则终止
NOVA_STOP_HOOK_BLOCK_PATTERN=I cannot complete this task
```

### 3.3 PreToolUse deny-list 的实现

```rust
// hooks/tool_flow.rs
let deny_list = config.csv_lower_list("NOVA_PRE_TOOL_DENY_TOOLS");
let tool_lower = tool_name.to_ascii_lowercase();
if deny_list.iter().any(|name| name == &tool_lower) {
    out.override_error = Some(format!("Blocked by PreToolUse hook: '{}' is deny-listed", tool_name));
}
```

这完全在 `execute_single_tool_call` 的 PreHook 阶段执行，工具函数本身对此一无所知。

### 3.4 Stop Hook 的独特能力

Stop Hook 是唯一能**检查 assistant 历史输出文本**的 Hook：

```rust
// hooks/stop.rs
let assistant_text = latest_assistant_text(messages); // 取最新的 assistant 完整输出
if assistant_text.contains(block_pattern) {
    out.prevent_continuation = true;
}
```

同时支持 `NOVA_STOP_HOOK_MAX_ASSISTANT_MESSAGES` 限制轮次上限，防止 Agent 无限循环。

---

## 4. 记忆系统：文件级 MD 持久化 + 语义去重 + 冲突检测

记忆系统放在 `services/memory_dir/`，680 行纯逻辑，不依赖外部向量数据库。

### 4.1 存储格式

每个记忆是 `%APPDATA%/memory/{kind}/` 下的独立 `.md` 文件：

```markdown
---
id: 1234567890
kind: preference
source: user
hits: 5
created_at: 1750000000
updated_at: 1750000000
---
代码风格使用 4 空格缩进
```

`MEMORY.md` 索引文件列出全部条目。三类记忆：`preference`（偏好）、`rule`（规则）、`fact`（事实）。

### 4.2 upsert 去重：semantic_memory_key

核心去重逻辑不是字符串相等，而是 `semantic_memory_key` 归一化：

```rust
fn semantic_memory_key(kind, content) -> String {
    let stripped = strip_memory_prefixes(content);
    // 剥离口语前缀: "记住 " "以后 " "请 " "from now on " "prefer " 等
    if stripped.is_empty() { normalize_for_match(content) } else { stripped }
}
fn normalize_for_match(raw) -> String {
    // 去标点、去控制符、小写化、多余空格归一
}
```

upsert 时若已有记录匹配同一 `semantic_key`，则原地更新 `hits`、`updated_at`，不创建新文件。

### 4.3 冲突维度检测

对 `preference`/`rule` 类型，upsert 后触发 `purge_conflicting_records`：

```rust
fn conflict_dimensions(kind, content) -> HashSet<String> {
    // 分四个维度做关键词检测：
    //   language: 中文/英文/english/chinese
    //   verbosity: 简洁/详细/concise/detailed
    //   fallback_policy: 不要兜底/no fallback
    //   tool_behavior: 直接改/edit directly
    //   statement: 归一化后的首句（≥6 字符）作为语义签名
}
```

发现维度重叠的旧记录时自动删除，确保同维度不会积压冲突规则。

### 4.4 检索评分

```rust
fn score_record(query_tokens, record) -> i64 {
    // 基础分: rule=24, preference=18, fact=0
    // 匹配分: n-gram(>=3) 命中 +10, bigram 命中 +4
    // rule + preference 始终入选（按评分排序），fact 只在 token 匹配时返回
}
```

Query token 化使用 2-gram/3-gram 滑动窗口 + 按字母数字边界分词，避免小词误匹配。

---

## 5. 权限审批状态机：会话级缓存 + 签名去重 + 异步等待

`utils/permissions/mod.rs` 830 行实现了一个完整的三态权限状态机。

### 5.1 三级判决优先级

```
DenySession > AllowSession > AllowOnce (一次性消费) > AskUser
```

每次 `enforce_tool_permission` 调用：
1. 检查 `NOVA_ALLOW_UNSAFE_TOOLS` 环境变量（调试开关，全局放行）
2. 检查 `deny_session` 集合（被拒签名直接 Deny）
3. 检查 `allow_session` 集合（会话缓存直接 Allow）
4. 检查 `allow_once` 集合（命中后**消费移除**，只生效一次）
5. 生成危险操作描述 → AskUser

### 5.2 危险操作检测

**Shell 命令**：
- 单词级：命中 `rm`/`del`/`remove-item` 即拦截
- 模式级：`rm -rf`、`git reset --hard`、`shutdown /s` 等 18 条高危片段
- 路径级：系统目录（`/etc`、`/boot`、`C:\Windows` 等）和敏感目录（`.ssh`、`.aws`、`.gnupg` 等）

**文件操作**：对 `apply_patch` 工具检测目标路径是否命中受保护前缀或敏感标记。

**MCP 工具**：对 server/tool 名做模糊匹配（`bash`/`shell`/`file`/`write` 等），命中则按对应规则检测参数。

### 5.3 AskUser 流程

```rust
PermissionEnforcement::AskUser { request_id, payload } => {
    // 1. 生成 JSON 审批 payload（含操作描述 + 风险提示）
    // 2. 注册 oneshot channel waiter
    // 3. 前端弹出审批对话框，返回 allow_once / allow_session / deny_session
    // 4. 150ms 轮询等待，同时监听取消信号（超时 15 分钟）
    // 5. 决策写入会话状态，notify waiter
}
```

同一个操作签名在会话内只弹一次审批（`pending_by_signature` 去重），已审批的请求 ID 缓存 60 分钟。

---

## 6. 持久化 Shell 会话：CPW Bash/PowerShell + CWD 保持 + 超时重启

`shell_sessions/mod.rs` 710 行实现了一个关键的差异化能力：**每个对话拥有独立的持久化 Shell 进程**。

### 6.1 架构

- 每个 `conversation_id` 映射一个 `Arc<SessionHandle>`（`AsyncMutex<ShellSession>`）
- `ShellSession` 封装：`tokio::process::Child`（持久化的 pwsh/bash 进程）、`ChildStdin`、`mpsc::UnboundedReceiver<StreamEvent>`
- 进程启动时注入 bootstrap 脚本（设置 UTF-8 编码、关闭 ANSI 颜色、设置 ProgressPreference）
- CWD 在每条命令执行后通过 marker 协议回传并持久化

### 6.2 Marker 协议

每条命令执行完后在 stdout/stderr 注入 marker：

```
__NOVA_CMD_END__|<command_id>|<exit_code>|<cwd_base64>|<timed_out>
```

解析到当前命令的 marker 后，同时更新 `last_known_cwd` 和判定 stdout/stderr 流结束。

### 6.3 生命周期管理

- **前台命令**：最多 5 分钟超时（可调至 30 分钟），超时后**重启整个 Shell Session**（保证状态干净）
- **后台命令**：30 秒收集输出后放行，PID 记入 `background_pids`
- **取消**：`CancellationToken` 触发后重启 Session
- **崩溃恢复**：`ensure_session_alive` 每次执行前检测进程存活性，自动重建
- **Reset**：`restart_session` 杀进程树（`taskkill /T /F`）→ 清理后台 PID → 重建 → 恢复 CWD
- **关闭**：`close_session` 杀进程树 + 移除注册表

---

## 7. Nova Browser：Tauri Webview IPC 桥接 + Oneshot 请求-响应

`browser_sessions/mod.rs` 320 行实现了一个**非 HTTP 的浏览器自动化通道**。

### 7.1 通信模型

不是 CDP/WebDriver，而是：
- 后端 emit `nova-browser-command` 事件 → 前端 Browser Webview 接收
- 前端执行完成后 emit `nova-browser-result` → 后端 `complete_command` 解析
- 后端侧通过 `oneshot::Sender/Receiver` 做异步等待

### 7.2 请求追踪

```rust
fn next_request_id() -> String {
    // "browser-{timestamp_ms}-{counter}"（AtomicU64 自增）
}
```

`pending: HashMap<request_id, oneshot::Sender<Value>>` 存储等待中的请求，超时后移除并返回 JSON error。

### 7.3 会话感知

- 每个 browser session 绑定 `conversation_id`
- 启动时通过 `nova-browser-open-request` 事件通知前端打开浏览器窗口
- 等待 4 秒（40 次 × 100ms）检测 webview 存活性
- 页面 URL 更新通过 `update_session_url` 实时追踪

---

## 8. 流式引擎与 Provider 适配

### 8.1 协议无关的 Delta 抽象

`stream_runner.rs` 840 行定义了一套统一的流式解析框架：

```rust
pub enum Delta {
    Text(String),           Reasoning(String),
    ToolStart { id, name }, ToolArgsDelta { id, args },
    ToolsReady(Vec<ReadyToolCall>),
    Usage { input, output, cache_read, cache_creation },
    Stop { reason: Option<String> },
    ThinkingBlock { thinking, signature },
    Error(String),
}
```

`StreamParser` trait 只有两个方法：

```rust
trait StreamParser {
    fn parse_event(&mut self, data: &str) -> Result<Vec<Delta>, String>;
    fn flush(&mut self) -> Vec<Delta>; // SSE 流结束后清理残余状态
}
```

`run_streaming` 做所有通用工作：SSE 字节缓冲、取消监控、Delta → `chat-stream` 事件 emit、工具执行、`AssistantOutputBuilder` 维护输出顺序。添加新 Provider 只需实现 `StreamParser` trait。

### 8.2 AnthropicAdapter 的关键细节

`AnthropicAdapter` 维护累加状态：
- `current_tool_name` / `current_tool_input`：增量累积 tool_use 参数 JSON
- `current_thinking` / `current_sig`：增量累积 thinking block
- `pending_tool_calls`：`content_block_stop` 事件触发时，将已完成累积的 tool_use 转为 `ToolsReady`

### 8.3 取消时的部分内容保留

取消事件通过 `tokio::select!` 上的 `cancel_token.cancelled()` 分支处理。取消时不是丢弃已收到的输出，而是通过 `build_partial_cancelled_messages` 将已累积的 `AssistantOutputBuilder` 状态打包返回——**确保 UI 中已经显示的内容和 turn_snapshot 保持一致**。

---

## 9. 上下文装配器：三段注入 + 幂等 Marker

`utils/context_assembler.rs` 在每轮请求前执行：

```
assembled_messages = [Global Memory] + (optional)[Session Restore] + working_messages + (optional)[AssemblerContext]
```

- **Global Memory** 前置：从记忆系统检索相关偏好/规则/事实，插入 `[Global Memory]` 标记的消息到最前面
- **Session Restore**（默认关闭）：从 compact 记录恢复会话摘要上下文
- **Env Context**（默认关闭）：读取 `NOVA_ASSEMBLER_CONTEXT` 环境变量

所有注入通过 marker 字符串做幂等检查（`has_global_memory_marker` / `has_session_restore_marker`），同一轮不重复塞入。重要的是：**assembler 只做临时注入**——保存 turn_snapshot 前通过 `strip_injected_context` 移除所有动态上下文（14 个 marker），确保持久化快照只含真实对话内容。

---

## 10. Turn Snapshot 恢复：可信上下文保证

`send_chat_message` 的上下文恢复策略是 Nova 区别于其他 Agent 的关键设计：

```
首轮: snapshot 为空 → 用前端当前输入启动 → OK
非首轮: load_turn_snapshot → strip_injected_context → + 本轮新输入 → OK
非首轮且 snapshot 缺失: 直接 Err，拒绝用前端 UI 历史兜底
```

这意味着**后端不信任前端传来的历史消息**用于多轮上下文。前端 UI 历史（`conversation_messages` 表）只用于显示；模型上下文来自上一轮保存的完整 snapshot（含 tool_use/tool_result blocks）。这是一个极为保守但安全的设计——即使前端数据不同步或损坏，模型仍看到正确上下文。

Snapshot 保存时机：每轮正常结束后保存 strip 后的 `current_messages`；错误路径做 best-effort partial 保存。

---

## 11. Plan Mode 与 Subagent 特殊语义

### 11.1 Plan 控制流

`EnterPlanModeTool` / `ExitPlanModeTool` / `PlanForApprovalTool` 三种工具对应的角色不是简单的工具调用，而是通过 `ToolRegistrationRole` 标记特殊语义：

- `SubagentStart`：`is_subagent_start_tool` 匹配时，在工具执行**前**就触发 `run_subagent_start_hooks`
- `SubagentStop`：`is_subagent_stop_tool` 匹配时，在工具执行**后**触发 `run_subagent_stop_hooks`

这样 Hook 系统可以在子代理启停时注入上下文或阻断执行，而无需工具代码感知 Hook 的存在。

### 11.2 Goal 跟踪

`GoalTool` 让模型在任意时刻声明当前目标，这些目标文本被持久化并通过 `conversation_meta` 表跟踪。与其他 Agent 的"任务列表"不同，Nova 的 Goal 是**模型自主声明的**，不是预设模板。

---

## 12. 对话级隔离全景

| 资源 | 隔离方式 |
|------|----------|
| Shell Session | `SessionKey(conversation_id) → ShellState`，独立 Tokio Child 进程 |
| Browser | `BrowserSession { label, current_url }`，独立 webview 窗口 |
| CWD | Shell Session 的 `last_known_cwd` 持久化 |
| 工作区 | 每个对话的可选 `working_dir` |
| 权限 | `ConversationPermissionState`（allow/deny 集合按 conversation 分区） |
| 取消 | `CancellationToken` 按 conversation_id 分组 |
| Live Turn | `HashMap<conversation_id, LiveTurnStatus>` 进程内分组 |
| 记忆 | 全局共享（设计意图是跨会话复用偏好/规则） |

---

## 13. 数据流全景：一次用户输入的完整旅程

```
用户输入 (Vue ChatInput)
  → Tauri IPC invoke("send_chat_message", { conversationId, messages })
  → command::history::add_user_message (持久化 UI 历史)
  → llm::query::send_chat_message
       ├─ latest_user_query_text → RAG query
       ├─ run_user_prompt_submit_hooks → [UserPromptSubmit] context
       ├─ run_session_start_hooks (首轮) → [SessionStart] context
       ├─ load_turn_snapshot → strip_injected_context → +本轮新消息
       ├─ assemble_messages_for_turn → [Global Memory] + working
       ├─ compact_messages_for_turn → Micro/Full 压缩
       ├─ run_pre_compact_hooks / run_post_compact_hooks
       ├─ build_session_rag_context_message → [Session RAG Context]
       ├─ build_mcp_server_context_message → [MCP Server Catalog]
       └─ LLM 主循环
            ├─ apply_tool_result_context_editing
            ├─ provider.send_request → stream_runner → SSE parse → emit chat-stream
            ├─ 工具执行: execute_tool_calls_with_app → 读写分批 → Hook 拦截
            ├─ 工具结果回灌 → loop continue
            └─ 完成: run_stop_hooks → run_session_end_hooks
                     → strip_injected_context → save_turn_snapshot
                     → emit stop event
```

---

## 14. 代码规模与模块边界

| 模块 | 行数 | 职责 |
|------|------|------|
| `llm/query.rs` | 1,111 | Agent 主循环：上下文恢复 → 装配 → 压缩 → Provider → 工具 → Snapshot |
| `llm/services/compact/` | 999 | 三层压缩 + Token 估算 + Context Editing + 熔断 |
| `llm/providers/stream_runner.rs` | 834 | 协议无关 SSE 循环 + Delta 抽象 |
| `llm/utils/permissions/` | 832 | 权限状态机 + 危险操作检测 + AskUser 异步等待 |
| `llm/tools/mod.rs` | 731 | 工具注册 + 读写分批并发 + 单工具执行全流程 |
| `llm/services/shell_sessions/` | 714 | 持久化 Shell 进程 + Marker 协议 + 生命周期 |
| `llm/services/memory_dir/` | 681 | MD 文件记忆 + 语义去重 + 冲突检测 + 检索评分 |
| `llm/providers/adapters/anthropic/` | 575 | Anthropic 协议适配 + SSE Delta 解析 |
| `llm/services/hooks/` + `stop.rs` | ~400 | 8 类 Hook + 环境变量驱动配置 |

---

## 15. 工程层面的关键设计决策

**值得借鉴的**：

1. **编译期工具注册**：`build.rs` 自动扫描 + `include!` 嵌入，新增工具零修改中心模块。这种模式也适用于任何需要插件化扩展的 Rust 项目。

2. **读写分批并发**：只读工具汇入 JoinSet 并发、写工具打断 flush、取消信号贯穿——在正确性和吞吐之间找到了很好的平衡。

3. **Hook 环境变量驱动**：零侵入，所有拦截在中心执行函数中完成，Hook 能力与工具代码完全解耦。

4. **Turn Snapshot 信任模型**：后端不信任前端历史——这是一种保守但正确的选择，避免了 UI 状态与模型上下文不同步的诡异 bug。

5. **权限签名去重**：同一操作签名会话内只弹一次审批框，`allow_once` 消费后自毁——减少了用户的审批疲劳。

**可能的问题**：

1. **Token 估算误差**：字符分桶对中英混合文本的估算偏差可能较大，`estimate_tokens_for_serializable` 在多轮累积后误差会滚雪球。

2. **Hook 只读环境变量**：无法在运行时动态调整 Hook 配置——修改需要重启应用或至少重建 `HookConfig`。

3. **权限 AskUser 的 15 分钟 TTL**：如果用户离开桌面，超时后 Deny 的默认行为可能导致 Agent 卡住。

4. **Full Compact 摘要质量依赖模型能力**：摘要丢失关键信息（如精确文件路径、错误堆栈）时，后续对话会降级——Compact State 熔断机制只是防止连续 API 调用失败，不防止语义丢失。

---

**技术栈**：Tauri 2 / Rust 2021 / Tokio / SQLx / reqwest / Vue 3.5 / Pinia / TailwindCSS 4 / xterm / ECharts / TypeScript 5.6 / Vite 6
