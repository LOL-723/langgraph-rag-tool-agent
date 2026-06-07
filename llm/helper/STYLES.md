# LangGraph Code Style Guide

本文档约束本项目 LangGraph 相关代码的组织格式和风格。以后新增或修改 agent、graph、node、route、verifier、tool 调用等代码时，应优先遵守本文档，以提高可读性并降低代码腐化风险。

## 适用范围

- 主要适用于 `llm/langgraph.py`。
- 调用 LangGraph 的入口代码，例如 `llm/client.py` 中的 `run_langgraph()`、结果格式化、初始 state 构造，也应保持与本文档一致。
- 工具注册、RAG 服务、API 路由不需要强行搬入 LangGraph 文件；只在图节点中通过清晰边界调用它们。

## 文件组织顺序

LangGraph 文件应按以下顺序组织，避免把节点、私有工具函数、图构建逻辑混在一起：

1. 标准库 import。
2. 第三方库 import。
3. 项目内 import。
4. 类型别名与 `TypedDict` state 定义。
5. 常量、重试预算、prompt 模板。
6. 通用 state 工具函数，例如 `add_log()`。
7. 节点函数，例如 `router_node()`、`rag_node()`、`answer_node()`。
8. 路由决策函数，例如 `route_decision()`、`verifier_decision()`。
9. `build_graph()`，集中声明所有节点和边。
10. 私有辅助函数，以 `_` 开头，例如 `_verify_answer()`、`_chat_completion()`。

除非有明确收益，不要在文件中间插入新的大段逻辑。新增能力时优先放入对应分区。

## 命名规则

- 节点函数统一使用 `*_node` 后缀，例如 `router_node`、`tool_executor_node`。
- 条件边决策函数使用 `*_decision` 后缀，返回值必须是受 `Literal` 约束的节点名或终止标识。
- 私有辅助函数使用 `_snake_case`，只暴露给本模块内部使用。
- State 字段使用 `snake_case`，语义要稳定，不要用临时缩写。
- Graph 节点名字符串必须和节点函数名保持一致，例如 `"router_node"` 对应 `router_node`。
- Route 名、状态名、终止状态等有限集合必须用 `Literal` 类型表达。

## State 设计

- 所有跨节点传递的数据必须先写入 `LangGraphState`。
- `LangGraphState` 使用 `TypedDict, total=False`，允许节点只返回自己更新的字段。
- 必填输入字段应在调用入口构造，例如 `question`、`system_prompt`、`use_rag`、计数器、`logs`。
- 节点返回增量 state，不要原地修改传入的 `state`。
- 列表字段更新时返回新列表，例如 `logs + [log_item]`，避免隐藏副作用。
- 新增重试或校验字段时，同时补齐：
  - `LangGraphState` 字段定义。
  - 初始 state。
  - 相关最大重试常量。
  - verifier 或 route 逻辑。
  - 输出格式化中需要暴露的元数据。

## 节点函数规范

每个节点函数应保持单一职责：

- `router_node` 只负责决定 route，不执行 RAG、工具或回答生成。
- `rag_node` 只负责检索、重排和写入检索上下文。
- `tool_selector_node` 只负责选择工具调用。
- `tool_executor_node` 只负责执行已选择工具并记录结果。
- `answer_node` 只负责生成回答。
- `verifier_node` 只负责验证回答、决定下一步和写入终止状态。

节点函数的推荐结构：

1. 从 `state` 提取输入。
2. 调用外部服务或私有辅助函数完成本节点职责。
3. 构造并返回 `LangGraphState` 增量。
4. 通过 `add_log()` 记录关键行为。

节点函数不应直接构建 graph，不应直接格式化 API 响应，不应处理 HTTP 异常。

## Graph 构建规范

`build_graph()` 是唯一声明拓扑的位置，保持可扫读：

- 先创建 `StateGraph(LangGraphState)`。
- 再连续声明所有 `add_node()`。
- 然后声明 `START` 到入口节点。
- 再声明条件边。
- 最后声明普通边和终止边。
- 函数末尾只做 `return graph_builder.compile()`。

条件边映射必须显式列出所有可能返回值，不要依赖隐式约定。节点名字符串不要散落在辅助函数内部；如果节点数量继续增加，应提取节点名常量。

## Prompt 与模型调用

- Prompt 模板集中放在文件顶部常量区，使用全大写命名，例如 `VERIFIER_PROMPT`。
- Prompt 应明确返回格式，尤其是 JSON 输出必须要求模型返回单个合法 JSON object。
- 所有模型 JSON 输出必须经过 `json.loads()` 和类型检查。
- 解析失败时返回保守默认值，不要让无效模型输出破坏图执行。
- `_openai_client()` 统一封装客户端创建，节点函数不直接实例化客户端。

## 日志规范

- 所有节点都应写入 `logs`。
- 日志至少包含：
  - `node`
  - `message`
  - 必要的 `extra` 元数据
- `message` 使用稳定的英文短语，便于前端或日志系统筛选。
- 不要把大文本、完整 prompt、完整回答反复写入 logs；只记录诊断需要的摘要和计数。

## 错误和重试

- 重试预算集中定义为 `MAX_*_RETRIES` 常量。
- verifier 决定失败后的下一步，其他节点不应自行跳转。
- 达到重试上限时必须设置：
  - `verifier_next`
  - `end_status`
  - 可供调用方展示的 `answer` 或失败原因
- 重试计数器必须是 state 字段，不能依赖模块级可变变量。

## 与调用侧的边界

- `llm/client.py` 负责校验用户输入、上传文件、构造初始 state、调用 graph、格式化结果。
- `llm/langgraph.py` 负责图执行和中间状态，不负责 HTTP 语义。
- API 路由只捕获异常并返回响应，不直接操作 graph state。
- RAG、tool、client 等外部能力通过清晰函数调用接入，不把实现细节复制到 graph 文件。
- `rag_node` 只服务 LangGraph workflow；Agent 工作流需要 RAG 时只能调用 `llm/rag_tools.py` 暴露的 tool 函数，不应反向调用 `llm/langgraph.py`。

## 编码与可读性

- 所有源码和文档使用 UTF-8。
- 中文字符串必须保持可读，不能提交乱码文本。
- 类型标注要覆盖函数参数和返回值，尤其是 node、decision、helper 函数。
- 单个函数过长时优先抽取私有辅助函数；不要在节点函数里堆叠复杂分支。
- 注释只解释不明显的业务约束或安全原因，不解释显而易见的赋值。

## 新增 LangGraph 功能检查清单

新增节点、route、工具路径或验证路径前，至少检查：

- 是否已在 `LangGraphState` 中定义所有跨节点字段。
- 是否已在 `build_graph()` 中注册节点和边。
- 条件边返回值是否和映射完全一致。
- 是否补充了日志。
- 是否补充了重试预算或明确不需要重试。
- 调用侧初始 state 是否需要新增默认值。
- 输出格式化是否需要暴露新的元数据。
- 模型 JSON 输出是否有解析失败兜底。
- 文件编码是否仍为 UTF-8，中文是否可读。
