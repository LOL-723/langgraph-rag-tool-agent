# Annotation Style Skill

本文档用于约束以后生成代码时的注释风格。需要写注释时，优先使用这里定义的风格：注释要贴近代码、解释当前业务里的原因和数据流，帮助读者理解“为什么这样写”和“这一行在做什么转换”。

## 风格定位

注释面向正在理解代码的人，而不是只面向熟悉框架的人。写法可以直接、口语化，但必须准确。

推荐注释覆盖：

- 入口函数的职责和返回值。
- 外部库 API 的参数形状和返回值形状。
- 数据结构转换，例如从 `X` 变成 `[X]`，从多个列表 `zip` 成元组。
- fallback 分支为什么存在。
- 排序、过滤、截断等关键业务逻辑。
- 容易误解的 Python 语法，例如 `or [[]]`、`lambda item: item[1]`。

不推荐注释覆盖：

- 显而易见的赋值，例如 `count = 0`。
- 和当前业务无关的泛泛解释。
- 重复函数名已经说明清楚的内容。
- 大段理论说明，尤其是会打断代码阅读节奏的段落。

## 注释位置

- 函数入口注释放在 `def` 上方，用一行说明该函数在流程中的角色。
- 对单行代码的解释放在代码上方；只有非常短的补充才允许放在行尾。
- 多行逻辑前写一到三行注释，说明这段逻辑处理的输入、输出或原因。
- 注释应贴近被解释的代码，不要集中堆在函数开头。

示例：

```python
# rag_node召回入口，返回从向量库或本地文件中召回的片段
def retrieve_with_mode(...):
    ...
```

```python
# Chroma支持一次查询多个问题，所以这里要把单个问题向量包成列表
query_kwargs = {
    "query_embeddings": [query_embedding],
    "n_results": top_k,
}
```

## 语言风格

- 使用中文注释。
- 语气直接，尽量像给自己或同项目同事解释代码。
- 可以使用“这里”“所以”“由于”“避免”等连接词，把原因讲清楚。
- 注释中可以保留英文术语，例如 `RAG`、`Chroma`、`query_embeddings`、`zip`、`lambda`。
- 中文和英文、代码标识符之间适当留空，提高可读性。

推荐：

```python
# query_embedding 自身就是问题向量，但 Chroma 支持一次查询多个问题，
# 所以 query_embeddings 需要传向量列表，从 [X] 变成 [[X]]
```

不推荐：

```python
# 查询
```

```python
# This queries Chroma.
```

## 解释颗粒度

注释应解释到读者能顺着代码继续看下去，但不要替代代码本身。

合适的颗粒度：

```python
# results 中 documents、metadatas、ids 都按查询问题分组；
# 当前只有一个问题向量，所以取第 0 组结果
documents = (results.get("documents") or [[]])[0]
```

过细：

```python
# results 是变量
# get 是字典方法
# documents 是 key
```

过粗：

```python
# 处理结果
```

## 数据结构转换注释

遇到外部库返回嵌套结构、多个列表并行处理、排序截断时，优先解释“转换前后是什么”。

推荐：

```python
# zip(sources, scores) 将片段和分数配对，
# 例如 sources=["A", "B"], scores=[0.9, 0.8] 会变成 [("A", 0.9), ("B", 0.8)]
ranked = sorted(zip(sources, scores), key=lambda item: float(item[1]), reverse=True)
```

```python
# key=lambda item: float(item[1]) 表示按元组里的第二个元素，也就是相关性分数排序
```

## fallback 注释

fallback 分支必须说明触发条件和目的，避免读者误以为这是主路径。

推荐：

```python
# Chroma 不可用时走本地 chunks.json 召回，保证缺少向量库依赖时 RAG 仍能工作
if collection is None:
    return self._retrieve_from_local_chunks(...), "local"
```

## 业务流程注释

对 RAG、tool、agent graph 等流程代码，注释应优先说明该函数在流程中的位置。

推荐：

```python
# rag_node 重排入口，返回相关性最高的 top_k 个片段
def rerank(...):
    ...
```

```python
# 每个召回片段和问题组成键值对，再交给 CrossEncoder 计算相关性分数
pairs = [(query, source.content) for source in sources]
```

## 注释长度

- 单行注释优先控制在一行内。
- 如果一句话超过可读长度，拆成两到三行连续注释。
- 不要写成段落式长注释；需要长解释时，抽到文档，不放在代码里。

## 注释与代码同步

- 修改代码逻辑时必须同步更新附近注释。
- 如果注释只是重复旧逻辑而代码已经变化，应删除或重写注释。
- 注释中不要写不稳定的绝对判断，例如“永远不会失败”；除非代码确实保证。
- 注释解释外部库行为时，要以当前调用方式为准，不写超出代码范围的推断。

## 生成注释时的检查清单

写完注释后检查：

- 是否解释了当前代码为什么这样写。
- 是否说明了关键数据结构的输入和输出形状。
- 是否贴近被解释的代码。
- 是否避免了无意义的“执行某操作”式注释。
- 中文是否可读，不能出现乱码。
- 注释是否会在代码修改后变成误导信息。
