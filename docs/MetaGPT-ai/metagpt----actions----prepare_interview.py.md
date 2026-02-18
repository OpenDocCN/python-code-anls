
# `.\MetaGPT\metagpt\actions\prepare_interview.py` 详细设计文档

该代码定义了一个名为 PrepareInterview 的动作类，其核心功能是使用大语言模型（LLM）根据给定的上下文（如简历）自动生成一份面试问题列表。它通过封装一个预定义的 ActionNode（QUESTIONS）来标准化问题的生成流程，旨在为面试官提供结构化的提问参考。

## 整体流程

```mermaid
graph TD
    A[开始: 调用 PrepareInterview.run] --> B[输入: context (如简历文本)]
    B --> C[调用 QUESTIONS.fill 方法]
    C --> D[向 LLM 发送请求]
    D --> E{LLM 生成结果}
    E --> F[返回: 面试问题列表 (list[str])]
    F --> G[结束]
```

## 类结构

```
Action (来自 metagpt.actions)
└── PrepareInterview
```

## 全局变量及字段


### `QUESTIONS`
    
一个预定义的 ActionNode 实例，用于生成面试问题列表，其指令要求根据简历上下文创建至少10个前端或后端开发相关的面试问题。

类型：`ActionNode`
    


### `PrepareInterview.name`
    
Action 的名称，在此类中固定为 'PrepareInterview'，用于标识该动作。

类型：`str`
    
    

## 全局函数及方法

### `PrepareInterview.run`

该方法是一个异步方法，用于根据给定的上下文（例如简历内容）生成一份面试问题列表。它通过调用一个预定义的 `ActionNode` 对象 `QUESTIONS` 来利用大语言模型（LLM）完成此任务。

参数：

- `context`：`str`，包含生成面试问题所需信息的上下文，例如求职者的简历内容。

返回值：`list[str]`，返回一个由大语言模型生成的面试问题字符串列表。

#### 流程图

```mermaid
flowchart TD
    A[开始: run(context)] --> B[调用 QUESTIONS.fill<br>传入 context 和 self.llm]
    B --> C[LLM 处理请求<br>基于 context 生成问题列表]
    C --> D[返回生成的<br>问题列表]
    D --> E[结束]
```

#### 带注释源码

```python
async def run(self, context):
    # 调用 QUESTIONS ActionNode 的 fill 方法。
    # 参数 `req` 接收上下文信息（如简历），`llm` 参数传入当前 Action 实例所持有的大语言模型接口。
    # 该方法将利用 LLM 根据上下文生成并返回一个面试问题列表。
    return await QUESTIONS.fill(req=context, llm=self.llm)
```

## 关键组件


### ActionNode

ActionNode 是 MetaGPT 框架中用于定义和封装结构化动作输出的核心组件，它通过指定期望的输出类型、指令和示例，指导大语言模型生成格式化的内容。

### PrepareInterview Action

PrepareInterview 是一个具体的 Action 类，它继承自 MetaGPT 框架的基础 Action 类，其核心功能是利用 QUESTIONS ActionNode 和给定的上下文（简历），通过大语言模型生成一份面试问题列表。

### 大语言模型集成

代码通过 `self.llm` 属性集成了大语言模型，作为 QUESTIONS ActionNode 执行 `fill` 方法时的推理引擎，用于根据指令和上下文生成最终的面试问题列表。


## 问题及建议


### 已知问题

-   **硬编码的提示词与角色**：`QUESTIONS` ActionNode 中的 `instruction` 字段包含了硬编码的角色描述（“You are an interviewer of our company...”）。这使得代码的复用性降低，难以适应不同的面试场景（如不同公司、不同职位、不同面试官风格）。
-   **缺乏配置与上下文定制**：`PrepareInterview` 类的 `run` 方法直接将 `context` 作为 `req` 参数传递给 `QUESTIONS.fill`。`context` 的内容和结构未被明确定义或验证，可能导致 `ActionNode` 无法正确解析输入，从而生成不相关或低质量的问题。
-   **输出格式僵化**：`QUESTIONS` 节点要求输出为特定的 Markdown 列表格式。虽然这提供了结构，但也限制了输出的灵活性。如果需求变为生成 JSON、纯文本或其他结构化数据，则需要修改代码。
-   **示例问题过于简单**：`QUESTIONS` 节点的 `example` 字段仅提供了格式示例（`["1. What ...", "2. How ..."]`），缺乏具体、有代表性的问题内容。这可能无法有效地引导大语言模型生成高质量、有针对性的面试问题。
-   **类设计过于简单**：`PrepareInterview` 类目前仅是一个对 `ActionNode` 的简单包装。它没有提供任何额外的逻辑来处理输入、验证输出、或管理对话状态。这限制了其作为独立、可复用组件的潜力。

### 优化建议

-   **参数化提示词与角色**：将 `QUESTIONS` ActionNode 中的 `instruction` 内容（特别是角色描述和具体要求）提取为类属性或构造参数。这样可以在创建 `PrepareInterview` 实例时动态指定面试官角色、公司背景、问题数量要求等，提高灵活性。
-   **明确输入上下文契约**：在 `PrepareInterview` 类的文档或方法签名中，明确 `run` 方法中 `context` 参数的预期格式和内容（例如，它应包含求职者的简历文本）。可以考虑使用 Pydantic 模型来定义和验证输入数据结构。
-   **增强输出处理与验证**：在 `run` 方法中，不仅调用 `QUESTIONS.fill`，还可以增加对返回结果的后续处理。例如，验证返回的列表是否满足最小数量要求，对问题进行去重、排序或格式化，或者将结果转换为更易用的数据结构（如字典列表，每个问题附带类型标签如“技术”、“行为”等）。
-   **提供更丰富的示例**：更新 `QUESTIONS` 节点的 `example` 字段，包含更具体、多样化的真实面试问题示例。这能更好地指导大语言模型生成符合预期的内容。
-   **扩展类功能与职责**：考虑让 `PrepareInterview` 类承担更多职责。例如：
    -   集成从不同来源（文件、数据库、API）加载简历的逻辑。
    -   支持根据不同的职位描述（JD）来调整问题生成策略。
    -   实现一个缓存机制，对于相似的简历和职位组合，避免重复调用大模型以节省成本。
    -   添加日志记录和指标收集，用于监控问题生成的质量和性能。
-   **考虑错误处理与降级策略**：在 `run` 方法中添加异常处理逻辑。如果大语言模型调用失败或返回了不符合格式要求的结果，应有相应的错误处理机制（如重试、返回默认问题列表、或抛出清晰的异常信息）。


## 其它


### 设计目标与约束

本模块的核心设计目标是提供一个可复用的、基于大语言模型（LLM）的面试问题生成动作（Action）。它旨在接收一份简历文本作为上下文（context），并自动生成一份至少包含10个问题的面试问题列表。其设计遵循了MetaGPT框架的`Action`基类规范，确保能够无缝集成到更复杂的智能体（Agent）工作流中。主要约束包括：1) 依赖于外部LLM服务的可用性与性能；2) 生成的面试问题质量受限于提示词（Prompt）的设计和LLM本身的能力；3) 输出格式被严格限定为Markdown列表样式。

### 错误处理与异常设计

当前代码未显式包含错误处理逻辑。`PrepareInterview.run`方法直接调用`QUESTIONS.fill`并返回其结果。潜在的异常可能来自：1) `self.llm`调用失败（如网络错误、API错误）；2) `QUESTIONS.fill`内部处理异常；3) 输入的`context`不符合预期。这些异常会向上抛出，由调用者（通常是MetaGPT框架的运行时）捕获和处理。建议的优化是：在`run`方法内部添加`try-except`块，捕获`LLMError`、`ValidationError`等特定异常，并返回一个包含错误信息的结构化结果或默认问题列表，以提高动作的健壮性。

### 数据流与状态机

本模块的数据流相对简单直接：
1.  **输入**：`PrepareInterview.run`方法接收一个`context`参数（字符串类型，代表简历文本）。
2.  **处理**：将`context`和`self.llm`实例传递给预定义的`QUESTIONS`（`ActionNode`对象）。`QUESTIONS`内部会构建包含指令、示例和上下文的Prompt，调用LLM，并解析LLM的返回结果。
3.  **输出**：`QUESTIONS.fill`返回一个`list[str]`，即生成的面试问题列表，该列表由`run`方法直接返回。
模块本身是无状态的（Stateless），不维护任何内部状态。每次调用都是独立的，输出完全由当前输入和LLM决定。

### 外部依赖与接口契约

1.  **外部依赖**：
    *   **MetaGPT框架**：强依赖`metagpt.actions.Action`基类和`metagpt.actions.action_node.ActionNode`类。这决定了模块的编程接口和运行方式。
    *   **大语言模型（LLM）服务**：通过`self.llm`属性（在父类`Action`中初始化）进行交互，是核心功能依赖。具体的LLM提供商（如OpenAI、Azure、Ollama等）由框架配置决定。
2.  **接口契约**：
    *   **输入契约**：`run`方法的`context`参数应为字符串类型，内容为待分析的简历文本。调用者需确保其有效性。
    *   **输出契约**：`run`方法返回一个`list[str]`，即生成的面试问题列表。列表应至少包含10个元素，每个元素是一个字符串形式的问题。返回格式符合`QUESTIONS`节点定义的`expected_type`。
    *   **父类契约**：必须实现`Action`基类定义的`run`异步方法，并遵循其生命周期约定。

### 配置与参数化

模块的行为主要通过`QUESTIONS`这个`ActionNode`对象的配置来驱动：
*   **`instruction`**：定义了LLM的角色、任务和具体要求（如至少10个问题）。这是控制生成内容质量和方向的关键参数。
*   **`example`**：提供了输出格式的示例，引导LLM生成符合`list[str]`类型且格式规范（如“1. ...”）的内容。
*   **`expected_type`**：指定了输出数据的类型（`list[str]`），用于指导`ActionNode`对LLM原始输出的解析。
目前，这些配置是硬编码（Hard-coded）在源码中的。一个优化方向是将`instruction`和`example`等内容外部化（如放到配置文件或数据库中），以便在不修改代码的情况下调整面试问题的生成策略，例如针对不同岗位（前端、后端、算法）定制不同的提示词。

    