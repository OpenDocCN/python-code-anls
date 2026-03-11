
# `graphrag\packages\graphrag\graphrag\prompts\query\global_search_reduce_system_prompt.py` 详细设计文档

Global Search系统提示词配置文件，定义了用于指导AI综合多个分析师报告的系统提示词模板，包含角色定义、目标说明、响应格式要求、数据引用规范和长度限制等关键配置

## 整体流程

```mermaid
graph TD
    A[开始] --> B[定义REDUCE_SYSTEM_PROMPT模板]
    B --> C[定义NO_DATA_ANSWER常量]
    C --> D[模板包含动态变量: {max_length}]
    D --> E[模板包含动态变量: {response_type}]
    E --> F[模板包含动态变量: {report_data}]
    F --> G[提示词包含角色定义和目标说明]
    G --> H[提示词包含数据引用规范]
    H --> I[提示词包含响应格式要求]
```

## 类结构

```
无类层次结构 - 纯配置文件
```

## 全局变量及字段


### `REDUCE_SYSTEM_PROMPT`
    
A multi-line system prompt template for the Global Search system that instructs an AI assistant to synthesize and summarize multiple analyst reports into a coherent response with formatting guidelines.

类型：`str`
    


### `NO_DATA_ANSWER`
    
A default apology message returned when the system cannot answer a user's question based on the provided data.

类型：`str`
    


    

## 全局函数及方法



## 关键组件





### REDUCE_SYSTEM_PROMPT (系统提示模板)

一个复杂的多行字符串，定义了全局搜索系统的核心提示模板，用于将多个分析师的报告综合成最终响应。包含角色定义、响应目标、数据引用格式规范（每条引用最多5个记录ID）、模态动词保留规则，以及{max_length}、{response_type}、{report_data}三个模板变量占位符。

### NO_DATA_ANSWER (无数据响应)

一个字符串常量，当提供的报告中没有足够信息来回答用户问题时，返回的默认道歉回复。

### 角色定义模块 (Role Definition)

提示模板中定义的人工智能助手角色规范，明确设定为"帮助性助手"，通过综合多个分析师的观点来回应关于数据集的问题。

### 分析师报告排序逻辑 (Report Ranking)

提示模板中指定分析师报告按"降序重要性"排列的规则，要求优先使用最重要的报告内容，最终响应中不得提及多位分析师的分析过程角色。

### 数据引用格式化规范 (Data Reference Formatting)

明确限制每条数据引用最多列出5个记录ID，超过时显示"+more"，格式为`[Data: Reports (id1, id2, id3, id4, id5, +more)]`。

### 模板变量系统 (Template Variables)

提示模板中使用的占位符变量：{max_length}控制响应字数限制，{response_type}指定目标响应长度和格式，{report_data}嵌入分析师报告的实际数据内容。

### 响应合成规则 (Response Synthesis)

定义了如何清理和合并分析师报告的规则：移除无关信息、保留模态动词（shall/may/will）、保持数据引用、生成带有适当章节和评论的综合答案。



## 问题及建议



### 已知问题

-   **重复代码（DRY 原则违反）**：REDUCE_SYSTEM_PROMPT 中的 "---Goal---" 和 "---Target response length and format---" 段落完全重复了两次，这违反了 DRY 原则，增加了维护成本。
-   **硬编码提示词**：所有提示词内容直接硬编码在字符串中，缺乏灵活性和可配置性，不利于多语言支持和提示词版本管理。
-   **魔法字符串未提取**："Data: Reports"、"[Data: Reports (...)]" 等格式化字符串散落在提示词中，应提取为常量以提高可维护性。
-   **缺乏模块级文档**：文件缺少模块级 docstring 说明该文件的功能和用途。
-   **提示词结构不清晰**：提示词的各个逻辑部分（角色、目标、格式、示例等）混在一个大字符串中，难以阅读和维护。
-   **可测试性差**：提示词与代码紧耦合，难以进行单元测试验证提示词内容的正确性。
-   **字符串格式化方式**：可考虑使用更现代的格式化方式（如 f-string 或模板库）替代当前的方式。

### 优化建议

-   将重复的提示词段落提取为单独的字符串常量或配置项，避免重复内容。
-   使用配置驱动的方式管理提示词，可将提示词模板存储在外部配置文件（如 YAML、JSON）或数据库中。
-   提取常用格式化字符串为具名常量，例如 `DATA_REFERENCE_TEMPLATE = "Data: Reports ({ids})"`。
-   为模块添加清晰的 docstring，说明这是 Global Search 系统的提示词定义文件。
-   将提示词的不同部分（角色定义、输出格式、示例等）分离为独立配置项或类，提高可读性。
-   考虑引入提示词模板引擎（如 Jinja2），将占位符变量与模板结构分离。

## 其它





### 设计目标与约束

本模块的设计目标是定义Global Search系统的核心提示模板，指导AI助手如何综合多个分析师的报告生成最终响应。约束条件包括：响应长度限制由{max_length}参数控制，数据引用最多5条记录ID，保留原始modal verbs（shall/may/will），不编造信息，仅基于提供的报告数据生成答案。

### 错误处理与异常设计

当提供的报告数据不足或无法回答用户问题时，系统应返回预定义的`NO_DATA_ANSWER`常量："I am sorry but I am unable to answer this question given the provided data." 代码不包含运行时异常处理逻辑，因其为纯数据定义模块。调用方需确保{report_data}、{max_length}、{response_type}占位符被正确替换。

### 数据流与状态机

本模块为静态数据提供模块，无动态状态机设计。数据流如下：调用方传入{max_length}（目标响应字数）、{response_type}（响应格式类型）、{report_data}（分析师报告内容），系统提示模板通过字符串格式化填充这些参数，最终生成完整的system prompt传递给下游LLM处理。

### 外部依赖与接口契约

本模块无外部Python依赖，仅依赖Python标准库的字符串格式化机制。接口契约要求调用方必须提供：report_data（分析师报告字符串）、max_length（整数或可转换为整数的字符串）、response_type（响应类型描述字符串）。模块输出为格式化后的system prompt字符串。

### 安全性考虑

系统提示中包含明确指令要求AI不编造信息（"Do not make anything up"），不包含无法验证的数据（"Do not include information where the supporting evidence for it is not provided"），这体现了数据安全性设计。模块本身不处理敏感数据，仅定义提示规则。

### 性能要求与扩展性

作为静态字符串定义模块，无性能瓶颈。扩展性方面：可通过修改提示模板内容添加新的约束规则（如新的数据引用格式要求），或通过调用方传入不同的response_type实现多种响应格式支持。

### 配置管理

本模块采用硬编码提示模板方式，无动态配置管理机制。如需灵活配置max_length等参数，需由调用方在字符串格式化时传入，体现了我全网搜索结果中关于配置外部化最佳实践的一致性。

### 版本历史与变更日志

当前版本基于MIT License（2024 Microsoft Corporation）。提示模板内容表明其为Global Search系统的核心组件，用于多分析师报告综合场景。变更日志需维护在项目级别文档中。


    