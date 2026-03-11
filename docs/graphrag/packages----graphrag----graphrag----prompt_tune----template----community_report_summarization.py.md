
# `graphrag\packages\graphrag\graphrag\prompt_tune\template\community_report_summarization.py` 详细设计文档

该文件定义了一个用于社区报告摘要的微调提示模板（Prompt），用于指导大型语言模型生成包含标题、摘要、评级、评级解释和详细发现的结构化JSON格式社区评估报告。

## 整体流程

```mermaid
graph TD
A[开始] --> B[定义提示模板字符串]
B --> C[模板包含占位符: {persona}, {role}, {report_rating_description}, {language}, {input_text}]
C --> D[用户调用时填充占位符]
D --> E[生成完整的提示指令]
E --> F[将提示发送给语言模型]
F --> G[模型返回JSON格式的社区报告]
```

## 类结构

```
该文件为扁平结构，无类层次
仅包含一个全局常量: COMMUNITY_REPORT_SUMMARIZATION_PROMPT
```

## 全局变量及字段


### `COMMUNITY_REPORT_SUMMARIZATION_PROMPT`
    
用于社区报告摘要的微调提示模板，包含角色定义、报告结构、输出格式、示例输入输出和 grounding 规则

类型：`str`
    


    

## 全局函数及方法



## 关键组件





### 社区报告摘要提示模板

这是一个大型多行字符串模板，用于指导语言模型生成社区评估报告。模板定义了报告的结构、输出格式、JSON规范、基础规则和示例，帮助AI系统根据输入的实体和关系数据生成结构化的社区评估报告。

### 报告结构定义

定义输出报告应包含的五个主要部分：TITLE（社区名称）、SUMMARY（执行摘要）、REPORT RATING（评级）、RATING EXPLANATION（评级说明）和DETAILED FINDINGS（详细发现），其中详细发现包含5-10个关键洞察。

### JSON输出格式规范

定义了严格的JSON输出格式规范，包括嵌套的对象结构、字段名称（title、summary、rating、rating_explanation、findings）以及findings数组中每个元素包含的summary和explanation字段，并要求输出可被json.loads解析。

### 数据基础规则

定义了证据引用的规范格式，要求使用"[Data: <数据集名称> (记录ids)]"格式，并限制单次引用最多5个记录ID，超过时使用"+more"表示。

### 角色和语言占位符

使用{persona}、{role}、{report_rating_description}和{language}作为动态占位符，允许在运行时替换为具体值以定制化报告生成的各个方面。

### 输入文本占位符

使用{input_text}作为输入数据占位符，在实际调用时替换为包含实体和关系数据的文本内容。



## 问题及建议





### 已知问题

- **JSON格式表示不一致**：代码中使用了 `{{{{` 和 `}}}}` 来表示JSON对象的大括号，但这种双重大括号的写法仅在Python f-strings中用于转义单个大括号，在实际使用该提示时会导致输出格式错误
- **占位符定义不完整**：`{language}`, `{role}`, `{report_rating_description}`, `{persona}` 等占位符未提供默认值或说明文档，使用者可能不清楚这些参数的有效值和来源
- **字符串格式化方式混用**：使用传统的 `{placeholder}` 风格但未显式调用 `.format()` 或使用 f-string 注释，可能导致格式化失败或产生意外行为
- **缺乏输入验证**：没有对输入文本 `{{input_text}}` 的格式、大小或内容进行任何校验或说明
- **硬编码常量缺乏灵活性**：将提示模板作为单一常量定义，无法支持不同场景下的参数化配置

### 优化建议

- **统一JSON格式**：移除示例输出中的多余大括号，正确使用 `{{` 和 `}}` 进行转义，确保提示中的示例与实际期望的JSON输出一致
- **添加参数化支持**：使用 dataclass 或 Pydantic 模型定义提示参数结构，明确每个占位符的类型、默认值和可选值
- **采用类型安全**：使用 f-string（添加 `f` 前缀）或 `.format()` 方法进行字符串格式化，提高代码的可读性和可靠性
- **模块化拆分**：将提示模板拆分为多个逻辑部分（元数据、规则、示例等），便于维护和版本管理
- **添加文档注释**：为常量添加 docstring，说明其用途、参数要求和预期行为



## 其它





### 设计目标与约束

**设计目标**：为LLM提供一个结构化的提示模板，使其能够基于输入的实体和关系数据生成符合特定格式要求的社区评估报告，包含标题、摘要、评级、评级解释和详细发现。

**设计约束**：
- 输出必须为有效的JSON格式字符串，可被json.loads解析
- 报告必须包含5-10个关键发现
- 每个发现需包含摘要和解释两部分
- 数据引用需遵循特定的格式规范（数据集名称加记录ID）
- 单个引用最多包含5个记录ID，超过时需使用"+more"标注
- 报告语言需通过占位符{language}动态指定

### 错误处理与异常设计

- **JSON解析异常**：提示中明确要求" Don't use any unnecessary escape sequences"以确保输出的JSON可被正确解析
- **数据缺失处理**：提示要求"Do not include information where the supporting evidence for it is not provided"，即没有支持证据的信息不应包含在报告中
- **记录ID格式**：要求使用记录的human_readable_id而非索引，确保引用准确性
- **输出格式验证**：通过明确的JSON结构模板约束输出格式

### 数据流与状态机

**数据输入流**：
- 输入文本（{input_text}）包含实体表格（human_readable_id, title, description）和关系表格（human_readable_id, source, target, description）
- 角色定义（{role}）决定报告的视角和语气
- Persona（{persona}）定义报告者的身份特征
- 评级描述（{report_rating_description}）定义评级标准
- 语言设置（{language}）决定输出语言

**处理流程**：
- LLM解析输入的实体和关系数据
- 识别关键实体和关系模式
- 生成结构化的分析发现
- 按JSON模板格式组织输出

**数据输出流**：
- 输出为JSON格式的字符串
- 包含title、summary、rating、rating_explanation、findings字段
- findings数组包含5-10个发现对象

### 外部依赖与接口契约

**外部依赖**：
- LLM模型：用于生成报告内容
- json库：用于解析输出的JSON字符串
- 输入数据源：实体和关系数据（CSV格式）

**接口契约**：
- 输入接口：提供包含实体和关系数据的文本
- 输出接口：返回符合JSON schema的结构化报告
- 占位符：{persona}、{role}、{report_rating_description}、{language}、{input_text}需在使用前进行替换

### 安全性考虑

- **数据泄露防护**：提示中不包含任何实际敏感数据，仅为模板
- **JSON转义**：明确要求不使用不必要的转义序列，防止JSON解析错误导致的安全问题
- **证据要求**：强制要求报告内容必须有数据支持，防止生成虚假信息

### 性能要求

- 提示长度需在LLM的上下文窗口限制内
- JSON输出需保持精简，避免过大的响应体积
- 发现数量限制为5-10个，平衡详尽性与性能

### 可扩展性

- 支持多语言：通过{language}占位符支持不同语言输出
- 可自定义角色：通过{role}和{persona}占位符适配不同场景
- 可扩展的报告结构：findings数组可根据需要调整发现数量
- 可配置的评级系统：通过{report_rating_description}自定义评级标准

### 国际化/本地化

- 语言参数化：{language}占位符支持任意语言
- 示例包含英文示例
- 输出要求"Your answers should be in {language}"确保语言一致性

### 配置管理

- 所有占位符通过配置或代码动态替换
- 报告结构模板硬编码在提示中，确保一致性
- Grounding规则标准化，确保数据引用格式统一

### 版本兼容性

- 当前版本为MIT许可的开源项目
- 代码无外部依赖，可在不同环境中运行
- JSON输出格式遵循标准json库规范

### 潜在技术债务与优化空间

1. **硬编码的JSON模板结构**：当前使用四层大括号{{{{ }}}}进行转义，可考虑使用textwrap.dedent或模板引擎简化
2. **缺乏输入验证**：未对输入的实体/关系数据格式进行预验证
3. **固定的发现数量范围**：5-10个发现的限制可能不适用于所有场景
4. **缺乏错误重试机制**：当LLM输出不符合JSON格式时的处理逻辑缺失
5. **Grounding规则复杂度**：数据引用格式规则较复杂，可能导致LLM理解偏差
6. **缺少日志记录**：无法追踪模板使用情况和调试信息
7. **缺乏单元测试**：未提供针对提示模板的测试用例
8. **CSV输入格式固定**：仅支持特定格式的CSV输入，扩展性受限

### 关键组件信息

1. **COMMUNITY_REPORT_SUMMARIZATION_PROMPT**：核心提示模板，包含完整的报告生成指令和输出格式规范
2. **角色定义模块**（{persona}、{role}）：用于定义报告生成者的身份和视角
3. **评级系统**（{report_rating_description}）：可配置的评估标准框架
4. **Grounding规则引擎**：确保报告内容与输入数据一致性的验证规则
5. **JSON格式化器**：将分析结果结构化为标准JSON输出格式


    