
# `graphrag\packages\graphrag\graphrag\prompts\index\community_report.py` 详细设计文档

这是一个Prompt定义文件，包含用于生成社区报告的完整提示词模板。该提示词指导AI助手如何分析社区实体、关系和声明，生成包含标题、摘要、影响评级（0-10）和5-10个详细发现的JSON格式结构化报告。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[加载COMMUNITY_REPORT_PROMPT模板]
B --> C[将{max_report_length}和{input_text}注入模板]
C --> D[调用大语言模型生成报告]
D --> E[模型输出JSON格式报告]
E --> F{解析JSON}
F --> G[返回结构化报告对象]
F --> H[解析失败则返回原始字符串]
```

## 类结构

```

```

## 全局变量及字段


### `COMMUNITY_REPORT_PROMPT`
    
一个多行字符串提示词模板，用于指导AI模型生成社区报告，包含报告结构、输出格式、 grounding 规则和示例输入输出等内容

类型：`str`
    


    

## 全局函数及方法



## 关键组件





### Community Report Prompt Template

这是一个大型语言模型的提示词模板，用于指导AI助手分析社区数据并生成综合报告。模板定义了报告的目标、结构、输出格式、 grounding 规则以及具体的输入输出规范。

### Goal Definition (目标定义)

明确AI助手作为人类分析师的助手，帮助执行一般信息发现任务，识别和评估与网络中特定实体（如组织和个人）相关的信息。

### Report Structure Specification (报告结构规范)

定义了报告应包含的五个主要部分：TITLE（标题）、SUMMARY（执行摘要）、IMPACT SEVERITY RATING（影响严重程度评分）、RATING EXPLANATION（评分说明）、DETAILED FINDINGS（详细发现）。

### JSON Output Format (JSON输出格式)

定义了输出的JSON结构，包括title、summary、rating、rating_explanation和findings数组，其中findings包含summary和explanation字段。

### Grounding Rules ( grounding 规则)

定义了在报告中引用数据的方法论，包括使用[Data: <dataset name> (record ids)]格式，单个引用最多5个记录ID，以及使用"+more"表示更多记录。

### Input Placeholders (输入占位符)

模板中使用{input_text}作为实际社区数据的占位符，{max_report_length}用于限制报告长度。

### Data Reference Format (数据引用格式)

定义了如何引用不同数据集（Entities、Relationships、Claims、Reports等）的记录ID，以及示例展示如何组合多个数据源。

### Example Input/Output Demonstration (示例输入输出演示)

通过完整的示例展示了从输入数据（实体和关系）到JSON输出的转换过程，帮助模型理解任务要求。

### Impact Severity Rating System (影响严重程度评分系统)

定义评分为0-10之间的浮点数，用于表示社区内实体带来的影响严重程度。



## 问题及建议




### 已知问题

- **缺少文档和类型注解**：代码没有文档字符串（docstring）说明该变量的用途、参数要求和使用方式，缺乏类型注解。
- **硬编码的占位符说明**：提示中包含 `{max_report_length}` 和 `{input_text}` 占位符，但没有提供这些变量的默认值、类型约束或使用说明。
- **JSON格式示例转义问题**：提示中使用 `{{` 和 `}}` 来转义花括号以显示JSON格式示例，但这种转义方式容易导致实际使用时格式混淆，读者可能不清楚哪些是实际JSON的一部分，哪些是提示的一部分。
- **提示模板缺乏模块化**：整个提示是一个巨大的硬编码字符串，如果需要修改某个部分（如调整报告结构或 grounding rules），必须修改整个字符串，缺乏灵活性和可维护性。
- **缺少输入验证机制**：直接使用 `{input_text}` 占位符，没有提供对输入文本的验证、清理或安全处理（如防止提示注入）。
- **缺乏国际化/本地化支持**：提示内容完全硬编码为英文，无法轻松适配其他语言版本的报告生成需求。

### 优化建议

- **添加文档字符串**：为 `COMMUNITY_REPORT_PROMPT` 添加详细的 docstring，说明其用途、占位符含义、预期输入输出格式。
- **模块化提示构建**：将提示的不同部分（如系统指令、报告结构、grounding rules、示例等）拆分为独立的字符串常量，通过模板拼接方式组合，提高可维护性。
- **参数化配置**：将 `{max_report_length}` 等参数提取为配置常量或函数参数，并添加类型注解和默认值说明。
- **分离示例和指令**：将 JSON 格式示例与提示指令明确分离，使用更清晰的标记或注释说明示例的作用。
- **添加输入验证**：在文档中说明 `{input_text}` 的预期格式和安全要求，提供输入验证或清理函数。
- **支持多语言**：考虑将提示内容与语言相关部分分离，支持不同语言的报告生成需求。


## 其它




### 设计目标与约束

设计目标：定义一个结构化的prompt模板，用于指导大型语言模型生成社区报告（Community Report），该报告包含标题、摘要、影响严重性评级、评级解释和详细发现。约束：输出必须为JSON格式，报告长度限制为{max_report_length} words，必须遵循数据 grounding规则，且输入数据包含Entities和Relationships表格。

### 错误处理与异常设计

本代码为纯静态prompt定义文件，不涉及运行时错误处理。潜在问题包括：1) 若{max_report_length}模板变量未正确替换，可能导致输出过长；2) 若输入数据格式不符合预期，prompt中的占位符{input_text}无法被正确填充，导致模型无法生成有效报告。建议在使用时确保模板变量被正确传入。

### 数据流与状态机

数据流：外部系统传入{input_text}（包含Entities和Relationships数据）和{max_report_length}参数 → 填充到prompt模板的占位符 → 发送给大型语言模型 → 模型解析数据并生成JSON格式报告。状态机不适用，本文件为无状态的prompt模板定义。

### 外部依赖与接口契约

外部依赖：1) 大型语言模型API（如OpenAI GPT系列），用于执行prompt；2) 输入数据源，需提供符合格式的Entities和Relationships数据（CSV格式）。接口契约：调用方需提供{input_text}（字符串）和{max_report_length}（整数）两个参数，prompt模板返回完整的prompt字符串供模型使用。

### 关键组件信息

组件名称：COMMUNITY_REPORT_PROMPT
一句话描述：一个用于指导AI生成社区报告的字符串模板，包含详细的输出格式规范、grounding规则和示例数据。

### 潜在的技术债务或优化空间

1. Prompt硬编码：prompt模板直接作为常量字符串，可考虑外部化到配置文件或数据库；2. 缺乏输入验证：未验证max_report_length的有效范围（如正整数）；3. 结构可扩展性差：若需添加新的报告章节，需修改prompt字符串，可考虑模块化prompt构建；4. 国际化支持缺失：prompt为英文，若需支持其他语言需重新设计。

### 其它项目

1. 版本信息与变更日志：建议添加版本号和变更说明，便于追踪prompt迭代；2. 使用示例与文档：虽然代码中包含示例输入输出，但建议添加更完整的使用说明；3. 性能考量：本文件无性能影响，但使用此prompt的调用需考虑LLM API的响应时间和token消耗；4. 安全与隐私：prompt中要求模型不要编造信息（"Do not include information where the supporting evidence for it is not provided"），这是良好的安全实践。

    