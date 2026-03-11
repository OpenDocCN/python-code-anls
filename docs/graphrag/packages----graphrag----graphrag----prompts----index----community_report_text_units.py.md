
# `graphrag\packages\graphrag\graphrag\prompts\index\community_report_text_units.py` 详细设计文档

该文件定义了用于生成社区报告的LLM提示模板，包含报告结构、 grounding 规则、输出格式规范和示例输入输出，用于指导AI模型生成结构化的社区分析报告。

## 整体流程

```mermaid
graph TD
    A[外部模块调用] --> B[加载COMMUNITY_REPORT_TEXT_PROMPT]
B --> C[替换{input_text}占位符]
C --> D[将提示发送给LLM]
D --> E[接收LLM生成的JSON响应]
E --> F[解析JSON得到结构化报告]
```

## 类结构

```
无类层次结构 - 该文件仅包含全局常量
```

## 全局变量及字段


### `COMMUNITY_REPORT_TEXT_PROMPT`
    
一个用于生成社区报告的文本提示模板，定义了报告的结构（包括标题、摘要、重要性评分、详细发现和日期范围）、输出JSON格式要求、数据接地规则以及示例输入输出

类型：`str`
    


    

## 全局函数及方法



## 关键组件




### COMMUNITY_REPORT_TEXT_PROMPT

核心prompt模板常量，定义了生成社区报告的完整指令系统，包含任务目标、输出结构、 grounding rules 和示例数据。

### 报告结构定义

定义了JSON输出的六个核心字段：title（社区名称）、summary（执行摘要）、rating（重要性评分0-10）、rating_explanation（评分说明）、findings（详细发现列表）、date_range（日期范围）。

### Grounding Rules 规范

数据引用规范，要求每个数据点引用不超过5个record id，超出时使用"+more"标记，并指定数据集名称和日期范围。

### JSON输出格式规范

明确定义了输出JSON的严格结构，包含嵌套的findings数组，每个finding包含summary和explanation字段，以及单层级的title、summary、rating等字段。

### 模板变量 {max_report_length}

动态占位符，允许调用方控制生成报告的最大字数，实现输出长度的灵活控制。

### 示例输入输出模块

包含完整的Enron邮件数据示例及对应的结构化JSON输出，用于Few-shot learning指导模型理解期望的输出格式。


## 问题及建议



### 已知问题

- **JSON格式不一致**：prompt示例中的`findings`数组里，`explanation`字段的引号未正确闭合（如`"explanation": "<insight_1_explanation"`末尾缺少闭合引号），可能导致LLM解析错误
- **占位符缺乏默认值**：`{max_report_length}`和`{input_text}`占位符没有提供默认值或参数校验，调用方未传入时会触发运行时错误
- **硬编码的业务参数**：关键数字如"5-10个关键发现"、"最多5条记录ID"被直接写在prompt中，修改需改代码
- **缺乏版本管理**：prompt模板没有版本号或变更记录，难以追踪和回溯
- **字符串过长**：单一字符串变量包含超过300行内容，维护和阅读困难
- **缺少使用文档**：没有注释说明该prompt的适用场景、输入要求或调用示例

### 优化建议

- **修复JSON语法错误**：检查并修正prompt示例中的JSON格式，确保所有字段引号闭合
- **添加参数验证**：在代码层面验证`max_report_length`为正整数，`input_text`非空
- **外部化配置**：将"5-10"、"max_report_length"默认值等业务参数提取到配置文件中
- **拆分长字符串**：将prompt按逻辑章节拆分为多个常量或使用模板文件（如Jinja2）
- **添加Prompt版本**：引入版本号支持多版本管理，便于A/B测试和回滚
- **增加单元测试**：验证模板渲染后的JSON格式正确性

## 其它





### 设计目标与约束

**设计目标：**
1. 为AI模型提供结构化的社区报告生成指令，确保输出符合预定义的JSON schema
2. 保持报告的专业性和可读性，同时确保数据可追溯性
3. 支持灵活的最大报告长度配置

**设计约束：**
1. 输出必须为有效的JSON格式，可通过json.loads解析
2. 报告长度受max_report_length参数约束
3. 数据引用需遵循特定的格式规范（数据集名称+记录ID）
4. 日期范围必须使用YYYY-MM-DD格式

### 错误处理与异常设计

**输入验证：**
- input_text为必需参数，若为空可能导致输出格式不符合预期
- max_report_length参数需为正整数，否则可能影响报告质量

**格式异常处理：**
- 当输入文本不包含足够信息时，报告的date_range可能为空或无效
- findings数组长度可能少于预期的5-10条，取决于输入数据的丰富程度
- JSON输出中的转义字符需要严格控制，避免使用不必要的转义

### 数据流与状态机

**数据输入流程：**
```
外部输入 → 模板格式化(max_report_length填充) → 完整提示词 → AI模型 → JSON输出
```

**关键状态：**
1. 模板状态：原始提示词模板定义
2. 格式化状态：填充变量后的完整提示词
3. 输出状态：AI模型返回的JSON字符串结果

### 外部依赖与接口契约

**依赖项：**
- json模块：用于解析最终输出的JSON字符串
- AI语言模型：负责根据提示词生成报告

**接口契约：**
- 输入：包含max_report_length变量的格式化字符串和input_text原始文本
- 输出：符合JSON schema的字符串，包含title、summary、rating、rating_explanation、findings、date_range字段
- 契约约束：输出必须为单行JSON对象，不包含不必要的转义序列

### 性能考虑

**优化点：**
1. 模板为静态字符串常量，无运行时计算开销
2. 字符串格式化操作（format）性能开销可忽略
3. 建议：对于大量重复调用，可考虑缓存格式化后的提示词

### 安全考虑

**数据安全：**
1. 提示词中不包含任何敏感信息或凭证
2. 输出JSON中的数据引用遵循最小化原则（最多5个记录ID）
3. 需注意input_text可能包含敏感信息，AI模型处理时需确保数据隐私

### 可维护性与扩展性

**扩展性设计：**
1. 模板结构化良好，便于添加新的报告章节
2. 可通过修改模板轻松调整报告格式或添加新的 grounding rules
3. max_report_length参数化设计便于控制输出长度

**维护建议：**
1. JSON schema定义可考虑提取为独立常量或配置文件
2. Grounding rules的格式规范可考虑抽取为独立模块
3. 建议添加模板版本管理机制

### 测试策略

**测试重点：**
1. 模板格式化后的字符串完整性验证
2. AI输出JSON格式的有效性验证
3. max_report_length参数边界值测试（极小值、极大值）
4. 空输入或特殊字符输入的处理验证

**验证方法：**
- 使用json.loads验证输出可解析性
- 使用JSON schema validator验证字段完整性
- 检查转义字符是否符合"不必要"的标准

### 配置与常量定义

**关键配置项：**
- max_report_length：控制报告最大词数，需根据实际需求调整
- findings数量范围：5-10条 insight
- rating评分范围：0-10的浮点数
- date_range格式：[YYYY-MM-DD, YYYY-MM-DD]


    