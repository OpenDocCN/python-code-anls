
# `.\AutoGPT\classic\benchmark\agbenchmark\utils\prompts.py` 详细设计文档

这是一个AI评分提示模板库，提供了多种用于评估机器生成响应的提示模板（reference、rubric、question、custom），支持percentage、scale、binary三种评分方式，通过模板替换机制生成最终的用户提示词。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[选择评分类型]
    B --> C{评分类型}
    C -->|reference| D[使用REFERENCE_PROMPT]
    C -->|rubric| E[使用RUBRIC_PROMPT]
    C -->|question| F[使用QUESTION_PROMPT]
    C -->|custom| G[使用CUSTOM_PROMPT]
    D --> H[替换{scoring}{task}{answer}{response}占位符]
    E --> H
    F --> H
    G --> H
    H --> I[拼接END_PROMPT]
    I --> J[返回完整提示词]
```

## 类结构

```
无类层次结构（纯配置模块）
├── 全局常量
│   ├── SCORING_MAP (评分方式描述)
│   ├── REFERENCE_PROMPT (参考模板)
│   ├── RUBRIC_PROMPT (评分标准模板)
│   ├── QUESTION_PROMPT (问题模板)
│   ├── FEW_SHOT_EXAMPLES (少样本示例)
│   ├── CUSTOM_PROMPT (自定义模板)
│   ├── PROMPT_MAP (模板映射表)
│   └── END_PROMPT (结束提示)
```

## 全局变量及字段


### `SCORING_MAP`
    
A dictionary mapping scoring type names ('percentage', 'scale', 'binary') to their corresponding evaluation instructions and descriptions.

类型：`Dict[str, Tuple[str, ...]]`
    


### `REFERENCE_PROMPT`
    
A prompt template string for evaluating machine generated responses against a reference/ideal answer using distance-based scoring.

类型：`str`
    


### `RUBRIC_PROMPT`
    
A prompt template string for evaluating machine generated responses using a rubric-based scoring approach that factors in predefined criteria.

类型：`str`
    


### `QUESTION_PROMPT`
    
A prompt template string for evaluating machine generated responses based on how well they answer a specific question.

类型：`str`
    


### `FEW_SHOT_EXAMPLES`
    
A placeholder string template for including few-shot examples in evaluation prompts to guide the scoring behavior.

类型：`str`
    


### `CUSTOM_PROMPT`
    
A flexible prompt template that combines custom instructions with scoring methodology specifications.

类型：`str`
    


### `PROMPT_MAP`
    
A dictionary mapping prompt type identifiers ('rubric', 'reference', 'question', 'custom') to their corresponding prompt template strings.

类型：`Dict[str, str]`
    


### `END_PROMPT`
    
A constant string instructing the language model to conclude its response with only a float score value.

类型：`str`
    


    

## 全局函数及方法



## 关键组件




### 概述

该代码定义了一套用于评估机器生成文本质量的提示词系统和评分映射规则，核心功能是通过不同的提示模板（参考评估、评分标准评估、问题评估）引导语言模型对生成内容进行打分，支持百分比、十分制和二进制三种评分模式。

### 文件整体运行流程

1. **初始化阶段**：定义评分规则映射（SCORING_MAP）和各类提示词模板
2. **运行时阶段**：根据评估类型从PROMPT_MAP选择对应模板，填充{scoring}、{task}、{answer}、{response}等变量
3. **输出阶段**：拼接END_PROMPT要求模型返回浮点数评分

### 全局变量详细信息

| 名称 | 类型 | 描述 |
|------|------|------|
| SCORING_MAP | dict | 评分映射表，定义percentage/scale/binary三种评分方式及描述 |
| REFERENCE_PROMPT | str | 参考提示模板，用于比较机器生成回答与理想回答的接近程度 |
| RUBRIC_PROMPT | str | 评分标准提示模板，基于评分标准评估生成文本 |
| QUESTION_PROMPT | str | 问题提示模板，评估生成回答是否正确回答问题 |
| FEW_SHOT_EXAMPLES | str | 少样本示例模板，用于提供评分示例 |
| CUSTOM_PROMPT | str | 自定义提示模板，支持自定义评分规则 |
| PROMPT_MAP | dict | 提示映射字典，将评估类型映射到对应模板 |
| END_PROMPT | str | 结束提示，要求模型仅返回浮点数分数 |

### 关键组件信息

#### 1. SCORING_MAP

定义了三种评分策略：percentage（0-100百分制）、scale（1-10十分制）、binary（0/1二进制），每种策略包含评分规则描述。

#### 2. 提示模板系统（REFERENCE_PROMPT/RUBRIC_PROMPT/QUESTION_PROMPT）

通过变量替换机制实现灵活评估，{scoring}变量注入评分策略，{task}/{answer}/{response}注入评估数据。

#### 3. PROMPT_MAP

提供统一的提示模板访问接口，支持rubric、reference、question、custom四种评估模式。

#### 4. END_PROMPT

强制约束输出格式，确保模型仅返回浮点数评分结果。

### 潜在技术债务与优化空间

1. **字符串硬编码**：所有提示模板作为字符串常量内嵌，缺乏外部配置化支持
2. **变量安全风险**：使用Python字符串format()方法存在安全风险（代码中未显示但隐含），建议使用f-string或模板引擎
3. **缺乏验证机制**：没有对输入变量占位符的完整性校验
4. **国际化局限**：提示模板为英文，不支持多语言评估场景
5. **重复代码结构**：各提示模板结构相似，可抽象基类或组合模式

### 其他项目

#### 设计目标与约束
- 目标：构建灵活可扩展的LLM输出评估框架
- 约束：要求模型返回单一浮点数，禁止附加解释

#### 错误处理与异常设计
- 当前实现无错误处理机制
- 建议：添加模板变量缺失、评分策略不支持等异常场景处理

#### 数据流与状态机
- 数据流：评估请求 → 选择提示模板 → 填充变量 → 发送给LLM → 解析响应
- 状态：静态配置阶段 → 运行时填充阶段 → 输出阶段

#### 外部依赖与接口契约
- 依赖：Python 3.6+字符串格式化
- 接口契约：调用方需提供{scoring}、{task}、{answer}、{response}四个必需变量


## 问题及建议




### 已知问题

-   **重复代码片段**：多个prompt模板中包含相同的指令文本（如"Return nothing but a float score"），未提取为共享常量，导致维护成本高
-   **END_PROMPT与评分类型不匹配**：END_PROMPT固定要求返回float分数，但SCORING_MAP中的binary评分应返回整数0或1，scale评分应为1-10的整数，类型不一致
-   **字符串格式化方式过时**：使用 `%` 格式化风格（`{scoring}`），不如f-string简洁且错误提示不友好
-   **缺少输入验证**：scoring参数未验证是否为有效值（percentage/scale/binary），可能导致后续逻辑错误
-   **未使用的常量**：FEW_SHOT_EXAMPLES已定义但在PROMPT_MAP中没有对应入口，属于死代码
-   **硬编码的prompt内容**：所有文本内容硬编码，不支持国际化或多租户定制需求
-   **类型提示缺失**：无任何函数签名类型注解，影响代码可读性和静态分析工具的效能
-   **行长度问题**：使用 `# noqa: E501` 跳过E501检查而非调整代码格式，表明代码风格不够严谨

### 优化建议

-   **提取通用指令**：将重复的指令文本提取为独立常量，如 `SCORE_RETURN_INSTRUCTION = "Return nothing but a float score."`
-   **增强END_PROMPT灵活性**：根据scoring类型生成对应的END_PROMPT，或在文档中明确约束返回类型
-   **迁移至f-string**：将 `%` 格式化替换为f-string，提升可读性和调试体验
-   **添加输入验证**：在函数入口处校验scoring值，参考SCORING_MAP.keys()进行白名单验证
-   **清理死代码**：补充FEW_SHOT_EXAMPLES到PROMPT_MAP，或删除未使用的常量
-   **引入配置化prompt**：将prompt模板外部化，支持通过配置文件或参数注入实现定制化
-   **补充类型注解**：为函数参数和返回值添加Type Hints，利用IDE和mypy进行类型检查
-   **修复代码格式**：合理拆分长行，移除noqa注释，遵循PEP 8规范（建议行长≤79或≤120）


## 其它





### 设计目标与约束

该模块的核心设计目标是提供一个灵活、可扩展的评分框架，用于评估机器生成文本与人类答案的接近程度。支持三种评分模式（百分比、1-10量表、二元判断），通过模板化提示词系统适配不同的评估场景。设计约束包括：评分逻辑完全依赖外部LLM API实现，评分精度受限于提示词模板设计，不支持实时流式评分结果解析。

### 错误处理与异常设计

本模块主要依赖外部LLM调用返回结果，未在代码层面实现显式的异常捕获机制。潜在异常场景包括：LLM API返回非浮点数格式、API调用超时或失败、提示词模板参数缺失。对于API返回格式异常，当前代码假设LLM会严格遵循"Return nothing but a float score"的指令，因此缺乏对异常返回值的容错处理。建议在实际调用层增加结果校验逻辑，确保返回值可解析为浮点数。

### 外部依赖与接口契约

主要外部依赖为LLM API（如OpenAI GPT系列）。接口契约如下：输入为包含任务描述、参考答案、机器生成响应的提示词模板；输出要求为仅包含浮点数分数的纯文本响应。PROMPT_MAP定义了四种提示词模板类型（rubric、reference、question、custom），调用方需根据评分场景选择合适的模板并填充相应参数。END_PROMPT为固定结尾指令，确保LLM输出格式一致。

### 使用场景与示例

该模块适用于基于LLM的自动化评估流水线。典型使用场景包括：1）使用reference模式评估生成文本与标准答案的相似度；2）使用rubric模式基于评分规则（rubric）评估响应质量；3）使用question模式验证生成内容是否正确回答了指定问题；4）使用custom模式支持自定义评分标准。评分结果可用于模型微调、采样策略优化、A/B测试等下游任务。

### 安全考虑

提示词模板中包含"Ignore previous directions"等指令注入模式，用于覆盖LLM可能存在的系统提示。虽然在评估场景下这是预期行为，但需注意：1）此类模式可能被恶意用户利用进行提示注入攻击；2）评分结果仅作参考，重要业务决策需人工复核；3）如果评分涉及敏感内容，提示词模板需进行脱敏处理。

### 性能考虑

当前实现为同步调用模式，每次评分请求需等待LLM API响应。性能瓶颈主要在于：1）网络延迟（取决于LLM服务商地理位置）；2）LLM推理时间（受模型复杂度、输入长度影响）。优化方向包括：1）实现批量评分接口，减少网络往返；2）考虑使用更小、更快的评估模型；3）增加评分结果缓存机制（针对相同输入）；4）异步处理多个评分请求。

### 配置说明

SCORING_MAP为评分方式配置字典，可按需扩展新的评分模式。PROMPT_MAP为提示词模板映射，可通过修改模板内容调整评分行为。提示词模板支持占位符替换，常见占位符包括：{scoring}（评分方式描述）、{task}（任务描述）、{answer}（参考答案/评分标准）、{response}（待评估响应）、{examples}（few-shot示例）、{custom}（自定义内容）。

### 关键数据流转

输入数据流：调用方准备task/answer/response → 填充对应提示词模板占位符 → 构建完整提示词 → 发送给LLM API。输出数据流：LLM返回文本响应 → 解析提取浮点数分数 → 返回给调用方。评分方式描述（{scoring}）从SCORING_MAP中获取对应评分规则的文字描述，并注入到提示词模板中，引导LLM按照指定方式评分。


    