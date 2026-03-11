
# `diffusers\src\diffusers\modular_pipelines\qwenimage\prompt_templates.py` 详细设计文档

该模块集中管理QwenImage管道中使用的所有提示词模板，支持四种管道变体：QwenImage（文本到图像）、QwenImage Edit（单图像编辑）、QwenImage Edit Plus（多参考图像编辑）和QwenImage Layered（自动标注），通过定义不同场景下的系统提示、用户提示和助手响应格式来统一图像生成与编辑的输入规范。

## 整体流程

```mermaid
graph TD
    A[外部调用方] --> B[选择对应模板]
    B --> C{管道类型?}
    C -->|Text-to-Image| D[QWENIMAGE_PROMPT_TEMPLATE]
    C -->|Single-Image Edit| E[QWENIMAGE_EDIT_PROMPT_TEMPLATE]
    C -->|Multi-Image Edit| F[QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE + IMG_TEMPLATE]
    C -->|Layered/Auto-Caption| G[QWENIMAGE_LAYERED_CAPTION_PROMPT_EN/CN]
    D --> H[填充用户输入{}]
    E --> H
    F --> H
    G --> H
    H --> I[送入VL模型或文本编码器]
```

## 类结构

```
该文件为纯模块文件，无类定义，仅包含模块级常量变量
```

## 全局变量及字段


### `QWENIMAGE_PROMPT_TEMPLATE`
    
文本到图像生成的提示词模板，包含系统消息和用户消息的结构化格式，用于描述图像的颜色、形状、尺寸等属性

类型：`str`
    


### `QWENIMAGE_PROMPT_TEMPLATE_START_IDX`
    
基础提示词模板中assistant回复部分的起始索引位置，用于定位生成内容的开始位置

类型：`int`
    


### `QWENIMAGE_EDIT_PROMPT_TEMPLATE`
    
单图像编辑的提示词模板，包含视觉编码标签(vision_start/vision_end)和图像填充符(image_pad)，用于将图像与文本指令结合进行编辑

类型：`str`
    


### `QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX`
    
编辑提示词模板中assistant回复部分的起始索引位置，用于定位生成内容的开始位置

类型：`int`
    


### `QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE`
    
多图像编辑(多参考图)的提示词模板，允许多个图像与文本指令结合进行复杂编辑任务

类型：`str`
    


### `QWENIMAGE_EDIT_PLUS_IMG_TEMPLATE`
    
多图像编辑中用于格式化每个输入图像的模板，使用占位符{}插入图像编号，配合视觉标签和图像填充符

类型：`str`
    


### `QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE_START_IDX`
    
多图像编辑提示词模板中assistant回复部分的起始索引位置

类型：`int`
    


### `QWENIMAGE_LAYERED_CAPTION_PROMPT_EN`
    
英文版分层图像标注提示词模板，用于自动图像标注任务，指导模型生成包含对象属性、视觉关系、环境细节和文字内容(引号强调)的描述

类型：`str`
    


### `QWENIMAGE_LAYERED_CAPTION_PROMPT_CN`
    
中文版分层图像标注提示词模板，功能与英文版相同，用于中文语境下的自动图像标注任务

类型：`str`
    


    

## 全局函数及方法



## 关键组件





### QWENIMAGE_PROMPT_TEMPLATE

用于纯文本到图像生成的基础提示词模板，包含系统消息和用户消息的结构化格式，用于指导模型生成详细的图像描述。

### QWENIMAGE_PROMPT_TEMPLATE_START_IDX

标记QWENIMAGE_PROMPT_TEMPLATE中assistant角色开始的token位置，用于精确控制提示词结构。

### QWENIMAGE_EDIT_PROMPT_TEMPLATE

用于单图像编辑的视觉-语言编码提示词模板，结合图像token和文本指令，支持图像修改任务的提示词格式化。

### QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX

标记QWENIMAGE_EDIT_PROMPT_TEMPLATE中assistant角色开始的token位置。

### QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE

用于多参考图像编辑的提示词模板，支持多个图像输入的上下文整合和指令跟随。

### QWENIMAGE_EDIT_PLUS_IMG_TEMPLATE

多图像编辑中的单个图像格式化模板，用于将每个图像嵌入到提示词中。

### QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE_START_IDX

标记QWENIMAGE_EDIT_PLUS_PROMPT_TEMPLATE中assistant角色开始的token位置。

### QWENIMAGE_LAYERED_CAPTION_PROMPT_EN

用于图像分层处理的英文自动标注提示词，指导模型生成包含对象属性、视觉关系和环境细节的详细图像描述。

### QWENIMAGE_LAYERED_CAPTION_PROMPT_CN

用于图像分层处理的中文自动标注提示词，支持多语言图像标注场景。



## 问题及建议



### 已知问题

-   **魔法数字（Magic Numbers）**：使用了硬编码的索引值（如 `QWENIMAGE_PROMPT_TEMPLATE_START_IDX = 34`、`QWENIMAGE_EDIT_PROMPT_TEMPLATE_START_IDX = 64`），这些数字没有任何注释说明其含义和计算依据，当模板内容变化时需要手动重新计算，容易出错且难以维护。
-   **模板内容重复**：多个模板之间存在大量重复的系统提示部分（如 `<|im_start|>system\n` 开头的描述），未进行提取复用，增加了维护成本。
-   **缺乏国际化架构**：虽然提供了中英文两种 `QWENIMAGE_LAYERED_CAPTION_PROMPT`，但命名方式不一致（_EN/_CN 后缀），且未建立统一的多语言管理机制，其他语言扩展困难。
-   **硬编码的特殊标签**：`<|vision_start|>`、`<|image_pad|>`、`<|vision_end|>` 等标签在多个模板中重复硬编码，若标签格式需要调整，需逐一修改所有位置。
-   **无输入验证**：没有对模板占位符（`{}`）的数量或格式进行校验，使用时若占位符不匹配可能导致运行时错误。
-   **缺乏类型注解**：作为配置模块，未使用类型提示（Type Hints）声明常量类型，降低了代码的可读性和 IDE 辅助支持。
-   **无错误处理机制**：模板被错误使用（如索引越界、格式不匹配）时无法提供有意义的错误信息或异常抛出。

### 优化建议

-   **消除魔法数字**：将 `START_IDX` 改为动态计算（例如通过模板字符串的 `.find()` 方法定位），或使用具名常量配合详细注释说明其用途和计算逻辑。
-   **模板抽象**：提取公共的系统提示部分为独立常量，通过字符串拼接或模板继承机制生成各类型模板，减少重复代码。
-   **建立国际化框架**：使用字典或类结构管理多语言模板（如 `QWENIMAGE_LAYERED_CAPTION_PROMPT["en"]`），或采用 gettext 等成熟方案。
-   **配置化标签**：将 `vision_start`、`image_pad`、`vision_end` 等标签提取为全局配置常量，便于统一管理和未来修改。
-   **添加校验函数**：提供 `validate_template()` 函数检查模板格式和占位符完整性，确保模板在使用前符合预期。
-   **补充类型注解**：为所有常量添加类型注解（如 `QWENIMAGE_PROMPT_TEMPLATE: str`），提升代码可读性。
-   **错误处理设计**：在模板访问模块中增加异常类（如 `TemplateError`），对不合法访问抛出明确异常。

## 其它





### 设计目标与约束

**设计目标**：
- 集中管理QwenImage所有pipeline变体的提示模板，确保模板格式一致性
- 支持4种不同的图像处理场景：纯文本生成、单图像编辑、多图像编辑、分层标注
- 提供多语言支持（英文、中文）的图像标注提示

**约束条件**：
- 模板字符串中的索引常量（如`START_IDX`）必须与实际tokenizer配置精确匹配
- 所有模板必须遵循`<|im_start|>`、`<|vision_start|>`等特殊标记的闭合规范
- 图像pad占位符`<|image_pad|>`的数量由外部调用方根据实际图像数量动态插入

### 错误处理与异常设计

**预期异常场景**：
- 模板索引越界：`QWENIMAGE_PROMPT_TEMPLATE_START_IDX`与实际模板字符串长度不匹配时，可能导致tokenizer生成错误的attention mask
- 格式不匹配：外部代码未按模板要求插入图像占位符，导致模型输入格式错误

**处理方式**：
- 本模块为纯数据定义模块，不包含运行时错误处理逻辑
- 错误检测应在调用方（pipeline代码）进行，调用前需验证模板格式完整性

### 数据流与状态机

**数据流**：
```
外部调用方（Pipeline）
    ↓ 传入用户文本/图像
    ↓ 选择对应模板
Prompt Template（模板字符串）
    ↓ 格式化用户输入（str.format()）
    ↓ 拼接特殊标记（im_start, vision_start等）
    ↓ 传入Tokenizer/Model
```

**状态说明**：
- 本模块不维护状态，为无状态模块
- 模板使用方式分为静态替换（用户文本直接替换`{}`）和动态扩展（多图像场景需预先插入多个`img_template`）

### 外部依赖与接口契约

**依赖关系**：
- 无Python代码依赖
- 依赖外部配置：Tokenizer必须支持`im_start`、`vision_start`、`image_pad`等特殊token
- 依赖模型配置：`START_IDX`值需与模型词汇表配置一致

**接口契约**：
- 所有模板变量通过`str.format()`方法注入用户输入
- 调用方必须保证传入字符串不包含`{}`占位符（避免格式化冲突）
- 图像场景下，调用方需自行处理`<|image_pad|>`的重复次数（代表图像embedding数量）

### 性能考虑与扩展性

**性能考量**：
- 模板定义为Python字符串常量，加载时无额外计算开销
- 模板字符串长度较小（<500字符），内存占用可忽略

**扩展性**：
- 新增pipeline变体时，只需定义新的模板变量（遵循命名规范`QWENIMAGE_{场景}_PROMPT_TEMPLATE`）
- 新增语言支持时，只需添加对应的`{场景}_PROMPT_{LANG}`变量

### 安全与合规

**安全检查**：
- 模板中不包含用户可控制的内容，仅包含系统指令
- 用户输入通过`str.format()`直接替换，无执行风险

**合规性**：
- 符合Apache 2.0开源许可协议要求
- 模板内容为通用图像描述指令，无特定敏感领域限制

### 测试策略

**测试要点**：
- 验证模板字符串格式正确性：检查`<|im_start|>`等标记是否成对出现
- 验证索引常量准确性：`START_IDX`指向的位置是否为`assistant`角色开始的正确位置
- 验证格式化功能：使用示例输入执行`str.format()`不抛出异常
- 回归测试：新增或修改模板后，验证模型输出质量未下降

### 配置与部署说明

**配置要求**：
- 部署时需确保与HuggingFace Transformers库版本兼容
- 需确保模型权重包含`im_start`、`vision_start`等特殊token

**部署方式**：
- 作为QwenImage库的子模块直接导入使用
- 无独立运行能力，仅提供数据定义


    