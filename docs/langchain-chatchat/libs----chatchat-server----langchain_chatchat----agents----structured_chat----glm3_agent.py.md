
# `Langchain-Chatchat\libs\chatchat-server\langchain_chatchat\agents\structured_chat\glm3_agent.py` 详细设计文档

该文件是一个用于创建 ChatGLM3-6B 代理的模块，基于 LangChain 框架修改而来。它提供了工具渲染功能和结构化代理创建功能，支持将工具描述格式化为 JSON 并绑定到语言模型，以实现带工具调用的智能代理。

## 整体流程

```mermaid
graph TD
A[开始] --> B[render_glm3_json 渲染工具列表]
B --> C{tools_renderer 是否为 None?}
C -- 是 --> D[使用默认的 render_glm3_json]
C -- 否 --> E[使用自定义 tools_renderer]
D --> F[create_structured_glm3_chat_agent 创建代理]
E --> F
F --> G{验证 prompt 变量}
G -- 缺少变量 --> H[抛出 ValueError]
G -- 验证通过 --> I[部分填充 prompt]
I --> J{stop_sequence 设置}
J -- True --> K[添加 <|observation|> 停止符]
J -- False --> L[不添加停止符]
J -- List --> M[使用自定义停止符列表]
K --> N[绑定停止符到 LLM]
L --> N
M --> N
N --> O[构建 Runnable 序列]
O --> P[format_to_platform_tool_messages 格式化中间步骤]
P --> Q[执行 prompt 和 LLM]
Q --> R[PlatformToolsAgentOutputParser 解析输出]
R --> S[返回 AgentAction 或 AgentFinish]
```

## 类结构

```
模块: langchain_chatchat.agents
├── render_glm3_json (工具渲染函数)
└── create_structured_glm3_chat_agent (代理工厂函数)
```

## 全局变量及字段


### `logger`
    
用于记录日志的全局变量

类型：`logging.Logger`
    


    

## 全局函数及方法



### `render_glm3_json`

该函数是一个工具渲染函数，接收工具列表（List[BaseTool]）作为输入，遍历每个工具获取其参数模式（schema），从工具描述中提取有效信息，处理参数定义（移除冗余字段），最终将所有工具转换为符合ChatGLM3模型要求的JSON格式字符串输出。

参数：

-  `tools`：`List[BaseTool]`，待渲染的工具列表，每个元素为一个工具实例

返回值：`str`，JSON格式的工具描述字符串，多个工具之间用换行符分隔

#### 流程图

```mermaid
flowchart TD
    A[开始 render_glm3_json] --> B[初始化空列表 tools_json]
    B --> C{遍历 tools 中的每个 tool}
    C -->|是| D[获取 tool.args_schema]
    D --> E{tool.args_schema 存在?}
    E -->|是| F[调用 model_schema 获取完整schema]
    E -->|否| G[设置 tool_schema 为空字典]
    F --> H
    G --> H[提取 description]
    H --> I{tool.description 包含 ' - '?]
    I -->|是| J[取 " - " 后的第二部分并strip]
    I -->|否| K[直接使用原 description]
    J --> L
    K --> L[处理 parameters]
    L --> M[遍历 tool_schema.properties]
    M --> N{过滤条件: sub_k != 'title'}
    N --> O[构建 parameters 字典]
    O --> P[构建简化配置 simplified_config_langchain]
    P --> Q[将配置添加到 tools_json]
    Q --> C
    C -->|否| R[遍历结束]
    R --> S[拼接 JSON 字符串]
    S --> T[返回结果]
```

#### 带注释源码

```python
def render_glm3_json(tools: List[BaseTool]) -> str:
    """
    将工具列表渲染为JSON格式字符串，供ChatGLM3模型使用
    
    参数:
        tools: List[BaseTool], 待渲染的工具列表
        
    返回:
        str: JSON格式的工具描述字符串
    """
    # 1. 初始化存储渲染结果的列表
    tools_json = []
    
    # 2. 遍历每个工具
    for tool in tools:
        # 3. 获取工具的参数schema，若无schema则为空字典
        #    model_schema 来自 chatchat.server.pydantic_v1，用于生成Pydantic模型的JSON schema
        tool_schema = model_schema(tool.args_schema) if tool.args_schema else {}
        
        # 4. 处理工具描述
        #    工具描述通常格式为 "工具名 - 描述信息"，需要提取第二部分
        #    例如: "weather_tool - 查询天气信息的工具" -> "查询天气信息的工具"
        if tool.description and " - " in tool.description:
            # 按 " - " 分割，取第二部分并去除首尾空格
            description = tool.description.split(" - ")[1].strip()
        else:
            # 若无 " - " 格式，直接使用原描述
            description = tool.description
        
        # 5. 处理工具参数定义
        #    从schema的properties中提取参数信息，并过滤掉'title'字段
        #    title字段在JSON Schema中用于描述参数名称，不需要传递给模型
        parameters = {
            k: {sub_k: sub_v for sub_k, sub_v in v.items() if sub_k != "title"}
            for k, v in tool_schema.get("properties", {}).items()
        }
        
        # 6. 构建简化的工具配置字典
        #    仅包含模型需要的核心字段：name, description, parameters
        simplified_config_langchain = {
            "name": tool.name,          # 工具名称
            "description": description,  # 工具描述
            "parameters": parameters,   # 工具参数定义
        }
        
        # 7. 将配置添加到结果列表
        tools_json.append(simplified_config_langchain)
    
    # 8. 将每个工具配置转换为JSON字符串，用换行符连接后返回
    #    indent=4: 缩进4个空格
    #    ensure_ascii=False: 允许非ASCII字符（如中文）正常显示
    return "\n".join(
        [json.dumps(tool, indent=4, ensure_ascii=False) for tool in tools_json]
    )
```



### `create_structured_glm3_chat_agent`

创建并返回一个可运行的工具调用代理，用于与ChatGLM3-6B模型交互。该代理通过提示模板、LLM和输出解析器构建完整的工具调用流程，支持平台工具消息格式化和停止序列控制。

参数：

- `llm`：`BaseLanguageModel`，用于执行语言模型推理
- `tools`：`Sequence[BaseTool]` ，代理可访问的工具序列
- `prompt`：`ChatPromptTemplate`，必须包含 `tools` 和 `agent_scratchpad` 输入变量的提示模板
- `tools_renderer`：`ToolsRenderer = render_glm3_json`，控制如何将工具转换为字符串传递给LLM，默认使用JSON格式渲染
- `stop_sequence`：`Union[bool, List[str]] = True`，停止序列控制。True 时添加 `<|observation|>` 作为停止标记，False 不添加，自定义列表使用提供的字符串作为停止标记
- `llm_with_platform_tools`：`List[Dict[str, Any]] = []`，平台工具的字典列表，长度≥0

返回值：`Runnable`，表示代理的可运行序列，接受与提示模板相同的输入变量，返回 `AgentAction` 或 `AgentFinish`

#### 流程图

```mermaid
flowchart TD
    A[开始创建代理] --> B{检查prompt是否包含必需变量}
    B -->|缺少变量| C[抛出ValueError异常]
    B -->|变量完整| D[partial填充tools和tool_names]
    D --> E{stop_sequence是否为True}
    E -->|True| F[设置stop为<|observation|>]
    E -->|自定义列表| G[使用自定义stop列表]
    E -->|False| H[不添加stop]
    F --> I[llm.bind绑定stop]
    G --> I
    H --> I
    I --> J[构建Runnable序列]
    J --> K[assign添加agent_scratchpad]
    K --> L[依次执行: prompt → llm_with_stop → PlatformToolsAgentOutputParser]
    L --> M[返回agent Runnable]
```

#### 带注释源码

```python
def create_structured_glm3_chat_agent(
        llm: BaseLanguageModel,              # 语言模型实例
        tools: Sequence[BaseTool],           # 可用工具序列
        prompt: ChatPromptTemplate,          # 提示模板（需包含tools和agent_scratchpad变量）
        tools_renderer: ToolsRenderer = render_glm3_json,  # 工具渲染函数，默认JSON格式
        *,                                    # 以下为关键字参数
        stop_sequence: Union[bool, List[str]] = True,  # 停止序列配置
        llm_with_platform_tools: List[Dict[str, Any]] = [],  # 平台工具列表
) -> Runnable:
    """Create an agent that uses tools.

    Args:
        llm: LLM to use as the agent.
        tools: Tools this agent has access to.
        prompt: The prompt to use, must have input keys
            `tools`: contains descriptions for each tool.
            `agent_scratchpad`: contains previous agent actions and tool outputs.
        tools_renderer: This controls how the tools are converted into a string and
            then passed into the LLM. Default is `render_text_description`.
        stop_sequence: bool or list of str.
            If True, adds a stop token of "<|observation|>" to avoid hallucinates.
            If False, does not add a stop token.
            If a list of str, uses the provided list as the stop tokens.

            Default is True. You may to set this to False if the LLM you are using
            does not support stop sequences.
        llm_with_platform_tools: length ge 0 of dict tools for platform

    Returns:
        A Runnable sequence representing an agent. It takes as input all the same input
        variables as the prompt passed in does. It returns as output either an
        AgentAction or AgentFinish.

    """
    # 验证prompt是否包含必需的变量：tools和agent_scratchpad
    missing_vars = {"tools", "agent_scratchpad"}.difference(
        prompt.input_variables + list(prompt.partial_variables)
    )
    if missing_vars:
        raise ValueError(f"Prompt missing required variables: {missing_vars}")

    # 使用tools_renderer将工具列表渲染为字符串，并partial填充到prompt中
    prompt = prompt.partial(
        tools=tools_renderer(list(tools)),  # 将工具渲染为JSON字符串
        tool_names=", ".join([t.name for t in tools]),  # 工具名称逗号分隔
    )
    
    # 处理停止序列配置
    if stop_sequence:
        # 根据stop_sequence参数设置停止标记
        stop = ["<|observation|>"] if stop_sequence is True else stop_sequence
        llm_with_stop = llm.bind(stop=stop)  # 绑定停止序列到LLM
    else:
        llm_with_stop = llm  # 不添加停止序列

    # 构建Runnable代理序列
    agent = (
            RunnablePassthrough.assign(
                # 将中间步骤格式化为平台工具消息
                agent_scratchpad=lambda x: format_to_platform_tool_messages(x["intermediate_steps"]),
            )
            | prompt                    # 首先应用prompt模板
            | llm_with_stop            # 然后调用带停止序列的LLM
            | PlatformToolsAgentOutputParser(instance_type="glm3")  # 最后解析输出
    )
    return agent  # 返回可运行的代理
```

## 关键组件





### 工具渲染组件 (render_glm3_json)

将BaseTool列表转换为JSON格式字符串，供LLM理解工具的结构和参数

### 代理工厂函数 (create_structured_glm3_chat_agent)

创建并返回一个Runnable序列的代理，用于执行结构化的GLM3聊天代理，支持工具调用和停止序列控制

### 平台工具消息格式化 (format_to_platform_tool_messages)

将中间步骤（代理行为和工具输出）格式化为平台工具消息，供代理的scratchpad使用

### GLM3输出解析器 (PlatformToolsAgentOutputParser)

解析LLM输出，识别平台工具调用，返回AgentAction或AgentFinish

### 结构化GLM3聊天输出解析器 (StructuredGLM3ChatOutputParser)

将LLM输出解析为结构化的代理行为，支持工具参数提取

### 提示词部分变量管理

通过prompt.partial()方法动态注入tools和tool_names变量，实现工具描述的运行时更新

### 停止序列绑定

根据stop_sequence参数配置LLM的停止token，避免LLM产生幻觉输出

### LangChainRunnable管道

通过RunnablePassthrough、prompt和LLM的链式组合，构建完整的代理执行流程



## 问题及建议



### 已知问题

-   **硬编码的停止符**: 代码中硬编码 `<|observation|>` 作为默认停止符，缺乏灵活性，不支持自定义平台特定的停止符
-   **工具描述解析风险**: `render_glm3_json` 中使用 `tool.description.split(" - ")[1]` 解析描述，若工具描述格式不符预期会触发 `IndexError`，缺少异常处理
-   **未使用的参数**: `llm_with_platform_tools` 参数被接收但在函数体中完全未使用，造成接口冗余
-   **日志记录缺失**: 导入了 `logger` 但代码中未使用任何日志记录，调试困难
-   **硬编码输出解析器类型**: `PlatformToolsAgentOutputParser` 的 `instance_type` 被硬编码为 `"glm3"`，缺乏可配置性
-   **类型注解不一致**: `llm_with_platform_tools` 文档说明 "length ge 0"，但类型标注未体现此约束
-   **参数清理逻辑冗余**: 参数清理使用了 `if sub_k != "title"` 的黑名单方式，更易维护的方式是使用白名单
-   **缺少输入验证**: 对 `intermediate_steps` 传递给 `format_to_platform_tool_messages` 前缺少类型和结构验证
-   **partial_variables 处理不完整**: 只检查了 `input_variables` 和 `partial_variables` 是否包含必需变量，但未处理其他可能的缺失变量场景

### 优化建议

-   **移除未使用参数或实现功能**: 删除 `llm_with_platform_tools` 参数，或在 agent 创建逻辑中正确集成平台工具
-   **添加错误处理和默认值**: 为工具描述解析添加 try-except 和默认值，避免解析失败导致整个 agent 创建失败
-   **引入日志记录**: 在关键路径添加 logger.debug/info 记录，便于生产环境调试
-   **参数化输出解析器类型**: 将 `instance_type` 作为可选参数，允许创建不同类型的 agent
-   **统一类型注解**: 使用 `Sequence` 替代 `List` 增强类型泛化性，确保文档与代码一致
-   **增强参数验证**: 对 stop_sequence 类型进行运行时验证，确保为 bool 或 List[str]
-   **使用白名单过滤参数**: 将参数过滤改为显式白名单方式，提高代码可读性和安全性
-   **添加单元测试**: 针对工具渲染、参数验证等关键逻辑补充测试用例

## 其它




### 设计目标与约束

本模块旨在为ChatGLM3-6B模型创建一个结构化的代理框架，支持工具调用能力。设计约束包括：1) 必须遵循LangChain的Runnable接口规范；2) prompt必须包含`tools`和`agent_scratchpad`变量；3) 工具描述生成需要从原始description中提取简化版本；4) 支持自定义停止序列以避免模型幻觉。

### 错误处理与异常设计

代码中的错误处理主要包括：1) 在`create_structured_glm3_chat_agent`函数中检查prompt是否缺少必需变量`tools`和`agent_scratchpad`，若缺失则抛出`ValueError`；2) LangChain的`OutputParserException`用于处理输出解析失败的情况；3) `PlatformToolsAgentOutputParser`内部可能抛出各种解析异常。错误传播机制遵循LangChain的统一异常体系。

### 数据流与状态机

数据输入流：llm（语言模型）、tools（工具列表）、prompt（提示模板）、tools_renderer（工具渲染器）、stop_sequence（停止序列）、llm_with_platform_tools（平台工具）。处理流程：1) 验证prompt变量完整性；2) 使用tools_renderer渲染工具为JSON字符串；3) 绑定停止序列到LLM；4) 通过RunnablePassthrough获取中间步骤并格式化为消息；5) 传递给LLM生成响应；6) 通过PlatformToolsAgentOutputParser解析输出。输出状态为AgentAction（需要执行工具）或AgentFinish（完成执行）。

### 外部依赖与接口契约

主要外部依赖包括：1) `langchain`核心模块：BaseLanguageModel、BaseTool、ChatPromptTemplate、AgentAction、AgentFinish、OutputParserException；2) `langchain_core.runnables`：Runnable、RunnablePassthrough；3) `chatchat.server.pydantic_v1`：Field、model_schema、typing；4) `chatchat.utils`：build_logger；5) `langchain_chatchat.agents.format_scratchpad.all_tools`：format_to_platform_tool_messages；6) `langchain_chatchat.agents.output_parsers`：StructuredGLM3ChatOutputParser、PlatformToolsAgentOutputParser。接口契约要求tools必须实现BaseTool接口，prompt必须支持partial变量更新。

### 性能考虑与优化空间

当前实现中潜在的性能问题：1) 工具描述渲染在每次agent创建时执行，如果工具集合固定可考虑缓存；2) lambda函数`lambda x: format_to_platform_tool_messages(x["intermediate_steps"])`每次调用都会创建新的函数对象；3) `tools_renderer`默认实现遍历所有工具并做JSON处理。优化建议：1) 对于固定工具集，可预先渲染工具描述并缓存；2) 可使用`functools.partial`替代lambda；3) 工具schema的model_schema调用可以考虑缓存。

### 安全性考虑

安全相关设计：1) 工具描述提取时使用split(" - ")[1]假设description格式为"tool_name - description"，需确保输入格式安全；2) JSON序列化使用ensure_ascii=False支持Unicode；3) stop_sequence参数可自定义，需验证用户输入的停止符不会导致安全问题；4) 工具参数过滤排除title字段防止信息泄露。

### 配置参数详细说明

| 参数名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| llm | BaseLanguageModel | 是 | - | 用于代理的语言模型实例 |
| tools | Sequence[BaseTool] | 是 | - | 代理可访问的工具列表 |
| prompt | ChatPromptTemplate | 是 | - | 提示模板，需包含tools和agent_scratchpad变量 |
| tools_renderer | ToolsRenderer | 否 | render_glm3_json | 工具转字符串的渲染函数 |
| stop_sequence | Union[bool, List[str]] | 否 | True | 停止序列设置，True使用默认"<|observation|>" |
| llm_with_platform_tools | List[Dict[str, Any]] | 否 | [] | 平台工具列表 |

### 使用示例与调用模式

基本调用示例：
```python
from langchain.chains import LLMChain
from chatchat.agents import create_structured_glm3_chat_agent

# 创建代理
agent = create_structured_glm3_chat_agent(
    llm=llm,
    tools=[tool1, tool2],
    prompt=prompt_template,
    stop_sequence=True,
    llm_with_platform_tools=[]
)

# 执行代理
result = agent.invoke({"input": "用户问题"})
```

### 版本历史与变更记录

本文件标注为"modified version for ChatGLM3-6B"，基于LangChain仓库的glm3_agent.py修改而来。主要变更：将原始GLM3代理适配为支持平台工具的版本，引入了PlatformToolsAgentOutputParser和format_to_platform_tool_messages集成。

### 测试策略建议

测试应覆盖：1) 工具描述渲染函数render_glm3_json的JSON输出格式；2) create_structured_glm3_chat_agent对缺失变量的验证；3) 停止序列的正确绑定；4) 代理链的完整执行流程；5) 不同tools_renderer的兼容性；6) 输出解析器对各种LLM响应的处理。

### 部署与集成注意事项

部署要求：1) 依赖LangChain 0.1+版本和langchain-core；2) 需要正确安装chatchat和langchain_chatchat包；3) 环境变量需包含LLM服务配置；4) 建议使用相同的stop_sequence设置以保持一致性；5) 集成时应确保prompt模板包含正确的输入变量。

    