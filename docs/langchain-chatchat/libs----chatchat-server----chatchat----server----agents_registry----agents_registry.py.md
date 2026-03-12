
# `Langchain-Chatchat\libs\chatchat-server\chatchat\server\agents_registry\agents_registry.py` 详细设计文档

这是一个Agent工厂注册函数，根据不同的agent_type（glm3, qwen, platform-agent, structured-chat-agent, default, openai-functions, openai-tools, tool-calling, platform-knowledge-mode）创建相应的LangChain AgentExecutor，支持多种大语言模型和工具集成。

## 整体流程

```mermaid
graph TD
    A[开始 agents_registry] --> B{agent_type == 'glm3'}
    B -- 是 --> C[创建GLM3结构化聊天Agent]
    B -- 否 --> D{agent_type == 'qwen'}
    D -- 是 --> E[创建Qwen聊天Agent]
    D -- 否 --> F{agent_type == 'platform-agent'}
    F -- 是 --> G[创建平台工具Agent]
    F -- 否 --> H{agent_type == 'structured-chat-agent'}
    H -- 是 --> I[创建结构化聊天Agent]
    H -- 否 --> J{agent_type == 'default'}
    J -- 是 --> K[创建默认Agent]
    J -- 否 --> L{agent_type == 'openai-functions'}
    L -- 是 --> M[创建OpenAI函数Agent]
    L -- 否 --> N{agent_type in ('openai-tools', 'tool-calling')}
    N -- 是 --> O[创建OpenAI工具Agent]
    N -- 否 --> P{agent_type == 'platform-knowledge-mode'}
    P -- 是 --> Q[创建平台知识模式Agent]
    P -- 否 --> R[抛出ValueError: 不支持的agent类型]
    C --> S[返回PlatformToolsAgentExecutor]
    E --> S
    G --> S
    I --> S
    K --> T[返回AgentExecutor]
    M --> T
    O --> T
    Q --> S
```

## 类结构

```
agents_registry (函数模块)
└── 无类层次结构，仅包含一个工厂函数
```

## 全局变量及字段


### `asyncio`
    
Python标准库异步编程模块

类型：`module`
    


### `sys`
    
Python标准库系统相关模块

类型：`module`
    


### `AsyncExitStack`
    
异步上下文管理器，用于管理异步资源的退出

类型：`class`
    


### `RunnableMultiActionAgent`
    
LangChain的多动作Agent基类，支持处理多个动作

类型：`class`
    


### `SystemMessage`
    
系统消息类型，用于定义Agent的系统提示

类型：`class`
    


### `AIMessage`
    
AI回复消息类型

类型：`class`
    


### `ChatPromptTemplate`
    
聊天提示模板，用于构建对话格式的提示

类型：`class`
    


### `HumanMessagePromptTemplate`
    
人类消息提示模板，用于用户输入

类型：`class`
    


### `MessagesPlaceholder`
    
消息占位符，用于动态插入消息历史

类型：`class`
    


### `BaseModel`
    
Pydantic基础模型类，用于数据验证

类型：`class`
    


### `get_prompt_template_dict`
    
获取指定类型和Agent的提示模板字典

类型：`function`
    


### `PlatformToolsAgentExecutor`
    
平台工具Agent执行器，用于执行平台级工具

类型：`class`
    


### `create_prompt_glm3_template`
    
创建GLM3模型的提示模板

类型：`function`
    


### `create_prompt_structured_react_template`
    
创建结构化ReAct提示模板

类型：`function`
    


### `create_prompt_platform_template`
    
创建平台Agent提示模板

类型：`function`
    


### `create_prompt_gpt_tool_template`
    
创建GPT工具调用提示模板

类型：`function`
    


### `create_prompt_platform_knowledge_mode_template`
    
创建平台知识模式提示模板

类型：`function`
    


### `create_structured_glm3_chat_agent`
    
创建结构化GLM3聊天Agent

类型：`function`
    


### `create_platform_knowledge_agent`
    
创建平台知识模式Agent

类型：`function`
    


### `create_platform_tools_agent`
    
创建平台工具Agent

类型：`function`
    


### `create_qwen_chat_agent`
    
创建Qwen聊天Agent

类型：`function`
    


### `create_chat_agent`
    
创建通用聊天Agent

类型：`function`
    


### `Any`
    
任意类型标注

类型：`type`
    


### `AsyncIterable`
    
异步可迭代对象类型

类型：`type`
    


### `Awaitable`
    
可等待对象类型

类型：`type`
    


### `Callable`
    
可调用对象类型

类型：`type`
    


### `Dict`
    
字典类型

类型：`type`
    


### `List`
    
列表类型

类型：`type`
    


### `Optional`
    
可选类型

类型：`type`
    


### `Sequence`
    
序列类型

类型：`type`
    


### `Tuple`
    
元组类型

类型：`type`
    


### `Type`
    
类型对象

类型：`type`
    


### `Union`
    
联合类型

类型：`type`
    


### `cast`
    
类型强制转换函数

类型：`function`
    


### `hub`
    
LangChain提示模板中心

类型：`module`
    


### `AgentExecutor`
    
Agent执行器，用于运行Agent并执行工具

类型：`class`
    


### `create_openai_tools_agent`
    
创建OpenAI工具Agent

类型：`function`
    


### `create_tool_calling_agent`
    
创建工具调用Agent

类型：`function`
    


### `BaseCallbackHandler`
    
回调处理器基类，用于处理Agent执行过程中的事件

类型：`class`
    


### `BaseLanguageModel`
    
基础语言模型接口

类型：`class`
    


### `BaseTool`
    
基础工具接口

类型：`class`
    


### `MCPStructuredTool`
    
MCP结构化工具，用于Model Context Protocol

类型：`class`
    


### `global function.agents_registry`
    
Agent工厂函数，根据agent_type参数创建并返回对应的Agent执行器实例，支持多种主流LLM Agent类型

类型：`function`
    
    

## 全局函数及方法



### `agents_registry`

该函数是一个代理工厂（Agent Factory）函数，根据传入的 `agent_type` 参数动态创建不同类型的代理执行器（AgentExecutor），支持 GLM3、Qwen、Platform-Agent、Structured-Chat-Agent、Default、OpenAI-Functions、OpenAI-Tools、Tool-Calling、Platform-Knowledge-Mode 等多种代理类型，并返回配置好的 `PlatformToolsAgentExecutor` 或 `AgentExecutor` 实例。

参数：

- `agent_type`：`str`，代理类型标识符，用于决定创建哪种类型的代理
- `llm`：`BaseLanguageModel`，语言模型实例，用于代理进行推理和生成
- `llm_with_platform_tools`：`List[Dict[str, Any]] = []`，包含平台工具的语言模型配置列表
- `tools`：`Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = []`，可供代理调用的工具序列
- `mcp_tools`：`Sequence[MCPStructuredTool] = []`，MCP（Model Context Protocol）结构化工具序列
- `callbacks`：`List[BaseCallbackHandler] = []`，回调处理器列表，用于监控代理执行过程
- `verbose`：`bool = False`，是否输出详细日志信息
- `**kwargs`：`Any`，其他可选关键字参数，如 `current_working_directory`、`FUNCTIONS_PREFIX`、`FUNCTIONS_SUFFIX` 等

返回值：`Union[PlatformToolsAgentExecutor, AgentExecutor]`，返回配置好的代理执行器实例，用于执行代理任务

#### 流程图

```mermaid
flowchart TD
    A[开始 agents_registry] --> B{agent_type == 'glm3'?}
    B -->|Yes| C[创建 GLM3 代理模板]
    B -->|No| D{agent_type == 'qwen'?}
    D -->|Yes| E[创建 Qwen 代理模板<br/>llm.streaming = False]
    D -->|No| F{agent_type == 'platform-agent'?}
    F -->|Yes| G[创建 Platform Agent 模板]
    F -->|No| H{agent_type == 'structured-chat-agent'?}
    H -->|Yes| I[创建 Structured Chat Agent 模板]
    H -->|No| J{agent_type == 'default'?}
    J -->|Yes| K[创建 Default 代理模板]
    J -->|No| L{agent_type == 'openai-functions'?}
    L -->|Yes| M[创建 OpenAI Functions 代理<br/>部分填充 tool_names]
    L -->|No| N{agent_type in ('openai-tools', 'tool-calling')?}
    N -->|Yes| O[创建 OpenAI Tools/Tool-Calling 代理]
    N -->|No| P{agent_type == 'platform-knowledge-mode'?}
    P -->|Yes| Q[创建 Platform Knowledge Mode 代理]
    P -->|No| R[抛出 ValueError 不支持的代理类型]
    
    C --> S[创建 PlatformToolsAgentExecutor]
    E --> S
    G --> S
    I --> S
    K --> T[创建 AgentExecutor]
    M --> T
    O --> T
    Q --> S
    
    S --> U[返回 PlatformToolsAgentExecutor]
    T --> V[返回 AgentExecutor]
    
    R --> W[结束 - 抛出异常]
```

#### 带注释源码

```python
def agents_registry(
        agent_type: str,
        llm: BaseLanguageModel,
        llm_with_platform_tools: List[Dict[str, Any]] = [],
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]] = [],
        mcp_tools: Sequence[MCPStructuredTool] = [],
        callbacks: List[BaseCallbackHandler] = [],
        verbose: bool = False,
        **kwargs: Any,
):
    """
    代理注册工厂函数，根据 agent_type 创建不同类型的代理执行器
    
    Args:
        agent_type: 代理类型标识符
        llm: 语言模型实例
        llm_with_platform_tools: 带有平台工具的LLM配置列表
        tools: 可用工具序列
        mcp_tools: MCP结构化工具序列
        callbacks: 回调处理器列表
        verbose: 是否详细输出
        **kwargs: 其他可选参数
    
    Returns:
        配置好的 AgentExecutor 或 PlatformToolsAgentExecutor 实例
    """
    
    # Write any optimized method here.
    # TODO agent params of PlatformToolsAgentExecutor or AgentExecutor  enable return_intermediate_steps=True,
    
    # ============================================================
    # 分支1: GLM3 系列模型代理
    # ============================================================
    if "glm3" == agent_type:
        # 从提示词模板字典中获取 GLM3 的模板配置
        template = get_prompt_template_dict("action_model", agent_type)
        # 创建 GLM3 专用的提示词模板
        prompt = create_prompt_glm3_template(agent_type, template=template)
        # 创建结构化的 GLM3 聊天代理
        agent = create_structured_glm3_chat_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            llm_with_platform_tools=llm_with_platform_tools
        )

        # 使用 PlatformToolsAgentExecutor 包装代理
        agent_executor = PlatformToolsAgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,  # 返回中间步骤结果
        )
        return agent_executor
    
    # ============================================================
    # 分支2: Qwen 系列模型代理
    # ============================================================
    elif "qwen" == agent_type:
        # Qwen 代理不支持流式输出，强制关闭
        llm.streaming = False

        # 获取 Qwen 的提示词模板
        template = get_prompt_template_dict("action_model", agent_type)
        prompt = create_prompt_structured_react_template(agent_type, template=template)
        # 创建 Qwen 聊天代理
        agent = create_qwen_chat_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            llm_with_platform_tools=llm_with_platform_tools
        )

        # 使用 PlatformToolsAgentExecutor 包装
        agent_executor = PlatformToolsAgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
        )
        return agent_executor
    
    # ============================================================
    # 分支3: Platform Agent 代理
    # ============================================================
    elif "platform-agent" == agent_type:
        # 获取平台代理的提示词模板
        template = get_prompt_template_dict("action_model", agent_type)
        prompt = create_prompt_platform_template(agent_type, template=template)
        # 创建平台工具代理
        agent = create_platform_tools_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            llm_with_platform_tools=llm_with_platform_tools
        )

        agent_executor = PlatformToolsAgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
        )
        return agent_executor
    
    # ============================================================
    # 分支4: Structured Chat Agent 代理
    # ============================================================
    elif agent_type == 'structured-chat-agent':
        # 获取结构化聊天代理的提示词模板
        template = get_prompt_template_dict("action_model", agent_type)
        prompt = create_prompt_structured_react_template(agent_type, template=template)
        # 创建通用聊天代理
        agent = create_chat_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            llm_with_platform_tools=llm_with_platform_tools
        )

        agent_executor = PlatformToolsAgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
        )
        return agent_executor
    
    # ============================================================
    # 分支5: Default 代理（单聊模式）
    # ============================================================
    elif agent_type == 'default':
        # 此代理用于单聊场景
        template = get_prompt_template_dict("action_model", "default")
        # 从系统提示词创建 ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=template.get("SYSTEM_PROMPT"))]
        )

        # 创建通用聊天代理
        agent = create_chat_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            llm_with_platform_tools=llm_with_platform_tools
        )

        # 使用标准 AgentExecutor（不是 PlatformToolsAgentExecutor）
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
            **kwargs,  # 传递额外参数
        )
        return agent_executor

    # ============================================================
    # 分支6: OpenAI Functions 代理
    # ============================================================
    elif agent_type == "openai-functions":
        # 仅使用工具的代理，支持历史消息
        template = get_prompt_template_dict("action_model", agent_type)
        prompt = create_prompt_gpt_tool_template(agent_type, template=template)

        # 预先部分填充 tool_names 变量
        prompt = prompt.partial(
            tool_names=", ".join([t.name for t in tools]),
        )
        
        # 创建 OpenAI 工具代理
        runnable = create_openai_tools_agent(llm, tools, prompt)
        
        # 创建多动作代理
        agent = RunnableMultiActionAgent(
            runnable=runnable,
            input_keys_arg=["input"],
            return_keys_arg=["output"],
            **kwargs,
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
            **kwargs,
        )
        return agent_executor
    
    # ============================================================
    # 分支7: OpenAI Tools / Tool-Calling 代理
    # ============================================================
    elif agent_type in ("openai-tools", "tool-calling"):
        # 仅使用工具的代理，不支持历史消息
        function_prefix = kwargs.get("FUNCTIONS_PREFIX")
        function_suffix = kwargs.get("FUNCTIONS_SUFFIX")
        
        # 构建消息模板：系统消息 + 用户输入 + AI响应 + Agent草稿板
        messages = [
            SystemMessage(content=cast(str, function_prefix)),
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=function_suffix),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        
        # 根据类型选择创建函数
        if agent_type == "openai-tools":
            runnable = create_openai_tools_agent(llm, tools, prompt)
        else:
            runnable = create_tool_calling_agent(llm, tools, prompt)
            
        agent = RunnableMultiActionAgent(
            runnable=runnable,
            input_keys_arg=["input"],
            return_keys_arg=["output"],
            **kwargs,
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
            **kwargs,
        )
        return agent_executor

    # ============================================================
    # 分支8: Platform Knowledge Mode 代理
    # ============================================================
    elif "platform-knowledge-mode" == agent_type:
        # 平台知识模式代理，支持 MCP 工具
        template = get_prompt_template_dict("action_model", agent_type)
        prompt = create_prompt_platform_knowledge_mode_template(agent_type, template=template)
        
        # 创建平台知识代理
        agent = create_platform_knowledge_agent(
            llm=llm,
            current_working_directory=kwargs.get("current_working_directory", "/tmp"),
            tools=tools,
            mcp_tools=mcp_tools,
            llm_with_platform_tools=llm_with_platform_tools,
            prompt=prompt
        )

        agent_executor = PlatformToolsAgentExecutor(
            agent=agent,
            tools=tools,
            mcp_tools=mcp_tools,  # 传递 MCP 工具
            verbose=verbose,
            callbacks=callbacks,
            return_intermediate_steps=True,
        )
        return agent_executor

    # ============================================================
    # 异常处理: 不支持的代理类型
    # ============================================================
    else:
        raise ValueError(
            f"Agent type {agent_type} not supported at the moment. Must be one of "
            "'tool-calling', 'openai-tools', 'openai-functions', "
            "'default','ChatGLM3','structured-chat-agent','platform-agent','qwen','glm3'"
        )
```

## 关键组件




### 多种代理类型支持（Multi-Agent Type Support）

代码通过agents_registry函数支持9种不同的代理类型：glm3、qwen、platform-agent、structured-chat-agent、default、openai-functions、openai-tools、tool-calling和platform-knowledge-mode，每种类型针对不同的LLM模型和应用场景进行优化。

### 平台工具代理执行器（PlatformToolsAgentExecutor）

使用PlatformToolsAgentExecutor作为主要代理执行器，支持return_intermediate_steps=True以返回中间步骤，提供verbose和callbacks支持，实现对代理执行过程的完整控制。

### 提示模板创建系统（Prompt Template Creation）

通过create_prompt_glm3_template、create_prompt_structured_react_template、create_prompt_platform_template、create_prompt_gpt_tool_template和create_prompt_platform_knowledge_mode_template等函数为不同代理类型动态创建提示模板，实现灵活的提示工程。

### 工具与MCP工具集成（Tools & MCP Tools Integration）

支持多种工具格式：Dict、BaseModel、Callable和BaseTool，同时通过MCPStructuredTool支持MCP（Model Context Protocol）结构化工具，实现与外部工具生态的集成。

### LLM与平台工具绑定（LLM with Platform Tools）

通过llm_with_platform_tools参数支持将平台特定工具绑定到LLM，允许代理访问平台扩展能力，实现更强大的工具调用功能。

### 结构化聊天代理（Structured Chat Agent）

create_chat_agent、create_platform_tools_agent、create_platform_knowledge_agent和create_qwen_chat_agent等函数提供结构化的聊天代理创建能力，支持复杂的对话流程和多工具协同。

### LangChain框架集成（LangChain Framework Integration）

深度集成LangChain生态，使用AgentExecutor、RunnableMultiActionAgent、create_openai_tools_agent和create_tool_calling_agent等组件，提供标准化的代理执行接口。


## 问题及建议



### 已知问题

- **可变默认参数**: `llm_with_platform_tools: List[Dict[str, Any]] = []` 使用了可变默认参数，这是Python反模式，可能导致意外的数据共享和修改
- **魔法字符串**: 代理类型使用硬编码的字符串比较（如"glm3"、"qwen"等），容易产生拼写错误且难以维护
- **重复代码**: 创建`PlatformToolsAgentExecutor`的代码块高度重复，每次都需要设置相同的参数（tools、verbose、callbacks、return_intermediate_steps=True）
- **错误信息不一致**: 错误信息中列出'ChatGLM3'但代码实际使用'glm3'，导致用户困惑
- **直接修改输入参数**: `llm.streaming = False` 直接修改传入的LLM对象，可能产生副作用
- **类型提示不完整**: 函数没有声明返回类型，应返回`AgentExecutor | PlatformToolsAgentExecutor`
- **kwargs处理不一致**: 部分分支将kwargs传递给AgentExecutor，部分分支没有传递，导致行为不一致
- **TODO未完成**: 代码中有关于`return_intermediate_steps=True`的TODO注释，表明有未完成的优化

### 优化建议

- 将可变默认参数改为`None`，在函数内部初始化为空列表
- 使用枚举或常量类定义代理类型，避免魔法字符串
- 提取重复的AgentExecutor创建逻辑为私有方法
- 修正错误信息中的代理类型名称，与实际代码保持一致
- 考虑使用配置对象而非直接修改llm对象的属性
- 添加显式的返回类型注解
- 统一kwargs的传递行为
- 清理TODO注释或完成相关优化工作

## 其它




### 设计目标与约束

该模块的核心设计目标是提供一个统一的Agent工厂，根据不同的agent_type动态创建适合的Agent Executor，支持多种大语言模型（LLM）和工具集成。设计约束包括：必须支持LangChain生态的Agent接口规范；每个Agent类型需要返回中间步骤（return_intermediate_steps=True）；qwen类型需要禁用流式输出；所有Agent必须支持callback机制。

### 错误处理与异常设计

当传入不支持的agent_type时，函数抛出ValueError异常，明确列出支持的Agent类型列表。函数内部依赖外部函数（如get_prompt_template_dict、create_*_template等）若失败会向上传播异常。建议在调用agents_registry前验证agent_type的有效性，并在UI层捕获ValueError异常向用户提示。

### 数据流与状态机

数据流为：输入agent_type、llm、tools等参数 → 根据agent_type选择对应的Agent创建流程 → 调用对应的prompt template生成函数 → 创建具体Agent对象 → 包装为AgentExecutor或PlatformToolsAgentExecutor → 返回执行器。无复杂状态机，仅根据agent_type进行分支处理。

### 外部依赖与接口契约

主要依赖包括：langchain.agents的AgentExecutor和RunnableMultiActionAgent；langchain_core的BaseLanguageModel、BaseTool、BaseCallbackHandler；langchain_chatchat各子模块的create_*_agent函数；pydantic的BaseModel。llm参数需实现BaseLanguageModel接口；tools需为Sequence[Union[Dict, Type[BaseModel], Callable, BaseTool]]类型；mcp_tools需为Sequence[MCPStructuredTool]类型。

### 性能考虑与限制

qwen类型会修改llm对象的streaming属性为False。PlatformToolsAgentExecutor和AgentExecutor均开启return_intermediate_steps=True，会增加内存占用。所有Agent创建后为同步对象，实际执行时为同步调用（未使用async def）。

### 安全性考虑

代码本身无直接安全风险，但需注意：llm_with_platform_tools参数可能包含敏感的平台工具权限配置；callback机制可能记录敏感交互数据；kwargs透传需确保不注入不安全参数。

### 配置与可扩展性

通过agent_type字符串扩展新Agent类型，只需在对应elif分支添加创建逻辑。kwargs字典允许传递额外配置参数（如current_working_directory、FUNCTIONS_PREFIX等）。工具列表支持多种格式（Dict/BaseModel/Callable/BaseTool），具有良好的扩展性。

### 日志与监控

代码中未包含显式日志记录，通过verbose参数控制AgentExecutor的详细输出。建议在生产环境集成Python logging模块记录Agent创建和执行过程。

### 版本兼容性

依赖langchain 0.1.x版本序列，create_openai_tools_agent和create_tool_calling_agent为较新API。部分Agent类型（glm3、qwen等）依赖特定模型行为，需确认LLM实现版本兼容性。

    