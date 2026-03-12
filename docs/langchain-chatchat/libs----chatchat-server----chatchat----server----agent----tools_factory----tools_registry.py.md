
# `Langchain-Chatchat\libs\chatchat-server\chatchat\server\agent\tools_factory\tools_registry.py` 详细设计文档

该代码是一个工具注册和上下文格式化的辅助模块，通过扩展langchain的BaseTool添加了title等额外字段支持，实现了工具自动注册到全局注册表的功能，并提供了将知识库文档输出格式化为LLM上下文字符串的工具函数。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[导入模块]
B --> C[patch BaseTool._parse_input]
C --> D[patch BaseTool._to_args_and_kwargs]
D --> E{调用regist_tool}
E -->|作为装饰器| F[wrapper函数]
E -->|直接调用| G[tool函数]
F --> H[执行partial_工具装饰]
G --> I[执行tool函数]
H --> J[_parse_tool解析]
I --> J
J --> K[添加到全局_TOOLS_REGISTRY]
K --> L[返回BaseTool实例]
L --> M{调用format_context}
M --> N[解析docs数据]
N --> O[构建source_documents列表]
O --> P{文档数量为0?}
P -->|是 --> Q[返回无文档提示]
P -->|否 --> R[拼接文档内容]
R --> S[返回格式化上下文]
```

## 类结构

```
模块级别
├── 全局变量
│   └── _TOOLS_REGISTRY (工具注册表字典)
├── Patch函数
│   ├── _new_parse_input (扩展BaseTool输入解析)
│   └── _new_to_args_and_kwargs (扩展参数转换)
├── 工具函数
│   ├── regist_tool (工具注册装饰器)
│   └── format_context (上下文格式化)
└── 导入的外部类型
    ├── BaseTool (langchain)
    ├── BaseModel (pydantic)
    └── DocumentWithVSId (知识库文档)
```

## 全局变量及字段


### `_TOOLS_REGISTRY`
    
全局工具注册表，存储已注册的工具实例，以工具名称（字符串）为键

类型：`Dict[str, BaseTool]`
    


### `BaseTool.Config.Extra`
    
Pydantic模型配置类，用于允许BaseTool工具类拥有额外的自定义字段（如title等）

类型：`Type[Extra]`
    
    

## 全局函数及方法



### `_new_parse_input`

该函数是 langchain `BaseTool._parse_input` 方法的补丁实现，用于将工具输入转换为 pydantic 模型。它首先获取工具的参数模式（args_schema），然后根据输入类型（字符串或字典）进行相应的验证或解析，最终返回符合 pydantic 模型定义的字典或原始字符串。

参数：

- `self`：`BaseTool`，LangChain 工具基类实例，隐式参数
- `tool_input`：`Union[str, Dict]`，工具输入，可以是字符串形式或字典形式

返回值：`Union[str, Dict[str, Any]]`，如果输入是字符串则返回原始字符串；如果输入是字典且有 args_schema，则返回验证后的字典，否则返回原始字典

#### 流程图

```mermaid
flowchart TD
    A[开始: _new_parse_input] --> B[获取 self.args_schema]
    B --> C{tool_input 是字符串?}
    C -->|是| D{input_args 存在?}
    C -->|否| E{input_args 存在?}
    D -->|是| F[使用 input_args 验证: {key_: tool_input}]
    D -->|否| G[返回原始 tool_input]
    E -->|是| I[使用 input_args.parse_obj 解析 tool_input]
    E -->|否| H[返回原始 tool_input]
    F --> J[返回原始 tool_input]
    I --> K[返回 result.dict()]
    H --> L[返回原始 tool_input]
```

#### 带注释源码

```python
def _new_parse_input(
    self,           # BaseTool 实例，隐式传入
    tool_input: Union[str, Dict],  # 工具输入，字符串或字典类型
) -> Union[str, Dict[str, Any]]:
    """Convert tool input to pydantic model."""
    # 从工具实例获取参数模式（pydantic 模型类）
    input_args = self.args_schema
    
    # 判断输入是否为字符串类型
    if isinstance(tool_input, str):
        # 如果存在参数模式定义
        if input_args is not None:
            # 获取参数模式的第一个字段名
            key_ = next(iter(input_args.__fields__.keys()))
            # 使用该字段验证字符串输入（仅验证，不保存结果）
            input_args.validate({key_: tool_input})
        # 字符串输入直接返回
        return tool_input
    else:
        # 输入为字典类型
        if input_args is not None:
            # 使用 pydantic 模型解析字典输入
            result = input_args.parse_obj(tool_input)
            # 返回解析后的字典形式
            return result.dict()
        # 无参数模式时返回原始字典
        return tool_input
```



### `_new_to_args_and_kwargs`

该函数是 `BaseTool` 类的 monkey patch 方法，用于将工具输入（tool_input）转换为位置参数元组和关键字参数字典，以保持与旧版 LangChain 工具调用的向后兼容性。特别处理了使用 `*args` 参数定义的工具场景。

参数：

- `self`：`BaseTool`，BaseTool 实例本身
- `tool_input`：`Union[str, Dict]`，工具输入，可以是字符串或字典形式

返回值：`Tuple[Tuple, Dict]`，包含位置参数元组和关键字参数字典的元组

#### 流程图

```mermaid
flowchart TD
    A[开始: tool_input] --> B{tool_input 是否为字符串?}
    B -->|是| C[返回 (tool_input,), {}]
    B -->|否| D{tool_input 中是否有 'args' 键?}
    D -->|否| E[返回 (), tool_input]
    D -->|是| F{args 是否为 None?}
    F -->|是| G[弹出 'args', 返回 (), tool_input]
    F -->|否| H{args 是否为 tuple?}
    H -->|是| I[弹出 'args', 返回 args, tool_input]
    H -->|否| J[返回 (), tool_input]
```

#### 带注释源码

```python
def _new_to_args_and_kwargs(self, tool_input: Union[str, Dict]) -> Tuple[Tuple, Dict]:
    # 用于向后兼容：如果 run_input 是字符串，
    # 作为位置参数传递
    if isinstance(tool_input, str):
        # 字符串输入作为单个位置参数返回
        return (tool_input,), {}
    else:
        # 对于使用 `*args` 参数定义的工具
        # args_schema 有一个名为 `args` 的字段
        # 它应该展开为实际的 *args
        # 例如：test_tools.test_named_tool_decorator_return_direct.search_api
        if "args" in tool_input:
            args = tool_input["args"]
            if args is None:
                # 如果 args 为 None，移除该键并返回空位置参数
                tool_input.pop("args")
                return (), tool_input
            elif isinstance(args, tuple):
                # 如果 args 是元组，展开为位置参数
                tool_input.pop("args")
                return args, tool_input
        # 默认情况：无位置参数，只有关键字参数
        return (), tool_input
```



### `regist_tool`

该函数是一个工具注册装饰器，用于包装 LangChain 的 `tool` 装饰器，实现自动将工具添加到全局注册表 `_TOOLS_REGISTRY` 中，并支持自定义标题、描述和参数模式等配置。

参数：

- `*args`：`Any`，可变位置参数，用于传递给 LangChain 的 `tool` 装饰器
- `title`：`str`，工具的标题，默认为空字符串，将自动从工具名称转换而来
- `description`：`str`，工具描述，默认为空字符串，将自动从函数文档字符串获取
- `return_direct`：`bool`，是否直接返回工具输出，默认为 `False`
- `args_schema`：`Optional[Type[BaseModel]]`，Pydantic 模型定义工具参数模式，默认为 `None`
- `infer_schema`：`bool`，是否自动推断参数模式，默认为 `True`

返回值：`Union[Callable, BaseTool]`，当作为装饰器使用时返回 `BaseTool`，当直接调用时返回装饰器函数或 `BaseTool`

#### 流程图

```mermaid
flowchart TD
    A[调用 regist_tool] --> B{len(args) == 0?}
    B -->|是| C[返回 wrapper 装饰器函数]
    B -->|否| D[直接调用 tool 装饰器]
    D --> E[_parse_tool 处理工具]
    E --> F[返回 BaseTool 实例]
    
    C --> G[使用 @regist_tool 装饰]
    G --> H[调用 wrapper 装饰函数]
    H --> I[调用 tool 装饰器]
    I --> J[_parse_tool 处理工具]
    J --> K[返回 BaseTool 实例]
    
    E --> L[将工具添加到 _TOOLS_REGISTRY]
    L --> M[设置/更新 description]
    M --> N[自动生成 title]
    N --> O[设置 title 属性]
```

#### 带注释源码

```python
def regist_tool(
    *args: Any,
    title: str = "",
    description: str = "",
    return_direct: bool = False,
    args_schema: Optional[Type[BaseModel]] = None,
    infer_schema: bool = True,
) -> Union[Callable, BaseTool]:
    """
    wrapper of langchain tool decorator
    add tool to regstiry automatically
    """

    def _parse_tool(t: BaseTool):
        nonlocal description, title

        # 将工具添加到全局注册表，以工具名称为键
        _TOOLS_REGISTRY[t.name] = t

        # 如果未提供描述，则从函数或协程的文档字符串中获取
        if not description:
            if t.func is not None:
                description = t.func.__doc__
            elif t.coroutine is not None:
                description = t.coroutine.__doc__
        
        # 将多行描述合并为单行，去除多余空白
        t.description = " ".join(re.split(r"\n+\s*", description))
        
        # 如果未提供标题，则从工具名称自动生成（将下划线分隔的单词首字母大写）
        if not title:
            title = "".join([x.capitalize() for x in t.name.split("_")])
        
        # 设置工具的标题属性
        t.title = title

    def wrapper(def_func: Callable) -> BaseTool:
        # 使用 LangChain 的 tool 装饰器包装函数
        partial_ = tool(
            *args,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
        )
        # 执行装饰
        t = partial_(def_func)
        # 解析和处理工具
        _parse_tool(t)
        return t

    # 根据参数数量决定行为：
    # 无参数时返回装饰器函数
    if len(args) == 0:
        return wrapper
    else:
        # 有参数时直接调用 tool 装饰器并解析
        t = tool(
            *args,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
        )
        _parse_tool(t)
        return t
```



### `format_context`

将包含知识库输出的 `ToolOutput` 对象格式化为 LLM 所需的字符串格式，提取知识库文档的 page_content 并进行拼接处理。

参数：

- `self`：`BaseToolOutput`，调用该函数的工具输出对象，通过 `self.data["docs"]` 访问知识库文档列表

返回值：`str`，格式化后的上下文字符串。如果没有找到相关文档，返回提示信息"没有找到相关文档,请更换关键词重试"；否则返回所有文档内容用双换行符拼接的字符串。

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[初始化 context 为空字符串]
    B --> C[从 self.data 提取 docs 列表]
    C --> D[遍历 docs 列表]
    D --> E[使用 DocumentWithVSId.parse_obj 解析每个文档]
    E --> F[提取 doc.page_content 并添加到 source_documents]
    F --> G{是否遍历完成}
    G -->|否| D
    G -->|是| H{len(source_documents) == 0}
    H -->|是| I[context = '没有找到相关文档,请更换关键词重试']
    H -->|否| J[拼接所有 doc + '\n\n' 到 context]
    I --> K[返回 context]
    J --> K
```

#### 带注释源码

```python
def format_context(self: BaseToolOutput) -> str:
    '''
    将包含知识库输出的ToolOutput格式化为 LLM 需要的字符串
    '''
    # 1. 初始化结果字符串
    context = ""
    
    # 2. 从工具输出对象中提取知识库文档列表
    # self.data 是字典，包含 'docs' 键，存储 DocumentWithVSId 对象列表
    docs = self.data["docs"]
    
    # 3. 用于存储解析后的文档内容
    source_documents = []

    # 4. 遍历知识库文档列表，逐个解析
    for inum, doc in enumerate(docs):
        # 使用 pydantic 模型解析文档对象
        # DocumentWithVSId 包含 page_content 等字段
        doc = DocumentWithVSId.parse_obj(doc)
        
        # 提取页面内容并添加到列表
        # 注意：此处 inum 变量未被使用，可能为遗留代码
        source_documents.append(doc.page_content)

    # 5. 检查是否有检索到的文档
    if len(source_documents) == 0:
        # 无文档时返回提示信息，引导用户更换关键词
        context = "没有找到相关文档,请更换关键词重试"
    else:
        # 6. 有文档时，使用双换行符拼接所有文档内容
        for doc in source_documents:
            context += doc + "\n\n"

    # 7. 返回格式化后的上下文字符串
    return context
```

## 关键组件





### BaseTool补丁模块

对LangChain的BaseTool进行扩展，支持额外字段和参数解析，解决langchain #15855问题

### regist_tool函数

装饰器包装器，自动将工具注册到全局注册表并设置描述和标题

### format_context函数

将包含知识库输出的ToolOutput格式化为LLM需要的字符串格式

### _TOOLS_REGISTRY全局变量

存储所有已注册工具的全局字典，以工具名称为键

### _new_parse_input函数

解析工具输入的补丁方法，支持字符串和字典两种输入格式

### _new_to_args_and_kwargs函数

将工具输入转换为位置参数和关键字参数的处理函数

### DocumentWithVSId模型

来自chatchat server的知识库文档模型，用于解析文档内容



## 问题及建议



### 已知问题

-   **Monkey Patching 风险**：直接修改 `BaseTool._parse_input` 和 `BaseTool._to_args_and_kwargs` 是危险的 monkey patching，注释中标记为 workaround，可能在 langchain 版本升级后失效，缺乏版本兼容性检查
-   **全局状态管理缺陷**：`_TOOLS_REGISTRY` 是全局字典，但没有任何清理机制（无 `unregister_tool` 或 `clear_registry` 方法），在测试环境或长时间运行的服务中可能导致内存泄漏
-   **类型注解错误**：`format_context` 函数签名写为 `def format_context(self: BaseToolOutput)` 但实际是模块级函数而非方法，`self` 参数没有实际作用且会造成类型检查工具的误报
-   **缺失错误处理**：`format_context` 中 `DocumentWithVSId.parse_obj(doc)` 可能抛出验证异常但无 try-except 保护；`format_context` 的 `self.data["docs"]` 访问无默认值检查，若 key 不存在会抛出 KeyError
-   **字符串处理问题**：使用 `re.split(r"\n+\s*", description)` 去除换行符的方式较为粗糙，可能意外移除有意义的格式；`title` 的生成逻辑 `"".join([x.capitalize() for x in t.name.split("_")])` 未考虑边界情况（如连续下划线、空格）
-   **未使用的导入**：`typing` 中的 `List` 被导入但未使用

### 优化建议

-   将 monkey patching 封装为带版本检测的条件逻辑，或考虑 fork langchain 源码/提交 PR 从根本上解决问题，而非依赖临时 workaround
-   为 `_TOOLS_REGISTRY` 实现完整的生命周期管理：添加 `unregister_tool(name)` 方法、注册表大小限制、序列化/反序列化能力
-   修正 `format_context` 函数签名为 `def format_context(tool_output: BaseToolOutput) -> str`，移除无效的 self 参数
-   为 `format_context` 添加完整的错误处理：try-except 捕获验证异常、添加 `self.data.get("docs", [])` 的默认值处理、返回有意义的错误信息而非直接崩溃
-   改进字符串处理：使用更健壮的标题生成算法（如考虑首字母大写例外）、添加正则表达式的边界测试
-   清理未使用的导入，使用静态分析工具（如 ruff、pylint）定期检查

## 其它





### 设计目标与约束

本模块旨在为chatchat项目提供一个灵活的LangChain工具注册和管理框架，支持自动将Python函数注册为LLM工具，并提供知识库上下文格式化能力。约束条件包括：依赖langchain库的BaseTool实现，支持pydantic v1版本，需要Python 3.8+环境。

### 错误处理与异常设计

代码中的错误处理主要体现在：1) input_args.validate和input_args.parse_obj调用时会自动触发pydantic验证错误；2) format_context函数中处理docs为空或解析失败的情况；3) 工具名称作为字典键可能产生键冲突风险。当args_schema验证失败时，langchain会抛出ValidationError；当工具函数执行异常时，会通过BaseTool的错误传播机制传递给调用方。

### 外部依赖与接口契约

主要依赖包括：langchain.agents.tool装饰器、langchain_core.tools.BaseTool、chatchat.server.knowledge_base.kb_doc_api.DocumentWithVSId、chatchat.server.pydantic_v1.BaseModel。外部接口契约：regist_tool返回BaseTool实例并自动注册到_TOOLS_REGISTRY全局字典；format_context接收BaseToolOutput实例并返回格式化字符串；所有注册工具可通过_TOOLS_REGISTRY[name]访问。

### 安全性考虑

代码存在以下安全风险：1) _TOOLS_REGISTRY全局字典无访问控制，可能被恶意注入；2) format_context直接解析用户提供的docs数据，存在反序列化风险；3) 工具描述通过__doc__动态获取，可能泄露函数内部实现细节。建议增加命名空间隔离、输入数据消毒、敏感信息过滤机制。

### 性能考虑

潜在性能瓶颈：1) re.split(r"\n+\s*", description)在每次工具解析时执行；2) format_context中DocumentWithVSId.parse_obj对每个doc逐个解析；3) _TOOLS_REGISTRY字典无大小限制可能导致内存泄漏。优化建议：缓存正则编译结果，批量解析文档，考虑使用弱引用或LRU缓存限制注册表大小。

### 配置说明

本模块无需显式配置，但regist_tool函数支持以下参数：title用于设置工具显示名称，description覆盖自动提取的描述，return_direct控制是否直接返回结果，args_schema定义pydantic格式的参数模式，infer_schema控制是否自动推断参数类型。BaseTool.Config.extra = Extra.allow允许动态添加额外字段。

### 使用示例

```python
from chatchat.server.utils.regist_tool import regist_tool, format_context

@regist_tool(title="知识库搜索", description="从知识库中搜索相关文档")
def search_knowledge_base(query: str) -> dict:
    """根据关键词搜索知识库"""
    # 实现逻辑
    return {"docs": [...]}

# 访问注册的工具
tool = _TOOLS_REGISTRY["search_knowledge_base"]
```

### 版本历史

当前版本基于LangChain 0.1.x系列设计，包含针对langchain #15855问题的补丁。补丁内容：_new_parse_input修复pydantic模型解析逻辑，_new_to_args_and_kwargs处理args参数展开逻辑。这些补丁为临时解决方案，后续LangChain官方修复后应移除。

### 扩展性设计

模块设计支持以下扩展场景：1) 通过BaseTool.Config.extra = Extra.allow支持自定义工具元数据；2) _TOOLS_REGISTRY采用字典结构便于扩展为命名空间隔离；3) format_context函数可通过注册机制扩展支持不同的文档格式；4) 可通过中间件模式添加工具调用日志、限流等横切关注点。


    