
# `AutoGPT\autogpt_platform\backend\backend\api\features\chat\tools\agent_generator\errors.py` 详细设计文档

该代码是一个错误处理工具模块，主要包含两个核心功能：一是利用正则表达式对原始错误详情进行脱敏处理，移除文件路径、数据库连接串、URL凭据等敏感信息；二是根据不同的错误类型（如LLM解析错误、超时、限流等）生成面向用户的友好错误提示，并可选地附加处理过的技术细节。

## 整体流程

```mermaid
graph TD
    A[开始: get_user_message_for_error] --> B{检查 error_type}
    B -- llm_parse_error --> C[设置 base_message 为 LLM 解析提示]
    B -- validation_error --> D[设置 base_message 为 验证失败提示]
    B -- patch_error --> E[设置 base_message 为 补丁失败提示]
    B -- timeout/llm_timeout --> F[设置 base_message 为 超时提示]
    B -- rate_limit/llm_rate_limit --> G[设置 base_message 为 限流提示]
    B -- 其他 --> H[设置 base_message 为 通用默认提示]
    C --> I{是否存在 error_details?}
    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
    I -- 否 --> J[返回 base_message]
    I -- 是 --> K[调用 _sanitize_error_details]
    K --> K1[替换文件路径为 [path]]
    K1 --> K2[替换数据库连接串为 [database_url]]
    K2 --> K3[替换带凭据的 URL 为 [url]]
    K3 --> K4[移除行号和文件引用]
    K4 --> K5[截断字符串至 200 字符]
    K5 --> L[拼接 base_message 和 details]
    L --> J
```

## 类结构

```
未定义类 (No Classes Defined)
├── 全局函数 (Global Functions)
│   ├── _sanitize_error_details
│   └── get_user_message_for_error
```

## 全局变量及字段




    

## 全局函数及方法


### `_sanitize_error_details`

清理错误详情字符串，移除敏感信息（如文件路径、数据库连接字符串、带凭证的URL以及堆栈跟踪内部信息），使其可以安全地展示给用户。

参数：

- `details`：`str`，包含潜在敏感数据的原始错误详情字符串。

返回值：`str`，清理后安全用于用户展示的错误详情字符串。

#### 流程图

```mermaid
graph TD
    A([开始]) --> B[/输入: details/]
    B --> C[替换 Unix 风格文件路径<br>匹配 .py, .js 等扩展名<br>替换为 '[path]']
    C --> D[替换 Windows 风格文件路径<br>匹配盘符及路径<br>替换为 '[path]']
    D --> E[替换数据库连接字符串<br>匹配 postgres, mysql 等协议<br>替换为 '[database_url]']
    E --> F[替换带凭证的 URL<br>匹配 http(s)://user:pass@host<br>替换为 '[url]']
    F --> G[移除行号信息<br>匹配 ', line \d+' 模式]
    G --> H[移除堆栈文件声明<br>匹配 'File \"...\"' 模式]
    H --> I[去除字符串首尾空格]
    I --> J([返回: sanitized])
```

#### 带注释源码

```python
def _sanitize_error_details(details: str) -> str:
    """Sanitize error details to remove sensitive information.

    Strips common patterns that could expose internal system info:
    - File paths (Unix and Windows)
    - Database connection strings
    - URLs with credentials
    - Stack trace internals

    Args:
        details: Raw error details string

    Returns:
        Sanitized error details safe for user display
    """
    # 替换 Unix 风格的文件路径 (例如 /home/user/project/app.py) 为 "[path]"
    # 匹配包含常见扩展名 (py, js, ts, json, yaml, yml) 的路径
    sanitized = re.sub(
        r"/[a-zA-Z0-9_./\-]+\.(py|js|ts|json|yaml|yml)", "[path]", details
    )
    # 替换 Windows 风格的文件路径 (例如 C:\Users\project\app) 为 "[path]"
    sanitized = re.sub(r"[A-Z]:\\[a-zA-Z0-9_\\.\\-]+", "[path]", sanitized)
    # 替换数据库连接字符串 (例如 postgresql://user:pass@localhost/db) 为 "[database_url]"
    # 涵盖 postgres, mysql, mongodb, redis
    sanitized = re.sub(
        r"(postgres|mysql|mongodb|redis)://[^\s]+", "[database_url]", sanitized
    )
    # 替换包含用户名和密码的 URL (例如 http://admin:12345@api.example.com) 为 "[url]"
    sanitized = re.sub(r"https?://[^:]+:[^@]+@[^\s]+", "[url]", sanitized)
    # 移除具体的行号信息 (例如 ", line 42")，避免暴露代码具体位置
    sanitized = re.sub(r", line \d+", "", sanitized)
    # 移除堆栈跟踪中的文件名引用 (例如 File "script.py", line 10)
    sanitized = re.sub(r'File "[^"]+",?', "", sanitized)

    # 返回处理后的字符串，并去除首尾可能产生的多余空格
    return sanitized.strip()
```



### `get_user_message_for_error`

该函数负责根据内部错误类型、操作描述及可选的自定义信息，生成并返回用户友好的错误消息。同时，如果提供了错误详情，会对详情进行脱敏处理并截断后追加到消息末尾。

参数：

- `error_type`：`str`，来自外部服务的错误类型（例如 "llm_parse_error", "timeout", "rate_limit"），用于确定主要错误消息的内容。
- `operation`：`str`，对失败操作的描述，用于在默认错误消息中说明具体的操作内容（例如 "analyze the goal"）。
- `llm_parse_message`：`str | None`，针对 `llm_parse_error` 类型的自定义消息，如果提供则覆盖默认消息。
- `validation_message`：`str | None`，针对 `validation_error` 类型的自定义消息，如果提供则覆盖默认消息。
- `error_details`：`str | None`，关于错误的附加技术细节，将在脱敏和截断后追加到用户消息中。

返回值：`str`，经过格式化和处理的、适合向最终用户展示的错误消息。

#### 流程图

```mermaid
flowchart TD
    A[开始: get_user_message_for_error] --> B{检查 error_type}
    B -->|llm_parse_error| C{llm_parse_message 是否存在?}
    C -->|是| D[使用 llm_parse_message]
    C -->|否| E[使用默认 LLM 解析错误消息]
    
    B -->|validation_error| F{validation_message 是否存在?}
    F -->|是| G[使用 validation_message]
    F -->|否| H[使用默认验证错误消息]
    
    B -->|patch_error| I[设置 Patch 错误消息]
    B -->|timeout 或 llm_timeout| J[设置超时消息]
    B -->|rate_limit 或 llm_rate_limit| K[设置速率限制消息]
    B -->|其他| L[使用带有 operation 的默认兜底消息]
    
    D --> M{error_details 是否存在?}
    E --> M
    G --> M
    H --> M
    I --> M
    J --> M
    K --> M
    L --> M

    M -->|是| N[调用 _sanitize_error_details]
    N --> O{长度 > 200?}
    O -->|是| P[截断至 200 字符并添加省略号]
    O -->|否| Q[保持原样]
    P --> R[追加技术详情到 base_message]
    Q --> R
    M -->|否| S[保持 base_message 不变]

    R --> T[返回 base_message]
    S --> T
```

#### 带注释源码

```python
def get_user_message_for_error(
    error_type: str,
    operation: str = "process the request",
    llm_parse_message: str | None = None,
    validation_message: str | None = None,
    error_details: str | None = None,
) -> str:
    """Get a user-friendly error message based on error type.

    This function maps internal error types to user-friendly messages,
    providing a consistent experience across different agent operations.

    Args:
        error_type: The error type from the external service
            (e.g., "llm_parse_error", "timeout", "rate_limit")
        operation: Description of what operation failed, used in the default
            message (e.g., "analyze the goal", "generate the agent")
        llm_parse_message: Custom message for llm_parse_error type
        validation_message: Custom message for validation_error type
        error_details: Optional additional details about the error

    Returns:
        User-friendly error message suitable for display to the user
    """
    # 初始化基础消息变量
    base_message = ""

    # 根据具体的错误类型，选择或生成对应的用户友好的提示信息
    if error_type == "llm_parse_error":
        # 如果提供了自定义的 LLM 解析错误消息则使用，否则使用默认文案
        base_message = (
            llm_parse_message
            or "The AI had trouble processing this request. Please try again."
        )
    elif error_type == "validation_error":
        # 如果提供了自定义的验证错误消息则使用，否则使用默认文案
        base_message = (
            validation_message
            or "The generated agent failed validation. "
            "This usually happens when the agent structure doesn't match "
            "what the platform expects. Please try simplifying your goal "
            "or breaking it into smaller parts."
        )
    elif error_type == "patch_error":
        # 针对 Patch 错误的特定文案
        base_message = (
            "Failed to apply the changes. The modification couldn't be "
            "validated. Please try a different approach or simplify the change."
        )
    elif error_type in ("timeout", "llm_timeout"):
        # 针对超时错误的文案
        base_message = (
            "The request took too long to process. This can happen with "
            "complex agents. Please try again or simplify your goal."
        )
    elif error_type in ("rate_limit", "llm_rate_limit"):
        # 针对速率限制错误的文案
        base_message = "The service is currently busy. Please try again in a moment."
    else:
        # 其他未知错误，使用通用的操作描述作为兜底
        base_message = f"Failed to {operation}. Please try again."

    # 如果提供了错误详情，则进行处理
    if error_details:
        # 调用辅助函数去除敏感信息（如路径、数据库连接串等）
        details = _sanitize_error_details(error_details)
        # 如果详情过长，截断到200个字符，避免 UI 显示问题
        if len(details) > 200:
            details = details[:200] + "..."
        # 将处理后的技术详情追加到基础消息后
        base_message += f"\n\nTechnical details: {details}"

    return base_message
```


## 关键组件


### 敏感信息净化组件

利用正则表达式匹配与替换算法，从原始错误堆栈或详情字符串中识别并移除文件路径、数据库连接字符串、包含认证信息的 URL 以及具体的行号信息，以防止系统内部实现细节泄露，确保面向用户的信息安全。

### 用户友好的错误消息映射器

一种基于策略的错误处理组件，将内部的技术错误类型（如 `llm_parse_error`、`rate_limit` 等）映射为通俗易懂的用户提示语，支持自定义消息覆盖，并能安全地追加经过净化的技术细节，从而提升系统的用户体验。


## 问题及建议


### 已知问题

-   Windows 路径正则表达式不够健壮：当前正则 `[A-Z]:\\[a-zA-Z0-9_\\.\\-]+` 无法处理包含空格的常见路径（如 `C:\Program Files\app.exe`），导致部分路径信息未被正确脱敏，存在信息泄露风险。
-   错误处理扩展性差：代码使用 `if-elif-else` 链式结构映射错误类型，违反了开闭原则，每次新增错误类型都需要修改函数核心逻辑，增加了维护成本和出错概率。
-   字符串截断逻辑粗糙：`details[:200]` 直接按长度截断，可能会在单词或句子中间切断，导致显示的“技术细节”结尾出现乱码或语义不完整，影响用户体验。

### 优化建议

-   引入字典映射策略：使用字典（例如 `ERROR_MESSAGE_MAP`）将错误类型映射到对应的错误消息模板，实现逻辑与数据分离，提高代码的可读性和可扩展性。
-   提取正则常量并增强匹配：将硬编码在函数内的正则表达式提取为模块级常量；优化 Windows 路径正则以支持空格和更多特殊字符，同时补充单元测试以覆盖各种边缘情况。
-   实现国际化（i18n）支持：将硬编码的英文提示语提取到资源文件（如 `.po`、`.json`）中，利用国际化库（如 `gettext`）根据用户语言环境动态加载消息，以支持多语言环境。
-   优化字符串截断算法：在截断错误详情时，采用智能截断逻辑（例如在最近的空格或换行符处截断），或者仅截断并显示特定关键字段，以保证展示内容的可读性。
-   完善日志与脱敏的架构设计：建议在架构层面明确区分“日志记录”与“用户展示”。确保在调用此模块生成用户消息之前，原始的详细错误信息已被完整记录到日志系统，以便开发人员调试，而后再进行脱敏处理返回给用户。


## 其它


### 设计目标与约束

**设计目标：**
1.  **用户隐私与安全**：核心目标是防止敏感信息（如文件路径、数据库连接串、密码、内部堆栈结构）通过错误消息泄露给最终用户。
2.  **用户体验一致性**：将底层的、技术性强的错误代码转换为统一风格、易于理解的用户提示语，降低用户的认知负荷。
3.  **可维护性与扩展性**：集中管理错误消息映射，便于后续添加新的错误类型或修改提示语文案，而不需要在业务逻辑代码中散落硬编码。

**约束：**
1.  **零外部依赖**：模块仅依赖 Python 标准库（`re`），不引入任何第三方库，保持轻量级。
2.  **无状态性**：所有函数均为纯函数，不依赖全局状态，不产生副作用，确保线程安全。
3.  **性能要求**：错误处理通常在异常流程中触发，对性能要求不高，但正则替换操作应避免过于复杂，防止在错误风暴中导致系统资源耗尽。

### 错误处理与异常设计

该模块本身是一个“错误处理处理器”，并不直接抛出业务异常，而是对已发生的异常进行二次加工。

1.  **防御性编程**：
    *   `get_user_message_for_error` 函数对可选参数（如 `llm_parse_message`, `validation_message`）进行了处理，即使调用方未提供特定消息，也能回退到默认消息，确保流程不中断。
    *   在 `error_details` 处理中，通过截断机制（超过200字符）防止日志或消息面板被过长的错误信息撑爆。
2.  **输入假设**：
    *   假设输入的 `error_details` 为字符串类型。如果传入非字符串类型，函数内部可能会抛出 `AttributeError` 或 `TypeError`。当前设计未在函数内部显式捕获此类异常，遵循“Python 鸭子类型”原则，即假定调用方传入正确类型。
3.  **容错机制**：
    *   正则表达式替换是按顺序执行的。如果某个替换模式失败（例如正则语法错误），程序会崩溃。但在当前设计中，正则均为静态常量，风险较低。通过连续的 `re.sub` 调用，确保多重敏感信息能被层层过滤。

### 数据流与状态机

**数据流：**
该模块的数据处理是单向的流水线过程，不存在循环或复杂的状态转换。

1.  **输入阶段**：接收原始错误上下文（`error_type`, `operation`, `raw_details` 等）。
2.  **映射与决策阶段**：系统根据 `error_type` 查找对应的用户友好消息模板。如果存在特定错误类型的自定义消息，则优先使用；否则使用默认模板。
3.  **清洗阶段**：
    *   如果提供了 `error_details`，数据流进入 `_sanitize_error_details` 函数。
    *   数据依次通过一系列“过滤器”（正则替换规则），移除路径、URL、行号等敏感元数据。
4.  **格式化输出阶段**：将清洗后的详情（截断至200字符）追加到基础消息后，组合成最终字符串返回给调用者。

**状态机：**
本模块无状态机设计。它是无状态的函数式编程风格，每次调用都是独立的，不保留前一次调用的上下文。

### 外部依赖与接口契约

**外部依赖：**
*   **Python 标准库 `re`**：用于执行正则表达式匹配和替换操作，是脱敏功能的核心依赖。

**接口契约：**

1.  **`_sanitize_error_details(details: str) -> str`**
    *   **前置条件**：输入 `details` 应为字符串。如果为 `None`，调用方需自行处理（虽然 Python 动态类型允许传入，但在该函数文档类型注解中未声明 `None`，理论上不应为空）。
    *   **后置条件**：返回值必然是字符串，且其中不包含符合特定正则模式的敏感路径或凭证信息。

2.  **`get_user_message_for_error(...) -> str`**
    *   **前置条件**：
        *   `error_type` 必须是预期的字符串之一（如 "llm_parse_error", "timeout" 等），否则将触发 `else` 分支的默认处理。
        *   字符串参数建议为 UTF-8 编码。
    *   **后置条件**：
        *   返回值必定非空字符串。
        *   返回值不包含内部文件路径结构。
        *   返回值长度受控（虽然未硬性限制总长度，但详情部分被限制在200字符以内）。

### 安全与合规

**安全设计：**
1.  **信息泄露防护 (Info Leakage Prevention)**：
    *   实施了严格的正则过滤策略，针对 Unix (`/...`) 和 Windows (`C:\...`) 文件路径进行脱敏。
    *   特别针对数据库连接字符串（如 `postgres://...`, `mysql://...`）和带有认证信息的 HTTP URL 进行脱敏，防止凭证外泄。
    *   移除了 Python 特有的堆栈信息（如 `line \d+`, `File "..."`），避免攻击者通过报错推断后端代码结构。
2.  **输入净化**：
    *   虽然主要用于输出净化，但也间接防止了将未经过滤的系统错误日志直接输出到前端可能导致的 XSS 风险（视前端渲染框架而定，去除特定字符有助于减少此类风险）。

**合规性考虑：**
*   该模块有助于满足 GDPR 等数据保护法规中关于“不泄露不必要的技术细节”的要求，确保在发生故障时，仅向用户展示解决问题所需的最小信息量，而非系统内部架构。

    