
# `graphrag\packages\graphrag\graphrag\cache\cache_key_creator.py` 详细设计文档

该代码是GraphRAG项目的缓存键生成模块，通过将输入参数与版本号结合，生成具有缓存失效机制的缓存键，以确保在配置变更或缓存格式变化时能够自动刷新缓存。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[接收输入参数 input_args]
B --> C{input_args是否为有效字典?}
C -- 否 --> D[传入create_cache_key处理]
C -- 是 --> D
D --> E[获取基础缓存键 base_key]
E --> F[拼接版本号 _CACHE_VERSION]
F --> G[返回格式: {base_key}_v{version}]
G --> H[结束]
```

## 类结构

```
无类层次结构（仅包含模块级函数和变量）
```

## 全局变量及字段


### `_CACHE_VERSION`
    
全局版本号，用于控制缓存失效

类型：`int`
    


    

## 全局函数及方法



### `cache_key_creator`

该函数是 Graphrag 缓存系统的核心组件，用于根据输入参数生成带有版本号的缓存键。它接收模型调用的输入参数字典，通过底层的 `create_cache_key` 函数生成基础键值，然后附加当前缓存版本号以确保缓存隔离和失效策略的执行。

**参数：**

- `input_args`：`dict[str, Any]`，包含模型调用的输入参数字典，用于生成缓存键的依据

**返回值：**`str`，返回格式为 `{base_key}_v{版本号}` 的缓存键字符串

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[接收 input_args 参数]
    B --> C[调用 create_cache_key 函数]
    C --> D[生成 base_key 基础键]
    D --> E[拼接版本号 _CACHE_VERSION]
    E --> F[返回完整缓存键]
    F --> G[结束]
    
    C -.->|使用输入参数| D
    E -.->|格式: {base_key}_v{version}| F
```

#### 带注释源码

```python
# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""Cache key creation for Graphrag."""

from typing import Any

# 从 graphrag_llm.cache 模块导入底层缓存键生成函数
from graphrag_llm.cache import create_cache_key

# 缓存版本号全局常量
# 如果缓存内容发生破坏性变更，应递增此版本号以使现有缓存失效
# fnllm 使用缓存版本 2，虽然生成的缓存键相似，但 litellm 存储的对象不同
# 使用 litellm 模型提供者无法复用 fnllm 生成的缓存，因此从版本 3 开始
# graphrag-llm 包现在是版本 4，用于适应从 graphrag 分离时对 ModelConfig 的修改
# graphrag-llm 现在支持指标缓存，这些指标之前未被缓存
_CACHE_VERSION = 4


def cache_key_creator(
    input_args: dict[str, Any],
) -> str:
    """Generate a cache key based on input arguments.

    Args
    ____
        input_args: dict[str, Any]
            The input arguments for the model call.

    Returns
    -------
        str
            The generated cache key in the format
            `{prefix}_{data_hash}_v{version}` if prefix is provided.
    """
    # 调用底层函数根据输入参数生成基础缓存键
    base_key = create_cache_key(input_args)

    # 将基础键与版本号拼接，形成完整的缓存键
    # 格式: {base_key}_v{_CACHE_VERSION}
    return f"{base_key}_v{_CACHE_VERSION}"
```

## 关键组件





### 缓存版本管理 (_CACHE_VERSION)

全局常量，值为4，用于控制缓存版本。当缓存格式发生破坏性变化时递增版本号以使旧缓存失效。

### 缓存键创建函数 (cache_key_creator)

主函数，接受输入参数字典，调用底层create_cache_key生成基础键，并在键后附加版本号形成最终缓存键。

### 底层缓存键生成 (create_cache_key)

从graphrag_llm.cache导入的依赖函数，负责根据输入参数生成基础缓存键的具体实现。

### 版本注释文档

记录了缓存版本从2->3->4的演进历史，说明了fnllm到litellm的变更以及graphrag-llm包拆分的影响。



## 问题及建议





### 已知问题

-   **文档与实现不一致**：函数文档注释描述返回格式为`{prefix}_{data_hash}_v{version}`，表示支持prefix参数，但实际函数签名并未包含prefix参数，导致文档误导性
-   **硬编码版本号**：`_CACHE_VERSION`为硬编码常量，版本更新需要修改源码，缺乏灵活配置机制
-   **错误处理缺失**：调用`create_cache_key(input_args)`时未处理可能的异常情况（如返回None或抛出异常），可能导致运行时错误
-   **类型提示过于宽泛**：`input_args: dict[str, Any]`中Any类型过于宽泛，无法约束输入参数的结构，影响代码可维护性和类型安全
-   **缺少日志记录**：缓存键生成过程没有任何日志记录，难以追踪缓存行为和调试问题

### 优化建议

-   根据实际需求补充prefix参数支持，或修正文档描述以保持一致性
-   考虑将版本号改为可配置参数，支持从环境变量或配置文件读取，提高灵活性
-   添加try-except块处理`create_cache_key`的异常情况，并返回有意义的错误信息
-   定义更具体的输入参数类型（如 TypedDict 或 dataclass），替代泛型的dict[str, Any]
-   添加日志记录功能，记录缓存键生成的时间戳和关键参数，便于问题排查和监控
-   考虑添加缓存键格式验证逻辑，确保生成的键符合预期格式



## 其它





### 设计目标与约束

本模块的设计目标是提供一个可靠的缓存键生成机制，用于Graphrag系统的缓存管理。核心约束包括：1）缓存键格式必须包含版本号以支持缓存失效；2）依赖外部的create_cache_key函数生成基础键；3）版本号变更需遵循语义化版本控制原则，仅在破坏性变更时递增。

### 错误处理与异常设计

本模块的错误处理主要依赖于底层create_cache_key函数的异常传播。当输入参数为空或格式不正确时，底层函数可能抛出TypeError或ValueError。由于函数设计为纯函数无状态，暂无业务层面的自定义异常设计。建议调用方在传入input_args前进行参数校验。

### 数据流与状态机

数据流描述：调用方传入input_args字典 → cache_key_creator函数接收 → 调用create_cache_key生成base_key → 拼接版本号后缀 → 返回完整缓存键字符串。本模块不涉及状态机设计，为无状态纯函数。

### 外部依赖与接口契约

外部依赖：graphrag_llm.cache模块的create_cache_key函数。接口契约：输入参数input_args类型为dict[str, Any]，返回类型为str。调用方需保证input_args可被pickle序列化以支持缓存哈希计算。

### 性能考虑

本模块的性能开销主要来自create_cache_key的哈希计算。为优化性能，建议：1）对于相同输入复用缓存键；2）避免在热路径中频繁调用；3）版本号变更时需清理旧缓存以释放存储空间。

### 安全考虑

本模块本身不涉及敏感数据处理，但需注意：1）input_args中不应包含明文密码或密钥等敏感信息；2）缓存键会持久化存储，需确保底层缓存存储的安全性；3）版本号不应包含任何敏感标识信息。

### 兼容性考虑

版本兼容性：_CACHE_VERSION从2（fnllm）演进到3（litellm）再到4（graphrag-llm），每次变更都会导致旧缓存失效。跨版本缓存复用需自行实现键映射逻辑。API兼容性：函数签名保持稳定，无breaking change计划。

### 配置说明

_CACHE_VERSION为模块级配置常量，当前值为4。修改此值会立即使所有现有缓存失效，适用于需要强制刷新缓存的场景。在发布破坏性变更前必须递增此版本号。

### 使用示例

```python
# 基本用法
input_args = {"model": "gpt-4", "messages": [{"role": "user", "content": "hello"}]}
cache_key = cache_key_creator(input_args)
# 返回类似: "abc123_v4" 的字符串

# 用于缓存查询
def get_cached_result(args):
    key = cache_key_creator(args)
    if key in cache_store:
        return cache_store[key]
    return None
```

### 测试策略

建议测试用例：1）空输入字典；2）包含嵌套结构的参数字典；3）版本号正确拼接验证；4）相同输入返回相同键的幂等性测试；5）版本号变更后键值变化的验证。

### 缓存失效策略

本模块采用显式版本号控制失效策略。当_CACHE_VERSION递增时，所有历史版本缓存自动失效。失效机制为键名后缀匹配，非时间-based失效。生产环境建议配合TTL机制处理极端情况。


    