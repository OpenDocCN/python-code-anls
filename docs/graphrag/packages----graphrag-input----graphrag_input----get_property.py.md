
# `graphrag\packages\graphrag-input\graphrag_input\get_property.py` 详细设计文档

一个工具函数，用于通过点符号（dot notation）从嵌套字典中检索属性值，支持深层嵌套访问，并在路径不存在时抛出KeyError异常。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[接收data字典和path字符串]
    B --> C[将path用.分割成keys列表]
    C --> D[初始化current为data]
    D --> E{遍历keys中的每个key}
    E --> F{current是字典且key存在?}
    F -- 否 --> G[抛出KeyError异常]
    F -- 是 --> H[current = current[key]]
    H --> E
    E --> I{遍历完成?}
    I -- 是 --> J[返回current值]
```

## 类结构

```
无类层次结构（仅包含一个全局函数）
```

## 全局变量及字段


### `keys`
    
存储分割后的路径节点

类型：`list[str]`
    


### `current`
    
当前遍历的字典层级

类型：`dict[str, Any]`
    


    

## 全局函数及方法



### `get_property`

该函数是一个工具函数，用于从嵌套字典中检索属性，支持使用点号（"."）分隔的路径字符串来访问深层嵌套的值，如果路径不存在则抛出 KeyError。

参数：

- `data`：`dict[str, Any]`，要从中检索属性的字典
- `path`：`str`，点分隔的路径字符串（例如 "foo.bar.baz"）

返回值：`Any`，指定路径处的值

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[将 path 按 '.' 分割成 keys 列表]
    B --> C[初始化 current = data]
    C --> D{遍历 keys 中的每个 key}
    D --> E{current 是 dict 且 key 在 current 中?}
    E -->|是| F[current = current[key]]
    F --> D
    E -->|否| G[抛出 KeyError: Property '{path}' not found]
    G --> H[结束 - 异常]
    D --> I{还有更多 key?}
    I -->|否| J[返回 current]
    J --> K[结束 - 成功]
```

#### 带注释源码

```python
# 从嵌套字典中检索属性的工具函数
# 支持使用点号分隔的路径访问深层嵌套值
def get_property(data: dict[str, Any], path: str) -> Any:
    """Retrieve a property from a dictionary using dot notation.

    Parameters
    ----------
    data : dict[str, Any]
        The dictionary to retrieve the property from.
    path : str
        A dot-separated string representing the path to the property (e.g., "foo.bar.baz").

    Returns
    -------
    Any
        The value at the specified path.

    Raises
    ------
    KeyError
        If the path does not exist in the dictionary.
    """
    # 步骤1: 将路径字符串按 "." 分割成键列表
    # 例如: "foo.bar.baz" -> ["foo", "bar", "baz"]
    keys = path.split(".")
    
    # 步骤2: 从根字典开始遍历
    current = data
    
    # 步骤3: 逐层遍历键路径
    for key in keys:
        # 步骤4: 类型检查和键存在性检查
        # 确保当前层级是字典类型，且目标键存在于当前字典中
        if not isinstance(current, dict) or key not in current:
            # 步骤5: 路径不存在时抛出明确的 KeyError
            msg = f"Property '{path}' not found"
            raise KeyError(msg)
        
        # 步骤6: 移动到下一层嵌套
        current = current[key]
    
    # 步骤7: 返回最终找到的值
    return current
```

## 关键组件




### get_property 函数

核心工具函数，支持通过点符号路径从嵌套字典中检索任意深度的属性值，路径不存在时抛出 KeyError。

### 路径解析模块

将点分隔的路径字符串（如 "foo.bar.baz"）拆分为键序列的逻辑，是实现嵌套访问的基础。

### 嵌套遍历机制

逐层遍历字典结构的循环逻辑，负责在每一层验证数据类型并访问子字典。

### 错误处理机制

当路径不存在或中间节点类型不为字典时，构造并抛出描述性 KeyError 异常的逻辑。

### 类型注解

使用 typing.Any 和 dict[str, Any] 进行静态类型标注，确保函数签名的类型安全。

### 数据验证逻辑

在遍历过程中检查当前值是否为字典类型，以及目标键是否存在于当前字典中的验证步骤。


## 问题及建议



### 已知问题

-   不支持列表/数组索引访问：当前实现仅支持字典类型，无法处理路径中包含列表索引的情况（如 "users.0.name"）
-   错误信息不够精确：虽然抛出了 KeyError，但未指出具体是哪个嵌套键不存在，调试困难
-   缺乏默认值支持：对比标准库的 dict.get() 和常见实现，缺少默认值参数选项，使用时不够灵活
-   未处理 None 值场景：当嵌套路径中存在 None 值时，会抛出不明确的类型错误而非有意义的 KeyError
-   硬编码分隔符：点号（"."）作为分隔符被写死，无法适配不同场景需求

### 优化建议

-   添加列表索引支持：扩展功能以支持 "items.0.name" 这类路径访问
-   增加默认值参数：参考 dict.get() 设计，提供 default 参数避免异常
-   改进错误信息：在 KeyError 中包含具体失败的键和当前可用的键路径
-   支持自定义分隔符：添加可选的 delimiter 参数
-   考虑缓存机制：对于高频调用相同路径的场景，可添加 lru_cache 装饰器
-   添加类型安全的变体：提供泛型版本返回具体类型而非 Any

## 其它




### 设计目标与约束

本工具函数的设计目标是提供一种简洁、高效的方式从嵌套字典中检索属性，使用户能够通过点符号路径（如"foo.bar.baz"）快速访问深层嵌套的值。约束包括：仅支持字典类型的嵌套结构，不支持列表索引；路径必须为点分隔的字符串；不支持默认值机制，路径不存在时直接抛出异常。

### 错误处理与异常设计

函数采用显式异常传播机制。当指定路径在字典中不存在时，抛出带有明确错误信息的KeyError异常。错误信息格式为"Property '{path}' not found"，其中{path}为用户传入的完整路径，便于调试定位问题。异常设计遵循Python社区惯例，调用方可通过try-except块捕获KeyError进行处理。

### 数据流与状态机

数据流分为三个阶段：1）解析阶段：将输入路径字符串按"."分割为键序列；2）遍历阶段：依次遍历键序列，从外层字典向内层字典逐层检索；3）返回阶段：到达最后一层时返回目标值。状态机包含两种状态：遍历中（当前为字典类型且键存在）和终止（返回最终值或抛出异常）。

### 外部依赖与接口契约

本函数为无依赖模块，仅使用Python标准库typing模块进行类型注解。接口契约明确：输入data参数必须为dict类型，path参数必须为str类型；返回值为Any类型，表示任意Python对象；函数具有幂等性，不修改输入字典。

### 边界条件与限制

边界情况包括：1）空路径字符串返回整个字典；2）单层路径直接返回对应值；3）路径以点开头或结尾会导致空键，产生KeyError；4）连续的点（如"a..b"）会产生空键，触发异常；5）值为None时正常返回None而非视为不存在。

### 使用示例与调用方式

基本用法：get_property({"a": {"b": "c"}}, "a.b") 返回"c"。嵌套调用：get_property({"users": {"alice": {"age": 30}}}, "users.alice.age") 返回30。错误处理示例：try...except KeyError as e处理不存在路径。

### 安全性考虑

函数本身不涉及敏感数据处理，但由于支持任意深度的字典访问，需注意：1）不提供访问对象内部属性或方法的能力，仅限于字典键；2）无命令注入风险，因仅执行字典键查找；3）在多租户场景中需确保传入的data来源可信。

    