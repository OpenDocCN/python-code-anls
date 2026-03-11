
# `graphrag\packages\graphrag\graphrag\graphs\edge_weights.py` 详细设计文档

该代码实现了两条边权重计算工具函数，分别基于点互信息(PMI)和倒数排名融合(RRF)算法，用于计算图中节点之间的关联强度。

## 整体流程

```mermaid
graph TD
    A[开始] --> B[calculate_pmi_edge_weights]
    B --> C[计算节点频率比例]
    C --> D[计算边权重比例]
    D --> E[合并源节点和目标节点的频率信息]
    E --> F[计算PMI权重: weight * log2(weight / (source_prop * target_prop))]
    F --> G[返回PMI边权重数据框]
    H[开始] --> I[calculate_rrf_edge_weights]
    I --> J[调用calculate_pmi_edge_weights获取PMI权重]
    J --> K[计算PMI排名和原始权重排名]
    K --> L[应用RRF公式: 1/(k+pmi_rank) + 1/(k+raw_weight_rank)]
    L --> M[返回RRF边权重数据框]
```

## 类结构

```
该文件不包含类定义，仅包含两个全局函数模块
```

## 全局变量及字段


### `np`
    
NumPy库别名，用于数值计算和数学运算

类型：`module`
    


### `pd`
    
Pandas库别名，用于数据框操作和数据处理

类型：`module`
    


    

## 全局函数及方法



### `calculate_pmi_edge_weights`

计算点互信息（PMI）边权重，使用考虑了低频事件偏差的PMI变体公式。PMI(x,y) = p(x,y) * log2(p(x,y) / (p(x)*p(y)))，其中 p(x,y) 是边权重在总边权重中的比例，p(x) 是节点频率在总频率中的比例。

参数：

- `nodes_df`：`pd.DataFrame`，节点数据框，包含节点名称和频率信息
- `edges_df`：`pd.DataFrame`，边数据框，包含边权重、源节点和目标节点信息
- `node_name_col`：`str`，节点名称列名，默认为 "title"
- `node_freq_col`：`str`，节点频率列名，默认为 "frequency"
- `edge_weight_col`：`str`，边权重列名，默认为 "weight"
- `edge_source_col`：`str`，边源节点列名，默认为 "source"
- `edge_target_col`：`str`，边目标节点列名，默认为 "target"

返回值：`pd.DataFrame`，返回更新边权重后的边数据框，原有的临时列（prop_weight、source_prop、target_prop）会被删除

#### 流程图

```mermaid
flowchart TD
    A[开始 calculate_pmi_edge_weights] --> B[复制 nodes_df 的 name 和 freq 列]
    B --> C[计算 total_edge_weights 和 total_freq_occurrences]
    C --> D[计算每个节点的 prop_occurrence]
    D --> E[计算每条边的 prop_weight]
    E --> F[merge 源节点频率比例到边表]
    F --> G[merge 目标节点频率比例到边表]
    G --> H[计算 PMI 权重: prop_weight * log2(prop_weight / source_prop * target_prop)]
    H --> I[删除临时列并返回结果]
    
    style A fill:#f9f,color:#000
    style I fill:#9f9,color:#000
```

#### 带注释源码

```python
def calculate_pmi_edge_weights(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_name_col: str = "title",
    node_freq_col: str = "frequency",
    edge_weight_col: str = "weight",
    edge_source_col: str = "source",
    edge_target_col: str = "target",
) -> pd.DataFrame:
    """Calculate pointwise mutual information (PMI) edge weights.

    Uses a variant of PMI that accounts for bias towards low-frequency events.
    pmi(x,y) = p(x,y) * log2(p(x,y)/ (p(x)*p(y))
    p(x,y) = edge_weight(x,y) / total_edge_weights
    p(x) = freq_occurrence(x) / total_freq_occurrences.
    """
    # 步骤1：复制节点数据框，只保留节点名称和频率列
    copied_nodes_df = nodes_df[[node_name_col, node_freq_col]]

    # 步骤2：计算总边权重和总频率出现次数，用于后续的概率计算
    total_edge_weights = edges_df[edge_weight_col].sum()
    total_freq_occurrences = nodes_df[node_freq_col].sum()
    
    # 步骤3：计算每个节点的频率比例（prop_occurrence）
    # p(x) = freq_occurrence(x) / total_freq_occurrences
    copied_nodes_df["prop_occurrence"] = (
        copied_nodes_df[node_freq_col] / total_freq_occurrences
    )
    
    # 步骤4：只保留节点名称和频率比例列
    copied_nodes_df = copied_nodes_df.loc[:, [node_name_col, "prop_occurrence"]]

    # 步骤5：计算每条边的权重比例（prop_weight）
    # p(x,y) = edge_weight(x,y) / total_edge_weights
    edges_df["prop_weight"] = edges_df[edge_weight_col] / total_edge_weights
    
    # 步骤6：将源节点的频率比例合并到边表
    edges_df = (
        edges_df
        .merge(
            copied_nodes_df,
            left_on=edge_source_col,
            right_on=node_name_col,
            how="left",
        )
        .drop(columns=[node_name_col])
        .rename(columns={"prop_occurrence": "source_prop"})
    )
    
    # 步骤7：将目标节点的频率比例合并到边表
    edges_df = (
        edges_df
        .merge(
            copied_nodes_df,
            left_on=edge_target_col,
            right_on=node_name_col,
            how="left",
        )
        .drop(columns=[node_name_col])
        .rename(columns={"prop_occurrence": "target_prop"})
    )
    
    # 步骤8：使用 PMI 公式计算新的边权重
    # pmi(x,y) = p(x,y) * log2(p(x,y) / (p(x)*p(y))
    edges_df[edge_weight_col] = edges_df["prop_weight"] * np.log2(
        edges_df["prop_weight"] / (edges_df["source_prop"] * edges_df["target_prop"])
    )

    # 步骤9：删除临时计算列，返回最终的边数据框
    return edges_df.drop(columns=["prop_weight", "source_prop", "target_prop"])
```



### `calculate_rrf_edge_weights`

该函数通过结合点互信息（PMI）权重和节点频率，计算Reciprocal Rank Fusion（倒数秩融合）边权重，用于图中边的权重评估。

参数：

- `nodes_df`：`pd.DataFrame`，包含节点的DataFrame，必须包含节点名称和频率列
- `edges_df`：`pd.DataFrame`，包含边的DataFrame，必须包含源节点、目标节点和权重列
- `node_name_col`：`str`，节点名称列名，默认为 "title"
- `node_freq_col`：`str`，节点频率列名，默认为 "freq"
- `edge_weight_col`：`str`，边权重列名，默认为 "weight"
- `edge_source_col`：`str`，边源节点列名，默认为 "source"
- `edge_target_col`：`str`，边目标节点列名，默认为 "target"
- `rrf_smoothing_factor`：`int`，RRF平滑因子，用于避免除零错误，默认为 60

返回值：`pd.DataFrame`，包含融合后边权重的DataFrame

#### 流程图

```mermaid
flowchart TD
    A[开始] --> B[调用calculate_pmi_edge_weights计算PMI权重]
    B --> C[对PMI权重进行排名<br/>pmi_rank = rank, ascending=False]
    C --> D[对原始权重进行排名<br/>raw_weight_rank = rank, ascending=False]
    D --> E[应用RRF公式计算新权重<br/>weight = 1/(k+pmi_rank) + 1/(k+raw_weight_rank)]
    E --> F[删除临时列pmi_rank和raw_weight_rank]
    F --> G[返回结果DataFrame]
```

#### 带注释源码

```python
def calculate_rrf_edge_weights(
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_name_col: str = "title",
    node_freq_col: str = "freq",
    edge_weight_col: str = "weight",
    edge_source_col: str = "source",
    edge_target_col: str = "target",
    rrf_smoothing_factor: int = 60,
) -> pd.DataFrame:
    """Calculate reciprocal rank fusion (RRF) edge weights.

    Combines PMI weight and combined freq of source and target.
    """
    # 第一步：调用PMI计算函数，获取PMI边权重
    # 该函数计算 pmi(x,y) = p(x,y) * log2(p(x,y)/ (p(x)*p(y))
    edges_df = calculate_pmi_edge_weights(
        nodes_df,
        edges_df,
        node_name_col,
        node_freq_col,
        edge_weight_col,
        edge_source_col,
        edge_target_col,
    )

    # 第二步：对PMI权重计算排名（降序，权重越大排名越小）
    # method="min"表示相同值给予相同最小排名
    edges_df["pmi_rank"] = edges_df[edge_weight_col].rank(method="min", ascending=False)
    
    # 第三步：对原始权重也进行排名（此处实际上与PMI权重相同，因为edge_weight_col已被PMI值覆盖）
    edges_df["raw_weight_rank"] = edges_df[edge_weight_col].rank(
        method="min", ascending=False
    )
    
    # 第四步：应用RRF公式融合两种排名
    # RRF公式：score = 1/(k + rank1) + 1/(k + rank2)
    # k为平滑因子(rrf_smoothing_factor)，避免除零并控制排名差异的敏感度
    edges_df[edge_weight_col] = edges_df.apply(
        lambda x: (
            (1 / (rrf_smoothing_factor + x["pmi_rank"]))
            + (1 / (rrf_smoothing_factor + x["raw_weight_rank"]))
        ),
        axis=1,  # 按行应用函数
    )

    # 第五步：删除临时排名列，返回只包含原始列的DataFrame
    return edges_df.drop(columns=["pmi_rank", "raw_weight_rank"])
```

## 关键组件





### PMI边权重计算器

负责计算点互信息（Pointwise Mutual Information）边权重，通过概率比例计算节点间的关联强度，并使用log2变换避免数值下溢。该组件整合了节点频次和边权重数据，通过三次DataFrame合并操作构建源节点和目标节点的概率分布，最终输出PMI加权的边数据框。

### RRF边权重融合器

实现倒数排名融合（Reciprocal Rank Fusion）算法，将PMI权重与原始权重进行排名融合，通过倒数排名求和方式平衡不同度量标准的贡献。该组件内部调用PMI计算器，并引入平滑因子（默认为60）防止除零错误，输出融合后的标准化边权重。

### 数据转换管道

包含多个pandas merge操作的级联流水线，负责将边数据与节点数据进行关联计算。该管道通过左连接（left join）将源节点和目标节点的频次概率prop_occurrence映射到边数据中，形成source_prop和target_prop中间列，供后续数学计算使用。

### 数学计算引擎

基于NumPy的向量化计算模块，提供log2变换和排名计算功能。其中log2用于PMI公式的对数运算，rank方法（method="min"）实现最小值排名策略，确保相同权重值获得相同排名，避免排名跳跃问题。



## 问题及建议



### 已知问题

-   **效率问题 - apply 方法**：`calculate_rrf_edge_weights` 中使用 `df.apply(lambda..., axis=1)` 进行逐行计算，在大数据集上性能极差，应改用向量化操作替代。
-   **冗余计算**：`pmi_rank` 和 `raw_weight_rank` 计算了完全相同的值（都基于 `edge_weight_col` 升序排列），存在代码冗余。
-   **除零风险**：PMI 计算中 `source_prop * target_prop` 可能为 0 或极小值，会导致除零错误或产生 NaN/Inf 值，缺乏边界处理。
-   **参数不一致**：两个函数默认参数不统一，`node_freq_col` 分别为 "frequency" 和 "freq"，容易引发使用混淆。
-   **缺少输入校验**：未验证 DataFrame 是否为空、必需列是否存在、列类型是否正确，运行时可能抛出难以追踪的错误。
-   **内存效率**：多次 merge 操作创建中间 DataFrame 副本，处理大规模数据时可能导致内存压力。
-   **Magic Number**：`rrf_smoothing_factor=60` 硬编码在函数签名中，缺乏配置灵活性和文档说明。

### 优化建议

-   将 `apply` 替换为向量化计算，例如直接使用 `1 / (rrf_smoothing_factor + edges_df["pmi_rank"])` 避免逐行迭代。
-   移除重复的 `pmi_rank` 和 `raw_weight_rank` 计算，保留一个即可。
-   在 PMI 计算前添加 `np.where` 或 `np.finfo` 防护处理，避免除零和极端值。
-   统一两个函数的默认参数，或通过共享配置对象传递参数。
-   在函数入口添加输入校验，检查必要列的存在性和数据有效性。
-   使用链式操作或原地操作减少中间 DataFrame 创建，提升内存效率。
-   将 `rrf_smoothing_factor` 的默认值提取为常量或配置参数，并添加注释说明其业务含义。

## 其它





### 设计目标与约束

本模块的设计目标是提供两种边权重计算方法：PMI（点互信息）和RRF（倒数排名融合），用于图谱构建中的边权重优化。约束条件包括：输入的nodes_df和edges_df必须包含指定的列名；频率列和权重列必须为数值类型；节点名称列必须为字符串类型。默认列名假设基于title/frequency标准的DataFrame格式。

### 错误处理与异常设计

代码未实现显式的错误处理和异常捕获机制。潜在错误场景包括：除零错误（当total_edge_weights或total_freq_occurrences为0时）、类型不匹配错误（传入非数值列）、缺失值导致的NaN传播（merge操作中无法匹配的节点）。建议添加try-except块捕获ZeroDivisionError、KeyError和TypeError，并返回有意义的错误信息或默认值。

### 数据流与状态机

数据流遵循以下流程：输入原始nodes_df和edges_df → 计算节点频率比例和边权重比例 → 通过merge操作关联源节点和目标节点的频率比例 → 应用PMI公式计算新权重 → 返回更新后的edges_df。RRF流程在此基础上增加排名计算步骤，结合PMI排名和原始权重排名进行融合。状态转换：无状态函数，每次调用独立完成计算。

### 外部依赖与接口契约

本模块依赖两个外部库：numpy（版本要求>=1.x，用于log2运算）和pandas（版本要求>=1.x，用于DataFrame操作）。接口契约规定：calculate_pmi_edge_weights接受7个参数，返回包含计算后weight列的DataFrame；calculate_rrf_edge_weights接受8个参数（增加rrf_smoothing_factor），返回融合后的weight列。调用方需保证传入的DataFrame中包含指定的列名，且node_name_col对应的列无重复值。

### 性能考虑与复杂度分析

calculate_pmi_edge_weights的时间复杂度为O(n + m)，其中n为节点数，m为边数，主要消耗在两次merge操作上。calculate_rrf_edge_weights额外增加了O(m log m)的排名计算和O(m)的apply操作。空间复杂度为O(m)，因为需要创建临时列。对于大规模数据集（边数>100万），建议预先过滤低频节点以减少merge开销，或考虑使用向量化操作替代apply。

### 安全性考虑

代码本身不涉及用户输入处理或网络通信，安全性风险较低。但需注意：当处理来自不可信来源的DataFrame时，应验证数值列不包含特殊值（如Inf、NaN）以避免后续计算异常。merge操作的how="left"会在目标节点不存在时产生NaN，需确保数据质量。

### 可扩展性设计

当前实现了PMI和RRF两种权重计算方法。扩展方向包括：添加其他权重计算方法（如TF-IDF变体、朴素贝叶斯权重）；支持自定义权重融合公式；增加缓存机制避免重复计算。模块采用函数式设计，易于添加新函数或修改现有逻辑。

### 测试策略

建议的测试用例包括：基本功能测试（已知输入的预期输出）；边界条件测试（空DataFrame、单节点、单边）；异常输入测试（缺失列、类型错误、重复节点名）；数值精度测试（验证PMI计算结果与手动计算一致）；大规模性能测试（验证处理时间在可接受范围内）。

### 使用示例

```python
import pandas as pd
import numpy as np

# 创建示例数据
nodes = pd.DataFrame({
    "title": ["A", "B", "C"],
    "frequency": [10, 20, 30]
})
edges = pd.DataFrame({
    "source": ["A", "B"],
    "target": ["B", "C"],
    "weight": [5, 15]
})

# 计算PMI权重
pmi_result = calculate_pmi_edge_weights(nodes, edges)
print(pmi_result)

# 计算RRF权重
rrf_result = calculate_rrf_edge_weights(nodes, edges)
print(rrf_result)
```

### 版本历史与变更记录

当前版本为初始版本（v1.0），基于MIT License发布。后续可能的变更包括：添加类型注解（typing.Optional等）；优化大规模数据性能；增加配置类封装默认参数；添加日志记录便于调试。


    