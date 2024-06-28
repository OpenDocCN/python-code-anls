# `.\ollama_knowlage_graph\helpers\df_helpers.py`

```
import uuid  # 导入生成唯一标识符的模块
import pandas as pd  # 导入处理数据的 Pandas 库
import numpy as np  # 导入处理数据的 NumPy 库
from .prompts import extractConcepts  # 从 prompts 模块中导入 extractConcepts 函数
from .prompts import graphPrompt  # 从 prompts 模块中导入 graphPrompt 函数


def documents2Dataframe(documents) -> pd.DataFrame:
    rows = []  # 初始化空列表 rows，用于存储数据框的行
    for chunk in documents:
        row = {
            "text": chunk.page_content,  # 将文本内容存储在 "text" 键下
            **chunk.metadata,  # 使用 ** 操作符将 chunk.metadata 的所有键值对添加到字典中
            "chunk_id": uuid.uuid4().hex,  # 生成一个新的唯一标识符并将其存储在 "chunk_id" 键下
        }
        rows = rows + [row]  # 将当前行添加到 rows 列表中

    df = pd.DataFrame(rows)  # 使用 Pandas 创建数据框，数据来自 rows 列表
    return df  # 返回生成的数据框


def df2ConceptsList(dataframe: pd.DataFrame) -> list:
    # dataframe.reset_index(inplace=True)  # 重置数据框的索引（注释掉的代码，未使用）
    results = dataframe.apply(
        lambda row: extractConcepts(
            row.text, {"chunk_id": row.chunk_id, "type": "concept"}
        ),  # 对每一行应用 extractConcepts 函数，传递文本和元数据
        axis=1,  # 按行处理数据框
    )
    # invalid json results in NaN  # 处理无效的 JSON 结果，会生成 NaN 值
    results = results.dropna()  # 删除包含 NaN 值的行
    results = results.reset_index(drop=True)  # 重置结果的索引，丢弃原始索引

    ## 将列表扁平化为单一的实体列表。
    concept_list = np.concatenate(results).ravel().tolist()  # 扁平化嵌套列表，转换为普通列表
    return concept_list  # 返回包含所有概念的列表


def concepts2Df(concepts_list) -> pd.DataFrame:
    ## Remove all NaN entities  # 删除所有 NaN 实体
    concepts_dataframe = pd.DataFrame(concepts_list).replace(" ", np.nan)  # 创建包含概念列表的数据框，替换空格为 NaN
    concepts_dataframe = concepts_dataframe.dropna(subset=["entity"])  # 删除 "entity" 列中的 NaN 值
    concepts_dataframe["entity"] = concepts_dataframe["entity"].apply(
        lambda x: x.lower()  # 将 "entity" 列中的每个值转换为小写
    )

    return concepts_dataframe  # 返回处理后的数据框


def df2Graph(dataframe: pd.DataFrame, model=None) -> list:
    # dataframe.reset_index(inplace=True)  # 重置数据框的索引（注释掉的代码，未使用）
    results = dataframe.apply(
        lambda row: graphPrompt(row.text, {"chunk_id": row.chunk_id}, model), axis=1
        # 对每一行应用 graphPrompt 函数，传递文本、元数据和可选模型参数
    )
    # invalid json results in NaN  # 处理无效的 JSON 结果，会生成 NaN 值
    results = results.dropna()  # 删除包含 NaN 值的行
    results = results.reset_index(drop=True)  # 重置结果的索引，丢弃原始索引

    ## 将列表扁平化为单一的实体列表。
    concept_list = np.concatenate(results).ravel().tolist()  # 扁平化嵌套列表，转换为普通列表
    return concept_list  # 返回包含所有概念的列表


def graph2Df(nodes_list) -> pd.DataFrame:
    ## 删除所有 NaN 实体  # 删除所有 NaN 实体
    graph_dataframe = pd.DataFrame(nodes_list).replace(" ", np.nan)  # 创建包含节点列表的数据框，替换空格为 NaN
    graph_dataframe = graph_dataframe.dropna(subset=["node_1", "node_2"])  # 删除 "node_1" 和 "node_2" 列中的 NaN 值
    graph_dataframe["node_1"] = graph_dataframe["node_1"].apply(
        lambda x: x.lower()  # 将 "node_1" 列中的每个值转换为小写
    )
    graph_dataframe["node_2"] = graph_dataframe["node_2"].apply(
        lambda x: x.lower()  # 将 "node_2" 列中的每个值转换为小写
    )

    return graph_dataframe  # 返回处理后的数据框
```