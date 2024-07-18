# `.\graphrag\graphrag\index\verbs\graph\unpack.py`

```py
# 定义一个名为 unpack_graph 的自定义动词，用于从 graphml 图中解包节点或边，返回节点或边的列表
@verb(name="unpack_graph")
def unpack_graph(
    input: VerbInput,  # 输入参数对象
    callbacks: VerbCallbacks,  # 回调函数对象，用于进度追踪等
    column: str,  # 包含图表的列的名称
    type: str,  # 解包的数据类型，可以是 'node' 或 'edge'
    copy: list[str] | None = None,  # 要复制到输出的列的列表，默认为默认复制列表
    embeddings_column: str = "embeddings",  # 包含嵌入向量的列的名称，默认为 'embeddings'
    **kwargs,  # 其他关键字参数
) -> TableContainer:  # 返回一个 TableContainer 对象，包含解包后的数据表
    """
    Unpack nodes or edges from a graphml graph, into a list of nodes or edges.

    This verb will create columns for each attribute in a node or edge.

    ## Usage
    ```yaml
    verb: unpack_graph
    args:
        type: node # The type of data to unpack, one of: node, edge. node will create a node list, edge will create an edge list
        column: <column name> # The name of the column containing the graph, should be a graphml graph
    ```py
    """
    if copy is None:
        copy = default_copy  # 如果未提供 copy 参数，则使用默认的复制列表

    input_df = input.get_input()  # 获取输入数据作为 DataFrame
    num_total = len(input_df)  # 输入数据的总行数
    result = []  # 初始化结果列表

    copy = [col for col in copy if col in input_df.columns]  # 筛选在输入数据中存在的复制列
    has_embeddings = embeddings_column in input_df.columns  # 检查是否存在嵌入向量列

    # 迭代处理输入数据的每一行，显示进度并合并原始行与解包的图表项
    for _, row in progress_iterable(input_df.iterrows(), callbacks.progress, num_total):
        cleaned_row = {col: row[col] for col in copy}  # 提取复制列的数据作为清理后的行
        embeddings = (
            cast(dict[str, list[float]], row[embeddings_column])  # 获取嵌入向量数据，如果有的话
            if has_embeddings
            else {}
        )

        # 将清理后的行与每个解包后的图表项合并，添加到结果列表中
        result.extend([
            {**cleaned_row, **graph_id}
            for graph_id in _run_unpack(
                cast(str | nx.Graph, row[column]),  # 强制类型转换，获取当前行中指定的图表数据
                type,  # 解包的类型，节点或边
                embeddings,  # 嵌入向量数据
                kwargs,  # 其他关键字参数
            )
        ])

    output_df = pd.DataFrame(result)  # 将结果列表转换为 DataFrame
    return TableContainer(table=output_df)  # 返回包含结果数据表的 TableContainer 对象


# 解包节点的私有函数，返回包含每个节点属性的字典列表
def _unpack_nodes(
    graph: nx.Graph,  # 输入的 NetworkX 图对象
    embeddings: dict[str, list[float]],  # 节点嵌入向量的字典
    _args: dict[str, Any]  # 其他参数，暂时未使用
) -> list[dict[str, Any]]:  # 返回一个字典列表，包含解包后的节点数据
    return [
        {
            "label": label,  # 节点标签
            **(node_data or {}),  # 节点数据中的所有属性
            "graph_embedding": embeddings.get(label),  # 节点的嵌入向量
        }
        for label, node_data in graph.nodes(data=True)  # 遍历图中的每个节点及其数据
    ]


# 解包边的私有函数，返回包含每条边属性的字典列表
def _unpack_edges(
    graph: nx.Graph,  # 输入的 NetworkX 图对象
    _args: dict[str, Any]  # 其他参数，暂时未使用
) -> list[dict[str, Any]]:  # 返回一个字典列表，包含解包后的边数据
    # 返回图中每条边及其属性的字典
    return [
        {
            "source": source,  # 边的起点节点
            "target": target,  # 边的终点节点
            **edge_data,  # 边的其他属性
        }
        for source, target, edge_data in graph.edges(data=True)  # 遍历图中的每条边及其数据
    ]


# 运行解包函数的私有函数，根据解包类型调用相应的解包方法
def _run_unpack(
    graphml_or_graph: str | nx.Graph,  # 表示图表的字符串或 NetworkX 图对象
    unpack_type: str,  # 解包的类型，可以是 'nodes' 或 'edges'
    embeddings: dict[str, list[float]],  # 节点嵌入向量的字典
    args: dict[str, Any],  # 其他参数
) -> list[dict[str, Any]]:  # 返回一个字典列表，包含解包后的数据
    graph = load_graph(graphml_or_graph)  # 载入图表数据，转换为 NetworkX 图对象
    if unpack_type == "nodes":
        return _unpack_nodes(graph, embeddings, args)  # 解包节点数据
    if unpack_type == "edges":
        return _unpack_edges(graph, args)  # 解包边数据
    msg = f"Unknown type {unpack_type}"  # 未知的解包类型异常信息
    raise ValueError(msg)  # 抛出值错误异常
    # 返回一个包含源节点、目标节点和边数据的列表
    return [
        {
            # 源节点的 ID
            "source": source_id,
            # 目标节点的 ID
            "target": target_id,
            # 边的数据，如果数据为空则使用空字典
            **(edge_data or {}),
        }
        # 遍历图中所有边，包括数据
        for source_id, target_id, edge_data in graph.edges(data=True)  # type: ignore
    ]
```