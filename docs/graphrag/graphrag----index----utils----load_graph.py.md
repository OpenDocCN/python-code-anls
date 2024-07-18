# `.\graphrag\graphrag\index\utils\load_graph.py`

```py
# 导入networkx库，用于处理图数据
import networkx as nx

# 定义函数load_graph，用于加载图数据
def load_graph(graphml: str | nx.Graph) -> nx.Graph:
    """
    从GraphML文件或networkx图中加载图数据。

    参数:
    graphml (str or nx.Graph): 要加载的GraphML文件路径或者已有的networkx图对象。

    返回:
    nx.Graph: 返回加载后的networkx图对象。

    Raises:
    如果graphml参数不是str类型也不是nx.Graph类型，则可能引发异常。

    注意:
    如果graphml参数是str类型，则解析对应的GraphML文件并返回一个新的networkx图对象。
    如果graphml参数是nx.Graph类型，则直接返回该图对象本身，不进行任何加载操作。
    """
    # 如果graphml参数是字符串类型，则解析对应的GraphML文件并返回一个新的networkx图对象
    return nx.parse_graphml(graphml) if isinstance(graphml, str) else graphml
```