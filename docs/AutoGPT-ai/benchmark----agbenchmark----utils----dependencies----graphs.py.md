# `.\AutoGPT\benchmark\agbenchmark\utils\dependencies\graphs.py`

```py
# 导入所需的库
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pyvis.network import Network

# 导入自定义的模块
from agbenchmark.generate_test import DATA_CATEGORY
from agbenchmark.utils.utils import write_pretty_json

# 获取 logger 对象
logger = logging.getLogger(__name__)

# 定义生成贝塞尔曲线点的函数
def bezier_curve(
    src: np.ndarray, ctrl: List[float], dst: np.ndarray
) -> List[np.ndarray]:
    """
    Generate Bézier curve points.

    Args:
    - src (np.ndarray): The source point.
    - ctrl (List[float]): The control point.
    - dst (np.ndarray): The destination point.

    Returns:
    - List[np.ndarray]: The Bézier curve points.
    """
    curve = []
    # 在 0 到 1 之间生成 100 个点，计算贝塞尔曲线上的点
    for t in np.linspace(0, 1, num=100):
        curve_point = (
            np.outer((1 - t) ** 2, src)
            + 2 * np.outer((1 - t) * t, ctrl)
            + np.outer(t**2, dst)
        )
        curve.append(curve_point[0])
    return curve

# 绘制同一水平线上节点之间的曲线边
def curved_edges(
    G: nx.Graph, pos: Dict[Any, Tuple[float, float]], dist: float = 0.2
) -> None:
    """
    Draw curved edges for nodes on the same level.

    Args:
    - G (Any): The graph object.
    - pos (Dict[Any, Tuple[float, float]]): Dictionary with node positions.
    - dist (float, optional): Distance for curvature. Defaults to 0.2.

    Returns:
    - None
    """
    # 获取当前的坐标轴对象
    ax = plt.gca()
    # 遍历图 G 的边，同时获取边的起点、终点和数据
    for u, v, data in G.edges(data=True):
        # 获取起点的坐标
        src = np.array(pos[u])
        # 获取终点的坐标
        dst = np.array(pos[v])

        # 判断起点和终点是否在同一水平线上
        same_level = abs(src[1] - dst[1]) < 0.01

        # 如果起点和终点在同一水平线上
        if same_level:
            # 计算控制点的坐标
            control = [(src[0] + dst[0]) / 2, src[1] + dist]
            # 根据起点、控制点和终点生成贝塞尔曲线
            curve = bezier_curve(src, control, dst)
            # 创建箭头对象
            arrow = patches.FancyArrowPatch(
                posA=curve[0],  # 起点坐标
                posB=curve[-1],  # 终点坐标
                connectionstyle=f"arc3,rad=0.2",  # 连接样式
                color="gray",  # 箭头颜色
                arrowstyle="-|>",  # 箭头样式
                mutation_scale=15.0,  # 箭头缩放比例
                lw=1,  # 箭头线宽
                shrinkA=10,  # 起点缩进
                shrinkB=10,  # 终点缩进
            )
            # 将箭头添加到图形对象 ax 上
            ax.add_patch(arrow)
        # 如果起点和终点不在同一水平线上
        else:
            # 在起点和终点之间添加箭头
            ax.annotate(
                "",
                xy=dst,  # 终点坐标
                xytext=src,  # 起点坐标
                arrowprops=dict(
                    arrowstyle="-|>", color="gray", lw=1, shrinkA=10, shrinkB=10
                ),  # 箭头属性
            )
# 计算树状布局的节点位置，以根节点为中心，交替垂直偏移
def tree_layout(graph: nx.DiGraph, root_node: Any) -> Dict[Any, Tuple[float, float]]:
    """Compute positions as a tree layout centered on the root with alternating vertical shifts."""
    # 从根节点开始进行广度优先搜索，构建树
    bfs_tree = nx.bfs_tree(graph, source=root_node)
    # 计算每个节点到根节点的深度
    levels = {
        node: depth
        for node, depth in nx.single_source_shortest_path_length(
            bfs_tree, root_node
        ).items()
    }

    pos = {}  # 存储节点位置的字典
    max_depth = max(levels.values())  # 获取最大深度
    level_positions = {i: 0 for i in range(max_depth + 1)}  # type: ignore

    # 计算每个层级的节点数量，用于计算宽度
    level_count: Any = {}
    for node, level in levels.items():
        level_count[level] = level_count.get(level, 0) + 1

    vertical_offset = (
        0.07  # 同一层级内每个节点的垂直偏移量
    )

    # 分配节点位置
    for node, level in sorted(levels.items(), key=lambda x: x[1]):
        total_nodes_in_level = level_count[level]
        horizontal_spacing = 1.0 / (total_nodes_in_level + 1)
        pos_x = (
            0.5
            - (total_nodes_in_level - 1) * horizontal_spacing / 2
            + level_positions[level] * horizontal_spacing
        )

        # 在同一层级内交替向上和向下偏移节点
        pos_y = (
            -level
            + (level_positions[level] % 2) * vertical_offset
            - ((level_positions[level] + 1) % 2) * vertical_offset
        )
        pos[node] = (pos_x, pos_y)

        level_positions[level] += 1

    return pos


# 使用Spring布局绘制图形
def graph_spring_layout(
    dag: nx.DiGraph, labels: Dict[Any, str], tree: bool = True
) -> None:
    num_nodes = len(dag.nodes())
    # 设置图形和坐标轴
    fig, ax = plt.subplots()
    ax.axis("off")  # 关闭坐标轴

    base = 3.0

    if num_nodes > 10:
        base /= 1 + math.log(num_nodes)
        font_size = base * 10

    font_size = max(10, base * 10)
    node_size = max(300, base * 1000)
    # 如果传入了树结构，则找到根节点并计算树的布局
    if tree:
        # 找到入度为0的节点作为根节点
        root_node = [node for node, degree in dag.in_degree() if degree == 0][0]
        # 根据根节点计算树的布局
        pos = tree_layout(dag, root_node)
    else:
        # 根据节点数量调整弹簧布局的参数 k
        k_value = 3 / math.sqrt(num_nodes)

        # 使用弹簧布局算法计算节点的位置
        pos = nx.spring_layout(dag, k=k_value, iterations=50)

    # 绘制节点和标签
    nx.draw_networkx_nodes(dag, pos, node_color="skyblue", node_size=int(node_size))
    nx.draw_networkx_labels(dag, pos, labels=labels, font_size=int(font_size))

    # 绘制曲线边
    curved_edges(dag, pos)  # type: ignore

    # 调整布局并显示图形
    plt.tight_layout()
    plt.show()
# 将 RGB 颜色值转换为十六进制表示的颜色字符串
def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


# 获取不同类别的颜色映射字典
def get_category_colors(categories: Dict[Any, str]) -> Dict[str, str]:
    # 获取唯一的类别集合
    unique_categories = set(categories.values())
    # 使用 tab10 调色板创建颜色映射
    colormap = plt.cm.get_cmap("tab10", len(unique_categories))  # type: ignore
    return {
        category: rgb_to_hex(colormap(i)[:3])
        for i, category in enumerate(unique_categories)
    }


# 绘制交互式网络图
def graph_interactive_network(
    dag: nx.DiGraph,
    labels: Dict[Any, Dict[str, Any]],
    html_graph_path: str = "",
) -> None:
    # 创建一个交互式网络图对象
    nt = Network(notebook=True, width="100%", height="800px", directed=True)

    # 获取类别颜色映射
    category_colors = get_category_colors(DATA_CATEGORY)

    # 为 pyvis 网络图添加节点和边
    for node, json_data in labels.items():
        label = json_data.get("name", "")
        # 去除标签的前四个字符
        label_without_test = label[4:]
        node_id_str = node.nodeid

        # 获取该标签的类别
        category = DATA_CATEGORY.get(
            label, "unknown"
        )  # 如果找不到标签，则默认为 'unknown'

        # 获取该类别的颜色
        color = category_colors.get(category, "grey")

        nt.add_node(
            node_id_str,
            label=label_without_test,
            color=color,
            data=json_data,
        )

    # 为 pyvis 网络图添加边
    for edge in dag.edges():
        source_id_str = edge[0].nodeid
        target_id_str = edge[1].nodeid
        edge_id_str = (
            f"{source_id_str}_to_{target_id_str}"  # 构建唯一的边标识
        )
        if not (source_id_str in nt.get_nodes() and target_id_str in nt.get_nodes()):
            logger.warning(
                f"Skipping edge {source_id_str} -> {target_id_str} due to missing nodes"
            )
            continue
        nt.add_edge(source_id_str, target_id_str, id=edge_id_str)
    # 为层次布局配置物理引擎参数
    hierarchical_options = {
        "enabled": True,
        "levelSeparation": 200,  # 增加层级之间的垂直间距
        "nodeSpacing": 250,  # 增加同一层级节点之间的间距
        "treeSpacing": 250,  # 增加不同树之间的间距（用于森林）
        "blockShifting": True,
        "edgeMinimization": True,
        "parentCentralization": True,
        "direction": "UD",
        "sortMethod": "directed",
    }

    physics_options = {
        "stabilization": {
            "enabled": True,
            "iterations": 1000,  # 默认值通常在100左右
        },
        "hierarchicalRepulsion": {
            "centralGravity": 0.0,
            "springLength": 200,  # 增加边的长度
            "springConstant": 0.01,
            "nodeDistance": 250,  # 增加节点之间的最小距离
            "damping": 0.09,
        },
        "solver": "hierarchicalRepulsion",
        "timestep": 0.5,
    }

    nt.options = {
        "nodes": {
            "font": {
                "size": 20,  # 增加标签的字体大小
                "color": "black",  # 设置可读的字体颜色
            },
            "shapeProperties": {"useBorderWithImage": True},
        },
        "edges": {
            "length": 250,  # 增加边的长度
        },
        "physics": physics_options,
        "layout": {"hierarchical": hierarchical_options},
    }

    # 将图形序列化为 JSON 并保存在适当的位置
    graph_data = {"nodes": nt.nodes, "edges": nt.edges}
    logger.debug(f"Generated graph data:\n{json.dumps(graph_data, indent=4)}")

    # FIXME: 使用更可靠的方法找到这些文件的正确位置。
    #   这将在除了从我们的存储库根目录运行的情况下都会失败。
    home_path = Path.cwd()
    write_pretty_json(graph_data, home_path / "frontend" / "public" / "graph.json")
    # 定义 Flutter 应用路径为 home_path 的父目录下的 frontend/assets 目录
    flutter_app_path = home_path.parent / "frontend" / "assets"

    # 可选地，将数据保存到文件中
    # 与 Flutter UI 同步
    # 这部分代码只在 AutoGPT 仓库中有效，但如果 BUILD_SKILL_TREE 为 false，则不会执行到这部分代码
    write_pretty_json(graph_data, flutter_app_path / "tree_structure.json")
    validate_skill_tree(graph_data, "")

    # 提取类别为 "coding" 的节点 ID

    # 提取基于类别 "coding" 的子图
    coding_tree = extract_subgraph_based_on_category(graph_data.copy(), "coding")
    validate_skill_tree(coding_tree, "coding")
    write_pretty_json(
        coding_tree,
        flutter_app_path / "coding_tree_structure.json",
    )

    # 提取基于类别 "data" 的子图
    data_tree = extract_subgraph_based_on_category(graph_data.copy(), "data")
    # validate_skill_tree(data_tree, "data")
    write_pretty_json(
        data_tree,
        flutter_app_path / "data_tree_structure.json",
    )

    # 提取基于类别 "general" 的子图
    general_tree = extract_subgraph_based_on_category(graph_data.copy(), "general")
    validate_skill_tree(general_tree, "general")
    write_pretty_json(
        general_tree,
        flutter_app_path / "general_tree_structure.json",
    )

    # 提取基于类别 "scrape_synthesize" 的子图
    scrape_synthesize_tree = extract_subgraph_based_on_category(
        graph_data.copy(), "scrape_synthesize"
    )
    validate_skill_tree(scrape_synthesize_tree, "scrape_synthesize")
    write_pretty_json(
        scrape_synthesize_tree,
        flutter_app_path / "scrape_synthesize_tree_structure.json",
    )

    # 如果存在 html_graph_path，则将文件路径解析为绝对路径
    if html_graph_path:
        file_path = str(Path(html_graph_path).resolve())

        # 将图数据写入 HTML 文件
        nt.write_html(file_path)
# 根据指定类别提取子图，包括到达所有具有指定类别节点所需的所有节点和边

def extract_subgraph_based_on_category(graph, category):
    # 初始化子图字典，包含节点和边
    subgraph = {"nodes": [], "edges": []}
    # 初始化已访问节点集合
    visited = set()

    # 定义反向深度优先搜索函数
    def reverse_dfs(node_id):
        # 如果节点已经访问过，则返回
        if node_id in visited:
            return
        visited.add(node_id)

        # 获取节点数据
        node_data = next(node for node in graph["nodes"] if node["id"] == node_id)

        # 如果节点不在子图中，则添加到子图中
        if node_data not in subgraph["nodes"]:
            subgraph["nodes"].append(node_data)

        # 遍历边，找到指向当前节点的边，将其添加到子图中，并继续递归搜索
        for edge in graph["edges"]:
            if edge["to"] == node_id:
                if edge not in subgraph["edges"]:
                    subgraph["edges"].append(edge)
                reverse_dfs(edge["from"])

    # 找到具有目标类别的节点，并从这些节点开始进行反向深度优先搜索
    nodes_with_target_category = [
        node["id"] for node in graph["nodes"] if category in node["data"]["category"]
    ]

    for node_id in nodes_with_target_category:
        reverse_dfs(node_id)

    return subgraph

# 判断图是否存在环
def is_circular(graph):
    # 深度优先搜索函数，用于检测图中是否存在环
    def dfs(node, visited, stack, parent_map):
        # 将当前节点标记为已访问
        visited.add(node)
        # 将当前节点添加到栈中
        stack.add(node)
        # 遍历图中的边
        for edge in graph["edges"]:
            # 如果边的起始节点是当前节点
            if edge["from"] == node:
                # 如果边的终点在栈中，说明存在环
                if edge["to"] in stack:
                    # 检测到环，记录环的路径
                    cycle_path = []
                    current = node
                    while current != edge["to"]:
                        cycle_path.append(current)
                        current = parent_map.get(current)
                    cycle_path.append(edge["to])
                    cycle_path.append(node)
                    return cycle_path[::-1]
                # 如果边的终点未被访问过
                elif edge["to"] not in visited:
                    # 更新父节点映射
                    parent_map[edge["to"]] = node
                    # 递归调用深度优先搜索
                    cycle_path = dfs(edge["to"], visited, stack, parent_map)
                    if cycle_path:
                        return cycle_path
        # 将当前节点从栈中移除
        stack.remove(node)
        return None

    # 初始化已访问节点集合、栈和父节点映射
    visited = set()
    stack = set()
    parent_map = {}
    # 遍历图中的节点
    for node in graph["nodes"]:
        node_id = node["id"]
        # 如果节点未被访问过，则进行深度优先搜索
        if node_id not in visited:
            cycle_path = dfs(node_id, visited, stack, parent_map)
            if cycle_path:
                return cycle_path
    # 如果没有检测到环，则返回 None
    return None
# 返回图的根节点。根节点是没有入边的节点。
def get_roots(graph):
    """
    Return the roots of a graph. Roots are nodes with no incoming edges.
    """
    # 创建一个包含所有节点 ID 的集合
    all_nodes = {node["id"] for node in graph["nodes"]}

    # 创建一个包含有入边的节点的集合
    nodes_with_incoming_edges = {edge["to"] for edge in graph["edges"]}

    # 根节点是没有入边的节点
    roots = all_nodes - nodes_with_incoming_edges

    return list(roots)


# 验证给定的图是否代表一个有效的技能树，并在无效时引发适当的异常。
def validate_skill_tree(graph, skill_tree_name):
    """
    Validate if a given graph represents a valid skill tree and raise appropriate exceptions if not.

    :param graph: A dictionary representing the graph with 'nodes' and 'edges'.
    :raises: ValueError with a description of the invalidity.
    """
    # 检查是否存在循环
    cycle_path = is_circular(graph)
    if cycle_path:
        cycle_str = " -> ".join(cycle_path)
        raise ValueError(
            f"{skill_tree_name} skill tree is circular! Circular path detected: {cycle_str}."
        )

    # 检查是否存在多个根节点
    roots = get_roots(graph)
    if len(roots) > 1:
        raise ValueError(f"{skill_tree_name} skill tree has multiple roots: {roots}.")
    elif not roots:
        raise ValueError(f"{skill_tree_name} skill tree has no roots.")
```