# `.\graphrag\graphrag\index\graph\visualization\compute_umap_positions.py`

```py
    # 使用 umap 库对给定的嵌入向量进行降维投影到 2D/3D 空间
    embedding_positions = umap.UMAP(
        min_dist=min_dist,              # UMAP 中的最小距离参数
        n_neighbors=n_neighbors,        # UMAP 中的邻居数量参数
        spread=spread,                  # UMAP 中的扩展参数
        n_components=n_components,      # UMAP 中的降维后的维度数目
        metric=metric,                  # UMAP 使用的距离度量
        random_state=random_state,      # 随机状态用于确定初始化
    ).fit_transform(embedding_vectors)  # 对嵌入向量进行拟合和转换得到降维后的位置

    # 初始化储存节点位置信息的列表
    embedding_position_data: list[NodePosition] = []
    # 遍历节点标签列表，同时获取索引和节点名称
    for index, node_name in enumerate(node_labels):
        # 获取当前节点的嵌入位置坐标
        node_points = embedding_positions[index]  # type: ignore
        # 确定当前节点的类别（如果未提供类别，则默认为1）
        node_category = 1 if node_categories is None else node_categories[index]
        # 确定当前节点的大小（如果未提供大小，则默认为1）
        node_size = 1 if node_sizes is None else node_sizes[index]
    
        # 检查嵌入位置坐标的维度，如果是二维，则处理如下
        if len(node_points) == 2:
            # 将节点信息添加到嵌入位置数据中，包括节点名称、x 和 y 坐标、类别和大小
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=float(node_points[0]),
                    y=float(node_points[1]),
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
        else:
            # 如果嵌入位置坐标是三维，则处理如下
            # 将节点信息添加到嵌入位置数据中，包括节点名称、x、y 和 z 坐标、类别和大小
            embedding_position_data.append(
                NodePosition(
                    label=str(node_name),
                    x=float(node_points[0]),
                    y=float(node_points[1]),
                    z=float(node_points[2]),
                    cluster=str(int(node_category)),
                    size=int(node_size),
                )
            )
    
    # 返回处理后的嵌入位置数据列表
    return embedding_position_data
def visualize_embedding(
    graph,
    umap_positions: list[dict],
):
    """Project embedding down to 2D using UMAP and visualize."""
    # 清空当前图形
    plt.clf()
    # 获取当前图形对象
    figure = plt.gcf()
    # 获取当前轴对象
    ax = plt.gca()

    # 设置轴的隐藏
    ax.set_axis_off()
    # 设置图形大小为10x10英寸
    figure.set_size_inches(10, 10)
    # 设置图形分辨率为400 DPI
    figure.set_dpi(400)

    # 创建节点位置字典，将标签映射到UMAP坐标 (x, y)
    node_position_dict = {
        str(position["label"]): (position["x"], position["y"])
        for position in umap_positions
    }
    # 创建节点类别字典，将标签映射到类别
    node_category_dict = {
        str(position["label"]): position["category"]
        for position in umap_positions
    }
    # 创建节点大小列表
    node_sizes = [position["size"] for position in umap_positions]
    # 获取节点颜色列表，根据节点类别字典
    node_colors = gc.layouts.categorical_colors(node_category_dict)  # type: ignore

    # 初始化节点列表和颜色列表
    vertices = []
    node_color_list = []
    # 遍历节点位置字典，填充节点和颜色列表
    for node in node_position_dict:
        vertices.append(node)
        node_color_list.append(node_colors[node])

    # 绘制网络节点
    nx.draw_networkx_nodes(
        graph,
        pos=node_position_dict,
        nodelist=vertices,
        node_color=node_color_list,  # type: ignore
        alpha=1.0,
        linewidths=0.01,
        node_size=node_sizes,  # type: ignore
        node_shape="o",
        ax=ax,
    )
    # 显示图形
    plt.show()
```