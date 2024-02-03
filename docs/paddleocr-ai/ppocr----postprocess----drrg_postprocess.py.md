# `.\PaddleOCR\ppocr\postprocess\drrg_postprocess.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据许可证分发，基于"按原样"的基础，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和
# 限制
"""
# 本代码参考自:
# https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textdet/postprocess/drrg_postprocessor.py

# 导入必要的库
import functools
import operator
import numpy as np
import paddle
from numpy.linalg import norm
import cv2

# 定义节点类
class Node:
    def __init__(self, ind):
        self.__ind = ind
        self.__links = set()

    @property
    def ind(self):
        return self.__ind

    @property
    def links(self):
        return set(self.__links)

    # 添加连接节点
    def add_link(self, link_node):
        self.__links.add(link_node)
        link_node.__links.add(self)

# 图传播函数
def graph_propagation(edges, scores, text_comps, edge_len_thr=50.):
    # 断言检查输入数据维度和类型
    assert edges.ndim == 2
    assert edges.shape[1] == 2
    assert edges.shape[0] == scores.shape[0]
    assert text_comps.ndim == 2
    assert isinstance(edge_len_thr, float)

    # 对边进行排序
    edges = np.sort(edges, axis=1)
    # 初始化得分字典
    score_dict = {}
    # 遍历边的列表，同时获取索引和边
    for i, edge in enumerate(edges):
        # 如果文本组件不为空
        if text_comps is not None:
            # 获取边的两个端点的坐标，reshape成4x2的矩阵
            box1 = text_comps[edge[0], :8].reshape(4, 2)
            box2 = text_comps[edge[1], :8].reshape(4, 2)
            # 计算两个端点的中心点坐标
            center1 = np.mean(box1, axis=0)
            center2 = np.mean(box2, axis=0)
            # 计算两个中心点之间的距离
            distance = norm(center1 - center2)
            # 如果距离大于阈值，则将对应边的分数设为0
            if distance > edge_len_thr:
                scores[i] = 0
        # 如果边的两个端点在分数字典中
        if (edge[0], edge[1]) in score_dict:
            # 更新分数字典中对应边的分数为原分数和当前分数的平均值
            score_dict[edge[0], edge[1]] = 0.5 * (
                score_dict[edge[0], edge[1]] + scores[i])
        else:
            # 如果边的两个端点不在分数字典中，则将当前分数加入分数字典
            score_dict[edge[0], edge[1]] = scores[i]

    # 获取所有节点的唯一值并排序
    nodes = np.sort(np.unique(edges.flatten()))
    # 创建一个映射数组，将节点映射到新的索引
    mapping = -1 * np.ones((np.max(nodes) + 1), dtype=np.int32)
    mapping[nodes] = np.arange(nodes.shape[0])
    # 根据映射关系更新边的顺序索引
    order_inds = mapping[edges]
    # 创建节点对象列表
    vertices = [Node(node) for node in nodes]
    # 根据顺序索引将节点连接起来
    for ind in order_inds:
        vertices[ind[0]].add_link(vertices[ind[1]])

    # 返回节点列表和分数字典
    return vertices, score_dict
# 计算连接组件，返回连接组件列表
def connected_components(nodes, score_dict, link_thr):
    # 确保 nodes 是列表
    assert isinstance(nodes, list)
    # 确保 nodes 列表中的元素都是 Node 类型
    assert all([isinstance(node, Node) for node in nodes])
    # 确保 score_dict 是字典类型
    assert isinstance(score_dict, dict)
    # 确保 link_thr 是浮点数类型
    assert isinstance(link_thr, float)

    # 初始化 clusters 列表
    clusters = []
    # 将 nodes 转换为集合
    nodes = set(nodes)
    # 当 nodes 集合非空时循环
    while nodes:
        # 从 nodes 集合中弹出一个节点
        node = nodes.pop()
        # 初始化 cluster 集合，包含当前节点
        cluster = {node}
        # 初始化 node_queue 列表，包含当前节点
        node_queue = [node]
        # 当 node_queue 列表非空时循环
        while node_queue:
            # 从 node_queue 列表中取出第一个节点
            node = node_queue.pop(0)
            # 获取当前节点的邻居节点，满足连接阈值条件
            neighbors = set([
                neighbor for neighbor in node.links if
                score_dict[tuple(sorted([node.ind, neighbor.ind]))] >= link_thr
            ])
            # 从邻居节点中去除已经在 cluster 中的节点
            neighbors.difference_update(cluster)
            # 从 nodes 中去除邻居节点
            nodes.difference_update(neighbors)
            # 将邻居节点加入 cluster
            cluster.update(neighbors)
            # 将邻居节点加入 node_queue
            node_queue.extend(neighbors)
        # 将 cluster 转换为列表并添加到 clusters 列表中
        clusters.append(list(cluster))
    # 返回连接组件列表
    return clusters


# 将连接组件转换为节点标签
def clusters2labels(clusters, num_nodes):
    # 确保 clusters 是列表
    assert isinstance(clusters, list)
    # 确保 clusters 列表中的元素都是列表
    assert all([isinstance(cluster, list) for cluster in clusters])
    # 确保 clusters 中的节点都是 Node 类型
    assert all(
        [isinstance(node, Node) for cluster in clusters for node in cluster])
    # 确保 num_nodes 是整数类型
    assert isinstance(num_nodes, int)

    # 初始化节点标签数组
    node_labels = np.zeros(num_nodes)
    # 遍历 clusters 列表
    for cluster_ind, cluster in enumerate(clusters):
        # 遍历每个连接组件中的节点
        for node in cluster:
            # 将节点的标签设置为连接组件的索引
            node_labels[node.ind] = cluster_ind
    # 返回节点标签数组
    return node_labels


# 移除单个节点连接组件
def remove_single(text_comps, comp_pred_labels):
    # 确保 text_comps 是二维数组
    assert text_comps.ndim == 2
    # 确保 text_comps 的行数与 comp_pred_labels 的长度相同
    assert text_comps.shape[0] == comp_pred_labels.shape[0]

    # 初始化单个节点标志数组
    single_flags = np.zeros_like(comp_pred_labels)
    # 获取预测标签中的唯一值
    pred_labels = np.unique(comp_pred_labels)
    # 遍历每个预测标签
    for label in pred_labels:
        # 获取当前标签的布尔标志
        current_label_flag = (comp_pred_labels == label)
        # 如果当前标签只有一个节点
        if np.sum(current_label_flag) == 1:
            # 将该节点标记为单个节点
            single_flags[np.where(current_label_flag)[0][0]] = 1
    # 保留非单个节点的索引
    keep_ind = [i for i in range(len(comp_pred_labels)) if not single_flags[i]]
    # 根据索引过滤文本组件和标签
    filtered_text_comps = text_comps[keep_ind, :]
    filtered_labels = comp_pred_labels[keep_ind]

    # 返回过滤后的文本组件和标签
    return filtered_text_comps, filtered_labels
# 计算两个点之间的欧几里德距离
def norm2(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

# 寻找连接所有点的最短路径
def min_connect_path(points):
    # 断言输入参数为列表
    assert isinstance(points, list)
    # 断言所有元素为列表
    assert all([isinstance(point, list) for point in points])
    # 断言所有坐标为整数
    assert all([isinstance(coord, int) for point in points for coord in point])

    # 复制输入点列表
    points_queue = points.copy()
    shortest_path = []
    current_edge = [[], []]

    edge_dict0 = {}
    edge_dict1 = {}
    current_edge[0] = points_queue[0]
    current_edge[1] = points_queue[0]
    points_queue.remove(points_queue[0])
    while points_queue:
        for point in points_queue:
            length0 = norm2(point, current_edge[0])
            edge_dict0[length0] = [point, current_edge[0]]
            length1 = norm2(current_edge[1], point)
            edge_dict1[length1] = [current_edge[1], point]
        key0 = min(edge_dict0.keys())
        key1 = min(edge_dict1.keys())

        if key0 <= key1:
            start = edge_dict0[key0][0]
            end = edge_dict0[key0][1]
            shortest_path.insert(0, [points.index(start), points.index(end)])
            points_queue.remove(start)
            current_edge[0] = start
        else:
            start = edge_dict1[key1][0]
            end = edge_dict1[key1][1]
            shortest_path.append([points.index(start), points.index(end)])
            points_queue.remove(end)
            current_edge[1] = end

        edge_dict0 = {}
        edge_dict1 = {}

    # 将路径列表展开并去重排序
    shortest_path = functools.reduce(operator.concat, shortest_path)
    shortest_path = sorted(set(shortest_path), key=shortest_path.index)

    return shortest_path

# 判断点是否在轮廓内部
def in_contour(cont, point):
    x, y = point
    is_inner = cv2.pointPolygonTest(cont, (int(x), int(y)), False) > 0.5
    return is_inner

# 修正角点
def fix_corner(top_line, bot_line, start_box, end_box):
    # 断言输入参数类型
    assert isinstance(top_line, list)
    assert all(isinstance(point, list) for point in top_line)
    assert isinstance(bot_line, list)
    # 确保 bot_line 中的每个元素都是列表类型
    assert all(isinstance(point, list) for point in bot_line)
    # 确保 start_box 和 end_box 的形状都是 (4, 2)
    assert start_box.shape == end_box.shape == (4, 2)

    # 将 top_line 和 bot_line 反转后合并成一个 NumPy 数组
    contour = np.array(top_line + bot_line[::-1])
    # 计算起始框的左中点和右中点
    start_left_mid = (start_box[0] + start_box[3]) / 2
    start_right_mid = (start_box[1] + start_box[2]) / 2
    # 计算结束框的左中点和右中点
    end_left_mid = (end_box[0] + end_box[3]) / 2
    end_right_mid = (end_box[1] + end_box[2]) / 2

    # 如果起始框的左中点不在轮廓内，则将起始框的左上角和左下角插入到 top_line 和 bot_line 中
    if not in_contour(contour, start_left_mid):
        top_line.insert(0, start_box[0].tolist())
        bot_line.insert(0, start_box[3].tolist())
    # 如果起始框的右中点不在轮廓内，则将起始框的右上角和右下角插入到 top_line 和 bot_line 中
    elif not in_contour(contour, start_right_mid):
        top_line.insert(0, start_box[1].tolist())
        bot_line.insert(0, start_box[2].tolist())
    
    # 如果结束框的左中点不在轮廓内，则将结束框的左上角和左下角追加到 top_line 和 bot_line 中
    if not in_contour(contour, end_left_mid):
        top_line.append(end_box[0].tolist())
        bot_line.append(end_box[3].tolist())
    # 如果结束框的右中点不在轮廓内，则将结束框的右上角和右下角追加到 top_line 和 bot_line 中
    elif not in_contour(contour, end_right_mid):
        top_line.append(end_box[1].tolist())
        bot_line.append(end_box[2].tolist())
    
    # 返回更新后的 top_line 和 bot_line
    return top_line, bot_line
# 将文本组件合并并构建文本实例的边界
def comps2boundaries(text_comps, comp_pred_labels):
    # 断言文本组件的维度为2
    assert text_comps.ndim == 2
    # 断言文本组件数量与组件预测标签数量相等
    assert len(text_comps) == len(comp_pred_labels)
    # 初始化边界列表
    boundaries = []
    # 如果文本组件数量小于1，则返回空边界列表
    if len(text_comps) < 1:
        return boundaries
    # 遍历每个聚类的索引
    for cluster_ind in range(0, int(np.max(comp_pred_labels)) + 1):
        # 获取属于当前聚类的文本组件索引
        cluster_comp_inds = np.where(comp_pred_labels == cluster_ind)
        # 提取文本组件的边界框坐标并重塑为4x2的数组
        text_comp_boxes = text_comps[cluster_comp_inds, :8].reshape(
            (-1, 4, 2)).astype(np.int32)
        # 计算当前聚类的平均分数
        score = np.mean(text_comps[cluster_comp_inds, -1])

        # 如果文本组件数量小于1，则继续下一次循环
        if text_comp_boxes.shape[0] < 1:
            continue

        # 如果文本组件数量大于1
        elif text_comp_boxes.shape[0] > 1:
            # 计算文本组件中心点坐标并找到最短连接路径
            centers = np.mean(text_comp_boxes, axis=1).astype(np.int32).tolist()
            shortest_path = min_connect_path(centers)
            text_comp_boxes = text_comp_boxes[shortest_path]
            # 计算顶部线和底部线的坐标
            top_line = np.mean(
                text_comp_boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
            bot_line = np.mean(
                text_comp_boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
            # 修正顶部线和底部线的坐标
            top_line, bot_line = fix_corner(
                top_line, bot_line, text_comp_boxes[0], text_comp_boxes[-1])
            # 构建边界点列表
            boundary_points = top_line + bot_line[::-1]

        # 如果文本组件数量为1
        else:
            # 计算顶部线和底部线的坐标
            top_line = text_comp_boxes[0, 0:2, :].astype(np.int32).tolist()
            bot_line = text_comp_boxes[0, 2:4:-1, :].astype(np.int32).tolist()
            # 构建边界点列表
            boundary_points = top_line + bot_line

        # 将边界点和分数组合成边界
        boundary = [p for coord in boundary_points for p in coord] + [score]
        # 将边界添加到边界列表中
        boundaries.append(boundary)

    # 返回边界列表

    return boundaries


class DRRGPostprocess(object):
    """合并文本组件并构建文本实例的边界。

    Args:
        link_thr (float): 边缘分数阈值。
    """

    def __init__(self, link_thr, **kwargs):
        # 断言边缘分数阈值为浮点数
        assert isinstance(link_thr, float)
        # 初始化边缘分数阈值
        self.link_thr = link_thr
    # 定义一个方法，用于对预测结果进行处理并返回文本实例的边界
    def __call__(self, preds, shape_list):
        """
        Args:
            edges (ndarray): The edge array of shape N * 2, each row is a node
                index pair that makes up an edge in graph.
            scores (ndarray): The edge score array of shape (N,).
            text_comps (ndarray): The text components.

        Returns:
            List[list[float]]: The predicted boundaries of text instances.
        """
        # 解析预测结果
        edges, scores, text_comps = preds
        # 如果存在边
        if edges is not None:
            # 将边转换为numpy数组
            if isinstance(edges, paddle.Tensor):
                edges = edges.numpy()
            # 将得分转换为numpy数组
            if isinstance(scores, paddle.Tensor):
                scores = scores.numpy()
            # 将文本组件转换为numpy数组
            if isinstance(text_comps, paddle.Tensor):
                text_comps = text_comps.numpy()
            # 检查边和得分数组的长度是否一致
            assert len(edges) == len(scores)
            # 检查文本组件的维度是否为2
            assert text_comps.ndim == 2
            # 检查文本组件的第二个维度是否为9
            assert text_comps.shape[1] == 9

            # 进行图传播，得到顶点和得分字典
            vertices, score_dict = graph_propagation(edges, scores, text_comps)
            # 进行连通组件分析，得到聚类结果
            clusters = connected_components(vertices, score_dict, self.link_thr)
            # 将聚类结果转换为标签
            pred_labels = clusters2labels(clusters, text_comps.shape[0])
            # 移除单独的文本组件
            text_comps, pred_labels = remove_single(text_comps, pred_labels)
            # 将文本组件和标签转换为边界
            boundaries = comps2boundaries(text_comps, pred_labels)
        else:
            # 如果不存在边，则边界为空列表
            boundaries = []

        # 调整边界大小并返回边界框
        boundaries, scores = self.resize_boundary(
            boundaries, (1 / shape_list[0, 2:]).tolist()[::-1])
        boxes_batch = [dict(points=boundaries, scores=scores)]
        return boxes_batch
    # 通过缩放因子 scale_factor 重新调整边界

    # 参数 boundaries 是包含边界列表的列表。每个边界都有大小为 2k+1，其中 k>=4。
    # 参数 scale_factor 是大小为 (4,) 的缩放因子。

    # 返回值 boundaries 是缩放后的边界列表。

    def resize_boundary(self, boundaries, scale_factor):
        # 初始化空列表用于存储调整后的边界框和分数
        boxes = []
        scores = []
        
        # 遍历每个边界
        for b in boundaries:
            # 获取边界的长度
            sz = len(b)
            # 将边界的最后一个元素作为分数存储起来
            scores.append(b[-1])
            # 对边界进行缩放操作
            b = (np.array(b[:sz - 1]) *
                 (np.tile(scale_factor[:2], int(
                     (sz - 1) / 2)).reshape(1, sz - 1))).flatten().tolist()
            # 将调整后的边界框添加到 boxes 列表中
            boxes.append(np.array(b).reshape([-1, 2]))
        
        # 返回调整后的边界框列表和分数列表
        return boxes, scores
```