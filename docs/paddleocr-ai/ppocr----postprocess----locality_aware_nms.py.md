# `.\PaddleOCR\ppocr\postprocess\locality_aware_nms.py`

```py
# 导入所需的库
import numpy as np
from shapely.geometry import Polygon

# 计算两个多边形的交集
def intersection(g, p):
    """
    Intersection.
    """
    # 将坐标转换为多边形对象
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    # 对多边形进行缓冲处理
    g = g.buffer(0)
    p = p.buffer(0)
    # 检查多边形是否有效
    if not g.is_valid or not p.is_valid:
        return 0
    # 计算交集和并集的面积
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union

# 计算两个多边形的交集比例
def intersection_iog(g, p):
    """
    Intersection_iog.
    """
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = p.area
    if union == 0:
        print("p_area is very small")
        return 0
    else:
        return inter / union

# 权重合并两个多边形
def weighted_merge(g, p):
    """
    Weighted merge.
    """
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])
    g[8] = (g[8] + p[8])
    return g

# 标准非极大值抑制
def standard_nms(S, thres):
    """
    Standard nms.
    """
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]

# 标准非极大值抑制，返回索引
def standard_nms_inds(S, thres):
    """
    Standard nms, retun inds.
    """
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return keep

# 非极大值抑制
def nms(S, thres):
    """
    nms.
    """
    order = np.argsort(S[:, 8])[::-1]
    keep = []
    # 当订单大小大于0时，进入循环
    while order.size > 0:
        # 获取订单中的第一个元素
        i = order[0]
        # 将第一个元素添加到保留列表中
        keep.append(i)
        # 计算当前元素与订单中其他元素的交集
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])

        # 找到交集小于阈值的元素的索引
        inds = np.where(ovr <= thres)[0]
        # 更新订单，去除交集小于阈值的元素
        order = order[inds + 1]

    # 返回保留列表
    return keep
# 定义软非极大值抑制函数，用于对输入的边界框进行处理
def soft_nms(boxes_in, Nt_thres=0.3, threshold=0.8, sigma=0.5, method=2):
    """
    soft_nms
    :para boxes_in, N x 9 (coords + score)  # 输入参数为 N 行 9 列的数组，包含坐标和分数信息
    :para threshould, eliminate cases min score(0.001)  # 分数低于阈值的边界框将被消除
    :para Nt_thres, iou_threshi  # IOU 阈值
    :para sigma, gaussian weght  # 高斯权重参数
    :method, linear or gaussian  # 方法选择，线性或高斯
    """
    # 复制输入的边界框数组
    boxes = boxes_in.copy()
    # 获取边界框的数量
    N = boxes.shape[0]
    # 如果边界框数量为空或小于1，则返回空数组
    if N is None or N < 1:
        return np.array([])
    # 初始化变量
    pos, maxpos = 0, 0
    weight = 0.0
    # 生成索引数组
    inds = np.arange(N)
    # 复制第一个边界框作为 tbox 和 sbox
    tbox, sbox = boxes[0].copy(), boxes[0].copy()
    # 遍历 N 个框
    for i in range(N):
        # 获取当前框的最大分数和位置
        maxscore = boxes[i, 8]
        maxpos = i
        # 复制当前框的信息
        tbox = boxes[i].copy()
        # 获取当前框的索引
        ti = inds[i]
        # 设置下一个位置
        pos = i + 1
        # 获取最大框
        while pos < N:
            # 如果当前位置的框分数比最大分数大，则更新最大分数和位置
            if maxscore < boxes[pos, 8]:
                maxscore = boxes[pos, 8]
                maxpos = pos
            pos = pos + 1
        # 将最大框添加为检测结果
        boxes[i, :] = boxes[maxpos, :]
        inds[i] = inds[maxpos]
        # 交换框的位置
        boxes[maxpos, :] = tbox
        inds[maxpos] = ti
        tbox = boxes[i].copy()
        pos = i + 1
        # NMS 迭代
        while pos < N:
            # 复制当前框信息
            sbox = boxes[pos].copy()
            # 计算当前框和最大框的交并比
            ts_iou_val = intersection(tbox, sbox)
            # 根据不同方法计算权重
            if ts_iou_val > 0:
                if method == 1:
                    if ts_iou_val > Nt_thres:
                        weight = 1 - ts_iou_val
                    else:
                        weight = 1
                elif method == 2:
                    weight = np.exp(-1.0 * ts_iou_val**2 / sigma)
                else:
                    if ts_iou_val > Nt_thres:
                        weight = 0
                    else:
                        weight = 1
                boxes[pos, 8] = weight * boxes[pos, 8]
                # 如果框的分数低于阈值，则丢弃该框
                if boxes[pos, 8] < threshold:
                    boxes[pos, :] = boxes[N - 1, :]
                    inds[pos] = inds[N - 1]
                    N = N - 1
                    pos = pos - 1
            pos = pos + 1

    # 返回前 N 个框的信息
    return boxes[:N]
# 定义一个函数，实现EAST算法的局部感知非极大值抑制（NMS）
def nms_locality(polys, thres=0.3):
    """
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    """
    # 初始化一个空列表S，用于存储合并后的多边形
    S = []
    # 初始化变量p为None
    p = None
    # 遍历输入的多边形数组
    for g in polys:
        # 如果p不为None且当前多边形与上一个多边形的交集大于阈值thres
        if p is not None and intersection(g, p) > thres:
            # 对当前多边形和上一个多边形进行加权合并
            p = weighted_merge(g, p)
        else:
            # 如果p不为None且当前多边形与上一个多边形的交集不大于阈值thres
            if p is not None:
                # 将上一个多边形加入到列表S中
                S.append(p)
            # 更新p为当前多边形
            p = g
    # 将最后一个多边形加入到列表S中
    if p is not None:
        S.append(p)

    # 如果列表S为空，则返回一个空的numpy数组
    if len(S) == 0:
        return np.array([])
    # 对列表S中的多边形进行标准的非极大值抑制处理，并返回结果
    return standard_nms(np.array(S), thres)


if __name__ == '__main__':
    # 创建一个四边形对象，计算其面积并打印输出
    print(
        Polygon(np.array([[343, 350], [448, 135], [474, 143], [369, 359]]))
        .area)
```