# `.\PaddleOCR\ppocr\utils\e2e_metric\Deteval.py`

```py
# 版权声明和许可证信息
# 本代码版权归 PaddlePaddle 作者所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”提供的，没有任何明示或暗示的保证或条件。
# 请查看许可证以获取有关权限和限制的详细信息。

# 导入所需的库
import json
import numpy as np
import scipy.io as io

# 从 ppocr.utils.utility 模块中导入 check_install 函数
from ppocr.utils.utility import check_install

# 从 ppocr.utils.e2e_metric.polygon_fast 模块中导入 iod, area_of_intersection, area 函数

# 定义函数 get_socre_A，接受 gt_dir 和 pred_dict 两个参数
def get_socre_A(gt_dir, pred_dict):
    # 初始化变量 allInputs 为 1
    allInputs = 1

    # 定义内部函数 input_reading_mod，用于从 txt 文件中读取输入
    def input_reading_mod(pred_dict):
        """This helper reads input from txt files"""
        # 初始化空列表 det
        det = []
        # 获取预测字典的长度
        n = len(pred_dict)
        # 遍历预测字典
        for i in range(n):
            # 获取预测字典中第 i 个元素的 'points' 和 'texts' 字段值
            points = pred_dict[i]['points']
            text = pred_dict[i]['texts']
            # 将 points 数组展平并转换为字符串，用逗号分隔
            point = ",".join(map(str, points.reshape(-1, )))
            # 将展平后的点坐标和文本信息添加到 det 列表中
            det.append([point, text])
        # 返回处理后的结果列表 det
        return det
    # 定义一个函数，用于从 mat 文件中读取 groundtruths
    def gt_reading_mod(gt_dict):
        """This helper reads groundtruths from mat files"""
        # 初始化一个空列表用于存储 groundtruths
        gt = []
        # 获取 groundtruths 字典的长度
        n = len(gt_dict)
        # 遍历每个 groundtruth
        for i in range(n):
            # 获取 groundtruth 中的 points 列表并转换为 Python 列表
            points = gt_dict[i]['points'].tolist()
            # 获取 points 列表的长度
            h = len(points)
            # 获取 groundtruth 中的 text
            text = gt_dict[i]['text']
            # 初始化一个包含 x, y, #, # 的列表
            xx = [
                np.array(['x:'], dtype='<U2'), 0, np.array(['y:'], dtype='<U2'), 0, np.array(['#'], dtype='<U1'), np.array(['#'], dtype='<U1')
            ]
            # 初始化两个空列表用于存储 x 和 y 坐标
            t_x, t_y = [], []
            # 遍历 points 列表，分别将 x 和 y 坐标存入对应的列表中
            for j in range(h):
                t_x.append(points[j][0])
                t_y.append(points[j][1])
            # 将 x 坐标列表转换为 numpy 数组并存入 xx 列表
            xx[1] = np.array([t_x], dtype='int16')
            # 将 y 坐标列表转换为 numpy 数组并存入 xx 列表
            xx[3] = np.array([t_y], dtype='int16')
            # 如果 text 不为空，则将其转换为 numpy 数组并存入 xx 列表
            if text != "":
                xx[4] = np.array([text], dtype='U{}'.format(len(text)))
                xx[5] = np.array(['c'], dtype='<U1')
            # 将当前 groundtruth 添加到 gt 列表中
            gt.append(xx)
        # 返回处理后的 groundtruth 列表
        return gt

    # 定义一个函数，用于过滤检测结果
    def detection_filtering(detections, groundtruths, threshold=0.5):
        # 遍历每个 groundtruth
        for gt_id, gt in enumerate(groundtruths):
            # 如果 groundtruth 的第五个元素为 '#' 且 x 坐标数组的长度大于 1
            if (gt[5] == '#') and (gt[1].shape[1] > 1):
                # 将 x 坐标数组转换为 Python 列表
                gt_x = list(map(int, np.squeeze(gt[1])))
                # 将 y 坐标数组转换为 Python 列表
                gt_y = list(map(int, np.squeeze(gt[3]))
                # 遍历每个检测结果
                for det_id, detection in enumerate(detections):
                    # 备份原始检测结果
                    detection_orig = detection
                    # 将检测结果字符串转换为列表，并转换为整数类型
                    detection = [float(x) for x in detection[0].split(',')]
                    detection = list(map(int, detection))
                    # 提取检测结果中的 x 和 y 坐标
                    det_x = detection[0::2]
                    det_y = detection[1::2]
                    # 计算检测结果与 groundtruth 的 IoU
                    det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                    # 如果 IoU 大于阈值，则将该检测结果置为空列表
                    if det_gt_iou > threshold:
                        detections[det_id] = []

                # 移除所有空列表元素
                detections[:] = [item for item in detections if item != []]
        # 返回过滤后的检测结果列表
        return detections
    # 计算 sigma 值，即交集面积除以真实框面积
    def sigma_calculation(det_x, det_y, gt_x, gt_y):
        """
        sigma = inter_area / gt_area
        """
        return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) /
                         area(gt_x, gt_y)), 2)

    # 计算 tau 值，即交集面积除以检测框面积
    def tau_calculation(det_x, det_y, gt_x, gt_y):
        if area(det_x, det_y) == 0.0:
            return 0
        return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) /
                         area(det_x, det_y)), 2)

    ##############################Initialization###################################
    # 初始化全局变量
    # global_sigma = []
    # global_tau = []
    # global_pred_str = []
    # global_gt_str = []
    ###############################################################################

    # 创建一个字典存储单个数据
    single_data = {}
    single_data['sigma'] = global_sigma
    single_data['global_tau'] = global_tau
    single_data['global_pred_str'] = global_pred_str
    single_data['global_gt_str'] = global_gt_str
    # 返回单个数据字典
    return single_data
def get_socre_B(gt_dir, img_id, pred_dict):
    allInputs = 1

    def input_reading_mod(pred_dict):
        """This helper reads input from txt files"""
        # 初始化一个空列表用于存储检测结果
        det = []
        # 获取预测字典的长度
        n = len(pred_dict)
        # 遍历预测字典
        for i in range(n):
            # 获取预测结果中的点坐标和文本信息
            points = pred_dict[i]['points']
            text = pred_dict[i]['texts']
            # 将点坐标转换为字符串形式
            point = ",".join(map(str, points.reshape(-1, )))
            # 将点坐标和文本信息添加到检测结果列表中
            det.append([point, text])
        return det

    def gt_reading_mod(gt_dir, gt_id):
        # 从指定路径读取真实标注数据
        gt = io.loadmat('%s/poly_gt_img%s.mat' % (gt_dir, gt_id))
        gt = gt['polygt']
        return gt

    def detection_filtering(detections, groundtruths, threshold=0.5):
        # 遍历真实标注数据
        for gt_id, gt in enumerate(groundtruths):
            # 判断是否为有效标注
            if (gt[5] == '#') and (gt[1].shape[1] > 1):
                # 获取真实标注的 x 和 y 坐标
                gt_x = list(map(int, np.squeeze(gt[1])))
                gt_y = list(map(int, np.squeeze(gt[3]))
                # 遍历检测结果
                for det_id, detection in enumerate(detections):
                    detection_orig = detection
                    # 将检测结果转换为列表形式
                    detection = [float(x) for x in detection[0].split(',')]
                    detection = list(map(int, detection))
                    det_x = detection[0::2]
                    det_y = detection[1::2]
                    # 计算检测结果与真实标注的 IoU
                    det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                    # 如果 IoU 大于阈值，则将该检测结果置为空列表
                    if det_gt_iou > threshold:
                        detections[det_id] = []

                # 移除空列表元素
                detections[:] = [item for item in detections if item != []]
        return detections

    def sigma_calculation(det_x, det_y, gt_x, gt_y):
        """
        sigma = inter_area / gt_area
        """
        # 计算 sigma 值
        return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) /
                         area(gt_x, gt_y)), 2)

    def tau_calculation(det_x, det_y, gt_x, gt_y):
        if area(det_x, det_y) == 0.0:
            return 0
        # 计算 tau 值
        return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) /
                         area(det_x, det_y)), 2)
    ##############################Initialization###################################
    # 初始化全局变量 global_sigma 为空列表
    # 初始化全局变量 global_tau 为空列表
    # 初始化全局变量 global_pred_str 为空列表
    # 初始化全局变量 global_gt_str 为空列表
    ###############################################################################

    # 创建一个空字典 single_data
    single_data = {}
    # 将全局变量 global_sigma 存入字典 single_data 的 'sigma' 键中
    single_data['sigma'] = global_sigma
    # 将全局变量 global_tau 存入字典 single_data 的 'global_tau' 键中
    single_data['global_tau'] = global_tau
    # 将全局变量 global_pred_str 存入字典 single_data 的 'global_pred_str' 键中
    single_data['global_pred_str'] = global_pred_str
    # 将全局变量 global_gt_str 存入字典 single_data 的 'global_gt_str' 键中
    single_data['global_gt_str'] = global_gt_str
    # 返回字典 single_data
    return single_data
# 定义一个函数，用于计算 CentripetalText (CT) 预测的得分
def get_score_C(gt_label, text, pred_bboxes):
    """
    get score for CentripetalText (CT) prediction.
    """
    # 检查并安装 Polygon 库
    check_install("Polygon", "Polygon3")
    # 导入 Polygon 库并重命名为 plg
    import Polygon as plg

    # 定义一个辅助函数，用于从 mat 文件中读取 groundtruths
    def gt_reading_mod(gt_label, text):
        """This helper reads groundtruths from mat files"""
        groundtruths = []
        nbox = len(gt_label)
        for i in range(nbox):
            # 将每个 groundtruth 转换为字典格式，包括文本和坐标点
            label = {"transcription": text[i][0], "points": gt_label[i].numpy()}
            groundtruths.append(label)

        return groundtruths

    # 计算两个多边形的并集面积
    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    # 计算两个多边形的交集面积
    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    # 过滤检测结果，根据阈值删除与 groundtruth 的 IoU 大于阈值的检测结果
    def detection_filtering(detections, groundtruths, threshold=0.5):
        for gt in groundtruths:
            point_num = gt['points'].shape[1] // 2
            if gt['transcription'] == '###' and (point_num > 1):
                gt_p = np.array(gt['points']).reshape(point_num, 2).astype('int32')
                gt_p = plg.Polygon(gt_p)

                for det_id, detection in enumerate(detections):
                    det_y = detection[0::2]
                    det_x = detection[1::2]

                    det_p = np.concatenate((np.array(det_x), np.array(det_y)))
                    det_p = det_p.reshape(2, -1).transpose()
                    det_p = plg.Polygon(det_p)

                    try:
                        det_gt_iou = get_intersection(det_p, gt_p) / det_p.area()
                    except:
                        print(det_x, det_y, gt_p)
                    if det_gt_iou > threshold:
                        detections[det_id] = []

                detections[:] = [item for item in detections if item != []]
        return detections
    # 计算 sigma 值，即交集面积除以 groundtruth 面积
    def sigma_calculation(det_p, gt_p):
        """
        sigma = inter_area / gt_area
        """
        # 如果 groundtruth 面积为 0，则返回 0
        if gt_p.area() == 0.:
            return 0
        # 返回交集面积除以 groundtruth 面积的结果
        return get_intersection(det_p, gt_p) / gt_p.area()

    # 计算 tau 值，即交集面积除以 detection 面积
    def tau_calculation(det_p, gt_p):
        """
        tau = inter_area / det_area
        """
        # 如果 detection 面积为 0，则返回 0
        if det_p.area() == 0.:
            return 0
        # 返回交集面积除以 detection 面积的结果
        return get_intersection(det_p, gt_p) / det_p.area()

    # 初始化 detections 列表
    detections = []

    # 将预测框中的坐标转换并添加到 detections 列表中
    for item in pred_bboxes:
        detections.append(item[:, ::-1].reshape(-1))

    # 读取 groundtruths 数据
    groundtruths = gt_reading_mod(gt_label, text)

    # 过滤掉与 DC 区域重叠的检测结果
    detections = detection_filtering(
        detections, groundtruths)

    # 移除 groundtruths 中 transcription 为 '###' 的数据
    for idx in range(len(groundtruths) - 1, -1, -1):
        # 注意：源代码中使用 'orin' 表示 '#'，这里使用 'anno'，可能会导致 fscore 稍微下降，约为 0.12
        if groundtruths[idx]['transcription'] == '###':
            groundtruths.pop(idx)

    # 初始化 local_sigma_table 和 local_tau_table
    local_sigma_table = np.zeros((len(groundtruths), len(detections)))
    local_tau_table = np.zeros((len(groundtruths), len(detections)))

    # 计算每个 groundtruth 与 detection 之间的 sigma 和 tau 值
    for gt_id, gt in enumerate(groundtruths):
        if len(detections) > 0:
            for det_id, detection in enumerate(detections):
                point_num = gt['points'].shape[1] // 2

                gt_p = np.array(gt['points']).reshape(point_num,
                                                      2).astype('int32')
                gt_p = plg.Polygon(gt_p)

                det_y = detection[0::2]
                det_x = detection[1::2]

                det_p = np.concatenate((np.array(det_x), np.array(det_y)))

                det_p = det_p.reshape(2, -1).transpose()
                det_p = plg.Polygon(det_p)

                local_sigma_table[gt_id, det_id] = sigma_calculation(det_p,
                                                                     gt_p)
                local_tau_table[gt_id, det_id] = tau_calculation(det_p, gt_p)

    # 初始化 data 字典
    data = {}
    # 将本地 sigma 表赋值给 data 字典中的 'sigma' 键
    data['sigma'] = local_sigma_table
    # 将本地 tau 表赋值给 data 字典中的 'global_tau' 键
    data['global_tau'] = local_tau_table
    # 将空字符串赋值给 data 字典中的 'global_pred_str' 键
    data['global_pred_str'] = ''
    # 将空字符串赋值给 data 字典中的 'global_gt_str' 键
    data['global_gt_str'] = ''
    # 返回更新后的 data 字典
    return data
# 组合多个数据结果，计算综合评估指标
def combine_results(all_data, rec_flag=True):
    # 初始化参数
    tr = 0.7
    tp = 0.6
    fsc_k = 0.8
    k = 2
    global_sigma = []
    global_tau = []
    global_pred_str = []
    global_gt_str = []

    # 遍历所有数据，将各项指标存入全局列表中
    for data in all_data:
        global_sigma.append(data['sigma'])
        global_tau.append(data['global_tau'])
        global_pred_str.append(data['global_pred_str'])
        global_gt_str.append(data['global_gt_str'])

    # 初始化累积召回率、累积精确率、总真实标注数、总检测数、匹配字符串数、匹配数
    global_accumulative_recall = 0
    global_accumulative_precision = 0
    total_num_gt = 0
    total_num_det = 0
    hit_str_count = 0
    hit_count = 0

    # 计算召回率，处理除零错误
    try:
        recall = global_accumulative_recall / total_num_gt
    except ZeroDivisionError:
        recall = 0

    # 计算精确率，处理除零错误
    try:
        precision = global_accumulative_precision / total_num_det
    except ZeroDivisionError:
        precision = 0

    # 计算 F1 分数，处理除零错误
    try:
        f_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f_score = 0

    # 计算序列错误率，处理除零错误
    try:
        seqerr = 1 - float(hit_str_count) / global_accumulative_recall
    except ZeroDivisionError:
        seqerr = 1

    # 计算端到端召回率，处理除零错误
    try:
        recall_e2e = float(hit_str_count) / total_num_gt
    except ZeroDivisionError:
        recall_e2e = 0

    # 计算端到端精确率，处理除零错误
    try:
        precision_e2e = float(hit_str_count) / total_num_det
    except ZeroDivisionError:
        precision_e2e = 0

    # 计算端到端 F1 分数，处理除零错误
    try:
        f_score_e2e = 2 * precision_e2e * recall_e2e / (precision_e2e + recall_e2e)
    except ZeroDivisionError:
        f_score_e2e = 0

    # 构建最终结果字典
    final = {
        'total_num_gt': total_num_gt,
        'total_num_det': total_num_det,
        'global_accumulative_recall': global_accumulative_recall,
        'hit_str_count': hit_str_count,
        'recall': recall,
        'precision': precision,
        'f_score': f_score,
        'seqerr': seqerr,
        'recall_e2e': recall_e2e,
        'precision_e2e': precision_e2e,
        'f_score_e2e': f_score_e2e
    }
    # 返回最终结果
    return final
```