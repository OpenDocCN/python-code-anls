# `.\PaddleOCR\ppstructure\kie\tools\eval_with_label_end2end.py`

```
# 导入所需的库
import os
import re
import sys
import shapely
from shapely.geometry import Polygon
import numpy as np
from collections import defaultdict
import operator
from rapidfuzz.distance import Levenshtein
import argparse
import json
import copy

# 解析文本文件中的结果数据，返回一个字典
def parse_ser_results_fp(fp, fp_type="gt", ignore_background=True):
    # 确保文件类型为"gt"或"pred"
    assert fp_type in ["gt", "pred"]
    # 根据文件类型选择关键字
    key = "label" if fp_type == "gt" else "pred"
    # 创建一个空字典用于存储结果数据
    res_dict = dict()
    # 打开文件并读取内容
    with open(fp, "r", encoding='utf-8') as fin:
        # 逐行读取文件内容
        lines = fin.readlines()
    # 遍历每一行数据，获取图片路径和信息
    for _, line in enumerate(lines):
        img_path, info = line.strip().split("\t")
        # 获取图片文件名
        image_name = os.path.basename(img_path)
        # 初始化结果字典中该图片对应的值为空列表
        res_dict[image_name] = []
        # 解析信息字符串为 JSON 格式
        json_info = json.loads(info)
        # 遍历 OCR 信息列表
        for single_ocr_info in json_info["ocr_info"]:
            # 获取标签信息并转换为大写
            label = single_ocr_info[key].upper()
            # 如果标签为 "O", "OTHERS", "OTHER" 中的一个，则统一为 "O"
            if label in ["O", "OTHERS", "OTHER"]:
                label = "O"
            # 如果忽略背景并且标签为 "O"，则跳过当前循环
            if ignore_background and label == "O":
                continue
            # 将标签信息添加到单个 OCR 信息中
            single_ocr_info["label"] = label
            # 将处理后的 OCR 信息添加到结果字典中对应图片的值列表中
            res_dict[image_name].append(copy.deepcopy(single_ocr_info))
    # 返回结果字典
    return res_dict
# 从字符串中创建一个 shapely 多边形对象
def polygon_from_str(polygon_points):
    # 将多边形点坐标转换为 numpy 数组，并重塑为 4 行 2 列的形状
    polygon_points = np.array(polygon_points).reshape(4, 2)
    # 创建一个多边形对象，并获取其凸包
    polygon = Polygon(polygon_points).convex_hull
    return polygon

# 计算两个 shapely 多边形之间的交并比
def polygon_iou(poly1, poly2):
    # 如果两个多边形不相交，则交并比为 0
    if not poly1.intersects(poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            # 计算两个多边形的交集面积和并集面积，然后计算交并比
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            # 如果出现拓扑错误，则将交并比设置为 0
            print('shapely.geos.TopologicalError occurred, iou set to 0')
            iou = 0
    return iou

# 将字符串转换为 Levenshtein 距离，根据参数进行忽略空格和大小写处理
def ed(args, str1, str2):
    if args.ignore_space:
        # 如果需要忽略空格，则将字符串中的空格替换为空
        str1 = str1.replace(" ", "")
        str2 = str2.replace(" ", "")
    if args.ignore_case:
        # 如果需要忽略大小写，则将字符串转换为小写
        str1 = str1.lower()
        str2 = str2.lower()
    return Levenshtein.distance(str1, str2)

# 将边界框转换为多边形
def convert_bbox_to_polygon(bbox):
    """
    bbox  : [x1, y1, x2, y2]
    output: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    """
    xmin, ymin, xmax, ymax = bbox
    # 根据边界框的坐标信息创建多边形的点坐标列表
    poly = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    return poly

# 评估端到端的结果
def eval_e2e(args):
    # 解析 gt 结果
    gt_results = parse_ser_results_fp(args.gt_json_path, "gt", args.ignore_background)
    # 解析 pred 结果
    dt_results = parse_ser_results_fp(args.pred_json_path, "pred", args.ignore_background)
    iou_thresh = args.iou_thres
    num_gt_chars = 0
    gt_count = 0
    dt_count = 0
    hit = 0
    ed_sum = 0
    # 遍历检测结果中的每个图像名称
    for img_name in dt_results:
        # 获取对应图像名称的真实标注信息
        gt_info = gt_results[img_name]
        # 累加真实标注信息中的数量
        gt_count += len(gt_info)

        # 获取对应图像名称的检测结果信息
        dt_info = dt_results[img_name]
        # 累加检测结果信息中的数量
        dt_count += len(dt_info)

        # 初始化用于记录检测结果是否匹配的列表
        dt_match = [False] * len(dt_info)
        gt_match = [False] * len(gt_info)

        # 初始化存储所有 IOU 值的字典
        all_ious = defaultdict(tuple)
        # 遍历真实标注信息和检测结果信息，计算它们之间的 IOU 值
        for index_gt, gt in enumerate(gt_info):
            if "poly" not in gt:
                gt["poly"] = convert_bbox_to_polygon(gt["bbox"])
            gt_poly = polygon_from_str(gt["poly"])
            for index_dt, dt in enumerate(dt_info):
                if "poly" not in dt:
                    dt["poly"] = convert_bbox_to_polygon(dt["bbox"])
                dt_poly = polygon_from_str(dt["poly"])
                iou = polygon_iou(dt_poly, gt_poly)
                if iou >= iou_thresh:
                    all_ious[(index_gt, index_dt)] = iou
        # 根据 IOU 值对字典进行排序
        sorted_ious = sorted(
            all_ious.items(), key=operator.itemgetter(1), reverse=True)
        sorted_gt_dt_pairs = [item[0] for item in sorted_ious]

        # 匹配真实标注和检测结果
        for gt_dt_pair in sorted_gt_dt_pairs:
            index_gt, index_dt = gt_dt_pair
            if gt_match[index_gt] == False and dt_match[index_dt] == False:
                gt_match[index_gt] = True
                dt_match[index_dt] = True
                # 获取真实标注和检测结果的文本信息
                gt_text = gt_info[index_gt]["text"]
                dt_text = dt_info[index_dt]["text"]

                # 获取真实标注和检测结果的标签信息
                gt_label = gt_info[index_gt]["label"]
                dt_label = dt_info[index_dt]["pred"]

                # 如果条件为真，则执行以下操作
                if True:  # ignore_masks[index_gt] == '0':
                    # 计算编辑距离
                    ed_sum += ed(args, gt_text, dt_text)
                    num_gt_chars += len(gt_text)
                    # 如果真实标注和检测结果的文本相同
                    if gt_text == dt_text:
                        # 如果忽略 SER 预测或者真实标签和预测标签相同
                        if args.ignore_ser_prediction or gt_label == dt_label:
                            # 命中数加一
                            hit += 1
# 未匹配的 dt
        # 遍历 dt_match 列表，找出未匹配的 dt
        for tindex, dt_match_flag in enumerate(dt_match):
            # 如果 dt 未匹配
            if dt_match_flag == False:
                # 获取未匹配的 dt 的文本信息
                dt_text = dt_info[tindex]["text"]
                gt_text = ""
                # 计算编辑距离并累加到 ed_sum 中
                ed_sum += ed(args, dt_text, gt_text)

# 未匹配的 gt
        # 遍历 gt_match 列表，找出未匹配的 gt
        for tindex, gt_match_flag in enumerate(gt_match):
            # 如果 gt 未匹配
            if gt_match_flag == False:
                dt_text = ""
                # 获取未匹配的 gt 的文本信息
                gt_text = gt_info[tindex]["text"]
                # 计算编辑距离并累加到 ed_sum 中，同时计算 gt 文本字符数并累加到 num_gt_chars 中
                ed_sum += ed(args, gt_text, dt_text)
                num_gt_chars += len(gt_text)

    # 定义一个极小值 eps
    eps = 1e-9
    # 打印配置参数 args
    print("config: ", args)
    # 打印命中数、dt 数量、gt 数量
    print('hit, dt_count, gt_count', hit, dt_count, gt_count)
    # 计算精确率、召回率、F1 值
    precision = hit / (dt_count + eps)
    recall = hit / (gt_count + eps)
    fmeasure = 2.0 * precision * recall / (precision + recall + eps)
    # 计算平均编辑距离（图片级别和字段级别）、字符准确率
    avg_edit_dist_img = ed_sum / len(gt_results)
    avg_edit_dist_field = ed_sum / (gt_count + eps)
    character_acc = 1 - ed_sum / (num_gt_chars + eps)

    # 打印字符准确率、字段级别平均编辑距离、图片级别平均编辑距离、精确率、召回率、F1 值
    print('character_acc: %.2f' % (character_acc * 100) + "%")
    print('avg_edit_dist_field: %.2f' % (avg_edit_dist_field))
    print('avg_edit_dist_img: %.2f' % (avg_edit_dist_img))
    print('precision: %.2f' % (precision * 100) + "%")
    print('recall: %.2f' % (recall * 100) + "%")
    print('fmeasure: %.2f' % (fmeasure * 100) + "%")

    # 返回
    return


def parse_args():
    """
    解析命令行参数
    """

    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser()
    ## 必需参数
    # 添加 gt_json_path 参数
    parser.add_argument(
        "--gt_json_path",
        default=None,
        type=str,
        required=True, )
    # 添加 pred_json_path 参数
    parser.add_argument(
        "--pred_json_path",
        default=None,
        type=str,
        required=True, )

    # 添加 iou_thres 参数
    parser.add_argument("--iou_thres", default=0.5, type=float)

    # 添加 ignore_case 参数
    parser.add_argument(
        "--ignore_case",
        default=False,
        type=str2bool,
        help="whether to do lower case for the strs")
    # 添加一个命令行参数，用于指定是否忽略空格，默认为True
    parser.add_argument(
        "--ignore_space",
        default=True,
        type=str2bool,
        help="whether to ignore space")

    # 添加一个命令行参数，用于指定是否忽略背景标签，默认为True
    parser.add_argument(
        "--ignore_background",
        default=True,
        type=str2bool,
        help="whether to ignore other label")

    # 添加一个命令行参数，用于指定是否忽略OCR预测结果，默认为False
    parser.add_argument(
        "--ignore_ser_prediction",
        default=False,
        type=str2bool,
        help="whether to ignore ocr pred results")

    # 解析命令行参数并返回结果
    args = parser.parse_args()
    return args
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == '__main__':
    # 解析命令行参数并存储在args中
    args = parse_args()
    # 调用eval_e2e函数，并传入args参数
    eval_e2e(args)
```