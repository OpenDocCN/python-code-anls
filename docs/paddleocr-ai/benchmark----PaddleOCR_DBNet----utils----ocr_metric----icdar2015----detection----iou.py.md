# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\ocr_metric\icdar2015\detection\iou.py`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入必要的库
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
import cv2

# 定义一个函数用于计算旋转框之间的 IoU
def iou_rotate(box_a, box_b, method='union'):
    # 计算 box_a 和 box_b 的最小外接矩形
    rect_a = cv2.minAreaRect(box_a)
    rect_b = cv2.minAreaRect(box_b)
    # 计算两个旋转矩形的交集
    r1 = cv2.rotatedRectangleIntersection(rect_a, rect_b)
    # 如果交集为空，则返回 IoU 为 0
    if r1[0] == 0:
        return 0
    else:
        # 计算交集的面积
        inter_area = cv2.contourArea(r1[1])
        # 计算 box_a 和 box_b 的面积
        area_a = cv2.contourArea(box_a)
        area_b = cv2.contourArea(box_b)
        # 计算并集的面积
        union_area = area_a + area_b - inter_area
        # 如果并集面积或交集面积为 0，则返回 IoU 为 0
        if union_area == 0 or inter_area == 0:
            return 0
        # 根据计算方法计算 IoU
        if method == 'union':
            iou = inter_area / union_area
        elif method == 'intersection':
            iou = inter_area / min(area_a, area_b)
        else:
            raise NotImplementedError
        return iou

# 定义一个类用于检测 IoU 的评估器
class DetectionIoUEvaluator(object):
    def __init__(self,
                 is_output_polygon=False,
                 iou_constraint=0.5,
                 area_precision_constraint=0.5):
        # 初始化评估器的参数
        self.is_output_polygon = is_output_polygon
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint
    # 合并多个结果，计算全局关注的真实值数量、检测值数量和匹配数量的总和
    def combine_results(self, results):
        # 初始化全局关注的真实值数量、检测值数量和匹配数量的总和
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        # 遍历每个结果
        for result in results:
            # 累加全局关注的真实值数量
            numGlobalCareGt += result['gtCare']
            # 累加全局关注的检测值数量
            numGlobalCareDet += result['detCare']
            # 累加匹配数量
            matchedSum += result['detMatched']

        # 计算方法的召回率，如果全局关注的真实值数量为0，则召回率为0，否则计算匹配数量除以全局关注的真实值数量
        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        # 计算方法的精确率，如果全局关注的检测值数量为0，则精确率为0，否则计算匹配数量除以全局关注的检测值数量
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        # 计算方法的 F1 值，如果召回率和精确率之和为0，则 F1 值为0，否则计算 F1 值公式
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)

        # 构建方法的评估指标字典，包括精确率、召回率和 F1 值
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        # 返回方法的评估指标字典
        return methodMetrics
# 如果当前脚本作为主程序运行
if __name__ == '__main__':
    # 创建一个检测器评估器对象
    evaluator = DetectionIoUEvaluator()
    # 预测结果列表，每个元素是一个字典，包含检测框的坐标、文本和是否忽略等信息
    preds = [[{
        'points': [(0.1, 0.1), (0.5, 0), (0.5, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(0.5, 0.1), (1, 0), (1, 1), (0.5, 1)],
        'text': 5678,
        'ignore': False,
    }]]
    # 真实标注结果列表，每个元素是一个字典，包含检测框的坐标、文本和是否忽略等信息
    gts = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    # 存储评估结果的列表
    results = []
    # 遍历真实标注结果和预测结果，进行评估
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    # 将评估结果合并得到最终指标
    metrics = evaluator.combine_results(results)
    # 打印最终指标
    print(metrics)
```