# `.\PaddleOCR\ppocr\metrics\eval_det_iou.py`

```
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 导入必要的库
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
"""
reference from :
https://github.com/MhLiao/DB/blob/3c32b808d4412680310d3d28eeb6a2d5bf1566c5/concern/icdar2015_eval/detection/iou.py#L8
"""

# 定义一个检测器的IoU评估器类
class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        # 初始化IoU约束和面积精度约束
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    # 合并结果的方法
    def combine_results(self, results):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        # 遍历结果列表
        for result in results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        # 计算方法的召回率、精确率和F1分数
        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics

# 主函数入口
if __name__ == '__main__':
    # 创建一个检测器的IoU评估器对象
    evaluator = DetectionIoUEvaluator()
    # 定义一些ground truth和predictions
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': False,
    }]]
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],
        'text': 123,
        'ignore': False,
    }]]
    results = []
    # 遍历真实标签和预测结果的列表，将每一对真实标签和预测结果传入评估器进行评估，并将评估结果添加到结果列表中
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    # 将所有评估结果合并为一个整体指标
    metrics = evaluator.combine_results(results)
    # 打印合并后的评估指标
    print(metrics)
```