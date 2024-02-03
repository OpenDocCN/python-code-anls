# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\ocr_metric\icdar2015\detection\icdar2013.py`

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 导入 math 模块
import math
# 导入 namedtuple 类
from collections import namedtuple
# 导入 numpy 模块并使用别名 np
import numpy as np
# 导入 Polygon 类
from shapely.geometry import Polygon

# 定义 DetectionICDAR2013Evaluator 类
class DetectionICDAR2013Evaluator(object):
    # 初始化函数，设置默认参数
    def __init__(self,
                 area_recall_constraint=0.8,
                 area_precision_constraint=0.4,
                 ev_param_ind_center_diff_thr=1,
                 mtype_oo_o=1.0,
                 mtype_om_o=0.8,
                 mtype_om_m=1.0):

        self.area_recall_constraint = area_recall_constraint
        self.area_precision_constraint = area_precision_constraint
        self.ev_param_ind_center_diff_thr = ev_param_ind_center_diff_thr
        self.mtype_oo_o = mtype_oo_o
        self.mtype_om_o = mtype_om_o
        self.mtype_om_m = mtype_om_m

    # 合并结果函数
    def combine_results(self, results):
        numGt = 0
        numDet = 0
        methodRecallSum = 0
        methodPrecisionSum = 0

        # 遍历结果列表
        for result in results:
            numGt += result['gtCare']
            numDet += result['detCare']
            methodRecallSum += result['recallAccum']
            methodPrecisionSum += result['precisionAccum']

        # 计算方法的召回率、精确率和 F1 值
        methodRecall = 0 if numGt == 0 else methodRecallSum / numGt
        methodPrecision = 0 if numDet == 0 else methodPrecisionSum / numDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
            methodRecall + methodPrecision)

        # 存储方法的评估指标
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }

        return methodMetrics

# 主程序入口
if __name__ == '__main__':
    # 创建 DetectionICDAR2013Evaluator 实例
    evaluator = DetectionICDAR2013Evaluator()
    # 设置 ground truth 数据
    gts = [[{
        'points': [(0, 0), (1, 0), (1, 1), (0, 1)],
        'text': 1234,
        'ignore': False,
    }, {
        'points': [(2, 2), (3, 2), (3, 3), (2, 3)],
        'text': 5678,
        'ignore': True,
    }]]
    # 定义预测结果列表，每个元素是一个包含预测信息的字典
    preds = [[{
        'points': [(0.1, 0.1), (1, 0), (1, 1), (0, 1)],  # 定义预测框的四个顶点坐标
        'text': 123,  # 预测的文本内容
        'ignore': False,  # 是否忽略该预测结果
    }]]
    # 初始化结果列表
    results = []
    # 遍历真实标签和预测结果，计算评估指标
    for gt, pred in zip(gts, preds):
        results.append(evaluator.evaluate_image(gt, pred))
    # 组合所有评估结果
    metrics = evaluator.combine_results(results)
    # 打印评估指标
    print(metrics)
```