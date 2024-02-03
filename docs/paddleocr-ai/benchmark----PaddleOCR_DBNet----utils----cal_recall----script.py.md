# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\cal_recall\script.py`

```
# 指定 Python 解释器路径和编码格式
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 导入必要的模块和函数
from collections import namedtuple
from . import rrc_evaluation_funcs
import Polygon as plg
import numpy as np

# 定义默认的评估参数
def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'IOU_CONSTRAINT': 0.5,
        'AREA_PRECISION_CONSTRAINT': 0.5,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'LTRB':
        False,  # LTRB:2points(left,top,right,bottom) or 4 points(x1,y1,x2,y2,x3,y3,x4,y4)
        'CRLF': False,  # Lines are delimited by Windows CRLF format
        'CONFIDENCES':
        False,  # Detections must include confidence value. AP will be calculated
        'PER_SAMPLE_RESULTS':
        True  # Generate per sample results and produce data for visualization
    }

# 验证数据的有效性
def validate_data(gtFilePath, submFilePath, evaluationParams):
    """
    Method validate_data: validates that all files in the results folder are correct (have the correct name contents).
                            Validates also that there are no missing files in the folder.
                            If some error detected, the method raises the error
    """
    # 加载 GroundTruth 文件夹中的文件
    gt = rrc_evaluation_funcs.load_folder_file(
        gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])

    # 加载提交结果文件夹中的文件
    subm = rrc_evaluation_funcs.load_folder_file(
        submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    # 验证 GroundTruth 的格式
    for k in gt:
        rrc_evaluation_funcs.validate_lines_in_file(
            k, gt[k], evaluationParams['CRLF'], evaluationParams['LTRB'], True)

    # 验证结果的格式
    # 遍历提交的结果字典中的每个键
    for k in subm:
        # 如果键不在参考结果字典中，则抛出异常
        if (k in gt) == False:
            raise Exception("The sample %s not present in GT" % k)

        # 调用验证函数，验证提交结果中的每行是否在参考结果中
        rrc_evaluation_funcs.validate_lines_in_file(
            k, subm[k], evaluationParams['CRLF'], evaluationParams['LTRB'],
            False, evaluationParams['CONFIDENCES'])
# 定义一个评估方法，用于评估方法并返回结果
def evaluate_method(gtFilePath, submFilePath, evaluationParams):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    # 从点列表中返回一个多边形对象，用于与 Polygon2 类一起使用，点列表包含 8 个点：x1,y1,x2,y2,x3,y3,x4,y4
    def polygon_from_points(points):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)

    # 将矩形转换为多边形
    def rectangle_to_polygon(rect):
        resBoxes = np.empty([1, 8], dtype='int32')
        resBoxes[0, 0] = int(rect.xmin)
        resBoxes[0, 4] = int(rect.ymax)
        resBoxes[0, 1] = int(rect.xmin)
        resBoxes[0, 5] = int(rect.ymin)
        resBoxes[0, 2] = int(rect.xmax)
        resBoxes[0, 6] = int(rect.ymin)
        resBoxes[0, 3] = int(rect.xmax)
        resBoxes[0, 7] = int(rect.ymax)

        pointMat = resBoxes[0].reshape([2, 4]).T

        return plg.Polygon(pointMat)

    # 将矩形转换为点列表
    def rectangle_to_points(rect):
        points = [
            int(rect.xmin), int(rect.ymax), int(rect.xmax), int(rect.ymax),
            int(rect.xmax), int(rect.ymin), int(rect.xmin), int(rect.ymin)
        ]
        return points

    # 获取两个多边形的并集
    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)
    # 计算两个集合的交集与并集的比值
    def get_intersection_over_union(pD, pG):
        try:
            # 调用计算交集函数和并集函数，计算交集与并集的比值
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            # 如果出现异常（比如除数为0），返回0
            return 0

    # 计算两个集合的交集
    def get_intersection(pD, pG):
        # 取两个集合的交集
        pInt = pD & pG
        # 如果交集为空集，返回0
        if len(pInt) == 0:
            return 0
        # 返回交集的面积
        return pInt.area()

    # 计算平均准确率（Average Precision）
    def compute_ap(confList, matchList, numGtCare):
        correct = 0
        AP = 0
        # 如果置信度列表不为空
        if len(confList) > 0:
            # 将置信度列表和匹配列表转换为NumPy数组
            confList = np.array(confList)
            matchList = np.array(matchList)
            # 根据置信度对数组进行降序排序
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            # 遍历置信度列表
            for n in range(len(confList)):
                match = matchList[n]
                # 如果匹配成功
                if match:
                    correct += 1
                    # 计算准确率
                    AP += float(correct) / (n + 1)

            # 如果有关注的真实值数量大于0，计算平均准确率
            if numGtCare > 0:
                AP /= numGtCare

        # 返回平均准确率
        return AP

    # 初始化每个样本的指标字典
    perSampleMetrics = {}

    # 初始化匹配总数
    matchedSum = 0

    # 定义一个命名元组Rectangle，表示矩形
    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

    # 从文件中加载真实值和检测值
    gt = rrc_evaluation_funcs.load_folder_file(
        gtFilePath, evaluationParams['GT_SAMPLE_NAME_2_ID'])
    subm = rrc_evaluation_funcs.load_folder_file(
        submFilePath, evaluationParams['DET_SAMPLE_NAME_2_ID'], True)

    # 初始化全局关注的真实值和检测值数量
    numGlobalCareGt = 0
    numGlobalCareDet = 0

    # 初始化全局置信度列表和匹配列表
    arrGlobalConfidences = []
    arrGlobalMatches = []

    # 计算平均准确率（Average Precision）
    AP = 0
    if evaluationParams['CONFIDENCES']:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    # 计算方法召回率和精确率
    methodRecall = 0 if numGlobalCareGt == 0 else float(
        matchedSum) / numGlobalCareGt
    methodPrecision = 0 if numGlobalCareDet == 0 else float(
        matchedSum) / numGlobalCareDet
    # 计算方法的调和平均值
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * methodRecall * methodPrecision / (
        methodRecall + methodPrecision)

    # 存储方法的指标信息
    methodMetrics = {
        'precision': methodPrecision,
        'recall': methodRecall,
        'hmean': methodHmean,
        'AP': AP
    }
    # 创建一个包含计算结果的字典
    resDict = {
        'calculated': True,  # 表示计算已完成
        'Message': '',  # 存储消息，初始为空
        'method': methodMetrics,  # 存储方法指标数据
        'per_sample': perSampleMetrics  # 存储每个样本的指标数据
    }

    # 返回结果字典
    return resDict
# 计算召回率、精确率和 F1 值
def cal_recall_precison_f1(gt_path, result_path, show_result=False):
    # 构建参数字典，包括真实数据路径和结果数据路径
    p = {'g': gt_path, 's': result_path}
    # 调用 rrc_evaluation_funcs 模块的主要评估函数，传入参数字典、默认评估参数、验证数据函数、评估方法函数和是否显示结果标志
    result = rrc_evaluation_funcs.main_evaluation(p, default_evaluation_params,
                                                  validate_data,
                                                  evaluate_method, show_result)
    # 返回评估结果中的方法值
    return result['method']
```