# `.\PaddleOCR\tools\end2end\eval_end2end.py`

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
import editdistance

# 定义一个函数，将字符串中的全角字符转换为半角字符
def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

# 定义一个函数，根据给定的点坐标创建一个 shapely 多边形对象
def polygon_from_str(polygon_points):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(polygon_points).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon

# 定义一个函数，计算两个 shapely 多边形对象的交并比
def polygon_iou(poly1, poly2):
    """
    Intersection over union between two shapely polygons.
    """
    if not poly1.intersects(
            poly2):  # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.area + poly2.area - inter_area
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            # except Exception as e:
            #     print(e)
            print('shapely.geos.TopologicalError occurred, iou set to 0')
            iou = 0
    return iou

# 定义一个函数，计算两个字符串之间的编辑距离
def ed(str1, str2):
    # 返回两个字符串之间的编辑距离
    return editdistance.eval(str1, str2)
# 计算端到端评估指标，包括字符准确率、平均编辑距离等
def e2e_eval(gt_dir, res_dir, ignore_blank=False):
    # 打印开始测试信息
    print('start testing...')
    # 设置 IOU 阈值
    iou_thresh = 0.5
    # 获取真实标签文件夹下的文件名列表
    val_names = os.listdir(gt_dir)
    # 初始化变量
    num_gt_chars = 0
    gt_count = 0
    dt_count = 0
    hit = 0
    ed_sum = 0

    # 设置一个极小值
    eps = 1e-9
    # 打印命中数、检测数、真实标签数
    print('hit, dt_count, gt_count', hit, dt_count, gt_count)
    # 计算精确率
    precision = hit / (dt_count + eps)
    # 计算召回率
    recall = hit / (gt_count + eps)
    # 计算 F1 分数
    fmeasure = 2.0 * precision * recall / (precision + recall + eps)
    # 计算平均编辑距离（按图片计算）
    avg_edit_dist_img = ed_sum / len(val_names)
    # 计算平均编辑距离（按字段计算）
    avg_edit_dist_field = ed_sum / (gt_count + eps)
    # 计算字符准确率
    character_acc = 1 - ed_sum / (num_gt_chars + eps)

    # 打印字符准确率
    print('character_acc: %.2f' % (character_acc * 100) + "%")
    # 打印平均编辑距离（按字段计算）
    print('avg_edit_dist_field: %.2f' % (avg_edit_dist_field))
    # 打印平均编辑距离（按图片计算）
    print('avg_edit_dist_img: %.2f' % (avg_edit_dist_img))
    # 打印精确率
    print('precision: %.2f' % (precision * 100) + "%")
    # 打印召回率
    print('recall: %.2f' % (recall * 100) + "%")
    # 打印 F1 分数
    print('fmeasure: %.2f' % (fmeasure * 100) + "%")


if __name__ == '__main__':
    # 读取命令行参数，获取真实标签文件夹和预测结果文件夹
    # if len(sys.argv) != 3:
    #     print("python3 ocr_e2e_eval.py gt_dir res_dir")
    #     exit(-1)
    # gt_folder = sys.argv[1]
    # pred_folder = sys.argv[2]
    gt_folder = sys.argv[1]
    pred_folder = sys.argv[2]
    # 调用 e2e_eval 函数进行评估
    e2e_eval(gt_folder, pred_folder)
```