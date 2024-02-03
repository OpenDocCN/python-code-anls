# `.\PaddleOCR\ppocr\metrics\table_metric.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
import numpy as np
# 从 ppocr.metrics.det_metric 模块导入 DetMetric 类
from ppocr.metrics.det_metric import DetMetric

# 定义 TableStructureMetric 类
class TableStructureMetric(object):
    # 初始化方法，设置主要指标、eps、是否删除 thead 和 tbody 等参数
    def __init__(self,
                 main_indicator='acc',
                 eps=1e-6,
                 del_thead_tbody=False,
                 **kwargs):
        self.main_indicator = main_indicator
        self.eps = eps
        self.del_thead_tbody = del_thead_tbody
        # 重置指标
        self.reset()
    # 定义一个类方法，用于计算预测标签和真实标签之间的准确率
    def __call__(self, pred_label, batch=None, *args, **kwargs):
        # 解包预测标签和真实标签
        preds, labels = pred_label
        # 获取预测结构批次列表
        pred_structure_batch_list = preds['structure_batch_list']
        # 获取真实结构批次列表
        gt_structure_batch_list = labels['structure_batch_list']
        # 初始化正确预测数量和总数量
        correct_num = 0
        all_num = 0
        # 遍历预测结构批次列表和真实结构批次列表
        for (pred, pred_conf), target in zip(pred_structure_batch_list,
                                             gt_structure_batch_list):
            # 将预测结构和真实结构转换为字符串
            pred_str = ''.join(pred)
            target_str = ''.join(target)
            # 如果需要删除 <thead> 和 <tbody> 标签
            if self.del_thead_tbody:
                pred_str = pred_str.replace('<thead>', '').replace(
                    '</thead>', '').replace('<tbody>', '').replace('</tbody>',
                                                                   '')
                target_str = target_str.replace('<thead>', '').replace(
                    '</thead>', '').replace('<tbody>', '').replace('</tbody>',
                                                                   '')
            # 如果预测结构等于真实结构，则正确预测数量加一
            if pred_str == target_str:
                correct_num += 1
            # 总数量加一
            all_num += 1
        # 更新类属性中的正确预测数量和总数量
        self.correct_num += correct_num
        self.all_num += all_num

    # 定义一个方法，用于计算准确率并返回指标字典
    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        # 计算准确率
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        # 重置类属性
        self.reset()
        # 返回包含准确率的指标字典
        return {'acc': acc}

    # 定义一个方法，用于重置类属性
    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.len_acc_num = 0
        self.token_nums = 0
        self.anys_dict = dict()
class TableMetric(object):
    # 定义一个名为TableMetric的类
    def __init__(self,
                 main_indicator='acc',
                 compute_bbox_metric=False,
                 box_format='xyxy',
                 del_thead_tbody=False,
                 **kwargs):
        """
        初始化函数，设置类的属性和参数

        @param sub_metrics: configs of sub_metric
        @param main_matric: main_matric for save best_model
        @param kwargs:
        """
        # 创建TableStructureMetric对象并传入参数del_thead_tbody
        self.structure_metric = TableStructureMetric(
            del_thead_tbody=del_thead_tbody)
        # 根据compute_bbox_metric参数决定是否创建DetMetric对象
        self.bbox_metric = DetMetric() if compute_bbox_metric else None
        # 设置主指标、框格式等属性
        self.main_indicator = main_indicator
        self.box_format = box_format
        # 调用reset方法
        self.reset()

    def __call__(self, pred_label, batch=None, *args, **kwargs):
        # 调用TableStructureMetric对象的__call__方法
        self.structure_metric(pred_label)
        # 如果bbox_metric不为None，则调用prepare_bbox_metric_input方法
        if self.bbox_metric is not None:
            self.bbox_metric(*self.prepare_bbox_metric_input(pred_label))

    def prepare_bbox_metric_input(self, pred_label):
        # 初始化一些列表
        pred_bbox_batch_list = []
        gt_ignore_tags_batch_list = []
        gt_bbox_batch_list = []
        preds, labels = pred_label

        # 获取batch数量
        batch_num = len(preds['bbox_batch_list'])
        for batch_idx in range(batch_num):
            # 处理预测框
            pred_bbox_list = [
                self.format_box(pred_box)
                for pred_box in preds['bbox_batch_list'][batch_idx]
            ]
            pred_bbox_batch_list.append({'points': pred_bbox_list})

            # 处理真实框
            gt_bbox_list = []
            gt_ignore_tags_list = []
            for gt_box in labels['bbox_batch_list'][batch_idx]:
                gt_bbox_list.append(self.format_box(gt_box))
                gt_ignore_tags_list.append(0)
            gt_bbox_batch_list.append(gt_bbox_list)
            gt_ignore_tags_batch_list.append(gt_ignore_tags_list)

        # 返回处理后的数据
        return [
            pred_bbox_batch_list,
            [0, 0, gt_bbox_batch_list, gt_ignore_tags_batch_list]
        ]
    # 获取当前对象的度量值，包括结构度量和边界框度量
    def get_metric(self):
        # 获取结构度量值
        structure_metric = self.structure_metric.get_metric()
        # 如果边界框度量为空，则返回结构度量值
        if self.bbox_metric is None:
            return structure_metric
        # 获取边界框度量值
        bbox_metric = self.bbox_metric.get_metric()
        # 如果主要指标与边界框度量的主要指标相同
        if self.main_indicator == self.bbox_metric.main_indicator:
            # 输出结果为边界框度量值
            output = bbox_metric
            # 将结构度量值中的每个子键值对添加到输出结果中
            for sub_key in structure_metric:
                output["structure_metric_{}".format(
                    sub_key)] = structure_metric[sub_key]
        else:
            # 输出结果为结构度量值
            output = structure_metric
            # 将边界框度量值中的每个子键值对添加到输出结果中
            for sub_key in bbox_metric:
                output["bbox_metric_{}".format(sub_key)] = bbox_metric[sub_key]
        # 返回最终结果
        return output

    # 重置当前对象的度量值
    def reset(self):
        # 重置结构度量值
        self.structure_metric.reset()
        # 如果边界框度量不为空，则重置边界框度量值
        if self.bbox_metric is not None:
            self.bbox_metric.reset()

    # 格式化边界框的表示形式
    def format_box(self, box):
        # 如果边界框格式为 'xyxy'
        if self.box_format == 'xyxy':
            # 解析边界框坐标
            x1, y1, x2, y2 = box
            # 调整边界框格式为四个点的坐标
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        # 如果边界框格式为 'xywh'
        elif self.box_format == 'xywh':
            # 解析边界框坐标和宽高
            x, y, w, h = box
            # 计算边界框的四个点的坐标
            x1, y1, x2, y2 = x - w // 2, y - h // 2, x + w // 2, y + h // 2
            # 调整边界框格式为四个点的坐标
            box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        # 如果边界框格式为 'xyxyxyxy'
        elif self.box_format == 'xyxyxyxy':
            # 解析边界框四个点的坐标
            x1, y1, x2, y2, x3, y3, x4, y4 = box
            # 调整边界框格式为四个点的坐标
            box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # 返回格式化后的边界框
        return box
```