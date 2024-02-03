# `.\PaddleOCR\ppocr\metrics\det_metric.py`

```
# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 定义可以被导入的模块列表
__all__ = ['DetMetric', 'DetFCEMetric']

# 导入 DetectionIoUEvaluator 类
from .eval_det_iou import DetectionIoUEvaluator

# 定义 DetMetric 类
class DetMetric(object):
    # 初始化函数，设置主要指标和参数
    def __init__(self, main_indicator='hmean', **kwargs):
        # 创建 DetectionIoUEvaluator 实例
        self.evaluator = DetectionIoUEvaluator()
        # 设置主要指标
        self.main_indicator = main_indicator
        # 重置评估器状态
        self.reset()
    # 定义一个方法，用于评估模型预测结果的准确性
    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
       '''
        # 从批次中获取真实多边形和忽略标签
        gt_polyons_batch = batch[2]
        ignore_tags_batch = batch[3]
        # 遍历预测结果、真实多边形和忽略标签
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # 准备真实多边形信息列表
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # 准备检测多边形信息列表
            det_info_list = [{
                'points': det_polyon,
                'text': ''
            } for det_polyon in pred['points']]
            # 评估单张图片的结果
            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            # 将评估结果添加到结果列表中
            self.results.append(result)

    # 获取评估指标
    def get_metric(self):
        """
        return metrics {
                 'precision': 0,
                 'recall': 0,
                 'hmean': 0
            }
        """
        # 组合所有结果并计算指标
        metrics = self.evaluator.combine_results(self.results)
        # 重置结果列表
        self.reset()
        return metrics

    # 重置结果列表
    def reset(self):
        self.results = []  # clear results
class DetFCEMetric(object):
    # 定义一个检测指标类
    def __init__(self, main_indicator='hmean', **kwargs):
        # 初始化函数，设置主要指标和其他参数
        self.evaluator = DetectionIoUEvaluator()
        # 创建一个检测评估器对象
        self.main_indicator = main_indicator
        # 设置主要指标
        self.reset()
        # 重置函数

    def __call__(self, preds, batch, **kwargs):
        '''
       batch: a list produced by dataloaders.
           image: np.ndarray  of shape (N, C, H, W).
           ratio_list: np.ndarray  of shape(N,2)
           polygons: np.ndarray  of shape (N, K, 4, 2), the polygons of objective regions.
           ignore_tags: np.ndarray  of shape (N, K), indicates whether a region is ignorable or not.
       preds: a list of dict produced by post process
            points: np.ndarray of shape (N, K, 4, 2), the polygons of objective regions.
       '''
        # 定义调用函数，接受预测结果和批次数据
        gt_polyons_batch = batch[2]
        # 获取批次数据中的真实多边形
        ignore_tags_batch = batch[3]
        # 获取批次数据中的忽略标签

        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch,
                                                 ignore_tags_batch):
            # 遍历预测结果、真实多边形和忽略标签
            # prepare gt
            gt_info_list = [{
                'points': gt_polyon,
                'text': '',
                'ignore': ignore_tag
            } for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            # 准备真实多边形信息列表

            # prepare det
            det_info_list = [{
                'points': det_polyon,
                'text': '',
                'score': score
            } for det_polyon, score in zip(pred['points'], pred['scores'])]
            # 准备检测多边形信息列表

            for score_thr in self.results.keys():
                # 遍历结果字典中的阈值
                det_info_list_thr = [
                    det_info for det_info in det_info_list
                    if det_info['score'] >= score_thr
                ]
                # 根据阈值筛选检测多边形信息列表
                result = self.evaluator.evaluate_image(gt_info_list,
                                                       det_info_list_thr)
                # 评估图像，计算结果
                self.results[score_thr].append(result)
                # 将结果添加到结果字典中
    # 获取评估指标
    def get_metric(self):
        """
        返回指标字典 {'heman':0,
            'thr 0.3':'precision: 0 recall: 0 hmean: 0',
            'thr 0.4':'precision: 0 recall: 0 hmean: 0',
            'thr 0.5':'precision: 0 recall: 0 hmean: 0',
            'thr 0.6':'precision: 0 recall: 0 hmean: 0',
            'thr 0.7':'precision: 0 recall: 0 hmean: 0',
            'thr 0.8':'precision: 0 recall: 0 hmean: 0',
            'thr 0.9':'precision: 0 recall: 0 hmean: 0',
            }
        """
        # 初始化指标字典
        metrics = {}
        # 初始化 hmean 为 0
        hmean = 0
        # 遍历结果字典中的分数阈值
        for score_thr in self.results.keys():
            # 组合结果并计算指标
            metric = self.evaluator.combine_results(self.results[score_thr])
            # 格式化指标字符串
            metric_str = 'precision:{:.5f} recall:{:.5f} hmean:{:.5f}'.format(
                metric['precision'], metric['recall'], metric['hmean'])
            # 将指标字符串添加到指标字典中
            metrics['thr {}'.format(score_thr)] = metric_str
            # 更新 hmean 为最大值
            hmean = max(hmean, metric['hmean'])
        # 将最大的 hmean 添加到指标字典中
        metrics['hmean'] = hmean

        # 重置结果字典
        self.reset()
        # 返回指标字典
        return metrics

    # 重置结果字典
    def reset(self):
        self.results = {
            0.3: [],
            0.4: [],
            0.5: [],
            0.6: [],
            0.7: [],
            0.8: [],
            0.9: []
        }  # 清空结果
```