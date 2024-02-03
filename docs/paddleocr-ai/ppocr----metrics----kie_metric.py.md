# `.\PaddleOCR\ppocr\metrics\kie_metric.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权使用本文件
# 除非符合许可证的规定，否则不得使用本文件
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件，无论是明示还是暗示的
# 请查看许可证以获取有关特定语言的权限和限制
# 代码参考自: https://github.com/open-mmlab/mmocr/blob/main/mmocr/core/evaluation/kie_metric.py

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle

# 定义模块的公开接口
__all__ = ['KIEMetric']

# 定义 KIEMetric 类
class KIEMetric(object):
    # 初始化函数，设置主要指标和其他参数
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        # 重置评估指标
        self.reset()
        # 初始化节点和标签列表
        self.node = []
        self.gt = []

    # 定义调用函数，用于计算评估指标
    def __call__(self, preds, batch, **kwargs):
        # 获取预测结果中的节点和标签
        nodes, _ = preds
        gts, tag = batch[4].squeeze(0), batch[5].tolist()[0]
        # 根据标签截取真实标签数据
        gts = gts[:tag[0], :1].reshape([-1])
        # 将节点和标签数据添加到列表中
        self.node.append(nodes.numpy())
        self.gt.append(gts)
        # 计算 F1 分数
        # result = self.compute_f1_score(nodes, gts)
        # self.results.append(result)
    # 计算 F1 分数
    def compute_f1_score(self, preds, gts):
        # 定义需要忽略的类别
        ignores = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]
        # 获取预测结果的类别数
        C = preds.shape[1]
        # 生成不包含忽略类别的类别数组
        classes = np.array(sorted(set(range(C)) - set(ignores)))
        # 计算混淆矩阵
        hist = np.bincount(
            (gts * C).astype('int64') + preds.argmax(1), minlength=C**2).reshape([C, C]).astype('float32')
        # 获取混淆矩阵对角线元素
        diag = np.diag(hist)
        # 计算召回率
        recalls = diag / hist.sum(1).clip(min=1)
        # 计算精确率
        precisions = diag / hist.sum(0).clip(min=1)
        # 计算 F1 分数
        f1 = 2 * recalls * precisions / (recalls + precisions).clip(min=1e-8)
        # 返回不包含忽略类别的 F1 分数
        return f1[classes]

    # 合并结果
    def combine_results(self, results):
        # 将节点结果和真实标签结果连接起来
        node = np.concatenate(self.node, 0)
        gts = np.concatenate(self.gt, 0)
        # 计算 F1 分数
        results = self.compute_f1_score(node, gts)
        # 计算结果的均值
        data = {'hmean': results.mean()}
        return data

    # 获取评估指标
    def get_metric(self):
        # 合并结果并计算评估指标
        metrics = self.combine_results(self.results)
        # 重置结果
        self.reset()
        return metrics

    # 重置结果
    def reset(self):
        # 清空结果
        self.results = []  # clear results
        self.node = []
        self.gt = []
```