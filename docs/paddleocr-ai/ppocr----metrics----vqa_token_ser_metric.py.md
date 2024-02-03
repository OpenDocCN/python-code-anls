# `.\PaddleOCR\ppocr\metrics\vqa_token_ser_metric.py`

```
# 版权声明和许可证信息
# 本代码版权归 PaddlePaddle 作者所有，保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import paddle

# 定义模块可导出的内容
__all__ = ['KIEMetric']

# 定义 VQASerTokenMetric 类
class VQASerTokenMetric(object):
    # 初始化函数
    def __init__(self, main_indicator='hmean', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    # 调用函数
    def __call__(self, preds, batch, **kwargs):
        preds, labels = preds
        self.pred_list.extend(preds)
        self.gt_list.extend(labels)

    # 获取指标函数
    def get_metric(self):
        from seqeval.metrics import f1_score, precision_score, recall_score
        metrics = {
            "precision": precision_score(self.gt_list, self.pred_list),
            "recall": recall_score(self.gt_list, self.pred_list),
            "hmean": f1_score(self.gt_list, self.pred_list),
        }
        self.reset()
        return metrics

    # 重置函数
    def reset(self):
        self.pred_list = []
        self.gt_list = []
```