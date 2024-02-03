# `.\PaddleOCR\ppocr\metrics\ct_metric.py`

```
# 版权声明
# 2020年PaddlePaddle作者保留所有权利。
#
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何明示或暗示的担保或条件。
# 请参阅许可证以获取特定语言的权限和限制。

# 导入必要的库
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from scipy import io
import numpy as np

# 导入自定义的模块
from ppocr.utils.e2e_metric.Deteval import combine_results, get_score_C

# 定义CTMetric类
class CTMetric(object):
    def __init__(self, main_indicator, delimiter='\t', **kwargs):
        # 初始化CTMetric类的属性
        self.delimiter = delimiter
        self.main_indicator = main_indicator
        self.reset()

    # 重置结果
    def reset(self):
        self.results = []  # 清空结果

    # 调用函数
    def __call__(self, preds, batch, **kwargs):
        # 注意：目前仅支持bs=1，因为不同样本的标签长度不相等
        assert len(
            preds) == 1, "CentripetalText test now only suuport batch_size=1."
        label = batch[2]
        text = batch[3]
        pred = preds[0]['points']
        result = get_score_C(label, text, pred)

        self.results.append(result)

    # 获取评估指标
    def get_metric(self):
        """
        输入格式：y0,x0, ..... yn,xn。每个检测结果由换行符('\n')分隔
        """
        metrics = combine_results(self.results, rec_flag=False)
        self.reset()
        return metrics
```