# `.\PaddleOCR\ppocr\metrics\distillation_metric.py`

```
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

# 导入模块
import importlib
# 复制模块
import copy

# 导入各种评估指标类
from .rec_metric import RecMetric
from .det_metric import DetMetric
from .e2e_metric import E2EMetric
from .cls_metric import ClsMetric
from .vqa_token_ser_metric import VQASerTokenMetric
from .vqa_token_re_metric import VQAReTokenMetric

# 定义蒸馏评估指标类
class DistillationMetric(object):
    def __init__(self,
                 key=None,
                 base_metric_name=None,
                 main_indicator=None,
                 **kwargs):
        # 初始化主要指标
        self.main_indicator = main_indicator
        # 初始化关键字
        self.key = key
        # 初始化主要指标
        self.main_indicator = main_indicator
        # 初始化基础评估指标名称
        self.base_metric_name = base_metric_name
        # 初始化其他参数
        self.kwargs = kwargs
        # 初始化评估指标为空
        self.metrics = None

    # 初始化评估指标
    def _init_metrcis(self, preds):
        self.metrics = dict()
        # 导入当前模块
        mod = importlib.import_module(__name__)
        # 遍历预测结果
        for key in preds:
            # 根据基础评估指标名称创建评估指标对象
            self.metrics[key] = getattr(mod, self.base_metric_name)(
                main_indicator=self.main_indicator, **self.kwargs)
            # 重置评估指标
            self.metrics[key].reset()

    # 调用评估指标
    def __call__(self, preds, batch, **kwargs):
        assert isinstance(preds, dict)
        # 如果评估指标为空，则初始化评估指标
        if self.metrics is None:
            self._init_metrcis(preds)
        output = dict()
        # 遍历预测结果
        for key in preds:
            # 调用评估指标对象的__call__方法
            self.metrics[key].__call__(preds[key], batch, **kwargs)
    # 获取模型评估指标的数值，并返回一个包含各指标数值的字典
    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        # 初始化一个空字典用于存储指标数值
        output = dict()
        # 遍历所有指标
        for key in self.metrics:
            # 获取每个指标的数值
            metric = self.metrics[key].get_metric()
            # 如果是主要指标，则更新输出字典
            if key == self.key:
                output.update(metric)
            # 如果不是主要指标，则将指标数值添加到输出字典中
            else:
                for sub_key in metric:
                    output["{}_{}".format(key, sub_key)] = metric[sub_key]
        # 返回包含所有指标数值的字典
        return output

    # 重置所有指标的数值
    def reset(self):
        # 遍历所有指标，调用 reset 方法将其数值重置为初始值
        for key in self.metrics:
            self.metrics[key].reset()
```