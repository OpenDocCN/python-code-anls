# `.\PaddleOCR\ppocr\metrics\cls_metric.py`

```
# 定义一个类 ClsMetric，用于计算分类任务的指标
class ClsMetric(object):
    # 初始化函数，设置主要指标为准确率，初始化 eps 为 1e-5，并调用 reset 函数
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.eps = 1e-5
        self.reset()

    # 定义 __call__ 函数，用于计算准确率
    def __call__(self, pred_label, *args, **kwargs):
        # 解包预测值和标签值
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        # 遍历预测值和标签值，计算正确预测数量和总数量
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        # 更新正确预测数量和总数量
        self.correct_num += correct_num
        self.all_num += all_num
        # 返回准确率作为字典
        return {'acc': correct_num / (all_num + self.eps), }

    # 定义 get_metric 函数，用于获取指标
    def get_metric(self):
        """
        return metrics {
                 'acc': 0
            }
        """
        # 计算准确率
        acc = self.correct_num / (self.all_num + self.eps)
        # 调用 reset 函数重置计数器
        self.reset()
        # 返回准确率作为字典
        return {'acc': acc}

    # 定义 reset 函数，用于重置计数器
    def reset(self):
        self.correct_num = 0
        self.all_num = 0
```