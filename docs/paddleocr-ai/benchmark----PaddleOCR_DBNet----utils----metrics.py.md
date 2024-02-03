# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\metrics.py`

```
# 导入 numpy 库
import numpy as np

# 定义一个 runningScore 类
class runningScore(object):
    # 初始化方法，传入类别数目
    def __init__(self, n_classes):
        # 设置类别数目
        self.n_classes = n_classes
        # 初始化混淆矩阵为全零矩阵
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    # 计算混淆矩阵的方法
    def _fast_hist(self, label_true, label_pred, n_class):
        # 创建一个掩码，用于过滤无效的标签
        mask = (label_true >= 0) & (label_true < n_class)

        # 如果预测标签中存在小于 0 的值，则打印出来
        if np.sum((label_pred[mask] < 0)) > 0:
            print(label_pred[label_pred < 0])
        
        # 使用 numpy 的 bincount 方法计算直方图
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class**2).reshape(n_class, n_class)
        return hist

    # 更新混淆矩阵的方法
    def update(self, label_trues, label_preds):
        # 遍历真实标签和预测标签
        for lt, lp in zip(label_trues, label_preds):
            try:
                # 更新混淆矩阵
                self.confusion_matrix += self._fast_hist(lt.flatten(),
                                                         lp.flatten(),
                                                         self.n_classes)
            except:
                pass

    # 获取评估结果的方法
    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        # 获取混淆矩阵
        hist = self.confusion_matrix
        # 计算整体准确率
        acc = np.diag(hist).sum() / (hist.sum() + 0.0001)
        # 计算平均准确率
        acc_cls = np.diag(hist) / (hist.sum(axis=1) + 0.0001)
        acc_cls = np.nanmean(acc_cls)
        # 计算平均 IU
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 0.0001)
        mean_iu = np.nanmean(iu)
        # 计算频权准确率
        freq = hist.sum(axis=1) / (hist.sum() + 0.0001)
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        # 生成类别 IU 字典
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
            'Overall Acc': acc,
            'Mean Acc': acc_cls,
            'FreqW Acc': fwavacc,
            'Mean IoU': mean_iu,
        }, cls_iu
    # 重置混淆矩阵为全零矩阵，大小为类别数 x 类别数
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
```