# `.\PaddleOCR\ppocr\metrics\rec_metric.py`

```py
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

# 导入 Levenshtein 和 SequenceMatcher 模块
from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

# 导入 numpy 和 string 模块
import numpy as np
import string

# 定义 RecMetric 类
class RecMetric(object):
    # 初始化函数，设置主要指标、是否过滤、是否忽略空格等参数
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        # 重置函数
        self.reset()

    # 文本规范化函数，去除非数字和字母字符，并转换为小写
    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()
    # 定义一个方法，用于计算预测标签的准确率和编辑距离
    def __call__(self, pred_label, *args, **kwargs):
        # 将预测值和真实标签分开
        preds, labels = pred_label
        # 初始化正确预测数量、总数量和标准化编辑距离
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        # 遍历预测值和真实标签
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            # 如果需要忽略空格，则去除预测值和真实标签中的空格
            if self.ignore_space:
                pred = pred.replace(" ", "")
                target = target.replace(" ", "")
            # 如果需要过滤文本，则对预测值和真实标签进行规范化处理
            if self.is_filter:
                pred = self._normalize_text(pred)
                target = self._normalize_text(target)
            # 计算标准化编辑距离
            norm_edit_dis += Levenshtein.normalized_distance(pred, target)
            # 如果预测值等于真实标签，则正确预测数量加一
            if pred == target:
                correct_num += 1
            # 总数量加一
            all_num += 1
        # 更新正确预测数量、总数量和标准化编辑距离
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        # 返回准确率和标准化编辑距离
        return {
            'acc': correct_num / (all_num + self.eps),
            'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
        }

    # 获取评估指标
    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        # 计算准确率和标准化编辑距离
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        # 重置统计数据
        self.reset()
        # 返回准确率和标准化编辑距离
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    # 重置统计数据
    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0
class CNTMetric(object):
    # 初始化函数，设置主要指标和参数
    def __init__(self, main_indicator='acc', **kwargs):
        # 设置主要指标
        self.main_indicator = main_indicator
        # 设置一个很小的数值
        self.eps = 1e-5
        # 调用reset函数
        self.reset()

    # 调用对象时执行的函数
    def __call__(self, pred_label, *args, **kwargs):
        # 解包预测标签
        preds, labels = pred_label
        # 初始化正确数量和总数量
        correct_num = 0
        all_num = 0
        # 遍历预测和标签，计算正确数量和总数量
        for pred, target in zip(preds, labels):
            if pred == target:
                correct_num += 1
            all_num += 1
        # 更新正确数量和总数量
        self.correct_num += correct_num
        self.all_num += all_num
        # 返回准确率
        return {'acc': correct_num / (all_num + self.eps), }

    # 获取指标函数
    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
            }
        """
        # 计算准确率
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        # 重置正确数量和总数量
        self.reset()
        # 返回准确率
        return {'acc': acc}

    # 重置函数
    def reset(self):
        # 重置正确数量和总数量
        self.correct_num = 0
        self.all_num = 0


class CANMetric(object):
    # 初始化函数，设置主要指标和参数
    def __init__(self, main_indicator='exp_rate', **kwargs):
        # 设置主要指标
        self.main_indicator = main_indicator
        # 初始化单词正确列表和表达式正确列表
        self.word_right = []
        self.exp_right = []
        # 初始化单词总长度和表达式总数量
        self.word_total_length = 0
        self.exp_total_num = 0
        # 初始化单词正确率和表达式正确率
        self.word_rate = 0
        self.exp_rate = 0
        # 调用reset函数和epoch_reset函数
        self.reset()
        self.epoch_reset()
    # 定义一个方法，用于计算模型预测结果的指标
    def __call__(self, preds, batch, **kwargs):
        # 遍历关键字参数
        for k, v in kwargs.items():
            # 获取是否需要重置 epoch 的标志
            epoch_reset = v
            # 如果需要重置 epoch，则调用 epoch_reset 方法
            if epoch_reset:
                self.epoch_reset()
        # 获取模型预测结果
        word_probs = preds
        # 获取输入数据的标签和掩码
        word_label, word_label_mask = batch
        # 初始化正确预测的行数
        line_right = 0
        # 如果模型预测结果不为空
        if word_probs is not None:
            # 获取预测的词索引
            word_pred = word_probs.argmax(2)
        # 将预测结果转移到 CPU 并转换为 numpy 数组
        word_pred = word_pred.cpu().detach().numpy()
        # 计算每个样本的相似度得分
        word_scores = [
            SequenceMatcher(
                None,
                s1[:int(np.sum(s3))],
                s2[:int(np.sum(s3))],
                autojunk=False).ratio() * (
                    len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) /
            len(s1[:int(np.sum(s3))]) / 2
            for s1, s2, s3 in zip(word_label, word_pred, word_label_mask)
        ]
        # 获取批次大小
        batch_size = len(word_scores)
        # 统计正确预测的行数
        for i in range(batch_size):
            if word_scores[i] == 1:
                line_right += 1
        # 计算词级别和实体级别的正确率
        self.word_rate = np.mean(word_scores)  #float
        self.exp_rate = line_right / batch_size  #float
        # 获取标签的长度
        exp_length, word_length = word_label.shape[:2]
        # 更新词级别和实体级别的正确预测数
        self.word_right.append(self.word_rate * word_length)
        self.exp_right.append(self.exp_rate * exp_length)
        # 更新总词长度和总实体数
        self.word_total_length = self.word_total_length + word_length
        self.exp_total_num = self.exp_total_num + exp_length

    # 获取模型指标
    def get_metric(self):
        """
        return {
            'word_rate': 0,
            "exp_rate": 0,
        }
        """
        # 计算当前词级别和实体级别的正确率
        cur_word_rate = sum(self.word_right) / self.word_total_length
        cur_exp_rate = sum(self.exp_right) / self.exp_total_num
        # 重置指标
        self.reset()
        # 返回当前指标
        return {'word_rate': cur_word_rate, "exp_rate": cur_exp_rate}

    # 重置指标
    def reset(self):
        self.word_rate = 0
        self.exp_rate = 0

    # 重置 epoch
    def epoch_reset(self):
        self.word_right = []
        self.exp_right = []
        self.word_total_length = 0
        self.exp_total_num = 0
```