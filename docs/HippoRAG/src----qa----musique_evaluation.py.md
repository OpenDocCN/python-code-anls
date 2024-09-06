# `.\HippoRAG\src\qa\musique_evaluation.py`

```py
# 导入正则表达式模块
import re
# 导入字符串常量
import string
# 导入集合操作模块
import collections
# 导入类型提示
from typing import Tuple, List, Any, Dict


class Metric:
    """
    代表一个可以被累计的度量的抽象类。
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        # 抽象方法：根据预测和真实标签计算度量
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        计算并返回度量。可以选择调用 `self.reset`。
        """
        # 抽象方法：计算度量并返回结果
        raise NotImplementedError

    def reset(self) -> None:
        """
        重置任何累加器或内部状态。
        """
        # 抽象方法：重置状态
        raise NotImplementedError


def normalize_answer(s):
    """将文本转为小写，移除标点符号、冠词和多余空白。"""

    def remove_articles(text):
        # 使用正则表达式移除冠词
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        # 去除多余的空白
        return " ".join(text.split())

    def remove_punc(text):
        # 创建一个包含所有标点符号的集合
        exclude = set(string.punctuation)
        # 移除标点符号
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 将文本转为小写
        return text.lower()

    # 处理文本，移除标点符号和冠词，去除多余空白
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    # 如果字符串为空，返回空列表
    if not s:
        return []
    # 返回处理后的文本分词列表
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    # 比较处理后的黄金标准答案和预测答案是否相同，返回 1 或 0
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    # 计算黄金标准答案和预测答案的 F1 分数
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    # 计算两个列表中的公共词汇
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    # 如果任一答案为空，则如果它们相等，F1 为 1，否则为 0
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    # 如果没有公共词汇，F1 为 0
    if num_same == 0:
        return 0
    # 计算精确度和召回率
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    # 计算 F1 分数
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    # 针对每个真实答案计算度量，并返回最大值
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AnswerMetric(Metric):
    def __init__(self) -> None:
        # 初始化计数器
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(
            self,
            predicted_answer: str,
            ground_truth_answers: List[str],
    ):
        # 计算精确度和 F1 分数
        exact_scores = metric_max_over_ground_truths(
            compute_exact, predicted_answer, ground_truth_answers
        )
        f1_scores = metric_max_over_ground_truths(
            compute_f1, predicted_answer, ground_truth_answers
        )

        # 更新累计精确度和 F1 分数
        self._total_em += int(exact_scores)
        self._total_f1 += f1_scores
        self._count += 1
    # 定义获取度量指标的方法，支持重置
    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        # 计算精确匹配率，除数为 0 时返回 0
        exact_match = self._total_em / self._count if self._count > 0 else 0
        # 计算 F1 分数，除数为 0 时返回 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        # 如果 reset 为 True，则调用重置方法
        if reset:
            self.reset()
        # 返回精确匹配率和 F1 分数
        return exact_match, f1_score
    
    # 定义重置方法，重置所有统计数据
    def reset(self):
        # 将精确匹配总数重置为 0
        self._total_em = 0.0
        # 将 F1 总分数重置为 0
        self._total_f1 = 0.0
        # 将计数器重置为 0
        self._count = 0
# 定义函数以评估 MuSiQue 数据集中的一个样本
def evaluate(prediction: dict, gold: dict):
    """
    Evaluate one sample for MuSiQue dataset.
    :param prediction: the predicted answer
    :param gold: the ground truth answers from the original dataset
    :return: metrics including EM and F1
    """
    # 从预测结果中提取预测的答案
    pred = prediction["predicted_answer"]  # str
    # 从黄金标准中提取正确答案及其别名，组合成答案列表
    gold_answers = [gold["answer"]] + gold["answer_aliases"]  # List[str]

    # 计算精确度分数，找到与所有黄金标准答案匹配的最大分数
    exact_scores = metric_max_over_ground_truths(
        compute_exact, pred, gold_answers
    )
    # 计算 F1 分数，找到与所有黄金标准答案匹配的最大分数
    f1_scores = metric_max_over_ground_truths(
        compute_f1, pred, gold_answers
    )

    # 将精确度分数转换为整数
    em = int(exact_scores)
    # 返回精确度和 F1 分数
    return em, f1_scores
```