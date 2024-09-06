# `.\HippoRAG\src\qa\hotpotqa_evaluation.py`

```py
# 导入正则表达式库
import re
# 导入字符串处理库
import string
# 导入系统相关库
import sys
# 从 collections 模块导入 Counter 类
from collections import Counter

# 导入 ujson 库作为 json
import ujson as json


# 定义规范化答案的函数
def normalize_answer(s):
    # 定义去除文章的内部函数
    def remove_articles(text):
        # 使用正则表达式替换 'a', 'an', 'the' 为一个空格
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    # 定义修复空白的内部函数
    def white_space_fix(text):
        # 将多个空白字符合并为一个空格
        return ' '.join(text.split())

    # 定义去除标点符号的内部函数
    def remove_punc(text):
        # 创建标点符号集合
        exclude = set(string.punctuation)
        # 过滤掉标点符号
        return ''.join(ch for ch in text if ch not in exclude)

    # 定义将文本转换为小写的内部函数
    def lower(text):
        # 将文本转换为小写
        return text.lower()

    # 返回经过文章去除、标点去除、小写转换和空白修复的文本
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 定义 F1 分数计算函数
def f1_score(prediction, ground_truth):
    # 规范化预测和实际答案
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # 定义 F1 分数为零的常量
    ZERO_METRIC = (0, 0, 0)

    # 如果预测或实际答案是特定值且不匹配，则返回零分数
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 将规范化的预测和实际答案分割为词语列表
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    # 计算预测和实际答案的共同词语
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    # 计算共同词语的数量
    num_same = sum(common.values())
    # 如果没有共同词语，则返回零分数
    if num_same == 0:
        return ZERO_METRIC
    # 计算精确度
    precision = 1.0 * num_same / len(prediction_tokens)
    # 计算召回率
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # 计算 F1 分数
    f1 = (2 * precision * recall) / (precision + recall)
    # 返回 F1 分数、精确度和召回率
    return f1, precision, recall


# 定义精确匹配分数计算函数
def exact_match_score(prediction, ground_truth):
    # 如果预测和实际答案规范化后相等，则返回 1，否则返回 0
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


# 定义更新答案评估指标的函数
def update_answer(metrics, prediction, gold):
    # 计算精确匹配分数
    em = exact_match_score(prediction, gold)
    # 计算 F1 分数、精确度和召回率
    f1, precision, recall = f1_score(prediction, gold)
    # 更新指标字典中的值
    metrics['em'] += float(em)
    metrics['f1'] += f1
    metrics['precision'] += precision
    metrics['recall'] += recall
    # 返回更新后的值
    return em, f1, precision, recall


# 定义更新分段精确匹配指标的函数
def update_sp(metrics, prediction, gold):
    # 将预测和实际答案转换为集合形式
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    # 计算真正例和假正例
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    # 计算假负例
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    # 计算精确度
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    # 计算召回率
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    # 计算 F1 分数
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    # 如果没有假正例和假负例，则精确匹配分数为 1，否则为 0
    em = 1.0 if fp + fn == 0 else 0.0
    # 更新指标字典中的值
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    # 返回更新后的值
    return em, prec, recall


# 定义评估函数
def eval(prediction_file, gold_file):
    # 打开并加载预测文件
    with open(prediction_file) as f:
        prediction = json.load(f)
    # 打开并加载实际答案文件
    with open(gold_file) as f:
        gold = json.load(f)
    # 初始化指标字典，所有指标初始值为 0
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    # 遍历每个数据点
    for dp in gold:
        # 获取当前数据点的 ID
        cur_id = dp['_id']
        # 初始化是否可以评估联合指标的标志
        can_eval_joint = True
        # 如果预测中没有当前 ID 的答案
        if cur_id not in prediction['answer']:
            # 输出缺失答案的警告
            print('missing answer {}'.format(cur_id))
            # 设置无法评估联合指标
            can_eval_joint = False
        else:
            # 更新答案指标
            em, f1, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], dp['answer'])
        # 如果预测中没有当前 ID 的支持事实
        if cur_id not in prediction['sp']:
            # 输出缺失支持事实的警告
            print('missing sp fact {}'.format(cur_id))
            # 设置无法评估联合指标
            can_eval_joint = False
        else:
            # 更新支持事实指标
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        # 如果可以评估联合指标
        if can_eval_joint:
            # 计算联合精确度和召回率
            joint_prec = prec * sp_prec
            joint_recall = recall * sp_recall
            # 计算联合 F1 分数
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            # 计算联合答案准确率
            joint_em = em * sp_em

            # 累加联合指标
            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    # 计算每个指标的平均值
    N = len(gold)
    for k in metrics.keys():
        metrics[k] /= N

    # 打印最终的指标结果
    print(metrics)
# 如果当前脚本是主程序，则执行以下代码
if __name__ == '__main__':
    # 解析并执行命令行参数中的表达式
    eval(sys.argv[1], sys.argv[2])
```