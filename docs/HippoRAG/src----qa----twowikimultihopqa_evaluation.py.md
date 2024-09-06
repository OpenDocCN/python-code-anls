# `.\HippoRAG\src\qa\twowikimultihopqa_evaluation.py`

```py
"""
2Wiki-Multihop QA evaluation script
Adapted from HotpotQA evaluation at https://github.com/hotpotqa/hotpot
"""
# 导入 Python 标准库和第三方库
import sys
import ujson as json  # 使用 ujson 库来处理 JSON 数据
import re  # 正则表达式库
import string  # 字符串常量
import itertools  # 迭代器工具
from collections import Counter  # 用于计数的字典
import pickle  # 用于序列化和反序列化对象
import os  # 与操作系统交互的功能

# 函数：标准化答案文本
def normalize_answer(s):
    # 函数：移除文本中的冠词
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    # 函数：修复空格，将多余空格合并为一个
    def white_space_fix(text):
        return ' '.join(text.split())

    # 函数：移除文本中的标点符号
    def remove_punc(text):
        exclude = set(string.punctuation)  # 获取所有标点符号
        return ''.join(ch for ch in text if ch not in exclude)

    # 函数：将文本转换为小写
    def lower(text):
        return text.lower()

    # 顺序调用上述函数来标准化答案
    return white_space_fix(remove_articles(remove_punc(lower(s))))

# 函数：计算 F1 分数
def f1_score(prediction: str, ground_truth: str):
    # 标准化预测和真实答案
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # 定义零分指标
    ZERO_METRIC = (0, 0, 0)

    # 对于特定预测值，若不匹配则返回零分
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    # 将预测和真实答案分词
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    # 计算预测和真实答案的词汇交集
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())  # 计算交集中的词汇数量
    # 如果没有共同词汇，返回零分
    if num_same == 0:
        return ZERO_METRIC
    # 计算精确率、召回率和 F1 分数
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

# 函数：计算准确匹配分数
def exact_match_score(prediction, ground_truth):
    # 比较标准化后的预测和真实答案是否相同
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

# 函数：评估答案的准确匹配分数和 F1 分数
def eval_answer(prediction, gold):
    em = exact_match_score(prediction, gold)
    f1, prec, recall = f1_score(prediction, gold)
    return em, f1, prec, recall

# 函数：更新答案的评估指标
def update_answer(metrics, prediction, golds):
    max_em, max_f1, max_prec, max_recall = 0, 0, 0, 0

    # 对每个黄金标准答案计算指标，并更新最大值
    for gold in golds:
        em, f1, prec, recall = eval_answer(prediction, gold)

        max_em = max(max_em, em)
        max_f1 = max(max_f1, f1)
        max_prec = max(max_prec, prec)
        max_recall = max(max_recall, recall)

    # 更新最终的指标值
    metrics['em'] += float(max_em)
    metrics['f1'] += max_f1
    metrics['prec'] += max_prec
    metrics['recall'] += max_recall

    return max_em, max_prec, max_recall

# 函数：标准化特殊答案
def normalize_sp(sps):
    new_sps = []
    for sp in sps:
        sp = list(sp)  # 将每个答案转换为列表
        sp[0] = sp[0].lower()  # 将第一个元素转换为小写
        new_sps.append(sp)
    return new_sps

# 函数：更新特殊答案的评估指标
def update_sp(metrics, prediction, gold):
    # 标准化预测和黄金标准答案
    cur_sp_pred = normalize_sp(set(map(tuple, prediction)))
    gold_sp_pred = normalize_sp(set(map(tuple, gold)))
    tp, fp, fn = 0, 0, 0
    # 计算真正例、假正例和假负例
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    # 计算精确度，若 tp + fp 大于 0，则计算公式为 tp / (tp + fp)，否则为 0
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    # 计算召回率，若 tp + fn 大于 0，则计算公式为 tp / (tp + fn)，否则为 0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    # 计算 F1 分数，若 prec + recall 大于 0，则计算公式为 2 * prec * recall / (prec + recall)，否则为 0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    # 如果 fp + fn 等于 0，则设置 em 为 1.0，否则为 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    # 累加计算得到的 em 值到 metrics 字典中的 'sp_em' 键
    metrics['sp_em'] += em
    # 累加计算得到的 f1 值到 metrics 字典中的 'sp_f1' 键
    metrics['sp_f1'] += f1
    # 累加计算得到的 prec 值到 metrics 字典中的 'sp_prec' 键
    metrics['sp_prec'] += prec
    # 累加计算得到的 recall 值到 metrics 字典中的 'sp_recall' 键
    metrics['sp_recall'] += recall
    # 返回 em, prec, recall 的值
    return em, prec, recall
# 对证据进行标准化处理
def normalize_evi(evidences):
    # 去除文本中的多余空格
    def white_space_fix(text):
        return ' '.join(text.split())

    # 去除文本中的标点符号
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # 将文本转换为小写
    def lower(text):
        return text.lower()

    # 递归处理列表中的每个元素
    def recurse(arr):
        for i in range(len(arr)):
            if isinstance(arr[i], str):
                # 对字符串进行去空格、去标点和转小写处理
                arr[i] = white_space_fix(remove_punc(lower(arr[i])))
            else:
                # 如果是列表，则递归处理
                recurse(arr[i])

    # 处理输入的证据
    recurse(evidences)

    # 返回处理后的证据
    return evidences


# 更新指标
def update_evi(metrics, prediction, gold):
    # 标准化预测和黄金证据
    prediction_normalize = normalize_evi(prediction)
    gold_normalize = normalize_evi(gold)
    #
    # 将标准化的预测证据转换为集合形式
    cur_evi_pred = set(map(tuple, prediction_normalize))
    # 将标准化的黄金证据转换为集合形式的列表
    gold_evi_pred = list(map(lambda e: set(map(tuple, e)), gold_normalize))
    #
    # 初始化匹配数量和预测及黄金证据的数量
    num_matches = 0
    num_preds = len(cur_evi_pred)
    num_golds = len(gold_evi_pred)

    # 计算匹配的数量
    for pred_evidence in cur_evi_pred:
        for gold_evidences in gold_evi_pred:
            if pred_evidence in gold_evidences:
                num_matches += 1
                break

    # 计算精确度、召回率和 F1 分数
    prec = num_preds and num_matches / num_preds
    recall = num_golds and num_matches / num_golds
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    # 计算完全匹配度
    em = 1.0 if num_matches == num_preds == num_golds else 0.0

    # 更新指标
    metrics['evi_em'] += em
    metrics['evi_f1'] += f1
    metrics['evi_prec'] += prec
    metrics['evi_recall'] += recall

    # 返回评估结果
    return em, prec, recall


# 评估预测结果
def eval(prediction_file, gold_file, alias_file):
    aliases = {}

    # 读取预测文件
    with open(prediction_file) as f:
        prediction = json.load(f)
    # 读取黄金标准文件
    with open(gold_file) as f:
        gold = json.load(f)
    # 读取别名文件，并构建别名字典
    with open(alias_file) as f:
        for json_line in map(json.loads, f):
            aliases[json_line["Q_id"]] = {
                "aliases": set(json_line["aliases"] + json_line["demonyms"])
            }

    # 初始化指标字典
    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
               'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
               'evi_em': 0, 'evi_f1': 0, 'evi_prec': 0, 'evi_recall': 0,
               'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0}
    # 遍历所有黄金标准数据项
    for dp in gold:
        # 获取当前数据项的 ID
        cur_id = dp['_id']
        # 初始化标志，表明是否可以进行联合评估
        can_eval_joint = True
        # 答案预测任务
        if cur_id not in prediction['answer']:
            # 打印缺失的答案信息
            print('missing answer {}'.format(cur_id))
            # 设置标志，表示无法进行联合评估
            can_eval_joint = False
        else:
            # 形成一个包含黄金答案的集合
            gold_answers = {dp['answer']}  # Gold span

            # 如果存在别名，则将其添加到黄金答案集合中
            if dp['answer_id'] in aliases and aliases[dp['answer_id']]["aliases"]:
                gold_answers.update(aliases[dp['answer_id']]["aliases"])

            # 更新答案的评价指标
            em, prec, recall = update_answer(
                metrics, prediction['answer'][cur_id], gold_answers)
        # 句子级别的支持事实预测任务
        if cur_id not in prediction['sp']:
            # 打印缺失的支持事实信息
            print('missing sp fact {}'.format(cur_id))
            # 设置标志，表示无法进行联合评估
            can_eval_joint = False
        else:
            # 更新支持事实的评价指标
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])
        # 证据生成任务
        if cur_id not in prediction['evidence']:
            # 打印缺失的证据信息
            print('missing evidence {}'.format(cur_id))
            # 设置标志，表示无法进行联合评估
            can_eval_joint = False
        else:
            # 初始化黄金证据列表
            gold_evidences = []

            # 遍历所有证据
            for evidence_idx, (sub_str, rel_str, obj_str) in enumerate(dp['evidences']):
                # 形成主语和宾语的字符串集合
                sub_strs = {sub_str}
                obj_strs = {obj_str}

                if dp['evidences_id'] != []:
                    # 验证证据 ID 列表长度与证据列表长度一致
                    assert len(dp['evidences_id']) == len(dp['evidences'])
                    sub_id, rel_id, obj_id = dp['evidences_id'][evidence_idx]

                    # 验证关系 ID 与关系字符串一致
                    assert rel_id == rel_str

                    # 如果主语和宾语 ID 存在别名，则更新集合
                    if sub_id in aliases:
                        sub_strs.update(aliases[sub_id]["aliases"])
                    if obj_id in aliases:
                        obj_strs.update(aliases[obj_id]["aliases"])

                # 初始化黄金证据
                gold_evidence = []

                # 形成所有主语和宾语的组合，并将其添加到黄金证据中
                for sub_str, obj_str in itertools.product(sub_strs, obj_strs):
                    gold_evidence.append([sub_str, rel_str, obj_str])

                # 将黄金证据添加到黄金证据列表中
                gold_evidences.append(gold_evidence)

            # 更新证据的评价指标
            evi_em, evi_prec, evi_recall = update_evi(
                metrics, prediction['evidence'][cur_id], gold_evidences)

        # 如果可以进行联合评估
        if can_eval_joint:
            # 计算联合精确度和召回率
            joint_prec = prec * sp_prec * evi_prec
            joint_recall = recall * sp_recall * evi_recall
            # 计算联合 F1 值
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            # 计算联合 EM 值
            joint_em = em * sp_em * evi_em

            # 更新各项指标
            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    # 获取数据项总数
    N = len(gold)

    # 对每项指标进行归一化并四舍五入
    for k in metrics.keys():
        metrics[k] = round(metrics[k] / N * 100, 2)

    # 打印最终的指标结果
    print(json.dumps(metrics, indent=4))
# 如果当前模块是主程序
if __name__ == '__main__':
    """
    """
    # 使用 eval 函数执行命令行参数传递的表达式
    eval(sys.argv[1], sys.argv[2], sys.argv[3])
    # 注释掉的示例调用 eval 函数的实际参数
    # eval("pred.json", "gold.json", "id_aliases.json")
```