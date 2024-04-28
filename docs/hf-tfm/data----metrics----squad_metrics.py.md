# `.\transformers\data\metrics\squad_metrics.py`

```
# 导入必要的库和模块
import collections  # 用于处理集合数据类型
import json  # 用于处理 JSON 格式的数据
import math  # 用于数学计算
import re  # 用于正则表达式操作
import string  # 用于字符串操作

# 导入来自 HuggingFace 的 BERT 模型的基本分词器
from ...models.bert import BasicTokenizer
# 导入日志记录工具
from ...utils import logging

# 获取或创建当前模块的日志记录器
logger = logging.get_logger(__name__)


# 定义一个函数，用于规范化答案文本，将其转换为规范形式
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    # 移除文章中的冠词（a、an、the）
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    # 修正文本中的空白字符
    def white_space_fix(text):
        return " ".join(text.split())

    # 移除文本中的标点符号
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    # 将文本转换为小写
    def lower(text):
        return text.lower()

    # 依次应用以上函数，规范化答案文本
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# 定义一个函数，将答案文本转换为单词列表（tokens）
def get_tokens(s):
    if not s:  # 如果答案文本为空，则返回空列表
        return []
    return normalize_answer(s).split()  # 否则，返回规范化后的答案文本的单词列表


# 定义一个函数，计算精确匹配得分（exact match）
def compute_exact(a_gold, a_pred):
    # 如果规范化后的黄金答案等于规范化后的预测答案，则返回1，否则返回0
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


# 定义一个函数，计算 F1 分数
def compute_f1(a_gold, a_pred):
    # 获取黄金答案和预测答案的单词列表
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    # 计算共同单词的数量
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    # 如果黄金答案或预测答案为空，则返回1（完全匹配）或0（不匹配）
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    # 如果没有共同单词，则返回0
    if num_same == 0:
        return 0
    # 计算精确率、召回率和 F1 分数
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# 定义一个函数，从示例和模型预测结果中计算精确匹配和 F1 分数
def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    # 初始化精确匹配得分和 F1 分数的字典
    exact_scores = {}
    f1_scores = {}
    # 遍历每个示例
    for example in examples:
        # 获取问题的唯一标识符
        qas_id = example.qas_id
        # 获取示例中所有标准答案的文本，如果答案经过规范化处理后不为空
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        # 如果没有标准答案，则将唯一答案设为一个空字符串
        if not gold_answers:
            gold_answers = [""]
        
        # 如果预测结果中没有当前问题的预测值，则打印缺失预测的信息并继续下一个示例
        if qas_id not in preds:
            print(f"Missing prediction for {qas_id}")
            continue

        # 获取当前问题的预测值
        prediction = preds[qas_id]
        # 计算当前问题的精确匹配得分，取所有标准答案中最高的得分
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        # 计算当前问题的 F1 得分，取所有标准答案中最高的得分
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    # 返回所有问题的精确匹配得分和 F1 得分
    return exact_scores, f1_scores
# 应用无答案阈值，根据预测分数、无答案概率、问题ID到是否有答案的映射以及无答案阈值，调整分数
def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    # 新的分数字典
    new_scores = {}
    # 遍历每个问题ID和其对应的分数
    for qid, s in scores.items():
        # 预测是否为无答案
        pred_na = na_probs[qid] > na_prob_thresh
        # 如果预测为无答案
        if pred_na:
            # 将分数调整为无答案情况下的值（是否有答案的映射取反）
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            # 如果不是无答案，则保持原始分数不变
            new_scores[qid] = s
    # 返回调整后的分数字典
    return new_scores


# 构造评估结果字典，包括精确度和F1分数，可选地指定问题ID列表
def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    # 如果未指定问题ID列表，则使用所有的问题ID
    if not qid_list:
        # 总问题数量
        total = len(exact_scores)
        # 返回按顺序排列的字典，包括精确度、F1分数和总问题数量
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        # 如果指定了问题ID列表，则计算列表中问题的评估结果
        total = len(qid_list)
        # 返回按顺序排列的字典，包括精确度、F1分数和总问题数量
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


# 合并评估结果，将新的评估结果添加到主要评估结果字典中，可指定前缀
def merge_eval(main_eval, new_eval, prefix):
    # 遍历新评估结果字典的键
    for k in new_eval:
        # 将新评估结果添加到主要评估结果字典中，加上指定前缀
        main_eval[f"{prefix}_{k}"] = new_eval[k]


# 找到最佳阈值版本2，根据预测、分数、无答案概率和问题ID到是否有答案的映射找到最佳阈值及相关分数
def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    # 统计无答案问题数量
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    # 按照无答案概率排序的问题ID列表
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    # 遍历问题ID列表
    for i, qid in enumerate(qid_list):
        # 如果问题ID不存在于分数字典中，则跳过
        if qid not in scores:
            continue
        # 如果问题有答案
        if qid_to_has_ans[qid]:
            # 计算分数差异
            diff = scores[qid]
        else:
            # 如果问题无答案，根据预测值确定分数差异
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        # 更新当前分数
        cur_score += diff
        # 如果当前分数超过最佳分数，则更新最佳分数及相应阈值
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    # 统计有答案问题的分数总和及数量
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    # 返回最佳分数、最佳阈值及有答案问题的平均分数
    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


# 找到所有最佳阈值版本2，根据预测、精确度分数、F1分数、无答案概率和问题ID到是否有答案的映射找到所有最佳阈值及相关分数
def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    # 找到精确度最佳阈值及相关分数
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    # 找到F1最佳阈值及相关分数
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    # 将最佳阈值及相关分数添加到主要评估结果字典中
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


# 找到最佳阈值，根据预测、分数、无
    # 遍历问题ID列表，使用索引和问题ID
    for _, qid in enumerate(qid_list):
        # 如果问题ID不在得分字典中，则跳过当前循环
        if qid not in scores:
            continue
        # 如果问题ID对应有答案
        if qid_to_has_ans[qid]:
            # 将得分设置为问题ID对应的得分
            diff = scores[qid]
        else:
            # 如果问题ID对应没有答案
            if preds[qid]:
                # 将得分设置为-1
                diff = -1
            else:
                # 如果问题ID对应没有答案且预测也没有答案，则将得分设置为0
                diff = 0
        # 累加当前得分
        cur_score += diff
        # 如果当前得分大于最佳得分
        if cur_score > best_score:
            # 更新最佳得分和最佳阈值
            best_score = cur_score
            best_thresh = na_probs[qid]
    # 返回最佳得分占总得分的百分比和最佳阈值
    return 100.0 * best_score / len(scores), best_thresh
# 寻找最佳阈值，计算最佳的 exact 和 f1 值，并将结果保存到 main_eval 字典中
def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    # 计算最佳的 exact 值和阈值
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    # 计算最佳的 f1 值和阈值
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    # 将计算得到的最佳 exact 和 f1 值以及对应的阈值保存到 main_eval 字典中
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


# 对给定的 examples 和预测结果 preds 进行评估
def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    # 创建 qas_id 到是否有答案的映射字典
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    # 获取有答案的 qas_id 列表
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    # 获取无答案的 qas_id 列表
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    # 如果没有提供无答案概率，则初始化为 0.0
    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    # 计算 exact 和 f1 值
    exact, f1 = get_raw_scores(examples, preds)

    # 应用无答案阈值来调整 exact 值
    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    # 应用无答案阈值来调整 f1 值
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    # 生成评估结果字典
    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    # 如果有答案的 qas_id 列表不为空，则对有答案的 qas_id 进行评估
    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    # 如果无答案的 qas_id 列表不为空，则对无答案的 qas_id 进行评估
    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    # 如果提供了无答案概率，则寻找所有最佳阈值
    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    # 返回评估结果字典
    return evaluation


# 将标记化的预测结果映射回原始文本
def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # 当创建数据时，我们记录了原始（空格标记化）标记与我们的 WordPiece 标记之间的对齐。
    # 因此，现在 `orig_text` 包含了我们预测的原始文本对应的原始文本范围。
    #
    # 但是，`orig_text` 可能包含我们不希望在预测中的额外字符。
    #
    # 例如，假设：
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # 我们不希望返回 `orig_text`，因为它包含额外的 "'s"。
    #
    # 我们也不希望返回 `pred_text`，因为它已经被规范化
    # （SQuAD 评估脚本也会进行标点符号剥离/小写处理，但我们的标记器会进行额外的规范化，如剥离重音字符）。
    #
    # 我们真正想要返回的是 "Steve Smith"。
    #
    # 因此，我们必须应用一种半复杂的对齐启发式方法
    # 在 `pred_text` 和 `orig_text` 之间获取字符到字符的对齐。
    # 在某些情况下可能失败，此时我们只需返回原始文本。
    def _strip_spaces(text):
        # 创建一个空列表用于存储非空格字符
        ns_chars = []
        # 创建一个有序字典用于存储非空格字符在原始文本中的索引映射关系
        ns_to_s_map = collections.OrderedDict()
        # 遍历原始文本的字符
        for i, c in enumerate(text):
            # 如果字符是空格，则跳过
            if c == " ":
                continue
            # 记录非空格字符在原始文本中的索引映射关系
            ns_to_s_map[len(ns_chars)] = i
            # 将非空格字符添加到列表中
            ns_chars.append(c)
        # 将非空格字符列表连接成字符串
        ns_text = "".join(ns_chars)
        # 返回去除空格后的文本和字符索引映射关系字典
        return (ns_text, ns_to_s_map)
    
    # 首先对原始文本和预测文本进行分词、去除空格，并检查它们的长度是否相同。
    # 如果它们的长度不同，则启发式方法失败。如果它们的长度相同，我们假设字符是一对一对齐的。
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    
    # 对原始文本进行分词并去除空格
    tok_text = " ".join(tokenizer.tokenize(orig_text))
    
    # 查找预测文本在分词、去除空格后的原始文本中的起始位置
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # 如果找不到预测文本，则返回原始文本
        if verbose_logging:
            logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
        return orig_text
    # 计算预测文本在分词、去除空格后的原始文本中的结束位置
    end_position = start_position + len(pred_text) - 1
    
    # 对原始文本和分词、去除空格后的文本进行去除空格操作
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)
    
    # 如果去除空格后的原始文本和分词、去除空格后的文本长度不同，则返回原始文本
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'")
        return orig_text
    
    # 然后使用字符之间的对齐关系将预测文本中的字符投影回原始文本。
    tok_s_to_ns_map = {}
    # 构建分词、去除空格后的文本中字符索引到原始文本中字符索引的映射关系字典
    for i, tok_index in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i
    
    # 查找预测文本在分词、去除空格后的原始文本中的起始位置
    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]
    
    # 如果找不到预测文本在原始文本中的起始位置，则返回原始文本
    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text
    
    # 查找预测文本在分词、去除空格后的原始文本中的结束位置
    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]
    
    # 如果找不到预测文本在原始文本中的结束位置，则返回原始文本
    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text
    
    # 根据起始位置和结束位置在原始文本中提取预测文本，并返回结果
    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text
def _get_best_indexes(logits, n_best_size):
    """从列表中获取前 n 个最佳的 logits 索引。"""
    # 根据 logits 的值对索引进行排序
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """计算原始 logits 的 softmax 概率。"""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def compute_predictions_logits(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    do_lower_case,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    verbose_logging,
    version_2_with_negative,
    null_score_diff_threshold,
    tokenizer,
):
    """将最终预测写入 json 文件，并在需要时记录 null 的 log-odds。"""
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def compute_predictions_log_probs(
    all_examples,
    all_features,
    all_results,
    n_best_size,
    max_answer_length,
    output_prediction_file,
    output_nbest_file,
    output_null_log_odds_file,
    start_n_top,
    end_n_top,  # 控制预测结束标志的参数，用于top-k采样
    version_2_with_negative,  # 控制模型是否支持无答案的情况，用于问答模型
    tokenizer,  # 文本处理器，用于将输入文本转换为模型可接受的格式
    verbose_logging,  # 控制是否输出详细日志信息，影响训练或推理时的输出
    """
    XLNet write prediction logic (more complex than Bert's). Write final predictions to the json file and log-odds of
    null if needed.

    Requires utils_squad_evaluate.py
    """

    # 声明一个命名元组，用于存储预测结果的初步信息
    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"]
    )

    # 声明一个命名元组，用于存储最终的预测结果
    _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
    )

    # 打印输出预测结果写入的目标文件路径
    logger.info(f"Writing predictions to: {output_prediction_file}")

    # 将每个样本的特征按照样本索引组织成字典
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    # 将每个结果按照唯一标识索引组织成字典
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    # 初始化用于存储预测结果、最终候选答案和分数差异的字典
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    # 将预测结果写入指定的 JSON 文件
    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # 将最终候选答案写入指定的 JSON 文件
    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # 如果启用了对 null 的 log-odds 计算，则将结果写入指定的 JSON 文件
    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    # 返回所有预测结果
    return all_predictions
```