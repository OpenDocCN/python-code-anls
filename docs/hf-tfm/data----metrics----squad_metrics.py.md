# `.\data\metrics\squad_metrics.py`

```
# 导入必要的模块和库
import collections  # 导入collections模块，用于处理数据集合
import json  # 导入json模块，用于处理JSON格式数据
import math  # 导入math模块，提供数学运算函数
import re  # 导入re模块，提供正则表达式操作
import string  # 导入string模块，提供字符串处理功能

from ...models.bert import BasicTokenizer  # 从bert模型中导入BasicTokenizer类
from ...utils import logging  # 从工具模块中导入logging模块

# 获取logger对象用于记录日志
logger = logging.get_logger(__name__)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        # 定义函数移除文本中的冠词（a, an, the）
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        # 移除文本中多余的空白符
        return " ".join(text.split())

    def remove_punc(text):
        # 移除文本中的标点符号
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        # 将文本转换为小写
        return text.lower()

    # 对输入文本s进行规范化处理，依次去除冠词、修复空白符、移除标点、转换为小写
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    # 如果输入文本s为空，则返回空列表
    if not s:
        return []
    # 对规范化后的文本s进行分词处理，并返回分词结果列表
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    # 计算精确匹配得分，如果规范化后的答案a_gold与a_pred相同则返回1，否则返回0
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    # 计算F1得分
    gold_toks = get_tokens(a_gold)  # 获取标准答案的分词列表
    pred_toks = get_tokens(a_pred)  # 获取预测答案的分词列表
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)  # 计算分词列表的交集
    num_same = sum(common.values())  # 计算交集中元素的总数

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # 如果标准答案或预测答案的分词列表为空，则如果它们相等返回1，否则返回0
        return int(gold_toks == pred_toks)

    if num_same == 0:
        # 如果交集中没有相同的分词，则返回0
        return 0

    precision = 1.0 * num_same / len(pred_toks)  # 计算精确率
    recall = 1.0 * num_same / len(gold_toks)  # 计算召回率
    f1 = (2 * precision * recall) / (precision + recall)  # 计算F1得分
    return f1


def get_raw_scores(examples, preds):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}  # 初始化精确匹配得分字典
    f1_scores = {}  # 初始化F1得分字典
    # 对于每个示例中的问题，获取问题的唯一标识符
    qas_id = example.qas_id
    # 获取示例中所有答案的文本，仅保留标准化后不为空的答案
    gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

    # 如果没有标准化后不为空的答案，则对于不可回答的问题，正确答案设置为空字符串
    if not gold_answers:
        gold_answers = [""]

    # 如果预测结果中没有当前问题的预测值，则输出一条缺失预测的警告信息并跳过当前问题
    if qas_id not in preds:
        print(f"Missing prediction for {qas_id}")
        continue

    # 获取当前问题的预测值
    prediction = preds[qas_id]
    # 计算所有标准化后不为空的答案与预测值之间的最大精确匹配分数
    exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
    # 计算所有标准化后不为空的答案与预测值之间的最大 F1 分数
    f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

# 返回所有问题的精确匹配分数和 F1 分数
return exact_scores, f1_scores
# 根据预测分数、不可回答概率、问题ID到是否有答案的映射以及阈值，应用无答案阈值，并返回新的分数字典
def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    new_scores = {}
    # 遍历每个问题ID和对应的分数
    for qid, s in scores.items():
        # 预测该问题是否为无答案，根据不可回答概率和设定的阈值
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            # 如果预测为无答案，将该问题的分数设为0或1（取决于是否有答案）
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            # 如果预测为有答案，则保持原始分数
            new_scores[qid] = s
    return new_scores


# 根据精确匹配分数和F1分数以及指定的问题ID列表（如果没有提供，则使用全部问题ID），生成评估结果字典
def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


# 将新的评估结果合并到主要评估结果字典中，使用指定的前缀
def merge_eval(main_eval, new_eval, prefix):
    for k in new_eval:
        main_eval[f"{prefix}_{k}"] = new_eval[k]


# 找到最佳阈值的版本2，根据预测、分数、不可回答概率和问题ID到是否有答案的映射来确定
def find_best_thresh_v2(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    # 遍历排序后的问题ID列表，计算当前分数和最佳分数以及最佳阈值
    for i, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]

    has_ans_score, has_ans_cnt = 0, 0
    # 统计有答案的问题的分数总和和数量
    for qid in qid_list:
        if not qid_to_has_ans[qid]:
            continue
        has_ans_cnt += 1

        if qid not in scores:
            continue
        has_ans_score += scores[qid]

    return 100.0 * best_score / len(scores), best_thresh, 1.0 * has_ans_score / has_ans_cnt


# 找到所有版本2的最佳阈值，将精确匹配和F1分数的最佳结果合并到主要评估结果中
def find_all_best_thresh_v2(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    best_exact, exact_thresh, has_ans_exact = find_best_thresh_v2(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh, has_ans_f1 = find_best_thresh_v2(preds, f1_raw, na_probs, qid_to_has_ans)
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh
    main_eval["has_ans_exact"] = has_ans_exact
    main_eval["has_ans_f1"] = has_ans_f1


# 找到最佳阈值，根据预测、分数、不可回答概率和问题ID到是否有答案的映射来确定
def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    # 使用 enumerate 函数遍历 qid_list 中的元素，索引不关心
    for _, qid in enumerate(qid_list):
        # 如果 qid 不在 scores 字典中，跳过当前循环，继续下一个 qid
        if qid not in scores:
            continue
        # 如果 qid 对应的问题有答案（True），取出其对应的分数
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            # 如果 qid 对应的问题没有答案：
            # 如果 preds[qid] 为真，则设置 diff 为 -1
            if preds[qid]:
                diff = -1
            # 否则，设置 diff 为 0
            else:
                diff = 0
        # 将 diff 加到当前得分 cur_score 上
        cur_score += diff
        # 如果当前得分 cur_score 大于最佳得分 best_score
        if cur_score > best_score:
            # 更新最佳得分为当前得分
            best_score = cur_score
            # 更新最佳阈值为 na_probs[qid]
            best_thresh = na_probs[qid]
    # 计算最终得分比例并返回，乘以 100.0 以得到百分比
    return 100.0 * best_score / len(scores), best_thresh
def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    # 调用 find_best_thresh 函数获取最佳的 exact 和 exact 阈值
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    # 调用 find_best_thresh 函数获取最佳的 f1 和 f1 阈值
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    # 将计算得到的最佳 exact 和 exact 阈值存储到 main_eval 字典中
    main_eval["best_exact"] = best_exact
    main_eval["best_exact_thresh"] = exact_thresh
    # 将计算得到的最佳 f1 和 f1 阈值存储到 main_eval 字典中
    main_eval["best_f1"] = best_f1
    main_eval["best_f1_thresh"] = f1_thresh


def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    # 创建一个字典，记录每个示例的 qas_id 是否有答案
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    # 获取有答案的 qas_id 列表和没有答案的 qas_id 列表
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    # 如果没有提供 no_answer_probs，则初始化为所有预测结果的概率为 0.0
    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    # 计算 exact 和 f1 得分
    exact, f1 = get_raw_scores(examples, preds)

    # 应用 no_answer_probability_threshold 进行 exact 和 f1 阈值处理
    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    # 根据处理后的 exact_threshold 和 f1_threshold 创建评估字典
    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    # 如果存在有答案的 qas_id，则对有答案的部分进行评估
    if has_answer_qids:
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    # 如果存在没有答案的 qas_id，则对没有答案的部分进行评估
    if no_answer_qids:
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    # 查找所有最佳的阈值，并更新到 evaluation 字典中
    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_ans)

    # 返回最终的评估结果字典
    return evaluation


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # 在创建数据时，我们记录了原始（按空格分词的）tokens和我们的WordPiece分词tokens之间的对齐。
    # 现在，`orig_text`包含了我们预测的原始文本对应的原始文本段。
    #
    # 但是，`orig_text`可能包含我们不想要的额外字符。
    #
    # 例如，假设:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # 我们不希望返回`orig_text`，因为它包含额外的"'s"。
    #
    # 我们也不希望返回`pred_text`，因为它已经被标准化了
    # （SQuAD评估脚本也会去除标点符号/小写化，但我们的分词器会进行额外的标准化，比如去除重音字符）。
    #
    # 我们真正想返回的是"Steve Smith"。
    #
    # 因此，我们必须应用一种半复杂的对齐启发式方法，使`pred_text`和`orig_text`之间的字符对齐。
    # 在某些情况下可能会失败，此时我们只返回 `orig_text`。

    def _strip_spaces(text):
        # 初始化一个空列表，用于存储非空格字符
        ns_chars = []
        # 使用有序字典记录非空格字符在原始文本中的索引映射关系
        ns_to_s_map = collections.OrderedDict()
        # 遍历原始文本的字符和索引
        for i, c in enumerate(text):
            # 如果字符是空格，则跳过
            if c == " ":
                continue
            # 记录非空格字符在新文本中的索引，原始索引为 i
            ns_to_s_map[len(ns_chars)] = i
            # 将非空格字符添加到列表中
            ns_chars.append(c)
        # 构建新的没有空格的文本
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # 首先对 `orig_text` 和 `pred_text` 进行分词，去除空格，并检查它们是否长度相同。
    # 如果它们长度不相同，则启发式方法失败。如果它们长度相同，则假定字符是一对一对齐的。
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    # 对 `orig_text` 进行分词
    tok_text = " ".join(tokenizer.tokenize(orig_text))

    # 查找 `pred_text` 在 `tok_text` 中的起始位置
    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # 如果找不到 `pred_text`，且启用了详细日志记录，则记录日志并返回原始文本
        if verbose_logging:
            logger.info(f"Unable to find text: '{pred_text}' in '{orig_text}'")
        return orig_text
    # 计算 `pred_text` 在 `tok_text` 中的结束位置
    end_position = start_position + len(pred_text) - 1

    # 去除 `orig_text` 和 `tok_text` 中的空格，获取新的文本及其字符映射关系
    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    # 如果去除空格后 `orig_text` 和 `tok_text` 长度不相等，则记录日志并返回原始文本
    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info(f"Length not equal after stripping spaces: '{orig_ns_text}' vs '{tok_ns_text}'")
        return orig_text

    # 使用字符对齐映射将 `pred_text` 的字符映射回 `orig_text`
    tok_s_to_ns_map = {}
    for i, tok_index in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    # 如果起始位置在映射表中，则获取原始文本中的起始位置
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    # 如果无法映射起始位置，则记录日志并返回原始文本
    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    # 如果结束位置在映射表中，则获取原始文本中的结束位置
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    # 如果无法映射结束位置，则记录日志并返回原始文本
    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    # 根据映射的起始和结束位置从 `orig_text` 中提取输出文本
    output_text = orig_text[orig_start_position : (orig_end_position + 1)]
    return output_text
# 从给定的logits列表中获取前n_best_size个最高的索引
def _get_best_indexes(logits, n_best_size):
    # 对(索引, 分数)对进行排序，按照分数降序排列
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


# 计算原始logits的softmax概率
def _compute_softmax(scores):
    # 如果scores为空，则返回空列表
    if not scores:
        return []

    max_score = None
    # 找出scores中的最大值
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    # 计算softmax的分子部分（exp(score - max_score)）
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    # 计算softmax概率值
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


# 写最终预测结果到JSON文件，并在需要时记录空结果的log-odds
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
    # 如果需要，记录预测结果到output_prediction_file
    if output_prediction_file:
        logger.info(f"Writing predictions to: {output_prediction_file}")
    # 如果需要，记录nbest结果到output_nbest_file
    if output_nbest_file:
        logger.info(f"Writing nbest to: {output_nbest_file}")
    # 如果version_2_with_negative为True且需要，记录null_log_odds到output_null_log_odds_file
    if output_null_log_odds_file and version_2_with_negative:
        logger.info(f"Writing null_log_odds to: {output_null_log_odds_file}")

    # 根据example_index将all_features分组
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    # 将all_results转换为unique_id到result的映射
    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    # 定义用于存储预测结果的命名元组类型_PrelimPrediction
    _PrelimPrediction = collections.namedtuple(
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    # 用于存储所有预测结果的有序字典
    all_predictions = collections.OrderedDict()
    # 用于存储所有nbest结果的有序字典
    all_nbest_json = collections.OrderedDict()
    # 用于存储scores_diff的JSON结果的有序字典
    scores_diff_json = collections.OrderedDict()

    # 如果需要，将all_predictions写入到output_prediction_file中
    if output_prediction_file:
        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")

    # 如果需要，将all_nbest_json写入到output_nbest_file中
    if output_nbest_file:
        with open(output_nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    # 如果version_2_with_negative为True且需要，将scores_diff_json写入到output_null_log_odds_file中
    if output_null_log_odds_file and version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    # 返回所有预测结果的有序字典
    return all_predictions


# 计算预测结果的对数概率
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
):
    # 这部分代码未提供完整，无法添加注释
    end_n_top,  # 变量名，可能表示某种结束条件或者顶部数量
    version_2_with_negative,  # 变量名，可能表示一个标志或配置选项，用来指示是否包含负数版本
    tokenizer,  # 变量名，可能是一个用于文本处理的工具或者模块，例如分词器
    verbose_logging,  # 变量名，可能表示一个标志或配置选项，用于控制是否输出详细日志信息
# 定义一个命名元组 `_PrelimPrediction`，用于表示预测结果的初步信息，包括特征索引、起始位置索引、结束位置索引、起始对数概率和结束对数概率
_PrelimPrediction = collections.namedtuple(
    "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_log_prob", "end_log_prob"]
)

# 定义一个命名元组 `_NbestPrediction`，用于表示最终的预测结果，包括文本、起始对数概率和结束对数概率
_NbestPrediction = collections.namedtuple(
    "NbestPrediction", ["text", "start_log_prob", "end_log_prob"]
)

# 记录器输出信息，指示将预测结果写入指定的 JSON 文件
logger.info(f"Writing predictions to: {output_prediction_file}")

# 创建一个 defaultdict，用于按照示例索引将特征对象分组存储
example_index_to_features = collections.defaultdict(list)
for feature in all_features:
    example_index_to_features[feature.example_index].append(feature)

# 创建一个字典，将每个结果对象按照其唯一标识存储
unique_id_to_result = {}
for result in all_results:
    unique_id_to_result[result.unique_id] = result

# 创建一个有序字典，用于存储所有的预测结果
all_predictions = collections.OrderedDict()
all_nbest_json = collections.OrderedDict()
scores_diff_json = collections.OrderedDict()

# 将所有预测结果写入指定的 JSON 文件中，格式化输出并换行
with open(output_prediction_file, "w") as writer:
    writer.write(json.dumps(all_predictions, indent=4) + "\n")

# 将所有最佳预测结果写入指定的 JSON 文件中，格式化输出并换行
with open(output_nbest_file, "w") as writer:
    writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

# 如果设置了包含负面情况的版本标志，将分数差异信息写入指定的 JSON 文件中，格式化输出并换行
if version_2_with_negative:
    with open(output_null_log_odds_file, "w") as writer:
        writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

# 返回存储所有预测结果的有序字典
return all_predictions
```