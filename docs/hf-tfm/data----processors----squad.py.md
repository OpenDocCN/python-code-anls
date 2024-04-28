# `.\transformers\data\processors\squad.py`

```py
# 导入所需模块和库
import json  # 导入处理 JSON 格式数据的模块
import os  # 导入操作系统相关功能的模块
from functools import partial  # 导入 partial 函数，用于部分应用函数
from multiprocessing import Pool, cpu_count  # 导入多进程相关模块和获取 CPU 数目的函数

import numpy as np  # 导入处理数组和矩阵的库
from tqdm import tqdm  # 导入进度条模块 tqdm

# 导入特定模块中的类和函数
from ...models.bert.tokenization_bert import whitespace_tokenize  # 从 BERT 模型的 tokenization_bert 模块导入 whitespace_tokenize 函数
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy  # 从 tokenization_utils_base 模块导入 BatchEncoding、PreTrainedTokenizerBase、TruncationStrategy 类
from ...utils import is_tf_available, is_torch_available, logging  # 导入工具模块中的函数和类
from .utils import DataProcessor  # 从当前模块的 utils 模块导入 DataProcessor 类

# 存储插入了 2 个分隔符的分词器集合
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}  

# 检查是否可用 Torch 模块，并导入相关类
if is_torch_available():
    import torch  # 导入 PyTorch 模块
    from torch.utils.data import TensorDataset  # 从 torch.utils.data 模块导入 TensorDataset 类

# 检查是否可用 TensorFlow 模块，并导入相关类
if is_tf_available():
    import tensorflow as tf  # 导入 TensorFlow 模块

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义一个函数，用于优化回答的跨度
def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    # 将原始答案文本进行分词处理
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    # 在给定的起始和结束位置范围内搜索更匹配的答案跨度
    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:  # 如果找到了与原始答案文本相匹配的跨度
                return (new_start, new_end)  # 返回新的起始和结束位置

    return (input_start, input_end)  # 如果未找到更匹配的跨度，则返回原始的起始和结束位置

# 检查给定的 token 是否处于最大上下文范围内
def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    # 遍历所有文档跨度
    for span_index, doc_span in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        # 更新最佳文档跨度的得分和索引
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

# 新版检查给定 token 是否处于最大上下文范围内
def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    # 遍历文档片段列表，获取索引和每个文档片段
    for span_index, doc_span in enumerate(doc_spans):
        # 计算文档片段的结束位置
        end = doc_span["start"] + doc_span["length"] - 1
        # 如果当前位置在文档片段的起始位置之前，则跳过当前文档片段
        if position < doc_span["start"]:
            continue
        # 如果当前位置在文档片段的结束位置之后，则跳过当前文档片段
        if position > end:
            continue
        # 计算当前位置到文档片段起始位置的距离和到结束位置的距离
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        # 计算得分，取左右距离的最小值加上0.01乘以文档片段长度
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        # 如果当前得分是最佳得分或者是第一个得分，则更新最佳得分和最佳文档片段索引
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    # 返回当前文档片段索引是否等于最佳文档片段索引
    return cur_span_index == best_span_index
# 判断字符是否为空白字符的函数
def _is_whitespace(c):
    # 如果字符是空格、制表符、回车符、换行符或者是 U+202F（Narrow No-Break Space）则返回True，否则返回False
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


# 将 SQuAD 格式的示例转换为模型输入特征的函数
def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    # 存储特征的列表
    features = []
    # 如果是训练模式且示例中存在答案
    if is_training and not example.is_impossible:
        # 获取答案的起始和结束位置
        start_position = example.start_position
        end_position = example.end_position

        # 如果在文本中找不到答案，则跳过该示例
        actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)])
        cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text))
        if actual_text.find(cleaned_answer_text) == -1:
            logger.warning(f"Could not find answer: '{actual_text}' vs. '{cleaned_answer_text}'")
            return []

    # 存储从原始 token 到 sub-token 的索引的列表
    tok_to_orig_index = []
    # 存储从 sub-token 到原始 token 的索引的列表
    orig_to_tok_index = []
    # 存储所有文档 token 的列表
    all_doc_tokens = []
    # 遍历文档 token
    for i, token in enumerate(example.doc_tokens):
        # 添加当前所有文档 token 的长度到 orig_to_tok_index 中
        orig_to_tok_index.append(len(all_doc_tokens))
        # 根据 tokenizer 的不同类型进行不同的分词处理
        if tokenizer.__class__.__name__ in [
            "RobertaTokenizer",
            "LongformerTokenizer",
            "BartTokenizer",
            "RobertaTokenizerFast",
            "LongformerTokenizerFast",
            "BartTokenizerFast",
        ]:
            sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
        else:
            sub_tokens = tokenizer.tokenize(token)
        # 遍历分词后的 sub-token
        for sub_token in sub_tokens:
            # 添加当前文档 token 在原文中的索引到 tok_to_orig_index 中
            tok_to_orig_index.append(i)
            # 将 sub-token 添加到所有文档 token 的列表中
            all_doc_tokens.append(sub_token)

    # 如果是训练模式且示例中存在答案
    if is_training and not example.is_impossible:
        # 获取答案在 sub-token 中的起始和结束位置
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1

        # 优化答案跨度
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    # 存储答案的所有可能跨度
    spans = []

    # 对问题文本进行截断处理，确保长度不超过设定的最大长度
    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # 处理插入了 2 个 SEP 标记的分词器所添加 token 的掩码
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    # 存储文档 token 的跨度
    span_doc_tokens = all_doc_tokens
        # 当 spans 数组的长度乘以 doc_stride 小于 all_doc_tokens 的长度时，继续执行循环
        while len(spans) * doc_stride < len(all_doc_tokens):
            # 定义我们想要截断/填充的侧面以及文本/对排序
            # 如果填充的侧面是右侧
            if tokenizer.padding_side == "right":
                # 将 truncated_query 赋给 texts
                texts = truncated_query
                # 将 span_doc_tokens 赋给 pairs
                pairs = span_doc_tokens
                # 截断策略为仅对第二个值进行截断
                truncation = TruncationStrategy.ONLY_SECOND.value
            else:
                # 将 span_doc_tokens 赋给 texts
                texts = span_doc_tokens
                # 将 truncated_query 赋给 pairs
                pairs = truncated_query
                # 截断策略为仅对第一个值进行截断
                truncation = TruncationStrategy.ONLY_FIRST.value

            # 使用 tokenizer 对文本对进行编码，并返回一个字典
            encoded_dict = tokenizer.encode_plus(
                texts,
                pairs,
                truncation=truncation,
                padding=padding_strategy,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                return_token_type_ids=True,
            )

            # 计算段落长度
            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            # 如果编码后的 input_ids 中存在填充 token 的 id
            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                # 如果填充的侧面是右侧
                if tokenizer.padding_side == "right":
                    # 找到第一个填充 token 的 id 之前的非填充 token 的 id
                    non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
                else:
                    # 找到最后一个填充 token 的 id 的位置
                    last_padding_id_position = (
                        len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                    )
                    # 获取最后一个填充 token 之后的所有 token 的 id
                    non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]
            else:
                # 如果编码后的 input_ids 中不存在填充 token 的 id，则直接使用 input_ids
                non_padded_ids = encoded_dict["input_ids"]

            # 将非填充 token 的 id 转换为 token
            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            # 创建一个字典，将 token 的索引映射回原始文本的索引
            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            # 将一些额外的键值对添加到编码后的字典中
            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            # 将编码后的字典添加到 spans 数组中
            spans.append(encoded_dict)

            # 如果编码后的字典中不存在键为 "overflowing_tokens"，或者键为 "overflowing_tokens" 的值为空列表
            if "overflowing_tokens" not in encoded_dict or (
                "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
            ):
                # 跳出循环
                break
            # 更新 span_doc_tokens 为编码后的字典中的 "overflowing_tokens" 键的值
            span_doc_tokens = encoded_dict["overflowing_tokens"]
    # 遍历每个文档片段的索引
    for doc_span_index in range(len(spans)):
        # 遍历每个段落的索引
        for j in range(spans[doc_span_index]["paragraph_len"]):
            # 检查当前位置是否为最大上下文位置
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            # 根据填充位置确定索引
            index = (
                j
                if tokenizer.padding_side == "left"
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
            )
            # 将当前位置是否为最大上下文位置的结果存储到字典中
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context
    for span in spans:
        # 遍历每个 span

        # 找到 CLS token 的位置
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: 用于标记不能作为答案的 token（0 表示可以作为答案）
        # 原始 TF 实现也保留了分类 token（设为 0）
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()

        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # 将 cls 索引设为 0：CLS 索引可用于表示无法回答的情况
        p_mask[cls_index] = 0

        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # 对于训练，如果文档块中不包含注释，则丢弃该块，因为没有需要预测的内容。
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # 这里不能设置 unique_id 和 example_index，它们将在多次处理后设置。
                unique_id=0,
                paragraph_len=span["paragraph_len"],
                token_is_max_context=span["token_is_max_context"],
                tokens=span["tokens"],
                token_to_orig_map=span["token_to_orig_map"],
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible,
                qas_id=example.qas_id,
            )
        )
    return features
# 初始化转换所需的分词器，设置为全局变量
def squad_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert

# 将示例转换为模型可以直接接受的特征列表。此过程依赖于模型，利用分词器的许多功能创建模型的输入。
def squad_convert_examples_to_features(
    examples,
    tokenizer,
    max_seq_length,
    doc_stride,
    max_query_length,
    is_training,
    padding_strategy="max_length",
    return_dataset=False,
    threads=1,
    tqdm_enabled=True,
):
    # 定义辅助方法
    features = []  # 用于存储特征

    # 确定线程数，不能超过 CPU 核心数
    threads = min(threads, cpu_count())

    # 使用线程池处理示例，初始化分词器
    with Pool(threads, initializer=squad_convert_example_to_features_init, initargs=(tokenizer,)) as p:
        # 部分函数，用于处理单个示例的转换为特征
        annotate_ = partial(
            squad_convert_example_to_features,
            max_seq_length=max_seq_length,
            doc_stride=doc_stride,
            max_query_length=max_query_length,
            padding_strategy=padding_strategy,
            is_training=is_training,
        )
        # 使用线程池并行处理示例，得到特征列表
        features = list(
            tqdm(
                p.imap(annotate_, examples, chunksize=32),  # 使用进度条显示处理进度
                total=len(examples),
                desc="convert squad examples to features",
                disable=not tqdm_enabled,  # 控制进度条是否显示
            )
        )

    # 添加示例索引和唯一 ID
    new_features = []  # 用于存储新特征
    unique_id = 1000000000  # 初始唯一 ID
    example_index = 0  # 初始示例索引
    for example_features in tqdm(
        features, total=len(features), desc="add example index and unique id", disable=not tqdm_enabled
        # 使用进度条显示添加进度
    # 遍历输入的特征列表
    ):
        # 如果特征列表为空，则跳过当前迭代
        if not example_features:
            continue
        # 遍历每个特征对象
        for example_feature in example_features:
            # 为每个特征对象设置示例索引和唯一标识符
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            # 将当前特征对象添加到新特征列表中
            new_features.append(example_feature)
            # 更新唯一标识符
            unique_id += 1
        # 更新示例索引
        example_index += 1
    # 使用新特征列表替换原特征列表
    features = new_features
    # 删除新特征列表引用
    del new_features
    # 如果返回的数据集格式为 PyTorch 格式
    if return_dataset == "pt":
        # 检查是否安装了 PyTorch 库
        if not is_torch_available():
            # 如果没有安装 PyTorch，则抛出运行时错误
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # 转换特征列表中的数据为 PyTorch 张量，并构建数据集
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        # 如果不是训练模式
        if not is_training:
            # 创建不包含起始和结束位置索引的数据集
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            # 创建包含起始和结束位置索引的数据集
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
                all_is_impossible,
            )

        # 返回特征列表和构建的数据集
        return features, dataset
    else:
        # 如果返回的数据集格式不是 PyTorch 格式，则直接返回特征列表
        return features
class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        # 如果不是评估模式
        if not evaluate:
            # 获取答案文本并解码为 UTF-8 格式
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            # 获取答案起始位置
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            # 初始化答案列表
            answers = []
        else:
            # 如果是评估模式
            # 通过 zip 函数将答案起始位置和文本解码为 UTF-8 格式，构建答案列表
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text")
            ]

            # 初始化答案和答案起始位置
            answer = None
            answer_start = None

        # 返回 SquadExample 对象
        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of [`~data.processors.squad.SquadExample`] using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from *tensorflow_datasets.load("squad")*
            evaluate: Boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples:

        ```python
        >>> import tensorflow_datasets as tfds

        >>> dataset = tfds.load("squad")

        >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
        >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        ```py"""

        # 根据评估模式选择数据集
        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        # 初始化示例列表
        examples = []
        # 遍历数据集中的每个张量字典
        for tensor_dict in tqdm(dataset):
            # 将每个张量字典转换为 SquadExample 对象并添加到示例列表中
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        # 返回示例列表
        return examples
    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        # 如果数据目录为空，则设为空字符串
        if data_dir is None:
            data_dir = ""

        # 如果训练文件名为空，则抛出值错误
        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        # 打开训练数据文件，读取其中的数据
        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        # 创建训练数据的示例，并返回结果
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        # 如果数据目录为空，则设为空字符串
        if data_dir is None:
            data_dir = ""

        # 如果开发文件名为空，则抛出值错误
        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        # 打开开发数据文件，读取其中的数据
        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        # 创建开发数据的示例，并返回结果
        return self._create_examples(input_data, "dev")
    # 创建训练/测试样本集合
    def _create_examples(self, input_data, set_type):
        # 判断是否为训练集
        is_training = set_type == "train"
        # 初始化样本列表
        examples = []
        # 遍历输入数据
        for entry in tqdm(input_data):
            # 获取标题
            title = entry["title"]
            # 遍历段落
            for paragraph in entry["paragraphs"]:
                # 获取段落文本
                context_text = paragraph["context"]
                # 遍历问题-答案对
                for qa in paragraph["qas"]:
                    # 获取问题ID
                    qas_id = qa["id"]
                    # 获取问题文本
                    question_text = qa["question"]
                    # 初始化起始位置和答案文本
                    start_position_character = None
                    answer_text = None
                    answers = []

                    # 判断是否为不可能的问题
                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        # 如果不是不可能的问题
                        if is_training:
                            # 如果是训练集，获取答案信息
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            # 如果是测试集，获取所有答案信息
                            answers = qa["answers"]

                    # 创建SquadExample对象
                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    # 将样本添加到列表中
                    examples.append(example)
        # 返回样本列表
        return examples
class SquadV1Processor(SquadProcessor):
    # 定义处理Squad V1数据集的处理器类，继承自SquadProcessor
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"

class SquadV2Processor(SquadProcessor):
    # 定义处理Squad V2数据集的处理器类，继承自SquadProcessor
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"

class SquadExample:
    """
    用于Squad数据集的单个训练/测试示例，从磁盘加载。

    Args:
        qas_id: 示例的唯一标识符
        question_text: 问题字符串
        context_text: 上下文字符串
        answer_text: 答案字符串
        start_position_character: 答案开始位置的字符位置
        title: 示例的标题
        answers: 默认为None，在评估过程中使用。保存答案以及它们的起始位置。
        is_impossible: 默认为False，如果示例没有可能的答案，则设置为True。
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # 根据空格分割文本，以便将不同的标记归因于它们的原始位置。
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # 在评估过程中，开始和结束位置才有值。
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]

class SquadFeatures:
    """
    要馈送给模型的单个Squad示例特征。这些特征是特定于模型的，可以从[`~data.processors.squad.SquadExample`]使用
    :method:*~transformers.data.processors.squad.squad_convert_examples_to_features*方法来创建。
    Args:
        input_ids: 输入序列标记在词汇表中的索引。
        attention_mask: 避免在填充标记索引上执行注意力的掩码。
        token_type_ids: 段标记索引，指示输入的第一部分和第二部分。
        cls_index: CLS 标记的索引。
        p_mask: 用于识别可以作为答案的标记与不能作为答案的标记的掩码。
            对于不能作为答案的标记，掩码为 1，对于可以作为答案的标记，掩码为 0。
        example_index: 示例的索引。
        unique_id: 特征的唯一标识符。
        paragraph_len: 上下文的长度。
        token_is_max_context:
            布尔值列表，标识哪些标记在此特征对象中具有最大上下文。如果一个标记在此特征对象中没有最大上下文，
            则意味着另一个特征对象具有更多与该标记相关的信息，并且应优先于此特征对象处理该标记。
        tokens: 与输入标识对应的标记列表。
        token_to_orig_map: 标记与原始文本之间的映射，用于识别答案。
        start_position: 答案标记的起始索引。
        end_position: 答案标记的结束索引。
        encoding: 可选地存储具有快速标记器对齐方法的 BatchEncoding。
    """

    def __init__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
        is_impossible,
        qas_id: str = None,
        encoding: BatchEncoding = None,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.qas_id = qas_id

        self.encoding = encoding
# 定义一个 SquadResult 类，用于评估模型在 SQuAD 数据集上的输出
class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    # 初始化 SquadResult 对象
    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        # 设置起始位置的 logits
        self.start_logits = start_logits
        # 设置结束位置的 logits
        self.end_logits = end_logits
        # 设置唯一标识符
        self.unique_id = unique_id

        # 如果存在 start_top_index，则设置起始位置的 top 索引
        if start_top_index:
            self.start_top_index = start_top_index
            # 设置结束位置的 top 索引
            self.end_top_index = end_top_index
            # 设置 cls 的 logits
            self.cls_logits = cls_logits
```