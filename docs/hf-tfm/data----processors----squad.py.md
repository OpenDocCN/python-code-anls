# `.\data\processors\squad.py`

```py
# 版权声明和许可信息
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json  # 导入json模块
import os  # 导入os模块
from functools import partial  # 导入functools模块中的partial函数
from multiprocessing import Pool, cpu_count  # 导入multiprocessing模块中的Pool和cpu_count函数

import numpy as np  # 导入numpy库，并使用np作为别名
from tqdm import tqdm  # 从tqdm库中导入tqdm函数

from ...models.bert.tokenization_bert import whitespace_tokenize  # 从bert模型的tokenization_bert模块导入whitespace_tokenize函数
from ...tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy  # 导入tokenization_utils_base模块中的BatchEncoding, PreTrainedTokenizerBase, TruncationStrategy类
from ...utils import is_tf_available, is_torch_available, logging  # 从utils模块导入is_tf_available, is_torch_available, logging函数
from .utils import DataProcessor  # 从当前目录的utils模块中导入DataProcessor类

# 存储插入2个分隔符令牌的标记器集合
MULTI_SEP_TOKENS_TOKENIZERS_SET = {"roberta", "camembert", "bart", "mpnet"}

if is_torch_available():  # 如果torch可用
    import torch  # 导入torch库

    from torch.utils.data import TensorDataset  # 从torch.utils.data模块导入TensorDataset类

if is_tf_available():  # 如果tensorflow可用
    import tensorflow as tf  # 导入tensorflow库

logger = logging.get_logger(__name__)  # 获取当前模块的logger实例


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))  # 使用tokenizer对原始答案文本进行分词

    for new_start in range(input_start, input_end + 1):  # 遍历起始和结束位置之间的所有可能起始位置
        for new_end in range(input_end, new_start - 1, -1):  # 从结束位置向前遍历到起始位置
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])  # 根据新的起始和结束位置获取文本片段
            if text_span == tok_answer_text:  # 如果文本片段与tokenized答案文本匹配
                return (new_start, new_end)  # 返回新的起始和结束位置作为改进后的答案文本位置

    return (input_start, input_end)  # 如果找不到更好的匹配，返回原始的起始和结束位置


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None  # 初始化最佳分数
    best_span_index = None  # 初始化最佳span的索引
    for span_index, doc_span in enumerate(doc_spans):  # 遍历所有文档span的索引和文档span
        end = doc_span.start + doc_span.length - 1  # 计算span的结束位置
        if position < doc_span.start:  # 如果当前位置小于span的起始位置，则跳过
            continue
        if position > end:  # 如果当前位置大于span的结束位置，则跳过
            continue
        num_left_context = position - doc_span.start  # 计算当前位置左侧的上下文数量
        num_right_context = end - position  # 计算当前位置右侧的上下文数量
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length  # 计算当前span的分数
        if best_score is None or score > best_score:  # 如果当前分数是最佳分数或者比最佳分数更高
            best_score = score  # 更新最佳分数
            best_span_index = span_index  # 更新最佳span的索引

    return cur_span_index == best_span_index  # 返回当前span索引是否是最佳span的索引


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None  # 初始化最佳分数
    best_span_index = None  # 初始化最佳span的索引
    # 省略了部分未实现的代码，可能会返回True或False，根据具体情况而定
    # 遍历文档片段列表，获取每个片段的索引和内容
    for span_index, doc_span in enumerate(doc_spans):
        # 计算当前文档片段的结束位置
        end = doc_span["start"] + doc_span["length"] - 1
        # 如果当前位置在当前文档片段之前，继续下一个片段
        if position < doc_span["start"]:
            continue
        # 如果当前位置在当前文档片段之后，继续下一个片段
        if position > end:
            continue
        # 计算当前位置相对于文档片段起始位置的左侧上下文长度
        num_left_context = position - doc_span["start"]
        # 计算当前位置相对于文档片段结束位置的右侧上下文长度
        num_right_context = end - position
        # 计算当前片段的得分，考虑左右上下文和片段长度的加权
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        # 如果当前得分是最佳得分或者是第一个评分，更新最佳得分和最佳片段索引
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    # 返回当前片段索引是否等于最佳片段索引
    return cur_span_index == best_span_index
# 判断字符 c 是否为空白字符，包括空格、制表符、回车符、换行符和特定的不间断空白符（0x202F）
def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

# 将示例转换为特征集合
def squad_convert_example_to_features(
    example, max_seq_length, doc_stride, max_query_length, padding_strategy, is_training
):
    features = []
    
    # 如果是训练模式且示例不是不可能的情况
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

    # 映射表：tokenized 后的索引到原始 token 的索引
    tok_to_orig_index = []
    # 原始 token 的索引到 tokenized 后的索引
    orig_to_tok_index = []
    # 所有的文档 token
    all_doc_tokens = []
    
    # 遍历示例的每个 token
    for i, token in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        
        # 根据不同的 tokenizer 类型进行 tokenization
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
        
        # 遍历 tokenization 后的每个子 token
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    # 如果是训练模式且示例不是不可能的情况
    if is_training and not example.is_impossible:
        # 确定答案在 tokenized 后的起始和结束位置
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        
        # 改进答案跨度
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
        )

    # spans 是一个空列表
    spans = []

    # 截断后的查询 token 序列
    truncated_query = tokenizer.encode(
        example.question_text, add_special_tokens=False, truncation=True, max_length=max_query_length
    )

    # 对于插入两个 SEP token 在 <context> & <question> 之间的 tokenizer，需要特殊处理添加 token 的 mask 计算
    tokenizer_type = type(tokenizer).__name__.replace("Tokenizer", "").lower()
    sequence_added_tokens = (
        tokenizer.model_max_length - tokenizer.max_len_single_sentence + 1
        if tokenizer_type in MULTI_SEP_TOKENS_TOKENIZERS_SET
        else tokenizer.model_max_length - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

    # 文档 token 序列
    span_doc_tokens = all_doc_tokens
    while len(spans) * doc_stride < len(all_doc_tokens):
        # 如果 span 数组乘以文档步幅小于所有文档标记的长度，则继续执行循环

        # 定义我们希望截断/填充的一侧和文本/配对的排序
        if tokenizer.padding_side == "right":
            # 如果填充在右侧，则将截断后的查询设置为 texts，文档标记为 pairs
            texts = truncated_query
            pairs = span_doc_tokens
            truncation = TruncationStrategy.ONLY_SECOND.value
        else:
            # 否则，将文档标记设置为 texts，截断后的查询设置为 pairs
            texts = span_doc_tokens
            pairs = truncated_query
            truncation = TruncationStrategy.ONLY_FIRST.value

        encoded_dict = tokenizer.encode_plus(  # 使用 tokenizer 编码文本对
            texts,
            pairs,
            truncation=truncation,  # 设置截断策略
            padding=padding_strategy,  # 使用指定的填充策略
            max_length=max_seq_length,  # 设置最大序列长度
            return_overflowing_tokens=True,  # 返回溢出的标记
            stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,  # 设置步幅
            return_token_type_ids=True,  # 返回 token 类型 ID
        )

        paragraph_len = min(
            len(all_doc_tokens) - len(spans) * doc_stride,  # 段落长度限制
            max_seq_length - len(truncated_query) - sequence_pair_added_tokens,  # 最大序列长度限制
        )

        if tokenizer.pad_token_id in encoded_dict["input_ids"]:
            # 如果填充 token 在输入 IDs 中
            if tokenizer.padding_side == "right":
                # 如果填充在右侧，则获取非填充的 ID
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                # 如果填充在左侧，则找到最后一个填充 token 的位置
                last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(tokenizer.pad_token_id)
                )
                non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1 :]
        else:
            # 如果填充 token 不在输入 IDs 中，则所有 token 都是非填充的
            non_padded_ids = encoded_dict["input_ids"]

        tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)  # 将非填充的 ID 转换为 token

        token_to_orig_map = {}
        for i in range(paragraph_len):
            # 创建 token 到原始文档标记索引的映射
            index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
            token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

        encoded_dict["paragraph_len"] = paragraph_len  # 记录段落长度
        encoded_dict["tokens"] = tokens  # 记录 tokens
        encoded_dict["token_to_orig_map"] = token_to_orig_map  # 记录 token 到原始文档标记的映射
        encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
        encoded_dict["token_is_max_context"] = {}  # 记录 token 是否是最大上下文
        encoded_dict["start"] = len(spans) * doc_stride  # 记录起始位置
        encoded_dict["length"] = paragraph_len  # 记录段落长度

        spans.append(encoded_dict)  # 将编码后的字典添加到 spans 列表中

        if "overflowing_tokens" not in encoded_dict or (
            "overflowing_tokens" in encoded_dict and len(encoded_dict["overflowing_tokens"]) == 0
        ):
            # 如果没有溢出的 token，或者存在溢出的 token 且长度为 0，则跳出循环
            break
        span_doc_tokens = encoded_dict["overflowing_tokens"]  # 更新文档标记为溢出的 token
    # 遍历每一个文档片段的索引
    for doc_span_index in range(len(spans)):
        # 遍历当前文档片段中的段落长度
        for j in range(spans[doc_span_index]["paragraph_len"]):
            # 调用函数检查当前位置是否为最大上下文位置
            is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
            # 根据填充方式确定当前 token 的索引位置
            index = (
                j
                if tokenizer.padding_side == "left"  # 如果填充在左侧，则直接使用 j 作为索引
                else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j  # 如果填充在右侧，则加上查询和特殊标记的长度
            )
            # 记录当前 token 是否为最大上下文
            spans[doc_span_index]["token_is_max_context"][index] = is_max_context
    for span in spans:
        # 对于每一个文本片段，执行以下操作：

        # 找到CLS标记的位置
        cls_index = span["input_ids"].index(tokenizer.cls_token_id)

        # p_mask: 用于标记不能作为答案的token（0表示可以作为答案）
        # 原始的TF实现也保留了分类标记（设为0）
        p_mask = np.ones_like(span["token_type_ids"])
        if tokenizer.padding_side == "right":
            # 如果padding在右侧，设置超出截断查询部分后的token为0
            p_mask[len(truncated_query) + sequence_added_tokens :] = 0
        else:
            # 如果padding在左侧，设置从右侧截断token直到超出截断查询部分为止的token为0
            p_mask[-len(span["tokens"]) : -(len(truncated_query) + sequence_added_tokens)] = 0

        # 找到所有的pad token的索引并将其标记为1
        pad_token_indices = np.where(span["input_ids"] == tokenizer.pad_token_id)
        # 找到所有的特殊token的索引并将其标记为1
        special_token_indices = np.asarray(
            tokenizer.get_special_tokens_mask(span["input_ids"], already_has_special_tokens=True)
        ).nonzero()
        p_mask[pad_token_indices] = 1
        p_mask[special_token_indices] = 1

        # 将CLS标记的索引设为0，表示CLS标记可以用于不可能的答案
        p_mask[cls_index] = 0

        # 判断当前文本片段是否不可能有答案
        span_is_impossible = example.is_impossible
        start_position = 0
        end_position = 0
        if is_training and not span_is_impossible:
            # 对于训练集，如果文档片段不包含注释，丢弃该片段，因为无法预测。
            doc_start = span["start"]
            doc_end = span["start"] + span["length"] - 1
            out_of_span = False

            # 如果起始和结束位置不在文档片段范围内，则丢弃该片段
            if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                out_of_span = True

            if out_of_span:
                # 如果超出文档片段范围，则将起始和结束位置设为CLS标记的位置，标记为不可能有答案
                start_position = cls_index
                end_position = cls_index
                span_is_impossible = True
            else:
                if tokenizer.padding_side == "left":
                    doc_offset = 0
                else:
                    doc_offset = len(truncated_query) + sequence_added_tokens

                # 计算起始和结束位置相对于文档片段的偏移量
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        # 将当前文本片段的特征添加到列表中
        features.append(
            SquadFeatures(
                span["input_ids"],
                span["attention_mask"],
                span["token_type_ids"],
                cls_index,
                p_mask.tolist(),
                example_index=0,  # 这里不设置unique_id和example_index，它们将在后续处理中设置。
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
    # 定义全局变量 tokenizer，用于存储传入的 tokenizer 实例
    global tokenizer
    tokenizer = tokenizer_for_convert

# 将给定的示例列表转换为可以直接用作模型输入的特征列表。此函数依赖于特定模型，并利用 tokenizer 的多个特性来创建模型的输入。
# 参数:
#   examples: [`~data.processors.squad.SquadExample`] 的列表
#   tokenizer: [`PreTrainedTokenizer`] 的子类实例
#   max_seq_length: 输入的最大序列长度
#   doc_stride: 当上下文过大并被拆分为多个特征时使用的步幅
#   max_query_length: 查询的最大长度
#   is_training: 是否为模型训练创建特征，还是为评估创建特征
#   padding_strategy: 填充策略，默认为 "max_length"
#   return_dataset: 默认为 False。可以是 'pt' 或 'tf'。
#       如果为 'pt'：返回一个 torch.data.TensorDataset
#       如果为 'tf'：返回一个 tf.data.Dataset
#   threads: 多处理线程数
#   tqdm_enabled: 是否启用 tqdm 进度条，默认为 True
# 返回:
#   [`~data.processors.squad.SquadFeatures`] 的列表

# 示例：
# ```
# processor = SquadV2Processor()
# examples = processor.get_dev_examples(data_dir)
#
# features = squad_convert_examples_to_features(
#     examples=examples,
#     tokenizer=tokenizer,
#     max_seq_length=args.max_seq_length,
#     doc_stride=args.doc_stride,
#     max_query_length=args.max_query_length,
#     is_training=not evaluate,
# )
# ```
    ):
        # 如果 example_features 为空列表，则跳过当前循环
        if not example_features:
            continue
        # 遍历 example_features 列表中的每个元素
        for example_feature in example_features:
            # 设置 example_feature 的 example_index 属性为当前的 example_index 值
            example_feature.example_index = example_index
            # 设置 example_feature 的 unique_id 属性为当前的 unique_id 值
            example_feature.unique_id = unique_id
            # 将 example_feature 添加到 new_features 列表中
            new_features.append(example_feature)
            # 增加 unique_id 的值，用于下一个 example_feature 的 unique_id
            unique_id += 1
        # 增加 example_index 的值，用于下一个 example_features 的 example_index
        example_index += 1
    # 将 new_features 赋值给 features，更新 features 到新的特征列表
    features = new_features
    # 删除 new_features 列表，释放内存
    del new_features
    # 如果 return_dataset 等于 "pt"
    if return_dataset == "pt":
        # 检查是否有可用的 PyTorch 环境，如果没有则抛出 RuntimeError
        if not is_torch_available():
            raise RuntimeError("PyTorch must be installed to return a PyTorch dataset.")

        # 将 features 中的各项属性转换为 PyTorch 的 Tensor 类型，并构建数据集
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
        all_is_impossible = torch.tensor([f.is_impossible for f in features], dtype=torch.float)

        # 如果不是训练模式，则创建 TensorDataset
        if not is_training:
            # 创建包含 all_input_ids 大小范围的索引的 Tensor
            all_feature_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            # 创建 TensorDataset 包含所有的 input_ids, attention_masks, token_type_ids, feature_index, cls_index, p_mask
            dataset = TensorDataset(
                all_input_ids, all_attention_masks, all_token_type_ids, all_feature_index, all_cls_index, all_p_mask
            )
        else:
            # 如果是训练模式，还需要包含 start_positions 和 end_positions
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            # 创建 TensorDataset 包含所有的 input_ids, attention_masks, token_type_ids, start_positions, end_positions, cls_index, p_mask, is_impossible
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

        # 返回 features 和构建好的 dataset
        return features, dataset
    else:
        # 如果 return_dataset 不等于 "pt"，则直接返回 features 列表
        return features
    """
    Processor for the SQuAD data set. overridden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and
    version 2.0 of SQuAD, respectively.
    """
    
    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        # 如果不是评估模式，从张量字典中获取答案文本的第一个值并解码成 UTF-8 格式
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            # 获取答案起始位置的第一个值
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            # 初始化答案列表
            answers = []
        else:
            # 如果是评估模式，从张量字典中获取所有答案起始位置和文本并解码成 UTF-8 格式，存放在字典列表中
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        # 返回 SquadExample 对象，包含问题ID、问题文本、上下文文本、答案文本等信息
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

        ```
        >>> import tensorflow_datasets as tfds

        >>> dataset = tfds.load("squad")

        >>> training_examples = get_examples_from_dataset(dataset, evaluate=False)
        >>> evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        ```"""

        # 根据评估模式选择数据集的子集（训练集或验证集）
        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        # 遍历数据集中的每个张量字典，并将其转换为 SquadExample 对象，存入 examples 列表中
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        # 返回转换后的 SquadExample 对象列表
        return examples
    # 返回训练集示例，从指定的数据目录中获取数据文件。
    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        # 如果 data_dir 为 None，则设为空字符串
        if data_dir is None:
            data_dir = ""

        # 如果 self.train_file 为 None，则抛出数值错误
        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        # 打开指定的训练数据文件，使用 utf-8 编码读取
        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            # 加载 JSON 数据并提取其 "data" 字段
            input_data = json.load(reader)["data"]
        # 使用提取的数据创建示例，并标识为训练集
        return self._create_examples(input_data, "train")

    # 返回开发集示例，从指定的数据目录中获取数据文件。
    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `dev-v1.1.json` and `dev-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        # 如果 data_dir 为 None，则设为空字符串
        if data_dir is None:
            data_dir = ""

        # 如果 self.dev_file 为 None，则抛出数值错误
        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        # 打开指定的开发数据文件，使用 utf-8 编码读取
        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            # 加载 JSON 数据并提取其 "data" 字段
            input_data = json.load(reader)["data"]
        # 使用提取的数据创建示例，并标识为开发集
        return self._create_examples(input_data, "dev")
    # 定义一个私有方法，用于根据输入数据和设置类型创建示例列表
    def _create_examples(self, input_data, set_type):
        # 根据设置类型确定是否为训练模式
        is_training = set_type == "train"
        # 初始化示例列表为空
        examples = []
        # 遍历输入数据中的每一个条目
        for entry in tqdm(input_data):
            # 获取条目的标题
            title = entry["title"]
            # 遍历条目中的每一个段落
            for paragraph in entry["paragraphs"]:
                # 获取段落的文本内容
                context_text = paragraph["context"]
                # 遍历段落中的每一个问答对
                for qa in paragraph["qas"]:
                    # 获取问答对的唯一标识符
                    qas_id = qa["id"]
                    # 获取问答对的问题文本
                    question_text = qa["question"]
                    # 初始化答案起始位置和答案文本为 None
                    start_position_character = None
                    answer_text = None
                    # 初始化答案列表为空
                    answers = []

                    # 检查问答对是否为不可能的情况（较少见的情况）
                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        # 如果不是不可能的情况，根据训练模式选择处理方式
                        if is_training:
                            # 如果是训练模式，获取第一个答案作为标准答案
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            # 如果不是训练模式，获取所有可能的答案列表
                            answers = qa["answers"]

                    # 创建一个新的 SquadExample 对象，并将其加入示例列表
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
                    examples.append(example)
        # 返回创建好的示例列表
        return examples
class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"



# 定义处理SQuAD V1.1数据集的处理器，继承自SquadProcessor类
class SquadV1Processor(SquadProcessor):
    # 训练数据文件名
    train_file = "train-v1.1.json"
    # 开发数据文件名
    dev_file = "dev-v1.1.json"



class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"



# 定义处理SQuAD V2.0数据集的处理器，继承自SquadProcessor类
class SquadV2Processor(SquadProcessor):
    # 训练数据文件名
    train_file = "train-v2.0.json"
    # 开发数据文件名
    dev_file = "dev-v2.0.json"



class SquadExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
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
        self.qas_id = qas_id  # 唯一标识符
        self.question_text = question_text  # 问题文本
        self.context_text = context_text  # 上下文文本
        self.answer_text = answer_text  # 答案文本
        self.title = title  # 示例的标题
        self.is_impossible = is_impossible  # 是否不可能存在答案，默认为False
        self.answers = answers  # 答案及其起始位置，用于评估时使用，默认为空列表

        self.start_position, self.end_position = 0, 0

        doc_tokens = []  # 存储上下文文本的标记列表
        char_to_word_offset = []  # 字符到单词偏移量映射
        prev_is_whitespace = True

        # 根据空白字符分割文本，将不同的标记归属于原始位置
        for c in self.context_text:
            if _is_whitespace(c):  # 判断字符是否为空白字符的辅助函数
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens  # 上下文文本的标记列表
        self.char_to_word_offset = char_to_word_offset  # 字符到单词偏移量映射

        # 仅在评估时，起始和结束位置才有值
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]



class SquadFeatures:
    """
    Single squad example features to be fed to a model. Those features are model-specific and can be crafted from
    [`~data.processors.squad.SquadExample`] using the
    :method:*~transformers.data.processors.squad.squad_convert_examples_to_features* method.



# 单个SQuAD示例的特征，用于模型输入。这些特征是特定于模型的，
# 可以从[`~data.processors.squad.SquadExample`]使用
# :method:*~transformers.data.processors.squad.squad_convert_examples_to_features*方法进行构建。
class SquadFeatures:
    pass  # 这里仅占位，没有额外的实现
    # 初始化函数，用于创建一个新的对象来存储输入特征和答案相关信息
    def __init__(
        self,
        input_ids,                    # 输入序列的token索引列表
        attention_mask,               # 避免在填充token索引上执行注意力计算的掩码
        token_type_ids,               # 指示输入中第一部分和第二部分的段落token索引
        cls_index,                    # CLS（分类）token的索引位置
        p_mask,                       # 用于标识可以作为答案和不可以作为答案的token的掩码
                                      # 为不可作为答案的token设置为1，可以作为答案的设置为0
        example_index,                # 示例的索引
        unique_id,                    # 特征的唯一标识符
        paragraph_len,                # 上下文段落的长度
        token_is_max_context,         # 布尔值列表，标识哪些token在此特征对象中具有最大上下文。
                                      # 如果一个token没有在此特征对象中具有最大上下文，则意味着另一个特征对象对该token有更多相关信息，应优先考虑那个特征对象。
        tokens,                       # 输入ids对应的token列表
        token_to_orig_map,            # token到原始文本的映射，用于识别答案
        start_position,               # 答案起始token索引
        end_position,                 # 答案结束token索引
        is_impossible,                # 标识答案是否不可行
        qas_id: str = None,           # 问题-答案对的唯一标识符（可选）
        encoding: BatchEncoding = None,  # 可选，存储使用快速分词器对齐方法的BatchEncoding
    ):
        self.input_ids = input_ids                # 初始化对象属性：输入ids
        self.attention_mask = attention_mask      # 初始化对象属性：注意力掩码
        self.token_type_ids = token_type_ids      # 初始化对象属性：段落token类型ids
        self.cls_index = cls_index                # 初始化对象属性：CLS token索引
        self.p_mask = p_mask                      # 初始化对象属性：答案标记掩码
        self.example_index = example_index        # 初始化对象属性：示例索引
        self.unique_id = unique_id                # 初始化对象属性：唯一标识符
        self.paragraph_len = paragraph_len        # 初始化对象属性：段落长度
        self.token_is_max_context = token_is_max_context  # 初始化对象属性：最大上下文标记
        self.tokens = tokens                      # 初始化对象属性：tokens
        self.token_to_orig_map = token_to_orig_map  # 初始化对象属性：token到原始文本的映射
        self.start_position = start_position      # 初始化对象属性：答案起始位置
        self.end_position = end_position          # 初始化对象属性：答案结束位置
        self.is_impossible = is_impossible        # 初始化对象属性：是否不可行的标记
        self.qas_id = qas_id                      # 初始化对象属性：问题-答案对的唯一标识符
        self.encoding = encoding                  # 初始化对象属性：BatchEncoding对象
class SquadResult:
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    # 定义 SquadResult 类，用于存储 SQuAD 数据集上模型输出的结果
    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        # 初始化实例属性
        self.start_logits = start_logits  # 存储回答开始位置的 logits
        self.end_logits = end_logits  # 存储回答结束位置的 logits
        self.unique_id = unique_id  # 存储唯一标识符

        # 如果提供了 start_top_index 参数，则初始化以下属性
        if start_top_index:
            self.start_top_index = start_top_index  # 存储开始位置的 top index
            self.end_top_index = end_top_index  # 存储结束位置的 top index
            self.cls_logits = cls_logits  # 存储对应的 cls logits
```