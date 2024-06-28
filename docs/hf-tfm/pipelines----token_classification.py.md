# `.\pipelines\token_classification.py`

```py
import types
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

# 导入BasicTokenizer类，用于处理文本的基本分词操作
from ..models.bert.tokenization_bert import BasicTokenizer
# 导入必要的工具函数和类
from ..utils import (
    ExplicitEnum,
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
)
# 导入处理文本和数据的基础类和函数
from .base import ArgumentHandler, ChunkPipeline, Dataset, build_pipeline_init_args

# 如果TensorFlow可用，导入相关模型和函数
if is_tf_available():
    import tensorflow as tf
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES

# 如果PyTorch可用，导入相关模型和函数
if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES


class TokenClassificationArgumentHandler(ArgumentHandler):
    """
    Handles arguments for token classification.
    """

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        # 根据输入类型处理输入数据
        if inputs is not None and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
            inputs = list(inputs)
            batch_size = len(inputs)
        elif isinstance(inputs, str):
            inputs = [inputs]
            batch_size = 1
        elif Dataset is not None and isinstance(inputs, Dataset) or isinstance(inputs, types.GeneratorType):
            return inputs, None
        else:
            raise ValueError("At least one input is required.")

        # 处理偏移映射（offset_mapping）参数
        offset_mapping = kwargs.get("offset_mapping")
        if offset_mapping:
            if isinstance(offset_mapping, list) and isinstance(offset_mapping[0], tuple):
                offset_mapping = [offset_mapping]
            if len(offset_mapping) != batch_size:
                raise ValueError("offset_mapping should have the same batch size as the input")
        
        # 返回处理后的输入数据和偏移映射
        return inputs, offset_mapping


class AggregationStrategy(ExplicitEnum):
    """All the valid aggregation strategies for TokenClassificationPipeline"""

    # 定义TokenClassificationPipeline的有效聚合策略
    NONE = "none"
    SIMPLE = "simple"
    FIRST = "first"
    AVERAGE = "average"
    MAX = "max"


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True),
        r"""
        ignore_labels (`List[str]`, defaults to `["O"]`):
            A list of labels to ignore.
        grouped_entities (`bool`, *optional*, defaults to `False`):
            DEPRECATED, use `aggregation_strategy` instead. Whether or not to group the tokens corresponding to the
            same entity together in the predictions or not.
        stride (`int`, *optional*):
            If stride is provided, the pipeline is applied on all the text. The text is split into chunks of size
            model_max_length. Works only with fast tokenizers and `aggregation_strategy` different from `NONE`. The
            value of this argument defines the number of overlapping tokens between chunks. In other words, the model
            will shift forward by `tokenizer.model_max_length - stride` tokens each step.
        aggregation_strategy (`str`, *optional*, defaults to `"none"`):
            The strategy to fuse (or not) tokens based on the model prediction.

                - "none" : Will simply not do any aggregation and simply return raw results from the model
                - "simple" : Will attempt to group entities following the default schema. (A, B-TAG), (B, I-TAG), (C,
                  I-TAG), (D, B-TAG2) (E, B-TAG2) will end up being [{"word": ABC, "entity": "TAG"}, {"word": "D",
                  "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}] Notice that two consecutive B tags will end up as
                  different entities. On word based languages, we might end up splitting words undesirably : Imagine
                  Microsoft being tagged as [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity":
                  "NAME"}]. Look for FIRST, MAX, AVERAGE for ways to mitigate that and disambiguate words (on languages
                  that support that meaning, which is basically tokens separated by a space). These mitigations will
                  only work on real words, "New york" might still be tagged with two different entities.
                - "first" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Words will simply use the tag of the first token of the word when there
                  is ambiguity.
                - "average" : (works only on word based models) Will use the `SIMPLE` strategy except that words,
                  cannot end up with different tags. scores will be averaged first across tokens, and then the maximum
                  label is applied.
                - "max" : (works only on word based models) Will use the `SIMPLE` strategy except that words, cannot
                  end up with different tags. Word entity will simply be the token with the maximum score.
        """
)
class TokenClassificationPipeline(ChunkPipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.

    Example:

    ```
    >>> from transformers import pipeline

    >>> token_classifier = pipeline(model="Jean-Baptiste/camembert-ner", aggregation_strategy="simple")
    >>> sentence = "Je m'appelle jean-baptiste et je vis à montréal"
    >>> tokens = token_classifier(sentence)
    >>> tokens
    [{'entity_group': 'PER', 'score': 0.9931, 'word': 'jean-baptiste', 'start': 12, 'end': 26}, {'entity_group': 'LOC', 'score': 0.998, 'word': 'montréal', 'start': 38, 'end': 47}]

    >>> token = tokens[0]
    >>> # Start and end provide an easy way to highlight words in the original text.
    >>> sentence[token["start"] : token["end"]]
    ' jean-baptiste'

    >>> # Some models use the same idea to do part of speech.
    >>> syntaxer = pipeline(model="vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
    >>> syntaxer("My name is Sarah and I live in London")
    [{'entity_group': 'PRON', 'score': 0.999, 'word': 'my', 'start': 0, 'end': 2}, {'entity_group': 'NOUN', 'score': 0.997, 'word': 'name', 'start': 3, 'end': 7}, {'entity_group': 'AUX', 'score': 0.994, 'word': 'is', 'start': 8, 'end': 10}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'sarah', 'start': 11, 'end': 16}, {'entity_group': 'CCONJ', 'score': 0.999, 'word': 'and', 'start': 17, 'end': 20}, {'entity_group': 'PRON', 'score': 0.999, 'word': 'i', 'start': 21, 'end': 22}, {'entity_group': 'VERB', 'score': 0.998, 'word': 'live', 'start': 23, 'end': 27}, {'entity_group': 'ADP', 'score': 0.999, 'word': 'in', 'start': 28, 'end': 30}, {'entity_group': 'PROPN', 'score': 0.999, 'word': 'london', 'start': 31, 'end': 37}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This token recognition pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    """

    default_input_names = "sequences"

    def __init__(self, args_parser=TokenClassificationArgumentHandler(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 检查并设置模型类型，根据框架不同选择不同的模型映射名称
        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
        )

        # 初始化基本分词器，不进行大小写转换
        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        # 使用指定的参数解析器
        self._args_parser = args_parser
    # 定义一个方法 `_sanitize_parameters`，用于处理和清理输入的参数，确保它们符合预期格式
    def _sanitize_parameters(
        self,
        ignore_labels=None,  # 忽略的标签列表，可以为 None
        grouped_entities: Optional[bool] = None,  # 是否对实体进行分组的标志，可以为 None
        ignore_subwords: Optional[bool] = None,  # 是否忽略子词的标志，可以为 None
        aggregation_strategy: Optional[AggregationStrategy] = None,  # 聚合策略，可以为 None
        offset_mapping: Optional[List[Tuple[int, int]]] = None,  # 偏移映射的列表，可以为 None
        stride: Optional[int] = None,  # 步幅，可以为 None
    ):

        """
        实现 `__call__` 方法，用于对给定的文本输入进行令牌分类。

        Args:
            inputs (`str` or `List[str]`):
                一个或多个文本（或文本列表）用于令牌分类。

        Return:
            A list or a list of list of `dict`: 每个结果都作为一个字典列表返回（每个输入的每个令牌，或者如果此管道是
            使用了聚合策略实例化，则每个实体都对应一个字典）具有以下键：

            - **word** (`str`) -- 被分类的令牌/单词。这是通过解码所选令牌获得的。如果要获得原始句子中的确切字符串，请使用 `start` 和 `end`。
            - **score** (`float`) -- `entity` 的相应概率。
            - **entity** (`str`) -- 预测的令牌/单词的实体（当 *aggregation_strategy* 不是 `"none"` 时命名为 *entity_group*）。
            - **index** (`int`, 仅在 `aggregation_strategy="none"` 时存在) -- 句子中对应令牌的索引。
            - **start** (`int`, *可选*) -- 句子中对应实体的起始索引。仅在 tokenizer 中可用偏移时存在。
            - **end** (`int`, *可选*) -- 句子中对应实体的结束索引。仅在 tokenizer 中可用偏移时存在。
        """
        
        # 使用 `_args_parser` 方法解析输入，并获取偏移映射
        _inputs, offset_mapping = self._args_parser(inputs, **kwargs)
        
        # 如果存在偏移映射，则将其添加到 kwargs 中
        if offset_mapping:
            kwargs["offset_mapping"] = offset_mapping
        
        # 调用父类的 `__call__` 方法，执行实际的令牌分类任务，并返回结果
        return super().__call__(inputs, **kwargs)
    # 对输入句子进行预处理，返回生成器对象，每次生成一个模型输入字典
    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        # 提取预处理参数中的 tokenizer_params，并从 preprocess_params 中移除
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        # 根据模型的最大长度和是否启用截断来确定是否截断输入句子
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        # 使用 Tokenizer 对句子进行处理，返回模型输入字典
        inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,  # 返回张量格式由 self.framework 决定
            truncation=truncation,  # 是否截断输入句子
            return_special_tokens_mask=True,  # 返回特殊 token 掩码
            return_offsets_mapping=self.tokenizer.is_fast,  # 返回偏移映射（如果 Tokenizer 支持）
            **tokenizer_params,  # 其他 tokenizer 参数
        )
        # 移除字典中的 "overflow_to_sample_mapping" 键值对
        inputs.pop("overflow_to_sample_mapping", None)
        # 计算分块数量
        num_chunks = len(inputs["input_ids"])

        # 遍历每个分块，生成模型输入字典
        for i in range(num_chunks):
            if self.framework == "tf":
                # 如果使用 TensorFlow 框架，对每个值张量进行扩展维度
                model_inputs = {k: tf.expand_dims(v[i], 0) for k, v in inputs.items()}
            else:
                # 如果使用其他框架，对每个值张量进行 unsqueeze 操作
                model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            # 如果提供了偏移映射，则将其添加到模型输入中
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping
            # 将句子添加到模型输入中（仅在第一个分块时添加）
            model_inputs["sentence"] = sentence if i == 0 else None
            # 指示当前分块是否为最后一个分块
            model_inputs["is_last"] = i == num_chunks - 1

            # 使用生成器的 yield 返回模型输入字典
            yield model_inputs

    # 私有方法：模型的前向传播过程
    def _forward(self, model_inputs):
        # 提取模型输入字典中的特殊 token 掩码
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        # 提取模型输入字典中的偏移映射（如果存在）
        offset_mapping = model_inputs.pop("offset_mapping", None)
        # 提取模型输入字典中的句子
        sentence = model_inputs.pop("sentence")
        # 提取模型输入字典中的 is_last 标志
        is_last = model_inputs.pop("is_last")

        # 根据框架类型选择不同的前向传播方式
        if self.framework == "tf":
            # 如果使用 TensorFlow 框架，调用模型的前向传播，返回 logits
            logits = self.model(**model_inputs)[0]
        else:
            # 如果使用其他框架，调用模型的前向传播，获取输出
            output = self.model(**model_inputs)
            # 如果输出为字典，则从中提取 logits；否则，假设输出为 logits
            logits = output["logits"] if isinstance(output, dict) else output[0]

        # 返回包含各种信息的字典，包括 logits、特殊 token 掩码、偏移映射、句子和 is_last 标志
        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "is_last": is_last,
            **model_inputs,  # 将其余模型输入字典内容一并返回
        }
    # 对模型输出进行后处理，根据指定策略聚合结果
    def postprocess(self, all_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        # 如果未提供忽略标签，则默认忽略 "O" 标签
        if ignore_labels is None:
            ignore_labels = ["O"]
        # 存储所有实体的列表
        all_entities = []
        # 遍历所有模型的输出
        for model_outputs in all_outputs:
            # 获取模型预测的 logits，并转换为 NumPy 数组
            logits = model_outputs["logits"][0].numpy()
            # 获取句子文本，假设是所有输出中的第一个句子
            sentence = all_outputs[0]["sentence"]
            # 获取模型输入的 token IDs
            input_ids = model_outputs["input_ids"][0]
            # 获取偏移映射，如果存在的话，转换为 NumPy 数组
            offset_mapping = (
                model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            )
            # 获取特殊 token 掩码，并转换为 NumPy 数组
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

            # 对 logits 进行 softmax 处理，得到每个标签的概率分数
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            # 如果使用 TensorFlow 框架，将 input_ids 和 offset_mapping 转换为 NumPy 数组
            if self.framework == "tf":
                input_ids = input_ids.numpy()
                offset_mapping = offset_mapping.numpy() if offset_mapping is not None else None

            # 调用 gather_pre_entities 方法，获取预测的实体信息
            pre_entities = self.gather_pre_entities(
                sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
            )
            # 调用 aggregate 方法，根据指定策略聚合实体
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            # 过滤掉在 ignore_labels 中的实体或实体组
            entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in ignore_labels
                and entity.get("entity_group", None) not in ignore_labels
            ]
            # 将过滤后的实体列表添加到 all_entities 中
            all_entities.extend(entities)
        # 如果输出包含多个部分（chunks），则对实体进行重叠处理
        num_chunks = len(all_outputs)
        if num_chunks > 1:
            all_entities = self.aggregate_overlapping_entities(all_entities)
        # 返回所有处理后的实体列表
        return all_entities

    # 对重叠的实体进行聚合处理
    def aggregate_overlapping_entities(self, entities):
        # 如果实体列表为空，直接返回空列表
        if len(entities) == 0:
            return entities
        # 按照实体的起始位置进行排序
        entities = sorted(entities, key=lambda x: x["start"])
        # 存储聚合后的实体列表
        aggregated_entities = []
        # 初始化前一个实体为列表中的第一个实体
        previous_entity = entities[0]
        # 遍历所有实体进行聚合处理
        for entity in entities:
            # 如果当前实体的起始位置在前一个实体的范围内
            if previous_entity["start"] <= entity["start"] < previous_entity["end"]:
                # 比较当前实体和前一个实体的长度，选择长度更长或得分更高的实体
                current_length = entity["end"] - entity["start"]
                previous_length = previous_entity["end"] - previous_entity["start"]
                if current_length > previous_length:
                    previous_entity = entity
                elif current_length == previous_length and entity["score"] > previous_entity["score"]:
                    previous_entity = entity
            else:
                # 将前一个实体添加到聚合列表中，并更新为当前实体
                aggregated_entities.append(previous_entity)
                previous_entity = entity
        # 添加最后一个实体到聚合列表中
        aggregated_entities.append(previous_entity)
        # 返回聚合后的实体列表
        return aggregated_entities
    def gather_pre_entities(
        self,
        sentence: str,
        input_ids: np.ndarray,
        scores: np.ndarray,
        offset_mapping: Optional[List[Tuple[int, int]]],
        special_tokens_mask: np.ndarray,
        aggregation_strategy: AggregationStrategy,
    ) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        # 初始化空列表，用于存储预实体字典
        pre_entities = []
        
        # 遍历每个索引和对应的 token_scores
        for idx, token_scores in enumerate(scores):
            # 过滤掉特殊 token
            if special_tokens_mask[idx]:
                continue
            
            # 将输入 token ID 转换为词汇
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            
            # 如果提供了偏移映射，则获取起始和结束索引
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                
                # 处理非整数类型的索引（通常出现在 PyTorch 框架中）
                if not isinstance(start_ind, int):
                    if self.framework == "pt":
                        start_ind = start_ind.item()
                        end_ind = end_ind.item()
                
                # 根据偏移映射从原始句子中获取参考词
                word_ref = sentence[start_ind:end_ind]
                
                # 检查是否是子词（针对 BPE 类型的 tokenizer）
                if getattr(self.tokenizer, "_tokenizer", None) and getattr(
                    self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
                ):
                    # 这是一个 BPE、词感知型 tokenizer，有正确的方式来融合 token
                    is_subword = len(word) != len(word_ref)
                else:
                    # 这是一个回退启发式方法，对于文本和标点混合的情况可能无法正确识别为 "word"。
                    # 非词感知型模型在这种情况下通常无法做得更好。
                    if aggregation_strategy in {
                        AggregationStrategy.FIRST,
                        AggregationStrategy.AVERAGE,
                        AggregationStrategy.MAX,
                    }:
                        warnings.warn(
                            "Tokenizer does not support real words, using fallback heuristic",
                            UserWarning,
                        )
                    is_subword = start_ind > 0 and " " not in sentence[start_ind - 1 : start_ind + 1]
                
                # 如果输入 token 是未知标记，使用参考词替换并设置 is_subword 为 False
                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            else:
                # 如果没有提供偏移映射，则将索引设置为 None，并且 is_subword 设置为 False
                start_ind = None
                end_ind = None
                is_subword = False
            
            # 创建预实体字典
            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            
            # 将预实体字典添加到预实体列表中
            pre_entities.append(pre_entity)
        
        # 返回所有预实体的列表
        return pre_entities
    # 根据预先提供的实体列表和聚合策略，返回聚合后的实体列表
    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        # 检查聚合策略是否为NONE或SIMPLE
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            # 初始化一个空列表来存储聚合后的实体
            entities = []
            # 遍历预先提供的实体列表
            for pre_entity in pre_entities:
                # 获取具有最高分数的实体索引
                entity_idx = pre_entity["scores"].argmax()
                # 获取该实体的分数
                score = pre_entity["scores"][entity_idx]
                # 创建新的实体字典，包含实体名称、分数、索引、单词、起始位置和结束位置
                entity = {
                    "entity": self.model.config.id2label[entity_idx],  # 实体名称
                    "score": score,                                     # 实体分数
                    "index": pre_entity["index"],                       # 实体索引
                    "word": pre_entity["word"],                         # 实体单词
                    "start": pre_entity["start"],                       # 实体起始位置
                    "end": pre_entity["end"],                           # 实体结束位置
                }
                # 将新创建的实体添加到实体列表中
                entities.append(entity)
        else:
            # 使用指定的聚合策略对实体列表进行聚合
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        # 如果聚合策略为NONE，则直接返回实体列表
        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        # 否则，调用方法将实体列表按某种方式分组并返回
        return self.group_entities(entities)

    # 根据给定的实体列表和聚合策略，返回聚合后的单个实体字典
    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        # 将实体列表中的单词转换为字符串形式
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        
        # 根据聚合策略选择不同的聚合方式
        if aggregation_strategy == AggregationStrategy.FIRST:
            # 对于FIRST策略，选择第一个实体的分数最高的标签作为聚合后的实体
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            # 对于MAX策略，选择分数最高的实体作为聚合后的实体
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            # 对于AVERAGE策略，计算所有实体分数的平均值，并选择平均分数最高的实体作为聚合后的实体
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            # 若聚合策略不是NONE、SIMPLE、FIRST、MAX、AVERAGE中的任何一种，则抛出异常
            raise ValueError("Invalid aggregation_strategy")
        
        # 创建新的聚合后的实体字典，包含实体名称、分数、单词、起始位置和结束位置
        new_entity = {
            "entity": entity,                   # 实体名称
            "score": score,                     # 实体分数
            "word": word,                       # 实体单词
            "start": entities[0]["start"],      # 第一个实体的起始位置
            "end": entities[-1]["end"],         # 最后一个实体的结束位置
        }
        # 返回聚合后的实体字典
        return new_entity
    # 覆盖不同意的单词，强制在单词边界上达成一致的聚合策略
    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        # 检查聚合策略是否为 NONE 或 SIMPLE，这两种策略无效
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        # 存储聚合后的单词实体列表
        word_entities = []
        # 初始化单词组列表
        word_group = None
        # 遍历实体列表
        for entity in entities:
            # 如果当前单词组为空，则初始化为当前实体
            if word_group is None:
                word_group = [entity]
            # 如果当前实体是子词，则添加到当前单词组
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                # 否则，对当前单词组进行聚合并添加到结果列表中
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # 处理最后一个单词组
        if word_group is not None:
            word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    # 将相邻的具有相同预测实体的标记组合在一起
    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # 获取实体组中第一个实体的标记，去掉可能存在的 B- 或 I- 前缀
        entity = entities[0]["entity"].split("-", 1)[-1]
        # 计算实体组的分数均值
        scores = np.nanmean([entity["score"] for entity in entities])
        # 提取实体组中的单词列表
        tokens = [entity["word"] for entity in entities]

        # 构建实体组字典
        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    # 获取实体名称的标签类型（B- 或 I-）和实体标签
    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # 如果不以 B- 或 I- 开头，默认为 I- 类型（用于连续实体）
            bi = "I"
            tag = entity_name
        return bi, tag
    # 定义一个方法，用于将具有相同实体预测的相邻标记组合在一起
    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        # 初始化一个空列表，用于存储最终的实体组
        entity_groups = []
        # 初始化一个空列表，用于存储分解后的实体组
        entity_group_disagg = []

        # 遍历预测的每个实体
        for entity in entities:
            # 如果分解后的实体组为空，直接将当前实体加入其中
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # 获取当前实体和上一个实体的类型标签和前缀信息
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            # 如果当前实体的类型标签和前一个实体相同且是相邻的，并且不是"B"类型，将其加入分解后的实体组
            if tag == last_tag and bi != "B":
                entity_group_disagg.append(entity)
            else:
                # 如果当前实体和前一个实体不同，将分解后的实体组聚合为一个完整的实体组，并重新开始新的分解组
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]

        # 处理最后剩余的分解后的实体组
        if entity_group_disagg:
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        # 返回最终的实体组列表
        return entity_groups
# 将 TokenClassificationPipeline 类的别名设置为 NerPipeline
NerPipeline = TokenClassificationPipeline
```