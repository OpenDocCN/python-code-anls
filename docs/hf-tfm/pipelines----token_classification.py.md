# `.\transformers\pipelines\token_classification.py`

```
    # 添加结束文档字符串的装饰器，参数为 PIPELINE_INIT_ARGS 中定义的参数
    # 这是一个文档字符串,提供了一些参数的说明
    r"""
        ignore_labels (`List[str]`, defaults to `["O"]`):
            一个要忽略的标签列表。
        grouped_entities (`bool`, *optional*, defaults to `False`):
            DEPRECATED,请使用 `aggregation_strategy` 参数代替。是否将对应同一个实体的令牌进行分组。
        stride (`int`, *optional*):
            如果提供了 stride,管道会在整个文本上应用。文本会被分成大小为 model_max_length 的块。只有在使用快速分词器且 `aggregation_strategy` 不为 `NONE` 时才有效。此参数的值定义了块之间重叠的令牌数量。换句话说,模型每次会向前移动 `tokenizer.model_max_length - stride` 个令牌。
        aggregation_strategy (`str`, *optional*, defaults to `"none"`):
            基于模型预测结果进行融合(或不融合)令牌的策略。
            - "none" : 不进行任何融合,直接返回模型的原始结果
            - "simple" : 将尝试按照默认模式对实体进行分组。(A, B-TAG), (B, I-TAG), (C, I-TAG), (D, B-TAG2) (E, B-TAG2) 将最终变成 [{"word": ABC, "entity": "TAG"}, {"word": "D", "entity": "TAG2"}, {"word": "E", "entity": "TAG2"}]。请注意,两个连续的 B 标签将被视为不同的实体。在基于词的语言中,我们可能会不恰当地将词语分割:比如 Microsoft 被标记为 [{"word": "Micro", "entity": "ENTERPRISE"}, {"word": "soft", "entity": "NAME"}]。查看 FIRST、MAX、AVERAGE 以了解如何减轻这种问题并消除词语歧义(对于支持这种含义的语言,基本上就是由空格分隔的词)。这些减轻措施只能在真实词语上生效,"New york"可能仍被标记为两个不同的实体。
            - "first" : (仅适用于基于词的模型)将使用 `SIMPLE` 策略,但词语不会被标记为不同的标签。当存在歧义时,词语将简单地使用第一个令牌的标签。
            - "average" : (仅适用于基于词的模型)将使用 `SIMPLE` 策略,但词语不会被标记为不同的标签。首先会在令牌间进行平均打分,然后应用最大标签。
            - "max" : (仅适用于基于词的模型)将使用 `SIMPLE` 策略,但词语不会被标记为不同的标签。词语实体将简单地是得分最高的令牌。
    """ 
# 定义了一个TokenClassificationPipeline类，它是ChunkPipeline的子类，用于命名实体识别任务
class TokenClassificationPipeline(ChunkPipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.
    命名实体识别管道，使用任何`ModelForTokenClassification`。查看[named entity recognition示例](../task_summary#named-entity-recognition)获取更多信息。

    Example:
    示例：

    ```python
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
    在[pipeline tutorial](../pipeline_tutorial)中了解更多关于使用管道的基础知识。

    This token recognition pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location or miscellaneous).
    此标记识别管道目前可以使用以下任务标识符从[`pipeline`]加载：`"ner"`（用于预测序列中的标记类别：人物、组织、位置或其他）。

    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=token-classification).
    该管道可以使用的模型是在标记分类任务上进行了微调的模型。查看[huggingface.co/models](https://huggingface.co/models?filter=token-classification)上的可用模型的最新列表。
    """

    # 定义默认的输入名称为"sequences"
    default_input_names = "sequences"

    # 初始化函数，接受args_parser参数，默认为TokenClassificationArgumentHandler()，*args和**kwargs为可变参数
    def __init__(self, args_parser=TokenClassificationArgumentHandler(), *args, **kwargs):
        # 调用父类ChunkPipeline的初始化函数
        super().__init__(*args, **kwargs)
        # 检查模型类型是否为token分类映射名称
        self.check_model_type(
            TF_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
            if self.framework == "tf"
            else MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
        )

        # 初始化基本的分词器为BasicTokenizer，不进行小写转换
        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        # 初始化args_parser为传入的参数
        self._args_parser = args_parser
    # 清理并规范化参数，以备后续处理
    def _sanitize_parameters(
        self,
        ignore_labels=None,  # 忽略的标签列表，默认为 None
        grouped_entities: Optional[bool] = None,  # 是否对实体进行分组，默认为 None
        ignore_subwords: Optional[bool] = None,  # 是否忽略子词，默认为 None
        aggregation_strategy: Optional[AggregationStrategy] = None,  # 聚合策略，默认为 None
        offset_mapping: Optional[List[Tuple[int, int]]] = None,  # 偏移映射列表，默认为 None
        stride: Optional[int] = None,  # 步长，默认为 None
    # 调用函数，实现文本分类
    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.

        Args:
            inputs (`str` or `List[str]`):
                One or several texts (or one list of texts) for token classification.

        Return:
            A list or a list of list of `dict`: Each result comes as a list of dictionaries (one for each token in the
            corresponding input, or each entity if this pipeline was instantiated with an aggregation_strategy) with
            the following keys:

            - **word** (`str`) -- The token/word classified. This is obtained by decoding the selected tokens. If you
              want to have the exact string in the original sentence, use `start` and `end`.
            - **score** (`float`) -- The corresponding probability for `entity`.
            - **entity** (`str`) -- The entity predicted for that token/word (it is named *entity_group* when
              *aggregation_strategy* is not `"none"`.
            - **index** (`int`, only present when `aggregation_strategy="none"`) -- The index of the corresponding
              token in the sentence.
            - **start** (`int`, *optional*) -- The index of the start of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
            - **end** (`int`, *optional*) -- The index of the end of the corresponding entity in the sentence. Only
              exists if the offsets are available within the tokenizer
        """

        # 解析输入参数，获取规范化后的输入和偏移映射列表
        _inputs, offset_mapping = self._args_parser(inputs, **kwargs)
        # 如果存在偏移映射列表，则将其加入参数中
        if offset_mapping:
            kwargs["offset_mapping"] = offset_mapping

        # 调用父类的 __call__ 方法，进行文本分类，并返回结果
        return super().__call__(inputs, **kwargs)
    # 对输入进行预处理的方法，接受句子和偏移映射参数
    def preprocess(self, sentence, offset_mapping=None, **preprocess_params):
        # 获取预处理参数中的分词器参数
        tokenizer_params = preprocess_params.pop("tokenizer_params", {})
        # 如果分词器设置了截断且模型最大长度大于0，则设置截断为True，否则为False
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        # 使用分词器对句子进行处理，返回模型输入
        inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            **tokenizer_params,
        )
        # 移除模型输入中的"overflow_to_sample_mapping"键
        inputs.pop("overflow_to_sample_mapping", None)
        # 计算模型输入中的分块数量
        num_chunks = len(inputs["input_ids"])

        # 对每个分块进行处理
        for i in range(num_chunks):
            # 如果框架为TensorFlow，则对模型输入进行扩展维度
            if self.framework == "tf":
                model_inputs = {k: tf.expand_dims(v[i], 0) for k, v in inputs.items()}
            else:
                # 否则，对模型输入进行unsqueeze操作
                model_inputs = {k: v[i].unsqueeze(0) for k, v in inputs.items()}
            # 如果存在偏移映射，则将其添加到模型输入中
            if offset_mapping is not None:
                model_inputs["offset_mapping"] = offset_mapping
            # 将句子添加到模型输入中，如果是第一个分块则添加，否则设为None
            model_inputs["sentence"] = sentence if i == 0 else None
            # 将是否为最后一个分块的信息添加到模型输入中
            model_inputs["is_last"] = i == num_chunks - 1

            # 生成模型输入
            yield model_inputs

    # 模型前向传播的私有方法，接受模型输入并返回预测结果
    def _forward(self, model_inputs):
        # 获取模型输入中的特殊标记掩码、偏移映射、句子和是否为最后一个分块的信息
        special_tokens_mask = model_inputs.pop("special_tokens_mask")
        offset_mapping = model_inputs.pop("offset_mapping", None)
        sentence = model_inputs.pop("sentence")
        is_last = model_inputs.pop("is_last")
        # 根据框架类型选择相应的前向传播方法，并获取模型的输出logits
        if self.framework == "tf":
            logits = self.model(**model_inputs)[0]
        else:
            output = self.model(**model_inputs)
            logits = output["logits"] if isinstance(output, dict) else output[0]

        # 返回预测结果、特殊标记掩码、偏移映射、句子和是否为最后一个分块的信息，以及模型输入中的其余内容
        return {
            "logits": logits,
            "special_tokens_mask": special_tokens_mask,
            "offset_mapping": offset_mapping,
            "sentence": sentence,
            "is_last": is_last,
            **model_inputs,
        }
    # 对模型输出进行后处理，根据给定的聚合策略以及忽略的标签
    def postprocess(self, all_outputs, aggregation_strategy=AggregationStrategy.NONE, ignore_labels=None):
        # 如果忽略的标签未被指定，初始化为["O"]
        if ignore_labels is None:
            ignore_labels = ["O"]
        # 存储所有的实体
        all_entities = []
        # 遍历所有模型输出
        for model_outputs in all_outputs:
            # 获取logits，将其转换为numpy数组
            logits = model_outputs["logits"][0].numpy()
            # 获取句子
            sentence = all_outputs[0]["sentence"]
            # 获取input_ids
            input_ids = model_outputs["input_ids"][0]
            # 获取偏移映射
            offset_mapping = (
                model_outputs["offset_mapping"][0] if model_outputs["offset_mapping"] is not None else None
            )
            # 获取特殊标记的屏蔽
            special_tokens_mask = model_outputs["special_tokens_mask"][0].numpy()

            # 对logits进行处理，计算分数
            maxes = np.max(logits, axis=-1, keepdims=True)
            shifted_exp = np.exp(logits - maxes)
            scores = shifted_exp / shifted_exp.sum(axis=-1, keepdims=True)

            # 如果使用的是TensorFlow框架，将input_ids和offset_mapping转换为numpy数组
            if self.framework == "tf":
                input_ids = input_ids.numpy()
                offset_mapping = offset_mapping.numpy() if offset_mapping is not None else None

            # 聚合预先生成的实体
            pre_entities = self.gather_pre_entities(
                sentence, input_ids, scores, offset_mapping, special_tokens_mask, aggregation_strategy
            )
            # 对预先生成的实体进行聚合
            grouped_entities = self.aggregate(pre_entities, aggregation_strategy)
            # 过滤掉self.ignore_labels中的内容
            entities = [
                entity
                for entity in grouped_entities
                if entity.get("entity", None) not in ignore_labels
                and entity.get("entity_group", None) not in ignore_labels
            ]
            # 将筛选后的实体添加到all_entities中
            all_entities.extend(entities)
        # 如果有多个块输出，对实体进行聚合
        num_chunks = len(all_outputs)
        if num_chunks > 1:
            all_entities = self.aggregate_overlapping_entities(all_entities)
        # 返回所有实体
        return all_entities

    # 聚合重叠的实体
    def aggregate_overlapping_entities(self, entities):
        # 如果实体数量为0，则直接返回
        if len(entities) == 0:
            return entities
        # 对实体根据起始位置进行排序
        entities = sorted(entities, key=lambda x: x["start"])
        # 存储聚合后的实体
        aggregated_entities = []
        # 初始化上一个实体为第一个实体
        previous_entity = entities[0]
        # 遍历所有实体
        for entity in entities:
            # 如果当前实体的起始位置在前一个实体的范围内
            if previous_entity["start"] <= entity["start"] < previous_entity["end"]:
                # 比较当前实体和前一个实体的长度
                current_length = entity["end"] - entity["start"]
                previous_length = previous_entity["end"] - previous_entity["start"]
                # 根据长度和分数更新前一个实体
                if current_length > previous_length:
                    previous_entity = entity
                elif current_length == previous_length and entity["score"] > previous_entity["score"]:
                    previous_entity = entity
            else:
                # 将前一个实体添加到聚合后的实体中，并更新前一个实体为当前实体
                aggregated_entities.append(previous_entity)
                previous_entity = entity
        # 添加最后一个实体到聚合后的实体中
        aggregated_entities.append(previous_entity)
        # 返回聚合后的实体
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
        # 初始化一个空列表用于存储预实体
        pre_entities = []
        # 遍历每个 token 的得分
        for idx, token_scores in enumerate(scores):
            # 过滤特殊 token
            if special_tokens_mask[idx]:
                continue
            # 将 token id 转换成对应的词
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            # 如果有偏移映射信息
            if offset_mapping is not None:
                start_ind, end_ind = offset_mapping[idx]
                # 处理输出为张量的情况
                if not isinstance(start_ind, int):
                    if self.framework == "pt":
                        start_ind = start_ind.item()
                        end_ind = end_ind.item()
                # 从句子中抽取当前词的原始文本
                word_ref = sentence[start_ind:end_ind]
                # 判断当前 tokenizer 是否是 BPE，是否为子词
                if getattr(self.tokenizer, "_tokenizer", None) and getattr(
                    self.tokenizer._tokenizer.model, "continuing_subword_prefix", None
                ):
                    # 基于 BPE 的词，需要特殊处理
                    is_subword = len(word) != len(word_ref)
                else:
                    # 通用的词处理方法
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
                # 处理未知 token 情况
                if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                    word = word_ref
                    is_subword = False
            # 如果没有偏移映射信息
            else:
                start_ind = None
                end_ind = None
                is_subword = False
            # 将当前 token 的信息存储到预实体中
            pre_entity = {
                "word": word,
                "scores": token_scores,
                "start": start_ind,
                "end": end_ind,
                "index": idx,
                "is_subword": is_subword,
            }
            # 将当前预实体添加到列表中
            pre_entities.append(pre_entity)
        # 返回所有预实体
        return pre_entities
    # 将多个实体进行聚合处理，返回聚合后的实体列表
    def aggregate(self, pre_entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        # 如果聚合策略是 NONE 或 SIMPLE，则执行以下操作
        if aggregation_strategy in {AggregationStrategy.NONE, AggregationStrategy.SIMPLE}:
            # 初始化一个空的实体列表
            entities = []
            # 遍历预处理实体列表
            for pre_entity in pre_entities:
                # 获取最大分数对应的索引
                entity_idx = pre_entity["scores"].argmax()
                # 获取最大分数
                score = pre_entity["scores"][entity_idx]
                # 创建新实体对象
                entity = {
                    "entity": self.model.config.id2label[entity_idx],
                    "score": score,
                    "index": pre_entity["index"],
                    "word": pre_entity["word"],
                    "start": pre_entity["start"],
                    "end": pre_entity["end"],
                }
                # 将新实体添加到实体列表中
                entities.append(entity)
        # 如果聚合策略不是 NONE 或 SIMPLE，则执行以下操作
        else:
            # 调用 self.aggregate_words 方法进行实体聚合
            entities = self.aggregate_words(pre_entities, aggregation_strategy)

        # 如果聚合策略是 NONE，则直接返回实体列表
        if aggregation_strategy == AggregationStrategy.NONE:
            return entities
        # 否则，调用 self.group_entities 方法对实体进行分组
        return self.group_entities(entities)

    # 将多个单词实体进行聚合处理，返回聚合后的实体对象
    def aggregate_word(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> dict:
        # 将多个单词实体合并成一个词字符串
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        # 根据聚合策略不同选择不同的聚合方式
        if aggregation_strategy == AggregationStrategy.FIRST:
            scores = entities[0]["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.MAX:
            # 根据最大的分数实体进行聚合
            max_entity = max(entities, key=lambda entity: entity["scores"].max())
            scores = max_entity["scores"]
            idx = scores.argmax()
            score = scores[idx]
            entity = self.model.config.id2label[idx]
        elif aggregation_strategy == AggregationStrategy.AVERAGE:
            # 计算多个实体分数的平均值
            scores = np.stack([entity["scores"] for entity in entities])
            average_scores = np.nanmean(scores, axis=0)
            entity_idx = average_scores.argmax()
            entity = self.model.config.id2label[entity_idx]
            score = average_scores[entity_idx]
        else:
            # 若聚合策略不合法，则引发值错误
            raise ValueError("Invalid aggregation_strategy")
        # 创建新的实体对象
        new_entity = {
            "entity": entity,
            "score": score,
            "word": word,
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return new_entity
    def aggregate_words(self, entities: List[dict], aggregation_strategy: AggregationStrategy) -> List[dict]:
        """
        Override tokens from a given word that disagree to force agreement on word boundaries.

        Example: micro|soft| com|pany| B-ENT I-NAME I-ENT I-ENT will be rewritten with first strategy as microsoft|
        company| B-ENT I-ENT
        """
        # 按照给定的聚合策略对实体进行聚合
        if aggregation_strategy in {
            AggregationStrategy.NONE,
            AggregationStrategy.SIMPLE,
        }:
            raise ValueError("NONE and SIMPLE strategies are invalid for word aggregation")

        word_entities = []
        word_group = None
        for entity in entities:
            # 如果当前词组为空
            if word_group is None:
                word_group = [entity]
            # 如果当前实体是子词
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
                word_group = [entity]
        # 最后一个词组
        if word_group is not None:
            word_entities.append(self.aggregate_word(word_group, aggregation_strategy))
        return word_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """
        # 获取实体组中的第一个实体
        entity = entities[0]["entity"].split("-", 1)[-1]
        # 计算实体组中实体的平均分数
        scores = np.nanmean([entity["score"] for entity in entities])
        # 提取实体组中的词
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"],
            "end": entities[-1]["end"],
        }
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        # 如果实体名称以 "B-" 开头
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        # 如果实体名称以 "I-" 开头
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # 不符合 B-、I- 格式
            # 默认为 I- 用于表示继续
            bi = "I"
            tag = entity_name
        return bi, tag
    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.

        Args:
            entities (`dict`): The entities predicted by the pipeline.
        """

        # 初始化空列表用于存储最终的实体组
        entity_groups = []
        # 初始化空列表用于存储当前正在处理的实体组
        entity_group_disagg = []

        # 遍历每个实体
        for entity in entities:
            # 如果当前实体组为空，则直接添加当前实体到实体组中
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue

            # 获取当前实体和上一个实体的标签信息
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])

            # 如果当前实体与上一个实体相似且相邻，则将其添加到当前实体组中
            # 分割是为了考虑 "B" 和 "I" 前缀
            # 如果两个实体都是 B 类型，则不应合并
            if tag == last_tag and bi != "B":
                # 修改子词类型为上一个类型
                entity_group_disagg.append(entity)
            else:
                # 如果当前实体与上一个实体不同，则将当前实体组合并到最终实体组中
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        # 如果还有剩余的实体组，则将其添加到最终实体组中
        if entity_group_disagg:
            entity_groups.append(self.group_sub_entities(entity_group_disagg))

        # 返回最终的实体组列表
        return entity_groups
# 将 TokenClassificationPipeline 赋值给 NerPipeline，两者指向相同的对象
NerPipeline = TokenClassificationPipeline
```