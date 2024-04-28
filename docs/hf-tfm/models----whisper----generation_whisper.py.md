# `.\transformers\models\whisper\generation_whisper.py`

```
# 设置文件编码格式为 UTF-8
# 版权声明
# 根据 Apache 许可协议，进行代码许可
# 详细协议内容可在 http://www.apache.org/licenses/LICENSE-2.0 找到
# 除非符合相关法律或以书面形式同意，否则不得使用此文件
# 分发的软件基于 "按原样" 基础分发，不附带任何保证或条件，不论是明示的或暗示的
# 请查看许可协议，以获取特定语言下的权限和限制
# numpy 是用于进行数学计算的Python库
# typing 提供用于支持类型提示的类和函数
# torch 是深度学习框架
# ... 表示文件路径
# 从 tokenization_whisper 模块中引入 TASK_IDS, TO_LANGUAGE_CODE
# 引入模块 utils 和 logging
# 为当前模块设置日志记录器
def _median_filter(inputs: torch.Tensor, filter_width: int) -> torch.Tensor:
    # 对输入进行最后一个维度的中值滤波
    """
    Applies a median filter of width `filter_width` along the last dimension of the input.

    The `inputs` tensor is assumed to be 3- or 4-dimensional.
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        # 如果滤波宽度小于等于 0 或者为偶数，则抛出值错误
        raise ValueError("`filter_width` should be an odd number")

    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs

    # 如果输入的最后一个维度的形状小于等于 pad_width，则直接返回输入
    # 通过在左右边缘进行填充，使用反射模式
    # 对输入进行反射填充
    inputs = nn.functional.pad(inputs, (pad_width, pad_width, 0, 0), mode="reflect")

    # sort() 比 torch.median 更快
    # 得到中值滤波的结果
    result = inputs.unfold(-1, filter_width, 1).sort()[0][..., pad_width]
    return result


def _dynamic_time_warping(matrix: np.ndarray):
    # 应用于衡量两个时间序列之间的相似性: 输入音频和输出标记
    # 用于生成标记级别的时间戳
    """
    Measures similarity between two temporal sequences: the input audio and the output tokens. Used to generate
    token-level timestamps.
    """
    output_length, input_length = matrix.shape
    # 创建全为无穷大的矩阵，并初始化成本和跟踪矩阵
    cost = np.ones((output_length + 1, input_length + 1), dtype=np.float32) * np.inf
    trace = -np.ones((output_length + 1, input_length + 1), dtype=np.float32)

    cost[0, 0] = 0
    # 遍历输入序列的长度范围
    for j in range(1, input_length + 1):
        # 遍历输出序列的长度范围
        for i in range(1, output_length + 1):
            # 获取当前格子左上角、上方和左方的代价
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            # 根据三种移动方向中最小的代价确定当前格子的最小代价及移动方向
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            # 更新当前格子的代价和移动方向
            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t

    # 回溯过程
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    # 设置起始点的移动方向为左下角（2）
    trace[0, :] = 2
    trace[:, 0] = 1

    # 存储路径上的文本索引和时间索引
    text_indices = []
    time_indices = []
    # 从最后一个格子开始，逐步回溯路径
    while i > 0 or j > 0:
        # 将当前文本索引和时间索引加入列表
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        # 根据当前格子的移动方向更新文本索引和时间索引
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            # 抛出运行时错误，指示动态时间规整中的内部错误
            raise RuntimeError(
                f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]. Please file a bug report."
            )

    # 将文本索引和时间索引转换为数组并返回
    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices
# 从logits_processor中获取指定类型的logit_processor对象，并返回其指定属性的值
def _get_attr_from_logit_processors(logits_processor, logit_processor_class, attribute_name):
    # 在logits_processor中找到第一个类型为logit_processor_class的对象，赋值给logit_processor
    logit_processor = next((cls for cls in logits_processor if isinstance(cls, logit_processor_class)), None)
    if logit_processor:
        # 如果logit_processor存在，则返回其指定属性的值，否则返回None
        return getattr(logit_processor, attribute_name, None)
    return None


# 将当前segments填充到指定的最大长度并返回填充后的sequences
def _pad_to_max_length(current_segments, pad_token_id, padding="right", bos_token_tensor=None, cut_off_length=None):
    max_total_length = 0  # 初始化最大总长度为0
    sequences = []  # 初始化序列列表
    if padding not in ["right", "left"]:  # 如果padding不在["right", "left"]中，则抛出ValueError
        raise ValueError(f"`padding` must be either 'right' or 'left', not {padding}")

    for current_segment_list in current_segments:
        if current_segment_list is not None and len([d["tokens"] for d in current_segment_list]) > 0:
            # 将current_segment_list中的tokens拼接成一个序列
            sequence = torch.cat([d["tokens"] for d in current_segment_list], dim=-1)

            if cut_off_length is not None:
                sequence = sequence[-cut_off_length:]  # 如果cut_off_length不为None，则截取sequence

            if bos_token_tensor is not None:
                sequence = torch.cat([bos_token_tensor, sequence])  # 如果bos_token_tensor不为None，则将其与sequence拼接

            sequences.append(sequence)  # 将sequence添加到sequences中
            max_total_length = max(max_total_length, len(sequences[-1]))  # 更新最大总长度

        else:
            sequences.append(bos_token_tensor)  # 如果current_segment_list为None，则将bos_token_tensor添加到sequences中

    for i in range(len(current_segments)):
        pad_length = max_total_length - len(sequences[i])  # 计算需要填充的长度
        pad = (0, pad_length) if padding == "right" else (pad_length, 0)  # 根据padding确定填充的位置
        sequences[i] = F.pad(sequences[i], pad=pad, value=pad_token_id)  # 使用F.pad进行填充

    sequences = torch.stack(sequences, dim=0)  # 将sequences堆叠成一个torch的张量
    return sequences  # 返回填充后的sequences


# WhisperGenerationMixin类，包含generate方法
class WhisperGenerationMixin:
    def generate(
        self,
        input_features: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: bool = False,
        return_timestamps: Optional[bool] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        is_multilingual: Optional[bool] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        condition_on_prev_tokens: Optional[bool] = None,
        temperature: Optional[Union[float, Tuple[float, ...]]] = None,
        compression_ratio_threshold: Optional[float] = None,
        logprob_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = None,
        num_segment_frames: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        time_precision: float = 0.02,
        return_token_timestamps: Optional[bool] = None,
        return_segments: bool = False,
        return_dict_in_generate: Optional[bool] = None,
        **kwargs,
        # 带有后备机制的生成函数，处理输入段，解码器输入 ID，当前批次大小，批次索引映射，查找，段的帧数，最大帧数，温度，生成配置，逻辑处理器，停止条件，允许的前缀标记函数，同步的 GPU，返回标记时间戳，根据前一个标记进行条件处理，其他参数
        def generate_with_fallback(
            self,
            segment_input,
            decoder_input_ids,
            cur_bsz,
            batch_idx_map,
            seek,
            num_segment_frames,
            max_frames,
            temperatures,
            generation_config,
            logits_processor,
            stopping_criteria,
            prefix_allowed_tokens_fn,
            synced_gpus,
            return_token_timestamps,
            do_condition_on_prev_tokens,
            kwargs,
        
        # 后处理生成的输出，处理返回标记时间戳和生成配置
        def _postprocess_outputs(self, seek_outputs, return_token_timestamps, generation_config):
            if return_token_timestamps and hasattr(generation_config, "alignment_heads"):
                num_frames = getattr(generation_config, "num_frames", None)
                seek_outputs["token_timestamps"] = self._extract_token_timestamps(
                    seek_outputs, generation_config.alignment_heads, num_frames=num_frames
                )

            if generation_config.return_dict_in_generate:

                # 根据批次索引将值拆分成单个批次，并转移到 CPU 上
                def split_by_batch_index(values, key, batch_idx):
                    if key == "scores":
                        return [v[batch_idx].cpu() for v in values]
                    if key == "past_key_values":
                        # 我们不保存“past_key_values”，因为这太耗费资源
                        return None
                    return values[batch_idx].cpu()

                sequence_tokens = seek_outputs["sequences"]
                seek_outputs = [
                    {k: split_by_batch_index(v, k, i) for k, v in seek_outputs.items()}
                    for i in range(sequence_tokens.shape[0])
                ]
            else:
                sequence_tokens = seek_outputs

            return sequence_tokens, seek_outputs

        # 是否需要后备机制，处理查找序列，查找输出，索引，逻辑处理器，生成配置，词汇表大小，温度
        def _need_fallback(
            self,
            seek_sequence,
            seek_outputs,
            index,
            logits_processor,
            generation_config,
            vocab_size,
            temperature,
    # 以函数参数作为条件，决定生成的文本是否需要回滚到另一种方式
    def _check_needs_fallback(
        self, index, seek_sequence, seek_outputs, generation_config, temperature, logits_processor
    ):
        # 默认情况下不需要回滚
        needs_fallback = False
        should_skip = False
        # 如果压缩比例阈值不为空，则计算压缩比率并检查是否超过阈值
        if generation_config.compression_ratio_threshold is not None:
            compression_ratio = self._retrieve_compression_ratio(seek_sequence, vocab_size)

            if compression_ratio > generation_config.compression_ratio_threshold:
                needs_fallback = True

        # 如果对数概率阈值不为空，则检查生成的文本对数概率是否低于阈值
        if generation_config.logprob_threshold is not None:
            if "sequences_scores" in seek_outputs[0]:
                logprobs = [s["sequences_scores"] for s in seek_outputs][index]
            else:
                scores = seek_outputs[index]["scores"]
                logprobs = self._retrieve_avg_logprobs(
                    scores, seek_sequence, generation_config.eos_token_id, temperature
                )

            if logprobs < generation_config.logprob_threshold:
                needs_fallback = True

        # 如果无语音概率阈值不为空，则检查生成的文本对应的无语音概率是否超过阈值
        if generation_config.no_speech_threshold is not None:
            no_speech_prob = _get_attr_from_logit_processors(
                logits_processor, WhisperNoSpeechDetection, "no_speech_prob"
            )

            if (
                logprobs < generation_config.logprob_threshold
                and no_speech_prob[index] > generation_config.no_speech_threshold
            ):
                # 如果满足条件则不需要回滚，并且应该跳过当前文本
                needs_fallback = False
                should_skip = True

        # 返回是否需要回滚和是否应该跳过的结果
        return needs_fallback, should_skip

    # 设置无语音检测
    @staticmethod
    def _setup_no_speech_detection(logits_processor, segment_input, decoder_input_ids, kwargs):
        set_inputs = _get_attr_from_logit_processors(logits_processor, WhisperNoSpeechDetection, "set_inputs")
        extra_kwargs = {k: v for k, v in kwargs.items() if torch.is_tensor(v)}
        set_inputs({"inputs": segment_input, "decoder_input_ids": decoder_input_ids, **extra_kwargs})

    # 获取总输入帧数
    @staticmethod
    def _retrieve_total_input_frames(input_features, input_stride, kwargs):
        # 如果存在输入特征，则返回其最后一维的大小
        if input_features is not None:
            return input_features.shape[-1]

        # 如果 kwargs 中包含编码器输出，则根据输入步幅计算总输入帧数
        if "encoder_outputs" in kwargs:
            encoder_outputs_shape = (
                kwargs["encoder_outputs"][0].shape
                if isinstance(kwargs["encoder_outputs"], BaseModelOutput)
                else kwargs["encoder_outputs"].shape
            )
            return encoder_outputs_shape[1] * input_stride

        # 如果既没有输入特征也没有编码器输出，则抛出数值错误
        raise ValueError("Make sure to provide either `input_features` or `encoder_outputs` to `generate`.")

    # 可能警告未使用的输入
    @staticmethod
    def _maybe_warn_unused_inputs(
        condition_on_prev_tokens,
        temperature,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        total_input_frames,
        ):
            # 构建警告消息的前缀，指示音频输入仅包含特定帧数，并启用了短形式转录
            warning_prefix = (
                f"Audio input consists of only {total_input_frames}. "
                "Short-form transcription is activated."
                "{}, but will be ignored."
            )
            # 如果存在对前一个标记的条件，记录警告消息
            if condition_on_prev_tokens is not None:
                logger.warn(warning_prefix.format(f"condition_on_prev_tokens is set to {condition_on_prev_tokens}"))

            # 如果存在压缩比例阈值，记录警告消息
            if compression_ratio_threshold is not None:
                logger.warn(warning_prefix.format(f"compression_ratio_threshold is set to {compression_ratio_threshold}"))

            # 如果存在对数概率阈值，记录警告消息
            if logprob_threshold is not None:
                logger.warn(warning_prefix.format(f"logprob_threshold is set to {logprob_threshold}"))

            # 如果存在无语音阈值，记录警告消息
            if no_speech_threshold is not None:
                logger.warn(warning_prefix.format(f"no_speech_threshold is set to {no_speech_threshold}"))

            # 当温度被传递为列表时，它不能被忽略 => 在这种情况下抛出错误
            if isinstance(temperature, (list, tuple)):
                raise ValueError(
                    f"Audio input consists of only {total_input_frames}. Short-form transcription is activated."
                    f"temperature cannot be set to {temperature} which can only be used for temperature fallback for long-form generation. Make sure to set `temperature` to a float value or `None` for short-form generation."
                )

        # 设置生成过程中的返回输出
        @staticmethod
        def _set_return_outputs(
            return_dict_in_generate, return_token_timestamps, is_shortform, logprob_threshold, generation_config
        ):
            # 如果在生成中没有指定返回字典，则使用默认值
            if return_dict_in_generate is None:
                return_dict_in_generate = generation_config.return_dict_in_generate

            # 设置返回标记时间戳
            generation_config.return_token_timestamps = return_token_timestamps
            if return_token_timestamps:
                return_dict_in_generate = True
                generation_config.output_attentions = True
                generation_config.output_scores = True

            # 如果不是短形式且存在对数概率阈值，则设置返回字典和输出分数
            if not is_shortform and logprob_threshold is not None:
                return_dict_in_generate = True
                generation_config.output_scores = True

            # 更新生成配置中的返回字典设置
            generation_config.return_dict_in_generate = return_dict_in_generate

        @staticmethod
    # 设置返回时间戳
    def _set_return_timestamps(return_timestamps, is_shortform, generation_config):
        # 如果需要返回时间戳
        if return_timestamps is True:
            # 如果生成配置中没有no_timestamps_token_id属性，则抛出数值错误
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You are trying to return timestamps, but the generation config is not properly set. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                )
            generation_config.return_timestamps = True
        # 如果不是简短形式
        elif not is_shortform:
            # 如果不需要返回时间戳，则抛出数值错误
            if return_timestamps is False:
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the model to predict timestamp tokens. Please either pass `return_timestamps=True` or make sure to pass no more than 3000 mel input features."
                )
            # 如果生成配置中没有no_timestamps_token_id属性，则抛出数值错误
            if not hasattr(generation_config, "no_timestamps_token_id"):
                raise ValueError(
                    "You have passed more than 3000 mel input features (> 30 seconds) which automatically enables long-form generation which "
                    "requires the generation config to have `no_timestamps_token_id` correctly. "
                    "Make sure to initialize the generation config with the correct attributes that are needed such as `no_timestamps_token_id`. "
                    "For more details on how to generate the approtiate config, refer to https://github.com/huggingface/transformers/issues/21878#issuecomment-1451902363"
                    "or make sure to pass no more than 3000 mel input features."
                )
            # 输出日志信息
            logger.info("Setting `return_timestamps=True` for long-form generation.")
            generation_config.return_timestamps = True
        # 否则不返回时间戳
        else:
            generation_config.return_timestamps = False

    @staticmethod
    # 设置语言和任务，并根据是否多语言和生成配置进行更新
    def _set_language_and_task(language, task, is_multilingual, generation_config):
        # 如果指定了是否多语言
        if is_multilingual is not None:
            # 如果生成配置中没有 is_multilingual 属性，抛出数值错误
            if not hasattr(generation_config, "is_multilingual"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `is_multilingual` argument "
                    "to `generate`. Please update the generation config as per the instructions "
                    "https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            # 更新生成配置中的 is_multilingual 属性
            generation_config.is_multilingual = is_multilingual

        # 如果生成配置中有 is_multilingual 属性且为 False
        if hasattr(generation_config, "is_multilingual") and not generation_config.is_multilingual:
            # 如果指定了任务或语言，但是模型只能支持英语
            if task is not None or language is not None:
                raise ValueError(
                    "Cannot specify `task` or `language` for an English-only model. If the model is intended to be "
                    "multilingual, pass `is_multilingual=True` to generate, or update the generation config."
                )

        # 如果指定了语言
        if language is not None:
            # 如果生成配置中没有 lang_to_id 属性，抛出数值错误
            if not hasattr(generation_config, "lang_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `language` argument "
                    "to `generate`. Either set the language using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            # 将语言转换为小写，并更新生成配置中的 language 属性
            language = language.lower()
            generation_config.language = language

        # 如果指定了任务
        if task is not None:
            # 如果生成配置中没有 task_to_id 属性，抛出数值错误
            if not hasattr(generation_config, "task_to_id"):
                raise ValueError(
                    "The generation config is outdated and is thus not compatible with the `task` argument "
                    "to `generate`. Either set the task using the `forced_decoder_ids` in the model config, "
                    "or update the generation config as per the instructions https://github.com/huggingface/transformers/issues/25084#issuecomment-1664398224"
                )
            # 更新生成配置中的 task 属性
            generation_config.task = task
    # 设置token ids，从generation_config和config中获取对应的eos_token_id和decoder_start_token_id
    def _set_token_ids(generation_config, config, kwargs):
        # 从kwargs参数中取出eos_token_id和decoder_start_token_id，并从kwargs中删除
        eos_token_id = kwargs.pop("eos_token_id", None)
        decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        # 如果eos_token_id不为None，则使用其值，否则使用generation_config中的值
        eos_token_id = eos_token_id if eos_token_id is not None else generation_config.eos_token_id
        # 如果decoder_start_token_id不为None，则使用其值，否则使用generation_config中的值
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else generation_config.decoder_start_token_id
        )

        # 设置generation_config中的eos_token_id为eos_token_id的值，如果eos_token_id为None，则使用config中的eos_token_id
        generation_config.eos_token_id = eos_token_id if eos_token_id is not None else config.eos_token_id
        # 设置generation_config中的decoder_start_token_id为decoder_start_token_id的值，如果decoder_start_token_id为None，则使用config中的decoder_start_token_id
        generation_config.decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else config.decoder_start_token_id
        )

    @staticmethod
    # 设置生成的帧数，根据return_token_timestamps和generation_config设置帧数
    def _set_num_frames(return_token_timestamps, generation_config, kwargs):
        # 如果return_token_timestamps为True且generation_config中的task是'translate'，则发出警告
        if return_token_timestamps:
            if getattr(generation_config, "task", None) == "translate":
                logger.warning("Token-level timestamps may not be reliable for task 'translate'.")
            # 如果generation_config中没有'alignment_heads'，则抛出值错误
            if not hasattr(generation_config, "alignment_heads"):
                raise ValueError(
                    "Model generation config has no `alignment_heads`, token-level timestamps not available. "
                    "See https://gist.github.com/hollance/42e32852f24243b748ae6bc1f985b13a on how to add this property to the generation config."
                )

            # 从kwargs中取出num_frames，并删除
            generation_config.num_frames = kwargs.pop("num_frames", None)

    @staticmethod
    # 设置阈值和条件，根据logprob_threshold、compression_ratio_threshold、no_speech_threshold和condition_on_prev_tokens设置generation_config中的相应属性
    def _set_thresholds_and_condition(
        generation_config,
        logprob_threshold,
        compression_ratio_threshold,
        no_speech_threshold,
        condition_on_prev_tokens,
    ):
        # 设置generation_config中的logprob_threshold根据logprob_threshold的值或generation_config中的值
        generation_config.logprob_threshold = (
            logprob_threshold
            if logprob_threshold is not None
            else getattr(generation_config, "logprob_threshold", None)
        )
        # 设置generation_config中的compression_ratio_threshold根据compression_ratio_threshold的值或generation_config中的值
        generation_config.compression_ratio_threshold = (
            compression_ratio_threshold
            if compression_ratio_threshold is not None
            else getattr(generation_config, "compression_ratio_threshold", None)
        )
        # 设置generation_config中的no_speech_threshold根据no_speech_threshold的值或generation_config中的值
        generation_config.no_speech_threshold = (
            no_speech_threshold
            if no_speech_threshold is not None
            else getattr(generation_config, "no_speech_threshold", None)
        )
        # 设置generation_config中的condition_on_prev_tokens根据condition_on_prev_tokens的值或generation_config中的值
        generation_config.condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", None)
        )

    @staticmethod
    # 设置condition_on_prev_tokens，根据condition_on_prev_tokens和generation_config设置condition_on_prev_tokens
    def _set_condition_on_prev_tokens(condition_on_prev_tokens, generation_config):
        # 如果condition_on_prev_tokens不为None，则使用其值，否则使用generation_config中的值或False
        condition_on_prev_tokens = (
            condition_on_prev_tokens
            if condition_on_prev_tokens is not None
            else getattr(generation_config, "condition_on_prev_tokens", False)
        )
        # 设置generation_config中的condition_on_prev_tokens为condition_on_prev_tokens的值
        generation_config.condition_on_prev_tokens = condition_on_prev_tokens
    # 静态方法：从批量大小、注意力掩码和总输入帧数中提取最大帧数和查找值
    @staticmethod
    def _retrieve_max_frames_and_seek(batch_size, attention_mask, total_input_frames):
        # 如果批量大小大于1且注意力掩码为None，则引发值错误
        if batch_size > 1 and attention_mask is None:
            raise ValueError(
                "When doing long-form audio transcription, make sure to pass an `attention_mask`. You can retrieve the `attention_mask` by doing `processor(audio, ..., return_attention_mask=True)` "
            )
        # 如果批量大小大于1
        elif batch_size > 1:
            # 计算每个样本的最大帧数，并将结果转移到CPU上
            max_frames = attention_mask.sum(-1).cpu().to(torch.long)
            # 创建全零张量，形状为(batch_size,)，表示每个样本的查找值
            seek = torch.zeros((batch_size,), dtype=torch.long)
        else:
            # 创建长度为1的张量，值为总输入帧数，表示单个样本的最大帧数
            max_frames = torch.ones((1,), dtype=torch.long) * total_input_frames
            # 创建长度为1的张量，值为0，表示单个样本的查找值
            seek = torch.zeros((1,), dtype=torch.long)

        # 返回最大帧数和查找值
        return max_frames, seek

    # 静态方法：从强制解码器ID中提取初始标记
    @staticmethod
    def _retrieve_init_tokens_from_forced_decoder_ids(generation_config):
        # 初始化标记列表，包含生成配置中的解码器起始标记ID
        init_tokens = [generation_config.decoder_start_token_id]
        # 强制解码器ID列表
        forced_decoder_ids = generation_config.forced_decoder_ids
        # 如果强制解码器ID不为None且第一个元素的第一个值为1
        if forced_decoder_ids is not None and forced_decoder_ids[0][0] == 1:
            i = 1
            # 循环直到强制解码器ID列表为空或第一个元素的第一个值不等于i
            while len(forced_decoder_ids) > 0 and forced_decoder_ids[0][0] == i:
                # 将强制解码器ID列表中的值添加到初始标记列表中
                init_tokens += [forced_decoder_ids[0][1]]
                # 弹出已处理的第一个元素
                forced_decoder_ids = forced_decoder_ids[1:]
                i += 1

            # 如果强制解码器ID列表不为空，则更新生成配置中的强制解码器ID列表
            forced_decoder_ids = forced_decoder_ids if len(forced_decoder_ids) > 0 else None
            generation_config.forced_decoder_ids = forced_decoder_ids

        # 返回初始标记列表
        return init_tokens

    # 实例方法：从生成配置、logits处理器、无语音阈值、是否短形式和束搜索数量中提取logits处理器
    def _retrieve_logit_processors(
        self, generation_config, logits_processor, no_speech_threshold, is_shortform, num_beams
    ):
        # 这部分方法需要在给定的代码中补充完整才能进行注释

    # 静态方法：可能减少批处理
    @staticmethod
    def _maybe_reduce_batch(input_features, seek, max_frames, cur_bsz, batch_idx_map):
        # 保存先前的批处理大小
        prev_bsz = cur_bsz
        # 新的批处理索引映射列表
        new_batch_idx_map = []
        # 遍历先前的批处理
        for i in range(prev_bsz):
            # 获取先前批处理中的索引
            prev_i = batch_idx_map[i]
            # 如果查找值大于等于最大帧数
            if seek[prev_i] >= max_frames[prev_i]:
                # 计算需要切除的索引
                cut_index = i + (cur_bsz - prev_bsz)
                # 更新当前批处理大小
                cur_bsz -= 1
                # 切除已完成处理的样本
                input_features = torch.cat([input_features[:cut_index], input_features[cut_index + 1 :]], dim=0)
            else:
                # 将未完成处理的索引添加到新的批处理索引映射列表中
                new_batch_idx_map.append(prev_i)

        # 返回更新后的输入特征、当前批处理大小和新的批处理索引映射列表
        return input_features, cur_bsz, new_batch_idx_map
    # 获取输入特征的分段
    def _get_input_segment(input_features, seek, seek_num_frames, num_segment_frames, cur_bsz, batch_idx_map):
        # 创建一个空的分段输入列表
        segment_input = []
        # 遍历当前批次大小
        for i in range(cur_bsz):
            # 获取当前样本在上一个批次的索引
            prev_i = batch_idx_map[i]
            # 从输入特征中截取当前样本的分段特征
            segment_input_slice = input_features[i : i + 1, :, seek[prev_i] : seek[prev_i] + seek_num_frames[prev_i]]
            
            # 如果分段特征长度小于目标分段长度，则进行填充
            if segment_input_slice.shape[-1] < num_segment_frames:
                segment_input_slice = F.pad(
                    segment_input_slice, pad=(0, num_segment_frames - segment_input_slice.shape[-1])
                )
            
            # 将当前样本的分段特征添加到列表中
            segment_input.append(segment_input_slice)
        
        # 将分段输入列表拼接成一个批次tensor
        segment_input = torch.cat(segment_input, dim=0)
        
        # 返回分段输入tensor
        return segment_input
    
    # 准备解码器输入ID
    @staticmethod
    def _prepare_decoder_input_ids(
        cur_bsz,
        init_tokens,
        current_segments,
        batch_idx_map,
        do_condition_on_prev_tokens,
        generation_config,
        config,
        device,
        suppress_tokens,
        kwargs,
    ):
        # 设置解码器输入序列的最大长度
        cut_off_length = config.max_target_positions // 2 - 1
        
        # 创建一个全1的tensor作为辅助tensor
        one_tensor = torch.ones((cur_bsz, 1), device=device, dtype=torch.long)
        
        # 将初始token复制成批次大小的tensor，并拼接成解码器输入ID
        decoder_input_ids = torch.cat([t * one_tensor for t in init_tokens], dim=-1)
        
        # 获取前一个起始标记的ID
        prev_start_of_text = getattr(generation_config, "prev_sot_token_id", None)
        if prev_start_of_text is None:
            prev_start_of_text = suppress_tokens[-2] if suppress_tokens is not None else None
        
        # 如果需要基于前一个token进行解码
        if any(do_condition_on_prev_tokens) and len(current_segments[0]) > 0:
            # 根据do_condition_on_prev_tokens确定需要使用的前一个token
            active_segments = [current_segments[i] if do_condition_on_prev_tokens[i] else None for i in batch_idx_map]
            # 设置前一个起始标记的ID
            prev_start_of_text = getattr(generation_config, "prev_bos_token_id", None) or prev_start_of_text
            
            # 创建前一个起始标记的tensor
            bos_token_tensor = prev_start_of_text * one_tensor[0]
            
            # 将前一个token填充到最大长度,并拼接到解码器输入ID前面
            prev_tokens = _pad_to_max_length(
                active_segments,
                generation_config.pad_token_id,
                padding="left",
                bos_token_tensor=bos_token_tensor,
                cut_off_length=cut_off_length,
            )
            decoder_input_ids = torch.cat([prev_tokens, decoder_input_ids], dim=-1)
            
            # 设置解码器注意力掩码
            kwargs["decoder_attention_mask"] = decoder_input_ids != generation_config.pad_token_id
        else:
            # 确保"decoder_attention_mask"不会传递给forward方法
            kwargs.pop("decoder_attention_mask", None)
        
        # 返回解码器输入ID和其他参数
        return decoder_input_ids, kwargs
    
    @staticmethod
    # 设置最大新标记数和长度，根据给定的配置、解码器输入标记、生成配置和其他参数
    def _set_max_new_tokens_and_length(config, decoder_input_ids, generation_config, kwargs):
        # 计算初始标记数，取配置中最大目标位置的一半减1和解码器输入标记长度-1的较小值
        num_initial_tokens = min(config.max_target_positions // 2 - 1, decoder_input_ids.shape[-1] - 1)

        # 弹出已传递的最大长度和最大新标记数
        passed_max_length = kwargs.pop("max_length", None)
        passed_max_new_tokens = kwargs.pop("max_new_tokens", None)
        
        # 获取生成配置中的最大长度和最大新标记数
        max_length_config = getattr(generation_config, "max_length", None)
        max_new_tokens_config = getattr(generation_config, "max_new_tokens", None)

        # 初始化最大新标记数和长度
        max_new_tokens = None
        max_length = None

        # 确保不超过最大长度
        if passed_max_length is not None and passed_max_new_tokens is None:
            # 如果已传递最大长度而未传递最大新标记数，则将最大长度设置为传递的最大长度加上初始标记数，同时保证不超过配置中的最大目标位置
            max_length = min(passed_max_length + num_initial_tokens, config.max_target_positions)
            # 记录日志，说明由于输入依赖于先前段落，将最大长度增加了
            logger.info(
                f"Increase max_length from {passed_max_length} to {max_length} since input is conditioned on previous segment."
            )
        elif max_length_config is not None and passed_max_new_tokens is None and max_new_tokens_config is None:
            # 如果生成配置中存在最大长度，且未传递最大新标记数且生成配置中也未设置最大新标记数，则将最大长度设置为生成配置中的最大长度加上初始标记数，同时保证不超过配置中的最大目标位置
            max_length = min(generation_config.max_length + num_initial_tokens, config.max_target_positions)
            # 记录日志，说明由于输入依赖于先前段落，将最大长度增加了
            logger.info(
                f"Increase max_length from {max_length_config} to {max_length} since input is conditioned on previous segment."
            )
        elif (
            passed_max_new_tokens is not None
            and passed_max_new_tokens + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            # 如果已传递最大新标记数且加上解码器输入标记长度后超过了配置中的最大目标位置，则计算允许的最大新标记数
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]
        elif (
            passed_max_new_tokens is None
            and max_new_tokens_config is not None
            and max_new_tokens_config + decoder_input_ids.shape[-1] > config.max_target_positions
        ):
            # 如果未传递最大新标记数且生成配置中存在最大新标记数且加上解码器输入标记长度后超过了配置中的最大目标位置，则计算允许的最大新标记数
            max_new_tokens = config.max_target_positions - decoder_input_ids.shape[-1]

        # 如果存在最大新标记数，则将其添加到参数中
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens

        # 如果存在最大长度，则将其添加到参数中
        if max_length is not None:
            kwargs["max_length"] = max_length

        # 返回参数
        return kwargs

    # 静态方法：检索压缩比，根据给定的标记和词汇表大小
    @staticmethod
    def _retrieve_compression_ratio(tokens, vocab_size):
        """Compute byte length of zlib compressed token bytes vs. byte length of raw token bytes"""
        # 计算标记的字节长度，使用标记所需的字节长度和标记的字节长度的对数来计算
        length = int(math.log2(vocab_size) / 8) + 1
        # 将标记转换为字节，计算其压缩比
        token_bytes = b"".join([t.to_bytes(length, "little") for t in tokens.tolist()])
        compression_ratio = len(token_bytes) / len(zlib.compress(token_bytes))

        # 返回压缩比
        return compression_ratio

    # 静态方法
    @staticmethod
    # 定义一个方法用于检索平均对数概率
    def _retrieve_avg_logprobs(scores, tokens, eos_token_id, temperature):
        # 如果温度大于0，则使用输入的温度值，否则使用1进行重新缩放
        rescale_temperature = temperature if temperature > 0.0 else 1
        # 将得分值堆叠成张量，并转移到tokens所在的设备上
        scores = torch.stack(scores).to(tokens.device)
    
        # 如果得分张量的行数大于tokens张量的行数，则截取得分张量的前tokens行，否则截取tokens的后scores行
        if scores.shape[0] > tokens.shape[0]:
            scores = scores[: tokens.shape[0]]
        else:
            tokens = tokens[-scores.shape[0 :]
    
        # 对得分值乘以重新缩放的温度后，计算对数softmax得到对数概率
        logprobs = F.log_softmax((scores * rescale_temperature).float(), dim=-1).to(scores.dtype)
    
        # 检索所选token的对数概率并进行求和
        sum_logprobs = sum((logprobs[i][tokens[i]] * (tokens[i] != eos_token_id)) for i in range(logprobs.shape[0]))
        # 如果提供了终结符的id，则计算长度，否则返回tokens的行数作为长度
        length = (tokens != eos_token_id).sum(-1) if eos_token_id is not None else tokens.shape[0]
        # 计算平均对数概率
        avg_logprobs = sum_logprobs / (length + 1)
        return avg_logprobs
    
    # 静态方法，用于检索段
    @staticmethod
    def _retrieve_segment(
        # 输入参数包括寻找序列、寻找输出、时间偏移、时间戳开始、寻找帧数、时间精度、输入步长、前一个索引、当前索引
```