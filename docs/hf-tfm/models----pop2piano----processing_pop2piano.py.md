# `.\transformers\models\pop2piano\processing_pop2piano.py`

```
# 设置字符编码为utf-8
# 版权信息
# 包含了Apache 2.0许可证的文本
# 引入了必要的模块
# 引入了类型提示
# 引入了NumPy库
# 引入了特征提取工具
# 引入了处理工具
# 引入了分词工具
# 引入了TensorType工具
# 定义了Pop2PianoProcessor类，并继承ProcessorMixin类
# 构建了一个Pop2PianoProcessor类，将Pop2Piano Feature Extractor和Pop2Piano Tokenizer封装为一个单一处理器
# 定义了Pop2PianoProcessor类的属性
# 指定了feature_extractor_class属性为Pop2PianoFeatureExtractor类
# 指定了tokenizer_class属性为Pop2PianoTokenizer类
# 初始化Pop2PianoProcessor类的实例
# 定义了__call__方法，用于实现特征提取和编码
# 对音频进行特征提取和编码
# 对采样频率进行特征提取和编码
# 设置每拍的步长
# 设置是否对数据进行重新采样
# 对音符进行特征提取和编码
# 设置是否进行填充
# 设置是否进行截断
# 设置最大长度
# 设置填充的长度为的倍数
# 设置是否显示详细信息
# 其他参数
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        This method uses [`Pop2PianoFeatureExtractor.__call__`] method to prepare log-mel-spectrograms for the model,
        and [`Pop2PianoTokenizer.__call__`] to prepare token_ids from notes.

        Please refer to the docstring of the above two methods for more information.
        """

        # 检查是否提供了音频和采样率或者音符
        # 如果未提供上述信息，抛出 ValueError 异常
        if (audio is None and sampling_rate is None) and (notes is None):
            raise ValueError(
                "You have to specify at least audios and sampling_rate in order to use feature extractor or "
                "notes to use the tokenizer part."
            )

        # 如果提供了音频和采样率，使用 feature_extractor 方法对音频进行处理，得到输入
        if audio is not None and sampling_rate is not None:
            inputs = self.feature_extractor(
                audio=audio,
                sampling_rate=sampling_rate,
                steps_per_beat=steps_per_beat,
                resample=resample,
                **kwargs,
            )

        # 如果提供了音符，使用 tokenizer 方法对音符进行处理，得到输入
        if notes is not None:
            encoded_token_ids = self.tokenizer(
                notes=notes,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                pad_to_multiple_of=pad_to_multiple_of,
                verbose=verbose,
                **kwargs,
            )

        # 如果未提供音符，则返回音频处理结果
        if notes is None:
            return inputs

        # 如果未提供音频或采样率，则返回音符处理结果
        elif audio is None or sampling_rate is None:
            return encoded_token_ids

        # 如果同时提供了音频和采样率，则将音符处理结果的 token_ids 添加到音频处理结果中，并返回
        else:
            inputs["token_ids"] = encoded_token_ids["token_ids"]
            return inputs

    def batch_decode(
        self,
        token_ids,
        feature_extractor_output: BatchFeature,
        return_midi: bool = True,
    ) -> BatchEncoding:
        """
        This method uses [`Pop2PianoTokenizer.batch_decode`] method to convert model generated token_ids to midi_notes.

        Please refer to the docstring of the above two methods for more information.
        """

        # 使用 tokenizer 的 batch_decode 方法将模型生成的 token_ids 转换为 midi_notes
        return self.tokenizer.batch_decode(
            token_ids=token_ids, feature_extractor_output=feature_extractor_output, return_midi=return_midi
        )

    @property
    def model_input_names(self):
        # 获取 tokenizer 和 feature_extractor 的 model_input_names，并返回去重后的列表
        tokenizer_input_names = self.tokenizer.model_input_names
        feature_extractor_input_names = self.feature_extractor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + feature_extractor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        # 检查 save_directory 是否是一个文件，如果是，则抛出 ValueError 异常
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        
        # 如果 save_directory 不存在，则创建该目录
        os.makedirs(save_directory, exist_ok=True)
        
        # 调用父类的 save_pretrained 方法保存模型到 save_directory
        return super().save_pretrained(save_directory, **kwargs)

    @classmethod
    # 从预训练模型中加载模型实例的类方法
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # 调用类方法 _get_arguments_from_pretrained 获取预训练模型的参数
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 使用获取的参数创建并返回一个新的类实例
        return cls(*args)
```