# `.\models\rag\configuration_rag.py`

```py
# 设置文件编码为 UTF-8

# 版权声明和许可协议，此代码版权归 RAG 作者和 HuggingFace Inc. 团队所有，使用 Apache License, Version 2.0 许可
# 详细许可信息可以在 http://www.apache.org/licenses/LICENSE-2.0 获取

# 导入预训练配置类 PretrainedConfig 和辅助函数 add_start_docstrings
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings

# RAG 配置文档字符串，描述了 RagConfig 类的配置信息
RAG_CONFIG_DOC = r"""
    [`RagConfig`] 存储了 *RagModel* 的配置。配置对象继承自 [`PretrainedConfig`]，
    可以用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。
"""

# 使用 add_start_docstrings 函数为 RagConfig 类添加起始文档字符串
@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    # 指定模型类型为 "rag"
    model_type = "rag"
    # 表示 RagConfig 是由其他组件组合而成
    is_composition = True

    # 构造函数，初始化 RagConfig 类的配置参数
    def __init__(
        self,
        vocab_size=None,
        is_encoder_decoder=True,
        prefix=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decoder_start_token_id=None,
        title_sep=" / ",
        doc_sep=" // ",
        n_docs=5,
        max_combined_length=300,
        retrieval_vector_size=768,
        retrieval_batch_size=8,
        dataset="wiki_dpr",
        dataset_split="train",
        index_name="compressed",
        index_path=None,
        passages_path=None,
        use_dummy_dataset=False,
        reduce_loss=False,
        label_smoothing=0.0,
        do_deduplication=True,
        exclude_bos_score=False,
        do_marginalize=False,
        output_retrieved=False,
        use_cache=True,
        forced_eos_token_id=None,
        dataset_revision=None,
        **kwargs,
    ):
        # 调用父类 PretrainedConfig 的构造函数，初始化配置参数
        super().__init__(
            vocab_size=vocab_size,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            prefix=prefix,
            vocab_size=vocab_size,
            **kwargs,
        )
        # 调用父类的初始化方法，传入各种模型配置参数和额外的关键字参数

        assert (
            "question_encoder" in kwargs and "generator" in kwargs
        ), "Config has to be initialized with question_encoder and generator config"
        # 断言确保关键字参数中包含 "question_encoder" 和 "generator"，否则抛出异常信息

        question_encoder_config = kwargs.pop("question_encoder")
        # 从关键字参数中弹出 "question_encoder" 并赋值给变量 question_encoder_config
        question_encoder_model_type = question_encoder_config.pop("model_type")
        # 从 question_encoder_config 中弹出 "model_type" 并赋值给变量 question_encoder_model_type

        decoder_config = kwargs.pop("generator")
        # 从关键字参数中弹出 "generator" 并赋值给变量 decoder_config
        decoder_model_type = decoder_config.pop("model_type")
        # 从 decoder_config 中弹出 "model_type" 并赋值给变量 decoder_model_type

        from ..auto.configuration_auto import AutoConfig
        # 从自动生成的配置模块中导入 AutoConfig 类

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        # 使用 AutoConfig 根据 question_encoder_model_type 和 question_encoder_config 创建 question_encoder 实例
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)
        # 使用 AutoConfig 根据 decoder_model_type 和 decoder_config 创建 generator 实例

        self.reduce_loss = reduce_loss
        # 将 reduce_loss 参数赋值给实例变量 self.reduce_loss
        self.label_smoothing = label_smoothing
        # 将 label_smoothing 参数赋值给实例变量 self.label_smoothing
        self.exclude_bos_score = exclude_bos_score
        # 将 exclude_bos_score 参数赋值给实例变量 self.exclude_bos_score
        self.do_marginalize = do_marginalize
        # 将 do_marginalize 参数赋值给实例变量 self.do_marginalize

        self.title_sep = title_sep
        # 将 title_sep 参数赋值给实例变量 self.title_sep
        self.doc_sep = doc_sep
        # 将 doc_sep 参数赋值给实例变量 self.doc_sep
        self.n_docs = n_docs
        # 将 n_docs 参数赋值给实例变量 self.n_docs
        self.max_combined_length = max_combined_length
        # 将 max_combined_length 参数赋值给实例变量 self.max_combined_length

        self.dataset = dataset
        # 将 dataset 参数赋值给实例变量 self.dataset
        self.dataset_split = dataset_split
        # 将 dataset_split 参数赋值给实例变量 self.dataset_split
        self.index_name = index_name
        # 将 index_name 参数赋值给实例变量 self.index_name

        self.retrieval_vector_size = retrieval_vector_size
        # 将 retrieval_vector_size 参数赋值给实例变量 self.retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        # 将 retrieval_batch_size 参数赋值给实例变量 self.retrieval_batch_size
        self.passages_path = passages_path
        # 将 passages_path 参数赋值给实例变量 self.passages_path
        self.index_path = index_path
        # 将 index_path 参数赋值给实例变量 self.index_path
        self.use_dummy_dataset = use_dummy_dataset
        # 将 use_dummy_dataset 参数赋值给实例变量 self.use_dummy_dataset
        self.dataset_revision = dataset_revision
        # 将 dataset_revision 参数赋值给实例变量 self.dataset_revision

        self.output_retrieved = output_retrieved
        # 将 output_retrieved 参数赋值给实例变量 self.output_retrieved

        self.do_deduplication = do_deduplication
        # 将 do_deduplication 参数赋值给实例变量 self.do_deduplication

        self.use_cache = use_cache
        # 将 use_cache 参数赋值给实例变量 self.use_cache

        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, "forced_eos_token_id", None)
        # 如果实例变量 forced_eos_token_id 为 None，则尝试从 generator 中获取 "forced_eos_token_id" 并赋值给它

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs
    ) -> PretrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)
        # 使用 question_encoder_config 和 generator_config 的字典形式创建一个 EncoderDecoderConfig 实例，并返回
```