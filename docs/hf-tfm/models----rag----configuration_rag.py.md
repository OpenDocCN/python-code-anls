# `.\transformers\models\rag\configuration_rag.py`

```py
# 设置文件编码为utf-8
# 版权声明，版权属于 RAG 作者和 HuggingFace 公司团队
# 基于 Apache 许可证 2.0 版本发布，除许可证规定外不得使用此文件
# 可以在以下链接获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律要求或书面同意，有条件使用此软件
# 分发给任何人使用"原样"的基础，无论是明示还是暗示的任何形式的保证和条件
# 请参考授权许可，以获取更多有关控制模型输出的信息

# 导入预训练配置和工具类
from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings

# RAG_CONFIG_DOC 文档字符串
# 存储 *RagModel* 的配置
# 配置对象继承自 PretrainedConfig，可用于控制模型输出，阅读 PretrainedConfig 文档获取更多信息

# 使用 RAG_CONFIG_DOC 文档字符串装饰 RagConfig 类
class RagConfig(PretrainedConfig):
    # 模型类型为rag
    model_type = "rag"
    # 是一个组合模型

    # 初始化函数
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
        **kwargs,  # 允许用户在配置中传递额外的关键字参数
    # 调用父类的初始化方法
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
        # 断言是否包含"question_encoder"和"generator"，若不包含则抛出异常
        assert (
            "question_encoder" in kwargs and "generator" in kwargs
        ), "Config has to be initialized with question_encoder and generator config"
        # 弹出"question_encoder"和"generator"并获取其配置信息
        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        # 导入自动配置模块，并根据模型类型和配置信息创建question_encoder和generator对象
        from ..auto.configuration_auto import AutoConfig

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        # 设置一些属性
        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize
        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.output_retrieved = output_retrieved
        self.do_deduplication = do_deduplication
        self.use_cache = use_cache

        # 若forced_eos_token_id为空，则获取generator的forced_eos_token_id
        if self.forced_eos_token_id is None:
            self.forced_eos_token_id = getattr(self.generator, "forced_eos_token_id", None)

    # 类方法，根据question_encoder_config和generator_config创建配置实例
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
```