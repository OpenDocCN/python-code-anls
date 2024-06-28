# `.\models\xmod\convert_xmod_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明及许可信息
# 引入 argparse 用于命令行参数解析
# 从 pathlib 模块中引入 Path 类
# 引入 fairseq 库
# 引入 torch 库
# 从 fairseq 的 xmod 模块中引入 XMODModel 类别名为 FairseqXmodModel
# 从 packaging 模块中引入 version 函数
# 从 transformers 库中引入 XmodConfig, XmodForMaskedLM, XmodForSequenceClassification 类
# 从 transformers.utils 中引入 logging 模块

if version.parse(fairseq.__version__) < version.parse("0.12.2"):
    # 如果 fairseq 版本小于 0.12.2，则抛出异常
    raise Exception("requires fairseq >= 0.12.2")
if version.parse(fairseq.__version__) > version.parse("2"):
    # 如果 fairseq 版本大于 2，则抛出异常
    raise Exception("requires fairseq < v2")

# 设置日志输出等级为 INFO
logging.set_verbosity_info()
# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 示例文本
SAMPLE_TEXT = "Hello, World!"
# 示例语言标识
SAMPLE_LANGUAGE = "en_XX"

def convert_xmod_checkpoint_to_pytorch(
    xmod_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    # 数据目录
    data_dir = Path("data_bin")
    # 从预训练模型路径加载 FairseqXmodModel 模型
    xmod = FairseqXmodModel.from_pretrained(
        model_name_or_path=str(Path(xmod_checkpoint_path).parent),
        checkpoint_file=Path(xmod_checkpoint_path).name,
        _name="xmod_base",
        arch="xmod_base",
        task="multilingual_masked_lm",
        data_name_or_path=str(data_dir),
        bpe="sentencepiece",
        sentencepiece_model=str(Path(xmod_checkpoint_path).parent / "sentencepiece.bpe.model"),
        src_dict=str(data_dir / "dict.txt"),
    )
    # 设置模型为评估模式，禁用 dropout
    xmod.eval()
    # 打印模型信息
    print(xmod)

    # 获取 xmod 模型的句子编码器
    xmod_sent_encoder = xmod.model.encoder.sentence_encoder
    # 根据 xmod 模型的配置创建 XmodConfig 对象
    config = XmodConfig(
        vocab_size=xmod_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=xmod.cfg.model.encoder_embed_dim,
        num_hidden_layers=xmod.cfg.model.encoder_layers,
        num_attention_heads=xmod.cfg.model.encoder_attention_heads,
        intermediate_size=xmod.cfg.model.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch 默认值，与 fairseq 兼容
        pre_norm=xmod.cfg.model.encoder_normalize_before,
        adapter_reduction_factor=getattr(xmod.cfg.model, "bottleneck", 2),
        adapter_layer_norm=xmod.cfg.model.adapter_layer_norm,
        adapter_reuse_layer_norm=xmod.cfg.model.adapter_reuse_layer_norm,
        ln_before_adapter=xmod.cfg.model.ln_before_adapter,
        languages=xmod.cfg.model.languages,
    )
    # 如果需要分类头部，则设置配置对象的标签数量为模型特定分类头的输出权重行数
    if classification_head:
        config.num_labels = xmod.model.classification_heads["mnli"].out_proj.weight.shape[0]
    # 打印 X-MOD 的配置信息
    print("Our X-MOD config:", config)

    # 根据是否有分类头选择模型类型，并设置为评估模式
    model = XmodForSequenceClassification(config) if classification_head else XmodForMaskedLM(config)
    model.eval()

    # 复制所有权重
    # 嵌入层权重
    model.roberta.embeddings.word_embeddings.weight = xmod_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = xmod_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # 将其置零因为 xmod 不使用它们

    model.roberta.embeddings.LayerNorm.weight = xmod_sent_encoder.layernorm_embedding.weight
    model.roberta.embeddings.LayerNorm.bias = xmod_sent_encoder.layernorm_embedding.bias

    # 如果存在层归一化，则复制编码器层归一化的权重和偏置
    if xmod_sent_encoder.layer_norm is not None:
        model.roberta.encoder.LayerNorm.weight = xmod_sent_encoder.layer_norm.weight
        model.roberta.encoder.LayerNorm.bias = xmod_sent_encoder.layer_norm.bias

    # 如果是分类头，复制分类器的权重和偏置
    if classification_head:
        model.classifier.dense.weight = xmod.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = xmod.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = xmod.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = xmod.model.classification_heads["mnli"].out_proj.bias
    else:
        # 如果是语言模型头，复制语言模型头的权重和偏置
        model.lm_head.dense.weight = xmod.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = xmod.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = xmod.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = xmod.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = xmod.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = xmod.model.encoder.lm_head.bias

    # 检查模型输出是否一致
    input_ids = xmod.encode(SAMPLE_TEXT).unsqueeze(0)  # 批量大小为 1
    model.roberta.set_default_language(SAMPLE_LANGUAGE)

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = xmod.model.classification_heads["mnli"](xmod.extract_features(input_ids))
    else:
        their_output = xmod.model(input_ids, lang_id=[SAMPLE_LANGUAGE])[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 约为 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    # 创建目录以保存 PyTorch 模型
    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本作为主程序执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--xmod_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    # 添加一个参数，指定官方 PyTorch 模型的路径，类型为字符串，必选项

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加一个参数，指定输出 PyTorch 模型的文件夹路径，类型为字符串，必选项

    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 添加一个参数，表示是否要转换最终的分类头部，这是一个布尔值参数

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，将 xmod 模型转换为 PyTorch 模型
    convert_xmod_checkpoint_to_pytorch(
        args.xmod_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
```