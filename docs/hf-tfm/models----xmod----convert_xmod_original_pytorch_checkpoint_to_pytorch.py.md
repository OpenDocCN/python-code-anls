# `.\transformers\models\xmod\convert_xmod_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# 版权声明

"""Convert X-MOD checkpoint."""

# 导入需要的模块
import argparse
from pathlib import Path
import fairseq
import torch
from fairseq.models.xmod import XMODModel as FairseqXmodModel
from packaging import version
from transformers import XmodConfig, XmodForMaskedLM, XmodForSequenceClassification
from transformers.utils import logging

# 检查 fairseq 版本是否符合要求
if version.parse(fairseq.__version__) < version.parse("0.12.2"):
    raise Exception("requires fairseq >= 0.12.2")
if version.parse(fairseq.__version__) > version.parse("2"):
    raise Exception("requires fairseq < v2")

# 设置日志信息
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# 定义示例文本和语言
SAMPLE_TEXT = "Hello, World!"
SAMPLE_LANGUAGE = "en_XX"

# 将 X-MOD 检查点转换为 PyTorch 模型
def convert_xmod_checkpoint_to_pytorch(
    xmod_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool
):
    # 定义数据目录
    data_dir = Path("data_bin")
    # 从预训练模型中加载 X-MOD 模型
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
    # 关闭 dropout
    xmod.eval()
    print(xmod)

    # 获取 X-MOD 模型的句子编码器
    xmod_sent_encoder = xmod.model.encoder.sentence_encoder
    # 定义用于转换的配置
    config = XmodConfig(
        vocab_size=xmod_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=xmod.cfg.model.encoder_embed_dim,
        num_hidden_layers=xmod.cfg.model.encoder_layers,
        num_attention_heads=xmod.cfg.model.encoder_attention_heads,
        intermediate_size=xmod.cfg.model.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch 默认值
        pre_norm=xmod.cfg.model.encoder_normalize_before,
        adapter_reduction_factor=getattr(xmod.cfg.model, "bottleneck", 2),
        adapter_layer_norm=xmod.cfg.model.adapter_layer_norm,
        adapter_reuse_layer_norm=xmod.cfg.model.adapter_reuse_layer_norm,
        ln_before_adapter=xmod.cfg.model.ln_before_adapter,
        languages=xmod.cfg.model.languages,
    )
    # 如果有分类头，则设置标签数量
    if classification_head:
        config.num_labels = xmod.model.classification_heads["mnli"].out_proj.weight.shape[0]
    # 输出配置信息
    print("Our X-MOD config:", config)

    # 根据是否是分类任务选择不同的模型
    model = XmodForSequenceClassification(config) if classification_head else XmodForMaskedLM(config)
    model.eval()

    # 复制所有权重
    # 嵌入层权重
    model.roberta.embeddings.word_embeddings.weight = xmod_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = xmod_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # 将其归零，因为 xmod 不使用它们

    model.roberta.embeddings.LayerNorm.weight = xmod_sent_encoder.layernorm_embedding.weight
    model.roberta.embeddings.LayerNorm.bias = xmod_sent_encoder.layernorm_embedding.bias

    if xmod_sent_encoder.layer_norm is not None:
        model.roberta.encoder.LayerNorm.weight = xmod_sent_encoder.layer_norm.weight
        model.roberta.encoder.LayerNorm.bias = xmod_sent_encoder.layer_norm.bias

    if classification_head:
        # 分类头权重
        model.classifier.dense.weight = xmod.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = xmod.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = xmod.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = xmod.model.classification_heads["mnli"].out_proj.bias
    else:
        # 语言模型头权重
        model.lm_head.dense.weight = xmod.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = xmod.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = xmod.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = xmod.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = xmod.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = xmod.model.encoder.lm_head.bias

    # 检验两个模型输出是否相同
    input_ids = xmod.encode(SAMPLE_TEXT).unsqueeze(0)  # 批次大小为 1
    model.roberta.set_default_language(SAMPLE_LANGUAGE)

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = xmod.model.classification_heads["mnli"](xmod.extract_features(input_ids))
    else:
        their_output = xmod.model(input_ids, lang_id=[SAMPLE_LANGUAGE])[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # 大约为 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    # 创建存储路径
    Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--xmod_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--classification_head", action="store_true", help="Whether to convert a final classification head."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将XMOD检查点转换为PyTorch格式
    convert_xmod_checkpoint_to_pytorch(
        args.xmod_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head
    )
```