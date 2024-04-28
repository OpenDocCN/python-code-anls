# `.\transformers\convert_tf_hub_seq_to_seq_bert_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"原样"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""将 Seq2Seq TF Hub 检查点转换为 PyTorch 模型。"""

# 导入必要的库
import argparse

from . import (
    BertConfig,
    BertGenerationConfig,
    BertGenerationDecoder,
    BertGenerationEncoder,
    load_tf_weights_in_bert_generation,
    logging,
)

# 设置日志级别为 info
logging.set_verbosity_info()

# 将 TF 检查点转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path, is_encoder_named_decoder, vocab_size, is_encoder):
    # 初始化 PyTorch 模型
    bert_config = BertConfig.from_pretrained(
        "bert-large-cased",
        vocab_size=vocab_size,
        max_position_embeddings=512,
        is_decoder=True,
        add_cross_attention=True,
    )
    bert_config_dict = bert_config.to_dict()
    del bert_config_dict["type_vocab_size"]
    config = BertGenerationConfig(**bert_config_dict)
    if is_encoder:
        model = BertGenerationEncoder(config)
    else:
        model = BertGenerationDecoder(config)
    print(f"Building PyTorch model from configuration: {config}")

    # 从 TF 检查点加载权重
    load_tf_weights_in_bert_generation(
        model,
        tf_hub_path,
        model_class="bert",
        is_encoder_named_decoder=is_encoder_named_decoder,
        is_encoder=is_encoder,
    )

    # 保存 PyTorch 模型
    print(f"Save PyTorch model and config to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

# 如果作为独立脚本运行，则解析命令行参数并调用转换函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--tf_hub_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_encoder_named_decoder",
        action="store_true",
        help="If decoder has to be renamed to encoder in PyTorch model.",
    )
    parser.add_argument("--is_encoder", action="store_true", help="If model is an encoder.")
    parser.add_argument("--vocab_size", default=50358, type=int, help="Vocab size of model")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_hub_path,
        args.pytorch_dump_path,
        args.is_encoder_named_decoder,
        args.vocab_size,
        is_encoder=args.is_encoder,
    )
```