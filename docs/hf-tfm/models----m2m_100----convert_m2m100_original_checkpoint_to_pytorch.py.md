# `.\transformers\models\m2m_100\convert_m2m100_original_checkpoint_to_pytorch.py`

```
# 导入所需的库
import argparse
import torch
from torch import nn
from transformers import M2M100Config, M2M100ForConditionalGeneration

# 定义一个函数，用于移除特定的键值对
def remove_ignore_keys_(state_dict):
    # 定义要移除的键的列表
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "encoder.embed_positions._float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    # 遍历要移除的键，如果存在则从 state_dict 中删除
    for k in ignore_keys:
        state_dict.pop(k, None)

# 定义一个函数，根据给定的嵌入层创建一个线性层
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    # 创建一个线性层，权重初始化为给定嵌入层的权重
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 定义一个函数，用于从磁盘中的 Fairseq M2M100 检查点文件转换模型
def convert_fairseq_m2m100_checkpoint_from_disk(checkpoint_path):
    # 加载 Fairseq 模型的检查点文件
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")
    args = m2m_100["args"] or m2m_100["cfg"]["model"]
    state_dict = m2m_100["model"]
    # 移除特定的键
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    # 创建 M2M100Config 对象，指定模型的参数
    config = M2M100Config(
        vocab_size=vocab_size,
        max_position_embeddings=1024,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        encoder_attention_heads=args.encoder_attention_heads,
        decoder_attention_heads=args.decoder_attention_heads,
        encoder_ffn_dim=args.encoder_ffn_embed_dim,
        decoder_ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.encoder_embed_dim,
        encoder_layerdrop=args.encoder_layerdrop,
        decoder_layerdrop=args.decoder_layerdrop,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="relu",
    )

    # 为共享的权重分配解码器的嵌入层权重
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    # 创建 M2M100ForConditionalGeneration 模型
    model = M2M100ForConditionalGeneration(config)
    # 加载模型参数
    model.model.load_state_dict(state_dict, strict=False)
    model.lm_head = make_linear_from_emb(model.model.shared)

    return model

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加必要参数
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数，返回解析后的参数对象
    args = parser.parse_args()
    # 从指定路径加载并转换 fairseq m2m-100 模型检查点，返回模型实例
    model = convert_fairseq_m2m100_checkpoint_from_disk(args.fairseq_pathß)
    # 将模型保存到指定的路径
    model.save_pretrained(args.pytorch_dump_folder_path)
```