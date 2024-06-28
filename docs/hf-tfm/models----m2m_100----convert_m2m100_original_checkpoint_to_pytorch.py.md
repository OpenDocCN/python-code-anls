# `.\models\m2m_100\convert_m2m100_original_checkpoint_to_pytorch.py`

```
# 导入命令行参数解析库
import argparse

# 导入PyTorch库
import torch
from torch import nn

# 导入transformers库中的M2M100Config和M2M100ForConditionalGeneration类
from transformers import M2M100Config, M2M100ForConditionalGeneration


# 定义函数，用于移除状态字典中指定的键
def remove_ignore_keys_(state_dict):
    # 要移除的键列表
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
    # 逐个移除指定键
    for k in ignore_keys:
        state_dict.pop(k, None)


# 定义函数，从给定的嵌入层创建一个线性层
def make_linear_from_emb(emb):
    # 获取嵌入层的词汇量大小和嵌入维度大小
    vocab_size, emb_size = emb.weight.shape
    # 创建一个无偏置的线性层
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重设置为嵌入层的权重
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# 定义函数，从Fairseq的M2M100模型检查点文件中转换为transformers的M2M100模型
def convert_fairseq_m2m100_checkpoint_from_disk(checkpoint_path):
    # 从硬盘加载Fairseq的M2M100模型
    m2m_100 = torch.load(checkpoint_path, map_location="cpu")
    # 获取模型参数
    args = m2m_100["args"] or m2m_100["cfg"]["model"]
    # 获取模型状态字典
    state_dict = m2m_100["model"]
    # 移除状态字典中不需要的键
    remove_ignore_keys_(state_dict)
    # 获取词汇量大小
    vocab_size = state_dict["encoder.embed_tokens.weight"].shape[0]

    # 根据Fairseq的参数创建transformers的配置对象
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

    # 调整状态字典以适应transformers的模型结构
    state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
    # 创建M2M100ForConditionalGeneration模型
    model = M2M100ForConditionalGeneration(config)
    # 加载模型的状态字典（允许部分严格性）
    model.model.load_state_dict(state_dict, strict=False)
    # 将语言模型头部设置为从嵌入层创建的线性层
    model.lm_head = make_linear_from_emb(model.model.shared)

    return model


# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需的命令行参数：fairseq模型检查点文件的路径
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    # 添加可选的命令行参数：输出PyTorch模型的文件夹路径
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数，获取用户输入的参数值
    args = parser.parse_args()
    
    # 调用函数 convert_fairseq_m2m100_checkpoint_from_disk，从磁盘中加载 Fairseq M2M100 模型的检查点
    model = convert_fairseq_m2m100_checkpoint_from_disk(args.fairseq_path)
    
    # 将转换后的 PyTorch 模型保存到指定的文件夹路径 args.pytorch_dump_folder_path
    model.save_pretrained(args.pytorch_dump_folder_path)
```