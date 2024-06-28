# `.\models\xglm\convert_xglm_original_ckpt_to_trfms.py`

```py
import argparse                            # 导入argparse库，用于处理命令行参数
from argparse import Namespace             # 导入Namespace类，用于创建命名空间

import torch                               # 导入PyTorch库
from torch import nn                       # 导入神经网络模块

from transformers import XGLMConfig, XGLMForCausalLM   # 导入transformers库中的XGLMConfig和XGLMForCausalLM类


def remove_ignore_keys_(state_dict):
    # 定义函数，用于从状态字典中移除特定的键
    ignore_keys = [
        "decoder.version",                             # 忽略的键1
        "decoder.output_projection.weight",             # 忽略的键2
        "_float_tensor",                               # 忽略的键3
        "decoder.embed_positions._float_tensor",       # 忽略的键4
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    # 定义函数，从给定的嵌入矩阵创建一个线性层
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)   # 创建线性层，无偏置
    lin_layer.weight.data = emb.weight.data                   # 将权重数据设置为输入嵌入的权重数据
    return lin_layer


def convert_fairseq_xglm_checkpoint_from_disk(checkpoint_path):
    # 定义函数，从Fairseq的检查点文件中加载模型并转换为XGLM模型

    # 加载检查点文件
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    args = Namespace(**checkpoint["cfg"]["model"])   # 从检查点中读取模型参数并创建命名空间对象
    state_dict = checkpoint["model"]                 # 从检查点中读取模型的状态字典
    remove_ignore_keys_(state_dict)                  # 调用函数移除状态字典中的特定键

    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]   # 获取词汇表大小

    # 重命名状态字典中的键，将"decoder"替换为"model"
    state_dict = {key.replace("decoder", "model"): val for key, val in state_dict.items()}

    # 根据配置创建XGLMConfig对象
    config = XGLMConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.max_target_positions,
        num_layers=args.decoder_layers,
        attention_heads=args.decoder_attention_heads,
        ffn_dim=args.decoder_ffn_embed_dim,
        d_model=args.decoder_embed_dim,
        layerdrop=args.decoder_layerdrop,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        activation_dropout=args.activation_dropout,
        activation_function="gelu",
        scale_embedding=not args.no_scale_embedding,
        tie_word_embeddings=args.share_decoder_input_output_embed,
    )

    model = XGLMForCausalLM(config)     # 创建XGLM模型对象
    missing = model.load_state_dict(state_dict, strict=False)   # 加载状态字典到模型，允许不严格匹配
    print(missing)                     # 打印加载时缺失的键信息
    model.lm_head = make_linear_from_emb(model.model.embed_tokens)   # 根据嵌入矩阵创建线性头部

    return model                        # 返回转换后的XGLM模型对象


if __name__ == "__main__":
    parser = argparse.ArgumentParser()   # 创建参数解析器对象
    # 添加必需的命令行参数
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()           # 解析命令行参数
    model = convert_fairseq_xglm_checkpoint_from_disk(args.fairseq_path)   # 转换Fairseq检查点文件为XGLM模型
    model.save_pretrained(args.pytorch_dump_folder_path)   # 将模型保存到指定路径
```