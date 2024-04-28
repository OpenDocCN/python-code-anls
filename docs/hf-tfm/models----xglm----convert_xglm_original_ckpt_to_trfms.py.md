# `.\transformers\models\xglm\convert_xglm_original_ckpt_to_trfms.py`

```
# 导入必要的库
import argparse
from argparse import Namespace

import torch
from torch import nn

from transformers import XGLMConfig, XGLMForCausalLM

# 定义函数去除不需要加载的参数
def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "decoder.version",
        "decoder.output_projection.weight",
        "_float_tensor",
        "decoder.embed_positions._float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 从一个嵌入矩阵（emb）中构建一个线性层
def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer

# 从磁盘读取 Fairseq-XGLM 模型的检查点并转换为 PyTorch 模型
def convert_fairseq_xglm_checkpoint_from_disk(checkpoint_path):
    # 从磁盘加载检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # 从检查点中获取模型参数
    args = Namespace(**checkpoint["cfg"]["model"])
    state_dict = checkpoint["model"]
    # 去除不需要加载的参数
    remove_ignore_keys_(state_dict)
    vocab_size = state_dict["decoder.embed_tokens.weight"].shape[0]
    
    # 替换字典中的键名，将 "decoder" 替换为 "model"
    state_dict = {key.replace("decoder", "model"): val for key, val in state_dict.items()}

    # 配置 XGLM 模型的参数
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

    # 构建 XGLM 模型
    model = XGLMForCausalLM(config)
    # 加载模型参数
    missing = model.load_state_dict(state_dict, strict=False)
    print(missing)
    # 构建线性层，使用嵌入矩阵中的数据
    model.lm_head = make_linear_from_emb(model.model.embed_tokens)

    return model

# 当作为单独文件运行时的入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument("fairseq_path", type=str, help="path to a model.pt on local filesystem.")
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    args = parser.parse_args()
    # 转换 Fairseq-XGLM 检查点并保存为 PyTorch 模型
    model = convert_fairseq_xglm_checkpoint_from_disk(args.fairseq_path)
    model.save_pretrained(args.pytorch_dump_folder_path)
```