# `.\models\reformer\convert_reformer_trax_checkpoint_to_pytorch.py`

```
# 导入必要的库和模块
import argparse  # 用于解析命令行参数
import pickle    # 用于序列化和反序列化 Python 对象

import numpy as np   # 导入 NumPy 库，用于处理数组
import torch         # 导入 PyTorch 库
from torch import nn  # 导入 PyTorch 的神经网络模块

from transformers import ReformerConfig, ReformerModelWithLMHead  # 导入 transformers 库中的 Reformer 模型相关类
from transformers.utils import logging   # 导入 logging 模块，用于日志记录

logging.set_verbosity_info()  # 设置日志记录级别为 info

def set_param(torch_layer, weight, bias=None):
    # 设置一个神经网络层的参数
    assert torch_layer.weight.shape == weight.shape, f"{torch_layer} layer.weight does not match"
    # 将给定的权重设为神经网络层的权重参数
    torch_layer.weight = nn.Parameter(weight)
    if bias is not None:
        assert torch_layer.bias.shape == bias.shape, f"{torch_layer} layer.bias does not match"
        # 将给定的偏置设为神经网络层的偏置参数
        torch_layer.bias = nn.Parameter(bias)

def set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    # 设置 Torch 中 LSH（Locality-Sensitive Hashing）层的权重
    np_query_key = np.asarray(weights[0])
    np_value = np.asarray(weights[1])
    np_dense = np.asarray(weights[2])

    # 设置自注意力机制中 query_key 的权重参数
    set_param(
        torch_layer.self_attention.query_key,
        torch.tensor(np_query_key).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    # 设置自注意力机制中 value 的权重参数
    set_param(
        torch_layer.self_attention.value,
        torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    # 设置输出层的密集连接层（dense layer）的权重参数
    set_param(
        torch_layer.output.dense,
        torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
    )

def set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    # 设置 Torch 中 Local 层的权重
    np_query = np.asarray(weights[0])
    np_key = np.asarray(weights[1])
    np_value = np.asarray(weights[2])
    np_dense = np.asarray(weights[3])

    # 设置自注意力机制中 query 的权重参数
    set_param(
        torch_layer.self_attention.query,
        torch.tensor(np_query).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    # 设置自注意力机制中 key 的权重参数
    set_param(
        torch_layer.self_attention.key,
        torch.tensor(np_key).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    # 设置自注意力机制中 value 的权重参数
    set_param(
        torch_layer.self_attention.value,
        torch.tensor(np_value).transpose(1, 2).contiguous().view(-1, hidden_size),
    )
    # 设置输出层的密集连接层（dense layer）的权重参数
    set_param(
        torch_layer.output.dense,
        torch.tensor(np_dense).view(-1, hidden_size).contiguous().transpose(0, 1),
    )

def set_block_weights_in_torch(weights, torch_block, hidden_size):
    # 设置 Torch 中的块（block）的权重
    # layernorm 1
    layer_norm_1 = weights[0][0][0]
    layer_norm_1_weight = np.asarray(layer_norm_1[0])
    # 将 layer_norm_1 的偏置项转换为 NumPy 数组
    layer_norm_1_bias = np.asarray(layer_norm_1[1])
    
    # 设置注意力层的参数，包括层归一化的权重和偏置
    set_param(
        torch_block.attention.layer_norm,  # 设置注意力层的层归一化
        torch.tensor(layer_norm_1_weight),  # 转换为 PyTorch 张量并设置层归一化的权重
        torch.tensor(layer_norm_1_bias),    # 转换为 PyTorch 张量并设置层归一化的偏置
    )
    
    # 获取注意力权重
    attn_weights = weights[0][1]
    
    # 根据注意力权重的长度选择设置分片式或局部的注意力层权重
    if len(attn_weights) < 4:
        set_layer_weights_in_torch_lsh(attn_weights, torch_block.attention, hidden_size)
    else:
        set_layer_weights_in_torch_local(attn_weights, torch_block.attention, hidden_size)
    
    # 获取中间权重
    intermediate_weights = weights[2][0][1][2]
    
    # 如果中间权重长度为 4，则选择其中的第三个权重作为 Chunked Feed Forward 的权重
    if len(intermediate_weights) == 4:
        intermediate_weights = intermediate_weights[2]
    
    # 设置第二个层归一化的权重和偏置
    layer_norm_2_weight = np.asarray(intermediate_weights[0][0])
    layer_norm_2_bias = np.asarray(intermediate_weights[0][1])
    set_param(
        torch_block.feed_forward.layer_norm,  # 设置前馈层的层归一化
        torch.tensor(layer_norm_2_weight),    # 转换为 PyTorch 张量并设置层归一化的权重
        torch.tensor(layer_norm_2_bias),      # 转换为 PyTorch 张量并设置层归一化的偏置
    )
    
    # 设置中间密集层的权重和偏置
    inter_dense_weight = np.asarray(intermediate_weights[1][0])
    inter_dense_bias = np.asarray(intermediate_weights[1][1])
    set_param(
        torch_block.feed_forward.dense.dense,  # 设置前馈层的密集层权重
        torch.tensor(inter_dense_weight).transpose(0, 1).contiguous(),  # 转换为 PyTorch 张量并设置密集层的权重
        torch.tensor(inter_dense_bias),        # 转换为 PyTorch 张量并设置密集层的偏置
    )
    
    # 设置中间输出层的权重和偏置
    out_dense_weight = np.asarray(intermediate_weights[4][0])
    out_dense_bias = np.asarray(intermediate_weights[4][1])
    set_param(
        torch_block.feed_forward.output.dense,  # 设置前馈层的输出层权重
        torch.tensor(out_dense_weight).transpose(0, 1).contiguous(),  # 转换为 PyTorch 张量并设置输出层的权重
        torch.tensor(out_dense_bias),            # 转换为 PyTorch 张量并设置输出层的偏置
    )
# 将给定的权重设置到指定的 PyTorch 模型中
def set_model_weights_in_torch(weights, torch_model, hidden_size):
    # 获取 PyTorch 模型中的 reformer 部分
    torch_model_reformer = torch_model.reformer

    # 从权重中获取词嵌入
    word_embeddings = np.asarray(weights[1])
    # 设置词嵌入参数
    set_param(
        torch_model_reformer.embeddings.word_embeddings,
        torch.tensor(word_embeddings),
    )

    # 如果权重的第 3 项是元组
    if isinstance(weights[3], tuple):
        # 获取位置嵌入
        position_embeddings = torch_model_reformer.embeddings.position_embeddings
        # 遍历位置嵌入的权重
        for emb_idx in range(len(position_embeddings.weights)):
            emb_weights = np.asarray(weights[3][emb_idx][0])
            # 断言确保位置嵌入的形状匹配
            assert (
                position_embeddings.weights[emb_idx].shape == emb_weights.shape
            ), f"{position_embeddings[emb_idx]} emb does not match"
            # 设置位置嵌入参数为可训练的 Tensor
            position_embeddings.weights[emb_idx] = nn.Parameter(torch.tensor(emb_weights))

    # 获取 Trax 模型的层权重
    trax_layer_weights = weights[5]
    # 断言确保编码器层的数量匹配
    assert len(torch_model_reformer.encoder.layers) * 4 == len(
        trax_layer_weights
    ), "HF and trax model do not have the same number of layers"
    # 遍历编码器的每一层并设置权重
    for layer_idx, layer in enumerate(torch_model_reformer.encoder.layers):
        block_weights = trax_layer_weights[4 * layer_idx : 4 * (layer_idx + 1)]
        set_block_weights_in_torch(block_weights, layer, hidden_size)

    # 设置输出层的 LayerNorm 参数
    layer_norm_out_weight = np.asarray(weights[7][0])
    layer_norm_out_bias = np.asarray(weights[7][1])
    set_param(
        torch_model_reformer.encoder.layer_norm,
        torch.tensor(layer_norm_out_weight),
        torch.tensor(layer_norm_out_bias),
    )

    # 设置输出嵌入层的参数
    output_embed_weights = np.asarray(weights[9][0])
    output_embed_bias = np.asarray(weights[9][1])
    set_param(
        torch_model.lm_head.decoder,
        torch.tensor(output_embed_weights).transpose(0, 1).contiguous(),
        torch.tensor(output_embed_bias),
    )


# 将 Trax 的检查点文件转换为 PyTorch 模型并保存
def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path):
    # 从配置文件中加载 Reformer 模型配置
    config = ReformerConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 根据配置创建 PyTorch 模型
    model = ReformerModelWithLMHead(config)

    # 从 Trax 检查点文件中加载权重
    with open(trax_model_pkl_path, "rb") as f:
        model_weights = pickle.load(f)["weights"]

    # 将加载的权重设置到 PyTorch 模型中
    set_model_weights_in_torch(model_weights, model, config.hidden_size)

    # 保存转换后的 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--trax_model_pkl_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained Reformer model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加一个命令行参数，用于指定输出的 PyTorch 模型的路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()
    # 调用函数 convert_trax_checkpoint_to_pytorch，将转换 Trax 模型为 PyTorch 模型
    # 使用 args 对象中的 trax_model_pkl_path（Trax 模型路径）、config_file（配置文件路径）和 pytorch_dump_path（输出 PyTorch 模型路径）作为参数
    convert_trax_checkpoint_to_pytorch(args.trax_model_pkl_path, args.config_file, args.pytorch_dump_path)
```