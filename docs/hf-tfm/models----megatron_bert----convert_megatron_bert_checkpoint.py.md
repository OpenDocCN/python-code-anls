# `.\models\megatron_bert\convert_megatron_bert_checkpoint.py`

```
# 引入 argparse 库用于解析命令行参数
import argparse
# 引入 os 库用于与操作系统交互
import os
# 引入 re 库用于正则表达式操作
import re
# 引入 zipfile 库用于 ZIP 文件操作
import zipfile

# 引入 torch 库
import torch

# 从 transformers 库中引入 MegatronBertConfig 类
from transformers import MegatronBertConfig


def recursive_print(name, val, spaces=0):
    # 递归打印字典或者 Tensor 的内容
    # 格式化消息
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # 打印消息并递归打印（如果需要）
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # 重新排列 param 张量的布局为 [num_splits * num_heads * hidden_size, :]
    # 以便与后续版本的 NVIDIA Megatron-LM 兼容
    # 在 Megatron-LM 内部执行逆操作以读取检查点：
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # 如果 param 是 self-attention 块的权重张量，则返回的张量还需要再次转置才能被 HuggingFace BERT 读取
    input_shape = param.size()
    # 如果版本号为 1.0：
    if checkpoint_version == 1.0:
        # 版本 1.0 存储形状为 [num_heads * hidden_size * num_splits, :] 的参数
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        # 调整参数的形状为 saved_shape
        param = param.view(*saved_shape)
        # 将维度 0 和 2 进行转置
        param = param.transpose(0, 2)
        # 将维度 1 和 2 进行转置并保证内存连续性
        param = param.transpose(1, 2).contiguous()
    # 如果版本号大于或等于 2.0：
    elif checkpoint_version >= 2.0:
        # 其他版本存储形状为 [num_heads * num_splits * hidden_size, :] 的参数
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        # 调整参数的形状为 saved_shape
        param = param.view(*saved_shape)
        # 将维度 0 和 1 进行转置并保证内存连续性
        param = param.transpose(0, 1).contiguous()
    # 最终将参数的形状调整为 input_shape
    param = param.view(*input_shape)
    # 返回调整形状后的参数
    return param
# 定义一个函数，用于转换 Megatron-LM 的检查点到适用于 Transformers 框架的格式
def convert_megatron_checkpoint(args, input_state_dict, config):
    # 输出的模型状态字典
    output_state_dict = {}

    # 旧版本可能没有存储训练参数
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # 如果存在训练参数，将其配置信息应用到转换后的配置中
        config.tokenizer_type = ds_args.tokenizer_type
        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        config.intermediate_size = ds_args.ffn_hidden_size if "ffn_hidden_size" in ds_args else 4 * ds_args.hidden_size

    # 注意力头的数量
    heads = config.num_attention_heads
    # 每个注意力头的隐藏大小
    hidden_size_per_head = config.hidden_size // heads
    # Megatron-LM 的检查点版本
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # 模型
    model = input_state_dict["model"]
    # 语言模型
    lm = model["language_model"]
    # 嵌入层
    embeddings = lm["embedding"]

    # 词嵌入
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # 截断嵌入表到指定的词汇表大小
    word_embeddings = word_embeddings[: config.vocab_size, :]
    # 存储词嵌入
    output_state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings

    # 位置嵌入
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    assert pos_embeddings.size(0) == config.max_position_embeddings and pos_embeddings.size(1) == config.hidden_size
    # 存储位置嵌入
    output_state_dict["bert.embeddings.position_embeddings.weight"] = pos_embeddings

    # 类型嵌入
    tokentype_embeddings = embeddings["tokentype_embeddings"]["weight"]
    # 存储类型嵌入
    output_state_dict["bert.embeddings.token_type_embeddings.weight"] = tokentype_embeddings

    # Transformer 模块
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # 用于提取层名称的正则表达式
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # Megatron-LM 到 Transformers 的简单名称映射
    megatron_to_transformers = {
        "attention.dense": ".attention.output.dense.",
        "self_attention.dense": ".attention.output.dense.",
        "mlp.dense_h_to_4h": ".intermediate.dense.",
        "mlp.dense_4h_to_h": ".output.dense.",
    }
    # 跟踪注意力/查询/值张量的变量，初始设为None
    attention_qkv_weight = None

    # 提取模型的各层参数并存储到输出状态字典中

    # 存储最终的层归一化权重
    output_state_dict["bert.encoder.ln.weight"] = transformer["final_layernorm.weight"]
    # 存储最终的层归一化偏置
    output_state_dict["bert.encoder.ln.bias"] = transformer["final_layernorm.bias"]

    # 提取并存储池化器的权重和偏置
    pooler = lm["pooler"]
    output_state_dict["bert.pooler.dense.weight"] = pooler["dense.weight"]
    output_state_dict["bert.pooler.dense.bias"] = pooler["dense.bias"]

    # 从 Megatron 的语言模型头部提取并存储 LM 头部的权重和偏置

    # 提取转换矩阵的权重
    output_state_dict["cls.predictions.transform.dense.weight"] = lm_head["dense.weight"]
    # 提取转换矩阵的偏置
    output_state_dict["cls.predictions.transform.dense.bias"] = lm_head["dense.bias"]

    # 提取转换层归一化的权重
    output_state_dict["cls.predictions.transform.LayerNorm.weight"] = lm_head["layernorm.weight"]
    # 提取转换层归一化的偏置
    output_state_dict["cls.predictions.transform.LayerNorm.bias"] = lm_head["layernorm.bias"]

    # 对于解码器，复制词嵌入的权重并存储到输出状态字典中
    output_state_dict["cls.predictions.decoder.weight"] = word_embeddings
    # 存储 LM 头部的偏置
    output_state_dict["cls.predictions.bias"] = lm_head["bias"]

    # 从 Megatron 的二元分类器提取并存储分类器的权重和偏置

    # 存储序列关系分类器的权重
    output_state_dict["cls.seq_relationship.weight"] = binary_head["weight"]
    # 存储序列关系分类器的偏置
    output_state_dict["cls.seq_relationship.bias"] = binary_head["bias"]

    # 返回最终的输出状态字典
    return output_state_dict
# 定义程序的主函数
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加用于打印检查点结构的参数
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    # 添加指向包含检查点的 ZIP 文件路径的参数
    parser.add_argument("path_to_checkpoint", type=str, help="Path to the ZIP file containing the checkpoint")
    # 添加可选的配置文件参数，描述预训练模型的配置
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 提取路径的基本名称部分
    basename = os.path.dirname(args.path_to_checkpoint)

    # 加载模型
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
    # 如果路径以 .zip 结尾，则使用 zipfile 模块解压缩
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            # 使用 zipfile 中的文件打开函数获取 PyTorch 状态字典
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        # 否则直接加载 PyTorch 状态字典
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    # 根据配置文件是否为空，选择相应的 MegatronBertConfig
    if args.config_file == "":
        # 默认使用 Megatron-BERT 345m 的配置
        config = MegatronBertConfig()
        # 根据输入状态字典调整词汇表大小
        config.vocab_size = input_state_dict["model"]["lm_head"]["bias"].numel()
    else:
        # 从 JSON 文件加载 MegatronBertConfig
        config = MegatronBertConfig.from_json_file(args.config_file)

    # 执行转换
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # 如果需要打印检查点结构，则递归打印输出状态字典
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # 将配置保存到文件中
    print("Saving config")
    config.save_pretrained(basename)

    # 将输出的状态字典保存到文件中
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


if __name__ == "__main__":
    # 如果是直接执行本脚本，则调用主函数
    main()
```