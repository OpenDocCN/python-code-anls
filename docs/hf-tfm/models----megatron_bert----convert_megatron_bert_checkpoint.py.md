# `.\transformers\models\megatron_bert\convert_megatron_bert_checkpoint.py`

```
# 版权声明和许可信息
# 版权所有（c）2021年，NVIDIA公司。保留所有权利。
# 根据Apache许可证2.0版（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言的权限和限制。

# 如果在运行此转换脚本时出现异常：
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# 您需要告诉Python在哪里找到Megatron-LM的克隆版本，例如：
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_bert/convert_megatron_bert_checkpoint.py ...
#
# 如果您已经在其他地方克隆了它，请简单地调整到现有路径
#
# 如果训练是使用Megatron-LM的分支进行的，例如，
# https://github.com/microsoft/Megatron-DeepSpeed/，那么您可能需要将其添加到路径中，即，/path/to/Megatron-DeepSpeed/

# 导入所需的库
import argparse
import os
import re
import zipfile

import torch

from transformers import MegatronBertConfig

# 递归打印函数
def recursive_print(name, val, spaces=0):
    # 格式化消息
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # 打印并递归（如果需要）
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)

# 修复查询键值排序函数
def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # 重新排列参数张量的布局为[num_splits * num_heads * hidden_size, :]
    # 以便与后续版本的NVIDIA Megatron-LM兼容。
    # 在Megatron-LM内部执行反向操作以读取检查点：
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # 如果param是自注意力块的权重张量，则返回的张量
    # 将需要再次转置才能被HuggingFace BERT读取。
    input_shape = param.size()
    # 如果检查点版本为1.0：
    if checkpoint_version == 1.0:
        # 存储格式为 [num_heads * hidden_size * num_splits, :]，计算保存的形状
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        # 重新调整参数的形状
        param = param.view(*saved_shape)
        # 转置参数的维度0和2
        param = param.transpose(0, 2)
        # 再次转置参数的维度1和2，并确保内存连续性
        param = param.transpose(1, 2).contiguous()
    # 如果检查点版本大于等于2.0：
    elif checkpoint_version >= 2.0:
        # 存储格式为 [num_heads * num_splits * hidden_size, :]，计算保存的形状
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        # 重新调整参数的形状
        param = param.view(*saved_shape)
        # 转置参数的维度0和1，并确保内存连续性
        param = param.transpose(0, 1).contiguous()
    # 将参数重新调整为输入形状
    param = param.view(*input_shape)
    # 返回参数
    return param
####################################################################################################

# 将 Megatron-LM 检查点转换为 Transformers 模型检查点
def convert_megatron_checkpoint(args, input_state_dict, config):
    # 转换后的输出模型状态字典
    output_state_dict = {}

    # 旧版本未存储训练参数
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # 如果检查点中已包含确切的维度/大小信息，则无需让用户编写配置文件
        config.tokenizer_type = ds_args.tokenizer_type
        config.vocab_size = ds_args.padded_vocab_size
        config.max_position_embeddings = ds_args.max_position_embeddings
        config.hidden_size = ds_args.hidden_size
        config.num_hidden_layers = ds_args.num_layers
        config.num_attention_heads = ds_args.num_attention_heads
        config.intermediate_size = ds_args.ffn_hidden_size if "ffn_hidden_size" in ds_args else 4 * ds_args.hidden_size

    # 头部数量
    heads = config.num_attention_heads
    # 每个头部的隐藏大小
    hidden_size_per_head = config.hidden_size // heads
    # Megatron-LM 检查点版本
    if "checkpoint_version" in input_state_dict.keys():
        checkpoint_version = input_state_dict["checkpoint_version"]
    else:
        checkpoint_version = 0.0

    # 模型
    model = input_state_dict["model"]
    # 语言模型
    lm = model["language_model"]
    # 嵌入
    embeddings = lm["embedding"]

    # 单词嵌入
    word_embeddings = embeddings["word_embeddings"]["weight"]
    # 截断嵌入表至 vocab_size 行
    word_embeddings = word_embeddings[: config.vocab_size, :]
    # 存储单词嵌入
    output_state_dict["bert.embeddings.word_embeddings.weight"] = word_embeddings

    # 位置嵌入
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    assert pos_embeddings.size(0) == config.max_position_embeddings and pos_embeddings.size(1) == config.hidden_size
    # 存储位置嵌入
    output_state_dict["bert.embeddings.position_embeddings.weight"] = pos_embeddings

    # 标记类型嵌入
    tokentype_embeddings = embeddings["tokentype_embeddings"]["weight"]
    # 存储标记类型嵌入
    output_state_dict["bert.embeddings.token_type_embeddings.weight"] = tokentype_embeddings

    # Transformer
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # 提取层名称的正则表达式
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # "自动化" 规则的简单名��映射
    megatron_to_transformers = {
        "attention.dense": ".attention.output.dense.",
        "self_attention.dense": ".attention.output.dense.",
        "mlp.dense_h_to_4h": ".intermediate.dense.",
        "mlp.dense_4h_to_h": ".output.dense.",
    }
    # 用于跟踪注意力/查询/值张量
    attention_qkv_weight = None

    # 提取层
    # 最终的 LayerNorm
    output_state_dict["bert.encoder.ln.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["bert.encoder.ln.bias"] = transformer["final_layernorm.bias"]

    # 池化器
    pooler = lm["pooler"]

    # 存储矩阵和偏置
    output_state_dict["bert.pooler.dense.weight"] = pooler["dense.weight"]
    output_state_dict["bert.pooler.dense.bias"] = pooler["dense.bias"]

    # 来自 Megatron 的 LM 头（用于 RACE）
    lm_head = model["lm_head"]

    # 变换矩阵
    output_state_dict["cls.predictions.transform.dense.weight"] = lm_head["dense.weight"]
    output_state_dict["cls.predictions.transform.dense.bias"] = lm_head["dense.bias"]

    # 变换的 LayerNorm
    output_state_dict["cls.predictions.transform.LayerNorm.weight"] = lm_head["layernorm.weight"]
    output_state_dict["cls.predictions.transform.LayerNorm.bias"] = lm_head["layernorm.bias"]

    # 对于解码器，我们复制权重
    output_state_dict["cls.predictions.decoder.weight"] = word_embeddings
    output_state_dict["cls.predictions.bias"] = lm_head["bias"]

    # 来自 Megatron 的分类器（用于 MLNI）
    binary_head = model["binary_head"]

    # 存储分类器
    output_state_dict["cls.seq_relationship.weight"] = binary_head["weight"]
    output_state_dict["cls.seq_relationship.bias"] = binary_head["bias"]

    # 完成！
    return output_state_dict
# 主函数，程序入口
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数选项
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    parser.add_argument("path_to_checkpoint", type=str, help="Path to the ZIP file containing the checkpoint")
    parser.add_argument(
        "--config_file",
        default="",
        type=str,
        help="An optional config json file describing the pre-trained model.",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 提取基本名称
    basename = os.path.dirname(args.path_to_checkpoint)

    # 加载模型
    # .zip 是可选的，为了向后兼容性，保留它
    print(f'Extracting PyTorch state dictionary from "{args.path_to_checkpoint}"')
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    if args.config_file == "":
        # 默认的 megatron-bert 345m 配置
        config = MegatronBertConfig()

        # 不同的 megatron-bert-*-345m 模型具有不同的词汇表大小，因此用实际的词汇维度覆盖默认配置
        config.vocab_size = input_state_dict["model"]["lm_head"]["bias"].numel()
    else:
        config = MegatronBertConfig.from_json_file(args.config_file)

    # 转换
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # 打印转换后状态字典的结构
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # 将配置保存到文件
    print("Saving config")
    config.save_pretrained(basename)

    # 将状态字典保存到文件
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)


if __name__ == "__main__":
    main()
```