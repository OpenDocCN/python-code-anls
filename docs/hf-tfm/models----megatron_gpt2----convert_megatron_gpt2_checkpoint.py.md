# `.\transformers\models\megatron_gpt2\convert_megatron_gpt2_checkpoint.py`

```py
####################################################################################################

# 版权声明，版权归 NVIDIA 公司所有
#
# 根据 Apache 许可证 2.0 版本使用此文件；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件。
# 请查看许可证以获取特定语言下的权限和限制。

####################################################################################################

#
# 注意：如果在运行此转换脚本时出现异常：
#     ModuleNotFoundError: No module named 'megatron.model.enums'
# 您需要告诉 Python 在哪里找到 Megatron-LM 的克隆版本，例如：
#
# cd /tmp
# git clone https://github.com/NVIDIA/Megatron-LM
# PYTHONPATH=/tmp/Megatron-LM python src/transformers/models/megatron_gpt2/convert_megatron_gpt2_checkpoint.py ...
#
# 如果您已经在其他地方克隆了它，请简单地调整到现有路径
#
# 如果训练是使用 Megatron-LM 的分支进行的，例如，
# https://github.com/microsoft/Megatron-DeepSpeed/ 那么您可能需要将其添加到路径中，即，/path/to/Megatron-DeepSpeed/

import argparse
import os
import re
import zipfile

import torch

from transformers import AutoTokenizer, GPT2Config


####################################################################################################


def recursive_print(name, val, spaces=0):
    # 递归打印参数名称和值
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


def fix_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # 修复参数张量的查询键值排序，以便与后续版本的 NVIDIA Megatron-LM 兼容
    # 将 param 张量的布局排列为 [num_splits * num_heads * hidden_size, :]
    # 在 Megatron-LM 内部执行逆操作以读取检查点：
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # 如果 param 是自注意力块的权重张量，则返回的张量将需要再次转置才能被 HuggingFace GPT2 读取。
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
# 定义一个函数，用于将 Megatron-LM 模型的检查点转换为 Transformers 模型的检查点
def convert_megatron_checkpoint(args, input_state_dict, config):
    # 存储转换后的输出模型的状态字典
    output_state_dict = {}

    # 旧版本没有存储训练参数
    ds_args = input_state_dict.get("args", None)
    if ds_args is not None:
        # 如果检查点中已经包含了确切的维度/大小信息，则不需要用户编写配置文件
        config.vocab_size = ds_args.padded_vocab_size
        config.n_positions = ds_args.max_position_embeddings
        config.n_embd = ds_args.hidden_size
        config.n_layer = ds_args.num_layers
        config.n_head = ds_args.num_attention_heads
        config.n_inner = ds_args.ffn_hidden_size

    # 头部的数量
    heads = config.n_head
    # 每个头部的隐藏大小
    hidden_size_per_head = config.n_embd // config.n_head
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
    output_state_dict["transformer.wte.weight"] = word_embeddings

    # 位置嵌入
    pos_embeddings = embeddings["position_embeddings"]["weight"]
    # 读取因果蒙版维度（序列长度）[max_sequence_length, hidden_size]
    n_positions = pos_embeddings.size(0)
    if n_positions != config.n_positions:
        raise ValueError(
            f"pos_embeddings.max_sequence_length={n_positions} and config.n_positions={config.n_positions} don't match"
        )
    # 存储位置嵌入
    output_state_dict["transformer.wpe.weight"] = pos_embeddings

    # Transformer
    transformer = lm["transformer"] if "transformer" in lm.keys() else lm["encoder"]

    # 用于提取层名称的正则表达式
    layer_re = re.compile(r"layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")

    # "自动"规则的简单名称映射
    megatron_to_transformers = {
        "attention.dense": ".attn.c_proj.",
        "self_attention.dense": ".attn.c_proj.",
        "mlp.dense_h_to_4h": ".mlp.c_fc.",
        "mlp.dense_4h_to_h": ".mlp.c_proj.",
    }

    # 提取层
    # 遍历 transformer 字典的键值对
    for key, val in transformer.items():
        # 匹配层名称的正则表达式
        m = layer_re.match(key)

        # 如果不是层，则停止
        if m is None:
            break

        # 层的索引
        layer_idx = int(m.group(1))
        # 操作的名称
        op_name = m.group(2)
        # 是权重还是偏置？
        weight_or_bias = m.group(3)

        # 层的名称
        layer_name = f"transformer.h.{layer_idx}"

        # 对于 layernorm，直接存储层归一化
        if op_name.endswith("layernorm"):
            ln_name = "ln_1" if op_name.startswith("input") else "ln_2"
            output_state_dict[layer_name + "." + ln_name + "." + weight_or_bias] = val

        # 转置 QKV 矩阵
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "weight":
            # 插入一个 1x1xDxD 的偏置张量
            causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.float16)).view(
                1, 1, n_positions, n_positions
            )
            output_state_dict[layer_name + ".attn.bias"] = causal_mask

            # 插入一个用于掩码偏置的“虚拟”张量
            masked_bias = torch.tensor(-1e4, dtype=torch.float16)
            output_state_dict[layer_name + ".attn.masked_bias"] = masked_bias

            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # Megatron 存储 (3*D) x D，但 transformers-GPT2 需要 D x (3*D)
            out_val = out_val.transpose(0, 1).contiguous()
            # 存储
            output_state_dict[layer_name + ".attn.c_attn.weight"] = out_val

        # 转置偏置
        elif (
            op_name == "attention.query_key_value" or op_name == "self_attention.query_key_value"
        ) and weight_or_bias == "bias":
            out_val = fix_query_key_value_ordering(val, checkpoint_version, 3, heads, hidden_size_per_head)
            # 存储，形状不变
            output_state_dict[layer_name + ".attn.c_attn.bias"] = out_val

        # 转置权重
        elif weight_or_bias == "weight":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "weight"] = val.transpose(0, 1)

        # 复制偏置
        elif weight_or_bias == "bias":
            out_name = megatron_to_transformers[op_name]
            output_state_dict[layer_name + out_name + "bias"] = val

    # DEBUG：确保层数与配置中的层数匹配
    assert config.n_layer == layer_idx + 1

    # 最终的 layernorm
    output_state_dict["transformer.ln_f.weight"] = transformer["final_layernorm.weight"]
    output_state_dict["transformer.ln_f.bias"] = transformer["final_layernorm.bias"]

    # 对于 LM 头，transformers 希望用于加权嵌入的矩阵
    output_state_dict["lm_head.weight"] = word_embeddings

    # 完成！
    # 返回函数的状态字典作为输出
    return output_state_dict
####################################################################################################

# 主函数
def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加用于打印检查点结构的参数
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    # 添加路径参数，用于指定检查点文件
    parser.add_argument(
        "path_to_checkpoint",
        type=str,
        help="Path to the checkpoint file (.zip archive or direct .pt file)",
    )
    # 添加配置文件参数
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
    # .zip 是可选的，为了向后兼容性
    print(f"Extracting PyTorch state dictionary from {args.path_to_checkpoint}")
    if args.path_to_checkpoint.endswith(".zip"):
        with zipfile.ZipFile(args.path_to_checkpoint, "r") as checkpoint:
            with checkpoint.open("release/mp_rank_00/model_optim_rng.pt") as pytorch_dict:
                input_state_dict = torch.load(pytorch_dict, map_location="cpu")
    else:
        input_state_dict = torch.load(args.path_to_checkpoint, map_location="cpu")

    ds_args = input_state_dict.get("args", None)

    # 读取配置，如果没有配置文件则默认使用 NVIDIA 发布的模型配置
    if args.config_file == "":
        if ds_args is not None:
            if ds_args.bias_gelu_fusion:
                activation_function = "gelu_fast"
            elif ds_args.openai_gelu:
                activation_function = "gelu_new"
            else:
                activation_function = "gelu"
        else:
            activation_function = "gelu_new"

        # 明确指定所有参数，以防默认值发生变化
        config = GPT2Config(
            vocab_size=50257,
            n_positions=1024,
            n_embd=1024,
            n_layer=24,
            n_head=16,
            n_inner=4096,
            activation_function=activation_function,
            resid_pdrop=0.1,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
        )
    else:
        # 从 JSON 文件加载配置
        config = GPT2Config.from_json_file(args.config_file)

    config.architectures = ["GPT2LMHeadModel"]

    # 转换模型
    print("Converting")
    output_state_dict = convert_megatron_checkpoint(args, input_state_dict, config)

    # 打印转换后状态字典的结构
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)
    # 根据 tokenizer_type 参数设置 tokenizer 的模型名称
    if ds_args is not None:
        tokenizer_type = ds_args.tokenizer_type
        # 如果 tokenizer_type 为 "GPT2BPETokenizer"，设置 tokenizer_model_name 为 "gpt2"
        if tokenizer_type == "GPT2BPETokenizer":
            tokenizer_model_name = "gpt2"
        # 如果 tokenizer_type 为 "PretrainedFromHF"，设置 tokenizer_model_name 为 ds_args.tokenizer_name_or_path
        elif tokenizer_type == "PretrainedFromHF":
            tokenizer_model_name = ds_args.tokenizer_name_or_path
        # 如果 tokenizer_type 不符合以上两种情况，抛出 ValueError 异常
        else:
            raise ValueError(f"Unrecognized tokenizer_type {tokenizer_type}")
    # 如果 ds_args 为 None，设置 tokenizer_model_name 为 "gpt2"
    else:
        tokenizer_model_name = "gpt2"
    
    # 根据 tokenizer_model_name 加载相应的 AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    # 获取 tokenizer 的类名
    tokenizer_class = type(tokenizer).__name__
    # 将 tokenizer 类名设置到配置对象的 tokenizer_class 属性中
    config.tokenizer_class = tokenizer_class
    
    # 将配置对象保存到指定路径
    print("Saving config")
    config.save_pretrained(basename)
    
    # 将 tokenizer 保存到指定路径
    print(f"Adding {tokenizer_class} tokenizer files")
    tokenizer.save_pretrained(basename)
    
    # 将输出的 state_dict 保存到指定路径
    output_checkpoint_file = os.path.join(basename, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(output_state_dict, output_checkpoint_file)
# 如果当前文件被作为脚本直接执行，则调用 main 函数
if __name__ == "__main__":
    main()
```