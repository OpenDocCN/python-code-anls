# `.\models\gpt_sw3\convert_megatron_to_pytorch.py`

```py
# 版权声明和许可信息
# 2022 年版权归 HuggingFace Inc. 团队和 AI-Sweden 团队所有，保留所有权利。
#
# 根据 Apache 许可证版本 2.0 授权；
# 除非符合许可证的条款，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，
# 不附带任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" 将 GPT-SW3 Megatron 检查点转换为 PyTorch 格式 """

import argparse
import os
from os.path import isfile

import torch

from transformers import GPT2Config


def recursive_print(name, val, spaces=0):
    # 递归打印函数名和对应的值
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    if isinstance(val, dict):
        # 如果值是字典，则递归打印键和值
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        # 如果值是 Torch 张量，则打印值的大小
        print(msg, ":", val.size())
    else:
        # 否则只打印消息和值
        print(msg, ":", val)


def fix_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # 调整参数张量的布局，以便与后续版本的 NVIDIA Megatron-LM 兼容
    # 如果 param 是自注意力模块的权重张量，则返回的张量需要再次转置才能被 HuggingFace GPT2 读取
    input_shape = param.size()
    saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
    param = param.view(*saved_shape)
    param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def convert_megatron_checkpoint(sd_megatron, config):
    """
    将 Megatron 检查点转换为 HuggingFace GPT-SW3 检查点。
    """
    n_positions = config.n_positions
    layers = config.n_layer
    vocab_size = config.vocab_size
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head

    word_embeddings = sd_megatron["model.language_model.embedding.word_embeddings.weight"][:vocab_size, :]
    # 定义一个字典 sd_hf，用于存储模型参数的映射关系
    sd_hf = {
        "transformer.wte.weight": word_embeddings,  # 将 word_embeddings 赋给键 "transformer.wte.weight"
        "transformer.wpe.weight": sd_megatron["model.language_model.embedding.position_embeddings.weight"],  # 将位置编码的权重赋给键 "transformer.wpe.weight"
        "transformer.ln_f.weight": sd_megatron["model.language_model.encoder.final_layernorm.weight"],  # 将最终层归一化层的权重赋给键 "transformer.ln_f.weight"
        "transformer.ln_f.bias": sd_megatron["model.language_model.encoder.final_layernorm.bias"],  # 将最终层归一化层的偏置赋给键 "transformer.ln_f.bias"
    }

    # 定义模型层的前缀字符串
    pf = "model.language_model.encoder.layers."

    # 遍历每个层，生成对应的参数映射
    for i in range(layers):
        # 创建一个因果掩码，限制自注意力只能关注当前及之前位置
        causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, n_positions, n_positions)
        sd_hf[f"transformer.h.{i}.attn.bias"] = causal_mask  # 将因果掩码赋给键 "transformer.h.{i}.attn.bias"
        
        # 设置自注意力层的偏置为一个很小的负数，用于掩盖无效位置的注意力分数
        sd_hf[f"transformer.h.{i}.attn.masked_bias"] = torch.tensor(-1e4, dtype=torch.bfloat16)  # 将负数偏置赋给键 "transformer.h.{i}.attn.masked_bias"

        # 以下为各个权重及偏置的赋值过程，从 Megatron 模型中提取对应层的参数
        sd_hf[f"transformer.h.{i}.ln_1.weight"] = sd_megatron[f"{pf}{i}.input_layernorm.weight"]
        sd_hf[f"transformer.h.{i}.ln_1.bias"] = sd_megatron[f"{pf}{i}.input_layernorm.bias"]

        val1 = sd_megatron[f"{pf}{i}.self_attention.query_key_value.weight"]
        val1 = fix_query_key_value_ordering(val1, 3, heads, hidden_size_per_head)
        sd_hf[f"transformer.h.{i}.attn.c_attn.weight"] = val1.transpose(0, 1).contiguous()

        val2 = sd_megatron[f"{pf}{i}.self_attention.query_key_value.bias"]
        val2 = fix_query_key_value_ordering(val2, 3, heads, hidden_size_per_head)
        sd_hf[f"transformer.h.{i}.attn.c_attn.bias"] = val2

        sd_hf[f"transformer.h.{i}.attn.c_proj.weight"] = sd_megatron[f"{pf}{i}.self_attention.dense.weight"].transpose(0, 1)
        sd_hf[f"transformer.h.{i}.attn.c_proj.bias"] = sd_megatron[f"{pf}{i}.self_attention.dense.bias"]
        sd_hf[f"transformer.h.{i}.ln_2.weight"] = sd_megatron[f"{pf}{i}.post_attention_layernorm.weight"]
        sd_hf[f"transformer.h.{i}.ln_2.bias"] = sd_megatron[f"{pf}{i}.post_attention_layernorm.bias"]
        sd_hf[f"transformer.h.{i}.mlp.c_fc.weight"] = sd_megatron[f"{pf}{i}.mlp.dense_h_to_4h.weight"].transpose(0, 1)
        sd_hf[f"transformer.h.{i}.mlp.c_fc.bias"] = sd_megatron[f"{pf}{i}.mlp.dense_h_to_4h.bias"]
        sd_hf[f"transformer.h.{i}.mlp.c_proj.weight"] = sd_megatron[f"{pf}{i}.mlp.dense_4h_to_h.weight"].transpose(0, 1)
        sd_hf[f"transformer.h.{i}.mlp.c_proj.bias"] = sd_megatron[f"{pf}{i}.mlp.dense_4h_to_h.bias"]

    # 对于语言模型头部，将词嵌入矩阵赋给 "lm_head.weight"
    sd_hf["lm_head.weight"] = word_embeddings

    # 返回完整的参数映射字典 sd_hf
    return sd_hf
# 主函数，接收参数 args
def main(args):
    # 打印输入的参数 args
    print(args)

    # 从参数中获取 checkpoint_path 和 save_path
    checkpoint_path = args.checkpoint_path
    save_path = args.save_path

    # 检查 checkpoint_path 是否为文件，如果是则抛出文件未找到异常
    if isfile(checkpoint_path):
        raise FileNotFoundError(f"ERROR! could not find file {checkpoint_path}")

    # 使用 torch 加载模型检查点
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 从检查点中加载 Megatron 的配置
    config_megatron = checkpoint["hyper_parameters"]["cfg"]

    # 创建一个新的 GPT2Config 对象，作为转换后的配置
    config_hf = GPT2Config()

    # 调用 copy_config 函数，将 Megatron 的配置复制到 config_hf 中
    config_hf = copy_config(config_hf=config_hf, config_megatron=config_megatron)

    # 设置模型的架构为 GPT2LMHeadModel
    config_hf.architectures = ["GPT2LMHeadModel"]

    # 从检查点中加载 Megatron 的状态字典
    sd_megatron = checkpoint["state_dict"]

    # 转换 Megatron 的状态字典到适用于 HF 的格式
    print("Converting")
    sd_hf = convert_megatron_checkpoint(sd_megatron, config_hf)

    # 如果设置了打印检查点结构的标志，递归打印转换后的状态字典结构
    if args.print_checkpoint_structure:
        recursive_print(None, sd_hf)

    # 设置 config_hf 的 tokenizer_class 为 "GPTSw3Tokenizer"
    config_hf.tokenizer_class = "GPTSw3Tokenizer"

    # 将配置保存到指定的 save_path
    print("Saving config")
    config_hf.save_pretrained(save_path)

    # 将转换后的状态字典保存到文件中
    output_checkpoint_file = os.path.join(save_path, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(sd_hf, output_checkpoint_file)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数：checkpoint_path，表示 Megatron 模型的检查点路径
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="e.g. megatron_gpt--val_loss=2.42-step=38000-consumed_samples=54720000",
    )

    # 添加命令行参数：save_path，表示 HF 模型的保存路径
    parser.add_argument("--save_path", type=str, required=True, help="e.g. /home/user/gpt-sw3/hf")

    # 添加命令行选项：print-checkpoint-structure，如果设置则打印转换后的检查点结构
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    # 解析命令行参数并存储到 _args 变量中
    _args = parser.parse_args()
    # 调用主函数 main，传入解析后的命令行参数作为参数
    main(_args)
```