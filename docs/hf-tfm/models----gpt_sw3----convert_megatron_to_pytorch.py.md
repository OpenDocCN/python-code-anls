# `.\models\gpt_sw3\convert_megatron_to_pytorch.py`

```py
# 版权声明
# 版权归 The HuggingFace Inc. team 和 AI-Sweden team 所有
# 根据 Apache License, Version 2.0 许可，除非遵守许可，否则不得使用此文件
# 您可以获取许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依照许可分发的软件是基于"按现状"分发的，不附带任何明示或暗示的担保或条件。请查看许可中关于具体语言管理权限和权限限制的信息

# 将 GPT-SW3 megatron 检查点转换为 pytorch
import argparse
import os
from os.path import isfile

import torch

from transformers import GPT2Config


# 递归打印函数
def recursive_print(name, val, spaces=0):
    # 格式化消息
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # 打印和递归（如果需要的话）
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


# 修复查询参数键值顺序
def fix_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # 将参数张量布局排列为 [num_splits * num_heads * hidden_size, :]，以便与后续版本的 NVIDIA Megatron-LM 兼容。
    # 在 Megatron-LM 内部执行的逆操作用于读取检查点：
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # 如果 param 是自注意力块的权重张量，返回的张量将需要再次转置才能被 HuggingFace GPT2 读取。
    input_shape = param.size()
    saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]  # 其他版本的存储形状为 [num_heads * num_splits * hidden_size, :]
    param = param.view(*saved_shape)
    param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


# 转换 Megatron 检查点为 HuggingFace 的 GPT-SW3 检查点
def convert_megatron_checkpoint(sd_megatron, config):
    n_positions = config.n_positions
    layers = config.n_layer
    vocab_size = config.vocab_size
    heads = config.n_head
    hidden_size_per_head = config.n_embd // config.n_head

    word_embeddings = sd_megatron["model.language_model.embedding.word_embeddings.weight"][:vocab_size, :]
    # 定义一个字典，包含了需要转换的参数名称和对应的数值
    sd_hf = {
        "transformer.wte.weight": word_embeddings,  # 设置 word_embeddings 为 "transformer.wte.weight" 的值
        "transformer.wpe.weight": sd_megatron["model.language_model.embedding.position_embeddings.weight"],  # 设置 sd_megatron 中的位置嵌入权重为 "transformer.wpe.weight" 的值
        "transformer.ln_f.weight": sd_megatron["model.language_model.encoder.final_layernorm.weight"],  # 设置 sd_megatron 中的最终层归一化权重为 "transformer.ln_f.weight" 的值
        "transformer.ln_f.bias": sd_megatron["model.language_model.encoder.final_layernorm.bias"],  # 设置 sd_megatron 中的最终层归一化偏置为 "transformer.ln_f.bias" 的值
    }

    # 循环遍历每个层，设置对应的参数值
    pf = "model.language_model.encoder.layers."
    for i in range(layers):
        # 创建一个下三角矩阵作为自回归遮罩
        causal_mask = torch.tril(torch.ones((n_positions, n_positions), dtype=torch.bool))
        causal_mask = causal_mask.view(1, 1, n_positions, n_positions)
        sd_hf[f"transformer.h.{i}.attn.bias"] = causal_mask  # 设置自回归遮罩为 "transformer.h.{i}.attn.bias" 的值
        sd_hf[f"transformer.h.{i}.attn.masked_bias"] = torch.tensor(-1e4, dtype=torch.bfloat16)  # 设置掩码偏置为 "transformer.h.{i}.attn.masked_bias" 的值

        # 设置每个层的参数值
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

    # 为语言模型头部设置权重矩阵
    sd_hf["lm_head.weight"] = word_embeddings

    return sd_hf
def copy_config(config_hf, config_megatron):
    """Copy the config from Megatron to hf."""
    # 将 Megatron 的配置复制到 hf 中
    config_hf.vocab_size = 64000
    config_hf.n_positions = config_megatron["encoder_seq_length"]
    config_hf.n_embd = config_megatron["hidden_size"]
    config_hf.n_layer = config_megatron["num_layers"]
    config_hf.n_head = config_megatron["num_attention_heads"]
    config_hf.n_inner = config_megatron["ffn_hidden_size"]
    config_hf.activation_function = "gelu"
    config_hf.resid_pdrop = 0.1
    config_hf.embd_pdrop = 0.1
    config_hf.attn_pdrop = 0.1
    config_hf.layer_norm_epsilon = config_megatron["layernorm_epsilon"]  # 1e-5
    config_hf.initializer_range = config_megatron["init_method_std"]  # 0.02
    config_hf.apply_query_key_layer_scaling = config_megatron["apply_query_key_layer_scaling"]  # True
    config_hf.normalize_attention_scores = True
    config_hf.use_cache = True

    # This identifies the 6.7B (7B) model which uses a different tokenizer
    if config_megatron["hidden_size"] == 4096:
        config_hf.bos_token_id = 1  # <|endoftext|>
        config_hf.eos_token_id = 1  # <|endoftext|>
        config_hf.pad_token_id = 0  # <unk>
    else:
        config_hf.bos_token_id = 2  # <s>
        config_hf.eos_token_id = 3  # <|endoftext|>
        config_hf.pad_token_id = 0  # <pad>

    return config_hf


def main(args):
    print(args)

    checkpoint_path = args.checkpoint_path
    save_path = args.save_path
    if isfile(checkpoint_path):
        raise FileNotFoundError(f"ERROR! could not find file {checkpoint_path}")

    # Load the model.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load the config.
    config_megatron = checkpoint["hyper_parameters"]["cfg"]
    config_hf = GPT2Config()
    config_hf = copy_config(config_hf=config_hf, config_megatron=config_megatron)
    config_hf.architectures = ["GPT2LMHeadModel"]

    sd_megatron = checkpoint["state_dict"]

    # Convert.
    print("Converting")
    sd_hf = convert_megatron_checkpoint(sd_megatron, config_hf)

    # Print the structure of converted state dict.
    if args.print_checkpoint_structure:
        recursive_print(None, sd_hf)

    config_hf.tokenizer_class = "GPTSw3Tokenizer"

    # Store the config to file.
    print("Saving config")
    config_hf.save_pretrained(save_path)

    # Store the state_dict to file.
    output_checkpoint_file = os.path.join(save_path, "pytorch_model.bin")
    print(f'Saving checkpoint to "{output_checkpoint_file}"')
    torch.save(sd_hf, output_checkpoint_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="e.g. megatron_gpt--val_loss=2.42-step=38000-consumed_samples=54720000",
    )
    parser.add_argument("--save_path", type=str, required=True, help="e.g. /home/user/gpt-sw3/hf")
    parser.add_argument("--print-checkpoint-structure", action="store_true")
    # 解析命令行参数并将其存储在_args变量中
    _args = parser.parse_args()
    # 调用main函数，并将_args作为参数传入
    main(_args)
```