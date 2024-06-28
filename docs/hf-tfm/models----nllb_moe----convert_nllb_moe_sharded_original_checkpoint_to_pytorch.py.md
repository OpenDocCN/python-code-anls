# `.\models\nllb_moe\convert_nllb_moe_sharded_original_checkpoint_to_pytorch.py`

```py
# 导入必要的库和模块
import argparse  # 用于命令行参数解析
import json  # 用于处理 JSON 格式数据
import os  # 提供操作系统相关功能的模块

import torch  # 张量计算库 PyTorch
from torch import nn  # PyTorch 的神经网络模块

# 从 transformers 库中导入模型和配置类
from transformers import NllbMoeConfig, NllbMoeModel
# 从 transformers 模块中导入数据类型相关的函数
from transformers.modeling_utils import dtype_byte_size
# 从 transformers 模块中导入权重相关的常量和函数
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME


def remove_ignore_keys_(state_dict):
    # 定义需要从 state_dict 中移除的键列表
    ignore_keys = [
        "encoder.version",  # 版本信息，不需保留
        "decoder.version",  # 版本信息，不需保留
        "model.encoder.version",  # 版本信息，不需保留
        "model.decoder.version",  # 版本信息，不需保留
        "decoder.output_projection.weight",  # 解码器输出投影权重，不需保留
        "_float_tensor",  # 浮点数张量，不需保留
        "encoder.embed_positions._float_tensor",  # 编码器位置嵌入的浮点数张量，不需保留
        "decoder.embed_positions._float_tensor",  # 解码器位置嵌入的浮点数张量，不需保留
    ]
    # 逐一移除 ignore_keys 中指定的键
    for k in ignore_keys:
        state_dict.pop(k, None)


def make_linear_from_emb(emb):
    # 根据嵌入层 emb 创建一个线性层
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    # 将线性层的权重设为与嵌入层相同的数据
    lin_layer.weight.data = emb.weight.data
    return lin_layer


def rename_fairseq_keys(state_dict, expert_idx=None):
    new_dict = {}
    # 遍历 state_dict 的键
    for old_key in state_dict.keys():
        key = old_key
        # 替换 moe_layer.experts. 为 ffn.experts.expert_，用于重命名键
        if "moe_layer.experts." in key:
            if expert_idx is not None:
                key = key.replace("moe_layer.experts.0", f"ffn.experts.expert_{expert_idx}")
            else:
                key = key.replace("moe_layer.experts.", "ffn.experts.expert_")
        # 将 gate 替换为 ffn.router.classifier，用于重命名键
        if "gate" in key:
            key = key.replace(".moe_layer.gate.wg", ".ffn.router.classifier")
        # 将 fc2 替换为 ffn.fc2，用于重命名键
        if "fc2" and "experts" not in key:
            key = key.replace(".fc2.", ".ffn.fc2.")
        # 将 fc1 替换为 ffn.fc1，用于重命名键
        if "fc1" and "experts" not in key:
            key = key.replace(".fc1.", ".ffn.fc1.")
        # 将 encoder_attn 替换为 cross_attention，用于重命名键
        if ".encoder_attn." in key:
            key = key.replace(".encoder_attn.", ".cross_attention.")
        # 将 encoder_attn_layer_norm 替换为 cross_attention_layer_norm，用于重命名键
        if "encoder_attn_layer_norm" in key:
            key = key.replace("encoder_attn_layer_norm", "cross_attention_layer_norm")
        # 将 final_layer_norm 替换为 ff_layer_norm，用于重命名键
        if "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ff_layer_norm")
        # 将新键值对加入到 new_dict 中
        new_dict[key] = state_dict[old_key]
    return new_dict


def shard_on_the_fly(switch_checkpoint_path, dump_path, num_experts, dtype, weights_name: str = WEIGHTS_NAME):
    sharded_state_dicts = []  # 初始化空的分片状态字典列表
    total_size = 0  # 初始化总大小为 0
    os.makedirs(dump_path, exist_ok=True)  # 创建 dump_path 目录，如果不存在的话
    # 遍历所有专家的范围，从0到num_experts-1
    for expert in range(num_experts):
        # 构造每个专家的检查点路径，形如"switch_checkpoint_path-rank-{expert}.pt"
        expert_path = switch_checkpoint_path + f"-rank-{expert}.pt"
        # 检查该路径是否是文件
        if os.path.isfile(expert_path):
            # 如果是文件，加载专家模型的状态字典
            expert_state = torch.load(expert_path)["model"]
            # 移除模型中要忽略的键
            remove_ignore_keys_(expert_state)
            # 重命名Fairseq模型中的键，使用专家的索引
            expert_state = rename_fairseq_keys(expert_state, expert)
            # 构造保存路径，使用weights_name替换后缀为".bin"的部分，形如"-{len(sharded_state_dicts)+1:05d}-of-???.bin"
            save_path = os.path.join(
                dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
            )
            # 保存专家模型的状态字典到指定路径
            torch.save(expert_state, save_path)
            # 将专家模型的键集合添加到sharded_state_dicts中
            sharded_state_dicts.append(expert_state.keys())
            # 更新总大小，计算专家模型中所有张量的总字节数
            total_size += sum([value.numel() for key, value in expert_state.items()]) * dtype_byte_size(
                expert_state[list(expert_state)[0]].dtype
            )

    # 添加共享权重模型的最后一个块
    # 构造保存路径，使用weights_name替换后缀为".bin"的部分，形如"-{len(sharded_state_dicts)+1:05d}-of-???.bin"
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    # 加载共享权重模型的状态字典
    shared_weights = torch.load(switch_checkpoint_path + "-shared.pt")["model"]
    # 移除模型中要忽略的键
    remove_ignore_keys_(shared_weights)
    # 重命名Fairseq模型中的键，此时专家为None
    shared_weights = rename_fairseq_keys(shared_weights, None)
    # 将共享权重中的"decoder.embed_tokens.weight"键映射到"shared.weight"
    shared_weights["shared.weight"] = shared_weights["decoder.embed_tokens.weight"]
    # 将共享权重模型的键集合添加到sharded_state_dicts中
    sharded_state_dicts.append(shared_weights.keys())

    # 如果只有共享权重（即dummy模型或专家保存在同一个文件中）
    if len(sharded_state_dicts) == 1:
        # 构造保存路径，直接使用weights_name
        save_path = os.path.join(dump_path, weights_name)
        # 保存共享权重模型的状态字典到指定路径
        torch.save(shared_weights, save_path)
        # 返回只包含一个元素的字典，表示文件名和sharded_state_dicts的第一个元素，以及None
        return {weights_name: sharded_state_dicts[0]}, None
    else:
        # 如果存在多个权重块，保存共享权重模型的状态字典到指定路径
        torch.save(shared_weights, save_path)

    # 否则，构建索引
    # 初始化权重映射字典
    weight_map = {}
    # 遍历所有权重块的索引和名称
    for idx, shard in enumerate(sharded_state_dicts):
        # 构造每个权重块的文件名，形如"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        shard_file = weights_name.replace(".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin")
        # 构造临时文件名，形如"-{idx+1:05d}-of-???.bin"
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        # 重命名临时文件为最终的权重块文件名
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        # 遍历当前权重块中的所有键，将其映射到对应的权重块文件名
        for key in shard:
            weight_map[key] = shard_file

    # 添加元数据
    # 构造包含总大小的元数据字典
    metadata = {"total_size": total_size}
    # 构造包含元数据和权重映射的索引字典
    index = {"metadata": metadata, "weight_map": weight_map}

    # 将索引字典以JSON格式写入文件
    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        # 将索引字典转换为格式化的JSON字符串并写入文件
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    # 返回元数据和索引字典
    return metadata, index
if __name__ == "__main__":
    # 如果脚本被直接执行，则开始执行以下操作

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--nllb_moe_checkpoint_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/model_moe_54b/checkpoint_2_300000",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    # 添加必需的参数：nllb_moe_checkpoint_path，表示模型检查点的路径

    parser.add_argument("--dtype", default="float32", type=str, required=False, help="dtype of the saved model")
    # 添加参数：dtype，默认为"float32"，表示保存模型的数据类型

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/home/arthur_huggingface_co/fairseq/weights/checkpoints/hf-converted-moe-54b",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    # 添加参数：pytorch_dump_folder_path，表示输出 PyTorch 模型的路径

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 对象中

    metadata, index = shard_on_the_fly(
        args.nllb_moe_checkpoint_path,
        args.pytorch_dump_folder_path,
        128,
        args.dtype,
    )
    # 调用 shard_on_the_fly 函数，使用命令行参数中的路径和参数来执行分片操作，并返回元数据和索引信息

    config = NllbMoeConfig.from_pretrained(
        "facebook/nllb-200-3.3B", encoder_sparse_step=4, decoder_sparse_step=4, num_experts=128
    )
    # 从预训练模型加载配置信息，指定了一些特定参数

    config.save_pretrained(args.pytorch_dump_folder_path)
    # 将配置信息保存到指定的 PyTorch 模型输出路径中

    model = NllbMoeModel.from_pretrained(args.pytorch_dump_folder_path)
    # 从指定路径加载预训练模型

    print("Done")
    # 打印提示信息，表明程序执行完成

    model.save_pretrained(args.pytorch_dump_folder_path)
    # 将加载的预训练模型保存到指定的 PyTorch 模型输出路径中
```