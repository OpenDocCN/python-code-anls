# `.\models\switch_transformers\convert_big_switch.py`

```py
# 导入必要的库
import argparse  # 命令行参数解析库
import json  # JSON 数据处理库
import os  # 系统操作库

import tensorstore as ts  # TensorStore 库
import torch  # PyTorch 深度学习库
from flax import serialization  # Flax 序列化库
from flax.traverse_util import flatten_dict, unflatten_dict  # Flax 的字典扁平化和反扁平化工具
from tensorflow.io import gfile  # TensorFlow 文件操作库

from transformers.modeling_utils import dtype_byte_size  # 计算数据类型字节大小的工具函数
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
    rename_keys,  # 从 Switch Transformers 原始 Flax 检查点转换到 PyTorch 的键重命名函数
)
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME  # Transformers 模型权重文件索引名称和通用权重名称
from transformers.utils.hub import convert_file_size_to_int  # 将文件大小转换为整数的函数


def rename_base_flax_keys(flax_key_tuple, flax_tensor):
    """
    对基本 JAX 键进行重命名以适配 PyTorch。
    """
    if flax_key_tuple[-1] == "kernel" and flax_tensor.ndim == 3:
        # 对专家层的特定处理
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
        flax_tensor = torch.permute(flax_tensor, (0, 2, 1))
    elif flax_key_tuple[-1] == "kernel" and ".".join(flax_key_tuple):
        # 对线性层的特定处理
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
        flax_tensor = flax_tensor.T
    elif flax_key_tuple[-1] in ["scale", "embedding"]:
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

    return flax_key_tuple, flax_tensor


def get_key_and_tensorstore_dict(layer, checkpoint_info, switch_checkpoint_path):
    """
    获取键和 TensorStore 字典。
    """
    if "metadata" in layer:
        split_layer = layer.split("metadata")
        curr_real_layer_name = "".join(split_layer[0])[:-1]
        split_layer = [tuple(("metadata" + split_layer[1]).split("/"))]
    elif "kvstore" in layer:
        split_layer = layer.split("kvstore")
        curr_real_layer_name = "".join(split_layer[0])[:-1]
        split_layer = [tuple(("kvstore" + split_layer[1]).split("/"))]
    else:
        split_layer = layer.split("/")
        curr_real_layer_name = "/".join(split_layer[:-1])
        split_layer[-1] = (split_layer[-1],)

    if "kvstore/path" in layer:
        content = f"{switch_checkpoint_path}/{checkpoint_info[layer]}"
    elif "kvstore/driver" in layer:
        content = "file"
    else:
        content = checkpoint_info[layer]

    return curr_real_layer_name, split_layer, content


def rename_and_save_block(current_block, save_path):
    """
    重命名当前块的键并保存。
    """
    current_block = rename_keys(current_block)
    new_current_block = {}
    for k, v in current_block.items():
        new_current_block[k.replace("/", ".")] = v
    current_block = new_current_block
    torch.save(current_block, save_path)


def shard_on_the_fly(switch_checkpoint_path, dump_path, max_shard_size, dtype, weights_name: str = WEIGHTS_NAME):
    """
    动态分片检查点文件。
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)  # 将最大分片大小转换为整数
    sharded_state_dicts = []  # 存储分片后的状态字典列表
    current_block = {}  # 当前块的状态字典
    current_block_size = 0  # 当前块的大小
    total_size = 0  # 总共的大小

    os.makedirs(dump_path, exist_ok=True)  # 确保转储路径存在，不存在则创建

    # 从检查点文件中恢复信息并扁平化
    with gfile.GFile(switch_checkpoint_path + "/checkpoint", "rb") as fp:
        checkpoint_info = serialization.msgpack_restore(fp.read())["optimizer"]["target"]
        checkpoint_info = flatten_dict(checkpoint_info, sep="/")

    all_layers = {}  # 所有层的字典，用于存储层信息
    # 遍历检查点信息中的每个层名称
    for layer in checkpoint_info.keys():
        # 获取真实的层名称、分割后的层名称及内容，通过函数获取
        curr_real_layer_name, split_layer, content = get_key_and_tensorstore_dict(
            layer, checkpoint_info, switch_checkpoint_path
        )
        # 如果当前真实层名称已经存在于所有层的字典中
        if curr_real_layer_name in all_layers:
            # 将内容存入已有的真实层名称对应的字典中的分割层中的最后一部分
            all_layers[curr_real_layer_name][split_layer[-1]] = content
        else:
            # 创建新的真实层名称键，并存入内容
            all_layers[curr_real_layer_name] = {split_layer[-1]: content}

    # 遍历所有层的键
    for key in all_layers.keys():
        # 使用 tensorstore 打开未展开的字典格式的所有层的数据
        raw_weights = ts.open(unflatten_dict(all_layers[key])).result().read().result()
        # 将原始权重数据转换为 PyTorch 的张量格式
        raw_weights = torch.tensor(raw_weights)
        # 计算权重张量的字节大小
        weight_size = raw_weights.numel() * dtype_byte_size(raw_weights.dtype)

        # 使用小型转换脚本中的重命名模式对键和原始权重进行重命名
        key, raw_weights = rename_base_flax_keys(tuple(key.split("/")), raw_weights)
        # 重新连接重命名后的键
        key = "/".join(key)

        # 如果当前块的大小加上权重大小超过了最大碎片大小
        if current_block_size + weight_size > max_shard_size:
            # 构建保存路径，包含碎片编号
            save_path = os.path.join(
                dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
            )
            # 重命名并保存当前块
            rename_and_save_block(current_block, save_path)
            # 添加当前块的键到碎片状态字典中
            sharded_state_dicts.append(current_block.keys())
            # 删除当前块
            del current_block
            # 重新创建空的当前块和当前块大小
            current_block = {}
            current_block_size = 0

        # 将处理后的原始权重数据添加到当前块中，转换为指定的数据类型
        current_block[key] = raw_weights.to(getattr(torch, dtype))
        # 更新当前块大小
        current_block_size += weight_size
        # 更新总大小
        total_size += weight_size

    # 添加最后一个块
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    rename_and_save_block(current_block, save_path)
    sharded_state_dicts.append(current_block.keys())

    # 如果只有一个碎片，直接返回
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # 否则，构建索引
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        # 构建每个碎片文件的名称，包含碎片编号和总碎片数
        shard_file = weights_name.replace(
            ".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        )
        # 临时文件名，用于重命名到最终的碎片文件
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        # 实际重命名文件到最终的碎片文件
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        # 记录每个碎片文件对应的碎片状态字典
        shards[shard_file] = shard
        # 遍历每个碎片的键
        for key in shard:
            # 记录每个键对应的碎片文件名称
            weight_map[key] = shard_file

    # 添加元数据
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}

    # 将索引写入文件
    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    # 返回元数据和索引
    return metadata, index
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128/checkpoint_634600",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    # 添加可选参数 max_shard_size，用于指定最大分片大小，默认为 "10GB"
    parser.add_argument("--max_shard_size", default="10GB", required=False, help="Max shard size")
    # 添加可选参数 dtype，用于指定保存模型的数据类型，默认为 "bfloat16"
    parser.add_argument("--dtype", default="bfloat16", type=str, required=False, help="dtype of the saved model")
    # 添加可选参数 pytorch_dump_folder_path，用于指定 PyTorch 模型输出的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128-converted",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    # 解析命令行参数并存储到 args 对象中
    args = parser.parse_args()
    # 调用 shard_on_the_fly 函数，传递解析后的参数进行处理
    shard_on_the_fly(
        args.switch_t5x_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.max_shard_size,
        args.dtype,
    )



def sanity_check():
    # 导入所需的类和函数
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, T5Tokenizer

    # 加载 Switch 模型的配置文件
    config = SwitchTransformersConfig.from_pretrained("google/switch-base-8")
    # 将配置保存到指定路径
    config.save_pretrained("/home/arthur_huggingface_co/transformers/switch_converted")
    # 加载转换后的 Switch 模型
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        "/home/arthur_huggingface_co/transformers/switch_converted", device_map="auto"
    )

    # 加载 T5Tokenizer，用于处理文本输入
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    # 指定一个文本输入
    text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."

    # 使用 tokenizer 对文本进行编码，生成输入的 token ids
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    # 使用模型生成输出
    out = model.generate(input_ids, decoder_start_token_id=0)
    # 解码输出并打印结果
    print(tokenizer.decode(out[0]))
```