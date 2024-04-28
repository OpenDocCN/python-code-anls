# `.\transformers\models\switch_transformers\convert_big_switch.py`

```py
import argparse
import json
import os

import tensorstore as ts  # 导入 tensorstore 库，用于处理数据存储和访问
import torch  # 导入 PyTorch 库，用于深度学习任务
from flax import serialization  # 从 flax 库中导入序列化模块，用于对象序列化和反序列化
from flax.traverse_util import flatten_dict, unflatten_dict  # 从 flax 库中导入遍历工具，用于扁平化和恢复字典
from tensorflow.io import gfile  # 从 tensorflow 库中导入 gfile 模块，用于文件操作

from transformers.modeling_utils import dtype_byte_size  # 从 transformers 库中导入模型工具，用于处理数据类型大小
from transformers.models.switch_transformers.convert_switch_transformers_original_flax_checkpoint_to_pytorch import (
    rename_keys,
)  # 从 transformers 库中导入模型转换工具，用于将 Switch Transformers 模型转换为 PyTorch 格式
from transformers.utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME  # 从 transformers 库中导入工具，用于处理权重文件名
from transformers.utils.hub import convert_file_size_to_int  # 从 transformers 库中导入 Hub 工具，用于文件大小转换

def rename_base_flax_keys(flax_key_tuple, flax_tensor):
    """
    Post renaming of basic JAX keys to pytorch.
    对基本的 JAX 键进行重命名为 PyTorch 键。
    """
    if flax_key_tuple[-1] == "kernel" and flax_tensor.ndim == 3:
        # 如果是 expert layer，调整键名为 "weight"，并对张量进行维度重排
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
        flax_tensor = torch.permute(flax_tensor, (0, 2, 1))
    elif flax_key_tuple[-1] == "kernel" and ".".join(flax_key_tuple):
        # 如果是 linear layer，调整键名为 "weight"，并对张量进行转置
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)
        flax_tensor = flax_tensor.T
    elif flax_key_tuple[-1] in ["scale", "embedding"]:
        # 如果是 scale 或 embedding 层，调整键名为 "weight"
        flax_key_tuple = flax_key_tuple[:-1] + ("weight",)

    return flax_key_tuple, flax_tensor

def get_key_and_tensorstore_dict(layer, checkpoint_info, switch_checkpoint_path):
    if "metadata" in layer:
        # 如果层名称包含 "metadata"，则对层名称进行分割和修正
        split_layer = layer.split("metadata")
        curr_real_layer_name = "".join(split_layer[0])[:-1]
        split_layer = [tuple(("metadata" + split_layer[1]).split("/"))]
    elif "kvstore" in layer:
        # 如果层名称包含 "kvstore"，则对层名称进行分割和修正
        split_layer = layer.split("kvstore")
        curr_real_layer_name = "".join(split_layer[0])[:-1]
        split_layer = [tuple(("kvstore" + split_layer[1]).split("/"))]
    else:
        # 否则，直接对层名称进行分割
        split_layer = layer.split("/")
        curr_real_layer_name = "/".join(split_layer[:-1])
        split_layer[-1] = (split_layer[-1],)

    if "kvstore/path" in layer:
        # 如果层名称包含 "kvstore/path"，则生成文件路径
        content = f"{switch_checkpoint_path}/{checkpoint_info[layer]}"
    elif "kvstore/driver" in layer:
        # 如果层名称包含 "kvstore/driver"，则设置内容为 "file"
        content = "file"
    else:
        # 否则，直接获取层的内容
        content = checkpoint_info[layer]

    return curr_real_layer_name, split_layer, content

def rename_and_save_block(current_block, save_path):
    """
    Renames keys in the current block and saves it to the specified path.
    对当前块中的键进行重命名，并将其保存到指定路径。
    """
    current_block = rename_keys(current_block)  # 使用转换工具重命名当前块的键
    new_current_block = {}
    for k, v in current_block.items():
        new_current_block[k.replace("/", ".")] = v  # 将键中的斜杠替换为点
    current_block = new_current_block
    torch.save(current_block, save_path)  # 使用 PyTorch 保存当前块到指定路径

def shard_on_the_fly(switch_checkpoint_path, dump_path, max_shard_size, dtype, weights_name: str = WEIGHTS_NAME):
    """
    Shards the checkpoint file based on the specified maximum shard size.
    根据指定的最大分片大小对检查点文件进行分片。
    """
    max_shard_size = convert_file_size_to_int(max_shard_size)  # 将最大分片大小转换为整数
    sharded_state_dicts = []
    current_block = {}
    current_block_size = 0
    total_size = 0

    os.makedirs(dump_path, exist_ok=True)  # 创建保存路径

    with gfile.GFile(switch_checkpoint_path + "/checkpoint", "rb") as fp:
        # 读取检查点文件中的信息
        checkpoint_info = serialization.msgpack_restore(fp.read())["optimizer"]["target"]
        checkpoint_info = flatten_dict(checkpoint_info, sep="/")  # 扁平化字典

    all_layers = {}  # 存储所有层信息的字典
```  
    # 遍历检查点信息字典的键
    for layer in checkpoint_info.keys():
        # 获得当前真实层名称、分割层和内容
        curr_real_layer_name, split_layer, content = get_key_and_tensorstore_dict(
            layer, checkpoint_info, switch_checkpoint_path
        )
        # 如果当前真实层名称在所有层的键中
        if curr_real_layer_name in all_layers:
            # 将内容添加到所有层中的当前真实层名称的最后一部分
            all_layers[curr_real_layer_name][split_layer[-1]] = content
        else:
            # 否则，将内容作为新字典添加到所有层中的当前真实层名称中
            all_layers[curr_real_layer_name] = {split_layer[-1]: content}

    # 遍历所有层的键
    for key in all_layers.keys():
        # 打开张量存储文件
        raw_weights = ts.open(unflatten_dict(all_layers[key])).result().read().result()
        # 将原始权重转换为张量
        raw_weights = torch.tensor(raw_weights)
        # 计算权重大小
        weight_size = raw_weights.numel() * dtype_byte_size(raw_weights.dtype)

        # 使用小型转换脚本的重命名模式
        key, raw_weights = rename_base_flax_keys(tuple(key.split("/")), raw_weights)
        # 重组键
        key = "/".join(key)

        # 如果当前块大小加上权重大小超过最大分片大小
        if current_block_size + weight_size > max_shard_size:
            # 保存路径为转储路径加上权重名称，并添加分片编号
            save_path = os.path.join(
                dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin")
            )
            # 重命名并保存当前块
            rename_and_save_block(current_block, save_path)
            # 将当前块的键添加到分片状态字典列表中
            sharded_state_dicts.append(current_block.keys())
            # 删除当前块
            del current_block
            # 重新初始化当前块和大小
            current_block = {}
            current_block_size = 0

        # 将原始权重添加到当前块中，并转换为指定数据类型
        current_block[key] = raw_weights.to(getattr(torch, dtype))
        # 更新当前块大小
        current_block_size += weight_size
        # 更新总大小
        total_size += weight_size

    # 添加最后一个块
    save_path = os.path.join(dump_path, weights_name.replace(".bin", f"-{len(sharded_state_dicts)+1:05d}-of-???.bin"))
    rename_and_save_block(current_block, save_path)
    sharded_state_dicts.append(current_block.keys())

    # 如果只有一个分片，直接返回
    if len(sharded_state_dicts) == 1:
        return {weights_name: sharded_state_dicts[0]}, None

    # 否则，构建索引
    weight_map = {}
    shards = {}
    for idx, shard in enumerate(sharded_state_dicts):
        # 创建分片文件名
        shard_file = weights_name.replace(
            ".bin", f"-{idx+1:05d}-of-{len(sharded_state_dicts):05d}.bin"
        )
        # 重命名分片文件
        temp_filename = os.path.join(dump_path, weights_name.replace(".bin", f"-{idx+1:05d}-of-???.bin"))
        os.rename(temp_filename, os.path.join(dump_path, shard_file))
        # 添加分片和对应的键到字典中
        shards[shard_file] = shard
        for key in shard:
            weight_map[key] = shard_file

    # 添加元数据
    metadata = {"total_size": total_size}
    index = {"metadata": metadata, "weight_map": weight_map}

    # 将索引写入文件
    with open(os.path.join(dump_path, WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    return metadata, index
# 当作为主程序运行时执行的代码
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 定义必需的命令行参数
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128/checkpoint_634600",
        type=str,
        required=False,
        help="Path to a directory containing a folder per layer. Follows the original Google format.",
    )
    parser.add_argument("--max_shard_size", default="10GB", required=False, help="Max shard size")
    parser.add_argument("--dtype", default="bfloat16", type=str, required=False, help="dtype of the saved model")
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/mnt/disks/disk_switch/original_checkpoints/switch-xxl-128-converted",
        type=str,
        required=False,
        help="Path to the output pytorch model.",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 shard_on_the_fly 函数，传入解析得到的参数
    shard_on_the_fly(
        args.switch_t5x_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.max_shard_size,
        args.dtype,
    )


# 定义一个名为 sanity_check 的函数
def sanity_check():
    # 导入相关的模块
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration, T5Tokenizer

    # 创建 SwitchTransformersConfig 对象，从 "google/switch-base-8" 预训练模型中获取
    config = SwitchTransformersConfig.from_pretrained("google/switch-base-8")
    # 将配置保存到指定路径
    config.save_pretrained("/home/arthur_huggingface_co/transformers/switch_converted")
    # 创建 SwitchTransformersForConditionalGeneration 对象，从保存的路径中加载
    model = SwitchTransformersForConditionalGeneration.from_pretrained(
        "/home/arthur_huggingface_co/transformers/switch_converted", device_map="auto"
    )

    # 创建 T5Tokenizer 对象，从 "t5-small" 预训练模型中获取
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # 定义一个示例输入文本
    text = "A <extra_id_0> walks into a bar a orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>."

    # 使用 tokenizer 处理输入文本，得到输入 ID 序列
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    # 使用模型生成输出
    out = model.generate(input_ids, decoder_start_token_id=0)
    # 使用 tokenizer 解码输出，并打印结果
    print(tokenizer.decode(out[0]))
```