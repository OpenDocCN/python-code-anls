# `.\models\switch_transformers\convert_switch_transformers_original_flax_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，声明代码版权归 The HuggingFace Inc. team 所有
#
# 根据 Apache 许可证 2.0 版本，使用本文件需要遵循许可证的规定
# 详细信息请参考 http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律另有规定或书面同意，本软件是基于“按原样提供”的基础分发的，不提供任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证条款。

"""将 SwitchTransformersX 仓库的检查点转换为 JAX/FLAX 模型。"""

import argparse  # 导入用于解析命令行参数的模块
import re  # 导入正则表达式模块

from flax.traverse_util import flatten_dict, unflatten_dict  # 导入用于扁平化和反扁平化字典的工具函数
from t5x import checkpoints  # 导入 SwitchTransformersX 仓库的检查点处理模块

from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration  # 导入 Switch Transformers 相关模型配置和生成模型类
from transformers.modeling_flax_pytorch_utils import load_flax_weights_in_pytorch_model  # 导入用于加载 FLAX 权重到 PyTorch 模型的工具函数
from transformers.utils import logging  # 导入日志记录模块

logging.set_verbosity_info()  # 设置日志输出级别为 INFO

# 应该不包括由 `from_pt` 参数已经完成的内容
# 定义从原始模型到 Switch Transformers 的层名称映射字典
MOE_LAYER_NAME_MAPPING = {
    "/attention/": "/0/SelfAttention/",
    "/self_attention/": "/0/SelfAttention/",
    "/encoder_decoder_attention/": "/1/EncDecAttention/",
    "value": "v",
    "query": "q",
    "key": "k",
    "out": "o",
    "pre_self_attention_layer_norm": "0/layer_norm",
    "pre_cross_attention_layer_norm": "1/layer_norm",
    "pre_attention_layer_norm": "0/layer_norm",  # 先前为 1，但似乎是错误的
    "token_embedder": "shared",
    "encoder_norm": "final_layer_norm",
    "decoder_norm": "final_layer_norm",
    "relpos_bias/rel_embedding": "block/0/layer/0/SelfAttention/relative_attention_bias/weight",
    "router/router_weights/w/": "router/classifier/",
    "roer/roer_weights/w/": "router/classifier/",
    "logits_dense": "lm_head",
}

def rename_keys(s_dict):
    # 在 HF T5 中，我们有 block.{x}.layer.{y}. 对应于原始模型中的 layer.{x}
    # 返回字典 s_dict 的键列表
    keys = list(s_dict.keys())
    # 1. Convert keys based on specified patterns
    for key in keys:
        # Define pattern to match and transform "layers_<number>" to "block/<number>/layer"
        layer_to_block_of_layer = r".*/layers_(\d+)"
        new_key = key
        if re.match(layer_to_block_of_layer, key):
            new_key = re.sub(r"layers_(\d+)", r"block/\1/layer", new_key)

        # Define pattern to match and transform "encoder/" or "decoder/" paths
        layer_to_block_of_layer = r"(encoder|decoder)\/"
        if re.match(layer_to_block_of_layer, key):
            groups = re.match(layer_to_block_of_layer, new_key).groups()
            if groups[0] == "encoder":
                new_key = re.sub(r"/mlp/", r"/1/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/1/layer_norm/", new_key)
            elif groups[0] == "decoder":
                new_key = re.sub(r"/mlp/", r"/2/mlp/", new_key)
                new_key = re.sub(r"/pre_mlp_layer_norm/", r"/2/layer_norm/", new_key)

        # 2. Convert keys using predefined mapping dictionary MOE_LAYER_NAME_MAPPING
        for old_key, temp_key in MOE_LAYER_NAME_MAPPING.items():
            if old_key in new_key:
                new_key = new_key.replace(old_key, temp_key)

        # Print the transformation from original key to new key
        print(f"{key} -> {new_key}")

        # Replace the original key in the dictionary with the transformed new_key
        s_dict[new_key] = s_dict.pop(key)

    # Adjust specific entries in the dictionary based on their keys
    if "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "encoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T
    if "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight" in s_dict:
        s_dict["decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"] = s_dict[
            "decoder/block/0/layer/0/SelfAttention/relative_attention_bias/weight"
        ].T

    # 3. Handle keys containing "expert" separately
    for key in list(s_dict.keys()):
        if "expert" in key:
            # Extract the number of experts and their weights
            num_experts = s_dict[key].shape[0]
            expert_weights = s_dict[key]
            # Iterate over each expert, renaming and adding to the dictionary
            for idx in range(num_experts):
                s_dict[key.replace("expert/", f"experts/expert_{idx}/")] = expert_weights[idx]
                print(f"{key} -> {key.replace('expert/', f'experts/expert_{idx}/')}")

            # Remove the original "expert" key from the dictionary
            s_dict.pop(key)

    # Return the modified dictionary
    return s_dict
# GIN_TO_CONFIG_MAPPING 定义了从 GIN 配置参数到 SwitchTransformersConfig 参数的映射关系
GIN_TO_CONFIG_MAPPING = {
    "NUM_ENCODER_LAYERS": "num_layers",
    "NUM_DECODER_LAYERS": "num_decoder_layers",
    "NUM_HEADS": "num_heads",
    "HEAD_DIM": "d_kv",
    "EMBED_DIM": "d_model",
    "MLP_DIM": "d_ff",
    "NUM_SELECTED_EXPERTS": "num_selected_experts",
    "NUM_ENCODER_SPARSE_LAYERS": "num_sparse_encoder_layers",
    "NUM_DECODER_SPARSE_LAYERS": "num_sparse_decoder_layers",
    "dense.MlpBlock.activations": "feed_forward_proj",
}

def convert_gin_to_config(gin_file, num_experts):
    # 将 Google 风格的配置文件转换为 Hugging Face 格式的配置
    import regex as re
    
    # 从文件中读取 GIN 配置内容
    with open(gin_file, "r") as f:
        raw_gin = f.read()

    # 使用正则表达式匹配参数和值
    regex_match = re.findall(r"(.*) = ([0-9.]*)", raw_gin)
    args = {}
    for param, value in regex_match:
        # 根据预定义的映射将参数名转换为 SwitchTransformersConfig 的参数名，并将值转换为相应类型
        if param in GIN_TO_CONFIG_MAPPING and value != "":
            args[GIN_TO_CONFIG_MAPPING[param]] = float(value) if "." in value else int(value)

    # 提取激活函数类型，并添加到参数字典中
    activation = re.findall(r"(.*activations) = \(\'(.*)\',\)", raw_gin)[0]
    args[GIN_TO_CONFIG_MAPPING[activation[0]]] = str(activation[1])

    # 添加 num_experts 参数到参数字典中
    args["num_experts"] = num_experts
    
    # 使用参数创建 SwitchTransformersConfig 对象
    config = SwitchTransformersConfig(**args)
    return config


def convert_flax_checkpoint_to_pytorch(
    flax_checkpoint_path, config_file, gin_file=None, pytorch_dump_path="./", num_experts=8
):
    # 初始化 PyTorch 模型

    # 打印正在加载的 flax 权重路径
    print(f"Loading flax weights from : {flax_checkpoint_path}")
    
    # 加载 flax 模型的参数
    flax_params = checkpoints.load_t5x_checkpoint(flax_checkpoint_path)

    if gin_file is not None:
        # 如果提供了 gin 文件，则根据 gin 文件和 num_experts 转换为 SwitchTransformersConfig 对象
        config = convert_gin_to_config(gin_file, num_experts)
    else:
        # 否则根据 config_file 创建 SwitchTransformersConfig 对象
        config = SwitchTransformersConfig.from_pretrained(config_file)

    # 使用配置文件创建 SwitchTransformersForConditionalGeneration 模型
    pt_model = SwitchTransformersForConditionalGeneration(config)

    # 将 flax 参数扁平化，重命名键名后再还原为字典
    flax_params = flax_params["target"]
    flax_params = flatten_dict(flax_params, sep="/")
    flax_params = rename_keys(flax_params)
    flax_params = unflatten_dict(flax_params, sep="/")

    # 加载 flax 参数到 PyTorch 模型中
    load_flax_weights_in_pytorch_model(pt_model, flax_params)

    # 打印保存 PyTorch 模型的路径
    print(f"Save PyTorch model to {pytorch_dump_path}")
    pt_model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必选参数
    parser.add_argument(
        "--switch_t5x_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained SwitchTransformers model. \nThis specifies the"
            " model architecture. If not provided, a `gin_file` has to be provided."
        ),
    )
    # 可选参数
    parser.add_argument(
        "--gin_file",
        default=None,
        type=str,
        required=False,
        help="Path to the gin config file. If not provided, a `config_file` has to be passed   ",
    )
    parser.add_argument(
        "--config_name", default=None, type=str, required=False, help="Config name of SwitchTransformers model."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output pytorch model."
    )
    parser.add_argument("--num_experts", default=8, type=int, required=False, help="Number of experts")
    args = parser.parse_args()
    convert_flax_checkpoint_to_pytorch(
        args.switch_t5x_checkpoint_path,
        args.config_name,
        args.gin_file,
        args.pytorch_dump_folder_path,
        args.num_experts,
    )



# 添加一个命令行参数，指定输出 PyTorch 模型的路径，参数为必填项
parser.add_argument(
    "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output pytorch model."
)
# 添加一个命令行参数，指定专家数量，默认为 8，非必填项
parser.add_argument("--num_experts", default=8, type=int, required=False, help="Number of experts")
# 解析命令行参数并将其存储在 args 变量中
args = parser.parse_args()
# 调用函数 convert_flax_checkpoint_to_pytorch，将 Flax 模型转换为 PyTorch 模型
convert_flax_checkpoint_to_pytorch(
    args.switch_t5x_checkpoint_path,  # Flax 模型的路径
    args.config_name,  # 配置名称
    args.gin_file,  # GIN 文件路径
    args.pytorch_dump_folder_path,  # 输出的 PyTorch 模型路径
    args.num_experts,  # 专家数量
)
```