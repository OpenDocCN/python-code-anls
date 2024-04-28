# `.\transformers\models\bloom\convert_bloom_original_checkpoint_to_pytorch.py`

```
# 指定 Python 文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可 2.0 版本（"许可"）获得许可；
# 除非符合许可的法律要求或书面同意，否则您不得使用此文件。
# 您可以在以下网址获得许可副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件按"原样"分发，
# 没有任何形式的明示或暗示担保或条件。
# 有关特定语言的权限，请参阅许可证。
"""转换 BigScience BLOOM 检查点。"""


import argparse  # 导入命令行参数解析模块
import json  # 导入 JSON 操作模块
import os  # 导入操作系统功能模块
import re  # 导入正则表达式模块

import torch  # 导入 PyTorch 模块

# 导入 transformers 库中的相关模块和函数
from transformers import BloomConfig, BloomModel
from transformers.file_utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils import logging  # 导入日志记录模块


logging.set_verbosity_info()  # 设置日志记录的详细程度为 info

# 需要进行平均的权重结尾
WEIGHTS_TO_AVERAGE_ENDSWITH = [
    "word_embeddings_layernorm.weight",
    "word_embeddings_layernorm.bias",
    "input_layernorm.weight",
    "input_layernorm.bias",
    "post_attention_layernorm.weight",
    "post_attention_layernorm.bias",
    "self_attention.dense.bias",
    "mlp.dense_4h_to_h.bias",
    "ln_f.weight",
    "ln_f.bias",
]

# 具有行并行性的权重包含
WEIGHTS_WITH_ROW_PARALLELISM_CONTAIN = [
    "mlp.dense_4h_to_h.weight",
    "self_attention.dense.weight",
]


def layer_name_mapping(key, file):
    """将 transformers 中的 TP/PP 权重映射转换为 Megatron-DeepSpeed 中的 TP 权重"""
    # 处理第一个和最后一个层
    layer_rename_map = {
        "word_embeddings.weight": "word_embeddings.weight",
        "word_embeddings.norm.weight": "word_embeddings_layernorm.weight",
        "word_embeddings.norm.bias": "word_embeddings_layernorm.bias",
        "weight": "ln_f.weight",
        "bias": "ln_f.bias",
    }

    if key in layer_rename_map:
        return layer_rename_map[key]

    # 处理 transformer blocks
    layer_number = int(re.match(r".*layer_(\d*).*", file)[1])
    layer_number -= 3
    return f"h.{layer_number}." + key


def get_dtype_size(dtype):
    if dtype == torch.bool:
        return 1 / 8
    bit_search = re.search(r"[^\d](\d+)$", str(dtype))
    if bit_search is None:
        raise ValueError(f"`dtype` is not a valid dtype: {dtype}.")
    bit_size = int(bit_search.groups()[0])
    return bit_size // 8


def convert_bloom_checkpoint_to_pytorch(
    bloom_checkpoint_path, bloom_config_file, pytorch_dump_folder_path, shard_model, pretraining_tp
):
    # 构造模型
    if bloom_config_file == "":
        config = BloomConfig()
    else:
        config = BloomConfig.from_json_file(bloom_config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # 必需参数
    parser.add_argument(
        "--bloom_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the Megatron-LM checkpoint path.",
    )
    # 添加命令行参数，指定输出 PyTorch 模型的路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 添加命令行参数，指定预训练模型的配置文件路径
    parser.add_argument(
        "--bloom_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained model. \n"
            "This specifies the model architecture."
        ),
    )
    # 添加命令行参数，用于设置是否对输出模型进行分片
    parser.add_argument(
        "--shard_model",
        action="store_true",
        help="An optional setting to shard the output model \nThis enables sharding the converted checkpoint",
    )
    # 添加命令行参数，指定在 Megatron-LM 训练模型时使用的预训练 TP 等级
    parser.add_argument(
        "--pretraining_tp",
        default=4,
        type=int,
        help="Pretraining TP rank that has been used when training the model in Megatron-LM \n",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 Bloom 格式的检查点转换为 PyTorch 格式
    convert_bloom_checkpoint_to_pytorch(
        args.bloom_checkpoint_path,
        args.bloom_config_file,
        args.pytorch_dump_folder_path,
        args.shard_model,
        args.pretraining_tp,
    )
```