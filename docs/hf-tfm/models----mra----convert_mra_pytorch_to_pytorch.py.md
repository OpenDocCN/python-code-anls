# `.\transformers\models\mra\convert_mra_pytorch_to_pytorch.py`

```
# coding=utf-8
# 指定文件编码为 UTF-8
# Copyright 2023 The HuggingFace Inc. team.
# 声明版权所有者为 HuggingFace Inc. 团队
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 2.0 版进行授权
# you may not use this file except in compliance with the License.
# 不得违反许可证的情况下使用此文件
# You may obtain a copy of the License at
# 可以在以下位置获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证的条款以"原样"分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 不提供任何明示或暗示的担保或条件
# See the License for the specific language governing permissions and
# 查看许可证中规定的权限和
# limitations under the License.
"""Convert MRA checkpoints from the original repository. URL: https://github.com/mlpen/mra-attention"""
# 该脚本的目的是将 MRA 检查点从原始仓库转换
import argparse
# 导入 argparse 模块用于解析命令行参数

import torch
# 导入 PyTorch 库

from transformers import MraConfig, MraForMaskedLM
# 从 transformers 库导入 MraConfig 和 MraForMaskedLM 类

def rename_key(orig_key):
    # 定义一个函数用于重命名 PyTorch 模型的键
    if "model" in orig_key:
        # 如果键包含 "model"，则删除 "model." 前缀
        orig_key = orig_key.replace("model.", "")
    if "norm1" in orig_key:
        # 如果键包含 "norm1"，则将其替换为 "attention.output.LayerNorm"
        orig_key = orig_key.replace("norm1", "attention.output.LayerNorm")
    if "norm2" in orig_key:
        # 如果键包含 "norm2"，则将其替换为 "output.LayerNorm"
        orig_key = orig_key.replace("norm2", "output.LayerNorm")
    if "norm" in orig_key:
        # 如果键包含 "norm"，则将其替换为 "LayerNorm"
        orig_key = orig_key.replace("norm", "LayerNorm")
    if "transformer" in orig_key:
        # 如果键包含 "transformer"，则根据层数替换为 "encoder.layer.{layer_num}"
        layer_num = orig_key.split(".")[0].split("_")[-1]
        orig_key = orig_key.replace(f"transformer_{layer_num}", f"encoder.layer.{layer_num}")
    if "mha.attn" in orig_key:
        # 如果键包含 "mha.attn"，则将其替换为 "attention.self"
        orig_key = orig_key.replace("mha.attn", "attention.self")
    if "mha" in orig_key:
        # 如果键包含 "mha"，则将其替换为 "attention"
        orig_key = orig_key.replace("mha", "attention")
    if "W_q" in orig_key:
        # 如果键包含 "W_q"，则将其替换为 "self.query"
        orig_key = orig_key.replace("W_q", "self.query")
    if "W_k" in orig_key:
        # 如果键包含 "W_k"，则将其替换为 "self.key"
        orig_key = orig_key.replace("W_k", "self.key")
    if "W_v" in orig_key:
        # 如果键包含 "W_v"，则将其替换为 "self.value"
        orig_key = orig_key.replace("W_v", "self.value")
    if "ff.0" in orig_key:
        # 如果键包含 "ff.0"，则将其替换为 "intermediate.dense"
        orig_key = orig_key.replace("ff.0", "intermediate.dense")
    if "ff.2" in orig_key:
        # 如果键包含 "ff.2"，则将其替换为 "output.dense"
        orig_key = orig_key.replace("ff.2", "output.dense")
    if "ff" in orig_key:
        # 如果键包含 "ff"，则将其替换为 "output.dense"
        orig_key = orig_key.replace("ff", "output.dense")
    if "mlm_class" in orig_key:
        # 如果键包含 "mlm_class"，则将其替换为 "cls.predictions.decoder"
        orig_key = orig_key.replace("mlm.mlm_class", "cls.predictions.decoder")
    if "mlm" in orig_key:
        # 如果键包含 "mlm"，则将其替换为 "cls.predictions.transform"
        orig_key = orig_key.replace("mlm", "cls.predictions.transform")
    if "backbone.backbone.encoders" in orig_key:
        # 如果键包含 "backbone.backbone.encoders"，则将其替换为 "encoder.layer"
        orig_key = orig_key.replace("backbone.backbone.encoders", "encoder.layer")
    if "cls" not in orig_key:
        # 如果键不包含 "cls"，则在前面添加 "mra."
        orig_key = "mra." + orig_key

    return orig_key
# 返回重命名后的键

def convert_checkpoint_helper(max_position_embeddings, orig_state_dict):
    # 定义一个函数用于转换检查点
    for key in orig_state_dict.copy().keys():
        # 遍历原始状态字典的键
        val = orig_state_dict.pop(key)
        # 获取对应的值并从原始状态字典中删除该键值对

        if ("pooler" in key) or ("sen_class" in key):
            # 如果键包含 "pooler" 或 "sen_class"，则跳过该键值对
            continue
        else:
            # 否则使用 rename_key 函数重命名键，并将键值对添加到新的状态字典中
            orig_state_dict[rename_key(key)] = val

    # 将 "cls.predictions.bias" 的值设置为 "cls.predictions.decoder.bias"
    orig_state_dict["cls.predictions.bias"] = orig_state_dict["cls.predictions.decoder.bias"]
    # 创建一个从 2 开始的位置 ID 序列，长度为 max_position_embeddings
    orig_state_dict["mra.embeddings.position_ids"] = torch.arange(max_position_embeddings).expand((1, -1)) + 2
    # 返回原始的 state_dict
    return orig_state_dict
# 将 MRA 模型的检查点转换为 PyTorch 格式
def convert_mra_checkpoint(checkpoint_path, mra_config_file, pytorch_dump_path):
    # 使用 CPU 加载 MRA 模型的原始状态字典
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model_state_dict"]
    # 从 JSON 文件中加载 MRA 模型的配置
    config = MraConfig.from_json_file(mra_config_file)
    # 根据配置创建 MRA 语言模型对象
    model = MraForMaskedLM(config)

    # 调用辅助函数，将原始状态字典转换为新的状态字典
    new_state_dict = convert_checkpoint_helper(config.max_position_embeddings, orig_state_dict)

    # 打印模型加载状态，并加载新的状态字典
    print(model.load_state_dict(new_state_dict))
    # 设置模型为评估模式
    model.eval()
    # 保存转换后的模型到指定路径
    model.save_pretrained(pytorch_dump_path)

    # 打印转换成功信息，包含保存路径
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--pytorch_model_path", default=None, type=str, required=True, help="Path to Mra pytorch checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for Mra model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入命令行参数
    convert_mra_checkpoint(args.pytorch_model_path, args.config_file, args.pytorch_dump_path)
```