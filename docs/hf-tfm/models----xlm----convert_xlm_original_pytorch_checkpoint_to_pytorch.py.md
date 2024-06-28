# `.\models\xlm\convert_xlm_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert OpenAI GPT checkpoint."""


import argparse  # 导入用于解析命令行参数的模块
import json  # 导入处理 JSON 格式数据的模块

import numpy  # 导入处理数值运算的模块
import torch  # 导入 PyTorch 深度学习框架

from transformers.models.xlm.tokenization_xlm import VOCAB_FILES_NAMES  # 导入 XLM 模型的词汇文件名
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging  # 导入配置文件、权重文件名以及日志模块


logging.set_verbosity_info()  # 设置日志输出级别为 info


def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    # Load checkpoint 加载模型检查点文件
    chkpt = torch.load(xlm_checkpoint_path, map_location="cpu")

    state_dict = chkpt["model"]  # 获取模型的状态字典

    # We have the base model one level deeper than the original XLM repository
    # 将模型的字典键名做适当的修改，使其符合转换后的 PyTorch 模型结构
    two_levels_state_dict = {}
    for k, v in state_dict.items():
        if "pred_layer" in k:
            two_levels_state_dict[k] = v
        else:
            two_levels_state_dict["transformer." + k] = v

    config = chkpt["params"]  # 获取模型的参数配置信息
    config = {n: v for n, v in config.items() if not isinstance(v, (torch.FloatTensor, numpy.ndarray))}  # 过滤掉浮点数类型的配置项

    vocab = chkpt["dico_word2id"]  # 获取词汇表
    vocab = {s + "</w>" if s.find("@@") == -1 and i > 13 else s.replace("@@", ""): i for s, i in vocab.items()}  # 处理词汇表内容

    # Save pytorch-model 保存转换后的 PyTorch 模型权重文件
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    torch.save(two_levels_state_dict, pytorch_weights_dump_path)

    # Save configuration file 保存模型的配置文件
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2) + "\n")

    # Save vocab file 保存模型的词汇表文件
    pytorch_vocab_dump_path = pytorch_dump_folder_path + "/" + VOCAB_FILES_NAMES["vocab_file"]
    with open(pytorch_vocab_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    # Required parameters 必须的命令行参数
    parser.add_argument(
        "--xlm_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()  # 解析命令行参数
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)  # 调用转换函数进行转换
```