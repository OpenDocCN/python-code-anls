# `.\transformers\models\xlm\convert_xlm_original_pytorch_checkpoint_to_pytorch.py`

```
# coding=utf-8
# 定义文件编码格式为 UTF-8
# 版权声明
# Copyright 2018 The HuggingFace Inc. team.
# 版权声明信息
# Licensed under the Apache License, Version 2.0 (the "License");
# 依据 Apache License 2.0 许可使用该文件
# You may not use this file except in compliance with the License;
# 除非符合许可使用，否则不得使用此文件
# You may obtain a copy of the License at
# 你可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，以"原样"分发的软件
# 不提供任何形式的担保或条件，无论是明示的还是暗示的。
# 请查阅许可证以了解具体语言管理权限和许可的限制准则
# 转换 OpenAI GPT 检查点
# 导入所需模块
import argparse
import json
# 导入numpy
import numpy
# 导入torch模块
import torch
# 从transformers模块中导入xlm模型的tokenization_xlm文件
from transformers.models.xlm.tokenization_xlm import VOCAB_FILES_NAMES
# 从transformers.utils模块中导入CONFIG_NAME、WEIGHTS_NAME、logging
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging
# 设置日志的详细程度为info
logging.set_verbosity_info()
# 定义一个函数，将XLM检查点转换为PyTorch模型
def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    # 加载检查点
    chkpt = torch.load(xlm_checkpoint_path, map_location="cpu")
    # 从检查点中获取模型状态字典
    state_dict = chkpt["model"]
    # 将模型状态字典调整为两级字典
    two_levels_state_dict = {}
    for k, v in state_dict.items():
        if "pred_layer" in k:
            two_levels_state_dict[k] = v
        else:
            two_levels_state_dict["transformer." + k] = v
    # 从检查点中获取参数
    config = chkpt["params"]
    # 从参数中排除torch.FloatTensor、numpy.ndarray类型的值
    config = {n: v for n, v in config.items() if not isinstance(v, (torch.FloatTensor, numpy.ndarray))}
    # 从检查点中获取词汇表
    vocab = chkpt["dico_word2id"]
    # 对词汇表进行处理，将@@替换为空格，添加"</w>"和替换词进行处理
    vocab = {s + "</w>" if s.find("@@") == -1 and i > 13 else s.replace("@@", ""): i for s, i in vocab.items()}
    # 保存PyTorch模型
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    torch.save(two_levels_state_dict, pytorch_weights_dump_path)
    # 保存配置文件
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2) + "\n")
    # 保存词汇表文件
    pytorch_vocab_dump_path = pytorch_dump_folder_path + "/" + VOCAB_FILES_NAMES["vocab_file"]
    with open(pytorch_vocab_dump_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vocab, indent=2) + "\n")
# 如果是主程序
if __name__ == "__main__":
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--xlm_checkpoint_path", default=None, type=str, required=True, help="Path the official PyTorch dump."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析参数
    args = parser.parse_args()
    # 调用函数，转换XLM检查点为PyTorch模型
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)
```