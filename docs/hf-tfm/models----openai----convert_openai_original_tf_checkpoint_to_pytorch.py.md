# `.\transformers\models\openai\convert_openai_original_tf_checkpoint_to_pytorch.py`

```py
# 引入编码方式 UTF-8 和 Hugging Face Inc. 团队版权信息
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# 导入 Apache 2.0 许可证相关内容
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# you may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，软件根据"按原样"分发,
# 没有任何明示或暗示的担保或条件。
# See the License for the specific language governing permissions and
# limitations under the License.

# 描述文件的用途：将 OpenAI GPT 检查点转换为 PyTorch 模型

import argparse # 导入参数解析库

import torch # 导入 PyTorch 库

# 导入 OpenAIGPTConfig、OpenAIGPTModel 和 load_tf_weights_in_openai_gpt 函数
from transformers import OpenAIGPTConfig, OpenAIGPTModel, load_tf_weights_in_openai_gpt
# 导入 CONFIG_NAME、WEIGHTS_NAME 和 logging 工具
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging

# 设置日志输出级别为 info
logging.set_verbosity_info()

# 定义转换 OpenAI GPT 检查点到 PyTorch 模型的函数
def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    # 根据配置文件创建 OpenAIGPTModel 模型
    if openai_config_file == "":
        config = OpenAIGPTConfig()
    else:
        config = OpenAIGPTConfig.from_json_file(openai_config_file)
    model = OpenAIGPTModel(config)

    # 从 NumPy 文件中加载权重到模型
    load_tf_weights_in_openai_gpt(model, config, openai_checkpoint_folder_path)

    # 保存 PyTorch 模型
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"Save configuration file to {pytorch_config_dump_path}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())

# 主函数入口
if __name__ == "__main__":
    # 定义命令行参数解析器
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--openai_checkpoint_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint path.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--openai_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained OpenAI model. \n"
            "This specifies the model architecture."
        ),
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数
    convert_openai_checkpoint_to_pytorch(
        args.openai_checkpoint_folder_path, args.openai_config_file, args.pytorch_dump_folder_path
    )
```