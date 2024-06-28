# `.\models\big_bird\convert_bigbird_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert BigBird checkpoint."""

# 引入处理命令行参数的库
import argparse

# 引入 BigBird 相关的模型配置和加载权重的方法
from transformers import BigBirdConfig, BigBirdForPreTraining, BigBirdForQuestionAnswering, load_tf_weights_in_big_bird
from transformers.utils import logging

# 设置日志的输出级别为信息级别
logging.set_verbosity_info()

# 定义函数，用于将 TensorFlow 的 checkpoint 转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, big_bird_config_file, pytorch_dump_path, is_trivia_qa):
    # 从 JSON 文件中读取 BigBird 的配置
    config = BigBirdConfig.from_json_file(big_bird_config_file)
    # 打印正在根据配置构建 PyTorch 模型
    print(f"Building PyTorch model from configuration: {config}")

    # 根据是否是 TriviaQA 模型选择相应的 BigBird 模型
    if is_trivia_qa:
        model = BigBirdForQuestionAnswering(config)
    else:
        model = BigBirdForPreTraining(config)

    # 加载 TensorFlow checkpoint 中的权重到 PyTorch 模型
    load_tf_weights_in_big_bird(model, tf_checkpoint_path, is_trivia_qa=is_trivia_qa)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    # 必须参数：TensorFlow checkpoint 的路径
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 必须参数：BigBird 模型的配置文件路径
    parser.add_argument(
        "--big_bird_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained BERT model. \n"
            "This specifies the model architecture."
        ),
    )
    # 必须参数：输出的 PyTorch 模型路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 可选参数：是否包含 TriviaQA 头部
    parser.add_argument(
        "--is_trivia_qa", action="store_true", help="Whether to convert a model with a trivia_qa head."
    )
    # 解析参数
    args = parser.parse_args()
    # 调用函数，执行 TensorFlow 到 PyTorch 模型的转换
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.big_bird_config_file, args.pytorch_dump_path, args.is_trivia_qa
    )
```