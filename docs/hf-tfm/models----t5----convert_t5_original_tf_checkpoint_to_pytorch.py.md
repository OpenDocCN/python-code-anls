# `.\transformers\models\t5\convert_t5_original_tf_checkpoint_to_pytorch.py`

```py
# 设置文件编码格式为 utf-8
# 版权归 2018 年 T5 作者和 HuggingFace 公司所有
#
# 根据 Apache 许可证 Version 2.0 许可
# 除非合规使用该文件，否则您不得使用此文件
# 您可以在以下网址获取许可副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则依照 "原样" 分发软件
# 没有任何明示或暗示的保证或条件
# 请查看许可证，以了解具体语言规定的权限和限制
"""Convert T5 checkpoint."""

import argparse

# 从 transformers 库中导入 T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
# 从 transformers.utils 中导入 logging
from transformers.utils import logging

# 设置日志输出级别为 info
logging.set_verbosity_info()

# 定义函数，将 TensorFlow 检查点转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # 根据 JSON 文件创建 T5Config 对象
    config = T5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 基于配置创建 T5ForConditionalGeneration 模型
    model = T5ForConditionalGeneration(config)

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

# 判断是否为主程序
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained T5 model. \nThis specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析参数
    args = parser.parse_args()
    # 调用转换函数
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
```