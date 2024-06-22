# `.\transformers\models\byt5\convert_byt5_original_tf_checkpoint_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 T5 作者和 HuggingFace 公司所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""Convert T5 checkpoint."""

# 导入必要的库
import argparse
# 从 transformers 库中导入 T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
from transformers import T5Config, T5ForConditionalGeneration, load_tf_weights_in_t5
# 从 transformers.utils 中导入 logging
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()

# 定义函数，将 TensorFlow checkpoint 转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # 初始化 PyTorch 模型
    config = T5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = T5ForConditionalGeneration(config)

    # 从 TensorFlow checkpoint 加载权重
    load_tf_weights_in_t5(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

# 如果作为独立脚本运行
if __name__ == "__main__":
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
    args = parser.parse_args()
    # 调用函数，将 TensorFlow checkpoint 转换为 PyTorch 模型
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
```