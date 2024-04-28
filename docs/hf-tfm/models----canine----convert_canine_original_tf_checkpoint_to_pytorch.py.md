# `.\transformers\models\canine\convert_canine_original_tf_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
"""将 CANINE 检查点转换为 PyTorch 检查点"""


# 导入所需的库
import argparse

from transformers import CanineConfig, CanineModel, CanineTokenizer, load_tf_weights_in_canine
from transformers.utils import logging


# 设置日志级别为 info
logging.set_verbosity_info()


# 定义函数，将 TensorFlow 检查点转换为 PyTorch 检查点
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, pytorch_dump_path):
    # 初始化 PyTorch 模型配置
    config = CanineConfig()
    # 创建 CANINE 模型
    model = CanineModel(config)
    # 设置为评估模式
    model.eval()

    # 打印正在构建的 PyTorch 模型的配置信息
    print(f"Building PyTorch model from configuration: {config}")

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_canine(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型（权重和配置）
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # 保存分词器文件
    tokenizer = CanineTokenizer()
    print(f"Save tokenizer files to {pytorch_dump_path}")
    tokenizer.save_pretrained(pytorch_dump_path)


# 如果作为独立脚本运行
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--tf_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the TensorFlow checkpoint. Should end with model.ckpt",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to a folder where the PyTorch model will be placed.",
    )
    # 解析参数
    args = parser.parse_args()
    # 调用函数，将 TensorFlow 检查点转换为 PyTorch 检查点
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.pytorch_dump_path)
```