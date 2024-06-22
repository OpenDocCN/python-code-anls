# `.\transformers\models\albert\convert_albert_original_tf_checkpoint_to_pytorch.py`

```py
# 引入 argparse 模块，用于解析命令行参数
import argparse
# 引入 torch 模块
import torch
# 引入 logging 模块，用于日志记录
from ...utils import logging
# 从当前目录的 __init__.py 中引入 AlbertConfig、AlbertForPreTraining 类以及 load_tf_weights_in_albert 函数
from . import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert

# 设置日志级别为 info
logging.set_verbosity_info()

# 定义函数 convert_tf_checkpoint_to_pytorch，用于将 TensorFlow 的检查点转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, albert_config_file, pytorch_dump_path):
    # 从 JSON 文件中读取 AlbertConfig 配置
    config = AlbertConfig.from_json_file(albert_config_file)
    # 打印配置信息
    print(f"Building PyTorch model from configuration: {config}")
    # 根据配置创建 AlbertForPreTraining 模型
    model = AlbertForPreTraining(config)

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_albert(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    # 将模型状态字典保存到指定路径
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 必选参数：TensorFlow 检查点路径
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 必选参数：预训练 ALBERT 模型的配置文件路径
    parser.add_argument(
        "--albert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained ALBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    # 必选参数：输出的 PyTorch 模型路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数 convert_tf_checkpoint_to_pytorch，执行转换
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.albert_config_file, args.pytorch_dump_path)
```