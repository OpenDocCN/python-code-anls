# `.\models\electra\convert_electra_original_tf_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# 版权所有 2018 年 HuggingFace 公司团队。
#
# 根据 Apache 许可证，版本 2.0 进行许可;
# 除非遵守许可证规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或以书面形式约定，否则依据"原样"提供软件，
# 没有任何明示或暗示的担保或条件。
# 请参考许可证以获取特定语言的权限和
# 下许可证规定的限制
"""Convert ELECTRA checkpoint."""

# 导入需要的库
import argparse
import torch
# 从 Transformers 库导入相应的类和函数
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from transformers.utils import logging

# 设置日志的输出级别为 info
logging.set_verbosity_info()

# 定义函数来将 TensorFlow 的 checkpoint 转换为 PyTorch 的模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    # 从配置文件中读取 Electra 模型的配置信息
    config = ElectraConfig.from_json_file(config_file)
    # 输出 PyTorch 模型正在根据配置文件构建的信息
    print(f"Building PyTorch model from configuration: {config}")

    # 根据所选择的是 discriminator 还是 generator 来初始化 PyTorch 模型
    if discriminator_or_generator == "discriminator":
        model = ElectraForPreTraining(config)
    elif discriminator_or_generator == "generator":
        model = ElectraForMaskedLM(config)
    else:
        raise ValueError("The discriminator_or_generator argument should be either 'discriminator' or 'generator'")

    # 从 TensorFlow 的 checkpoint 加载权重到 Electra 模型中
    load_tf_weights_in_electra(
        model, config, tf_checkpoint_path, discriminator_or_generator=discriminator_or_generator
    )

    # 保存 PyTorch 模型的状态字典
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)

# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
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
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--discriminator_or_generator",
        default=None,
        type=str,
        required=True,
        help=(
            "Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or "
            "'generator'."
        ),
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数进行 TensorFlow 到 PyTorch 的模型转换
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator
    )
```