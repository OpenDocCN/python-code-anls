# `.\transformers\models\mobilebert\convert_mobilebert_original_tf_checkpoint_to_pytorch.py`

```
import argparse  # 导入 argparse 模块，用于解析命令行参数

import torch  # 导入 PyTorch 库

from transformers import MobileBertConfig, MobileBertForPreTraining, load_tf_weights_in_mobilebert  # 导入 MobileBERT 相关类和函数
from transformers.utils import logging  # 导入 logging 模块

logging.set_verbosity_info()  # 设置 logging 模块的日志级别为 info


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, mobilebert_config_file, pytorch_dump_path):
    # 初始化 PyTorch 模型配置
    config = MobileBertConfig.from_json_file(mobilebert_config_file)
    # 输出 PyTorch 模型配置信息
    print(f"Building PyTorch model from configuration: {config}")
    # 根据配置创建 MobileBERT 预训练模型
    model = MobileBertForPreTraining(config)
    # 从 TensorFlow checkpoint 加载权重到 PyTorch 模型
    model = load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path)
    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    # 添加命令行参数
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--mobilebert_config_file",
        default=None,
        type=str,
        required=True,
        help=(
            "The config json file corresponding to the pre-trained MobileBERT model. \n"
            "This specifies the model architecture."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()  # 解析命令行参数
    # 调用函数将 TensorFlow checkpoint 转换为 PyTorch 模型
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.mobilebert_config_file, args.pytorch_dump_path)
```  
```