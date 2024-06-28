# `.\models\mobilebert\convert_mobilebert_original_tf_checkpoint_to_pytorch.py`

```py
# 导入必要的模块和库
import argparse  # 用于解析命令行参数

import torch  # 导入PyTorch库

# 从transformers库中导入MobileBertConfig、MobileBertForPreTraining和load_tf_weights_in_mobilebert函数
from transformers import MobileBertConfig, MobileBertForPreTraining, load_tf_weights_in_mobilebert

# 从transformers.utils中导入logging模块
from transformers.utils import logging

# 设置日志输出级别为info
logging.set_verbosity_info()

# 定义函数：将TensorFlow的checkpoint转换为PyTorch的模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, mobilebert_config_file, pytorch_dump_path):
    # 从配置文件中加载MobileBERT模型的配置
    config = MobileBertConfig.from_json_file(mobilebert_config_file)
    # 打印配置信息
    print(f"Building PyTorch model from configuration: {config}")
    # 根据配置创建MobileBERT的预训练模型
    model = MobileBertForPreTraining(config)
    # 加载TensorFlow的checkpoint中的权重到PyTorch模型中
    model = load_tf_weights_in_mobilebert(model, config, tf_checkpoint_path)
    # 打印保存PyTorch模型的路径
    print(f"Save PyTorch model to {pytorch_dump_path}")
    # 将PyTorch模型的状态字典保存到指定路径
    torch.save(model.state_dict(), pytorch_dump_path)


# 主程序入口
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加必选参数：TensorFlow的checkpoint路径
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    # 添加必选参数：MobileBERT模型配置文件的路径
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
    # 添加必选参数：输出的PyTorch模型路径
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，将TensorFlow的checkpoint转换为PyTorch模型
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.mobilebert_config_file, args.pytorch_dump_path)
```