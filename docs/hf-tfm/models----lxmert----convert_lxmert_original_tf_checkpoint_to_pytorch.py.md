# `.\transformers\models\lxmert\convert_lxmert_original_tf_checkpoint_to_pytorch.py`

```py
# 导入必要的库
import argparse  # 解析命令行参数
import torch  # 导入 PyTorch 库

# 从 transformers 库中导入所需的类和函数
from transformers import LxmertConfig, LxmertForPreTraining, load_tf_weights_in_lxmert
from transformers.utils import logging  # 导入日志记录工具

# 设置日志记录级别为信息
logging.set_verbosity_info()


# 定义函数，用于将 TensorFlow checkpoint 转换为 PyTorch 模型
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    # 从 JSON 文件中读取 LXMERT 模型的配置信息
    config = LxmertConfig.from_json_file(config_file)
    # 打印模型配置信息
    print(f"Building PyTorch model from configuration: {config}")
    # 基于配置信息初始化 PyTorch 模型
    model = LxmertForPreTraining(config)

    # 加载 TensorFlow checkpoint 中的权重到 PyTorch 模型中
    load_tf_weights_in_lxmert(model, config, tf_checkpoint_path)

    # 将 PyTorch 模型保存到指定路径
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
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，执行 TensorFlow checkpoint 到 PyTorch 模型的转换
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)
```