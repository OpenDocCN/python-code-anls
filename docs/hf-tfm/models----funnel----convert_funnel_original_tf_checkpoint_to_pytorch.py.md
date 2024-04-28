# `.\models\funnel\convert_funnel_original_tf_checkpoint_to_pytorch.py`

```
# 指定文件编码格式
# 版权声明
# 根据 Apache 许可证，除非遵守许可，否则不能使用此文件
# 获取许可的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则基于许可协议的软件按"原样"分发，没有任何类型的担保或条件，不管是明示的还是隐含的。
# 请查看许可协议以获取特定语言的权限和限制

# 导入需要的模块
import argparse
import torch
from transformers import FunnelBaseModel, FunnelConfig, FunnelModel, load_tf_weights_in_funnel
from transformers.utils import logging

# 设置日志级别
logging.set_verbosity_info()

# 定义函数，用于将 TensorFlow 的 checkpoint 转换为 PyTorch 格式
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, base_model):
    # 从配置文件中初始化 PyTorch 模型
    config = FunnelConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # 根据是否仅使用基础模型来选择初始化的模型类型
    model = FunnelBaseModel(config) if base_model else FunnelModel(config)

    # 从 TensorFlow checkpoint 中加载权重
    load_tf_weights_in_funnel(model, config, tf_checkpoint_path)

    # 保存 PyTorch 模型
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)

# 主程序入口
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
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--base_model", action="store_true", help="Whether you want just the base model (no decoder) or not."
    )
    args = parser.parse_args()
    # 调用转换函数
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.base_model
    )
```