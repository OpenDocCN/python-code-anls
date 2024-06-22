# `.\models\gpt2\convert_gpt2_original_tf_checkpoint_to_pytorch.py`

```py
# 定义编码格式
# 版权声明
#
# 根据 Apache 许可证 2.0 版本
#
# 除了遵守许可证，否则不得使用此文件。
#
# 您可以从以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于"AS IS"的基础上分发的
# 没有任何保证或条件，无论是明示的还是暗示的
# 请查看具体执行操作和许可证的限制
"""Convert OpenAI GPT checkpoint."""

# 导入必要的库
import argparse
import torch
# 从 transformers 库中导入 GPT2Config, GPT2Model 和 load_tf_weights_in_gpt2
from transformers import GPT2Config, GPT2Model, load_tf_weights_in_gpt2
# 从 transformers 库中导入 logging 模块
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging

# 设置日志级别为 info
logging.set_verbosity_info()

# 定义函数：将 GPT2 的 TensorFlow checkpoint 转换为 PyTorch 模型
def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
    # 构建模型
    if gpt2_config_file == "":
        config = GPT2Config()
    else:
        config = GPT2Config.from_json_file(gpt2_config_file)
    model = GPT2Model(config)

    # 从 numpy 加载权重
    load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path)

    # 保存 PyTorch 模型
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    print(f"保存 PyTorch 模型至 {pytorch_weights_dump_path}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"保存配置文件至 {pytorch_config_dump_path}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())

# 程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必须参数
    parser.add_argument(
        "--gpt2_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--gpt2_config_file",
        default="",
        type=str,
        help=(
            "An optional config json file corresponding to the pre-trained OpenAI model. \n"
            "This specifies the model architecture."
        ),
    )
    args = parser.parse_args()
    convert_gpt2_checkpoint_to_pytorch(args.gpt2_checkpoint_path, args.gpt2_config_file, args.pytorch_dump_folder_path)
```