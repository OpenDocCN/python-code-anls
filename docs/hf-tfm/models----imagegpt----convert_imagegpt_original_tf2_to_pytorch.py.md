# `.\models\imagegpt\convert_imagegpt_original_tf2_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的详细信息
"""将 OpenAI Image GPT 检查点转换为 PyTorch 模型。"""

# 导入所需库
import argparse
import torch
from transformers import ImageGPTConfig, ImageGPTForCausalLM, load_tf_weights_in_imagegpt
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME, logging

# 设置日志级别为信息
logging.set_verbosity_info()

# 定义函数，将 ImageGPT 检查点转换为 PyTorch 模型
def convert_imagegpt_checkpoint_to_pytorch(imagegpt_checkpoint_path, model_size, pytorch_dump_folder_path):
    # 根据模型大小构建配置
    MODELS = {"small": (512, 8, 24), "medium": (1024, 8, 36), "large": (1536, 16, 48)}
    n_embd, n_head, n_layer = MODELS[model_size]  # 设置模型超参数
    config = ImageGPTConfig(n_embd=n_embd, n_layer=n_layer, n_head=n_head)
    model = ImageGPTForCausalLM(config)

    # 从 numpy 加载权重
    load_tf_weights_in_imagegpt(model, config, imagegpt_checkpoint_path)

    # 保存 PyTorch 模型
    pytorch_weights_dump_path = pytorch_dump_folder_path + "/" + WEIGHTS_NAME
    pytorch_config_dump_path = pytorch_dump_folder_path + "/" + CONFIG_NAME
    print(f"保存 PyTorch 模型至 {pytorch_weights_dump_path}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"保存配置文件至 {pytorch_config_dump_path}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(config.to_json_string())

# 主程序入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--imagegpt_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="TensorFlow 检查点路径。",
    )
    parser.add_argument(
        "--model_size",
        default=None,
        type=str,
        required=True,
        help="模型大小（可以是'small'、'medium'或'large'之一）。",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="输出 PyTorch 模型的路径。"
    )
    args = parser.parse_args()
    convert_imagegpt_checkpoint_to_pytorch(
        args.imagegpt_checkpoint_path, args.model_size, args.pytorch_dump_folder_path
    )
```