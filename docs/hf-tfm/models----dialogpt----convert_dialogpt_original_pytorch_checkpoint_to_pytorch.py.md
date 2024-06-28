# `.\models\dialogpt\convert_dialogpt_original_pytorch_checkpoint_to_pytorch.py`

```py
# 导入必要的模块和库
import argparse  # 导入用于处理命令行参数的模块
import os  # 导入用于处理操作系统功能的模块

import torch  # 导入PyTorch深度学习库

from transformers.utils import WEIGHTS_NAME  # 从transformers库中导入WEIGHTS_NAME常量

# 预定义的DialoGPT模型名称列表
DIALOGPT_MODELS = ["small", "medium", "large"]

# 旧的模型权重键名和新的模型权重键名
OLD_KEY = "lm_head.decoder.weight"
NEW_KEY = "lm_head.weight"


def convert_dialogpt_checkpoint(checkpoint_path: str, pytorch_dump_folder_path: str):
    # 加载PyTorch模型检查点文件
    d = torch.load(checkpoint_path)
    # 将旧的权重键名映射到新的权重键名
    d[NEW_KEY] = d.pop(OLD_KEY)
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    # 保存修改后的模型检查点到目标文件夹中
    torch.save(d, os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME))


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加命令行参数--dialogpt_path，默认为当前目录
    parser.add_argument("--dialogpt_path", default=".", type=str)
    # 解析命令行参数
    args = parser.parse_args()
    
    # 遍历预定义的DialoGPT模型列表
    for MODEL in DIALOGPT_MODELS:
        # 构建模型检查点文件路径
        checkpoint_path = os.path.join(args.dialogpt_path, f"{MODEL}_ft.pkl")
        # 构建PyTorch模型转换后的输出文件夹路径
        pytorch_dump_folder_path = f"./DialoGPT-{MODEL}"
        # 转换模型检查点格式并保存到目标文件夹中
        convert_dialogpt_checkpoint(
            checkpoint_path,
            pytorch_dump_folder_path,
        )
```