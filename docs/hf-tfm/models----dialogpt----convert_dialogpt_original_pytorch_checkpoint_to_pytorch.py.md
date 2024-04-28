# `.\models\dialogpt\convert_dialogpt_original_pytorch_checkpoint_to_pytorch.py`

```
# 版权声明及许可信息
# 版权声明及许可信息
# 导入必要的库和模块
import argparse
import os
import torch
from transformers.utils import WEIGHTS_NAME

# 预定义对话生成模型的大小
DIALOGPT_MODELS = ["small", "medium", "large"]

# 定义需要转换的旧键和新键
OLD_KEY = "lm_head.decoder.weight"
NEW_KEY = "lm_head.weight"

# 转换对话生成模型的检查点文件
def convert_dialogpt_checkpoint(checkpoint_path: str, pytorch_dump_folder_path: str):
    # 加载原始检查点数据
    d = torch.load(checkpoint_path)
    # 将旧键对应的值移至新键
    d[NEW_KEY] = d.pop(OLD_KEY)
    # 创建目标文件夹路径（如果不存在）
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    # 保存新的检查点数据到目标文件夹路径下
    torch.save(d, os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME))

# 主程序入口
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogpt_path", default=".", type=str)
    args = parser.parse_args()
    # 遍历每个对话生成模型
    for MODEL in DIALOGPT_MODELS:
        # 组合检查点文件路径
        checkpoint_path = os.path.join(args.dialogpt_path, f"{MODEL}_ft.pkl")
        # 定义转换后的PyTorch模型文件夹路径
        pytorch_dump_folder_path = f"./DialoGPT-{MODEL}"
        # 进行检查点转换
        convert_dialogpt_checkpoint(
            checkpoint_path,
            pytorch_dump_folder_path,
        )
```