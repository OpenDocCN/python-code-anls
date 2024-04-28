# `.\models\flava\convert_dalle_to_flava_codebook.py`

```
# 这是 Python 文件的编码声明和版权信息
# 版权 2022 Meta Platforms 和 HuggingFace 团队，保留所有权利。
# 
# 根据 Apache License, Version 2.0 进行授权
# 如果没有法律要求，按照 "AS IS" 条款进行分发
# 查看许可证了解更多细节和限制
# 
# 导入 Python 的 argparse 和 os 模块，用于命令行解析和操作系统交互
import argparse
import os

# 导入 PyTorch，用于深度学习操作
import torch

# 从 transformers 导入 FlavaImageCodebook 和 FlavaImageCodebookConfig， 用于处理 FLAVA 模型
from transformers import FlavaImageCodebook, FlavaImageCodebookConfig


# 定义 rreplace 函数，用于从字符串 s 中将最后 occurrence 次 old 替换为 new
def rreplace(s, old, new, occurrence):
    # 用 old 分割字符串，限制分割次数为 occurrence，然后用 new 重新连接
    li = s.rsplit(old, occurrence)
    return new.join(li)


# 定义 count_parameters 函数，用于统计 state_dict 中参数的总和
def count_parameters(state_dict):
    # 由于原始 FLAVA 模型中 encoder.embeddings 会被双倍复制，所以忽略这些
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


# 定义 upgrade_state_dict 函数，用于升级给定的 state_dict
def upgrade_state_dict(state_dict):
    # 创建空字典，保存升级后的参数
    upgrade = {}

    # 定义需要升级的键的组
    group_keys = ["group_1", "group_2", "group_3", "group_4"]
    # 遍历 state_dict 中的每个键值对
    for key, value in state_dict.items():
        # 遍历 group_keys，看是否在键中
        for group_key in group_keys:
            if group_key in key:
                # 如果找到，则在键中替换 group_key 为 group_key.group
                key = key.replace(f"{group_key}.", f"{group_key}.group.")

        # 如果键中包含 "res_path"，替换为 "res_path.path."
        if "res_path" in key:
            key = key.replace("res_path.", "res_path.path.")

        # 如果键以 ".w" 结尾，替换为 ".weight"
        if key.endswith(".w"):
            key = rreplace(key, ".w", ".weight", 1)
        # 如果键以 ".b" 结尾，替换为 ".bias"
        if key.endswith(".b"):
            key = rreplace(key, ".b", ".bias", 1)

        # 将升级后的键值对添加到 upgrade 字典中，并将值转换为浮点数
        upgrade[key] = value.float()

    # 返回升级后的字典
    return upgrade


# 使用 torch.no_grad 装饰器，定义 convert_dalle_checkpoint 函数，以禁用梯度计算
@torch.no_grad()
def convert_dalle_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, save_checkpoint=True):
    """
    复制/粘贴/调整模型权重到 transformers 的设计。
    """
    # 从 dall_e 模块导入 Encoder 类
    from dall_e import Encoder

    # 创建一个 Encoder 对象
    encoder = Encoder()
    # 如果 checkpoint_path 存在，则从路径加载检查点
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
    else:
        # 否则从 URL 加载状态字典
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path)

    # 如果加载的 ckpt 是一个 Encoder 对象，则提取其 state_dict
    if isinstance(ckpt, Encoder):
        ckpt = ckpt.state_dict()
    # 使用检查点的状态字典来加载 encoder
    encoder.load_state_dict(ckpt)

    # 如果提供了 config_path，则从配置中创建 FlavaImageCodebookConfig 对象
    if config_path is not None:
        config = FlavaImageCodebookConfig.from_pretrained(config_path)
    else:
        # 否则创建默认配置
        config = FlavaImageCodebookConfig()

    # 创建 FlavaImageCodebook 对象，并设置为评估模式
    hf_model = FlavaImageCodebook(config).eval()
    # 获取 encoder 的状态字典
    state_dict = encoder.state_dict()

    # 使用 upgrade_state_dict 函数升级状态字典
    hf_state_dict = upgrade_state_dict(state_dict)
    # 使用升级后的状态字典加载模型
    hf_model.load_state_dict(hf_state_dict)
    # 获取模型的状态字典
    hf_state_dict = hf_model.state_dict()
    # 统计模型的参数数量
    hf_count = count_parameters(hf_state_dict)
    # 统计原始状态字典的参数数量
    state_dict_count = count_parameters(state_dict)

    # 断言两个参数数量非常接近
    assert torch.allclose(hf_count, state_dict_count, atol=1e-3)

    # 如果 save_checkpoint 为真，则将模型保存到 pytorch_dump_folder_path
    if save_checkpoint:
        hf_model.save_pretrained(pytorch_dump_folder_path)
    else:
        # 否则返回状态字典
        return hf_state_dict


# 如果当前模块是主程序，设置命令行参数解析
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 添加命令行参数解析器的一个参数，用于指定输出 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数解析器的一个参数，用于指定 flava 检查点的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    # 添加命令行参数解析器的一个参数，用于指定待转换模型的 HF 配置文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数，将其存储到 args 变量中
    args = parser.parse_args()
    
    # 调用函数，将 flava 检查点转换为 PyTorch 模型
    convert_dalle_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```