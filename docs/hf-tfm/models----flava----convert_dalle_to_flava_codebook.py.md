# `.\models\flava\convert_dalle_to_flava_codebook.py`

```py
# 导入必要的模块和库
import argparse  # 导入用于解析命令行参数的模块
import os  # 导入操作系统相关功能的模块

import torch  # 导入PyTorch深度学习库

from transformers import FlavaImageCodebook, FlavaImageCodebookConfig  # 导入transformers库中的模型和配置


def rreplace(s, old, new, occurrence):
    # 从字符串末尾向前查找并替换指定次数的子字符串
    li = s.rsplit(old, occurrence)
    return new.join(li)


def count_parameters(state_dict):
    # 统计模型参数数量
    # 对于不属于"encoder.embeddings"的参数，计算其总和
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


def upgrade_state_dict(state_dict):
    # 更新模型状态字典中的键名，以符合transformers的设计规范
    upgrade = {}

    group_keys = ["group_1", "group_2", "group_3", "group_4"]
    for key, value in state_dict.items():
        for group_key in group_keys:
            if group_key in key:
                key = key.replace(f"{group_key}.", f"{group_key}.group.")

        if "res_path" in key:
            key = key.replace("res_path.", "res_path.path.")

        if key.endswith(".w"):
            key = rreplace(key, ".w", ".weight", 1)
        if key.endswith(".b"):
            key = rreplace(key, ".b", ".bias", 1)

        upgrade[key] = value.float()

    return upgrade


@torch.no_grad()
def convert_dalle_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_path=None, save_checkpoint=True):
    """
    复制/粘贴/调整模型的权重以适应transformers设计。
    """
    from dall_e import Encoder  # 导入dall-e项目中的Encoder模型

    encoder = Encoder()  # 实例化Encoder模型对象
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)  # 如果本地存在checkpoint文件，则加载
    else:
        ckpt = torch.hub.load_state_dict_from_url(checkpoint_path)  # 否则，从URL加载模型权重

    if isinstance(ckpt, Encoder):
        ckpt = ckpt.state_dict()  # 如果加载的是Encoder对象，则获取其状态字典
    encoder.load_state_dict(ckpt)  # 加载Encoder模型的权重

    if config_path is not None:
        config = FlavaImageCodebookConfig.from_pretrained(config_path)  # 如果提供了配置文件路径，则从中加载配置
    else:
        config = FlavaImageCodebookConfig()  # 否则使用默认配置

    hf_model = FlavaImageCodebook(config).eval()  # 根据配置实例化FlavaImageCodebook模型，并设置为评估模式
    state_dict = encoder.state_dict()  # 获取Encoder模型的状态字典

    hf_state_dict = upgrade_state_dict(state_dict)  # 将Encoder模型的状态字典转换为适应transformers的格式
    hf_model.load_state_dict(hf_state_dict)  # 加载适应transformers格式的状态字典到FlavaImageCodebook模型
    hf_state_dict = hf_model.state_dict()  # 获取转换后的模型状态字典
    hf_count = count_parameters(hf_state_dict)  # 统计转换后模型的参数数量
    state_dict_count = count_parameters(state_dict)  # 统计原始Encoder模型的参数数量

    assert torch.allclose(hf_count, state_dict_count, atol=1e-3)  # 断言转换后的模型参数数量与原始模型参数数量的接近性

    if save_checkpoint:
        hf_model.save_pretrained(pytorch_dump_folder_path)  # 如果指定保存checkpoint，则保存模型到指定路径
    else:
        return hf_state_dict  # 否则返回转换后的模型状态字典


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器对象
    # 解析命令行参数，获取用户输入的 PyTorch 模型输出路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 解析命令行参数，获取用户输入的 flava 检查点路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    # 解析命令行参数，获取用户输入的模型配置文件路径（通常是一个 JSON 文件）
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数，将所有参数解析并存储在 args 对象中
    args = parser.parse_args()
    
    # 调用函数 convert_dalle_checkpoint，传递解析后的参数：
    # args.checkpoint_path：flava 检查点路径
    # args.pytorch_dump_folder_path：PyTorch 模型输出路径
    # args.config_path：模型配置文件路径
    convert_dalle_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```