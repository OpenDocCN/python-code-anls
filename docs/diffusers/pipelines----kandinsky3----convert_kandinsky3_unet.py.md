# `.\diffusers\pipelines\kandinsky3\convert_kandinsky3_unet.py`

```py
#!/usr/bin/env python3
# 指定脚本使用 Python 3 运行环境
import argparse
# 导入用于解析命令行参数的 argparse 模块
import fnmatch
# 导入用于匹配文件名的 fnmatch 模块

from safetensors.torch import load_file
# 从 safetensors.torch 导入 load_file 函数，用于加载模型文件

from diffusers import Kandinsky3UNet
# 从 diffusers 库导入 Kandinsky3UNet 模型


# 定义一个字典，用于映射原 U-Net 模型中的键到 Kandinsky3UNet 模型的键
MAPPING = {
    "to_time_embed.1": "time_embedding.linear_1",
    "to_time_embed.3": "time_embedding.linear_2",
    "in_layer": "conv_in",
    "out_layer.0": "conv_norm_out",
    "out_layer.2": "conv_out",
    "down_samples": "down_blocks",
    "up_samples": "up_blocks",
    "projection_lin": "encoder_hid_proj.projection_linear",
    "projection_ln": "encoder_hid_proj.projection_norm",
    "feature_pooling": "add_time_condition",
    "to_query": "to_q",
    "to_key": "to_k",
    "to_value": "to_v",
    "output_layer": "to_out.0",
    "self_attention_block": "attentions.0",
}

# 定义一个动态映射字典，用于处理带有通配符的键
DYNAMIC_MAP = {
    "resnet_attn_blocks.*.0": "resnets_in.*",
    "resnet_attn_blocks.*.1": ("attentions.*", 1),
    "resnet_attn_blocks.*.2": "resnets_out.*",
}
# DYNAMIC_MAP = {}  # 备用的动态映射字典


def convert_state_dict(unet_state_dict):
    """
    Args:
    Convert the state dict of a U-Net model to match the key format expected by Kandinsky3UNet model.
        unet_model (torch.nn.Module): The original U-Net model. unet_kandi3_model (torch.nn.Module): The Kandinsky3UNet
        model to match keys with.

    Returns:
        OrderedDict: The converted state dictionary.
    """
    # 创建一个空字典，用于存储转换后的状态字典
    converted_state_dict = {}
    # 遍历原 U-Net 模型的状态字典中的每个键
    for key in unet_state_dict:
        new_key = key  # 初始化新键为原键
        # 遍历映射字典，将原键中的模式替换为新模式
        for pattern, new_pattern in MAPPING.items():
            new_key = new_key.replace(pattern, new_pattern)

        # 处理动态映射
        for dyn_pattern, dyn_new_pattern in DYNAMIC_MAP.items():
            has_matched = False  # 初始化匹配标志
            # 如果新键匹配动态模式且尚未匹配
            if fnmatch.fnmatch(new_key, f"*.{dyn_pattern}.*") and not has_matched:
                # 提取模式中的数字部分
                star = int(new_key.split(dyn_pattern.split(".")[0])[-1].split(".")[1])

                # 处理动态模式为元组的情况
                if isinstance(dyn_new_pattern, tuple):
                    new_star = star + dyn_new_pattern[-1]  # 计算新数字
                    dyn_new_pattern = dyn_new_pattern[0]  # 提取新的模式
                else:
                    new_star = star  # 保持数字不变

                # 替换动态模式中的星号
                pattern = dyn_pattern.replace("*", str(star))
                new_pattern = dyn_new_pattern.replace("*", str(new_star))

                # 更新新键
                new_key = new_key.replace(pattern, new_pattern)
                has_matched = True  # 设置匹配标志为 True

        # 将转换后的新键与原状态字典中的值存入新的字典
        converted_state_dict[new_key] = unet_state_dict[key]

    return converted_state_dict  # 返回转换后的状态字典


def main(model_path, output_path):
    # 加载原 U-Net 模型的状态字典
    unet_state_dict = load_file(model_path)

    # 初始化 Kandinsky3UNet 模型的配置
    config = {}

    # 调用转换函数，转换状态字典
    converted_state_dict = convert_state_dict(unet_state_dict)

    # 实例化 Kandinsky3UNet 模型
    unet = Kandinsky3UNet(config)
    # 加载转换后的状态字典到模型中
    unet.load_state_dict(converted_state_dict)

    # 保存转换后的模型到指定路径
    unet.save_pretrained(output_path)
    # 打印保存模型的路径
    print(f"Converted model saved to {output_path}")


if __name__ == "__main__":
    # 创建命令行解析器，提供描述信息
    parser = argparse.ArgumentParser(description="Convert U-Net PyTorch model to Kandinsky3UNet format")
    # 添加命令行参数 '--model_path'，指定原始 U-Net PyTorch 模型的路径，类型为字符串，必填
        parser.add_argument("--model_path", type=str, required=True, help="Path to the original U-Net PyTorch model")
        # 添加命令行参数 '--output_path'，指定转换后模型的保存路径，类型为字符串，必填
        parser.add_argument("--output_path", type=str, required=True, help="Path to save the converted model")
    
        # 解析命令行参数并将其存储在 args 对象中
        args = parser.parse_args()
        # 调用 main 函数，传入模型路径和输出路径参数
        main(args.model_path, args.output_path)
```