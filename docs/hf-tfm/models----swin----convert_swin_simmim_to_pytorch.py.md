# `.\models\swin\convert_swin_simmim_to_pytorch.py`

```py
# 编码声明，指定使用 UTF-8 编码格式
# Copyright 2022 The HuggingFace Inc. team.
# 版权声明，版权归 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证版本 2.0 许可使用此文件；除非符合许可证的条款，否则不得使用此文件
# 您可以在以下网址获得许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件
# 无任何形式的明示或暗示担保或条件
# 请参阅许可证以了解特定的语言权限和限制

"""从原始存储库中转换 Swin SimMIM 检查点。

URL: https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#simmim-pretrained-swin-v1-models"""

# 导入必要的库和模块
import argparse  # 参数解析模块

import requests  # HTTP 请求库
import torch  # PyTorch 深度学习框架
from PIL import Image  # Python 图像处理库

from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor  # 导入 Transformers 库中的类


def get_swin_config(model_name):
    # 根据模型名称获取 Swin 模型配置
    config = SwinConfig(image_size=192)

    if "base" in model_name:
        # 如果模型名称包含“base”，设置特定参数
        window_size = 6
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    elif "large" in model_name:
        # 如果模型名称包含“large”，设置特定参数
        window_size = 12
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
    else:
        # 抛出错误，仅支持“base”和“large”变体的模型
        raise ValueError("Model not supported, only supports base and large variants")

    # 设置配置对象的参数
    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads

    return config


def rename_key(name):
    # 重命名模型的键名称，以便适应 Swin 模型的结构
    if "encoder.mask_token" in name:
        name = name.replace("encoder.mask_token", "embeddings.mask_token")
    if "encoder.patch_embed.proj" in name:
        name = name.replace("encoder.patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "encoder.patch_embed.norm" in name:
        name = name.replace("encoder.patch_embed.norm", "embeddings.norm")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    if name == "encoder.norm.weight":
        name = "layernorm.weight"
    if name == "encoder.norm.bias":
        name = "layernorm.bias"

    # 如果不包含“decoder”，则添加前缀“swin.”
    if "decoder" in name:
        pass
    else:
        name = "swin." + name

    return name


def convert_state_dict(orig_state_dict, model):
    # 定义函数，用于转换模型状态字典
    # 遍历原始状态字典中的键列表副本
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名包含 "attn_mask"，则跳过不处理
        if "attn_mask" in key:
            pass
        # 如果键名包含 "qkv"
        elif "qkv" in key:
            # 根据 "." 分割键名，提取层号和块号
            key_split = key.split(".")
            layer_num = int(key_split[2])
            block_num = int(key_split[4])
            # 获取当前注意力层的维度信息
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size

            # 如果键名包含 "weight"
            if "weight" in key:
                # 更新键名和对应的值到原始状态字典中，分别更新查询、键、值的权重部分
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新键名和对应的值到原始状态字典中，分别更新查询、键、值的偏置部分
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        else:
            # 使用自定义函数将键名转换后更新到原始状态字典中
            orig_state_dict[rename_key(key)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict
# 导入必要的库
import argparse
import requests
from PIL import Image
import torch
from transformers import SwinForMaskedImageModeling, ViTImageProcessor

# 定义函数，用于将指定的 Swin 模型检查点转换为 PyTorch 模型
def convert_swin_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    # 加载指定路径的检查点，并从中提取模型的状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 获取指定模型名称的配置
    config = get_swin_config(model_name)
    # 根据配置创建 Swin 模型对象
    model = SwinForMaskedImageModeling(config)
    # 设置模型为评估模式
    model.eval()

    # 转换模型的状态字典格式
    new_state_dict = convert_state_dict(state_dict, model)
    # 加载转换后的状态字典到模型中
    model.load_state_dict(new_state_dict)

    # 需要处理的图片的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 创建图像处理器对象，指定输出图像的大小
    image_processor = ViTImageProcessor(size={"height": 192, "width": 192})
    # 使用 requests 库获取并打开指定 URL 的图像，并用 PIL 库打开
    image = Image.open(requests.get(url, stream=True).raw)
    # 使用图像处理器处理图像，将图像转换为 PyTorch 张量格式
    inputs = image_processor(images=image, return_tensors="pt")

    # 关闭 PyTorch 自动求导功能，因为只需要进行推断
    with torch.no_grad():
        # 使用模型进行推断，获取输出 logits
        outputs = model(**inputs).logits

    # 打印模型输出的键
    print(outputs.keys())
    # 输出消息，确认一切正常
    print("Looks ok!")

    # 如果指定了输出目录路径
    if pytorch_dump_folder_path is not None:
        # 打印消息，保存模型到指定目录
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定的目录
        model.save_pretrained(pytorch_dump_folder_path)

        # 打印消息，保存图像处理器到指定目录
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        # 将图像处理器保存到指定的目录
        image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 hub
    if push_to_hub:
        # 打印消息，将模型和图像处理器推送到 hub
        print(f"Pushing model and image processor for {model_name} to hub")
        # 推送模型到 hub
        model.push_to_hub(f"microsoft/{model_name}")
        # 推送图像处理器到 hub
        image_processor.push_to_hub(f"microsoft/{model_name}")


# 如果该脚本作为主程序运行
if __name__ == "__main__":
    # 创建命令行参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加必需参数：模型名称
    parser.add_argument(
        "--model_name",
        default="swin-base-simmim-window6-192",
        type=str,
        choices=["swin-base-simmim-window6-192", "swin-large-simmim-window12-192"],
        help="Name of the Swin SimMIM model you'd like to convert.",
    )
    
    # 添加必需参数：原始检查点文件路径
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/SwinSimMIM/simmim_pretrain__swin_base__img192_window6__100ep.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
    )
    
    # 添加可选参数：输出 PyTorch 模型目录路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    
    # 添加可选标志：是否将模型推送到 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用转换函数，传入命令行参数
    convert_swin_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
```