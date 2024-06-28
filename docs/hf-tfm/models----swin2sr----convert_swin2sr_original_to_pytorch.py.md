# `.\models\swin2sr\convert_swin2sr_original_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert Swin2SR checkpoints from the original repository. URL: https://github.com/mv-lab/swin2sr"""

import argparse  # 导入解析命令行参数的模块

import requests  # 导入发送 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
from PIL import Image  # 导入处理图像的模块
from torchvision.transforms import Compose, Normalize, Resize, ToTensor  # 导入图像转换相关模块

from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor  # 导入转换器相关模块


def get_config(checkpoint_url):
    config = Swin2SRConfig()  # 创建 Swin2SRConfig 的实例

    if "Swin2SR_ClassicalSR_X4_64" in checkpoint_url:
        config.upscale = 4  # 设置放大倍数为4
    elif "Swin2SR_CompressedSR_X4_48" in checkpoint_url:
        config.upscale = 4  # 设置放大倍数为4
        config.image_size = 48  # 设置图像尺寸为48
        config.upsampler = "pixelshuffle_aux"  # 设置上采样方法为 pixelshuffle_aux
    elif "Swin2SR_Lightweight_X2_64" in checkpoint_url:
        config.depths = [6, 6, 6, 6]  # 设置深度参数列表
        config.embed_dim = 60  # 设置嵌入维度为60
        config.num_heads = [6, 6, 6, 6]  # 设置注意力头数列表
        config.upsampler = "pixelshuffledirect"  # 设置上采样方法为 pixelshuffledirect
    elif "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR" in checkpoint_url:
        config.upscale = 4  # 设置放大倍数为4
        config.upsampler = "nearest+conv"  # 设置上采样方法为 nearest+conv
    elif "Swin2SR_Jpeg_dynamic" in checkpoint_url:
        config.num_channels = 1  # 设置通道数为1
        config.upscale = 1  # 设置放大倍数为1
        config.image_size = 126  # 设置图像尺寸为126
        config.window_size = 7  # 设置窗口大小为7
        config.img_range = 255.0  # 设置图像像素范围为255.0
        config.upsampler = ""  # 设置上采样方法为空字符串

    return config  # 返回配置对象


def rename_key(name, config):
    if "patch_embed.proj" in name and "layers" not in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.patch_embeddings.layernorm")
    if "layers" in name:
        name = name.replace("layers", "encoder.stages")
    if "residual_group.blocks" in name:
        name = name.replace("residual_group.blocks", "layers")
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
    if "q_bias" in name:
        name = name.replace("q_bias", "query.bias")
    if "k_bias" in name:
        name = name.replace("k_bias", "key.bias")
    # 如果变量名中包含 "v_bias"，则替换为 "value.bias"
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    
    # 如果变量名中包含 "cpb_mlp"，则替换为 "continuous_position_bias_mlp"
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    
    # 如果变量名中包含 "patch_embed.proj"，则替换为 "patch_embed.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "patch_embed.projection")
    
    # 如果变量名为 "norm.weight"，则替换为 "layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    
    # 如果变量名为 "norm.bias"，则替换为 "layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"
    
    # 如果变量名中包含 "conv_first"，则替换为 "first_convolution"
    if "conv_first" in name:
        name = name.replace("conv_first", "first_convolution")
    
    # 如果变量名中包含以下任意一个字符串，将其替换为相应的名称或前缀
    if (
        "upsample" in name
        or "conv_before_upsample" in name
        or "conv_bicubic" in name
        or "conv_up" in name
        or "conv_hr" in name
        or "conv_last" in name
        or "aux" in name
    ):
        # 对于特定的字符串替换规则
        if "conv_last" in name:
            name = name.replace("conv_last", "final_convolution")
    
        # 根据 config.upsampler 的不同取值进行不同的替换
        if config.upsampler in ["pixelshuffle", "pixelshuffle_aux", "nearest+conv"]:
            if "conv_before_upsample.0" in name:
                name = name.replace("conv_before_upsample.0", "conv_before_upsample")
            if "upsample.0" in name:
                name = name.replace("upsample.0", "upsample.convolution_0")
            if "upsample.2" in name:
                name = name.replace("upsample.2", "upsample.convolution_1")
            # 统一添加前缀 "upsample."
            name = "upsample." + name
        elif config.upsampler == "pixelshuffledirect":
            # 特定替换规则
            name = name.replace("upsample.0.weight", "upsample.conv.weight")
            name = name.replace("upsample.0.bias", "upsample.conv.bias")
        else:
            pass
    else:
        # 如果不符合以上任何替换条件，则添加前缀 "swin2sr."
        name = "swin2sr." + name
    
    # 返回处理后的变量名
    return name
# 转换给定的原始状态字典，根据配置更新键名
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的复制键列表
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键名包含"qkv"
        if "qkv" in key:
            # 拆分键名以获取阶段号、块号和维度
            key_split = key.split(".")
            stage_num = int(key_split[1])
            block_num = int(key_split[4])
            dim = config.embed_dim

            # 如果键名中包含"weight"
            if "weight" in key:
                # 更新查询权重、键权重和值权重的新键名和对应的值
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.weight"
                ] = val[dim : dim * 2, :]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]
            else:
                # 更新查询偏置、键偏置和值偏置的新键名和对应的值
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
            pass
        else:
            # 对于其他键，使用配置中的重命名函数处理键名，并更新原始状态字典
            orig_state_dict[rename_key(key, config)] = val

    # 返回更新后的原始状态字典
    return orig_state_dict


def convert_swin2sr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    # 获取模型配置
    config = get_config(checkpoint_url)
    # 根据配置创建模型实例
    model = Swin2SRForImageSuperResolution(config)
    # 将模型设置为评估模式
    model.eval()

    # 从给定的 URL 加载模型状态字典到本地
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 使用转换函数将状态字典转换为适用于当前模型的新状态字典
    new_state_dict = convert_state_dict(state_dict, config)
    # 加载新的状态字典到模型中，并获取缺失键和意外键
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # 如果存在缺失键，抛出值错误
    if len(missing_keys) > 0:
        raise ValueError("Missing keys when converting: {}".format(missing_keys))
    # 对于每个意外的键，如果不包含指定的子字符串，则抛出值错误
    for key in unexpected_keys:
        if not ("relative_position_index" in key or "relative_coords_table" in key or "self_mask" in key):
            raise ValueError(f"Unexpected key {key} in state_dict")

    # 验证加载的图像 URL
    url = "https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true"
    # 使用请求获取并打开图像，并将其转换为 RGB 模式
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # 创建图像处理器实例
    processor = Swin2SRImageProcessor()
    
    # 根据模型类型设置图像大小
    image_size = 126 if "Jpeg" in checkpoint_url else 256
    # 定义图像转换步骤，包括调整大小、转换为张量和归一化处理
    transforms = Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # 对图像应用转换步骤，并扩展维度以匹配模型输入
    pixel_values = transforms(image).unsqueeze(0)

    # 如果配置中指定的通道数为 1，只保留第一个通道的像素值
    if config.num_channels == 1:
        pixel_values = pixel_values[:, 0, :, :].unsqueeze(1)
    # 使用模型对输入像素值进行推理，得到输出结果
    outputs = model(pixel_values)

    # 根据不同的 checkpoint_url 设置预期的输出形状和切片
    if "Swin2SR_ClassicalSR_X2_64" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor(
            [[-0.7087, -0.7138, -0.6721], [-0.8340, -0.8095, -0.7298], [-0.9149, -0.8414, -0.7940]]
        )
    elif "Swin2SR_ClassicalSR_X4_64" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor(
            [[-0.7775, -0.8105, -0.8933], [-0.7764, -0.8356, -0.9225], [-0.7976, -0.8686, -0.9579]]
        )
    elif "Swin2SR_CompressedSR_X4_48" in checkpoint_url:
        # TODO 值在这里并不完全匹配
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor(
            [[-0.8035, -0.7504, -0.7491], [-0.8538, -0.8124, -0.7782], [-0.8804, -0.8651, -0.8493]]
        )
    elif "Swin2SR_Lightweight_X2_64" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 512, 512])
        expected_slice = torch.tensor(
            [[-0.7669, -0.8662, -0.8767], [-0.8810, -0.9962, -0.9820], [-0.9340, -1.0322, -1.1149]]
        )
    elif "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR" in checkpoint_url:
        expected_shape = torch.Size([1, 3, 1024, 1024])
        expected_slice = torch.tensor(
            [[-0.5238, -0.5557, -0.6321], [-0.6016, -0.5903, -0.6391], [-0.6244, -0.6334, -0.6889]]
        )

    # 断言输出重建的形状是否与预期一致
    assert (
        outputs.reconstruction.shape == expected_shape
    ), f"Shape of reconstruction should be {expected_shape}, but is {outputs.reconstruction.shape}"

    # 断言输出重建的部分数据是否与预期一致，容差为 1e-3
    assert torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-3)
    
    # 打印提示信息，表明检查通过
    print("Looks ok!")

    # 将 checkpoint_url 映射到模型名称的字典
    url_to_name = {
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth": (
            "swin2SR-classical-sr-x2-64"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth": (
            "swin2SR-classical-sr-x4-64"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth": (
            "swin2SR-compressed-sr-x4-48"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_Lightweight_X2_64.pth": (
            "swin2SR-lightweight-x2-64"
        ),
        "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth": (
            "swin2SR-realworld-sr-x4-64-bsrgan-psnr"
        ),
    }
    
    # 根据 checkpoint_url 获取模型名称
    model_name = url_to_name[checkpoint_url]

    # 如果指定了 pytorch_dump_folder_path，保存模型和处理器到该路径
    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果设置了 push_to_hub 标志，将模型和处理器推送到 Hub
    if push_to_hub:
        model.push_to_hub(f"caidas/{model_name}")
        processor.push_to_hub(f"caidas/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    # 必需的参数设定
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth",
        type=str,
        help="URL of the original Swin2SR checkpoint you'd like to convert.",
    )
    # 添加名为 "checkpoint_url" 的参数，设定默认值为 Swin2SR 模型的下载地址，类型为字符串，帮助信息指定用途

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加名为 "pytorch_dump_folder_path" 的参数，设定默认值为 None，类型为字符串，帮助信息指定输出 PyTorch 模型的目录路径

    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")
    # 添加名为 "push_to_hub" 的参数，设定为布尔类型，表示是否将转换后的模型推送到模型中心（hub）

    args = parser.parse_args()
    # 解析命令行参数，并将结果存储在 args 对象中

    convert_swin2sr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 convert_swin2sr_checkpoint，传递解析后的参数对象 args 的相应属性作为函数参数
```