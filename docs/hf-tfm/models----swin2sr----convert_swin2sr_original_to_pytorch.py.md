# `.\transformers\models\swin2sr\convert_swin2sr_original_to_pytorch.py`

```
# 设置文件编码为 utf-8
# 版权声明，该代码文件使用 Apache License, Version 2.0 授权
# 获取授权副本的链接
# 如果软件按照许可证的规定分发，它将基于 "AS IS" 的基础分发，不提供任何明示或默示的担保或条件
# 根据许可证规则，具体语言支持语言规定和限制
# Swin2SR 模型的配置和转换逻辑，包括检查点的位置和配置参数
import argparse  # 用于解析命令行参数的库
import requests  # 用于发送 HTTP 请求的库
import torch  # PyTorch 深度学习框架
from PIL import Image  # 用于图像处理的 Python 库
from torchvision.transforms import Compose, Normalize, Resize, ToTensor  # 用于图像处理的 PyTorch 库
from transformers import Swin2SRConfig, Swin2SRForImageSuperResolution, Swin2SRImageProcessor  # Hugging Face 提供的 Swin2SR 相关类

# 根据检查点 URL 获取 Swin2SR 模型的配置
def get_config(checkpoint_url):
    config = Swin2SRConfig()  # 从 transformers 库中创建 Swin2SR 模型的配置对象

    # “Swin2SR_ClassicalSR_X4_64”检查点配置
    if "Swin2SR_ClassicalSR_X4_64" in checkpoint_url:
        config.upscale = 4  # 图像放大因子为 4
    # “Swin2SR_CompressedSR_X4_48”检查点配置
    elif "Swin2SR_CompressedSR_X4_48" in checkpoint_url:
        config.upscale = 4  # 图像放大因子为 4
        config.image_size = 48  # 图像尺寸设置为 48
        config.upsampler = "pixelshuffle_aux"  # 使用像素混洗辅助上采样器
    # “Swin2SR_Lightweight_X2_64”检查点配置
    elif "Swin2SR_Lightweight_X2_64" in checkpoint_url:
        config.depths = [6, 6, 6, 6]  # 网络深度分别为 [6, 6, 6, 6]
        config.embed_dim = 60  # 嵌入维度设置为 60
        config.num_heads = [6, 6, 6, 6]  # 注意力头数分别为 [6, 6, 6, 6]
        config.upsampler = "pixelshuffledirect"  # 使用像素直接混洗上采样器
    # “Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR”检查点配置
    elif "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR" in checkpoint_url:
        config.upscale = 4  # 图像放大因子为 4
        config.upsampler = "nearest+conv"  # 使用最近邻插值加卷积上采样器
    # “Swin2SR_Jpeg_dynamic”检查点配置
    elif "Swin2SR_Jpeg_dynamic" in checkpoint_url:
        config.num_channels = 1  # 图像通道数为 1
        config.upscale = 1  # 图像放大因子为 1
        config.image_size = 126  # 图像尺寸设置为 126
        config.window_size = 7  # 窗口大小为 7
        config.img_range = 255.0  # 图像数值范围为 255.0
        config.upsampler = ""  # 不使用上采样器

    return config  # 返回配置对象


# 重命名模型参数键名
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
    # 如果参数名中包含 "v_bias"，则替换为 "value.bias"
    if "v_bias" in name:
        name = name.replace("v_bias", "value.bias")
    
    # 如果参数名中包含 "cpb_mlp"，则替换为 "continuous_position_bias_mlp"
    if "cpb_mlp" in name:
        name = name.replace("cpb_mlp", "continuous_position_bias_mlp")
    
    # 如果参数名中包含 "patch_embed.proj"，则替换为 "patch_embed.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "patch_embed.projection")
    
    # 如果参数名为 "norm.weight"，则替换为 "layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    
    # 如果参数名为 "norm.bias"，则替换为 "layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"
    
    # 如果参数名中包含 "conv_first"，则替换为 "first_convolution"
    if "conv_first" in name:
        name = name.replace("conv_first", "first_convolution")
    
    # 如果参数名中包含以下字符串之一，则根据不同条件进行替换
    if (
        "upsample" in name
        or "conv_before_upsample" in name
        or "conv_bicubic" in name
        or "conv_up" in name
        or "conv_hr" in name
        or "conv_last" in name
        or "aux" in name
    ):
        # 如果参数名中包含 "conv_last"，则替换为 "final_convolution"
        if "conv_last" in name:
            name = name.replace("conv_last", "final_convolution")
        
        # 根据配置文件中的上采样方法进行不同的替换
        if config.upsampler in ["pixelshuffle", "pixelshuffle_aux", "nearest+conv"]:
            # 如果参数名中包含 "conv_before_upsample.0"，则替换为 "conv_before_upsample"
            if "conv_before_upsample.0" in name:
                name = name.replace("conv_before_upsample.0", "conv_before_upsample")
            # 如果参数名中包含 "upsample.0"，则替换为 "upsample.convolution_0"
            if "upsample.0" in name:
                name = name.replace("upsample.0", "upsample.convolution_0")
            # 如果参数名中包含 "upsample.2"，则替换为 "upsample.convolution_1"
            if "upsample.2" in name:
                name = name.replace("upsample.2", "upsample.convolution_1")
            # 在参数名前加上 "upsample."
            name = "upsample." + name
        elif config.upsampler == "pixelshuffledirect":
            # 将参数名中的 "upsample.0.weight" 替换为 "upsample.conv.weight"
            name = name.replace("upsample.0.weight", "upsample.conv.weight")
            # 将参数名中的 "upsample.0.bias" 替换为 "upsample.conv.bias"
            name = name.replace("upsample.0.bias", "upsample.conv.bias")
        else:
            pass
    else:
        # 如果以上条件都不满足，则在参数名前加上 "swin2sr."
        name = "swin2sr." + name
    
    # 返回替换后的参数名
    return name
# 将原始的模型参数字典转换为适用于Swin2SR模型的参数字典
def convert_state_dict(orig_state_dict, config):
    # 使用循环遍历原始参数字典的键，需要使用copy()方法以避免在迭代时修改原始字典
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键中包含"qkv"，则将参数转换为Swin2SR模型中对应的键
        if "qkv" in key:
            # 拆分键，获取阶段号、块号和维度信息
            key_split = key.split(".")
            stage_num = int(key_split[1])
            block_num = int(key_split[4])
            dim = config.embed_dim

            # 根据键中是否包含"weight"，将值映射到Swin2SR模型的对应键
            if "weight" in key:
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
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.query.bias"
                ] = val[:dim]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.key.bias"
                ] = val[dim : dim * 2]
                orig_state_dict[
                    f"swin2sr.encoder.stages.{stage_num}.layers.{block_num}.attention.self.value.bias"
                ] = val[-dim:]
        # 如果键中不包含"qkv"，则直接将键转换为适用于Swin2SR模型的键
        else:
            orig_state_dict[rename_key(key, config)] = val

    return orig_state_dict

# 加载Swin2SR模型的预训练检查点，转换参数，并加载到模型中
def convert_swin2sr_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub):
    # 获取Swin2SR模型的配置
    config = get_config(checkpoint_url)
    # 根据配置创建Swin2SR模型
    model = Swin2SRForImageSuperResolution(config)
    # 设置模型为评估模式
    model.eval()

    # 从URL加载预训练检查点的参数字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    # 转换参数字典
    new_state_dict = convert_state_dict(state_dict, config)
    # 将转换后的参数加载到模型中，strict=False表示允许缺失的键
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

    # 检查是否有缺失的键
    if len(missing_keys) > 0:
        raise ValueError("Missing keys when converting: {}".format(missing_keys))
    # 检查是否有意外的键
    for key in unexpected_keys:
        if not ("relative_position_index" in key or "relative_coords_table" in key or "self_mask" in key):
            raise ValueError(f"Unexpected key {key} in state_dict")

    # 验证转换后的参数是否正确
    url = "https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    processor = Swin2SRImageProcessor()
    # 对图像进行预处理
    image_size = 126 if "Jpeg" in checkpoint_url else 256
    transforms = Compose(
        [
            Resize((image_size, image_size)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    pixel_values = transforms(image).unsqueeze(0)

    # 如果图像通道数为1，则调整通道数
    if config.num_channels == 1:
        pixel_values = pixel_values[:, 0, :, :].unsqueeze(1)

# 暂时注释掉下面的代码，因为变量processor未被使用，可能需要后续补充
# pixel_values = processor(image, return_tensors="pt").pixel_values
    # 使用模型对输入的像素值进行推理，得到输出结果
    outputs = model(pixel_values)

    # 根据 checkpoint_url 的值确定预期的输出形状和切片值
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
        # TODO values didn't match exactly here
        # 根据 checkpoint_url 确定预期的输出形状和切片值（尚未精确匹配）
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

    # 检查模型的重构输出的形状是否符合预期
    assert (
        outputs.reconstruction.shape == expected_shape
    ), f"Shape of reconstruction should be {expected_shape}, but is {outputs.reconstruction.shape}"

    # 检查模型的重构输出的部分切片是否与预期相近
    assert torch.allclose(outputs.reconstruction[0, 0, :3, :3], expected_slice, atol=1e-3)

    # 打印提示信息
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

    # 获取模型名称
    model_name = url_to_name[checkpoint_url]

    # 如果指定了 pytorch_dump_folder_path，则保存模型和图像处理器
    if pytorch_dump_folder_path is not None:
        # 打印保存模型的提示信息
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 保存模型到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印保存图像处理器的提示信息
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        # 保存图像处理器到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果 push_to_hub 为真，则将模型和图像处理器推送到 Hub
    if push_to_hub:
        # 将模型推送到 Hub
        model.push_to_hub(f"caidas/{model_name}")
        # 将图像处理器推送到 Hub
        processor.push_to_hub(f"caidas/{model_name}")
# 如果当前脚本作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--checkpoint_url",  # 模型检查点的 URL 地址，默认为 Swin2SR 模型的一个版本的检查点
        default="https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X2_64.pth",
        type=str,
        help="URL of the original Swin2SR checkpoint you'd like to convert.",  # 提示信息：希望转换的原始 Swin2SR 检查点的 URL
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 转换后的 PyTorch 模型保存目录的路径，默认为 None
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",  # 提示信息：输出 PyTorch 模型的目录路径
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the converted model to the hub.")  # 是否将转换后的模型推送到 Hub

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 Swin2SR 模型检查点转换为 PyTorch 模型
    convert_swin2sr_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
```