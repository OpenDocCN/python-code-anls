# `.\models\focalnet\convert_focalnet_to_hf_format.py`

```
# 导入必要的库
import argparse  # 用于解析命令行参数
import json  # 用于 JSON 数据的读取和处理

import requests  # 用于发送 HTTP 请求
import torch  # PyTorch 库
from huggingface_hub import hf_hub_download  # 用于从 HF Hub 下载资源
from PIL import Image  # Python Imaging Library，用于图像处理
from torchvision import transforms  # PyTorch 的图像转换功能

# 导入 transformers 库中相关模块
from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling


# 定义函数以获取 FocalNet 模型的配置信息
def get_focalnet_config(model_name):
    # 根据模型名称确定不同的深度和是否使用不同的层级特性
    depths = [2, 2, 6, 2] if "tiny" in model_name else [2, 2, 18, 2]
    use_conv_embed = True if "large" in model_name or "huge" in model_name else False
    use_post_layernorm = True if "large" in model_name or "huge" in model_name else False
    use_layerscale = True if "large" in model_name or "huge" in model_name else False

    # 根据模型名称确定不同的焦点机制参数
    if "large" in model_name or "xlarge" in model_name or "huge" in model_name:
        if "fl3" in model_name:
            focal_levels = [3, 3, 3, 3]
            focal_windows = [5, 5, 5, 5]
        elif "fl4" in model_name:
            focal_levels = [4, 4, 4, 4]
            focal_windows = [3, 3, 3, 3]

    # 根据模型名称确定不同的焦点机制参数
    if "tiny" in model_name or "small" in model_name or "base" in model_name:
        focal_windows = [3, 3, 3, 3]
        if "lrf" in model_name:
            focal_levels = [3, 3, 3, 3]
        else:
            focal_levels = [2, 2, 2, 2]

    # 根据模型名称确定不同的嵌入维度
    if "tiny" in model_name:
        embed_dim = 96
    elif "small" in model_name:
        embed_dim = 96
    elif "base" in model_name:
        embed_dim = 128
    elif "large" in model_name:
        embed_dim = 192
    elif "xlarge" in model_name:
        embed_dim = 256
    elif "huge" in model_name:
        embed_dim = 352

    # 设置标签信息
    repo_id = "huggingface/label-files"
    if "large" in model_name or "huge" in model_name:
        filename = "imagenet-22k-id2label.json"
    else:
        filename = "imagenet-1k-id2label.json"

    # 从 HF Hub 下载标签文件并加载为字典形式
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 将键转换为整数型
    label2id = {v: k for k, v in id2label.items()}  # 创建反向映射关系的字典
    # 创建一个 FocalNetConfig 类的实例，传入各项参数
    config = FocalNetConfig(
        embed_dim=embed_dim,  # 嵌入维度
        depths=depths,  # 深度
        focal_levels=focal_levels,  # 焦点级别
        focal_windows=focal_windows,  # 焦点窗口大小
        use_conv_embed=use_conv_embed,  # 是否使用卷积嵌入
        id2label=id2label,  # ID 到标签的映射
        label2id=label2id,  # 标签到ID的映射
        use_post_layernorm=use_post_layernorm,  # 是否使用后层标准化
        use_layerscale=use_layerscale,  # 是否使用层标准化
    )
    
    # 返回 FocalNetConfig 实例
    return config
# 定义一个用于重命名键的函数
def rename_key(name):
    # 如果键包含 "patch_embed.proj"，将其替换为 "embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    # 如果键包含 "patch_embed.norm"，将其替换为 "embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    # 如果键包含 "layers"，在其前面加上 "encoder."
    if "layers" in name:
        name = "encoder." + name
    # 如果键包含 "encoder.layers"，将其替换为 "encoder.stages"
    if "encoder.layers" in name:
        name = name.replace("encoder.layers", "encoder.stages")
    # 如果键包含 "downsample.proj"，将其替换为 "downsample.projection"
    if "downsample.proj" in name:
        name = name.replace("downsample.proj", "downsample.projection")
    # 如果键包含 "blocks"，将其替换为 "layers"
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    # 如果键包含 "modulation.f.weight" 或 "modulation.f.bias"，将其替换为 "modulation.projection_in"
    if "modulation.f.weight" in name or "modulation.f.bias" in name:
        name = name.replace("modulation.f", "modulation.projection_in")
    # 如果键包含 "modulation.h.weight" 或 "modulation.h.bias"，将其替换为 "modulation.projection_context"
    if "modulation.h.weight" in name or "modulation.h.bias" in name:
        name = name.replace("modulation.h", "modulation.projection_context")
    # 如果键包含 "modulation.proj.weight" 或 "modulation.proj.bias"，将其替换为 "modulation.projection_out"
    if "modulation.proj.weight" in name or "modulation.proj.bias" in name:
        name = name.replace("modulation.proj", "modulation.projection_out")

    # 如果键等于 "norm.weight"，将其替换为 "layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    # 如果键等于 "norm.bias"，将其替换为 "layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"

    # 如果键包含 "head"，将其替换为 "classifier"
    if "head" in name:
        name = name.replace("head", "classifier")
    # 否则，在键的前面加上 "focalnet."
    else:
        name = "focalnet." + name

    # 返回修改后的键名
    return name


# 定义一个用于转换 FocalNet 检查点的函数
def convert_focalnet_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    # 格式化：关闭自动代码格式化
    # 定义模型名称到 URL 的映射
    model_name_to_url = {
        "focalnet-tiny": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_srf.pth",
        "focalnet-tiny-lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_tiny_lrf.pth",
        "focalnet-small": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_srf.pth",
        "focalnet-small-lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_small_lrf.pth",
        "focalnet-base": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_srf.pth",
        "focalnet-base-lrf": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_base_lrf.pth",
        "focalnet-large-lrf-fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384.pth",
        "focalnet-large-lrf-fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_large_lrf_384_fl4.pth",
        "focalnet-xlarge-lrf-fl3": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384.pth",
        "focalnet-xlarge-lrf-fl4": "https://projects4jw.blob.core.windows.net/focalnet/release/classification/focalnet_xlarge_lrf_384_fl4.pth",
    }
    # 格式化：开启自动代码格式化

    # 根据模型名称从字典中获取对应的检查点 URL
    checkpoint_url = model_name_to_url[model_name]
    # 打印检查点的 URL
    print("Checkpoint URL: ", checkpoint_url)
    # 从检查点 URL 加载状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # 重命名状态字典中的键
    # 遍历 state_dict 的复制，并删除原始 state_dict 中的每个键值对，将经过 rename_key 函数重命名后的键值对添加回 state_dict
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val

    # 根据模型名称获取 FocalNet 的配置信息
    config = get_focalnet_config(model_name)
    # 根据配置信息创建 FocalNetForImageClassification 模型
    model = FocalNetForImageClassification(config)
    # 设置模型为评估模式
    model.eval()

    # 加载状态字典到模型中
    model.load_state_dict(state_dict)

    # 验证模型转换后的效果
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 创建图像处理器，进行预处理
    processor = BitImageProcessor(
        do_resize=True,
        size={"shortest_edge": 256},
        resample=PILImageResampling.BILINEAR,
        do_center_crop=True,
        crop_size=224,
        do_normalize=True,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
    )
    # 从 URL 中读取图像
    image = Image.open(requests.get(url, stream=True).raw)
    # 对图像进行处理得到模型的输入
    inputs = processor(images=image, return_tensors="pt")

    # 定义图像变换
    image_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 获取原始像素值
    original_pixel_values = image_transforms(image).unsqueeze(0)

    # 验证像素值
    assert torch.allclose(inputs.pixel_values, original_pixel_values, atol=1e-4)

    # 对模型进行推断
    outputs = model(**inputs)

    # 获取预测的类别索引
    predicted_class_idx = outputs.logits.argmax(-1).item()
    # 打印预测的类别
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    # 打印 logits 的前三个值
    print("First values of logits:", outputs.logits[0, :3])

    # 根据模型名称确定期望的 logits 切片，并进行断言验证
    if model_name == "focalnet-tiny":
        expected_slice = torch.tensor([0.2166, -0.4368, 0.2191])
    elif model_name == "focalnet-tiny-lrf":
        expected_slice = torch.tensor([1.1669, 0.0125, -0.1695])
    elif model_name == "focalnet-small":
        expected_slice = torch.tensor([0.4917, -0.0430, 0.1341])
    elif model_name == "focalnet-small-lrf":
        expected_slice = torch.tensor([-0.2588, -0.5342, -0.2331])
    elif model_name == "focalnet-base":
        expected_slice = torch.tensor([-0.1655, -0.4090, -0.1730])
    elif model_name == "focalnet-base-lrf":
        expected_slice = torch.tensor([0.5306, -0.0483, -0.3928])
    # 断言模型输出的前三个 logits 与期望的值相近
    assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
    # 打印提示信息
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub，则推送模型和处理器
    if push_to_hub:
        print(f"Pushing model and processor of {model_name} to the hub...")
        model.push_to_hub(f"{model_name}")
        processor.push_to_hub(f"{model_name}")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",  # 模型名称参数
        default="focalnet-tiny",  # 默认值为"focalnet-tiny"
        type=str,  # 参数类型为字符串
        help="Name of the FocalNet model you'd like to convert.",  # 参数的帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # PyTorch 模型输出目录的参数
        default=None,  # 默认值为 None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 参数的帮助信息
    )
    parser.add_argument(
        "--push_to_hub",  # 是否将模型和处理器推送到 Hub 的参数
        action="store_true",  # 如果存在，则将其设置为 True
        help="Whether to push the model and processor to the hub."  # 参数的帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_focalnet_checkpoint 函数，将解析的参数传递给它
    convert_focalnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```