# `.\models\focalnet\convert_focalnet_to_hf_format.py`

```py
# 导入必要的库和模块
import argparse  # 导入命令行参数解析模块
import json  # 导入处理 JSON 格式数据的模块

import requests  # 导入发送 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数
from PIL import Image  # 导入 Python 图像处理库 PIL
from torchvision import transforms  # 导入 PyTorch 中用于图像处理的模块

from transformers import BitImageProcessor, FocalNetConfig, FocalNetForImageClassification  # 导入 Transformers 库中相关模型和配置
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling  # 导入图像处理的一些常量和函数


def get_focalnet_config(model_name):
    # 根据模型名称选择不同的深度配置
    depths = [2, 2, 6, 2] if "tiny" in model_name else [2, 2, 18, 2]
    # 根据模型名称选择是否使用卷积嵌入
    use_conv_embed = True if "large" in model_name or "huge" in model_name else False
    # 根据模型名称选择是否使用后层归一化
    use_post_layernorm = True if "large" in model_name or "huge" in model_name else False
    # 根据模型名称选择是否使用层缩放
    use_layerscale = True if "large" in model_name or "huge" in model_name else False

    if "large" in model_name or "xlarge" in model_name or "huge" in model_name:
        # 根据模型名称和类型设置聚焦层级和窗口大小
        if "fl3" in model_name:
            focal_levels = [3, 3, 3, 3]
            focal_windows = [5, 5, 5, 5]
        elif "fl4" in model_name:
            focal_levels = [4, 4, 4, 4]
            focal_windows = [3, 3, 3, 3]

    if "tiny" in model_name or "small" in model_name or "base" in model_name:
        # 根据模型名称设置默认的聚焦窗口大小和层级
        focal_windows = [3, 3, 3, 3]
        if "lrf" in model_name:
            focal_levels = [3, 3, 3, 3]
        else:
            focal_levels = [2, 2, 2, 2]

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

    # 从 Hugging Face Hub 下载标签文件，并加载为字典
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 转换字典键的数据类型为整数
    label2id = {v: k for k, v in id2label.items()}  # 创建反向标签到ID的映射字典
    # 使用提供的参数创建一个 FocalNetConfig 对象
    config = FocalNetConfig(
        embed_dim=embed_dim,  # 设定嵌入维度
        depths=depths,  # 设置深度参数
        focal_levels=focal_levels,  # 指定焦点级别
        focal_windows=focal_windows,  # 指定焦点窗口大小
        use_conv_embed=use_conv_embed,  # 标志是否使用卷积嵌入
        id2label=id2label,  # 用于标识到标签的映射
        label2id=label2id,  # 用于标签到标识的映射
        use_post_layernorm=use_post_layernorm,  # 标志是否使用层标准化后处理
        use_layerscale=use_layerscale,  # 标志是否使用层比例
    )
    
    # 返回创建的 FocalNetConfig 对象作为函数的结果
    return config
    # 检查名字中是否包含 "patch_embed.proj"，如果是则替换成 "embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "embeddings.patch_embeddings.projection")
    
    # 检查名字中是否包含 "patch_embed.norm"，如果是则替换成 "embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "embeddings.norm")
    
    # 如果名字中包含 "layers"，则在名字前加上 "encoder."
    if "layers" in name:
        name = "encoder." + name
    
    # 检查名字中是否包含 "encoder.layers"，如果是则替换成 "encoder.stages"
    if "encoder.layers" in name:
        name = name.replace("encoder.layers", "encoder.stages")
    
    # 检查名字中是否包含 "downsample.proj"，如果是则替换成 "downsample.projection"
    if "downsample.proj" in name:
        name = name.replace("downsample.proj", "downsample.projection")
    
    # 如果名字中包含 "blocks"，则替换成 "layers"
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    
    # 如果名字中包含 "modulation.f.weight" 或 "modulation.f.bias"，则替换成 "modulation.projection_in"
    if "modulation.f.weight" in name or "modulation.f.bias" in name:
        name = name.replace("modulation.f", "modulation.projection_in")
    
    # 如果名字中包含 "modulation.h.weight" 或 "modulation.h.bias"，则替换成 "modulation.projection_context"
    if "modulation.h.weight" in name or "modulation.h.bias" in name:
        name = name.replace("modulation.h", "modulation.projection_context")
    
    # 如果名字中包含 "modulation.proj.weight" 或 "modulation.proj.bias"，则替换成 "modulation.projection_out"
    if "modulation.proj.weight" in name or "modulation.proj.bias" in name:
        name = name.replace("modulation.proj", "modulation.projection_out")
    
    # 如果名字是 "norm.weight"，则替换成 "layernorm.weight"
    if name == "norm.weight":
        name = "layernorm.weight"
    
    # 如果名字是 "norm.bias"，则替换成 "layernorm.bias"
    if name == "norm.bias":
        name = "layernorm.bias"
    
    # 如果名字中包含 "head"，则替换成 "classifier"，否则加上 "focalnet."
    if "head" in name:
        name = name.replace("head", "classifier")
    else:
        name = "focalnet." + name
    
    # 返回修改后的名字
    return name
    # 使用循环遍历 state_dict 的键的副本，并逐一处理
    for key in state_dict.copy().keys():
        # 弹出当前键对应的值
        val = state_dict.pop(key)
        # 使用重命名函数处理键，并将键值对添加回 state_dict
        state_dict[rename_key(key)] = val

    # 根据模型名称获取配置信息
    config = get_focalnet_config(model_name)
    # 根据配置信息创建图像分类模型
    model = FocalNetForImageClassification(config)
    # 设置模型为评估模式
    model.eval()

    # 加载 state_dict 到模型中
    model.load_state_dict(state_dict)

    # 验证图像转换
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"

    # 创建图像处理器实例，进行预处理操作
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
    # 使用 requests 获取图像并用 PIL 打开
    image = Image.open(requests.get(url, stream=True).raw)
    # 对图像应用图像处理器，返回处理后的张量形式输入
    inputs = processor(images=image, return_tensors="pt")

    # 定义图像转换流水线
    image_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 获取原始像素值的张量表示
    original_pixel_values = image_transforms(image).unsqueeze(0)

    # 验证像素值是否接近
    assert torch.allclose(inputs.pixel_values, original_pixel_values, atol=1e-4)

    # 将输入传递给模型，获取输出
    outputs = model(**inputs)

    # 获取预测的类别索引
    predicted_class_idx = outputs.logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    # 打印 logits 的前三个值
    print("First values of logits:", outputs.logits[0, :3])

    # 根据模型名称选择预期的 logits 切片值，并进行验证
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
    # 验证预期的 logits 切片值是否与实际输出接近
    assert torch.allclose(outputs.logits[0, :3], expected_slice, atol=1e-4)
    print("Looks ok!")

    # 如果指定了 PyTorch 模型保存路径，保存模型和处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and processor of {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果设置了推送到 Hub 的标志，将模型和处理器推送到 Hub
    if push_to_hub:
        print(f"Pushing model and processor of {model_name} to the hub...")
        model.push_to_hub(f"{model_name}")
        processor.push_to_hub(f"{model_name}")
if __name__ == "__main__":
    # 如果作为主程序运行，执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="focalnet-tiny",
        type=str,
        help="Name of the FocalNet model you'd like to convert.",
    )
    # 添加一个必需的参数 `--model_name`，默认为 "focalnet-tiny"，类型为字符串，用于指定要转换的 FocalNet 模型的名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个参数 `--pytorch_dump_folder_path`，默认为 None，类型为字符串，用于指定输出的 PyTorch 模型目录的路径

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub.",
    )
    # 添加一个参数 `--push_to_hub`，如果存在则表示要推送模型和处理器到 hub

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 `args` 变量中

    convert_focalnet_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
    # 调用函数 `convert_focalnet_checkpoint`，传入从命令行解析得到的参数 `model_name`, `pytorch_dump_folder_path`, `push_to_hub`
```