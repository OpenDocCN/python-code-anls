# `.\models\efficientformer\convert_efficientformer_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置文件编码为UTF-8，确保可以正确解析中文和其他非ASCII字符
# 版权声明，声明代码版权归The HuggingFace Inc.团队所有，使用Apache License 2.0许可证
# 仅在符合许可证的情况下可以使用本文件
# 可以在以下链接获取完整的许可证文本：http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或书面同意的情况，软件按"原样"提供，不附带任何明示或暗示的担保或条件
# 详见许可证，限制软件使用的条件

"""从原始仓库转换EfficientFormer检查点。

URL: https://github.com/snap-research/EfficientFormer
"""

import argparse  # 导入用于解析命令行参数的模块
import re  # 导入正则表达式模块
from pathlib import Path  # 导入用于处理文件和目录路径的模块

import requests  # 导入用于发送HTTP请求的模块
import torch  # 导入PyTorch深度学习框架
from PIL import Image  # 导入PIL图像处理库中的Image模块
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor  # 导入图像转换函数

from transformers import (  # 导入transformers库中的相关类和函数
    EfficientFormerConfig,
    EfficientFormerForImageClassificationWithTeacher,
    EfficientFormerImageProcessor,
)
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling  # 导入图像处理相关常量和函数

def rename_key(old_name, num_meta4D_last_stage):
    new_name = old_name

    if "patch_embed" in old_name:  # 如果旧键名中包含'patch_embed'
        _, layer, param = old_name.split(".")  # 拆分键名，获取层和参数信息

        if layer == "0":  # 如果层为'0'
            new_name = old_name.replace("0", "convolution1")  # 替换为'convolution1'
        elif layer == "1":  # 如果层为'1'
            new_name = old_name.replace("1", "batchnorm_before")  # 替换为'batchnorm_before'
        elif layer == "3":  # 如果层为'3'
            new_name = old_name.replace("3", "convolution2")  # 替换为'convolution2'
        else:  # 其他情况
            new_name = old_name.replace("4", "batchnorm_after")  # 替换为'batchnorm_after'
    # 如果旧名称中包含"network"且包含形如数字.数字的模式
    if "network" in old_name and re.search(r"\d\.\d", old_name):
        # 匹配两位数字的正则表达式模式
        two_digit_num = r"\b\d{2}\b"
        # 如果旧名称中存在两位数字，则匹配数字.数字.的模式
        if bool(re.search(two_digit_num, old_name)):
            match = re.search(r"\d\.\d\d.", old_name).group()
        else:
            match = re.search(r"\d\.\d.", old_name).group()
        # 如果匹配到的第一个数字小于6
        if int(match[0]) < 6:
            # 删除匹配的部分，并替换"network"为第一个数字.meta4D_layers.blocks.剩余部分
            trimmed_name = old_name.replace(match, "")
            trimmed_name = trimmed_name.replace("network", match[0] + ".meta4D_layers.blocks." + match[2:-1])
            # 新名称为intermediate_stages.修剪后的名称
            new_name = "intermediate_stages." + trimmed_name
        else:
            # 删除匹配的部分，并根据条件替换"network"为不同的字符串
            trimmed_name = old_name.replace(match, "")
            if int(match[2]) < num_meta4D_last_stage:
                trimmed_name = trimmed_name.replace("network", "meta4D_layers.blocks." + match[2])
            else:
                layer_index = str(int(match[2]) - num_meta4D_last_stage)
                trimmed_name = trimmed_name.replace("network", "meta3D_layers.blocks." + layer_index)
                # 如果名称中包含"norm1"，替换为"layernorm1"
                if "norm1" in old_name:
                    trimmed_name = trimmed_name.replace("norm1", "layernorm1")
                elif "norm2" in old_name:
                    trimmed_name = trimmed_name.replace("norm2", "layernorm2")
                elif "fc1" in old_name:
                    trimmed_name = trimmed_name.replace("fc1", "linear_in")
                elif "fc2" in old_name:
                    trimmed_name = trimmed_name.replace("fc2", "linear_out")

            # 新名称为last_stage.修剪后的名称
            new_name = "last_stage." + trimmed_name

    # 如果旧名称中包含"network"且包含形如.数字.的模式
    elif "network" in old_name and re.search(r".\d.", old_name):
        # 将"network"替换为"intermediate_stages"
        new_name = old_name.replace("network", "intermediate_stages")

    # 如果新名称中包含"fc"，替换为"convolution"
    if "fc" in new_name:
        new_name = new_name.replace("fc", "convolution")
    # 如果新名称中包含"norm1"且不包含"layernorm1"，替换为"batchnorm_before"
    elif ("norm1" in new_name) and ("layernorm1" not in new_name):
        new_name = new_name.replace("norm1", "batchnorm_before")
    # 如果新名称中包含"norm2"且不包含"layernorm2"，替换为"batchnorm_after"
    elif ("norm2" in new_name) and ("layernorm2" not in new_name):
        new_name = new_name.replace("norm2", "batchnorm_after")
    # 如果新名称中包含"proj"，替换为"projection"
    if "proj" in new_name:
        new_name = new_name.replace("proj", "projection")
    # 如果新名称中包含"dist_head"，替换为"distillation_classifier"
    if "dist_head" in new_name:
        new_name = new_name.replace("dist_head", "distillation_classifier")
    # 如果新名称中包含"head"，替换为"classifier"
    elif "head" in new_name:
        new_name = new_name.replace("head", "classifier")
    # 如果新名称中包含"patch_embed"，在新名称前添加"efficientformer."
    elif "patch_embed" in new_name:
        new_name = "efficientformer." + new_name
    # 如果新名称为"norm.weight"或"norm.bias"，替换为"layernorm."并在新名称前添加"efficientformer."
    elif new_name == "norm.weight" or new_name == "norm.bias":
        new_name = new_name.replace("norm", "layernorm")
        new_name = "efficientformer." + new_name
    else:
        # 否则在新名称前添加"efficientformer.encoder."
        new_name = "efficientformer.encoder." + new_name

    # 返回处理后的新名称
    return new_name
# 将给定的检查点中的键重命名，返回更新后的检查点
def convert_torch_checkpoint(checkpoint, num_meta4D_last_stage):
    # 使用循环遍历检查点的副本中的所有键
    for key in checkpoint.copy().keys():
        # 弹出当前键对应的值，并用新键重新放置到检查点中
        val = checkpoint.pop(key)
        checkpoint[rename_key(key, num_meta4D_last_stage)] = val

    # 返回更新后的检查点
    return checkpoint


# 我们将在 COCO 图像上验证我们的结果
def prepare_img():
    # 定义 COCO 数据集中的图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库获取图像的原始字节流，并用 PIL 打开图像
    image = Image.open(requests.get(url, stream=True).raw)

    # 返回打开的图像对象
    return image


def convert_efficientformer_checkpoint(
    checkpoint_path: Path, efficientformer_config_file: Path, pytorch_dump_path: Path, push_to_hub: bool
):
    # 加载原始检查点的状态字典，定位到模型部分
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 从 JSON 文件中加载 EfficientFormer 的配置
    config = EfficientFormerConfig.from_json_file(efficientformer_config_file)
    # 创建 EfficientFormer 模型对象
    model = EfficientFormerForImageClassificationWithTeacher(config)
    # 提取模型名称，用于后续操作
    model_name = "_".join(checkpoint_path.split("/")[-1].split(".")[0].split("_")[:-1])

    # 计算最后一个阶段的元 4D 数量
    num_meta4D_last_stage = config.depths[-1] - config.num_meta3d_blocks + 1
    # 转换原始状态字典的键
    new_state_dict = convert_torch_checkpoint(orig_state_dict, num_meta4D_last_stage)

    # 加载转换后的状态字典到模型中
    model.load_state_dict(new_state_dict)
    # 将模型设置为评估模式
    model.eval()

    # 定义 Pillow 库中支持的图像重采样方式
    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # 准备图像
    image = prepare_img()
    image_size = 256
    crop_size = 224
    # 创建 EfficientFormer 图像处理器实例
    processor = EfficientFormerImageProcessor(
        size={"shortest_edge": image_size},
        crop_size={"height": crop_size, "width": crop_size},
        resample=pillow_resamplings["bicubic"],
    )
    # 使用处理器处理图像并获取像素值张量
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # 原始的图像处理流程
    image_transforms = Compose(
        [
            Resize(image_size, interpolation=pillow_resamplings["bicubic"]),
            CenterCrop(crop_size),
            ToTensor(),
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    # 对原始图像进行变换并添加批次维度
    original_pixel_values = image_transforms(image).unsqueeze(0)

    # 断言保证处理后的像素值与原始像素值的接近度
    assert torch.allclose(original_pixel_values, pixel_values)

    # 将像素值输入模型并获取输出
    outputs = model(pixel_values)
    logits = outputs.logits

    # 预期的输出形状
    expected_shape = (1, 1000)

    # 根据模型名称验证不同情况下的输出对数是否正确
    if "l1" in model_name:
        expected_logits = torch.Tensor(
            [-0.1312, 0.4353, -1.0499, -0.5124, 0.4183, -0.6793, -1.3777, -0.0893, -0.7358, -2.4328]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    elif "l3" in model_name:
        expected_logits = torch.Tensor(
            [-1.3150, -1.5456, -1.2556, -0.8496, -0.7127, -0.7897, -0.9728, -0.3052, 0.3751, -0.3127]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    # 如果模型名称包含 "l7"
    elif "l7" in model_name:
        # 预期的 logits 值，作为模型输出的期望值
        expected_logits = torch.Tensor(
            [-1.0283, -1.4131, -0.5644, -1.3115, -0.5785, -1.2049, -0.7528, 0.1992, -0.3822, -0.0878]
        )
        # 断言当前模型输出的形状符合预期的形状
        assert logits.shape == expected_shape
    else:
        # 如果模型名称不在已知的支持列表中，抛出异常
        raise ValueError(
            f"Unknown model checkpoint: {checkpoint_path}. Supported versions of efficientformer are l1, l3, and l7"
        )

    # 保存检查点到指定路径
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    # 将模型保存到 PyTorch 的预训练模型路径下
    model.save_pretrained(pytorch_dump_path)
    # 打印保存成功的信息和路径
    print(f"Checkpoint successfully converted. Model saved at {pytorch_dump_path}")
    # 将处理器保存到 PyTorch 的预训练模型路径下
    processor.save_pretrained(pytorch_dump_path)
    # 打印保存成功的信息和路径
    print(f"Processor successfully saved at {pytorch_dump_path}")

    # 如果需要推送到 Hub
    if push_to_hub:
        # 提示开始将模型推送到 Hub
        print("Pushing model to the hub...")
        # 将模型推送到指定的 Hub 仓库
        model.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add model",
            use_temp_dir=True,
        )
        # 将处理器推送到指定的 Hub 仓库
        processor.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
if __name__ == "__main__":
    # 如果作为主程序执行，开始解析命令行参数
    parser = argparse.ArgumentParser()

    # 必需的参数
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to EfficientFormer pytorch checkpoint.",
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for EfficientFormer model config.",
    )
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model."
    )

    # 可选参数：是否将模型推送到 hub
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and image processor to the hub",
    )
    parser.set_defaults(push_to_hub=True)

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数来转换 EfficientFormer 检查点
    convert_efficientformer_checkpoint(
        checkpoint_path=args.pytorch_model_path,
        efficientformer_config_file=args.config_file,
        pytorch_dump_path=args.pytorch_dump_path,
        push_to_hub=args.push_to_hub,
    )
```