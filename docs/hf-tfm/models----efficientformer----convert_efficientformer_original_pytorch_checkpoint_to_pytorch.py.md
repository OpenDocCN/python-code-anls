# `.\models\efficientformer\convert_efficientformer_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置编码格式为 utf-8
# 版权声明
# 在 Apache 许可证 2.0 下授权 huggingface 公司使用
# 除非符合许可证的规定，否则不得使用此文件
# 可在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件属于“按原样”分发，
# 没有任何明示或暗示的担保或条件。详细信息请参阅许可证。

"""将原始仓库中的 EfficientFormer 检查点转换为适用于 Hugging Face 模型库的格式。

URL: https://github.com/snap-research/EfficientFormer
"""

# 导入必要的库
import argparse
import re
from pathlib import Path
import requests
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import (
    EfficientFormerConfig,
    EfficientFormerForImageClassificationWithTeacher,
    EfficientFormerImageProcessor,
)
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling

# 定义一个函数用来重命名键名
def rename_key(old_name, num_meta4D_last_stage):
    new_name = old_name

    # 如果键名包含"patch_embed"字符
    if "patch_embed" in old_name:
        _, layer, param = old_name.split(".")

        # 根据不同的层进行重命名
        if layer == "0":
            new_name = old_name.replace("0", "convolution1")
        elif layer == "1":
            new_name = old_name.replace("1", "batchnorm_before")
        elif layer == "3":
            new_name = old_name.replace("3", "convolution2")
        else:
            new_name = old_name.replace("4", "batchnorm_after")
    # 判断旧名称是否包含"network"且符合正则表达式"\d\.\d"的模式
    if "network" in old_name and re.search(r"\d\.\d", old_name):
        # 定义匹配两位数字的正则表达式
        two_digit_num = r"\b\d{2}\b"
        # 判断旧名称中是否存在两位数字
        if bool(re.search(two_digit_num, old_name)):
            # 获取匹配到的数字部分
            match = re.search(r"\d\.\d\d.", old_name).group()
        else:
            match = re.search(r"\d\.\d.", old_name).group()
        # 判断匹配到的数字第一位是否小于6
        if int(match[0]) < 6:
            # 在旧名称中替换匹配到的数字部分，并创建新的修剪后名称
            trimmed_name = old_name.replace(match, "")
            trimmed_name = trimmed_name.replace("network", match[0] + ".meta4D_layers.blocks." + match[2:-1])
            # 在修剪后名称前添加"intermediate_stages."，创建新的名称
            new_name = "intermediate_stages." + trimmed_name
        else:
            # 在旧名称中替换匹配到的数字部分，并创建新的修剪后名称
            trimmed_name = old_name.replace(match, "")
            # 判断匹配到的数字第二位是否小于num_meta4D_last_stage
            if int(match[2]) < num_meta4D_last_stage:
                trimmed_name = trimmed_name.replace("network", "meta4D_layers.blocks." + match[2])
            else:
                # 在修剪后名称中根据匹配到的数字计算layer_index，并进行相应的替换
                layer_index = str(int(match[2]) - num_meta4D_last_stage)
                trimmed_name = trimmed_name.replace("network", "meta3D_layers.blocks." + layer_index)
                # 根据关键词替换特定部分的名称
                if "norm1" in old_name:
                    trimmed_name = trimmed_name.replace("norm1", "layernorm1")
                elif "norm2" in old_name:
                    trimmed_name = trimmed_name.replace("norm2", "layernorm2")
                elif "fc1" in old_name:
                    trimmed_name = trimmed_name.replace("fc1", "linear_in")
                elif "fc2" in old_name:
                    trimmed_name = trimmed_name.replace("fc2", "linear_out")
            # 在修剪后名称前添加"last_stage."，创建新的名称
            new_name = "last_stage." + trimmed_name
    
    # 判断旧名称是否包含"network"且符合正则表达式".\d."的模式
    elif "network" in old_name and re.search(r".\d.", old_name):
        # 在旧名称中替换"network"为"intermediate_stages"，创建新的名称
        new_name = old_name.replace("network", "intermediate_stages")
    
    # 判断新名称是否包含"fc"
    if "fc" in new_name:
        # 将新名称中的"fc"替换为"convolution"
        new_name = new_name.replace("fc", "convolution")
    # 判断新名称是否包含"norm1"且不包含"layernorm1"
    elif ("norm1" in new_name) and ("layernorm1" not in new_name):
        # 将新名称中的"norm1"替换为"batchnorm_before"
        new_name = new_name.replace("norm1", "batchnorm_before")
    # 判断新名称是否包含"norm2"且不包含"layernorm2"
    elif ("norm2" in new_name) and ("layernorm2" not in new_name):
        # 将新名称中的"norm2"替换为"batchnorm_after"
        new_name = new_name.replace("norm2", "batchnorm_after")
    # 判断新名称是否包含"proj"
    if "proj" in new_name:
        # 将新名称中的"proj"替换为"projection"
        new_name = new_name.replace("proj", "projection")
    # 判断新名称是否包含"dist_head"
    if "dist_head" in new_name:
        # 将新名称中的"dist_head"替换为"distillation_classifier"
        new_name = new_name.replace("dist_head", "distillation_classifier")
    # 判断新名称是否包含"head"
    elif "head" in new_name:
        # 将新名称中的"head"替换为"classifier"
        new_name = new_name.replace("head", "classifier")
    # 判断新名称是否包含"patch_embed"
    elif "patch_embed" in new_name:
        # 在新名称前添加"efficientformer."
        new_name = "efficientformer." + new_name
    # 判断新名称是否等于"norm.weight"或"norm.bias"
    elif new_name == "norm.weight" or new_name == "norm.bias":
        # 将新名称中的"norm"替换为"layernorm"
        new_name = new_name.replace("norm", "layernorm")
        # 在新名称前添加"efficientformer."
        new_name = "efficientformer." + new_name
    else:
        # 在新名称前添加"efficientformer.encoder."
        new_name = "efficientformer.encoder." + new_name
    
    # 返回新的名称
    return new_name
def convert_torch_checkpoint(checkpoint, num_meta4D_last_stage):
    # 遍历检查点中的所有键
    for key in checkpoint.copy().keys():
        # 弹出键值对并保存值
        val = checkpoint.pop(key)
        # 将键重命名后重新加入到检查点中
        checkpoint[rename_key(key, num_meta4D_last_stage)] = val

    # 返回更新后的检查点
    return checkpoint


# 我们将在 COCO 图像上验证我们的结果
def prepare_img():
    # 从 URL 下载并打开 COCO 图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    # 返回图片
    return image


def convert_efficientformer_checkpoint(
    checkpoint_path: Path, efficientformer_config_file: Path, pytorch_dump_path: Path, push_to_hub: bool
):
    # 从路径中加载原始状态字典
    orig_state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 从配置文件创建 EfficientFormerConfig 对象
    config = EfficientFormerConfig.from_json_file(efficientformer_config_file)
    # 创建 EfficientFormerForImageClassificationWithTeacher 模型
    model = EfficientFormerForImageClassificationWithTeacher(config)
    # 从模型路径解析模型名称
    model_name = "_".join(checkpoint_path.split("/")[-1].split(".")[0].split("_")[:-1])

    # 计算 num_meta4D_last_stage 值
    num_meta4D_last_stage = config.depths[-1] - config.num_meta3d_blocks + 1
    # 转换原始状态字典中的键名
    new_state_dict = convert_torch_checkpoint(orig_state_dict, num_meta4D_last_stage)

    # 加载模型状态字典
    model.load_state_dict(new_state_dict)
    # 设为评估模式
    model.eval()

    # 定义不同的 PIL 重采样方法
    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # 准备图像
    image = prepare_img()
    image_size = 256
    crop_size = 224
    # 创建 EfficientFormerImageProcessor 处理器
    processor = EfficientFormerImageProcessor(
        size={"shortest_edge": image_size},
        crop_size={"height": crop_size, "width": crop_size},
        resample=pillow_resamplings["bicubic"],
    )
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # 原始处理流程
    image_transforms = Compose(
        [
            # 重设大小并保持长宽比，采用 bicubic 差值
            Resize(image_size, interpolation=pillow_resamplings["bicubic"]),
            # 中心裁剪
            CenterCrop(crop_size),
            # 转换为张量
            ToTensor(),
            # 标准化
            Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ]
    )
    original_pixel_values = image_transforms(image).unsqueeze(0)

    # 断言原始处理流输出和 EfficientFormerImageProcessor 处理器输出一致
    assert torch.allclose(original_pixel_values, pixel_values)

    # 模型推理
    outputs = model(pixel_values)
    logits = outputs.logits

    expected_shape = (1, 1000)

    if "l1" in model_name:
        # 验证期望的 logits 输出（示例数值，仅作示范）
        expected_logits = torch.Tensor(
            [-0.1312, 0.4353, -1.0499, -0.5124, 0.4183, -0.6793, -1.3777, -0.0893, -0.7358, -2.4328]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    elif "l3" in model_name:
        # 验证期望的 logits 输出（示例数值，仅作示范）
        expected_logits = torch.Tensor(
            [-1.3150, -1.5456, -1.2556, -0.8496, -0.7127, -0.7897, -0.9728, -0.3052, 0.3751, -0.3127]
        )
        assert torch.allclose(logits[0, :10], expected_logits, atol=1e-3)
        assert logits.shape == expected_shape
    # 如果模型名称中包含"l7"
    elif "l7" in model_name:
        # 预期的logits值
        expected_logits = torch.Tensor(
            [-1.0283, -1.4131, -0.5644, -1.3115, -0.5785, -1.2049, -0.7528, 0.1992, -0.3822, -0.0878]
        )
        # 检查模型logits形状是否符合预期形状
        assert logits.shape == expected_shape
    else:
        # 抛出数值错误，指出未知的模型检查点
        raise ValueError(
            f"Unknown model checkpoint: {checkpoint_path}. Supported version of efficientformer are l1, l3 and l7"
        )

    # 保存检查点
    Path(pytorch_dump_path).mkdir(exist_ok=True)
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_path)
    # 打印保存成功信息
    print(f"Checkpoint successfuly converted. Model saved at {pytorch_dump_path}")
    # 保存处理器到指定路径
    processor.save_pretrained(pytorch_dump_path)
    # 打印处理器保存成功信息
    print(f"Processor successfuly saved at {pytorch_dump_path}")

    # 如果需要将模型推送到Hub上
    if push_to_hub:
        # 打印提示信息
        print("Pushing model to the hub...")

        # 将模型推送到Hub上
        model.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add model",
            use_temp_dir=True,
        )
        # 将处理器推送到Hub上
        processor.push_to_hub(
            repo_id=f"Bearnardd/{pytorch_dump_path}",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个必填参数，指定 EfficientFormer 的 PyTorch 模型路径
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        required=True,
        help="Path to EfficientFormer pytorch checkpoint.",
    )
    # 添加一个必填参数，指定 EfficientFormer 模型配置的 JSON 文件路径
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The json file for EfficientFormer model config.",
    )
    # 添加一个必填参数，指定输出的 PyTorch 模型的保存路径
    parser.add_argument(
        "--pytorch_dump_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model.",
    )

    # 添加一个可选参数，用于指示是否推送模型和图像处理器到 Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Push model and image processor to the hub")
    # 添加一个可选参数，用于指示不推送模型和图像处理器到 Hub
    parser.add_argument(
        "--no-push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not push model and image processor to the hub",
    )
    # 设置默认情况下推送到 Hub
    parser.set_defaults(push_to_hub=True)

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_efficientformer_checkpoint 函数进行模型转换
    convert_efficientformer_checkpoint(
        checkpoint_path=args.pytorch_model_path,
        efficientformer_config_file=args.config_file,
        pytorch_dump_path=args.pytorch_dump_path,
        push_to_hub=args.push_to_hub,
    )
```