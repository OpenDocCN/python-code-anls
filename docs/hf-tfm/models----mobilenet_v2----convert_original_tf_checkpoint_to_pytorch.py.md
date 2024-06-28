# `.\models\mobilenet_v2\convert_original_tf_checkpoint_to_pytorch.py`

```py
# 使用 UTF-8 编码声明文件编码方式
# 版权声明及许可信息，使用 Apache License 2.0
# 导入所需模块和库
import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 格式数据的模块
import re  # 导入正则表达式操作的模块
from pathlib import Path  # 导入处理文件和路径的模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载资源
from PIL import Image  # 导入 Python Imaging Library 处理图像的模块

from transformers import (  # 导入 Hugging Face 的 transformers 库中的类和函数
    MobileNetV2Config,  # MobileNetV2 模型的配置类
    MobileNetV2ForImageClassification,  # 用于图像分类任务的 MobileNetV2 模型
    MobileNetV2ForSemanticSegmentation,  # 用于语义分割任务的 MobileNetV2 模型
    MobileNetV2ImageProcessor,  # 处理 MobileNetV2 图像的类
    load_tf_weights_in_mobilenet_v2,  # 加载 TensorFlow 模型权重到 MobileNetV2 的函数
)
from transformers.utils import logging  # 导入 transformers 库的日志模块

# 设置日志输出级别为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器对象
logger = logging.get_logger(__name__)


def get_mobilenet_v2_config(model_name):
    # 创建 MobileNetV2 的配置对象，设置层标准化的 epsilon 值
    config = MobileNetV2Config(layer_norm_eps=0.001)

    # 如果模型名称包含 "quant"，则抛出值错误异常，不支持量化模型
    if "quant" in model_name:
        raise ValueError("Quantized models are not supported.")

    # 使用正则表达式匹配模型名称，提取深度乘数和图像大小信息
    matches = re.match(r"^.*mobilenet_v2_([^_]*)_([^_]*)$", model_name)
    if matches:
        config.depth_multiplier = float(matches[1])  # 设置深度乘数
        config.image_size = int(matches[2])  # 设置图像大小

    # 如果模型名称以 "deeplabv3_" 开头，则配置适用于 DeepLabV3 的特定参数
    if model_name.startswith("deeplabv3_"):
        config.output_stride = 8  # 设置输出步幅为 8
        config.num_labels = 21  # 设置类别数量为 21
        filename = "pascal-voc-id2label.json"  # 设置类别映射文件名
    else:
        # 对于其他 MobileNetV2 变体，默认设置为预测 1001 个类别（背景 + 1000 类别）
        config.num_labels = 1001  # 设置类别数量为 1001
        filename = "imagenet-1k-id2label.json"  # 设置类别映射文件名

    # 从 Hugging Face Hub 下载类别映射文件到本地，并加载为字典格式
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))

    # 根据类别数量调整类别映射字典
    if config.num_labels == 1001:
        id2label = {int(k) + 1: v for k, v in id2label.items()}
        id2label[0] = "background"  # 将索引 0 映射为背景类别
    else:
        id2label = {int(k): v for k, v in id2label.items()}

    # 将类别映射字典设置到配置对象中
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}  # 创建反向映射

    return config


# 准备用于测试的图像数据，从 COCO 数据集中下载一张可爱猫咪的图像
# 返回 PIL.Image 对象
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 使用装饰器标记，声明函数不需要进行梯度计算
@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    将模型的权重复制/粘贴/调整到我们的 MobileNetV2 结构中。
    """
    # 获取 MobileNetV2 的配置对象
    config = get_mobilenet_v2_config(model_name)

    # 加载 🤗 模型
    if model_name.startswith("deeplabv3_"):
        model = MobileNetV2ForSemanticSegmentation(config).eval()  # 创建语义分割任务的 MobileNetV2 模型对象
    else:
        # 如果不是从预训练模型加载，则创建一个 MobileNetV2ForImageClassification 实例并设置为评估模式
        model = MobileNetV2ForImageClassification(config).eval()

    # 从 TensorFlow 检查点加载权重到 MobileNetV2 模型
    load_tf_weights_in_mobilenet_v2(model, config, checkpoint_path)

    # 使用 MobileNetV2ImageProcessor 准备图像，设置裁剪大小和最短边大小
    image_processor = MobileNetV2ImageProcessor(
        crop_size={"width": config.image_size, "height": config.image_size},
        size={"shortest_edge": config.image_size + 32},
    )
    # 准备图像并编码
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 将编码后的图像输入模型得到输出
    outputs = model(**encoding)
    # 获取模型输出的 logits
    logits = outputs.logits

    # 如果模型名称以 "deeplabv3_" 开头
    if model_name.startswith("deeplabv3_"):
        # 确保 logits 的形状为 (1, 21, 65, 65)
        assert logits.shape == (1, 21, 65, 65)

        # 如果模型名称为 "deeplabv3_mobilenet_v2_1.0_513"
        if model_name == "deeplabv3_mobilenet_v2_1.0_513":
            # 预期的 logits 值
            expected_logits = torch.tensor(
                [
                    [[17.5790, 17.7581, 18.3355], [18.3257, 18.4230, 18.8973], [18.6169, 18.8650, 19.2187]],
                    [[-2.1595, -2.0977, -2.3741], [-2.4226, -2.3028, -2.6835], [-2.7819, -2.5991, -2.7706]],
                    [[4.2058, 4.8317, 4.7638], [4.4136, 5.0361, 4.9383], [4.5028, 4.9644, 4.8734]],
                ]
            )

        else:
            # 如果模型名称未知，抛出 ValueError 异常
            raise ValueError(f"Unknown model name: {model_name}")

        # 确保 logits 的前 3x3 子张量与预期值非常接近
        assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    else:
        # 如果模型名称不是以 "deeplabv3_" 开头，确保 logits 的形状为 (1, 1001)
        assert logits.shape == (1, 1001)

        # 根据模型名称选择预期的 logits 值
        if model_name == "mobilenet_v2_1.4_224":
            expected_logits = torch.tensor([0.0181, -1.0015, 0.4688])
        elif model_name == "mobilenet_v2_1.0_224":
            expected_logits = torch.tensor([0.2445, -1.1993, 0.1905])
        elif model_name == "mobilenet_v2_0.75_160":
            expected_logits = torch.tensor([0.2482, 0.4136, 0.6669])
        elif model_name == "mobilenet_v2_0.35_96":
            expected_logits = torch.tensor([0.1451, -0.4624, 0.7192])
        else:
            expected_logits = None

        # 如果预期的 logits 值不为 None，则确保 logits 的前 3 个值与预期值非常接近
        if expected_logits is not None:
            assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    # 确保 PyTorch dump 文件夹路径存在，如果不存在则创建
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    # 将模型保存到 PyTorch dump 文件夹路径
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到 PyTorch dump 文件夹路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        print("Pushing to the hub...")
        # 构建 repo_id，并推送 image_processor 和 model 到 Hub
        repo_id = "google/" + model_name
        image_processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
if __name__ == "__main__":
    # 如果这个脚本作为主程序运行，则执行以下代码块

    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="mobilenet_v2_1.0_224",
        type=str,
        help="Name of the MobileNetV2 model you'd like to convert. Should be in the form 'mobilenet_v2_<depth>_<size>'.",
    )

    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the original TensorFlow checkpoint (.ckpt file)."
    )

    # 添加必需的参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to the output PyTorch model directory."
    )

    # 添加可选的参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数来执行 TensorFlow 到 PyTorch 模型的转换
    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```