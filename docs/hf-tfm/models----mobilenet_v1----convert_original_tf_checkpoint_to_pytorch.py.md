# `.\models\mobilenet_v1\convert_original_tf_checkpoint_to_pytorch.py`

```py
# 设置编码格式为 UTF-8
# 版权声明：2022 年由 HuggingFace Inc. 团队拥有
#
# 根据 Apache 许可证 2.0 版本许可使用本文件
# 除非符合许可证的要求，否则不得使用本文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件分发时应以“原样”分发，
# 不附带任何形式的担保或条件。有关许可的详细信息，请参见许可证。
"""从 tensorflow/models 库中转换 MobileNetV1 检查点。"""


import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import re  # 导入正则表达式模块
from pathlib import Path  # 导入处理路径操作的模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数
from PIL import Image  # 导入处理图像的模块

from transformers import (  # 导入 Hugging Face Transformers 库中的相关模块和类
    MobileNetV1Config,
    MobileNetV1ForImageClassification,
    MobileNetV1ImageProcessor,
    load_tf_weights_in_mobilenet_v1,
)
from transformers.utils import logging  # 导入日志记录工具

logging.set_verbosity_info()  # 设置日志输出详细程度为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_mobilenet_v1_config(model_name):
    config = MobileNetV1Config(layer_norm_eps=0.001)  # 创建 MobileNetV1 配置对象，设置层归一化的 epsilon 值

    if "_quant" in model_name:
        raise ValueError("Quantized models are not supported.")  # 如果模型名中包含 "_quant"，则抛出异常

    # 使用正则表达式从模型名称中提取深度乘数和图像大小
    matches = re.match(r"^mobilenet_v1_([^_]*)_([^_]*)$", model_name)
    if matches:
        config.depth_multiplier = float(matches[1])  # 设置配置对象的深度乘数
        config.image_size = int(matches[2])  # 设置配置对象的图像大小

    # TensorFlow 版本的 MobileNetV1 预测 1001 类别而不是通常的 1000 类
    # 第一个类（索引 0）为“背景”
    config.num_labels = 1001  # 设置配置对象的类别数目为 1001
    filename = "imagenet-1k-id2label.json"
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))  # 从 Hub 下载并加载 ID 到标签的映射
    id2label = {int(k) + 1: v for k, v in id2label.items()}  # 调整 ID 映射
    id2label[0] = "background"  # 设置索引 0 的标签为“背景”
    config.id2label = id2label  # 设置配置对象的 ID 到标签的映射
    config.label2id = {v: k for k, v in id2label.items()}  # 设置配置对象的标签到 ID 的映射

    return config  # 返回配置对象


# 我们将在一张可爱猫咪的图像上验证我们的结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图像 URL
    im = Image.open(requests.get(url, stream=True).raw)  # 通过 HTTP 请求打开图像，并获取图像对象
    return im  # 返回图像对象


@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileNetV1 structure.
    将模型的权重复制/粘贴/调整到我们的 MobileNetV1 结构中。
    """
    config = get_mobilenet_v1_config(model_name)  # 获取 MobileNetV1 的配置

    # 加载 🤗 模型
    model = MobileNetV1ForImageClassification(config).eval()  # 创建 MobileNetV1 图像分类模型，并设置为评估模式

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_mobilenet_v1(model, config, checkpoint_path)  # 将 TensorFlow 检查点中的权重加载到模型中

    # 使用 MobileNetV1ImageProcessor 在图像上检查输出
    image_processor = MobileNetV1ImageProcessor(
        crop_size={"width": config.image_size, "height": config.image_size},  # 设置裁剪后的图像大小
        size={"shortest_edge": config.image_size + 32},  # 设置调整大小后的最短边长度
    )
    # 使用图像处理器处理准备好的图像，返回编码后的张量表示
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 使用模型对编码后的图像进行推理，得到输出
    outputs = model(**encoding)
    # 从输出中获取logits
    logits = outputs.logits

    # 断言logits的形状为(1, 1001)，即1个样本，1001个类别的预测值
    assert logits.shape == (1, 1001)

    # 根据模型名称选择预期的logits值
    if model_name == "mobilenet_v1_1.0_224":
        expected_logits = torch.tensor([-4.1739, -1.1233, 3.1205])
    elif model_name == "mobilenet_v1_0.75_192":
        expected_logits = torch.tensor([-3.9440, -2.3141, -0.3333])
    else:
        expected_logits = None

    # 如果预期的logits不为None，则断言模型输出的前三个类别的logits与预期值在给定的误差范围内相似
    if expected_logits is not None:
        assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    # 创建目录用于保存PyTorch模型和图像处理器
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的信息，包括模型名称和保存路径
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的信息，包括保存路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到Hub
    if push_to_hub:
        # 打印推送信息
        print("Pushing to the hub...")
        # 组合模型名称为库的ID
        repo_id = "google/" + model_name
        # 推送图像处理器到Hub
        image_processor.push_to_hub(repo_id)
        # 推送模型到Hub
        model.push_to_hub(repo_id)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="mobilenet_v1_1.0_224",
        type=str,
        help="Name of the MobileNetV1 model you'd like to convert. Should in the form 'mobilenet_v1_<depth>_<size>'."
    )
    # 添加一个必需的参数：模型名称，默认为"mobilenet_v1_1.0_224"

    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original TensorFlow checkpoint (.ckpt file)."
    )
    # 添加一个必需的参数：原始 TensorFlow checkpoint 文件的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个必需的参数：输出 PyTorch 模型的目录路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加一个参数：是否将转换后的模型推送到 🤗 hub

    args = parser.parse_args()
    # 解析命令行参数并将其存储在 args 变量中

    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
    # 调用转换函数，传入命令行参数中的模型名称、checkpoint路径、PyTorch输出路径和推送标志
```