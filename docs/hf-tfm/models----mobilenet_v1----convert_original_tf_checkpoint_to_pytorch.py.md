# `.\transformers\models\mobilenet_v1\convert_original_tf_checkpoint_to_pytorch.py`

```
# 设置编码为 UTF-8
# 版权声明标识
# 根据 Apache 许可证版本 2.0 进行许可
# 你可以在遵守许可证的前提下使用该文件
# 你可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则以“原样”分发软件
# 不带任何担保或条件，无论是明示的还是隐含的
# 请查看许可证以获取特定语言的权限和限制
# 从 tensorflow/models 库中转换 MobileNetV1 检查点

import argparse  # 导入参数解析模块
import json  # 导入 JSON 模块
import re  # 导入正则表达式模块
from pathlib import Path  # 导入路径模块

import requests  # 导入 requests 模块
import torch  # 导入 PyTorch 模块
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 模块导入 hf_hub_download 函数
from PIL import Image  # 从 PIL 模块导入 Image 类

from transformers import (  # 从 transformers 模块导入以下类和函数
    MobileNetV1Config, 
    MobileNetV1ForImageClassification, 
    MobileNetV1ImageProcessor,
    load_tf_weights_in_mobilenet_v1,
)
from transformers.utils import logging  # 从 transformers.utils 模块导入 logging

logging.set_verbosity_info()  # 设置日志级别为信息
logger = logging.get_logger(__name__)  # 获取日志记录器

# 获取 MobileNetV1 配置
def get_mobilenet_v1_config(model_name):
    config = MobileNetV1Config(layer_norm_eps=0.001)  # 初始化 MobileNetV1Config 对象

    if "_quant" in model_name:  # 如果模型名中包含 "_quant"
        raise ValueError("Quantized models are not supported.")  # 抛出错误信息

    # 匹配模型名称，提取深度乘子和图像尺寸
    matches = re.match(r"^mobilenet_v1_([^_]*)_([^_]*)$", model_name)
    if matches:
        config.depth_multiplier = float(matches[1])  # 设置深度乘子
        config.image_size = int(matches[2])  # 设置图像尺寸

    # TensorFlow 版本的 MobileNetV1 预测 1001 个类别，第一个类别（索引 0）为 "background"
    config.num_labels = 1001
    filename = "imagenet-1k-id2label.json"  # 文件名称
    repo_id = "huggingface/label-files"
    # 加载标签映射关系文件，并处理为 ID 到标签的映射
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k) + 1: v for k, v in id2label.items()}
    id2label[0] = "background"  # 将索引为 0 的标签设置为 "background"
    config.id2label = id2label  # 设置 ID 到标签的映射关系
    config.label2id = {v: k for k, v in id2label.items()}  # 设置标签到 ID 的映射关系

    return config  # 返回配置对象

# 准备图像
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图�� URL
    im = Image.open(requests.get(url, stream=True).raw)  # 打开网络图像
    return im  # 返回图像对象

@torch.no_grad()  # 禁止梯度计算
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    将模型的权重复制/粘贴/调整为我们的 MobileNetV1 结构。
    """
    config = get_mobilenet_v1_config(model_name)  # 获取 MobileNetV1 配置

    # 加载 🤗 模型
    model = MobileNetV1ForImageClassification(config).eval()  # 加载 MobileNetV1ForImageClassification 模型并设为评估模式

    # 从 TensorFlow 检查点加载权重
    load_tf_weights_in_mobilenet_v1(model, config, checkpoint_path)  # 加载 TensorFlow 检查点中的权重到模型

    # 使用 MobileNetV1ImageProcessor 准备图像，检查输出
    image_processor = MobileNetV1ImageProcessor(
        crop_size={"width": config.image_size, "height": config.image_size},  # 设置裁剪图像尺寸
        size={"shortest_edge": config.image_size + 32},  # 设置图像短边尺寸
    )
    # 对图像进行处理，返回编码后的张量
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 使用编码后的张量输入模型，获取输出
    outputs = model(**encoding)
    # 获取模型的 logits
    logits = outputs.logits
    
    # 确保 logits 的形状为 (1, 1001)
    assert logits.shape == (1, 1001)
    
    # 根据模型名称设置预期的 logits 值
    if model_name == "mobilenet_v1_1.0_224":
        expected_logits = torch.tensor([-4.1739, -1.1233, 3.1205])
    elif model_name == "mobilenet_v1_0.75_192":
        expected_logits = torch.tensor([-3.9440, -2.3141, -0.3333])
    else:
        expected_logits = None
    
    # 如果预期的 logits 存在，则确保模型输出的前3个值与预期值接近
    if expected_logits is not None:
        assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)
    
    # 创建存储模型和图像处理器的目录
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存模型至指定路径
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    # 保存图像处理器至指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)
    
    # 如果选择推送至 Hub
    if push_to_hub:
        print("Pushing to the hub...")
        repo_id = "google/" + model_name
        # 推送图像处理器至 Hub
        image_processor.push_to_hub(repo_id)
        # 推送模型至 Hub
        model.push_to_hub(repo_id)
# 如果当前脚本作为主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="mobilenet_v1_1.0_224",
        type=str,
        help="Name of the MobileNetV1 model you'd like to convert. Should in the form 'mobilenet_v1_<depth>_<size>'.",
    )
    parser.add_argument(
        "--checkpoint_path", required=True, type=str, help="Path to the original TensorFlow checkpoint (.ckpt file)."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 TensorFlow 检查点转换为 PyTorch 模型
    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```