# `.\transformers\models\mobilenet_v2\convert_original_tf_checkpoint_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，告知版权归作者团队所有，遵循 Apache License 2.0
# 包含有关许可的信息，可以在 http://www.apache.org/licenses/LICENSE-2.0 获取
# 根据适用法律或书面同意，根据许可发布的软件基础上分发，以"原样"方式分发，不提供任何保修或条件，明示或默示
# 请查看许可，了解特定语言执行权限和限制

"""Convert MobileNetV2 checkpoints from the tensorflow/models library."""

导入所需模块
import argparse
import json
import re
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    MobileNetV2Config,
    MobileNetV2ForImageClassification,
    MobileNetV2ForSemanticSegmentation,
    MobileNetV2ImageProcessor,
    load_tf_weights_in_mobilenet_v2,
)
from transformers.utils import logging

# 设置日志输出级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 根据模型名称获取 MobileNetV2 配置信息
def get_mobilenet_v2_config(model_name):
    # 创建 MobileNetV2 配置对象，指定层归一化 epsilon 值
    config = MobileNetV2Config(layer_norm_eps=0.001)
    
    # 如果模型名中包含 "quant"，则不支持量化模型，抛出异常
    if "quant" in model_name:
        raise ValueError("Quantized models are not supported.")
    
    # 使用正则表达式从模型名称中匹配深度乘数和图像尺寸信息
    matches = re.match(r"^.*mobilenet_v2_([^_]*)_([^_]*)$", model_name)
    if matches:
        config.depth_multiplier = float(matches[1])
        config.image_size = int(matches[2])
    
    # 如果模型名以 "deeplabv3_" 开头
    if model_name.startswith("deeplabv3_"):
        config.output_stride = 8
        config.num_labels = 21
        filename = "pascal-voc-id2label.json"
    else:
        # TensorFlow 版本的 MobileNetV2 预测 1001 个类别，而不是通常的 1000 个
        config.num_labels = 1001
        filename = "imagenet-1k-id2label.json"
    
    # 从 Hugging Face 模型中下载 ID 到标签的映射文件
    repo_id = "huggingface/label-files"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    
    if config.num_labels == 1001:
        # 将 ID 映射表中的键加一，以匹配 PyTorch 中类别索引
        id2label = {int(k) + 1: v for k, v in id2label.items()}
        id2label[0] = "background"
    else:
        id2label = {int(k): v for k, v in id2label.items()}
    
    # 配置对象设置 ID 到标签及标签到 ID 的映射关系
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    
    return config

# 准备用于验证结果的猫图片
def prepare_img():
    # 下载一张可爱猫的图片
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

@torch.no_grad()
def convert_movilevit_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our MobileNetV2 structure.
    """
    # 获取 MobileNetV2 配置
    config = get_mobilenet_v2_config(model_name)

    # 加载 MobileNetV2 模型
    if model_name.startswith("deeplabv3_"):
        model = MobileNetV2ForSemanticSegmentation(config).eval()
    else:
        # 如果模型名称没有包含 MobileNetV2ForImageClassification，就创建一个 MobileNetV2ForImageClassification 对象并设置为推理模式
        model = MobileNetV2ForImageClassification(config).eval()

    # 从 TensorFlow 预训练模型中加载权重
    load_tf_weights_in_mobilenet_v2(model, config, checkpoint_path)

    # 创建一个 MobileNetV2ImageProcessor 对象，对图像进行预处理
    image_processor = MobileNetV2ImageProcessor(
        crop_size={"width": config.image_size, "height": config.image_size},
        size={"shortest_edge": config.image_size + 32},
    )
    # 对预处理后的图像进行编码
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 将编码后的图像输入模型得到输出
    outputs = model(**encoding)
    # 获取模型的分类结果
    logits = outputs.logits

    if model_name.startswith("deeplabv3_"):
        # 如果模型名称以 "deeplabv3_" 开头，则做以下断言判断
        assert logits.shape == (1, 21, 65, 65)

        if model_name == "deeplabv3_mobilenet_v2_1.0_513":
            # 如果模型名称是 "deeplabv3_mobilenet_v2_1.0_513"，设置期望的分类结果 logits
            expected_logits = torch.tensor(
                [
                    [[17.5790, 17.7581, 18.3355], [18.3257, 18.4230, 18.8973], [18.6169, 18.8650, 19.2187]],
                    [[-2.1595, -2.0977, -2.3741], [-2.4226, -2.3028, -2.6835], [-2.7819, -2.5991, -2.7706]],
                    [[4.2058,  4.8317,  4.7638], [4.4136,  5.0361,  4.9383], [4.5028,  4.9644,  4.8734]],
                ]
            )

        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # 断言模型输出的分类结果与期望的分类结果接近
        assert torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-4)
    else:
        # 如果模型名称不以 "deeplabv3_" 开头，则做以下断言判断
        assert logits.shape == (1, 1001)

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

        if expected_logits is not None:
            # 断言模型输出的分类结果与期望的分类结果接近
            assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    # 创建路径以保存模型和图像处理器
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # 如果需要将模型和图像处理器推送到 hub，则执行以下代码
        print("Pushing to the hub...")
        repo_id = "google/" + model_name
        image_processor.push_to_hub(repo_id)
        model.push_to_hub(repo_id)
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建参数解析对象
    parser = argparse.ArgumentParser()
    # 添加必填参数
    parser.add_argument(
        "--model_name",
        default="mobilenet_v2_1.0_224",
        type=str,
        help="Name of the MobileNetV2 model you'd like to convert. Should in the form 'mobilenet_v2_<depth>_<size>'.",
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
    # 调用convert_movilevit_checkpoint函数，传入解析得到的参数
    convert_movilevit_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```