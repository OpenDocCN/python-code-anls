# `.\transformers\models\bit\convert_bit_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可证信息
# 版权所有 2022 年由 HuggingFace Inc. 团队保留
#
# 根据 Apache 许可证版本 2.0（"许可证"）授权;
# 除非符合许可证规定，否则您不得使用此文件。
# 您可以在以下网址获得许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据"原样"提供，不提供任何形式的担保
# 或者暗示。有关许可证的特定语言
# 请参阅许可证。

# 导入必要的库
import argparse  # 导入参数解析器
import json  # 导入 JSON 模块，用于处理 JSON 数据
from pathlib import Path  # 导入 Path 对象，用于处理文件路径

import requests  # 导入 requests 库，用于 HTTP 请求
import torch  # 导入 PyTorch 框架
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 库导入 hf_hub_download 函数
from PIL import Image  # 导入 PIL 库中的 Image 模块，用于图像处理
from timm import create_model  # 从 timm 库导入 create_model 函数，用于创建模型
from timm.data import resolve_data_config  # 从 timm 库导入 resolve_data_config 函数，用于数据配置
from timm.data.transforms_factory import create_transform  # 从 timm 库导入 create_transform 函数，用于创建转换

from transformers import BitConfig, BitForImageClassification, BitImageProcessor  # 导入 transformers 模块中的类和函数
from transformers.image_utils import PILImageResampling  # 导入 transformers 模块中的 PILImageResampling 类，用于图像重采样
from transformers.utils import logging  # 导入 transformers 模块中的 logging 模块，用于日志记录


# 设置日志记录的详细程度为 INFO
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


# 获取 BiT 模型的配置信息
def get_config(model_name):
    # 设置标签文件的存储库 ID 和文件名
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    # 加载图像标签数据
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 转换标签 ID 为整数
    label2id = {v: k for k, v in id2label.items()}  # 创建标签到 ID 的映射表

    # 根据模型名确定是否需要使用标准卷积层
    conv_layer = "std_conv" if "bit" in model_name else False

    # 配置 BiT 模型
    config = BitConfig(
        conv_layer=conv_layer,  # 卷积层类型
        num_labels=1000,  # 标签数量
        id2label=id2label,  # ID 到标签的映射
        label2id=label2id,  # 标签到 ID 的映射
    )

    return config


# 重命名模型参数的键名
def rename_key(name):
    if "stem.conv" in name:
        name = name.replace("stem.conv", "bit.embedder.convolution")
    if "blocks" in name:
        name = name.replace("blocks", "layers")
    if "head.fc" in name:
        name = name.replace("head.fc", "classifier.1")
    if name.startswith("norm"):
        name = "bit." + name
    if "bit" not in name and "classifier" not in name:
        name = "bit.encoder." + name

    return name


# 准备图像数据，用于模型验证
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 图像 URL
    im = Image.open(requests.get(url, stream=True).raw)  # 使用 requests 获取图像
    return im


# 转换 BiT 模型的检查点
@torch.no_grad()
def convert_bit_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    复制/粘贴/调整模型权重到我们的 BiT 结构。
    """

    # 获取默认的 BiT 配置
    config = get_config(model_name)

    # 从 timm 加载原始模型
    timm_model = create_model(model_name, pretrained=True)  # 创建 timm 预训练模型
    timm_model.eval()  # 设置为评估模式
    # 加载原始模型的状态字典
    state_dict = timm_model.state_dict()
    # 遍历状态字典的键（拷贝的原因是在遍历时需要修改字典）
    for key in state_dict.copy().keys():
        # 弹出当前键对应的值
        val = state_dict.pop(key)
        # 根据键重命名后的值重新设置状态字典
        state_dict[rename_key(key)] = val.squeeze() if "head" in key else val

    # 加载 HuggingFace 模型
    model = BitForImageClassification(config)
    # 设置模型为评估模式
    model.eval()
    # 加载状态字典到模型
    model.load_state_dict(state_dict)

    # 创建图像处理器
    transform = create_transform(**resolve_data_config({}, model=timm_model))
    # 获取 timm_transforms 的引用
    timm_transforms = transform.transforms

    # 定义 Pillow 重采样方法的映射
    pillow_resamplings = {
        "bilinear": PILImageResampling.BILINEAR,
        "bicubic": PILImageResampling.BICUBIC,
        "nearest": PILImageResampling.NEAREST,
    }

    # 创建 BitImageProcessor 实例
    processor = BitImageProcessor(
        do_resize=True,  # 是否进行调整大小
        size={"shortest_edge": timm_transforms[0].size},  # 设置图像最短边的大小
        resample=pillow_resamplings[timm_transforms[0].interpolation.value],  # 设置重采样方法
        do_center_crop=True,  # 是否进行中心裁剪
        crop_size={"height": timm_transforms[1].size[0], "width": timm_transforms[1].size[1]},  # 设置裁剪大小
        do_normalize=True,  # 是否进行归一化
        image_mean=timm_transforms[-1].mean.tolist(),  # 设置图像均值
        image_std=timm_transforms[-1].std.tolist(),  # 设置图像标准差
    )

    # 准备图像
    image = prepare_img()
    # 对图像进行变换，并在第 0 维度上添加一个维度
    timm_pixel_values = transform(image).unsqueeze(0)
    # 使用图像处理器处理图像，返回像素值张量
    pixel_values = processor(image, return_tensors="pt").pixel_values

    # 验证像素值是否一致
    assert torch.allclose(timm_pixel_values, pixel_values)

    # 验证模型输出的逻辑是否一致
    with torch.no_grad():
        # 使用模型得到输出
        outputs = model(pixel_values)
        # 获取模型的逻辑
        logits = outputs.logits

    # 打印前三个逻辑值
    print("Logits:", logits[0, :3])
    # 打印预测的类别
    print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])
    # 使用 timm_model 获取逻辑值
    timm_logits = timm_model(pixel_values)
    # 断言 timm_logits 的形状和模型输出的逻辑形状一致
    assert timm_logits.shape == outputs.logits.shape
    # 断言 timm_logits 和模型输出的逻辑值在一定的误差范围内一致
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)
    # 打印提示信息
    print("Looks ok!")

    # 如果设置了 pytorch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型和处理器的信息
        print(f"Saving model {model_name} and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果设置了 push_to_hub
    if push_to_hub:
        # 打印将模型和处理器推送到 hub 的信息
        print(f"Pushing model {model_name} and processor to the hub")
        # 将模型推送到 hub
        model.push_to_hub(f"ybelkada/{model_name}")
        # 将处理器推送到 hub
        processor.push_to_hub(f"ybelkada/{model_name}")
# 如果当前脚本被直接执行（而不是被导入），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="resnetv2_50x1_bitm",
        type=str,
        help="Name of the BiT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub.",
    )

    # 解析命令行参数并将其存储在 args 变量中
    args = parser.parse_args()
    # 调用函数 convert_bit_checkpoint，并传递解析后的参数
    convert_bit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```