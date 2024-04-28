# `.\transformers\models\mobilevitv2\convert_mlcvnets_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# Copyright 2023 The HuggingFace Inc. team.
# 根据 Apache 许可证 2.0 版本进行授权，除非符合许可证，否则不得使用该文件，可以在以下链接获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据"原样"提供软件和不提供任何明示或暗示的担保或条件，有关权限原则见许可证
# 详细语言管理权限和权限，参见许可证
"""Convert MobileViTV2 checkpoints from the ml-cvnets library."""

# 导入必要的库
import argparse # 用于命令行解析
import collections # 用于集合数据类型
import json # 用于处理 JSON 数据
from pathlib import Path # 用于操作文件路径

import requests # 用于发送 HTTP 请求
import torch # PyTorch 深度学习库
import yaml # 用于处理 YAML 格式数据
from huggingface_hub import hf_hub_download # 从 Hugging Face hub 下载数据集、模型等资源
from PIL import Image # Python 图像处理库

from transformers import ( # 导入 Transformers 库中的模型和处理器
    MobileViTImageProcessor, # 图像处理器
    MobileViTV2Config, # MobileViTV2 模型配置
    MobileViTV2ForImageClassification, # MobileViTV2 图像分类模型
    MobileViTV2ForSemanticSegmentation, # MobileViTV2 图像语义分割模型
)
from transformers.utils import logging # 导入 Transformers 工具模块中的日志功能

# 设置日志记录级别为 info
logging.set_verbosity_info()
# 获取名为 __name__ 的日志记录器
logger = logging.get_logger(__name__)

# 加载原始配置文件函数
def load_orig_config_file(orig_cfg_file):
    print("Loading config file...")

    # 将嵌套的 YAML 结构数据展平为字典
    def flatten_yaml_as_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.abc.MutableMapping):
                items.extend(flatten_yaml_as_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    # 创建 argparse 命名空间对象
    config = argparse.Namespace()
    with open(orig_cfg_file, "r") as yaml_file:
        try:
            # 从 YAML 文件加载配置数据
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

            # 将展平的配置数据设置到 argparse 命名空间对象中
            flat_cfg = flatten_yaml_as_dict(cfg)
            for k, v in flat_cfg.items():
                setattr(config, k, v)
        except yaml.YAMLError as exc:
            logger.error("Error while loading config file: {}. Error message: {}".format(orig_cfg_file, str(exc)))
    return config

# 获取 MobileViTV2 配置函数
def get_mobilevitv2_config(task_name, orig_cfg_file):
    # 创建一个 MobileViTV2Config 对象
    config = MobileViTV2Config()

    is_segmentation_model = False

    # 根据任务名设置配置参数
    # 图像分类任务 imagemet1k
    if task_name.startswith("imagenet1k_"):
        config.num_labels = 1000
        if int(task_name.strip().split("_")[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-1k-id2label.json"
    # 图像分类任务 imagenet21k
    elif task_name.startswith("imagenet21k_to_1k_"):
        config.num_labels = 21000
        if int(task_name.strip().split("_")[-1]) == 384:
            config.image_size = 384
        else:
            config.image_size = 256
        filename = "imagenet-22k-id2label.json"
    # 图像分割任务 ade20k
    elif task_name.startswith("ade20k_"):
        config.num_labels = 151
        config.image_size = 512
        filename = "ade20k-id2label.json"
        is_segmentation_model = True
    # 如果任务名称以"voc_"开头，设置标签数为21，图像大小为512，文件名为"pascal-voc-id2label.json"，并定义为分割模型
    elif task_name.startswith("voc_"):
        config.num_labels = 21
        config.image_size = 512
        filename = "pascal-voc-id2label.json"
        is_segmentation_model = True

    # 加载原始配置文件
    orig_config = load_orig_config_file(orig_cfg_file)
    # 断言原始配置的分类名称为"mobilevit_v2"，如果不满足则输出"Invalid model"
    assert getattr(orig_config, "model.classification.name", -1) == "mobilevit_v2", "Invalid model"
    # 设置配置的宽度乘数为原始配置的"model.classification.mitv2.width_multiplier"，默认为1.0
    config.width_multiplier = getattr(orig_config, "model.classification.mitv2.width_multiplier", 1.0)
    # 断言原始配置的分类注意力规范化层为"layer_norm_2d"，如果不满足则输出"Norm layers other than layer_norm_2d is not supported"
    assert (
        getattr(orig_config, "model.classification.mitv2.attn_norm_layer", -1) == "layer_norm_2d"
    ), "Norm layers other than layer_norm_2d is not supported"
    # 设置配置的隐藏激活函数为原始配置的"model.classification.activation.name"，默认为"swish"
    config.hidden_act = getattr(orig_config, "model.classification.activation.name", "swish")
    
    # 如果是分割模型
    if is_segmentation_model:
        # 设置输出步幅为原始配置的"model.segmentation.output_stride"，默认为16
        config.output_stride = getattr(orig_config, "model.segmentation.output_stride", 16)
        # 如果任务名包含"_deeplabv3"
        if "_deeplabv3" in task_name:
            # 设置空洞卷积率为原始配置的"model.segmentation.deeplabv3.aspp_rates"，默认为[12, 24, 36]
            config.atrous_rates = getattr(orig_config, "model.segmentation.deeplabv3.aspp_rates", [12, 24, 36])
            # 设置ASPP输出通道数为原始配置的"model.segmentation.deeplabv3.aspp_out_channels"，默认为512
            config.aspp_out_channels = getattr(orig_config, "model.segmentation.deeplabv3.aspp_out_channels", 512)
            # 设置ASPP的丢弃概率为原始配置的"model.segmentation.deeplabv3.aspp_dropout"，默认为0.1
            config.aspp_dropout_prob = getattr(orig_config, "model.segmentation.deeplabv3.aspp_dropout", 0.1)

    # 下载ID到标签文件
    repo_id = "huggingface/label-files"
    # 从Hugging Face Hub下载文件，返回文件路径，文件类型为数据集
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    # 将键值转换为整数型的ID到标签字典
    id2label = {int(k): v for k, v in id2label.items()}
    # 设置配置的ID到���签字典为id2label
    config.id2label = id2label
    # 设置配置的标签到ID字典为标签到ID的反向键值对
    config.label2id = {v: k for k, v in id2label.items()}

    # 返回配置
    return config
# 定义函数以重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将旧键和对应的值以新键重新插入字典
    dct[new] = val


# 创建重命名键列表
def create_rename_keys(state_dict, base_model=False):
    # 如果是基础模型，则模型前缀为空字符串，否则为"mobilevitv2."
    if base_model:
        model_prefix = ""
    else:
        model_prefix = "mobilevitv2."

    # 初始化重命名键列表
    rename_keys = []
    # 返回重命名键列表
    return rename_keys


# 移除未使用的键
def remove_unused_keys(state_dict):
    """remove unused keys (e.g.: seg_head.aux_head)"""
    # 初始化要忽略的键列表
    keys_to_ignore = []
    # 遍历状态字典中的键
    for k in state_dict.keys():
        # 如果键以"seg_head.aux_head."开头，则将其添加到要忽略的键列表中
        if k.startswith("seg_head.aux_head."):
            keys_to_ignore.append(k)
    # 遍历要忽略的键列表
    for k in keys_to_ignore:
        # 从状态字典中弹出要忽略的键
        state_dict.pop(k, None)


# 准备图像，我们将在一张可爱猫咪的图片上验证结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从 URL 获取图像的字节流，并打开图像
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回打开的图像对象
    return im


# 无需梯度计算的装饰器函数
@torch.no_grad()
def convert_mobilevitv2_checkpoint(task_name, checkpoint_path, orig_config_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our MobileViTV2 structure.
    """
    # 获取 MobileViTV2 的配置信息
    config = get_mobilevitv2_config(task_name, orig_config_path)

    # 加载原始的状态字典
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 加载 HuggingFace 模型
    if task_name.startswith("ade20k_") or task_name.startswith("voc_"):
        # 用于语义分割的 MobileViTV2 模型
        model = MobileViTV2ForSemanticSegmentation(config).eval()
        base_model = False
    else:
        # 用于图像分类的 MobileViTV2 模型
        model = MobileViTV2ForImageClassification(config).eval()
        base_model = False

    # 移除并重命名一些键，加载原始模型
    state_dict = checkpoint
    # 移除未使用的键
    remove_unused_keys(state_dict)
    # 创建重命名键列表
    rename_keys = create_rename_keys(state_dict, base_model=base_model)
    # 遍历重命名键列表，对状态字典中的键进行重命名
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)

    # 加载修改后的状态字典
    model.load_state_dict(state_dict)

    # 在 MobileViTImageProcessor 准备的图像上检查输出
    image_processor = MobileViTImageProcessor(crop_size=config.image_size, size=config.image_size + 32)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    outputs = model(**encoding)

    # 验证分类模型
    if task_name.startswith("imagenet"):
        logits = outputs.logits
        # 获取预测的类别索引
        predicted_class_idx = logits.argmax(-1).item()
        # 打印预测的类别
        print("Predicted class:", model.config.id2label[predicted_class_idx])
        # 如果是 imagenet1k_256 的基础变体，并且宽度倍增因子为 1.0
        if task_name.startswith("imagenet1k_256") and config.width_multiplier == 1.0:
            # 基础变体的期望 logits
            expected_logits = torch.tensor([-1.6336e00, -7.3204e-02, -5.1883e-01])
            # 检查 logits 是否接近期望值
            assert torch.allclose(logits[0, :3], expected_logits, atol=1e-4)

    # 创建目录以保存 PyTorch 模型
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的信息
    print(f"Saving model {task_name} to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的信息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器的参数保存到指定路径
    
    
    image_processor.save_pretrained(pytorch_dump_folder_path)
    
    
    这行代码将图像处理器的参数保存到指定路径。其中，`image_processor`是图像处理器对象，`pytorch_dump_folder_path`是保存参数的路径。
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必选参数
    parser.add_argument(
        "--task",
        default="imagenet1k_256",
        type=str,
        help=(
            "Name of the task for which the MobileViTV2 model you'd like to convert is trained on . "
            """
                Classification (ImageNet-1k)
                    - MobileViTV2 (256x256) : imagenet1k_256
                    - MobileViTV2 (Trained on 256x256 and Finetuned on 384x384) : imagenet1k_384
                    - MobileViTV2 (Trained on ImageNet-21k and Finetuned on ImageNet-1k 256x256) :
                      imagenet21k_to_1k_256
                    - MobileViTV2 (Trained on ImageNet-21k, Finetuned on ImageNet-1k 256x256, and Finetuned on
                      ImageNet-1k 384x384) : imagenet21k_to_1k_384
                Segmentation
                    - ADE20K Dataset : ade20k_deeplabv3
                    - Pascal VOC 2012 Dataset: voc_deeplabv3
            """
        ),
        choices=[
            "imagenet1k_256",
            "imagenet1k_384",
            "imagenet21k_to_1k_256",
            "imagenet21k_to_1k_384",
            "ade20k_deeplabv3",
            "voc_deeplabv3",
        ],
    )

    # 添加原始检查点文件路径参数
    parser.add_argument(
        "--orig_checkpoint_path", required=True, type=str, help="Path to the original state dict (.pt file)."
    )
    # 添加原始配置文件路径参数
    parser.add_argument("--orig_config_path", required=True, type=str, help="Path to the original config file.")
    # 添加输出 PyTorch 模型目录路径参数
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model directory."
    )

    # 解析参数
    args = parser.parse_args()
    # 转换 MobileViTV2 检查点文件
    convert_mobilevitv2_checkpoint(
        args.task, args.orig_checkpoint_path, args.orig_config_path, args.pytorch_dump_folder_path
    )
```