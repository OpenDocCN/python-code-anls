# `.\transformers\models\swiftformer\convert_swiftformer_original_to_hf.py`

```py
# 设置脚本文件编码格式为 UTF-8
# 版权声明及使用许可说明
# 通过结合执行原始代码以及遵守许可协议规定使用该文件。
# 在需要的情况下，可以获取许可协议的副本

# 引入必要的模块和库
import argparse  # 用于解析命令行参数
import json  # 用于 JSON 数据的编码和解码
from pathlib import Path  # 提供了用于处理文件和目录的类
import requests  # 用于发送 HTTP 请求
import torch  # 用于 PyTorch 相关操作
from huggingface_hub import hf_hub_download  # 从 Hugging Face 模型存储库下载模型
from PIL import Image  # Python 图像库，用于打开，操作和显示图像

# 引入 Hugging Face 提供的相关模块
from transformers import (
    SwiftFormerConfig,  # SwiftFormer 模型的配置
    SwiftFormerForImageClassification,  # 用于图像分类任务的 SwiftFormer 模型
    ViTImageProcessor,  # Vision Transformer 图像处理器
)

# 引入日志记录模块
from transformers.utils import logging

# 设置日志信息输出级别为 info
logging.set_verbosity_info()
# 获取日志记录对象
logger = logging.get_logger(__name__)

# 设备设置为 CPU
device = torch.device("cpu")


# 准备需要用到的图像
def prepare_img():
    # 通过 URL 打开图像
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从 URL 获取图像数据
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 获取期望的输出结果
def get_expected_output(swiftformer_name):
    # 根据 SwiftFormer 模型的不同名称获取对应的期望输出
    if swiftformer_name == "swiftformer_xs":
        return torch.tensor([-2.1703e00, 2.1107e00, -2.0811e00, 8.8685e-01, 2.4360e-01])

    elif swiftformer_name == "swiftformer_s":
        return torch.tensor([3.9636e-01, 2.3478e-01, -1.6963e00, -1.7381e00, -8.6337e-01])

    elif swiftformer_name == "swiftformer_l1":
        return torch.tensor([-4.2768e-01, -4.7429e-01, -1.0897e00, -1.0248e00, 3.5523e-02])

    elif swiftformer_name == "swiftformer_l3":
        return torch.tensor([-2.5330e-01, 2.4211e-01, -6.0185e-01, -8.2789e-01, -6.0446e-02])


# 重命名字典中的键值对
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 创建要重命名的键值对列表
def create_rename_keys(state_dict):
    rename_keys = []
    for k in state_dict.keys():
        k_new = k
        if ".pwconv" in k:
            k_new = k_new.replace(".pwconv", ".point_wise_conv")
        if ".dwconv" in k:
            k_new = k_new.replace(".dwconv", ".depth_wise_conv")
        if ".Proj." in k:
            k_new = k_new.replace(".Proj.", ".proj.")
        if "patch_embed" in k_new:
            k_new = k_new.replace("patch_embed", "swiftformer.patch_embed.patch_embedding")
        if "network" in k_new:
            ls = k_new.split(".")
            if ls[2].isdigit():
                k_new = "swiftformer.encoder.network." + ls[1] + ".blocks." + ls[2] + "." + ".".join(ls[3:])
            else:
                k_new = k_new.replace("network", "swiftformer.encoder.network")
        rename_keys.append((k, k_new))
    return rename_keys


# 转换 SwiftFormer 模型检查点
@torch.no_grad()
def convert_swiftformer_checkpoint(swiftformer_name, pytorch_dump_folder_path, original_ckpt):
    """ 转换 SwiftFormer 模型的检查点
    Copy/paste/tweak model's weights to our SwiftFormer structure.
    """

    # 定义默认的 SwiftFormer 配置
    config = SwiftFormerConfig()

    # 数据集（仅使用 ImageNet-21k 还是在 ImageNet 2012 上微调），patch_size 和 image_size
    config.num_labels = 1000
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # 架构的大小
    if swiftformer_name == "swiftformer_xs":
        config.depths = [3, 3, 6, 4]
        config.embed_dims = [48, 56, 112, 220]

    elif swiftformer_name == "swiftformer_s":
        config.depths = [3, 3, 9, 6]
        config.embed_dims = [48, 64, 168, 224]

    elif swiftformer_name == "swiftformer_l1":
        config.depths = [4, 3, 10, 5]
        config.embed_dims = [48, 96, 192, 384]

    elif swiftformer_name == "swiftformer_l3":
        config.depths = [4, 4, 12, 6]
        config.embed_dims = [64, 128, 320, 512]

    # 加载原始模型的 state_dict，删除和重命名一些键
    if original_ckpt:
        if original_ckpt.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(original_ckpt, map_location="cpu", check_hash=True)
        else:
            checkpoint = torch.load(original_ckpt, map_location="cpu")
    state_dict = checkpoint

    rename_keys = create_rename_keys(state_dict)
    for rename_key_src, rename_key_dest in rename_keys:
        rename_key(state_dict, rename_key_src, rename_key_dest)

    # 加载 HuggingFace 模型
    hf_model = SwiftFormerForImageClassification(config).eval()
    hf_model.load_state_dict(state_dict)

    # 准备测试输入
    image = prepare_img()
    processor = ViTImageProcessor.from_pretrained("preprocessor_config")
    inputs = processor(images=image, return_tensors="pt")

    # 比较两个模型的输出
    timm_logits = get_expected_output(swiftformer_name)
    hf_logits = hf_model(inputs["pixel_values"]).logits

    assert hf_logits.shape == torch.Size([1, 1000])
    assert torch.allclose(hf_logits[0, 0:5], timm_logits, atol=1e-3)

    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {swiftformer_name} to {pytorch_dump_folder_path}")
    hf_model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下操作
if __name__ == "__main__":
    # 创建 ArgumentParser 对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--swiftformer_name",
        default="swiftformer_xs",
        choices=["swiftformer_xs", "swiftformer_s", "swiftformer_l1", "swiftformer_l3"],
        type=str,
        help="Name of the SwiftFormer model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="./converted_outputs/",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--original_ckpt", default=None, type=str, help="Path to the original model checkpoint.")
    
    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_swiftformer_checkpoint 函数，传入解析后的参数进行 SwiftFormer 模型转换
    convert_swiftformer_checkpoint(args.swiftformer_name, args.pytorch_dump_folder_path, args.original_ckpt)
```