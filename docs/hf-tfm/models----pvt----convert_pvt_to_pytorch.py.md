# `.\transformers\models\pvt\convert_pvt_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# 版权声明，作者和团队信息
# 版权告知，依照 Apache License, Version 2.0。未经授权，不得使用此文件。
# 可在 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的一份副本。
# 除非法律要求或书面同意，否则按“原样”分发，没有任何形式的担保或条件，明示或暗示。
# 请参阅许可证以获取特定语言约束和权限的相关内容。

"""从原始库中转换 Pvt 检查点。"""

# 导入必要的库
import argparse
from pathlib import Path
import requests
import torch
from PIL import Image

from transformers import PvtConfig, PvtForImageClassification, PvtImageProcessor
from transformers.utils import logging
# 设置日志级别为 info
logging.set_verbosity_info()

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 函数：创建重命名键列表
def create_rename_keys(config):
    rename_keys = []
    # 重命名 cls token
    rename_keys.extend(
        [
            ("cls_token", "pvt.encoder.patch_embeddings.3.cls_token"),
        ]
    )
    # 重命名 norm 层和分类器层
    rename_keys.extend(
        [
            ("norm.weight", "pvt.encoder.layer_norm.weight"),
            ("norm.bias", "pvt.encoder.layer_norm.bias"),
            ("head.weight", "classifier.weight"),
            ("head.bias", "classifier.bias"),
        ]
    )

    return rename_keys

# 函数：读取与键值相关的权重
def read_in_k_v(state_dict, config):
    # 对于每个编码器块：
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # 读取键和值的权重 + 偏置（在原始实现中是单个矩阵）
            kv_weight = state_dict.pop(f"pvt.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"pvt.encoder.block.{i}.{j}.attention.self.kv.bias")
            # 接下来，将键和值（按顺序）添加到状态字典中
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[: config.hidden_sizes[i], :]
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]

            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"pvt.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[config.hidden_sizes[i] :]


# 函数：重命名键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 准备图像以验证结果
def prepare_img():
    # 定义变量url，存储待下载图片的URL地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests库发送GET请求，返回的response对象取出原始数据流，并给Image.open()方法打开和读取
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回读取的图片对象
    return im
# 使用torch.no_grad()装饰器，确保在此函数中不会进行梯度计算
@torch.no_grad()
def convert_pvt_checkpoint(pvt_size, pvt_checkpoint, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重以适应我们的PVT结构。
    """

    # 定义默认的Pvt配置
    if pvt_size == "tiny":
        config_path = "Zetatech/pvt-tiny-224"
    elif pvt_size == "small":
        config_path = "Zetatech/pvt-small-224"
    elif pvt_size == "medium":
        config_path = "Zetatech/pvt-medium-224"
    elif pvt_size == "large":
        config_path = "Zetatech/pvt-large-224"
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but " f"'{pvt_size}' was given")
    # 根据配置路径创建PvtConfig对象
    config = PvtConfig(name_or_path=config_path)
    # 加载来自https://github.com/whai362/PVT的原始模型
    state_dict = torch.load(pvt_checkpoint, map_location="cpu")

    # 创建重命名键
    rename_keys = create_rename_keys(config)
    # 遍历重命名键并将状态字典中的键重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取状态字典中的键和值
    read_in_k_v(state_dict, config)

    # 加载HuggingFace模型
    model = PvtForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # 在由PVTFeatureExtractor准备的图像上检查输出
    image_processor = PvtImageProcessor(size=config.image_size)
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)
    logits = outputs.logits.detach().cpu()

    # 根据PVT模型大小确定预期的输出
    if pvt_size == "tiny":
        expected_slice_logits = torch.tensor([-1.4192, -1.9158, -0.9702])
    elif pvt_size == "small":
        expected_slice_logits = torch.tensor([0.4353, -0.1960, -0.2373])
    elif pvt_size == "medium":
        expected_slice_logits = torch.tensor([-0.2914, -0.2231, 0.0321])
    elif pvt_size == "large":
        expected_slice_logits = torch.tensor([0.3740, -0.7739, -0.4214])
    else:
        raise ValueError(f"Available model's size: 'tiny', 'small', 'medium', 'large', but " f"'{pvt_size}' was given")

    # 断言模型输出与预期输出在一定误差范围内相等
    assert torch.allclose(logits[0, :3], expected_slice_logits, atol=1e-4)

    # 创建输出PyTorch模型目录
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存模型权重文件
    print(f"Saving model pytorch_model.bin to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    # 保存图像处理器
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--pvt_size",
        default="tiny",
        type=str,
        help="Size of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pvt_checkpoint",
        default="pvt_tiny.pth",
        type=str,
        help="Checkpoint of the PVT pretrained model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    # 调用 convert_pvt_checkpoint 函数，传入参数为 args.pvt_size, args.pvt_checkpoint, args.pytorch_dump_folder_path
    convert_pvt_checkpoint(args.pvt_size, args.pvt_checkpoint, args.pytorch_dump_folder_path)
```