# `.\transformers\models\poolformer\convert_poolformer_original_to_pytorch.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""从原始存储库转换 PoolFormer 检查点。URL: https://github.com/sail-sg/poolformer"""

# 导入所需的库
import argparse
import json
from collections import OrderedDict
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# 导入 Transformers 库中的相关模块
from transformers import PoolFormerConfig, PoolFormerForImageClassification, PoolFormerImageProcessor
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger 对象
logger = logging.get_logger(__name__)

# 定义函数，用于替换键名并减去偏移量
def replace_key_with_offset(key, offset, original_name, new_name):
    """
    Replaces the key by subtracting the offset from the original layer number
    """
    # 获取要查找的字符串
    to_find = original_name.split(".")[0]
    key_list = key.split(".")
    # 获取原始块号和层号
    orig_block_num = int(key_list[key_list.index(to_find) - 2])
    layer_num = int(key_list[key_list.index(to_find) - 1])
    # 计算新的块号
    new_block_num = orig_block_num - offset

    # 替换键名中的原始块号和新块号
    key = key.replace(f"{orig_block_num}.{layer_num}.{original_name}", f"block.{new_block_num}.{layer_num}.{new_name}")
    return key

# 定义函数，用于重命名键名
def rename_keys(state_dict):
    # 创建一个有序字典对象
    new_state_dict = OrderedDict()
    # 初始化变量
    total_embed_found, patch_emb_offset = 0, 0
    # 遍历状态字典中的键值对
    for key, value in state_dict.items():
        # 如果键以"network"开头，则替换为"poolformer.encoder"
        if key.startswith("network"):
            key = key.replace("network", "poolformer.encoder")
        # 如果键中包含"proj"
        if "proj" in key:
            # 对于第一个嵌入和内部嵌入层都适用
            if key.endswith("bias") and "patch_embed" not in key:
                # 如果键以"bias"结尾且不包含"patch_embed"，则增加patch_emb_offset
                patch_emb_offset += 1
            # 找到"proj"之前的部分
            to_replace = key[: key.find("proj")]
            # 替换为"patch_embeddings.{total_embed_found}."
            key = key.replace(to_replace, f"patch_embeddings.{total_embed_found}.")
            # 将"proj"替换为"projection"
            key = key.replace("proj", "projection")
            # 如果键以"bias"结尾
            if key.endswith("bias"):
                # 增加total_embed_found
                total_embed_found += 1
        # 如果键中包含"patch_embeddings"
        if "patch_embeddings" in key:
            # 在键前面加上"poolformer.encoder."
            key = "poolformer.encoder." + key
        # 如果键中包含"mlp.fc1"
        if "mlp.fc1" in key:
            # 使用replace_key_with_offset函数替换键
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc1", "output.conv1")
        # 如果键中包含"mlp.fc2"
        if "mlp.fc2" in key:
            # 使用replace_key_with_offset函数替换键
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc2", "output.conv2")
        # 如果键中包含"norm1"
        if "norm1" in key:
            # 使用replace_key_with_offset函数替换键
            key = replace_key_with_offset(key, patch_emb_offset, "norm1", "before_norm")
        # 如果键中包含"norm2"
        if "norm2" in key:
            # 使用replace_key_with_offset函数替换键
            key = replace_key_with_offset(key, patch_emb_offset, "norm2", "after_norm")
        # 如果键中包含"layer_scale_1"
        if "layer_scale_1" in key:
            # 使用replace_key_with_offset函数替换键
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale_1", "layer_scale_1")
        # 如果键中包含"layer_scale_2"
        if "layer_scale_2" in key:
            # 使用replace_key_with_offset函数替换键
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale_2", "layer_scale_2")
        # 如果键中包含"head"
        if "head" in key:
            # 将"head"替换为"classifier"
            key = key.replace("head", "classifier")
        # 将新的键值对添加到new_state_dict中
        new_state_dict[key] = value
    # 返回新的状态字典
    return new_state_dict
# 准备一个 COCO 图像用于验证结果
def prepare_img():
    # 定义图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 URL 获取图像的字节流并打开为图像对象
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_poolformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重到我们的 PoolFormer 结构。
    """

    # 加载默认的 PoolFormer 配置
    config = PoolFormerConfig()

    # 根据 model_name 设置属性
    repo_id = "huggingface/label-files"
    size = model_name[-3:]
    config.num_labels = 1000
    filename = "imagenet-1k-id2label.json"
    expected_shape = (1, 1000)

    # 设置配置属性
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == "s12":
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pct = 0.9
    elif size == "s24":
        config.depths = [4, 4, 12, 4]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pct = 0.9
    elif size == "s36":
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pct = 0.9
    elif size == "m36":
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pct = 0.95
    elif size == "m48":
        config.depths = [8, 8, 24, 8]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pct = 0.95
    else:
        raise ValueError(f"Size {size} not supported")

    # 加载图像处理器
    image_processor = PoolFormerImageProcessor(crop_pct=crop_pct)

    # 准备图像
    image = prepare_img()
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    logger.info(f"Converting model {model_name}...")

    # 加载原始状态字典
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # 重命名键
    state_dict = rename_keys(state_dict)

    # 创建 HuggingFace 模型并加载状态���典
    model = PoolFormerForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 定义图像处理器
    image_processor = PoolFormerImageProcessor(crop_pct=crop_pct)
    pixel_values = image_processor(images=prepare_img(), return_tensors="pt").pixel_values

    # 前向传播
    outputs = model(pixel_values)
    logits = outputs.logits

    # 为不同模型定义预期的 logit 切片
    # 如果 size 为 "s12"，则设置期望的切片为指定的值
    if size == "s12":
        expected_slice = torch.tensor([-0.3045, -0.6758, -0.4869])
    # 如果 size 为 "s24"，则设置期望的切片为指定的值
    elif size == "s24":
        expected_slice = torch.tensor([0.4402, -0.1374, -0.8045])
    # 如果 size 为 "s36"，则设置期望的切片为指定的值
    elif size == "s36":
        expected_slice = torch.tensor([-0.6080, -0.5133, -0.5898])
    # 如果 size 为 "m36"，则设置期望的切片为指定的值
    elif size == "m36":
        expected_slice = torch.tensor([0.3952, 0.2263, -1.2668])
    # 如果 size 为 "m48"，则设置期望的切片为指定的值
    elif size == "m48":
        expected_slice = torch.tensor([0.1167, -0.0656, -0.3423])
    # 如果 size 不在以上指定的值中，则抛出异常
    else:
        raise ValueError(f"Size {size} not supported")

    # 验证 logits 的形状是否与期望的形状相同
    assert logits.shape == expected_shape
    # 验证 logits 的前三个元素是否与期望的切片值非常接近，允许误差为 1e-2
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-2)

    # 最后，保存模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    # 创建保存模型和图像处理器的文件夹
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加一个参数，指定模型名称，默认为"poolformer_s12"，类型为字符串，提供帮助信息
    parser.add_argument(
        "--model_name",
        default="poolformer_s12",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    # 添加一个参数，指定原始 PyTorch 检查点文件的路径，默认为None，类型为字符串，提供帮助信息
    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    # 添加一个参数，指定输出 PyTorch 模型的文件夹路径，默认为None，类型为字符串，提供帮助信息
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 解析命令行参数，并将其存储在args对象中
    args = parser.parse_args()
    # 调用函数，将指定模型的原始 PyTorch 检查点文件转换为输出文件夹中的 PyTorch 模型
    convert_poolformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
```