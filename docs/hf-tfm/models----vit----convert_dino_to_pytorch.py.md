# `.\transformers\models\vit\convert_dino_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“原样”分发软件
# 没有任何明示或暗示的担保或条件，包括但不限于特定用途的适用性
# 请查看许可证以获取特定语言的权限和限制
"""将使用 DINO 方法训练的 ViT 检查点转换为模型。"""

# 导入所需库
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor, ViTModel
from transformers.utils import logging

# 设置日志级别为信息
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger(__name__)

# 创建要重命名的所有键列表（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # 编码器层：输出投影、2个前馈神经网络和2个 layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # 投影层 + 位置嵌入
    rename_keys.extend(
        [
            ("cls_token", "vit.embeddings.cls_token"),
            ("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "vit.embeddings.position_embeddings"),
        ]
    )
    # 如果存在基础模型
    if base_model:
        # 添加 layernorm 和 pooler 相关的键值对到重命名列表中
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )

        # 如果只有基础模型，应该将所有以"vit"开头的键中的"vit"去除
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 添加 layernorm 和分类头相关的键值对到重命名列表中
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    # 返回重命名后的键值对列表
    return rename_keys
# 将每个编码器层的矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每个隐藏层
    for i in range(config.num_hidden_layers):
        # 如果是基础模型，则前缀为空字符串
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # 读取输入投影层的权重和偏置（在timm中，这是一个矩阵加偏置）
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 接下来，按顺序将查询（query）、键（key）和值（value）添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 移除分类头部的权重和偏置
def remove_classification_head_(state_dict):
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


# 重命名字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 在一张可爱猫咪的图片上准备进行验证
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 在没有梯度的情况下进行转换ViT检查点
@torch.no_grad()
def convert_vit_checkpoint(model_name, pytorch_dump_folder_path, base_model=True):
    """
    复制/粘贴/调整模型的权重到我们的ViT结构中。
    """

    # 定义默认的ViT配置
    config = ViTConfig()
    # 如果模型名称以"8"结尾，则设置patch_size为8
    if model_name[-1] == "8":
        config.patch_size = 8
    # 如果需要，设置标签
    if not base_model:
        config.num_labels = 1000
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    # 设置架构的大小
    if model_name in ["dino_vits8", "dino_vits16"]:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6

    # 从torch hub加载原始模型
    original_model = torch.hub.load("facebookresearch/dino:main", model_name)
    # 将原始模型设置为评估模式
    original_model.eval()

    # 加载原始模型的状态字典，并移除/重命名一些键
    state_dict = original_model.state_dict()
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model=base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # 加载 HuggingFace 模型
    if base_model:
        model = ViTModel(config, add_pooling_layer=False).eval()
    else:
        model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # 在由 ViTImageProcessor 准备的图像上检查输出
    image_processor = ViTImageProcessor()
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    if base_model:
        final_hidden_state_cls_token = original_model(pixel_values)
        assert torch.allclose(final_hidden_state_cls_token, outputs.last_hidden_state[:, 0, :], atol=1e-1)
    else:
        logits = original_model(pixel_values)
        assert logits.shape == outputs.logits.shape
        assert torch.allclose(logits, outputs.logits, atol=1e-3)

    # 创建目录以保存模型和图像处理器
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被作为主程序执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name",
        default="dino_vitb16",
        type=str,
        help="Name of the model trained with DINO you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--base_model",
        action="store_true",
        help="Whether to only convert the base model (no projection head weights).",
    )

    # 设置默认参数
    parser.set_defaults(base_model=True)
    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数将 ViT 模型转换为 PyTorch 模型
    convert_vit_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.base_model)
```