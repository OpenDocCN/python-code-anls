# `.\transformers\models\maskformer\convert_maskformer_swin_to_pytorch.py`

```
# 设置文件编码为 utf-8

# 版权声明，标明代码版权归 HuggingFace Inc. 团队所有，遵循 Apache 2.0 许可证
# 你可以在符合许可证的情况下使用此文件
# 你可以在以下链接获取许可证的副本：
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非法律要求或书面同意，本软件按"原样"分发，
# 没有任何明示或暗示的保证或条件。
# 请参阅许可证以获取具体语言和权限

"""从原始存储库中转换具有 Swin 骨干的 MaskFormer 检查点。URL: https://github.com/facebookresearch/MaskFormer"""

import argparse  # 导入解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import pickle  # 导入处理 pickle 序列化数据的模块
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 库
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载文件
from PIL import Image  # 导入处理图像的模块

from transformers import (  # 导入所需的 Transformers 库模块
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
    SwinConfig,
)
from transformers.utils import logging  # 导入用于日志记录的模块

logging.set_verbosity_info()  # 设置日志记录级别为 INFO
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def get_maskformer_config(model_name: str):
    # 从预训练的 Swin 模型加载骨干配置
    backbone_config = SwinConfig.from_pretrained(
        "microsoft/swin-tiny-patch4-window7-224", out_features=["stage1", "stage2", "stage3", "stage4"]
    )
    # 使用骨干配置创建 MaskFormer 配置
    config = MaskFormerConfig(backbone_config=backbone_config)

    repo_id = "huggingface/label-files"
    if "ade20k-full" in model_name:
        # 针对 ade20k-full 模型，设置类别数为 847
        config.num_labels = 847
        filename = "maskformer-ade20k-full-id2label.json"
    elif "ade" in model_name:
        # 针对 ade 模型，设置类别数为 150
        config.num_labels = 150
        filename = "ade20k-id2label.json"
    elif "coco-stuff" in model_name:
        # 针对 coco-stuff 模型，设置类别数为 171
        config.num_labels = 171
        filename = "maskformer-coco-stuff-id2label.json"
    elif "coco" in model_name:
        # TODO: 针对 coco 模型，设置类别数为 133（待实现）
        config.num_labels = 133
        filename = "coco-panoptic-id2label.json"
    elif "cityscapes" in model_name:
        # 针对 cityscapes 模型，设置类别数为 19
        config.num_labels = 19
        filename = "cityscapes-id2label.json"
    elif "vistas" in model_name:
        # 针对 vistas 模型，设置类别数为 65
        config.num_labels = 65
        filename = "mapillary-vistas-id2label.json"

    # 从 Hugging Face Hub 下载标签文件
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}  # 将标签映射转换为整数键值对

    return config  # 返回 MaskFormer 配置


def create_rename_keys(config):
    rename_keys = []  # 初始化重命名键列表

    # 添加需要重命名的键对
    rename_keys.append(
        (
            "backbone.patch_embed.proj.weight",
            "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.weight",
        )
    )
    rename_keys.append(
        (
            "backbone.patch_embed.proj.bias",
            "model.pixel_level_module.encoder.model.embeddings.patch_embeddings.projection.bias",
        )
    )
    rename_keys.append(
        ("backbone.patch_embed.norm.weight", "model.pixel_level_module.encoder.model.embeddings.norm.weight")
    )

    # fmt: off
    # 添加 backbone.patch_embed.norm.bias 到 model.pixel_level_module.encoder.model.embeddings.norm.bias 的重命名关系
    rename_keys.append(("backbone.patch_embed.norm.bias", "model.pixel_level_module.encoder.model.embeddings.norm.bias"))
    
    # 添加 sem_seg_head.layer_4.weight 到 model.pixel_level_module.decoder.fpn.stem.0.weight 的重命名关系
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    # 添加 sem_seg_head.layer_4.norm.weight 到 model.pixel_level_module.decoder.fpn.stem.1.weight 的重命名关系 
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    # 添加 sem_seg_head.layer_4.norm.bias 到 model.pixel_level_module.decoder.fpn.stem.1.bias 的重命名关系
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))
    
    # 添加从 sem_seg_head.adapter_3/2/1.weight 到 model.pixel_level_module.decoder.fpn.layers.2/1/0.proj.0.weight 的重命名关系
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
    # 添加从 sem_seg_head.adapter_3/2/1.norm.weight 到 model.pixel_level_module.decoder.fpn.layers.2/1/0.proj.1.weight 的重命名关系
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
    # 添加从 sem_seg_head.adapter_3/2/1.norm.bias 到 model.pixel_level_module.decoder.fpn.layers.2/1/0.proj.1.bias 的重命名关系
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
    # 添加从 sem_seg_head.layer_3/2/1.weight 到 model.pixel_level_module.decoder.fpn.layers.2/1/0.block.0.weight 的重命名关系
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
    # 添加从 sem_seg_head.layer_3/2/1.norm.weight 到 model.pixel_level_module.decoder.fpn.layers.2/1/0.block.1.weight 的重命名关系
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
    # 添加从 sem_seg_head.layer_3/2/1.norm.bias 到 model.pixel_level_module.decoder.fpn.layers.2/1/0.block.1.bias 的重命名关系
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))
    
    # 添加 sem_seg_head.mask_features.weight 到 model.pixel_level_module.decoder.mask_projection.weight 的重命名关系
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    # 添加 sem_seg_head.mask_features.bias 到 model.pixel_level_module.decoder.mask_projection.bias 的重命名关系
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    
    # Transformer decoder 部分
    # 遍历解码器层索引范围，用于重命名参数
    for idx in range(config.decoder_config.decoder_layers):
        # 为自注意力机制的输出投影重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        # 为交叉注意力机制的输出投影重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        # 为MLP1重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        # 为MLP2重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        # 为第1个层归一化（自注意力机制）重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        # 为第2个层归一化（交叉注意力机制）重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        # 为第3个层归一化（最终层归一化）重命名参数
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))
    
    # 最后一项，为整个解码器层归一化重命名参数
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    # 将模型参数的键值对添加到 rename_keys 列表中
    # 这些键值对用于将预训练模型的参数重新映射到当前模型的参数上
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))
    # 将 "sem_seg_head.predictor.query_embed.weight" 参数映射到 "model.transformer_module.queries_embedder.weight"
    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))
    # 将 "sem_seg_head.predictor.input_proj.weight" 参数映射到 "model.transformer_module.input_projection.weight"
    rename_keys.append(("sem_seg_head.predictor.input_proj.weight", "model.transformer_module.input_projection.weight"))
    # 将 "sem_seg_head.predictor.input_proj.bias" 参数映射到 "model.transformer_module.input_projection.bias"
    rename_keys.append(("sem_seg_head.predictor.input_proj.bias", "model.transformer_module.input_projection.bias"))
    # 将 "sem_seg_head.predictor.class_embed.weight" 参数映射到 "class_predictor.weight"
    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    # 将 "sem_seg_head.predictor.class_embed.bias" 参数映射到 "class_predictor.bias"
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))
    # 遍历 3 次，将 "sem_seg_head.predictor.mask_embed.layers.i.weight" 参数映射到 "mask_embedder.i.0.weight"
    # 同时将 "sem_seg_head.predictor.mask_embed.layers.i.bias" 参数映射到 "mask_embedder.i.0.bias"
    for i in range(3):
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.weight", f"mask_embedder.{i}.0.weight"))
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.bias", f"mask_embedder.{i}.0.bias"))
    # 返回 rename_keys 列表
    return rename_keys
# 重命名字典 dct 中的键 old 为 new，并保留对应值到变量 val
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 将编码器层每个矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_swin_q_k_v(state_dict, backbone_config):
    # 计算每个特征的数量
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # 读取输入投影层的权重与偏置（在原始实现中，这是单个矩阵加偏置）
            in_proj_weight = state_dict.pop(f"backbone.layers.{i}.blocks.{j}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.layers.{i}.blocks.{j}.attn.qkv.bias")
            # 添加查询、键和值到状态字典
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[dim : dim * 2, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[dim : dim * 2]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[-dim :, :]
            state_dict[f"model.pixel_level_module.encoder.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on


# 将解码器的每个层的矩阵拆分为查询、键和值
def read_in_decoder_q_k_v(state_dict, config):
    # fmt: off
    # 获取解码器的隐藏大小
    hidden_size = config.decoder_config.hidden_size
    # 遍历解码器中的每个层，进行下列操作
    for idx in range(config.decoder_config.decoder_layers):
        # 读取自注意力机制输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_bias")
        # 将查询、键和值（顺序排列）添加到状态字典中
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
        # 读取跨注意力机制输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_bias")
        # 将查询、键和值（顺序排列）添加到状态字典中
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
    # 结束 fmt 格式化
    # fmt: on
# 准备一张可爱猫咪的图片用于验证结果
def prepare_img() -> torch.Tensor:
    # 图片的URL链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests库获取图片并打开成Image对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    复制/粘贴/调整模型的权重到我们的 MaskFormer 结构中。
    """
    # 获取 MaskFormer 模型的配置信息
    config = get_maskformer_config(model_name)

    # 加载原始的 state_dict
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # 重命名键名
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_swin_q_k_v(state_dict, config.backbone_config)
    read_in_decoder_q_k_v(state_dict, config)

    # 更新为 torch 张量
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # 加载 🤗 模型
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    # 打印模型参数的名称和形状
    for name, param in model.named_parameters():
        print(name, param.shape)

    # 加载模型的参数
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == [
        "model.pixel_level_module.encoder.model.layernorm.weight",
        "model.pixel_level_module.encoder.model.layernorm.bias",
    ]
    assert len(unexpected_keys) == 0, f"Unexpected keys: {unexpected_keys}"

    # 验证结果
    image = prepare_img()
    if "vistas" in model_name:
        ignore_index = 65
    elif "cityscapes" in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    reduce_labels = True if "ade" in model_name else False
    image_processor = MaskFormerImageProcessor(ignore_index=ignore_index, reduce_labels=reduce_labels)

    inputs = image_processor(image, return_tensors="pt")

    outputs = model(**inputs)

    print("Logits:", outputs.class_queries_logits[0, :3, :3])

    if model_name == "maskformer-swin-tiny-ade":
        expected_logits = torch.tensor(
            [[3.6353, -4.4770, -2.6065], [0.5081, -4.2394, -3.5343], [2.1909, -5.0353, -1.9323]]
        )
    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and image processor to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing model and image processor to the hub...")
        model.push_to_hub(f"nielsr/{model_name}")
        image_processor.push_to_hub(f"nielsr/{model_name}")


if __name__ == "__main__":
    # 导入 argparse 模块，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加一个参数，用于指定要转换的 MaskFormer 模型的名称，默认为 "maskformer-swin-tiny-ade"
    parser.add_argument(
        "--model_name",
        default="maskformer-swin-tiny-ade",
        type=str,
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    # 添加一个参数，用于指定原始状态字典（.pth 文件）的路径，默认路径为给定的文件路径
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/MaskFormer_checkpoints/MaskFormer-Swin-tiny-ADE20k/model.pkl",
        type=str,
        help="Path to the original state dict (.pth file).",
    )
    # 添加一个参数，用于指定输出 PyTorch 模型目录的路径，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个参数，用于指定是否将转换后的模型推送到 🤗 hub，默认为 False
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数，并将结果存储在 args 变量中
    args = parser.parse_args()
    # 调用 convert_maskformer_checkpoint 函数，传递解析得到的参数
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
```