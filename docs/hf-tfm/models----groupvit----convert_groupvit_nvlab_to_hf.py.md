# `.\models\groupvit\convert_groupvit_nvlab_to_hf.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，禁止未经许可使用此文件
# 可以在以下链接获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制

"""
从原始存储库转换 GroupViT 检查点。

URL: https://github.com/NVlabs/GroupViT
"""

import argparse

import requests
import torch
from PIL import Image

from transformers import CLIPProcessor, GroupViTConfig, GroupViTModel

# 定义一个函数，用于重命名键名
def rename_key(name):
    # 视觉编码器
    if "img_encoder.pos_embed" in name:
        name = name.replace("img_encoder.pos_embed", "vision_model.embeddings.position_embeddings")
    if "img_encoder.patch_embed.proj" in name:
        name = name.replace("img_encoder.patch_embed.proj", "vision_model.embeddings.patch_embeddings.projection")
    if "img_encoder.patch_embed.norm" in name:
        name = name.replace("img_encoder.patch_embed.norm", "vision_model.embeddings.layernorm")
    if "img_encoder.layers" in name:
        name = name.replace("img_encoder.layers", "vision_model.encoder.stages")
    if "blocks" in name and "res" not in name:
        name = name.replace("blocks", "layers")
    if "attn" in name and "pre_assign" not in name:
        name = name.replace("attn", "self_attn")
    if "proj" in name and "self_attn" in name and "text" not in name:
        name = name.replace("proj", "out_proj")
    if "pre_assign_attn.attn.proj" in name:
        name = name.replace("pre_assign_attn.attn.proj", "pre_assign_attn.attn.out_proj")
    if "norm1" in name:
        name = name.replace("norm1", "layer_norm1")
    if "norm2" in name and "pre_assign" not in name:
        name = name.replace("norm2", "layer_norm2")
    if "img_encoder.norm" in name:
        name = name.replace("img_encoder.norm", "vision_model.layernorm")
    # 文本编码器
    if "text_encoder.token_embedding" in name:
        name = name.replace("text_encoder.token_embedding", "text_model.embeddings.token_embedding")
    if "text_encoder.positional_embedding" in name:
        name = name.replace("text_encoder.positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "text_encoder.transformer.resblocks." in name:
        name = name.replace("text_encoder.transformer.resblocks.", "text_model.encoder.layers.")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    # 如果文件名中包含"text_encoder"，则替换为"text_model"
    if "text_encoder" in name:
        name = name.replace("text_encoder", "text_model")
    # 如果文件名中包含"ln_final"，则替换为"final_layer_norm"
    if "ln_final" in name:
        name = name.replace("ln_final", "final_layer_norm")
    # 处理投影层
    # 如果文件名中包含"img_projector.linear_hidden."，则替换为"visual_projection."
    if "img_projector.linear_hidden." in name:
        name = name.replace("img_projector.linear_hidden.", "visual_projection.")
    # 如果文件名中包含"img_projector.linear_out."，则替换为"visual_projection.3."
    if "img_projector.linear_out." in name:
        name = name.replace("img_projector.linear_out.", "visual_projection.3.")
    # 如果文件名中包含"text_projector.linear_hidden"，则替换为"text_projection"
    if "text_projector.linear_hidden" in name:
        name = name.replace("text_projector.linear_hidden", "text_projection")
    # 如果文件名中包含"text_projector.linear_out"，则替换为"text_projection.3"
    if "text_projector.linear_out" in name:
        name = name.replace("text_projector.linear_out", "text_projection.3")

    # 返回处理后的文件名
    return name
# 将原始状态字典转换为新的状态字典
def convert_state_dict(orig_state_dict, config):
    return orig_state_dict


# 准备一张可爱猫咪的图片用于验证结果
def prepare_img():
    # 图片链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过链接获取图片
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 转换 GroupViT 检查点到 Transformers 设计
@torch.no_grad()
def convert_groupvit_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, model_name="groupvit-gcc-yfcc", push_to_hub=False
):
    """
    复制/粘贴/调整模型的权重以适应 Transformers 设计。
    """
    # 创建 GroupViT 配置
    config = GroupViTConfig()
    # 创建 GroupViT 模型并设置为评估模式
    model = GroupViTModel(config).eval()

    # 加载检查点的状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 转换状态字典
    new_state_dict = convert_state_dict(state_dict, config)
    # 加载新的状态字典到模型中
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    assert missing_keys == ["text_model.embeddings.position_ids"]
    assert (unexpected_keys == ["multi_label_logit_scale"]) or (len(unexpected_keys) == 0)

    # 验证结果
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = prepare_img()
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    if model_name == "groupvit-gcc-yfcc":
        expected_logits = torch.tensor([[13.3523, 6.3629]])
    elif model_name == "groupvit-gcc-redcaps":
        expected_logits = torch.tensor([[16.1873, 8.6230]])
    else:
        raise ValueError(f"Model name {model_name} not supported.")
    assert torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3)

    # 保存处理器和模型
    processor.save_pretrained(pytorch_dump_folder_path)
    model.save_pretrained(pytorch_dump_folder_path)
    print("Successfully saved processor and model to", pytorch_dump_folder_path)

    if push_to_hub:
        print("Pushing to the hub...")
        processor.push_to_hub(model_name, organization="nielsr")
        model.push_to_hub(model_name, organization="nielsr")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to dump the processor and PyTorch model."
    )
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to GroupViT checkpoint")
    parser.add_argument(
        "--model_name",
        default="groupvit-gccy-fcc",
        type=str,
        help="Name of the model. Expecting either 'groupvit-gcc-yfcc' or 'groupvit-gcc-redcaps'",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model and processor to the 🤗 hub using the provided `model_name`.",
    )
    args = parser.parse_args()

    # 调用转换函数
    convert_groupvit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```