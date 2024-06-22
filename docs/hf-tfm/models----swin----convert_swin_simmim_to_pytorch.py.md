# `.\transformers\models\swin\convert_swin_simmim_to_pytorch.py`

```
# 设置编码格式为 utf-8
# 版权声明，版权归The HuggingFace Inc.团队所有
# 根据Apache许可证2.0获得许可，除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则不得分发许可下的软件
# 根据许可证以“现状”基础分发，没有任何明示或暗示的担保或条件
# 详见许可证的特定语言管理权限和限制
"""从原始存储库中转换Swin SimMIM检查点。

URL：https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md#simmim-pretrained-swin-v1-models"""
# 导入依赖库
import argparse
import requests
import torch
from PIL import Image
from transformers import SwinConfig, SwinForMaskedImageModeling, ViTImageProcessor

# 获取Swin模型的配置
def get_swin_config(model_name):
    config = SwinConfig(image_size=192)

    if "base" in model_name:
        window_size = 6
        embed_dim = 128
        depths = (2, 2, 18, 2)
        num_heads = (4, 8, 16, 32)
    elif "large" in model_name:
        window_size = 12
        embed_dim = 192
        depths = (2, 2, 18, 2)
        num_heads = (6, 12, 24, 48)
    else:
        raise ValueError("Model not supported, only supports base and large variants")

    config.window_size = window_size
    config.embed_dim = embed_dim
    config.depths = depths
    config.num_heads = num_heads

    return config

# 重命名键值
def rename_key(name):
    if "encoder.mask_token" in name:
        name = name.replace("encoder.mask_token", "embeddings.mask_token")
    if "encoder.patch_embed.proj" in name:
        name = name.replace("encoder.patch_embed.proj", "embeddings.patch_embeddings.projection")
    if "encoder.patch_embed.norm" in name:
        name = name.replace("encoder.patch_embed.norm", "embeddings.norm")
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")

    if name == "encoder.norm.weight":
        name = "layernorm.weight"
    if name == "encoder.norm.bias":
        name = "layernorm.bias"

    if "decoder" in name:
        pass
    else:
        name = "swin." + name

    return name

# 转换状态字典
def convert_state_dict(orig_state_dict, model):
    # 遍历原始状态字典的键（key）
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值（val），同时从原始状态字典中删除该键值对
        val = orig_state_dict.pop(key)
    
        # 检查键名中是否包含"attn_mask"
        if "attn_mask" in key:
            # 如果包含"attn_mask"，则跳过不做处理
            pass
        # 检查键名中是否包含"qkv"
        elif "qkv" in key:
            # 拆分键名以获取层号和块号
            key_split = key.split(".")
            # 解析层号
            layer_num = int(key_split[2])
            # 解析块号
            block_num = int(key_split[4])
            # 获取当前块的注意力头大小
            dim = model.swin.encoder.layers[layer_num].blocks[block_num].attention.self.all_head_size
    
            # 检查键名中是否包含"weight"
            if "weight" in key:
                # 如果包含"weight"，则更新查询、键、值的权重
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.weight"
                ] = val[:dim, :]  # 更新查询权重
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.weight"] = val[
                    dim : dim * 2, :
                ]  # 更新键权重
                orig_state_dict[
                    f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.weight"
                ] = val[-dim:, :]  # 更新值权重
            else:
                # 如果不包含"weight"，则更新查询、键、值的偏置
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.query.bias"] = val[
                    :dim
                ]  # 更新查询偏置
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.key.bias"] = val[
                    dim : dim * 2
                ]  # 更新键偏置
                orig_state_dict[f"swin.encoder.layers.{layer_num}.blocks.{block_num}.attention.self.value.bias"] = val[
                    -dim:
                ]  # 更新值偏置
        else:
            # 如果键名不包含"attn_mask"或"qkv"，则通过指定的函数重命名键后将其重新放入原始状态字典
            orig_state_dict[rename_key(key)] = val
    
    # 返回更新后的原始状态字典
    return orig_state_dict
def convert_swin_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    # 从指定路径加载 PyTorch 模型检查点文件，使用 CPU 进行计算
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    
    # 根据模型名称获取 Swin 模型的配置参数
    config = get_swin_config(model_name)
    # 创建 Swin 模型对象并设置为评估模式
    model = SwinForMaskedImageModeling(config)
    model.eval()
    
    # 将加载的模型状态转换为新的状态字典，以便加载到 Swin 模型中
    new_state_dict = convert_state_dict(state_dict, model)
    model.load_state_dict(new_state_dict)
    
    # 需要处理的图片 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    
    # 创建 ViTImageProcessor 对象，指定图片大小为 192x192
    image_processor = ViTImageProcessor(size={"height": 192, "width": 192})
    # 从指定 URL 获取图片并使用 Image.open 打开，转换为 PyTorch 张量
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = image_processor(images=image, return_tensors="pt")
    
    # 使用模型推理处理输入数据
    with torch.no_grad():
        outputs = model(**inputs).logits
    
    # 打印模型输出的键
    print(outputs.keys())
    print("Looks ok!")
    
    if pytorch_dump_folder_path is not None:
        # 保存模型到指定目录
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        
        # 保存图像处理器到指定目录
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)
    
    if push_to_hub:
        # 推送模型和图像处理器至 🤗 hub
        print(f"Pushing model and image processor for {model_name} to hub")
        model.push_to_hub(f"microsoft/{model_name}")
        image_processor.push_to_hub(f"microsoft/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    # 模型名称参数，默认为 "swin-base-simmim-window6-192"
    parser.add_argument(
        "--model_name",
        default="swin-base-simmim-window6-192",
        type=str,
        choices=["swin-base-simmim-window6-192", "swin-large-simmim-window12-192"],
        help="Name of the Swin SimMIM model you'd like to convert.",
    )
    # 检查点路径参数，默认为指定的人员文件路径
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/SwinSimMIM/simmim_pretrain__swin_base__img192_window6__100ep.pth",
        type=str,
        help="Path to the original PyTorch checkpoint (.pth file).",
    )
    # PyTorch 模型输出目录路径参数，默认为 None
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 是否将模型推送至 🤗 hub 参数，默认为 False
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用 convert_swin_checkpoint 函数，根据传入的参数进行模型转换操作
    convert_swin_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
```