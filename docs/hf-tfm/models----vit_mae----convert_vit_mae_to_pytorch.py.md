# `.\transformers\models\vit_mae\convert_vit_mae_to_pytorch.py`

```
# 定义函数，用于重命名模型参数的键名以匹配当前模型配置
def rename_key(name):
    # 如果键名中包含"cls_token"，替换为"vit.embeddings.cls_token"
    if "cls_token" in name:
        name = name.replace("cls_token", "vit.embeddings.cls_token")
    # 如果键名中包含"mask_token"，替换为"decoder.mask_token"
    if "mask_token" in name:
        name = name.replace("mask_token", "decoder.mask_token")
    # 如果键名中包含"decoder_pos_embed"，替换为"decoder.decoder_pos_embed"
    if "decoder_pos_embed" in name:
        name = name.replace("decoder_pos_embed", "decoder.decoder_pos_embed")
    # 如果键名中包含"pos_embed"但不包含"decoder"，替换为"vit.embeddings.position_embeddings"
    if "pos_embed" in name and "decoder" not in name:
        name = name.replace("pos_embed", "vit.embeddings.position_embeddings")
    # 如果键名中包含"patch_embed.proj"，替换为"vit.embeddings.patch_embeddings.projection"
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "vit.embeddings.patch_embeddings.projection")
    # 如果键名中包含"patch_embed.norm"，替换为"vit.embeddings.norm"
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "vit.embeddings.norm")
    # 如果键名中包含"decoder_blocks"，替换为"decoder.decoder_layers"
    if "decoder_blocks" in name:
        name = name.replace("decoder_blocks", "decoder.decoder_layers")
    # 如果键名中包含"blocks"，替换为"vit.encoder.layer"
    if "blocks" in name:
        name = name.replace("blocks", "vit.encoder.layer")
    # 如果键名中包含"attn.proj"，替换为"attention.output.dense"
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")
    # 如果键名中包含"attn"，替换为"attention.self"
    if "attn" in name:
        name = name.replace("attn", "attention.self")
    # 如果键名中包含"norm1"，替换为"layernorm_before"
    if "norm1" in name:
        name = name.replace("norm1", "layernorm_before")
    # 如果键名中包含"norm2"，替换为"layernorm_after"
    if "norm2" in name:
        name = name.replace("norm2", "layernorm_after")
    # 如果键名中包含"mlp.fc1"，替换为"intermediate.dense"
    if "mlp.fc1" in name:
        name = name.replace("mlp.fc1", "intermediate.dense")
    # 如果键名中包含"mlp.fc2"，替换为"output.dense"
    if "mlp.fc2" in name:
        name = name.replace("mlp.fc2", "output.dense")
    # 如果键名中包含"decoder_embed"，替换为"decoder.decoder_embed"
    if "decoder_embed" in name:
        name = name.replace("decoder_embed", "decoder.decoder_embed")
    # 如果键名中包含"decoder_norm"，替换为"decoder.decoder_norm"
    if "decoder_norm" in name:
        name = name.replace("decoder_norm", "decoder.decoder_norm")
    # 如果键名中包含"decoder_pred"，替换为"decoder.decoder_pred"
    if "decoder_pred" in name:
        name = name.replace("decoder_pred", "decoder.decoder_pred")
    # 如果键名中包含"norm.weight"但不包含"decoder"，替换为"vit.layernorm.weight"
    if "norm.weight" in name and "decoder" not in name:
        name = name.replace("norm.weight", "vit.layernorm.weight")
    # 如果键名中包含"norm.bias"但不包含"decoder"，替换为"vit.layernorm.bias"
    if "norm.bias" in name and "decoder" not in name:
        name = name.replace("norm.bias", "vit.layernorm.bias")

    # 返回修改后的键名
    return name


# 定义函数，用于将模型参数的键名进行转换以匹配当前模型配置
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的拷贝的键
    for key in orig_state_dict.copy().keys():
        # 从原始状态字典中弹出指定键的值
        val = orig_state_dict.pop(key)

        # 如果键名包含"qkv"
        if "qkv" in key:
            # 将键名按"."分割成列表
            key_split = key.split(".")
            # 从列表中获取层数
            layer_num = int(key_split[1])

            # 如果键名包含"decoder_blocks"
            if "decoder_blocks" in key:
                # 设置维度为config.decoder_hidden_size
                dim = config.decoder_hidden_size
                # 设置前缀为"decoder.decoder_layers."
                prefix = "decoder.decoder_layers."
                # 如果键名包含"weight"
                if "weight" in key:
                    # 根据指定规则重命名键名，并赋予对应的值
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
                # 如果键名包含"bias"
                elif "bias" in key:
                    # 根据指定规则重命名键名，并赋予对应的值
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.bias"] = val[-dim:]
            else:
                # 设置维度为config.hidden_size
                dim = config.hidden_size
                # 设置前缀为"vit.encoder.layer."
                prefix = "vit.encoder.layer."
                # 如果键名包含"weight"
                if "weight" in key:
                    # 根据指定规则重命名键名，并赋予对应的值
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
                # 如果键名包含"bias"
                elif "bias" in key:
                    # 根据指定规则重命名键名，并赋予对应的值
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.bias"] = val[:dim]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.bias"] = val[dim : dim * 2]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.bias"] = val[-dim:]

        # 如果键名不包含"qkv"
        else:
            # 根据指定规则重命名键名，并赋予对应的值
            orig_state_dict[rename_key(key)] = val

    # 返回修改后的原始状态字典
    return orig_state_dict
def convert_vit_mae_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    # 创建ViTMAEConfig对象作为配置
    config = ViTMAEConfig()
    if "large" in checkpoint_url:
        # 配置较大模型的参数
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "huge" in checkpoint_url:
        # 配置巨大模型的参数
        config.patch_size = 14
        config.hidden_size = 1280
        config.intermediate_size = 5120
        config.num_hidden_layers = 32
        config.num_attention_heads = 16

    # 创建ViTMAEForPreTraining模型
    model = ViTMAEForPreTraining(config)

    # 从指定URL加载模型参数
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # 创建ViTMAEImageProcessor对象
    image_processor = ViTMAEImageProcessor(size=config.image_size)

    # 转换模型参数
    new_state_dict = convert_state_dict(state_dict, config)

    # 加载转换后的模型参数
    model.load_state_dict(new_state_dict)
    model.eval()

    url = "https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg"

    # 从URL下载并打开图像
    image = Image.open(requests.get(url, stream=True).raw)
    image_processor = ViTMAEImageProcessor(size=config.image_size)
    inputs = image_processor(images=image, return_tensors="pt")

    # 前向传播
    torch.manual_seed(2)
    outputs = model(**inputs)
    logits = outputs.logits

    if "large" in checkpoint_url:
        # 定义预期的输出片段（large模型）
        expected_slice = torch.tensor(
            [[-0.7309, -0.7128, -1.0169], [-1.0161, -0.9058, -1.1878], [-1.0478, -0.9411, -1.1911]]
        )
    elif "huge" in checkpoint_url:
        # 定义预期的输出片段（huge模型）
        expected_slice = torch.tensor(
            [[-1.1599, -0.9199, -1.2221], [-1.1952, -0.9269, -1.2307], [-1.2143, -0.9337, -1.2262]]
        )
    else:
        # 定义预期的输出片段（默认模型）
        expected_slice = torch.tensor(
            [[-0.9192, -0.8481, -1.1259], [-1.1349, -1.0034, -1.2599], [-1.1757, -1.0429, -1.2726]]
        )

    # 验证logits是否与预期输出片段相似
    assert torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4)

    # 保存模型到指定路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)

    # 保存图像处理器到指定路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    image_processor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必��参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth",
        type=str,
        help="URL of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_mae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
```