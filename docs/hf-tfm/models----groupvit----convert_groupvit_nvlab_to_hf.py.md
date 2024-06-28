# `.\models\groupvit\convert_groupvit_nvlab_to_hf.py`

```py
# 定义函数用于重命名模型参数键名
def rename_key(name):
    # 如果键名中包含 "img_encoder.pos_embed"，则替换为 "vision_model.embeddings.position_embeddings"
    if "img_encoder.pos_embed" in name:
        name = name.replace("img_encoder.pos_embed", "vision_model.embeddings.position_embeddings")
    # 如果键名中包含 "img_encoder.patch_embed.proj"，则替换为 "vision_model.embeddings.patch_embeddings.projection"
    if "img_encoder.patch_embed.proj" in name:
        name = name.replace("img_encoder.patch_embed.proj", "vision_model.embeddings.patch_embeddings.projection")
    # 如果键名中包含 "img_encoder.patch_embed.norm"，则替换为 "vision_model.embeddings.layernorm"
    if "img_encoder.patch_embed.norm" in name:
        name = name.replace("img_encoder.patch_embed.norm", "vision_model.embeddings.layernorm")
    # 如果键名中包含 "img_encoder.layers"，则替换为 "vision_model.encoder.stages"
    if "img_encoder.layers" in name:
        name = name.replace("img_encoder.layers", "vision_model.encoder.stages")
    # 如果键名中包含 "blocks" 且不包含 "res"，则替换为 "layers"
    if "blocks" in name and "res" not in name:
        name = name.replace("blocks", "layers")
    # 如果键名中包含 "attn" 且不包含 "pre_assign"，则替换为 "self_attn"
    if "attn" in name and "pre_assign" not in name:
        name = name.replace("attn", "self_attn")
    # 如果键名中包含 "proj" 且同时包含 "self_attn" 且不包含 "text"，则替换为 "out_proj"
    if "proj" in name and "self_attn" in name and "text" not in name:
        name = name.replace("proj", "out_proj")
    # 如果键名中包含 "pre_assign_attn.attn.proj"，则替换为 "pre_assign_attn.attn.out_proj"
    if "pre_assign_attn.attn.proj" in name:
        name = name.replace("pre_assign_attn.attn.proj", "pre_assign_attn.attn.out_proj")
    # 如果键名中包含 "norm1"，则替换为 "layer_norm1"
    if "norm1" in name:
        name = name.replace("norm1", "layer_norm1")
    # 如果键名中包含 "norm2" 且不包含 "pre_assign"，则替换为 "layer_norm2"
    if "norm2" in name and "pre_assign" not in name:
        name = name.replace("norm2", "layer_norm2")
    # 如果键名中包含 "img_encoder.norm"，则替换为 "vision_model.layernorm"
    if "img_encoder.norm" in name:
        name = name.replace("img_encoder.norm", "vision_model.layernorm")
    # 如果键名中包含 "text_encoder.token_embedding"，则替换为 "text_model.embeddings.token_embedding"
    if "text_encoder.token_embedding" in name:
        name = name.replace("text_encoder.token_embedding", "text_model.embeddings.token_embedding")
    # 如果键名中包含 "text_encoder.positional_embedding"，则替换为 "text_model.embeddings.position_embedding.weight"
    if "text_encoder.positional_embedding" in name:
        name = name.replace("text_encoder.positional_embedding", "text_model.embeddings.position_embedding.weight")
    # 如果键名中包含 "text_encoder.transformer.resblocks."，则替换为 "text_model.encoder.layers."
    if "text_encoder.transformer.resblocks." in name:
        name = name.replace("text_encoder.transformer.resblocks.", "text_model.encoder.layers.")
    # 如果键名中包含 "ln_1"，则替换为 "layer_norm1"
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    # 如果键名中包含 "ln_2"，则替换为 "layer_norm2"
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    # 如果键名中包含 "c_fc"，则替换为 "fc1"
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    # 如果键名中包含 "c_proj"，则替换为 "fc2"
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    # 如果变量 name 中包含字符串 "text_encoder"
    if "text_encoder" in name:
        # 将其替换为 "text_model"
        name = name.replace("text_encoder", "text_model")
    
    # 如果变量 name 中包含字符串 "ln_final"
    if "ln_final" in name:
        # 将其替换为 "final_layer_norm"
        name = name.replace("ln_final", "final_layer_norm")
    
    # 处理投影层的命名映射
    # 如果变量 name 中包含字符串 "img_projector.linear_hidden."
    if "img_projector.linear_hidden." in name:
        # 将其替换为 "visual_projection."
        name = name.replace("img_projector.linear_hidden.", "visual_projection.")
    
    # 如果变量 name 中包含字符串 "img_projector.linear_out."
    if "img_projector.linear_out." in name:
        # 将其替换为 "visual_projection.3."
        name = name.replace("img_projector.linear_out.", "visual_projection.3.")
    
    # 如果变量 name 中包含字符串 "text_projector.linear_hidden"
    if "text_projector.linear_hidden" in name:
        # 将其替换为 "text_projection"
        name = name.replace("text_projector.linear_hidden", "text_projection")
    
    # 如果变量 name 中包含字符串 "text_projector.linear_out"
    if "text_projector.linear_out" in name:
        # 将其替换为 "text_projection.3"
        name = name.replace("text_projector.linear_out", "text_projection.3")
    
    # 返回处理后的 name 变量作为结果
    return name
def convert_state_dict(orig_state_dict, config):
    # 简单地返回原始状态字典，未经任何改动
    return orig_state_dict


# 我们将在一张可爱猫咪的图像上验证我们的结果
def prepare_img():
    # 图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库获取图像的原始字节流，并由 PIL 打开
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_groupvit_checkpoint(
    checkpoint_path, pytorch_dump_folder_path, model_name="groupvit-gcc-yfcc", push_to_hub=False
):
    """
    复制/粘贴/调整模型的权重以符合 Transformers 设计。
    """
    # 创建 GroupViT 模型配置
    config = GroupViTConfig()
    # 初始化 GroupViT 模型并设置为评估模式
    model = GroupViTModel(config).eval()

    # 从指定路径加载 GroupViT 模型的状态字典到 CPU
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 将加载的状态字典转换为新的状态字典，使用给定的配置
    new_state_dict = convert_state_dict(state_dict, config)
    # 加载新的状态字典到模型中，允许不严格匹配
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    # 断言确实缺失的键为指定值
    assert missing_keys == ["text_model.embeddings.position_ids"]
    # 断言确实的键为指定值或长度为 0
    assert (unexpected_keys == ["multi_label_logit_scale"]) or (len(unexpected_keys) == 0)

    # 验证结果
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = prepare_img()
    # 准备输入数据，包括文本和图像，使用 CLIP 处理器
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, padding=True, return_tensors="pt")

    with torch.no_grad():
        # 在不进行梯度计算的情况下，向模型输入数据并获取输出
        outputs = model(**inputs)

    # 根据模型名称设置期望的 logits 值，并进行数值近似比较
    if model_name == "groupvit-gcc-yfcc":
        expected_logits = torch.tensor([[13.3523, 6.3629]])
    elif model_name == "groupvit-gcc-redcaps":
        expected_logits = torch.tensor([[16.1873, 8.6230]])
    else:
        raise ValueError(f"Model name {model_name} not supported.")
    assert torch.allclose(outputs.logits_per_image, expected_logits, atol=1e-3)

    # 将处理器和模型保存到指定路径
    processor.save_pretrained(pytorch_dump_folder_path)
    model.save_pretrained(pytorch_dump_folder_path)
    print("Successfully saved processor and model to", pytorch_dump_folder_path)

    # 如果设置了推送到 Hub，执行推送操作
    if push_to_hub:
        print("Pushing to the hub...")
        processor.push_to_hub(model_name, organization="nielsr")
        model.push_to_hub(model_name, organization="nielsr")


if __name__ == "__main__":
    # 创建命令行参数解析器
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
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，执行 GroupViT 模型检查点的转换
    convert_groupvit_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```