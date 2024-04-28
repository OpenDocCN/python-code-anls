# `.\transformers\models\clipseg\convert_clipseg_original_pytorch_to_hf.py`

```
# 指定文件编码为 UTF-8
# 版权声明，声明使用 Apache 许可证 2.0 版本
# 导入必要的库和模块
# argparse 用于解析命令行参数
# requests 用于发送 HTTP 请求
# torch 是 PyTorch 深度学习框架
# PIL 是 Python Imaging Library，用于图像处理
# transformers 是 Hugging Face 提供的自然语言处理模型库
# CLIPSegConfig 用于配置 CLIPSeg 模型
# CLIPSegForImageSegmentation 是用于图像分割的 CLIPSeg 模型
# CLIPSegProcessor 用于处理 CLIPSeg 模型的输入数据
# CLIPSegTextConfig 用于配置 CLIPSeg 模型的文本部分
# CLIPSegVisionConfig 用于配置 CLIPSeg 模型的视觉部分
# CLIPTokenizer 用于对文本进行分词
# ViTImageProcessor 用于处理图像数据

# 定义函数 get_clipseg_config，根据模型名称获取 CLIPSeg 的配置
def get_clipseg_config(model_name):
    # 创建 CLIPSeg 的文本部分配置对象
    text_config = CLIPSegTextConfig()
    # 创建 CLIPSeg 的视觉部分配置对象，并设置图像块大小为 16
    vision_config = CLIPSegVisionConfig(patch_size=16)

    # 根据模型名称确定是否使用复杂的转置卷积
    use_complex_transposed_convolution = True if "refined" in model_name else False
    # 根据模型名称确定是否降低维度
    reduce_dim = 16 if "rd16" in model_name else 64

    # 根据文本和视觉部分的配置，以及其他参数创建 CLIPSeg 的配置对象
    config = CLIPSegConfig.from_text_vision_configs(
        text_config,
        vision_config,
        use_complex_transposed_convolution=use_complex_transposed_convolution,
        reduce_dim=reduce_dim,
    )
    # 返回配置对象
    return config

# 定义函数 rename_key，用于重命名模型参数的键名
def rename_key(name):
    # 更新前缀
    if "clip_model" in name:
        name = name.replace("clip_model", "clip")
    if "transformer" in name:
        if "visual" in name:
            name = name.replace("visual.transformer", "vision_model")
        else:
            name = name.replace("transformer", "text_model")
    if "resblocks" in name:
        name = name.replace("resblocks", "encoder.layers")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if "attn" in name and "self" not in name:
        name = name.replace("attn", "self_attn")
    # 更新文本编码器的键名
    if "token_embedding" in name:
        name = name.replace("token_embedding", "text_model.embeddings.token_embedding")
    if "positional_embedding" in name and "visual" not in name:
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    # 更新视觉编码器的键名
    if "visual.class_embedding" in name:
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    # 检查是否包含 "visual.positional_embedding"，如果是，则替换为 "vision_model.embeddings.position_embedding.weight"
    if "visual.positional_embedding" in name:
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    # 检查是否包含 "visual.ln_pre"，如果是，则替换为 "vision_model.pre_layrnorm"
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layrnorm")
    # 检查是否包含 "visual.ln_post"，如果是，则替换为 "vision_model.post_layernorm"
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    # 检查是否包含 "visual.proj"，如果是，则替换为 "visual_projection.weight"
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    # 检查是否包含 "text_projection"，如果是，则替换为 "text_projection.weight"
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")
    # 检查是否包含 "trans_conv"，如果是，则替换为 "transposed_convolution"
    if "trans_conv" in name:
        name = name.replace("trans_conv", "transposed_convolution")
    # 如果包含 "film_mul"、"film_add"、"reduce" 或 "transposed_convolution"，则添加前缀 "decoder."
    if "film_mul" in name or "film_add" in name or "reduce" in name or "transposed_convolution" in name:
        name = "decoder." + name
    # 检查是否包含 "blocks"，如果是，则替换为 "decoder.layers"
    if "blocks" in name:
        name = name.replace("blocks", "decoder.layers")
    # 检查是否包含 "linear1"，如果是，则替换为 "mlp.fc1"
    if "linear1" in name:
        name = name.replace("linear1", "mlp.fc1")
    # 检查是否包含 "linear2"，如果是，则替换为 "mlp.fc2"
    if "linear2" in name:
        name = name.replace("linear2", "mlp.fc2")
    # 检查是否包含 "norm1" 且不包含 "layer_"，如果是，则替换为 "layer_norm1"
    if "norm1" in name and "layer_" not in name:
        name = name.replace("norm1", "layer_norm1")
    # 检查是否包含 "norm2" 且不包含 "layer_"，如果是，则替换为 "layer_norm2"
    if "norm2" in name and "layer_" not in name:
        name = name.replace("norm2", "layer_norm2")

    # 返回处理后的名称
    return name
# 转换状态字典的函数，将原始的状态字典转换为新的格式
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的键的副本
    for key in orig_state_dict.copy().keys():
        # 弹出原始状态字典中的键，并获取对应的值
        val = orig_state_dict.pop(key)

        # 如果键以"clip_model"开头且包含"attn.in_proj"
        if key.startswith("clip_model") and "attn.in_proj" in key:
            # 根据键的结构提取信息
            key_split = key.split(".")
            # 如果键中包含"visual"
            if "visual" in key:
                # 获取层编号和维度
                layer_num = int(key_split[4])
                dim = config.vision_config.hidden_size
                prefix = "vision_model"
            else:
                # 获取层编号和维度
                layer_num = int(key_split[3])
                dim = config.text_config.hidden_size
                prefix = "text_model"

            # 如果键中包含"weight"
            if "weight" in key:
                # 更新新的键值对，重命名键以匹配新模型结构
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                # 更新新的键值对，重命名键以匹配新模型结构
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"clip.{prefix}.encoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        # 如果键中包含"self_attn"但不包含"out_proj"
        elif "self_attn" in key and "out_proj" not in key:
            # 根据键的结构提取信息
            key_split = key.split(".")
            layer_num = int(key_split[1])
            dim = config.reduce_dim
            # 如果键中包含"weight"
            if "weight" in key:
                # 更新新的键值对，重命名键以匹配新模型结构
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.weight"] = val[:dim, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.weight"] = val[dim : dim * 2, :]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.weight"] = val[-dim:, :]
            else:
                # 更新新的键值对，重命名键以匹配新模型结构
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.q_proj.bias"] = val[:dim]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.k_proj.bias"] = val[dim : dim * 2]
                orig_state_dict[f"decoder.layers.{layer_num}.self_attn.v_proj.bias"] = val[-dim:]
        else:
            # 重命名键以匹配新模型结构
            new_name = rename_key(key)
            # 如果新键中包含"visual_projection"或"text_projection"，则对值进行转置
            if "visual_projection" in new_name or "text_projection" in new_name:
                val = val.T
            orig_state_dict[new_name] = val

    return orig_state_dict


# 我们将在一张可爱的猫的图片上验证结果
def prepare_img():
    # 图片的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 从 URL 中获取图片
    image = Image.open(requests.get(url, stream=True).raw)
    # 返回图片对象
    return image


def convert_clipseg_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path, push_to_hub):
    # 获取 CLIPSeg 的配置
    config = get_clipseg_config(model_name)
    # 创建 CLIPSeg 模型对象
    model = CLIPSegForImageSegmentation(config)
    # 设置模型为评估模式
    model.eval()

    # 加载模型的状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 移除一些键
``` 
    # 遍历状态字典的拷贝中的键
    for key in state_dict.copy().keys():
        # 如果键以 "model" 开头，则从状态字典中移除该键
        if key.startswith("model"):
            state_dict.pop(key, None)

    # 对一些键进行重命名
    state_dict = convert_state_dict(state_dict, config)
    # 载入模型的状态字典，允许缺失键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # 如果缺失的键不是 ["clip.text_model.embeddings.position_ids", "clip.vision_model.embeddings.position_ids"]，则引发值错误
    if missing_keys != ["clip.text_model.embeddings.position_ids", "clip.vision_model.embeddings.position_ids"]:
        raise ValueError("Missing keys that are not expected: {}".format(missing_keys))
    # 如果意外的键不是 ["decoder.reduce.weight", "decoder.reduce.bias"]，则引发值错误
    if unexpected_keys != ["decoder.reduce.weight", "decoder.reduce.bias"]:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")

    # 创建 ViTImageProcessor 对象，指定大小为 352
    image_processor = ViTImageProcessor(size=352)
    # 从预训练模型加载 CLIPTokenizer 对象
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # 创建 CLIPSegProcessor 对象，指定图像处理器和分词器
    processor = CLIPSegProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 准备图像数据
    image = prepare_img()
    # 准备文本数据
    text = ["a glass", "something to fill", "wood", "a jar"]

    # 处理文本和图像数据，返回 PyTorch 张量输入
    inputs = processor(text=text, images=[image] * len(text), padding="max_length", return_tensors="pt")

    # 关闭梯度计算
    with torch.no_grad():
        # 使用模型处理输入数据，获取输出
        outputs = model(**inputs)

    # 验证输出值是否符合预期
    expected_conditional = torch.tensor([0.1110, -0.1882, 0.1645])
    expected_pooled_output = torch.tensor([0.2692, -0.7197, -0.1328])
    if model_name == "clipseg-rd64-refined":
        expected_masks_slice = torch.tensor(
            [[-10.0407, -9.9431, -10.2646], [-9.9751, -9.7064, -9.9586], [-9.6891, -9.5645, -9.9618]]
        )
    elif model_name == "clipseg-rd64":
        expected_masks_slice = torch.tensor(
            [[-7.2877, -7.2711, -7.2463], [-7.2652, -7.2780, -7.2520], [-7.2239, -7.2204, -7.2001]]
        )
    elif model_name == "clipseg-rd16":
        expected_masks_slice = torch.tensor(
            [[-6.3955, -6.4055, -6.4151], [-6.3911, -6.4033, -6.4100], [-6.3474, -6.3702, -6.3762]]
        )
    else:
        # 如果模型名称不受支持，则引发值错误
        raise ValueError(f"Model name {model_name} not supported.")

    # 断言输出的 logits 的部分值接近预期的 masks_slice，容差为 1e-3
    assert torch.allclose(outputs.logits[0, :3, :3], expected_masks_slice, atol=1e-3)
    # 断言输出的 conditional_embeddings 的部分值接近预期的 conditional，容差为 1e-3
    assert torch.allclose(outputs.conditional_embeddings[0, :3], expected_conditional, atol=1e-3)
    # 断言输出的 pooled_output 的部分值接近预期的 pooled_output，容差为 1e-3
    assert torch.allclose(outputs.pooled_output[0, :3], expected_pooled_output, atol=1e-3)
    # 打印信息表明一切正常
    print("Looks ok!")

    # 如果指定了 PyTorch 转储文件夹路径
    if pytorch_dump_folder_path is not None:
        # 打印信息表明正在保存模型和处理器
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 打印信息表明正在推送模型和处理器到 Hub
        print(f"Pushing model and processor for {model_name} to the hub")
        # 将模型推送到 Hub
        model.push_to_hub(f"CIDAS/{model_name}")
        # 将处理器推送到 Hub
        processor.push_to_hub(f"CIDAS/{model_name}")
# 如果当前脚本被直接执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需参数
    parser.add_argument(
        "--model_name",
        default="clipseg-rd64",
        type=str,
        choices=["clipseg-rd16", "clipseg-rd64", "clipseg-rd64-refined"],
        help=(
            "Name of the model. Supported models are: clipseg-rd64, clipseg-rd16 and clipseg-rd64-refined (rd meaning"
            " reduce dimension)"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/Users/nielsrogge/Documents/CLIPSeg/clip_plus_rd64-uni.pth",
        type=str,
        help=(
            "Path to the original checkpoint. Note that the script assumes that the checkpoint includes both CLIP and"
            " the decoder weights."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 CLIPSeg 检查点转换为 PyTorch 模型
    convert_clipseg_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
```