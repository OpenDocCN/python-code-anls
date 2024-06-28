# `.\models\siglip\convert_siglip_to_hf.py`

```
# 获取 SigLIP 模型配置信息的函数
def get_siglip_config(model_name):
    # 创建 SigLIPConfig 对象
    config = SiglipConfig()

    # 根据模型名称确定词汇表大小
    vocab_size = 250000 if "i18n" in model_name else 32000
    # 根据模型名称确定图像大小
    image_size = model_name_to_image_size[model_name]
    # 根据模型名称确定补丁大小
    patch_size = 16 if "patch16" in model_name else 14

    # 设置视觉配置的图像大小和补丁大小
    config.vision_config.image_size = image_size
    config.vision_config.patch_size = patch_size
    # 设置文本配置的词汇表大小
    config.text_config.vocab_size = vocab_size

    # 如果模型名称包含 "base"，则无需额外操作
    if "base" in model_name:
        pass
    # 如果模型名称中包含"large"
    elif "large" in model_name:
        # 设置文本模型的隐藏层大小为1024
        config.text_config.hidden_size = 1024
        # 设置文本模型的中间层大小为4096
        config.text_config.intermediate_size = 4096
        # 设置文本模型的隐藏层数量为24
        config.text_config.num_hidden_layers = 24
        # 设置文本模型的注意力头数为16
        config.text_config.num_attention_heads = 16
        # 设置视觉模型的隐藏层大小为1024
        config.vision_config.hidden_size = 1024
        # 设置视觉模型的中间层大小为4096
        config.vision_config.intermediate_size = 4096
        # 设置视觉模型的隐藏层数量为24
        config.vision_config.num_hidden_layers = 24
        # 设置视觉模型的注意力头数为16
        config.vision_config.num_attention_heads = 16
    # 如果模型名称中包含"so400m"
    elif "so400m" in model_name:
        # 设置文本模型的隐藏层大小为1152
        config.text_config.hidden_size = 1152
        # 设置文本模型的中间层大小为4304
        config.text_config.intermediate_size = 4304
        # 设置文本模型的隐藏层数量为27
        config.text_config.num_hidden_layers = 27
        # 设置文本模型的注意力头数为16
        config.text_config.num_attention_heads = 16
        # 设置视觉模型的隐藏层大小为1152
        config.vision_config.hidden_size = 1152
        # 设置视觉模型的中间层大小为4304
        config.vision_config.intermediate_size = 4304
        # 设置视觉模型的隐藏层数量为27
        config.vision_config.num_hidden_layers = 27
        # 设置视觉模型的注意力头数为16
        config.vision_config.num_attention_heads = 16
    else:
        # 若模型名称不符合已知模型，则引发值错误异常
        raise ValueError("Model not supported")

    # 返回配置对象config
    return config
def create_rename_keys(config):
    rename_keys = []
    # fmt: off  # 关闭代码格式化，以便后续手动指定格式

    # vision encoder  # 以下是关于视觉编码器的重命名键设置

    # 将旧键 "params/img/embedding/kernel" 映射到新键 "vision_model.embeddings.patch_embedding.weight"，并添加到重命名键列表中
    rename_keys.append(("params/img/embedding/kernel", "vision_model.embeddings.patch_embedding.weight"))

    # 将旧键 "params/img/embedding/bias" 映射到新键 "vision_model.embeddings.patch_embedding.bias"，并添加到重命名键列表中
    rename_keys.append(("params/img/embedding/bias", "vision_model.embeddings.patch_embedding.bias"))

    # 将旧键 "params/img/pos_embedding" 映射到新键 "vision_model.embeddings.position_embedding.weight"，并添加到重命名键列表中
    rename_keys.append(("params/img/pos_embedding", "vision_model.embeddings.position_embedding.weight"))
    # 遍历从配置中获取的视觉模型的隐藏层数量次数，进行重命名键值对的添加
    for i in range(config.vision_config.num_hidden_layers):
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的LayerNorm_0层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/scale", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的LayerNorm_0层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_0/bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的LayerNorm_1层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/scale", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的LayerNorm_1层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/LayerNorm_1/bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MlpBlock_0层的第一层全连接层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MlpBlock_0层的第一层全连接层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MlpBlock_0层的第二层全连接层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MlpBlock_0层的第二层全连接层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的key投影层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的key投影层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的value投影层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的value投影层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的query投影层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的query投影层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的输出投影层的权重参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        # 添加重命名键值对，将旧参数路径映射到新的视觉模型编码器的第i层的MultiHeadDotProductAttention_0层的输出投影层的偏置参数路径
        rename_keys.append((f"params/img/Transformer/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"))
    
    # 添加重命名键值对，将旧参数路径映射到新的视觉模型的编码器层之后的LayerNorm层的权重参数路径
    rename_keys.append(("params/img/Transformer/encoder_norm/scale", "vision_model.post_layernorm.weight"))
    # 添加重命名键值对，将旧参数路径映射到新的视觉模型的编码器层之后的LayerNorm层的偏置参数路径
    rename_keys.append(("params/img/Transformer/encoder_norm/bias", "vision_model.post_layernorm.bias"))
    
    # 添加重命名键值对，将旧参数路径映射到新的视觉模型的头部模块的探测参数路径
    rename_keys.append(("params/img/MAPHead_0/probe", "vision_model.head.probe"))
    # 将键值对添加到 `rename_keys` 列表，用于指定源键和目标键的映射关系，用于重命名模型参数
    
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/scale", "vision_model.head.layernorm.weight"))
    rename_keys.append(("params/img/MAPHead_0/LayerNorm_0/bias", "vision_model.head.layernorm.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/kernel", "vision_model.head.mlp.fc1.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_0/bias", "vision_model.head.mlp.fc1.bias"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/kernel", "vision_model.head.mlp.fc2.weight"))
    rename_keys.append(("params/img/MAPHead_0/MlpBlock_0/Dense_1/bias", "vision_model.head.mlp.fc2.bias"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/kernel", "vision_model.head.attention.out_proj.weight"))
    rename_keys.append(("params/img/MAPHead_0/MultiHeadDotProductAttention_0/out/bias", "vision_model.head.attention.out_proj.bias"))
    
    # text encoder
    
    # 添加用于文本编码器的键值对映射，重命名模型参数
    rename_keys.append(("params/txt/Embed_0/embedding", "text_model.embeddings.token_embedding.weight"))
    rename_keys.append(("params/txt/pos_embedding", "text_model.embeddings.position_embedding.weight"))
    # 遍历配置中指定的文本模型隐藏层数量次数
    for i in range(config.text_config.num_hidden_layers):
        # 将参数重命名并添加到 rename_keys 列表中，映射到文本模型编码器每一层的 LayerNorm 层的权重和偏置
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/scale", f"text_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_0/bias", f"text_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/scale", f"text_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/LayerNorm_1/bias", f"text_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/kernel", f"text_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_0/bias", f"text_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/kernel", f"text_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MlpBlock_0/Dense_1/bias", f"text_model.encoder.layers.{i}.mlp.fc2.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/kernel", f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/key/bias", f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/kernel", f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/value/bias", f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/kernel", f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/query/bias", f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/kernel", f"text_model.encoder.layers.{i}.self_attn.out_proj.weight"))
        rename_keys.append((f"params/txt/Encoder_0/encoderblock_{i}/MultiHeadDotProductAttention_0/out/bias", f"text_model.encoder.layers.{i}.self_attn.out_proj.bias"))
    
    # 将最后几个参数重命名并添加到 rename_keys 列表中，映射到文本模型的最终归一化层、输出层权重和偏置
    rename_keys.append(("params/txt/Encoder_0/encoder_norm/scale", "text_model.final_layer_norm.weight"))
    rename_keys.append(("params/txt/Encoder_0/encoder_norm/bias", "text_model.final_layer_norm.bias"))
    rename_keys.append(("params/txt/head/kernel", "text_model.head.weight"))
    rename_keys.append(("params/txt/head/bias", "text_model.head.bias"))
    
    # 学习到的温度和偏置（此处的注释并没有提供代码细节，可能表示这部分信息是从数据中学习到的额外参数）
    # 将元组 ("params/t", "logit_scale") 添加到 rename_keys 列表中
    rename_keys.append(("params/t", "logit_scale"))
    # 将元组 ("params/b", "logit_bias") 添加到 rename_keys 列表中
    rename_keys.append(("params/b", "logit_bias"))

    # 返回 rename_keys 列表作为函数的结果
    return rename_keys
# 重命名字典中的键，并根据配置修改值的形状
def rename_key(dct, old, new, config):
    # 弹出旧键对应的值
    val = dct.pop(old)

    # 根据新键中的标识和配置调整值的形状
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if ("out_proj" in new or "v_proj" in new or "k_proj" in new or "q_proj" in new) and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    # 如果新键指定了特定的权重矩阵，进行转置操作
    if "patch_embedding.weight" in new:
        val = val.transpose(3, 2, 0, 1)
    elif new.endswith("weight") and "position_embedding" not in new and "token_embedding" not in new:
        val = val.T

    # 根据新键中的标识和配置再次调整值的形状
    if "position_embedding" in new and "vision" in new:
        val = val.reshape(-1, config.vision_config.hidden_size)
    if "position_embedding" in new and "text" in new:
        val = val.reshape(-1, config.text_config.hidden_size)

    # 如果新键是偏置项，将值调整为一维数组
    if new.endswith("bias"):
        val = val.reshape(-1)

    # 将处理后的值转换为 Torch 张量，并存入字典中
    dct[new] = torch.from_numpy(val)


# 从状态字典中读取注意力机制的输入投影层参数
def read_in_q_k_v_head(state_dict, config):
    # 弹出并重塑键为"key/kernel"的参数
    key_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    # 弹出并重塑键为"key/bias"的参数
    key_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/key/bias").reshape(-1)
    # 弹出并重塑键为"value/kernel"的参数
    value_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    # 弹出并重塑键为"value/bias"的参数
    value_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/value/bias").reshape(-1)
    # 弹出并重塑键为"query/kernel"的参数
    query_proj_weight = (
        state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/kernel")
        .reshape(-1, config.vision_config.hidden_size)
        .T
    )
    # 弹出并重塑键为"query/bias"的参数
    query_proj_bias = state_dict.pop("params/img/MAPHead_0/MultiHeadDotProductAttention_0/query/bias").reshape(-1)

    # 将重塑后的参数拼接成一个单一的矩阵和向量，并加入状态字典中
    state_dict["vision_model.head.attention.in_proj_weight"] = torch.from_numpy(
        np.concatenate([query_proj_weight, key_proj_weight, value_proj_weight], axis=0)
    )
    state_dict["vision_model.head.attention.in_proj_bias"] = torch.from_numpy(
        np.concatenate([query_proj_bias, key_proj_bias, value_proj_bias], axis=0)
    )
# 定义函数，用于将模型的权重转换到 SigLIP 结构
def convert_siglip_checkpoint(model_name, pytorch_dump_folder_path, verify_logits=True, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our SigLIP structure.
    """

    # 获取默认的 SigLIP 配置
    config = get_siglip_config(model_name)

    # 获取模型名称对应的检查点
    checkpoint = model_name_to_checkpoint[model_name]

    # 获取词汇文件路径
    if "i18n" in model_name:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/multilingual_vocab/sentencepiece.model"
    else:
        vocab_file = "/Users/nielsrogge/Documents/SigLIP/english_vocab/sentencepiece.model"

    # 加载原始状态字典
    data = load(checkpoint)
    state_dict = flatten_nested_dict(data)

    # 移除并重命名一些键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest, config)

    # 对注意力池化头的 qkv 矩阵需要特殊处理
    read_in_q_k_v_head(state_dict, config)

    # 加载 HuggingFace 模型
    model = SiglipModel(config).eval()
    model.load_state_dict(state_dict)

    # 创建处理器
    # 注意: 使得分词器不返回 attention_mask，因为原始模型不需要它
    image_size = config.vision_config.image_size
    size = {"height": image_size, "width": image_size}
    image_processor = SiglipImageProcessor(size=size)
    tokenizer = SiglipTokenizer(vocab_file=vocab_file, model_input_names=["input_ids"])
    processor = SiglipProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # 在虚拟图片和文本上进行验证
    url_1 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-ipod.jpg"
    image_1 = Image.open(requests.get(url_1, stream=True).raw).convert("RGB")
    url_2 = "https://cdn.openai.com/multimodal-neurons/assets/apple/apple-blank.jpg"
    image_2 = Image.open(requests.get(url_2, stream=True).raw).convert("RGB")
    texts = ["an apple", "a picture of an apple"]

    inputs = processor(images=[image_1, image_2], text=texts, return_tensors="pt", padding="max_length")

    # 针对输入的 input_ids 进行验证
    if image_size == 224:
        filename = "siglip_pixel_values.pt"
    elif image_size == 256:
        filename = "siglip_pixel_values_256.pt"
    elif image_size == 384:
        filename = "siglip_pixel_values_384.pt"
    elif image_size == 512:
        filename = "siglip_pixel_values_512.pt"
    else:
        raise ValueError("Image size not supported")

    # 下载并加载原始像素数值
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename=filename, repo_type="dataset")
    original_pixel_values = torch.load(filepath)

    # 下载并加载原始 input_ids
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="siglip_input_ids.pt", repo_type="dataset")
    original_input_ids = torch.load(filepath)

    # 如果模型名称不包含 "i18n"，则断言 inputs.input_ids 与 original_input_ids 相等
    if "i18n" not in model_name:
        assert inputs.input_ids.tolist() == original_input_ids.tolist()

    # 打印原始像素值的平均值
    print("Mean of original pixel values:", original_pixel_values.mean())
    # 输出新像素值的均值
    print("Mean of new pixel values:", inputs.pixel_values.mean())

    # 使用原始像素值进行测试，因为我们没有准确的像素值
    with torch.no_grad():
        # 使用模型进行推断，输入包括输入的 ID 和原始像素值
        outputs = model(input_ids=inputs.input_ids, pixel_values=original_pixel_values)

    # 输出前三行三列的 logits_per_image
    print(outputs.logits_per_image[:3, :3])

    # 计算输出的 logits_per_image 的 sigmoid 函数，得到概率值
    probs = torch.sigmoid(outputs.logits_per_image)
    # 打印第一张图像是 texts[0] 的概率
    print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
    # 打印第一张图像是 texts[1] 的概率
    print(f"{probs[0][1]:.1%} that image 0 is '{texts[1]}'")

    # 如果需要验证 logits
    if verify_logits:
        # 根据模型名称选择预期的 slice
        if model_name == "siglip-base-patch16-224":
            expected_slice = torch.tensor(
                [[-2.9621, -2.1672], [-0.2713, 0.2910]],
            )
        elif model_name == "siglip-base-patch16-256":
            expected_slice = torch.tensor(
                [[-3.1146, -1.9894], [-0.7312, 0.6387]],
            )
        elif model_name == "siglip-base-patch16-384":
            expected_slice = torch.tensor(
                [[-2.8098, -2.1891], [-0.4242, 0.4102]],
            )
        elif model_name == "siglip-base-patch16-512":
            expected_slice = torch.tensor(
                [[-2.7899, -2.2668], [-0.4295, -0.0735]],
            )
        elif model_name == "siglip-large-patch16-256":
            expected_slice = torch.tensor(
                [[-1.5827, -0.5801], [-0.9153, 0.1363]],
            )
        elif model_name == "siglip-large-patch16-384":
            expected_slice = torch.tensor(
                [[-2.1523, -0.2899], [-0.2959, 0.7884]],
            )
        elif model_name == "siglip-so400m-patch14-384":
            expected_slice = torch.tensor([[-1.2441, -0.6649], [-0.7060, 0.7374]])
        elif model_name == "siglip-base-patch16-256-i18n":
            expected_slice = torch.tensor(
                [[-0.9064, 0.1073], [-0.0299, 0.5304]],
            )

        # 断言前三行三列的 logits_per_image 与预期的 slice 相似
        assert torch.allclose(outputs.logits_per_image[:3, :3], expected_slice, atol=1e-4)
        print("Looks ok!")

    # 如果有指定的 pytorch_dump_folder_path
    if pytorch_dump_folder_path is not None:
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型和处理器的信息
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 打印保存处理器的信息
        print(f"Saving processor to {pytorch_dump_folder_path}")
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 推送模型到 Hub
        model.push_to_hub(f"nielsr/{model_name}")
        # 推送处理器到 Hub
        processor.push_to_hub(f"nielsr/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建命令行参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="siglip-base-patch16-224",
        type=str,
        choices=model_name_to_checkpoint.keys(),
        help="Name of the model you'd like to convert.",
    )
    # 添加一个必选的参数 `--model_name`，默认为 "siglip-base-patch16-224"，
    # 类型为字符串，可以从 `model_name_to_checkpoint` 字典的键中选择，
    # 用于指定要转换的模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个可选的参数 `--pytorch_dump_folder_path`，默认为 None，
    # 类型为字符串，用于指定输出 PyTorch 模型的目录路径

    parser.add_argument(
        "--verify_logits",
        action="store_false",
        help="Whether to verify logits against the original implementation.",
    )
    # 添加一个可选的开关参数 `--verify_logits`，
    # 当存在时将其设置为 False，用于指示是否对 logits 进行与原始实现的验证

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加一个可选的开关参数 `--push_to_hub`，
    # 当存在时设置为 True，用于指示是否将转换后的模型推送到 🤗 hub

    # 解析命令行参数
    args = parser.parse_args()

    # 调用转换函数，传入解析后的参数
    convert_siglip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
```