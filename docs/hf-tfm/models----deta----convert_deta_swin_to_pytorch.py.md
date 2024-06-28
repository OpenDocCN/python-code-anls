# `.\models\deta\convert_deta_swin_to_pytorch.py`

```
# 定义一个函数，用于生成 DETA 模型的配置信息
def get_deta_config(model_name):
    # 定义 Swin Transformer 的配置信息作为背骨网络配置
    backbone_config = SwinConfig(
        embed_dim=192,
        depths=(2, 2, 18, 2),
        num_heads=(6, 12, 24, 48),
        window_size=12,
        out_features=["stage2", "stage3", "stage4"],
    )

    # 定义 DETA 模型的总体配置
    config = DetaConfig(
        backbone_config=backbone_config,
        num_queries=900,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        num_feature_levels=5,
        assign_first_stage=True,
        with_box_refine=True,
        two_stage=True,
    )

    # 设置模型的标签信息
    repo_id = "huggingface/label-files"
    if "o365" in model_name:
        num_labels = 366
        filename = "object365-id2label.json"
    else:
        num_labels = 91
        filename = "coco-detection-id2label.json"

    # 加载并解析标签文件，设置模型的标签映射
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    # 返回生成的配置信息
    return config


# 定义一个函数，用于创建需要重命名的键值对列表
def create_rename_keys(config):
    rename_keys = []

    # stem（茎部分）的重命名操作
    # fmt: off
    rename_keys.append(("backbone.0.body.patch_embed.proj.weight", "model.backbone.model.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("backbone.0.body.patch_embed.proj.bias", "model.backbone.model.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("backbone.0.body.patch_embed.norm.weight", "model.backbone.model.embeddings.norm.weight"))
    rename_keys.append(("backbone.0.body.patch_embed.norm.bias", "model.backbone.model.embeddings.norm.bias"))
    # stages
    # 遍历配置中指定的每个深度值
    for i in range(len(config.backbone_config.depths)):
        # 根据每个深度值，再次遍历对应数量的层
        for j in range(config.backbone_config.depths[i]):
            # 将旧的键值对和新的键值对添加到重命名键列表中，用于重命名模型参数
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm1.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_before.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm1.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_before.bias"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.relative_position_bias_table", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_bias_table"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.relative_position_index", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.relative_position_index"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.proj.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.output.dense.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.attn.proj.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.output.dense.bias"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm2.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_after.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.norm2.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.layernorm_after.bias"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc1.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.intermediate.dense.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc1.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.intermediate.dense.bias"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc2.weight", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.output.dense.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.blocks.{j}.mlp.fc2.bias", f"model.backbone.model.encoder.layers.{i}.blocks.{j}.output.dense.bias"))

        # 对于前三层，额外重命名下采样模块的参数
        if i < 3:
            rename_keys.append((f"backbone.0.body.layers.{i}.downsample.reduction.weight", f"model.backbone.model.encoder.layers.{i}.downsample.reduction.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.downsample.norm.weight", f"model.backbone.model.encoder.layers.{i}.downsample.norm.weight"))
            rename_keys.append((f"backbone.0.body.layers.{i}.downsample.norm.bias", f"model.backbone.model.encoder.layers.{i}.downsample.norm.bias"))

    # 添加额外的重命名键，用于处理第一层的归一化权重和偏置
    rename_keys.append(("backbone.0.body.norm1.weight", "model.backbone.model.hidden_states_norms.stage2.weight"))
    rename_keys.append(("backbone.0.body.norm1.bias", "model.backbone.model.hidden_states_norms.stage2.bias"))
    # 添加需要重命名的键值对，将模型中的旧键名映射到新键名
    rename_keys.append(("backbone.0.body.norm2.weight", "model.backbone.model.hidden_states_norms.stage3.weight"))
    rename_keys.append(("backbone.0.body.norm2.bias", "model.backbone.model.hidden_states_norms.stage3.bias"))
    rename_keys.append(("backbone.0.body.norm3.weight", "model.backbone.model.hidden_states_norms.stage4.weight"))
    rename_keys.append(("backbone.0.body.norm3.bias", "model.backbone.model.hidden_states_norms.stage4.bias"))

    # 遍历所有的 transformer 编码层，将各层的权重和偏置映射到模型中对应层的新键名
    for i in range(config.encoder_layers):
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.weight", f"model.encoder.layers.{i}.self_attn.sampling_offsets.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.sampling_offsets.bias", f"model.encoder.layers.{i}.self_attn.sampling_offsets.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.weight", f"model.encoder.layers.{i}.self_attn.attention_weights.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.attention_weights.bias", f"model.encoder.layers.{i}.self_attn.attention_weights.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.weight", f"model.encoder.layers.{i}.self_attn.value_proj.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.value_proj.bias", f"model.encoder.layers.{i}.self_attn.value_proj.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.weight", f"model.encoder.layers.{i}.self_attn.output_proj.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.self_attn.output_proj.bias", f"model.encoder.layers.{i}.self_attn.output_proj.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.weight", f"model.encoder.layers.{i}.self_attn_layer_norm.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"model.encoder.layers.{i}.self_attn_layer_norm.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"model.encoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"model.encoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"model.encoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"model.encoder.layers.{i}.fc2.bias"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"model.encoder.layers.{i}.final_layer_norm.weight"))
        rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"model.encoder.layers.{i}.final_layer_norm.bias"))

    # transformer decoder
    # 循环遍历配置中的解码器层数，生成重命名键列表
    for i in range(config.decoder_layers):
        # 添加每一层解码器交叉注意力模块的权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.weight", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight"))
        # 添加每一层解码器交叉注意力模块的偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.sampling_offsets.bias", f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias"))
        # 添加每一层解码器交叉注意力模块的注意力权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.weight", f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight"))
        # 添加每一层解码器交叉注意力模块的注意力偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.attention_weights.bias", f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias"))
        # 添加每一层解码器交叉注意力模块的值投影权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.weight", f"model.decoder.layers.{i}.encoder_attn.value_proj.weight"))
        # 添加每一层解码器交叉注意力模块的值投影偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.value_proj.bias", f"model.decoder.layers.{i}.encoder_attn.value_proj.bias"))
        # 添加每一层解码器交叉注意力模块的输出投影权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.weight", f"model.decoder.layers.{i}.encoder_attn.output_proj.weight"))
        # 添加每一层解码器交叉注意力模块的输出投影偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.cross_attn.output_proj.bias", f"model.decoder.layers.{i}.encoder_attn.output_proj.bias"))
        # 添加每一层解码器第一个层归一化层的权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"))
        # 添加每一层解码器第一个层归一化层的偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"))
        # 添加每一层解码器自注意力模块的输出投影权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"model.decoder.layers.{i}.self_attn.out_proj.weight"))
        # 添加每一层解码器自注意力模块的输出投影偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"model.decoder.layers.{i}.self_attn.out_proj.bias"))
        # 添加每一层解码器第二个层归一化层的权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight"))
        # 添加每一层解码器第二个层归一化层的偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias"))
        # 添加每一层解码器第一个全连接层的权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        # 添加每一层解码器第一个全连接层的偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        # 添加每一层解码器第二个全连接层的权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        # 添加每一层解码器第二个全连接层的偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        # 添加每一层解码器第三个归一化层的权重重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight"))
        # 添加每一层解码器第三个归一化层的偏置重命名键
        rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias"))

    # 格式化选项重新开启
    # 返回生成的重命名键列表
    return rename_keys
# 重命名字典中的键，将旧键移除并用新键替换
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将该值与新键关联起来
    dct[new] = val


# 将每个编码器层的矩阵拆分为查询（query）、键（key）和值（value）
def read_in_swin_q_k_v(state_dict, backbone_config):
    # 计算每个特征的维度列表
    num_features = [int(backbone_config.embed_dim * 2**i) for i in range(len(backbone_config.depths))]
    # 遍历深度列表
    for i in range(len(backbone_config.depths)):
        dim = num_features[i]
        # 遍历每个深度内的层数
        for j in range(backbone_config.depths[i]):
            # fmt: off
            # 读取输入投影层权重和偏置（原始实现中，这是一个单独的矩阵加偏置）
            in_proj_weight = state_dict.pop(f"backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.weight")
            in_proj_bias = state_dict.pop(f"backbone.0.body.layers.{i}.blocks.{j}.attn.qkv.bias")
            # 将查询（query）、键（key）和值（value）依次添加到状态字典中
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.query.bias"] = in_proj_bias[: dim]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[
                dim : dim * 2, :
            ]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.key.bias"] = in_proj_bias[
                dim : dim * 2
            ]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[
                -dim :, :
            ]
            state_dict[f"model.backbone.model.encoder.layers.{i}.blocks.{j}.attention.self.value.bias"] = in_proj_bias[-dim :]
            # fmt: on


# 读取解码器的查询（query）、键（key）和值（value）信息
def read_in_decoder_q_k_v(state_dict, config):
    # 解码器自注意力层
    hidden_size = config.d_model
    # 遍历解码器层数
    for i in range(config.decoder_layers):
        # 读取自注意力层输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # 将查询（query）、键（key）和值（value）依次添加到状态字典中
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:hidden_size, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:hidden_size]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[
            hidden_size : hidden_size * 2, :
        ]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size:]
# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 获取图片的原始数据流，并用 PIL 库打开图片
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_deta_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub):
    """
    复制/粘贴/调整模型的权重到我们的 DETA 结构中。
    """

    # 加载配置
    config = get_deta_config(model_name)

    # 加载原始状态字典
    if model_name == "deta-swin-large":
        # 从 HuggingFace Hub 下载 adet_swin_ft.pth 文件
        checkpoint_path = hf_hub_download(repo_id="nielsr/deta-checkpoints", filename="adet_swin_ft.pth")
    elif model_name == "deta-swin-large-o365":
        # 从 HuggingFace Hub 下载 deta_swin_pt_o365.pth 文件
        checkpoint_path = hf_hub_download(repo_id="jozhang97/deta-swin-l-o365", filename="deta_swin_pt_o365.pth")
    else:
        raise ValueError(f"Model name {model_name} not supported")

    # 使用 torch.load 加载模型的状态字典，并将其放在 CPU 上
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]

    # 打印原始状态字典中每个参数的名称和形状
    for name, param in state_dict.items():
        print(name, param.shape)

    # 重命名键
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_swin_q_k_v(state_dict, config.backbone_config)
    read_in_decoder_q_k_v(state_dict, config)

    # 修正一些前缀
    for key in state_dict.copy().keys():
        if "transformer.decoder.class_embed" in key or "transformer.decoder.bbox_embed" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer.decoder", "model.decoder")] = val
        if "input_proj" in key:
            val = state_dict.pop(key)
            state_dict["model." + key] = val
        if "level_embed" in key or "pos_trans" in key or "pix_trans" in key or "enc_output" in key:
            val = state_dict.pop(key)
            state_dict[key.replace("transformer", "model")] = val

    # 最后，创建 HuggingFace 模型并加载状态字典
    model = DetaForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 根据 GPU 是否可用选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 加载图像处理器
    processor = DetaImageProcessor(format="coco_detection")

    # 验证在图像上的转换结果
    img = prepare_img()
    encoding = processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values.to(device))

    # 验证输出的 logits
    print("Logits:", outputs.logits[0, :3, :3])
    print("Boxes:", outputs.pred_boxes[0, :3, :3])
    if model_name == "deta-swin-large":
        # 预期的 logits 和 boxes
        expected_logits = torch.tensor(
            [[-7.6308, -2.8485, -5.3737], [-7.2037, -4.5505, -4.8027], [-7.2943, -4.2611, -4.6617]]
        )
        expected_boxes = torch.tensor([[0.4987, 0.4969, 0.9999], [0.2549, 0.5498, 0.4805], [0.5498, 0.2757, 0.0569]])
    # 如果模型名称为 "deta-swin-large-o365"，设置预期的逻辑回归输出张量
    expected_logits = torch.tensor(
        [[-8.0122, -3.5720, -4.9717], [-8.1547, -3.6886, -4.6389], [-7.6610, -3.6194, -5.0134]]
    )
    # 设置预期的边界框张量
    expected_boxes = torch.tensor([[0.2523, 0.5549, 0.4881], [0.7715, 0.4149, 0.4601], [0.5503, 0.2753, 0.0575]])

# 断言：验证模型输出的逻辑回归部分是否与预期值接近
assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
# 断言：验证模型输出的预测边界框部分是否与预期值接近
assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)

# 输出确认信息
print("Everything ok!")

# 如果指定了 PyTorch 模型保存路径
if pytorch_dump_folder_path:
    # 日志记录：保存 PyTorch 模型和处理器到指定路径
    logger.info(f"Saving PyTorch model and processor to {pytorch_dump_folder_path}...")
    # 创建保存路径（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 保存处理器到指定路径
    processor.save_pretrained(pytorch_dump_folder_path)

# 如果需要推送到 Hub
if push_to_hub:
    # 输出信息：推送模型和处理器到 Hub
    print("Pushing model and processor to hub...")
    # 推送模型到 Hub，使用指定的命名空间和模型名称
    model.push_to_hub(f"jozhang97/{model_name}")
    # 推送处理器到 Hub，使用指定的命名空间和模型名称
    processor.push_to_hub(f"jozhang97/{model_name}")
# 如果当前脚本被直接执行而非被导入为模块，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数：--model_name，类型为字符串，默认为"deta-swin-large"，
    # 可选值为["deta-swin-large", "deta-swin-large-o365"]，用于指定要转换的模型名称
    parser.add_argument(
        "--model_name",
        type=str,
        default="deta-swin-large",
        choices=["deta-swin-large", "deta-swin-large-o365"],
        help="Name of the model you'd like to convert.",
    )

    # 添加命令行参数：--pytorch_dump_folder_path，类型为字符串，默认为None，
    # 用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the folder to output PyTorch model.",
    )

    # 添加命令行参数：--push_to_hub，如果指定该参数，则设置为 True，否则为 False，
    # 用于指定是否将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数，并将它们保存到 args 对象中
    args = parser.parse_args()

    # 调用 convert_deta_checkpoint 函数，传入命令行参数中的模型名称、PyTorch 模型输出路径和推送到 hub 的标志
    convert_deta_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```