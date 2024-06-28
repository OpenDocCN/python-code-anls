# `.\models\vit\convert_vit_timm_to_pytorch.py`

```py
# 定义用于重命名权重键的函数，根据给定的配置和是否基于基础模型来生成重命名规则列表
def create_rename_keys(config, base_model=False):
    rename_keys = []
    # 遍历所有编码器层
    for i in range(config.num_hidden_layers):
        # 添加权重重命名规则：输入层的归一化权重
        rename_keys.append((f"blocks.{i}.norm1.weight", f"vit.encoder.layer.{i}.layernorm_before.weight"))
        # 添加权重重命名规则：输入层的归一化偏置
        rename_keys.append((f"blocks.{i}.norm1.bias", f"vit.encoder.layer.{i}.layernorm_before.bias"))
        # 添加权重重命名规则：注意力机制输出的投影权重
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"vit.encoder.layer.{i}.attention.output.dense.weight"))
        # 添加权重重命名规则：注意力机制输出的投影偏置
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"vit.encoder.layer.{i}.attention.output.dense.bias"))
        # 添加权重重命名规则：输出层的归一化权重
        rename_keys.append((f"blocks.{i}.norm2.weight", f"vit.encoder.layer.{i}.layernorm_after.weight"))
        # 添加权重重命名规则：输出层的归一化偏置
        rename_keys.append((f"blocks.{i}.norm2.bias", f"vit.encoder.layer.{i}.layernorm_after.bias"))
        # 添加权重重命名规则：中间层的全连接层1权重
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"vit.encoder.layer.{i}.intermediate.dense.weight"))
        # 添加权重重命名规则：中间层的全连接层1偏置
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"vit.encoder.layer.{i}.intermediate.dense.bias"))
        # 添加权重重命名规则：中间层的全连接层2权重
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"vit.encoder.layer.{i}.output.dense.weight"))
        # 添加权重重命名规则：中间层的全连接层2偏置
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"vit.encoder.layer.{i}.output.dense.bias"))

    # 添加权重重命名规则：CLS token
    rename_keys.append(("cls_token", "vit.embeddings.cls_token"))
    # 添加权重重命名规则：补丁嵌入的投影权重
    rename_keys.append(("patch_embed.proj.weight", "vit.embeddings.patch_embeddings.projection.weight"))
    # 添加权重重命名规则：补丁嵌入的投影偏置
    rename_keys.append(("patch_embed.proj.bias", "vit.embeddings.patch_embeddings.projection.bias"))
    # 添加权重重命名规则：位置嵌入
    rename_keys.append(("pos_embed", "vit.embeddings.position_embeddings"))

    return rename_keys
    # 如果存在基础模型（base_model不为None）
    if base_model:
        # 将以下键值对添加到rename_keys列表中，用于重命名模型参数：
        # 将"norm.weight"重命名为"layernorm.weight"
        # 将"norm.bias"重命名为"layernorm.bias"
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )

        # 如果仅有基础模型，需要移除所有以"vit"开头的键中的前缀"vit"
        # 对rename_keys中的每对键值对进行检查和可能的修改
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("vit") else pair for pair in rename_keys]
    else:
        # 如果不仅有基础模型，而是包含分类头（layernorm + classification head）
        # 将以下键值对添加到rename_keys列表中，用于重命名模型参数：
        # 将"norm.weight"重命名为"vit.layernorm.weight"
        # 将"norm.bias"重命名为"vit.layernorm.bias"
        # 将"head.weight"重命名为"classifier.weight"
        # 将"head.bias"重命名为"classifier.bias"
        rename_keys.extend(
            [
                ("norm.weight", "vit.layernorm.weight"),
                ("norm.bias", "vit.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    # 返回最终的重命名后的键值对列表
    return rename_keys
# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历每个编码器层，分离出查询(query)、键(keys)和值(values)的权重和偏置
    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
        # 读取输入投影层的权重和偏置（在timm中，这是一个包含权重和偏置的单一矩阵）
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 将查询(query)、键(keys)和值(values)按顺序添加到状态字典中
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


def remove_classification_head_(state_dict):
    # 移除状态字典中的分类头部权重和偏置项
    ignore_keys = ["head.weight", "head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    # 重命名字典中的键
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    # 准备一张可爱猫咪的图像来验证结果
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vit_checkpoint(vit_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our ViT structure.
    """

    # define default ViT configuration
    config = ViTConfig()
    base_model = False

    # load original model from timm
    timm_model = timm.create_model(vit_name, pretrained=True)
    timm_model.eval()

    # detect unsupported ViT models in transformers
    # fc_norm is present
    # 检测transformers中不支持的ViT模型
    if not isinstance(getattr(timm_model, "fc_norm", None), torch.nn.Identity):
        raise ValueError(f"{vit_name} is not supported in transformers because of the presence of fc_norm.")

    # use of global average pooling in combination (or without) class token
    # 检测在使用全局平均池化时（或没有类令牌时），transformers中不支持的ViT模型
    if getattr(timm_model, "global_pool", None) == "avg":
        raise ValueError(f"{vit_name} is not supported in transformers because of use of global average pooling.")

    # CLIP style vit with norm_pre layer present
    # 检测是否存在norm_pre层，以确定是否是类似CLIP风格的ViT模型
    # 检查是否为 CLIP 风格的 ViT，且其 norm_pre 层不是 torch.nn.Identity
    if "clip" in vit_name and not isinstance(getattr(timm_model, "norm_pre", None), torch.nn.Identity):
        raise ValueError(
            f"{vit_name} is not supported in transformers because it's a CLIP style ViT with norm_pre layer."
        )

    # 检查是否为 SigLIP 风格的 ViT，且具有 attn_pool 层
    if "siglip" in vit_name and getattr(timm_model, "global_pool", None) == "map":
        raise ValueError(
            f"{vit_name} is not supported in transformers because it's a SigLIP style ViT with attn_pool."
        )

    # 检查 ViT 模型的 blocks[0] 中是否使用了 layer scale
    if not isinstance(getattr(timm_model.blocks[0], "ls1", None), torch.nn.Identity) or not isinstance(
        getattr(timm_model.blocks[0], "ls2", None), torch.nn.Identity
    ):
        raise ValueError(f"{vit_name} is not supported in transformers because it uses a layer scale in its blocks.")

    # 检查是否为混合 ResNet-ViT 模型，即 patch_embed 不是 timm.layers.PatchEmbed 类型
    if not isinstance(timm_model.patch_embed, timm.layers.PatchEmbed):
        raise ValueError(f"{vit_name} is not supported in transformers because it is a hybrid ResNet-ViT.")

    # 从 patch embedding 子模块中获取 patch 大小和图像大小
    config.patch_size = timm_model.patch_embed.patch_size[0]
    config.image_size = timm_model.patch_embed.img_size[0]

    # 从 timm 模型中获取特定于架构的参数
    config.hidden_size = timm_model.embed_dim
    config.intermediate_size = timm_model.blocks[0].mlp.fc1.out_features
    config.num_hidden_layers = len(timm_model.blocks)
    config.num_attention_heads = timm_model.blocks[0].attn.num_heads

    # 检查模型是否有分类头
    if timm_model.num_classes != 0:
        # 设置分类标签数量
        config.num_labels = timm_model.num_classes
        # 推断出 timm 模型的 ImageNet 子集
        imagenet_subset = infer_imagenet_subset(timm_model)
        dataset_info = ImageNetInfo(imagenet_subset)
        # 设置 id 到 label 名称的映射和 label 名称到 id 的映射
        config.id2label = {i: dataset_info.index_to_label_name(i) for i in range(dataset_info.num_classes())}
        config.label2id = {v: k for k, v in config.id2label.items()}
    else:
        # 若没有分类头，则模型将被转换为仅提取特征的模式
        print(f"{vit_name} is going to be converted as a feature extractor only.")
        base_model = True

    # 加载原始模型的 state_dict
    state_dict = timm_model.state_dict()

    # 如果是基础模型，移除和重命名 state_dict 中的一些键
    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config, base_model)

    # 加载 HuggingFace 模型
    if base_model:
        model = ViTModel(config, add_pooling_layer=False).eval()
    else:
        model = ViTForImageClassification(config).eval()
    model.load_state_dict(state_dict)

    # 在图像处理器 ViTImageProcessor/DeiTImageProcessor 上检查图像的输出
    if "deit" in vit_name:
        image_processor = DeiTImageProcessor(size=config.image_size)
    # 如果存在基础模型，则使用 ViTImageProcessor 处理图像数据
    else:
        image_processor = ViTImageProcessor(size=config.image_size)
    
    # 对准备好的图像数据进行编码，返回 PyTorch 张量表示
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    
    # 提取像素数值
    pixel_values = encoding["pixel_values"]
    
    # 使用模型进行推理，得到输出结果
    outputs = model(pixel_values)

    # 如果存在基础模型：
    if base_model:
        # 使用 timm_model 提取特征，并进行形状断言
        timm_pooled_output = timm_model.forward_features(pixel_values)
        assert timm_pooled_output.shape == outputs.last_hidden_state.shape
        assert torch.allclose(timm_pooled_output, outputs.last_hidden_state, atol=1e-1)
    else:
        # 使用 timm_model 进行推理，得到 logits，并进行形状断言
        timm_logits = timm_model(pixel_values)
        assert timm_logits.shape == outputs.logits.shape
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    # 确保指定路径下的文件夹存在，用于保存 PyTorch 模型
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    
    # 打印保存模型的信息
    print(f"Saving model {vit_name} to {pytorch_dump_folder_path}")
    
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    
    # 打印保存图像处理器的信息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果作为主程序执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--vit_name",
        default="vit_base_patch16_224",
        type=str,
        help="Name of the ViT timm model you'd like to convert.",
    )
    # 添加一个名为--vit_name的参数，默认值为"vit_base_patch16_224"，类型为字符串，用于指定要转换的 ViT 模型的名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个名为--pytorch_dump_folder_path的参数，值默认为None，类型为字符串，用于指定输出 PyTorch 模型的目录路径

    args = parser.parse_args()
    # 解析命令行参数并存储到args变量中

    convert_vit_checkpoint(args.vit_name, args.pytorch_dump_folder_path)
    # 调用函数convert_vit_checkpoint，传递解析得到的--vit_name和--pytorch_dump_folder_path参数


这段代码是一个命令行程序的入口点，使用argparse模块解析命令行参数，然后调用函数`convert_vit_checkpoint`进行处理。
```