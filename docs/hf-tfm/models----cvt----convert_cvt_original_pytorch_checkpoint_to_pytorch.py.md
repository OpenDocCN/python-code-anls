# `.\models\cvt\convert_cvt_original_pytorch_checkpoint_to_pytorch.py`

```py
# 定义一个函数，用于重命名嵌入层权重的函数
def embeddings(idx):
    """
    The function helps in renaming embedding layer weights.

    Args:
        idx: stage number in original model
    """
    # 存储重命名后的权重名称和对应的原始名称
    embed = []
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.weight",
            f"stage{idx}.patch_embed.proj.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.bias",
            f"stage{idx}.patch_embed.proj.bias",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.weight",
            f"stage{idx}.patch_embed.norm.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.bias",
            f"stage{idx}.patch_embed.norm.bias",
        )
    )
    return embed


# 定义一个函数，用于重命名注意力层权重的函数
def attention(idx, cnt):
    """
    The function helps in renaming attention block layers weights.

    Args:
        idx: stage number in original model
        cnt: count of blocks in each stage
    """
    # 存储重命名后的权重名称和对应的原始名称
    attention_weights = []
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.bias",
        )
    )
    return attention_weights
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_mean",
        )
    )
    # 添加注意力权重元组，包含查询卷积投影的运行均值路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_var",
        )
    )
    # 添加注意力权重元组，包含查询卷积投影的运行方差路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.num_batches_tracked",
        )
    )
    # 添加注意力权重元组，包含查询卷积投影的批次追踪计数路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.conv.weight",
        )
    )
    # 添加注意力权重元组，包含键卷积投影的卷积权重路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.weight",
        )
    )
    # 添加注意力权重元组，包含键卷积投影的归一化权重路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.bias",
        )
    )
    # 添加注意力权重元组，包含键卷积投影的归一化偏置路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_mean",
        )
    )
    # 添加注意力权重元组，包含键卷积投影的归一化运行均值路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_var",
        )
    )
    # 添加注意力权重元组，包含键卷积投影的归一化运行方差路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.num_batches_tracked",
        )
    )
    # 添加注意力权重元组，包含键卷积投影的归一化批次追踪计数路径和对应的模型中的路径

    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.conv.weight",
        )
    )
    # 添加注意力权重元组，包含值卷积投影的卷积权重路径和对应的模型中的路径
    # 将注意力权重相关的两个路径添加到 attention_weights 列表中
    attention_weights.append(
        (
            # 第一个路径：注意力权重的卷积投影值的权重参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.weight",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.weight",
        )
    )
    attention_weights.append(
        (
            # 第二个路径：注意力权重的卷积投影值的偏置参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.bias",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.bias",
        )
    )
    attention_weights.append(
        (
            # 第三个路径：注意力权重的卷积投影值的归一化均值路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_mean",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_mean",
        )
    )
    attention_weights.append(
        (
            # 第四个路径：注意力权重的卷积投影值的归一化方差路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_var",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_var",
        )
    )
    attention_weights.append(
        (
            # 第五个路径：注意力权重的卷积投影值的归一化追踪批次路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.num_batches_tracked",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.num_batches_tracked",
        )
    )
    attention_weights.append(
        (
            # 第六个路径：注意力权重的查询投影矩阵的权重参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.weight",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj_q.weight",
        )
    )
    attention_weights.append(
        (
            # 第七个路径：注意力权重的查询投影矩阵的偏置参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.bias",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj_q.bias",
        )
    )
    attention_weights.append(
        (
            # 第八个路径：注意力权重的键投影矩阵的权重参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.weight",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj_k.weight",
        )
    )
    attention_weights.append(
        (
            # 第九个路径：注意力权重的键投影矩阵的偏置参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.bias",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj_k.bias",
        )
    )
    attention_weights.append(
        (
            # 第十个路径：注意力权重的值投影矩阵的权重参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.weight",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj_v.weight",
        )
    )
    attention_weights.append(
        (
            # 第十一个路径：注意力权重的值投影矩阵的偏置参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.bias",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj_v.bias",
        )
    )
    attention_weights.append(
        (
            # 第十二个路径：注意力权重的输出密集层的权重参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.weight",
            # 对应的 PyTorch 路径
            f"stage{idx}.blocks.{cnt}.attn.proj.weight",
        )
    )
    # 将权重名称映射为在模型中的具体位置，以便后续在模型中加载预训练的权重
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.bias",  # CVT模型第 idx 阶段第 cnt 层的注意力输出层偏置
            f"stage{idx}.blocks.{cnt}.attn.proj.bias",  # 转换为对应的第 idx 阶段第 cnt 个块的注意力投影层偏置
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.weight",  # CVT模型第 idx 阶段第 cnt 层的中间层权重
            f"stage{idx}.blocks.{cnt}.mlp.fc1.weight"  # 转换为对应的第 idx 阶段第 cnt 个块的多层感知机（MLP）第一层权重
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.bias",  # CVT模型第 idx 阶段第 cnt 层的中间层偏置
            f"stage{idx}.blocks.{cnt}.mlp.fc1.bias"  # 转换为对应的第 idx 阶段第 cnt 个块的多层感知机（MLP）第一层偏置
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.weight",  # CVT模型第 idx 阶段第 cnt 层的输出层权重
            f"stage{idx}.blocks.{cnt}.mlp.fc2.weight"  # 转换为对应的第 idx 阶段第 cnt 个块的多层感知机（MLP）第二层权重
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.bias",  # CVT模型第 idx 阶段第 cnt 层的输出层偏置
            f"stage{idx}.blocks.{cnt}.mlp.fc2.bias"  # 转换为对应的第 idx 阶段第 cnt 个块的多层感知机（MLP）第二层偏置
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.weight",  # CVT模型第 idx 阶段第 cnt 层的归一化前权重
            f"stage{idx}.blocks.{cnt}.norm1.weight"  # 转换为对应的第 idx 阶段第 cnt 个块的归一化层1的权重
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.bias",  # CVT模型第 idx 阶段第 cnt 层的归一化前偏置
            f"stage{idx}.blocks.{cnt}.norm1.bias"  # 转换为对应的第 idx 阶段第 cnt 个块的归一化层1的偏置
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.weight",  # CVT模型第 idx 阶段第 cnt 层的归一化后权重
            f"stage{idx}.blocks.{cnt}.norm2.weight"  # 转换为对应的第 idx 阶段第 cnt 个块的归一化层2的权重
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.bias",  # CVT模型第 idx 阶段第 cnt 层的归一化后偏置
            f"stage{idx}.blocks.{cnt}.norm2.bias"  # 转换为对应的第 idx 阶段第 cnt 个块的归一化层2的偏置
        )
    )
    return attention_weights  # 返回所有权重的列表
# 定义一个函数，用于生成 cls_token 的重命名信息列表
def cls_token(idx):
    token = []
    token.append((f"cvt.encoder.stages.{idx}.cls_token", "stage2.cls_token"))
    return token

# 定义一个函数，用于生成 final 层的重命名信息列表
def final():
    head = []
    head.append(("layernorm.weight", "norm.weight"))
    head.append(("layernorm.bias", "norm.bias"))
    head.append(("classifier.weight", "head.weight"))
    head.append(("classifier.bias", "head.bias"))
    return head

# 定义一个函数，将 Microsoft CVT 模型转换为 Huggingface 模型的检查点
def convert_cvt_checkpoint(cvt_model, image_size, cvt_file_name, pytorch_dump_folder):
    # 定义与 ImageNet 类别对应的标签文件
    img_labels_file = "imagenet-1k-id2label.json"
    num_labels = 1000

    # 下载并加载 ImageNet 类别到标签的映射关系
    repo_id = "huggingface/label-files"
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, img_labels_file, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    # 根据 id2label 创建 label2id 的反向映射
    label2id = {v: k for k, v in id2label.items()}

    # 创建 CVT 模型的配置对象
    config = CvtConfig(num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 根据 CVT 模型的命名规则设置不同的深度参数
    if cvt_model.rsplit("/", 1)[-1][4:6] == "13":
        config.depth = [1, 2, 10]
    elif cvt_model.rsplit("/", 1)[-1][4:6] == "21":
        config.depth = [1, 4, 16]
    else:
        config.depth = [2, 2, 20]
        config.num_heads = [3, 12, 16]
        config.embed_dim = [192, 768, 1024]

    # 创建 CVT 图像分类模型
    model = CvtForImageClassification(config)
    # 从预训练模型中加载图像处理器
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
    image_processor.size["shortest_edge"] = image_size

    # 加载原始 CVT 模型的权重
    original_weights = torch.load(cvt_file_name, map_location=torch.device("cpu"))

    # 创建一个空的 OrderedDict 存储 Huggingface 格式的权重
    huggingface_weights = OrderedDict()
    list_of_state_dict = []

    # 遍历每个阶段的深度，并根据不同的条件添加对应的模型参数重命名信息
    for idx in range(len(config.depth)):
        if config.cls_token[idx]:
            list_of_state_dict = list_of_state_dict + cls_token(idx)
        list_of_state_dict = list_of_state_dict + embeddings(idx)
        for cnt in range(config.depth[idx]):
            list_of_state_dict = list_of_state_dict + attention(idx, cnt)

    # 添加 final 层的重命名信息
    list_of_state_dict = list_of_state_dict + final()

    # 根据重命名信息，将原始权重映射到 Huggingface 格式
    for i in range(len(list_of_state_dict)):
        huggingface_weights[list_of_state_dict[i][0]] = original_weights[list_of_state_dict[i][1]]

    # 加载映射后的权重到模型
    model.load_state_dict(huggingface_weights)

    # 将模型保存为 Huggingface 格式的预训练模型
    model.save_pretrained(pytorch_dump_folder)
    image_processor.save_pretrained(pytorch_dump_folder)
    parser.add_argument(
        "--cvt_model",  # 定义一个命令行参数 `--cvt_model`，用于指定要转换的 CVT 模型名称
        default="cvt-w24",  # 默认参数为 "cvt-w24"
        type=str,  # 参数类型为字符串
        help="Name of the cvt model you'd like to convert."  # 帮助信息，说明此参数是用于指定要转换的 CVT 模型的名称
    )
    parser.add_argument(
        "--image_size",  # 定义一个命令行参数 `--image_size`，用于指定输入图像的尺寸
        default=384,  # 默认参数为 384
        type=int,  # 参数类型为整数
        help="Input Image Size"  # 帮助信息，说明此参数是用于指定输入图像的尺寸
    )
    parser.add_argument(
        "--cvt_file_name",  # 定义一个命令行参数 `--cvt_file_name`，用于指定 CVT 模型文件的路径和名称
        default=r"cvtmodels\CvT-w24-384x384-IN-22k.pth",  # 默认参数为指定的文件路径
        type=str,  # 参数类型为字符串
        help="Input Image Size"  # 帮助信息，应为 "Input CVT model file path and name"
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # 定义一个命令行参数 `--pytorch_dump_folder_path`，用于指定输出 PyTorch 模型的目录路径
        default=None,  # 默认参数为 None，即未指定输出目录
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 帮助信息，说明此参数是用于指定输出 PyTorch 模型的目录路径
    )

    args = parser.parse_args()  # 解析命令行参数，并将结果存储在 `args` 变量中
    convert_cvt_checkpoint(args.cvt_model, args.image_size, args.cvt_file_name, args.pytorch_dump_folder_path)  # 调用函数 `convert_cvt_checkpoint`，传入解析后的参数作为参数
```