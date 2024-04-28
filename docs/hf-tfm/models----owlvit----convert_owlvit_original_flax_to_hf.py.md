# `.\transformers\models\owlvit\convert_owlvit_original_flax_to_hf.py`

```
# 设置文件编码
# 版权信息
# 引入需要的模块
# 导入命令行参数解析模块
# 导入collections模块
# 导入jax模块
# 导入jax.numpy模块
# 导入torch模块
# 导入torch.nn模块
# 从clip模块中导入CLIP类
# 从flax.training模块中导入checkpoints模块
# 从huggingface_hub模块中导入Repository类
# 从transformers模块中导入CLIPTokenizer类
# 从transformers模块中导入OwlViTConfig类
# 从transformers模块中导入OwlViTForObjectDetection类
# 从transformers模块中导入OwlViTImageProcessor类
# 从transformers模块中导入OwlViTModel类
# 从transformers模块中导入OwlViTProcessor类
# 定义全局配置字典
# 将嵌套字典展开为平铺字典
# 将参数中的浮点数转为32位浮点数
# 复制注意力层
    # 设置自注意力层的查询投影矩阵的权重数据
    hf_attn_layer.q_proj.weight.data = q_proj
    # 设置自注意力层的查询投影矩阵的偏置数据
    hf_attn_layer.q_proj.bias.data = q_proj_bias
    
    # 设置自注意力层的键投影矩阵的权重数据
    hf_attn_layer.k_proj.weight.data = k_proj
    # 设置自注意力层的键投影矩阵的偏置数据
    hf_attn_layer.k_proj.bias.data = k_proj_bias
    
    # 设置自注意力层的值投影矩阵的权重数据
    hf_attn_layer.v_proj.weight.data = v_proj
    # 设置自注意力层的值投影矩阵的偏置数据
    hf_attn_layer.v_proj.bias.data = v_proj_bias
    
    # 设置自注意力层的输出投影矩阵的权重数据
    hf_attn_layer.out_proj.weight = out_proj_weights
    # 设置自注意力层的输出投影矩阵的偏置数据
    hf_attn_layer.out_proj.bias = out_proj_bias
def copy_mlp(hf_mlp, pt_mlp):
    # 复制 PyTorch 模型的第一个全连接层到 Hugging Face 模型的第一个全连接层
    copy_linear(hf_mlp.fc1, pt_mlp.c_fc)
    # 复制 PyTorch 模型的第二个全连接层到 Hugging Face 模型的第二个全连接层
    copy_linear(hf_mlp.fc2, pt_mlp.c_proj)


def copy_linear(hf_linear, pt_linear):
    # 复制 PyTorch 线性层的权重到 Hugging Face 线性层的权重
    hf_linear.weight = pt_linear.weight
    # 复制 PyTorch 线性层的偏置到 Hugging Face 线性层的偏置
    hf_linear.bias = pt_linear.bias


def copy_layer(hf_layer, pt_layer):
    # 复制层归一化
    copy_linear(hf_layer.layer_norm1, pt_layer.ln_1)
    copy_linear(hf_layer.layer_norm2, pt_layer.ln_2)

    # 复制 MLP
    copy_mlp(hf_layer.mlp, pt_layer.mlp)

    # 复制注意力机制
    copy_attn_layer(hf_layer.self_attn, pt_layer.attn)


def copy_layers(hf_layers, pt_layers):
    # 遍历每个层并复制
    for hf_layer, pt_layer in zip(hf_layers, pt_layers):
        copy_layer(hf_layer, pt_layer)


def copy_encoder(hf_encoder, pt_model):
    # 复制嵌入
    hf_encoder.embeddings.token_embedding.weight = pt_model.token_embedding.weight
    hf_encoder.embeddings.position_embedding.weight.data = pt_model.positional_embedding

    # 复制层归一化
    copy_linear(hf_encoder.final_layer_norm, pt_model.ln_final)

    # 复制隐藏层
    copy_layers(hf_encoder.encoder.layers, pt_model.transformer.resblocks)


def copy_text_model_and_projection(hf_model, pt_model):
    # 复制投影
    hf_model.text_projection.weight.data = pt_model.text_projection.data.T

    # 复制文本编码器
    copy_encoder(hf_model.text_model, pt_model)


def copy_vision_model_and_projection(hf_model, pt_model):
    # 复制投影
    hf_model.visual_projection.weight.data = pt_model.visual.proj.data.T

    # 复制层归一化
    copy_linear(hf_model.vision_model.pre_layernorm, pt_model.visual.ln_pre)
    copy_linear(hf_model.vision_model.post_layernorm, pt_model.visual.ln_post)

    # 复制嵌入
    hf_model.vision_model.embeddings.patch_embedding.weight.data = pt_model.visual.conv1.weight.data
    hf_model.vision_model.embeddings.class_embedding = pt_model.visual.class_embedding
    hf_model.vision_model.embeddings.position_embedding.weight.data = pt_model.visual.positional_embedding.data

    # 复制编码器
    copy_layers(hf_model.vision_model.encoder.layers, pt_model.visual.transformer.resblocks)


def copy_class_merge_token(hf_model, flax_params):
    # 将 Flax 参数中合并的类标记参数复制到 Hugging Face 模型的层归一化中
    flax_class_token_params = flatten_nested_dict(flax_params["backbone"]["merged_class_token"])

    weight = torch.from_numpy(flax_class_token_params["scale"])
    bias = torch.from_numpy(flax_class_token_params["bias"])
    hf_model.layer_norm.weight = nn.Parameter(weight)
    hf_model.layer_norm.bias = nn.Parameter(bias)


def copy_class_box_heads(hf_model, flax_params):
    pt_params = hf_model.state_dict()
    new_params = {}

    # 将类别预测头 Flax 参数重命名为 PyTorch HF 参数
    flax_class_params = flatten_nested_dict(flax_params["class_head"])
    # 遍历flax_class_params字典，获取键和值
    for flax_key, v in flax_class_params.items():
        # 将flax_key中的"/"替换为"."，更新torch_key
        torch_key = flax_key.replace("/", ".")
        # 将torch_key中".kernel"替换为".weight"
        torch_key = torch_key.replace(".kernel", ".weight")
        # 将torch_key中"Dense_0"替换为"dense0"
        torch_key = torch_key.replace("Dense_0", "dense0")
        # 在torch_key前面添加"class_head."
        torch_key = "class_head." + torch_key

        # 如果torch_key包含"weight"且v的维度为2
        if "weight" in torch_key and v.ndim == 2:
            # 转置v
            v = v.T

        # 将torch.from_numpy(v)包装成nn.Parameter，更新new_params字典
        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))

    # 重命名box prediction box flax params为pytorch HF
    # 将flax_params["obj_box_head"]中的内容扁平化为flax_box_params字典
    flax_box_params = flatten_nested_dict(flax_params["obj_box_head"])

    # 遍历flax_box_params字典，获取键和值
    for flax_key, v in flax_box_params.items():
        # 将flax_key中的"/"替换为"."，更新torch_key
        torch_key = flax_key.replace("/", ".")
        # 将torch_key中".kernel"替换为".weight"
        torch_key = torch_key.replace(".kernel", ".weight")
        # 将torch_key中"_"替换为空字符串，并转换为小写
        torch_key = torch_key.replace("_", "").lower()
        # 在torch_key前面添加"box_head."
        torch_key = "box_head." + torch_key

        # 如果torch_key包含"weight"且v的维度为2
        if "weight" in torch_key and v.ndim == 2:
            # 转置v
            v = v.T

        # 将torch.from_numpy(v)包装成nn.Parameter，更新new_params字典
        new_params[torch_key] = nn.Parameter(torch.from_numpy(v))

    # 将flax params复制到PyTorch params
    # 遍历new_params字典中的项
    for name, param in new_params.items():
        # 如果name在pt_params的键中
        if name in pt_params.keys():
            # 将param的值复制给pt_params中对应的项
            pt_params[name].copy_(param)
# 将 Flax CLIP 中的注意力参数复制到 HF PyTorch 参数中
def copy_flax_attn_params(hf_backbone, flax_attn_params):
    # 遍历 Flax CLIP 注意力参数字典
    for k, v in flax_attn_params.items():
        # 根据键名判断参数来源，生成对应的 PyTorch 键名
        if k.startswith("transformer"):
            torch_key = k.replace("transformer.resblocks", "text_model.encoder.layers")
        else:
            torch_key = k.replace("visual.transformer.resblocks", "vision_model.encoder.layers")

        # 替换部分键名中的字符串表示不同的参数
        torch_key = torch_key.replace("attn", "self_attn")
        torch_key = torch_key.replace("key", "k_proj")
        torch_key = torch_key.replace("value", "v_proj")
        torch_key = torch_key.replace("query", "q_proj")
        torch_key = torch_key.replace("out", "out_proj")

        # 根据键名和维度信息对值做相应处理
        if "bias" in torch_key and v.ndim == 2:
            shape = v.shape[0] * v.shape[1]
            v = v.reshape(shape)

        if "weight" in torch_key and "out" in torch_key:
            shape = (v.shape[0] * v.shape[1], v.shape[2])
            v = v.reshape(shape).T

        if "weight" in torch_key and "out" not in torch_key:
            shape = (v.shape[0], v.shape[1] * v.shape[2])
            v = v.reshape(shape).T

        # 将值转换为 PyTorch 张量，并复制到 HF 模型参数中
        v = torch.from_numpy(v)
        hf_backbone.state_dict()[torch_key].copy_(v)


# 转换 Flax CLIP 中的注意力层参数结构
def _convert_attn_layers(params):
    new_params = {}
    processed_attn_layers = []

    # 遍历参数字典，处理注意力层参数
    for k, v in params.items():
        if "attn." in k:
            base = k[: k.rindex("attn.") + 5]
            if base in processed_attn_layers:
                continue

            processed_attn_layers.append(base)
            dim = params[base + "out.weight"].shape[-1]
            # 调整权重矩阵的形状，并转置后放入新参数字典中
            new_params[base + "out_proj.weight"] = params[base + "out.weight"].reshape(dim, dim).T
            new_params[base + "out_proj.bias"] = params[base + "out.bias"]
        else:
            new_params[k] = v
    return new_params


# 将 Flax CLIP 参数转换为 HF PyTorch 参数
def convert_clip_backbone(flax_params, torch_config):
    # 实例化 HF PyTorch 的 CLIP 模型，并获取其模型参数
    torch_model = CLIP(**torch_config)
    torch_model.eval()
    torch_clip_params = torch_model.state_dict()

    # 展平 Flax CLIP 参数字典
    flax_clip_params = flatten_nested_dict(flax_params["backbone"]["clip"])
    new_torch_params = {}
    # 遍历 Flax CLIP 参数字典中的键值对
    for flax_key, v in flax_clip_params.items():
        # 将 Flax 参数键名中的斜杠替换为点号，用于匹配 PyTorch 参数键名
        torch_key = flax_key.replace("/", ".")
        # 将特定字符串替换，以匹配 PyTorch 参数键名的格式
        torch_key = torch_key.replace("text.token_embedding.embedding", "token_embedding.kernel")

        # 检查是否需要剔除特定前缀
        if (
            torch_key.startswith("text.transformer")
            or torch_key.startswith("text.text_projection")
            or torch_key.startswith("text.ln_final")
            or torch_key.startswith("text.positional_embedding")
        ):
            # 剔除特定前缀
            torch_key = torch_key[5:]

        # 进行额外的字符串替换，以匹配 PyTorch 参数键名的格式
        torch_key = torch_key.replace("text_projection.kernel", "text_projection")
        torch_key = torch_key.replace("visual.proj.kernel", "visual.proj")
        torch_key = torch_key.replace(".scale", ".weight")
        torch_key = torch_key.replace(".kernel", ".weight")

        # 检查是否需要转置参数
        if "conv" in torch_key or "downsample.0.weight" in torch_key:
            # 转置参数的维度
            v = v.transpose(3, 2, 0, 1)

        # 检查是否需要转置参数，并且参数的维度为 2，且不是嵌入层的情况下
        elif "weight" in torch_key and v.ndim == 2 and "embedding" not in torch_key:
            # 转置参数
            v = v.T

        # 将转换后的参数添加到新的 PyTorch 参数字典中
        new_torch_params[torch_key] = v

    # 将注意力层参数转换为 PyTorch 参数
    attn_params = _convert_attn_layers(new_torch_params)
    # 更新新的 PyTorch 参数字典
    new_torch_params.update(attn_params)
    # 清空注意力层参数字典
    attn_params = {}

    # 将 Flax CLIP 的背景骨干参数复制到 PyTorch 参数中
    for name, param in new_torch_params.items():
        # 检查键名是否在 PyTorch 参数字典中
        if name in torch_clip_params.keys():
            # 将 Numpy 数组转换为 PyTorch 张量，并复制到对应的 PyTorch 参数中
            new_param = torch.from_numpy(new_torch_params[name])
            torch_clip_params[name].copy_(new_param)
        else:
            # 若键名不在 PyTorch 参数字典中，则添加到注意力参数字典中
            attn_params[name] = param

    # 返回 PyTorch 参数字典、PyTorch 模型和注意力参数字典
    return torch_clip_params, torch_model, attn_params
# 使用 torch.no_grad() 装饰器，确保在此函数内不计算梯度，提高性能
@torch.no_grad()
# 定义函数，将 OwlViT 模型的权重转换为 transformers 设计
def convert_owlvit_checkpoint(pt_backbone, flax_params, attn_params, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 使用 Repository 类，初始化仓库，克隆指定地址的仓库到本地，如果已存在，则进行更新
    repo = Repository(pytorch_dump_folder_path, clone_from=f"google/{pytorch_dump_folder_path}")
    repo.git_pull()

    # 如果提供了配置路径，则从提供的路径加载配置，否则使用默认配置
    if config_path is not None:
        config = OwlViTConfig.from_pretrained(config_path)
    else:
        config = OwlViTConfig()

    # 初始化 HF 版本的 OwlViT 模型的骨干部分，并设置为评估模式
    hf_backbone = OwlViTModel(config).eval()
    # 初始化 HF 版本的 OwlViT 模型，并设置为评估模式
    hf_model = OwlViTForObjectDetection(config).eval()

    # 复制文本模型和投影头的权重参数
    copy_text_model_and_projection(hf_backbone, pt_backbone)
    # 复制视觉模型和投影头的权重参数
    copy_vision_model_and_projection(hf_backbone, pt_backbone)
    # 复制 PyTorch 版本的 logit_scale 参数到 HF 版本
    hf_backbone.logit_scale = pt_backbone.logit_scale
    # 复制 Flax 版本的注意力参数到 HF 版本
    copy_flax_attn_params(hf_backbone, attn_params)

    # 将 HF 版本的骨干部分设置为 HF 版本的 OwlViT 模型的骨干
    hf_model.owlvit = hf_backbone
    # 复制 Flax 版本的 class_merge_token 参数到 HF 版本
    copy_class_merge_token(hf_model, flax_params)
    # 复制 Flax 版本的 class_box_heads 参数到 HF 版本
    copy_class_box_heads(hf_model, flax_params)

    # 保存 HF 版本的模型到指定路径
    hf_model.save_pretrained(repo.local_dir)

    # 初始化图像处理器，设置图像大小和裁剪大小
    image_processor = OwlViTImageProcessor(
        size=config.vision_config.image_size, crop_size=config.vision_config.image_size
    )
    # 初始化 tokenizer，使用指定的预训练模型和 pad_token，设置最大模型长度为 16
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", pad_token="!", model_max_length=16)

    # 初始化 processor，包括图像处理器和 tokenizer
    processor = OwlViTProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(repo.local_dir)
    # 将 processor 保存到指定路径
    processor.save_pretrained(repo.local_dir)

    # 将本地变更添加到仓库
    repo.git_add()
    # 提交仓库变更，添加提交信息
    repo.git_commit("Upload model and processor")
    # 推送本地变更到远程仓库

if __name__ == "__main__":
    # 初始化参数解析器
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument(
        "--owlvit_version",
        default=None,
        type=str,
        required=True,
        help="OWL-ViT model name [clip_b16, clip_b32, clip_l14].",
    )
    parser.add_argument(
        "--owlvit_checkpoint", default=None, type=str, required=True, help="Path to flax model checkpoint."
    )
    parser.add_argument("--hf_config", default=None, type=str, required=True, help="Path to HF model config.")
    parser.add_argument(
        "--pytorch_dump_folder_path", default="hf_model", type=str, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()

    # 初始化 PyToch 版本的 CLIP 模型
    model_name = args.owlvit_version
    if model_name == "clip_b16":
        torch_config = CONFIGS["vit_b16"]
    elif model_name == "clip_b32":
        torch_config = CONFIGS["vit_b32"]
    elif model_name == "clip_l14":
        torch_config = CONFIGS["vit_l14"]

    # 从检查点加载并将参数转换为 float-32
    variables = checkpoints.restore_checkpoint(args.owlvit_checkpoint, target=None)["optimizer"]["target"]
    flax_params = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, variables)
    del variables

    # 转换 CLIP 骨干部分
    # 调用函数convert_clip_backbone将flax模型参数转换为pt_backbone_params, clip_pt, attn_params
    pt_backbone_params, clip_pt, attn_params = convert_clip_backbone(flax_params, torch_config)
    
    # 调用函数convert_owlvit_checkpoint将clip_pt, flax_params, attn_params转换为pytorch模型检查点，并保存在指定路径下
    convert_owlvit_checkpoint(clip_pt, flax_params, attn_params, args.pytorch_dump_folder_path, args.hf_config)
```