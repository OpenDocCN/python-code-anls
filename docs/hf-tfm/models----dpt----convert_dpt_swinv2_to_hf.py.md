# `.\models\dpt\convert_dpt_swinv2_to_hf.py`

```
#python
# here we list all keys to be renamed (original name on the left, our name on the right)
def create_rename_keys(config):
    rename_keys = []

    # fmt: off
    # stem
    # 返回由 stem 处理的键值对列表
    rename_keys.extend([
    # 添加需要重命名的键值对到列表中，映射预训练模型的权重到新模型的对应位置
    rename_keys.append(("pretrained.model.patch_embed.proj.weight", "backbone.embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("pretrained.model.patch_embed.proj.bias", "backbone.embeddings.patch_embeddings.projection.bias"))
    rename_keys.append(("pretrained.model.patch_embed.norm.weight", "backbone.embeddings.norm.weight"))
    rename_keys.append(("pretrained.model.patch_embed.norm.bias", "backbone.embeddings.norm.bias"))

    # 转换器编码器部分
    # 注意：非转换器(backbone)如Swinv2、LeViT等不需要后处理激活（读取投影 + 调整块）
    
    # refinenet部分（此处比较棘手）
    # 设置映射关系，将refinenet的输出通道映射到融合阶段的层
    mapping = {1:3, 2:2, 3:1, 4:0}

    # 遍历映射关系，生成重命名的键值对，并添加到列表中
    for i in range(1, 5):
        j = mapping[i]
        rename_keys.append((f"scratch.refinenet{i}.out_conv.weight", f"neck.fusion_stage.layers.{j}.projection.weight"))
        rename_keys.append((f"scratch.refinenet{i}.out_conv.bias", f"neck.fusion_stage.layers.{j}.projection.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit1.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer1.convolution2.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv1.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution1.bias"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.weight", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.weight"))
        rename_keys.append((f"scratch.refinenet{i}.resConfUnit2.conv2.bias", f"neck.fusion_stage.layers.{j}.residual_layer2.convolution2.bias"))

    # scratch卷积部分
    # 生成重命名的键值对，将scratch层的权重映射到融合阶段的卷积层
    for i in range(4):
        rename_keys.append((f"scratch.layer{i+1}_rn.weight", f"neck.convs.{i}.weight"))

    # 头部部分
    # 生成重命名的键值对，将scratch的输出卷积权重映射到头部的权重
    for i in range(0, 5, 2):
        rename_keys.append((f"scratch.output_conv.{i}.weight", f"head.head.{i}.weight"))
        rename_keys.append((f"scratch.output_conv.{i}.bias", f"head.head.{i}.bias"))

    # 返回所有重命名后的键值对列表
    return rename_keys
# 从状态字典中移除指定的键列表
def remove_ignore_keys_(state_dict):
    ignore_keys = ["pretrained.model.head.weight", "pretrained.model.head.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)

# 读取每个编码器层的查询（queries）、键（keys）和值（values）矩阵
def read_in_q_k_v(state_dict, config, model):
    for i in range(len(config.backbone_config.depths)):
        for j in range(config.backbone_config.depths[i]):
            # 获取当前注意力层的全头尺寸
            dim = model.backbone.encoder.layers[i].blocks[j].attention.self.all_head_size
            # 读取输入投影层权重和偏置（在原始实现中，这是一个单独的矩阵加偏置）
            in_proj_weight = state_dict.pop(f"pretrained.model.layers.{i}.blocks.{j}.attn.qkv.weight")
            # 将查询（query）、键（key）、值（value）依次添加到状态字典中
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.query.weight"] = in_proj_weight[:dim, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.key.weight"] = in_proj_weight[dim: dim * 2, :]
            state_dict[f"backbone.encoder.layers.{i}.blocks.{j}.attention.self.value.weight"] = in_proj_weight[-dim:, :]

# 重命名字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

# 准备图像数据，从指定的 URL 获取图像
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

# 将模型的权重复制/粘贴/调整到我们的 DPT 结构中
@torch.no_grad()
def convert_dpt_checkpoint(model_name, pytorch_dump_folder_path, verify_logits, push_to_hub):
    """
    Copy/paste/tweak model's weights to our DPT structure.
    """
    # 定义模型名称到 URL 的映射
    name_to_url = {
        "dpt-swinv2-tiny-256": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt",
        "dpt-swinv2-base-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt",
        "dpt-swinv2-large-384": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt",
    }
    
    # 根据模型名称获取检查点 URL
    checkpoint_url = name_to_url[model_name]
    # 根据 URL 获取 DPT 配置和图像大小
    config, image_size = get_dpt_config(model_name)
    # 从 URL 加载原始状态字典
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")
    
    # 加载 HuggingFace 模型
    model = DPTForDepthEstimation(config)
    
    # 移除特定的键
    remove_ignore_keys_(state_dict)
    # 创建键重命名映射
    rename_keys = create_rename_keys(config)
    # 对每对源键和目标键执行重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取 QKV 矩阵
    read_in_q_k_v(state_dict, config, model)
    
    # 使用非严格模式加载模型状态字典，并获取缺失和意外的键列表
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    # 将模型设置为评估模式
    model.eval()
    
    # 在图像上验证输出结果
    # 创建一个 DPTImageProcessor 对象，设置图像大小为指定的 image_size
    processor = DPTImageProcessor(size={"height": image_size, "width": image_size})

    # 准备图像数据
    image = prepare_img()
    # 使用 processor 对象处理图像数据，返回 PyTorch 张量格式
    processor(image, return_tensors="pt")

    # 如果需要验证 logits
    if verify_logits:
        # 导入必要的库和模块
        from torchvision import transforms
        
        # 从网络下载并打开指定 URL 的图像
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)

        # 定义图像转换操作序列
        transforms = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),  # 调整图像大小为指定尺寸
                transforms.ToTensor(),  # 转换图像为 PyTorch 张量格式
            ]
        )
        # 对图像进行转换操作
        pixel_values = transforms(image).unsqueeze(0)

        # 执行模型的前向传播
        with torch.no_grad():
            outputs = model(pixel_values)

        # 获取预测的深度图
        predicted_depth = outputs.predicted_depth

        # 打印预测深度图的形状信息
        print("Shape of predicted depth:", predicted_depth.shape)
        # 打印预测深度图的前三行三列数据
        print("First values of predicted depth:", predicted_depth[0, :3, :3])

        # 根据模型名称验证预测深度图的形状和部分切片值
        if model_name == "dpt-swinv2-base-384":
            # 确认预期形状和切片值（已验证）
            expected_shape = torch.Size([1, 384, 384])
            expected_slice = torch.tensor(
                [
                    [1998.5575, 1997.3887, 2009.2981],
                    [1952.8607, 1979.6488, 2001.0854],
                    [1953.7697, 1961.7711, 1968.8904],
                ],
            )
        elif model_name == "dpt-swinv2-tiny-256":
            # 确认预期形状和切片值（已验证）
            expected_shape = torch.Size([1, 256, 256])
            expected_slice = torch.tensor(
                [[978.9163, 976.5215, 978.5349], [974.1859, 971.7249, 975.8046], [971.3419, 970.3118, 971.6830]],
            )
        elif model_name == "dpt-swinv2-large-384":
            # 确认预期形状和切片值（已验证）
            expected_shape = torch.Size([1, 384, 384])
            expected_slice = torch.tensor(
                [
                    [1203.7206, 1200.1495, 1197.8234],
                    [1196.2484, 1183.5033, 1186.4640],
                    [1178.8131, 1182.3260, 1174.3975],
                ],
            )

        # 使用断言确认预测深度图的形状和切片值与期望相符
        assert predicted_depth.shape == torch.Size(expected_shape)
        assert torch.allclose(predicted_depth[0, :3, :3], expected_slice)
        # 打印确认信息
        print("Looks ok!")

    # 如果指定了 pytorch_dump_folder_path，则保存模型和处理器
    if pytorch_dump_folder_path is not None:
        # 创建目录（如果不存在）
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 打印保存模型和处理器的信息
        print(f"Saving model and processor to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型和处理器推送到 Hub
    if push_to_hub:
        # 打印推送模型和处理器到 Hub 的信息
        print("Pushing model and processor to hub...")
        # 推送模型到指定的 Hub 仓库
        model.push_to_hub(repo_id=f"Intel/{model_name}")
        # 推送处理器到指定的 Hub 仓库
        processor.push_to_hub(repo_id=f"Intel/{model_name}")
if __name__ == "__main__":
    # 如果当前脚本作为主程序运行，则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # 必选参数
    parser.add_argument(
        "--model_name",
        default="dpt-swinv2-base-384",
        type=str,
        choices=["dpt-swinv2-tiny-256", "dpt-swinv2-base-384", "dpt-swinv2-large-384"],
        help="Name of the model you'd like to convert.",
    )
    # 添加名为--model_name的参数，指定默认值为"dpt-swinv2-base-384"，类型为字符串
    # 可选值为指定的三种模型名称，用于选择要转换的模型

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    # 添加名为--pytorch_dump_folder_path的参数，指定默认值为None，类型为字符串
    # 用于指定输出的PyTorch模型存储目录的路径

    parser.add_argument(
        "--verify_logits",
        action="store_true",
        help="Whether to verify logits after conversion.",
    )
    # 添加名为--verify_logits的参数，当命令行中有该选项时，设置为True
    # 用于指定在转换后是否验证logits（输出层未归一化的概率分布）

    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model to the hub after conversion.",
    )
    # 添加名为--push_to_hub的参数，当命令行中有该选项时，设置为True
    # 用于指定是否在转换后将模型推送到hub（模型分享和托管服务）

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在args变量中

    convert_dpt_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.verify_logits, args.push_to_hub)
    # 调用函数convert_dpt_checkpoint，传入解析得到的参数
```