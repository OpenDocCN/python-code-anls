# `.\models\deformable_detr\convert_deformable_detr_to_pytorch.py`

```
# 从状态字典中重命名键，根据特定规则进行替换
def rename_key(orig_key):
    if "backbone.0.body" in orig_key:
        orig_key = orig_key.replace("backbone.0.body", "backbone.conv_encoder.model")
    if "transformer" in orig_key:
        orig_key = orig_key.replace("transformer.", "")
    if "norm1" in orig_key:
        # 根据上下文替换层次规范化的键名，区分编码器和解码器的情况
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm1", "self_attn_layer_norm")
        else:
            orig_key = orig_key.replace("norm1", "encoder_attn_layer_norm")
    if "norm2" in orig_key:
        # 根据上下文替换层次规范化的键名，区分编码器和解码器的情况
        if "encoder" in orig_key:
            orig_key = orig_key.replace("norm2", "final_layer_norm")
        else:
            orig_key = orig_key.replace("norm2", "self_attn_layer_norm")
    if "norm3" in orig_key:
        # 替换最终层次规范化的键名
        orig_key = orig_key.replace("norm3", "final_layer_norm")
    if "linear1" in orig_key:
        # 替换第一个线性层的键名
        orig_key = orig_key.replace("linear1", "fc1")
    if "linear2" in orig_key:
        # 替换第二个线性层的键名
        orig_key = orig_key.replace("linear2", "fc2")
    if "query_embed" in orig_key:
        # 替换查询位置嵌入的键名
        orig_key = orig_key.replace("query_embed", "query_position_embeddings")
    if "cross_attn" in orig_key:
        # 替换交叉注意力的键名
        orig_key = orig_key.replace("cross_attn", "encoder_attn")

    return orig_key


# 从状态字典中读取查询、键和值
def read_in_q_k_v(state_dict):
    # 循环遍历范围为0到5，共6次，处理每个自注意力层的权重和偏置
    for i in range(6):
        # 从状态字典中弹出当前自注意力层输入投影层的权重和偏置
        in_proj_weight = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"decoder.layers.{i}.self_attn.in_proj_bias")
        
        # 将权重切片分配给查询、键和值投影层的权重
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        # 将偏置切片分配给查询投影层的偏置
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        # 将权重切片分配给键投影层的权重
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        # 将偏置切片分配给键投影层的偏置
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        # 将权重切片分配给值投影层的权重
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        # 将偏置切片分配给值投影层的偏置
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片的 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过请求获取图片的原始二进制数据流，并用 PIL 打开这个图片
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_deformable_detr_checkpoint(
    checkpoint_path,
    single_scale,
    dilation,
    with_box_refine,
    two_stage,
    pytorch_dump_folder_path,
    push_to_hub,
):
    """
    复制/粘贴/调整模型的权重以适应我们的 Deformable DETR 结构。
    """

    # 加载默认配置
    config = DeformableDetrConfig()
    # 设置配置属性
    if single_scale:
        config.num_feature_levels = 1  # 设置特征层级数为1
    config.dilation = dilation  # 设置膨胀参数
    config.with_box_refine = with_box_refine  # 设置是否进行框调整
    config.two_stage = two_stage  # 设置是否为两阶段模型
    # 设置标签数目
    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    # 从 HuggingFace Hub 下载并加载 COCO 检测标签映射文件
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, filename, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label  # 设置 ID 到标签的映射
    config.label2id = {v: k for k, v in id2label.items()}  # 设置标签到 ID 的映射

    # 加载图像处理器
    image_processor = DeformableDetrImageProcessor(format="coco_detection")

    # 准备图片
    img = prepare_img()  # 调用准备图片函数获取图片对象
    encoding = image_processor(images=img, return_tensors="pt")  # 对图片进行编码处理
    pixel_values = encoding["pixel_values"]  # 获取像素数值

    logger.info("Converting model...")  # 记录日志，表示正在转换模型

    # 加载原始的状态字典
    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    # 重命名键名
    for key in state_dict.copy().keys():
        val = state_dict.pop(key)
        state_dict[rename_key(key)] = val
    # 查询、键、值矩阵需要特殊处理
    read_in_q_k_v(state_dict)
    # 重要：需要在每个基础模型键名前添加前缀，因为头部模型使用不同的属性
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_embed"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    # 最后，创建 HuggingFace 模型并加载状态字典
    model = DeformableDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"  # 检测设备是否支持 CUDA
    model.to(device)  # 将模型移动到指定设备
    # 验证转换结果
    outputs = model(pixel_values.to(device))

    expected_logits = torch.tensor(
        [[-9.6645, -4.3449, -5.8705], [-9.7035, -3.8504, -5.0724], [-10.5634, -5.3379, -7.5116]]
    )
    expected_boxes = torch.tensor([[0.8693, 0.2289, 0.2492], [0.3150, 0.5489, 0.5845], [0.5563, 0.7580, 0.8518]])

    if single_scale:
        expected_logits = torch.tensor(
            [[-9.9051, -4.2541, -6.4852], [-9.6947, -4.0854, -6.8033], [-10.0665, -5.8470, -7.7003]]
        )
        expected_boxes = torch.tensor([[0.7292, 0.4991, 0.5532], [0.7959, 0.2426, 0.4236], [0.7582, 0.3518, 0.4451]])
    # 如果选择了单尺度和扩张操作，则设定预期的分类 logits 和边界框
    if single_scale and dilation:
        expected_logits = torch.tensor(
            [[-8.9652, -4.1074, -5.6635], [-9.0596, -4.9447, -6.6075], [-10.1178, -4.5275, -6.2671]]
        )
        expected_boxes = torch.tensor([[0.7665, 0.4130, 0.4769], [0.8364, 0.1841, 0.3391], [0.6261, 0.3895, 0.7978]])

    # 如果需要进行边界框细化，则设定预期的分类 logits 和边界框
    if with_box_refine:
        expected_logits = torch.tensor(
            [[-8.8895, -5.4187, -6.8153], [-8.4706, -6.1668, -7.6184], [-9.0042, -5.5359, -6.9141]]
        )
        expected_boxes = torch.tensor([[0.7828, 0.2208, 0.4323], [0.0892, 0.5996, 0.1319], [0.5524, 0.6389, 0.8914]])

    # 如果同时需要边界框细化和两阶段操作，则设定预期的分类 logits 和边界框
    if with_box_refine and two_stage:
        expected_logits = torch.tensor(
            [[-6.7108, -4.3213, -6.3777], [-8.9014, -6.1799, -6.7240], [-6.9315, -4.4735, -6.2298]]
        )
        expected_boxes = torch.tensor([[0.2583, 0.5499, 0.4683], [0.7652, 0.9068, 0.4882], [0.5490, 0.2763, 0.0564]])

    # 打印模型输出的前三行三列的 logits
    print("Logits:", outputs.logits[0, :3, :3])

    # 断言模型输出的前三行三列的 logits 和预期的 logits 在给定的误差范围内相似
    assert torch.allclose(outputs.logits[0, :3, :3], expected_logits.to(device), atol=1e-4)
    # 断言模型输出的前三行三列的预测边界框和预期的边界框在给定的误差范围内相似
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_boxes.to(device), atol=1e-4)

    # 打印信息，表明一切正常
    print("Everything ok!")

    # 保存 PyTorch 模型和图像处理器到指定路径
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    # 确保保存模型和处理器的文件夹存在
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 调用模型的保存方法和图像处理器的保存方法
    model.save_pretrained(pytorch_dump_folder_path)
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 Hub 上，则进行相应操作
    if push_to_hub:
        # 构造模型的名称，根据选择的参数添加后缀
        model_name = "deformable-detr"
        model_name += "-single-scale" if single_scale else ""
        model_name += "-dc5" if dilation else ""
        model_name += "-with-box-refine" if with_box_refine else ""
        model_name += "-two-stage" if two_stage else ""
        # 打印提示信息，表明正在将模型推送到 Hub 上
        print("Pushing model to hub...")
        # 调用模型对象的推送到 Hub 的方法
        model.push_to_hub(repo_path_or_name=model_name, organization="nielsr", commit_message="Add model")
# 如果当前脚本作为主程序执行（而不是被导入为模块），则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数：checkpoint_path，用于指定 PyTorch checkpoint 文件的路径
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/niels/checkpoints/deformable_detr/r50_deformable_detr-checkpoint.pth",
        help="Path to Pytorch checkpoint (.pth file) you'd like to convert.",
    )

    # 添加命令行参数：single_scale，设置为 True 则设置 config.num_features_levels = 1
    parser.add_argument("--single_scale", action="store_true", help="Whether to set config.num_features_levels = 1.")

    # 添加命令行参数：dilation，设置为 True 则设置 config.dilation=True
    parser.add_argument("--dilation", action="store_true", help="Whether to set config.dilation=True.")

    # 添加命令行参数：with_box_refine，设置为 True 则设置 config.with_box_refine=True
    parser.add_argument("--with_box_refine", action="store_true", help="Whether to set config.with_box_refine=True.")

    # 添加命令行参数：two_stage，设置为 True 则设置 config.two_stage=True
    parser.add_argument("--two_stage", action="store_true", help="Whether to set config.two_stage=True.")

    # 添加命令行参数：pytorch_dump_folder_path，必需的参数，指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the folder to output PyTorch model.",
    )

    # 添加命令行参数：push_to_hub，设置为 True 则表示要将转换后的模型推送到 🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析命令行参数并将其保存到 args 变量中
    args = parser.parse_args()

    # 调用函数 convert_deformable_detr_checkpoint，并传入命令行参数中的相应值
    convert_deformable_detr_checkpoint(
        args.checkpoint_path,
        args.single_scale,
        args.dilation,
        args.with_box_refine,
        args.two_stage,
        args.pytorch_dump_folder_path,
        args.push_to_hub,
    )
```