# `.\models\poolformer\convert_poolformer_original_to_pytorch.py`

```py
# 设置日志输出级别为INFO，确保日志在运行时能够显示相关信息
logging.set_verbosity_info()

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个函数，用于替换模型权重字典中的键名，通过减去指定偏移量来实现
def replace_key_with_offset(key, offset, original_name, new_name):
    """
    Replaces the key by subtracting the offset from the original layer number
    
    Args:
        key (str): 需要替换的键名
        offset (int): 偏移量，用于计算新的块号
        original_name (str): 原始层名称，用于定位需要替换的部分
        new_name (str): 新的层名称，用于替换原始层名称
        
    Returns:
        str: 替换后的新键名
    """
    # 根据原始层名称获取需要替换的块号
    to_find = original_name.split(".")[0]
    key_list = key.split(".")
    orig_block_num = int(key_list[key_list.index(to_find) - 2])
    layer_num = int(key_list[key_list.index(to_find) - 1])
    
    # 计算新的块号
    new_block_num = orig_block_num - offset

    # 构建新的键名并进行替换
    key = key.replace(f"{orig_block_num}.{layer_num}.{original_name}", 
                      f"block.{new_block_num}.{layer_num}.{new_name}")
    return key


def rename_keys(state_dict):
    # 使用有序字典保存新的状态字典
    new_state_dict = OrderedDict()
    # 初始化嵌入层的计数和补丁嵌入偏移量
    total_embed_found, patch_emb_offset = 0, 0
    # 遍历给定状态字典中的键值对
    for key, value in state_dict.items():
        # 如果键以"network"开头，替换为"poolformer.encoder"
        if key.startswith("network"):
            key = key.replace("network", "poolformer.encoder")
        
        # 如果键包含"proj"，处理第一个嵌入和内部嵌入层的偏置项
        if "proj" in key:
            # 如果键以"bias"结尾且不包含"patch_embed"，增加嵌入偏置的偏移量
            if key.endswith("bias") and "patch_embed" not in key:
                patch_emb_offset += 1
            
            # 替换"proj"之前的部分为"patch_embeddings.{total_embed_found}."，
            # 并将"proj"替换为"projection"
            to_replace = key[: key.find("proj")]
            key = key.replace(to_replace, f"patch_embeddings.{total_embed_found}.")
            key = key.replace("proj", "projection")
            
            # 如果键以"bias"结尾，增加已找到的嵌入总数
            if key.endswith("bias"):
                total_embed_found += 1
        
        # 如果键包含"patch_embeddings"，在键前面添加"poolformer.encoder."
        if "patch_embeddings" in key:
            key = "poolformer.encoder." + key
        
        # 如果键包含"mlp.fc1"，调用函数替换键名，处理偏置项偏移
        if "mlp.fc1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc1", "output.conv1")
        
        # 如果键包含"mlp.fc2"，调用函数替换键名，处理偏置项偏移
        if "mlp.fc2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "mlp.fc2", "output.conv2")
        
        # 如果键包含"norm1"，调用函数替换键名，处理偏置项偏移
        if "norm1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "norm1", "before_norm")
        
        # 如果键包含"norm2"，调用函数替换键名，处理偏置项偏移
        if "norm2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "norm2", "after_norm")
        
        # 如果键为"layer_scale_1"，调用函数替换键名，处理偏置项偏移
        if "layer_scale_1" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale_1", "layer_scale_1")
        
        # 如果键为"layer_scale_2"，调用函数替换键名，处理偏置项偏移
        if "layer_scale_2" in key:
            key = replace_key_with_offset(key, patch_emb_offset, "layer_scale_2", "layer_scale_2")
        
        # 如果键包含"head"，将"head"替换为"classifier"
        if "head" in key:
            key = key.replace("head", "classifier")
        
        # 将处理后的新键值对存入新的状态字典中
        new_state_dict[key] = value
    
    # 返回处理后的新状态字典
    return new_state_dict
# We will verify our results on a COCO image
def prepare_img():
    # 定义 COCO 图像的 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 库获取图像的原始字节流，并用 PIL 库打开图像
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_poolformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PoolFormer structure.
    """

    # load default PoolFormer configuration
    config = PoolFormerConfig()

    # set attributes based on model_name
    repo_id = "huggingface/label-files"
    # 从模型名字中提取尺寸信息
    size = model_name[-3:]
    config.num_labels = 1000
    filename = "imagenet-1k-id2label.json"
    expected_shape = (1, 1000)

    # set config attributes
    # 从 HuggingFace Hub 下载并加载 id 到 label 的映射
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    
    # 根据模型尺寸设置不同的配置参数
    if size == "s12":
        config.depths = [2, 2, 6, 2]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pct = 0.9
    elif size == "s24":
        config.depths = [4, 4, 12, 4]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        crop_pct = 0.9
    elif size == "s36":
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [64, 128, 320, 512]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pct = 0.9
    elif size == "m36":
        config.depths = [6, 6, 18, 6]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pct = 0.95
    elif size == "m48":
        config.depths = [8, 8, 24, 8]
        config.hidden_sizes = [96, 192, 384, 768]
        config.mlp_ratio = 4.0
        config.layer_scale_init_value = 1e-6
        crop_pct = 0.95
    else:
        # 如果尺寸不在支持范围内，抛出异常
        raise ValueError(f"Size {size} not supported")

    # 加载 PoolFormerImageProcessor，用于处理图像
    image_processor = PoolFormerImageProcessor(crop_pct=crop_pct)

    # 准备图像数据
    image = prepare_img()
    # 使用图像处理器处理图像并获取像素值张量
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    # 打印日志，显示模型转换开始
    logger.info(f"Converting model {model_name}...")

    # 加载原始的模型状态字典
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    # 重命名模型状态字典的键名
    state_dict = rename_keys(state_dict)

    # 创建 HuggingFace 模型并加载状态字典
    model = PoolFormerForImageClassification(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 再次定义图像处理器
    image_processor = PoolFormerImageProcessor(crop_pct=crop_pct)
    # 使用 prepare_img 函数准备图像并获取像素值张量
    pixel_values = image_processor(images=prepare_img(), return_tensors="pt").pixel_values

    # 模型前向传播
    outputs = model(pixel_values)
    logits = outputs.logits

    # 定义不同模型的预期 logit 切片
    # 如果尺寸为 "s12"，设置预期切片为指定的张量
    if size == "s12":
        expected_slice = torch.tensor([-0.3045, -0.6758, -0.4869])
    # 如果尺寸为 "s24"，设置预期切片为指定的张量
    elif size == "s24":
        expected_slice = torch.tensor([0.4402, -0.1374, -0.8045])
    # 如果尺寸为 "s36"，设置预期切片为指定的张量
    elif size == "s36":
        expected_slice = torch.tensor([-0.6080, -0.5133, -0.5898])
    # 如果尺寸为 "m36"，设置预期切片为指定的张量
    elif size == "m36":
        expected_slice = torch.tensor([0.3952, 0.2263, -1.2668])
    # 如果尺寸为 "m48"，设置预期切片为指定的张量
    elif size == "m48":
        expected_slice = torch.tensor([0.1167, -0.0656, -0.3423])
    else:
        # 抛出异常，显示不支持的尺寸
        raise ValueError(f"Size {size} not supported")

    # 验证 logits 的形状是否符合预期形状
    assert logits.shape == expected_shape
    # 验证 logits 的前三个元素是否接近于预期切片，允许的绝对误差为 1e-2
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-2)

    # 最后，保存 PyTorch 模型和图像处理器
    logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
    # 创建保存路径（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的消息
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 将图像处理器保存到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本被直接执行而非被导入其他模块，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 添加一个命令行参数，用于指定模型的名称，默认为"poolformer_s12"
    parser.add_argument(
        "--model_name",
        default="poolformer_s12",
        type=str,
        help="Name of the model you'd like to convert.",
    )

    # 添加一个命令行参数，用于指定原始 PyTorch checkpoint 的路径（.pth 文件）
    parser.add_argument(
        "--checkpoint_path", 
        default=None, 
        type=str, 
        help="Path to the original PyTorch checkpoint (.pth file)."
    )

    # 添加一个命令行参数，用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        default=None, 
        type=str, 
        help="Path to the folder to output PyTorch model."
    )

    # 解析命令行参数，并将它们存储在 args 对象中
    args = parser.parse_args()

    # 调用 convert_poolformer_checkpoint 函数，传入命令行参数中指定的模型名称、原始 checkpoint 路径和输出文件夹路径
    convert_poolformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
```