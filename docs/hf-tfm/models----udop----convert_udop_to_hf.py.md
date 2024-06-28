# `.\models\udop\convert_udop_to_hf.py`

```py
    filepath = hf_hub_download(
        repo_id="hf-internal-testing/fixtures_docvqa", filename="document_2.png", repo_type="dataset"
    )
    # 使用 Hugging Face Hub 下载指定 repository 中的文件 'document_2.png'，返回其本地文件路径
    image = Image.open(filepath).convert("RGB")
    # 打开下载的图像文件，并转换为 RGB 模式的 PIL 图像对象

    return image
    # 返回处理后的图像对象作为函数的输出
    words = ['7', 'ITC', 'Limited', 'REPORT', 'AND', 'ACCOUNTS', '2013', 'ITC’s', 'Brands:', 'An', 'Asset', 'for', 'the', 'Nation', 'The', 'consumer', 'needs', 'and', 'aspirations', 'they', 'fulfil,', 'the', 'benefit', 'they', 'generate', 'for', 'millions', 'across', 'ITC’s', 'value', 'chains,', 'the', 'future-ready', 'capabilities', 'that', 'support', 'them,', 'and', 'the', 'value', 'that', 'they', 'create', 'for', 'the', 'country,', 'have', 'made', 'ITC’s', 'brands', 'national', 'assets,', 'adding', 'to', 'India’s', 'competitiveness.', 'It', 'is', 'ITC’s', 'aspiration', 'to', 'be', 'the', 'No', '1', 'FMCG', 'player', 'in', 'the', 'country,', 'driven', 'by', 'its', 'new', 'FMCG', 'businesses.', 'A', 'recent', 'Nielsen', 'report', 'has', 'highlighted', 'that', "ITC's", 'new', 'FMCG', 'businesses', 'are', 'the', 'fastest', 'growing', 'among', 'the', 'top', 'consumer', 'goods', 'companies', 'operating', 'in', 'India.', 'ITC', 'takes', 'justifiable', 'pride', 'that,', 'along', 'with', 'generating', 'economic', 'value,', 'these', 'celebrated', 'Indian', 'brands', 'also', 'drive', 'the', 'creation', 'of', 'larger', 'societal', 'capital', 'through', 'the', 'virtuous', 'cycle', 'of', 'sustainable', 'and', 'inclusive', 'growth.', 'DI', 'WILLS', '*', ';', 'LOVE', 'DELIGHTFULLY', 'SOFT', 'SKIN?', 'aia', 'Ans', 'Source:', 'https://www.industrydocuments.ucsf.edu/docs/snbx0223']
    # 定义一个包含文本的列表和边界框的空列表
    text_list = []
    bbox_list = []
    # 遍历每个词和对应的框
    for text, box in zip(words, boxes):
        # 如果文本为空，则跳过当前循环
        if text == "":
            continue
        # 对文本进行分词处理
        sub_tokens = tokenizer.tokenize(text)
        # 遍历每个子词并添加到文本列表中，同时将框添加到框列表中
        for sub_token in sub_tokens:
            text_list.append(sub_token)
            bbox_list.append(box)

    # 将文本列表转换为输入 ID 列表
    input_ids = tokenizer.convert_tokens_to_ids(text_list)

    # 将前面的提示 ID 与当前输入 ID 拼接
    input_ids = prompt_ids + input_ids
    # 将框列表与一个全零的框列表拼接
    bbox = [[0, 0, 0, 0]] * len(prompt_ids) + bbox_list

    # 使用图像处理器处理图像并获取像素值
    pixel_values = image_processor(image, return_tensors="pt").pixel_values
    # 使用原始变换处理图像，并展开维度
    original_pixel_values = original_transform(image, image_size=image_processor.size["height"]).unsqueeze(0)
    # 验证像素值是否相似
    assert torch.allclose(original_pixel_values, pixel_values)
    # 打印信息确认像素值正常
    print("Pixel values are ok!")

    # 返回输入 ID 的张量、边界框的张量和像素值
    return torch.tensor(input_ids).unsqueeze(0), torch.tensor(bbox).unsqueeze(0).float(), pixel_values
# 将给定模型名称映射到其对应的检查点路径
def convert_udop_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    # 不同模型名称到其对应的检查点路径的映射字典
    name_to_checkpoint_path = {
        "udop-large": "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-224/pytorch_model.bin",
        "udop-large-512": "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512/pytorch_model.bin",
        "udop-large-512-300k": "/Users/nielsrogge/Documents/UDOP/udop-unimodel-large-512-300k-steps/pytorch_model.bin",
    }

    # 根据模型名称获取其对应的检查点路径
    checkpoint_path = name_to_checkpoint_path[model_name]
    # 使用 torch 加载检查点，将其状态字典加载到 CPU 上
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # 打印加载的检查点路径
    print("Checkpoint path:", checkpoint_path)

    # 创建 HF 模型对象
    image_size = 512 if "512" in model_name else 224
    # 使用指定的配置创建 UDOP 模型配置对象
    config = UdopConfig(decoder_start_token_id=0, image_size=image_size)
    # 使用配置创建条件生成的 UDOP 模型
    model = UdopForConditionalGeneration(config)
    # 将模型设置为评估模式
    model.eval()

    # 重命名状态字典的键名中的特定子字符串
    state_dict = {k.replace("cell2dembedding", "cell_2d_embedding"): v for k, v in state_dict.items()}

    # 加载模型的权重，忽略不匹配的键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 打印缺失的键和不期待的键
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)
    # 断言确保缺失的键和不期待的键满足预期
    assert missing_keys == ["encoder.embed_patches.proj.weight", "encoder.embed_patches.proj.bias"]
    assert unexpected_keys == ["pos_embed"]

    # 准备虚拟输入
    # 从预训练模型 "t5-base" 创建 UDOP 分词器
    tokenizer = UdopTokenizer.from_pretrained("t5-base", legacy=True)
    # 设置图像处理器的大小
    size = {"height": image_size, "width": image_size}
    # 使用 LayoutLMv3 图像处理器创建 UDOP 处理器
    image_processor = LayoutLMv3ImageProcessor(
        image_mean=IMAGENET_DEFAULT_MEAN, image_std=IMAGENET_DEFAULT_STD, size=size
    )
    # 创建 UDOP 处理器
    processor = UdopProcessor(image_processor=image_processor, tokenizer=tokenizer)
    # 准备虚拟输入数据：分词输入 ID、边界框和图像
    input_ids, bbox, image = prepare_dummy_inputs(tokenizer, image_processor)
    # 指定提示文本
    prompt = "Question answering. In which year is the report made?"
    # 使用处理器编码图像和文本输入，返回 PyTorch 张量
    encoding = processor(images=get_image(), text=prompt, return_tensors="pt")

    # 获取输入文本的分词 ID
    input_ids = encoding.input_ids
    # 定义预期的输入 ID 张量，这是一个 2D 张量，表示模型的预期输入
    EXPECTED_INPUT_IDS = torch.tensor([[11860, 18243, 5, 86, 84, 215, 19, 8, 934, 263, 58, 1, 489, 27, 3838, 7363, 4083, 14536, 3430, 5686, 5911, 17161, 134, 2038, 27, 3838, 22, 7, 4688, 7, 10, 389, 18202, 21, 8, 11046, 37, 3733, 523, 11, 38, 2388, 1628, 3, 13133, 23334, 6, 8, 1656, 79, 3806, 21, 4040, 640, 27, 3838, 22, 7, 701, 16534, 6, 8, 3, 76, 2693, 18, 23015, 5644, 24, 380, 3, 6015, 6, 11, 8, 701, 24, 79, 482, 21, 3, 88, 684, 6, 43, 263, 27, 3838, 22, 7, 3635, 1157, 4089, 6, 2651, 12, 1547, 22, 7, 3265, 655, 5, 19, 27, 3838, 22, 7, 38, 2388, 257, 12, 36, 8, 465, 209, 13409, 12150, 1959, 16, 8, 684, 6, 6737, 57, 165, 126, 13409, 12150, 1623, 5, 71, 1100, 30298, 934, 65, 12566, 24, 27, 3838, 31, 7, 126, 13409, 12150, 1623, 33, 8, 10391, 1710, 859, 8, 420, 3733, 4968, 688, 2699, 16, 1547, 5, 27, 3838, 1217, 131, 99, 23, 179, 6064, 24, 6, 590, 28, 3, 11600, 1456, 701, 6, 175, 9443, 2557, 3635, 92, 1262, 8, 3409, 13, 2186, 3, 27908, 1784, 190, 8, 3, 5771, 17, 13281, 4005, 13, 5086, 11, 13066, 1170, 5, 10826, 16309, 134, 3, 2, 276, 26, 3, 55, 391, 13570, 5, 10315, 309, 3577, 19114, 371, 4254, 5121, 5055, 6245, 3, 10047, 3162, 58, 3, 9, 61, 1713, 2703, 476, 667, 25158, 301, 6058, 6038, 476, 3765, 9149, 10, 4893, 1303, 1986, 5, 13580, 7, 8224, 28244, 7, 5, 76, 75, 7, 89, 5, 15, 1259, 87, 7171, 7, 87, 7, 29, 115, 226, 4305, 2773, 1]])  # fmt: skip
    # 检查预期输入 ID 是否与实际输入 ID 相匹配，使用 torch.testing.assert_close 进行断言
    torch.testing.assert_close(EXPECTED_INPUT_IDS, input_ids)
    # 获得编码中的边界框，转换为浮点数类型
    bbox = encoding.bbox.float()
    # 获得编码中的像素值
    pixel_values = encoding.pixel_values

    # 如果出现异常，打印错误信息，并准备使用虚拟输入
    except Exception:
        print("Input_ids don't match, preparing dummy inputs")
        # 调用准备虚拟输入的函数，获取输入 ID、边界框和像素值
        input_ids, bbox, pixel_values = prepare_dummy_inputs(tokenizer, image_processor)

    # 验证单个前向传播过程
    print("Testing single forward pass..")
    # 禁用梯度计算的上下文管理器
    with torch.no_grad():
        # 设置解码器的输入 ID
        decoder_input_ids = torch.tensor([[101]])
        # 调用模型进行前向传播，获取输出结果
        outputs = model(input_ids=input_ids, bbox=bbox, pixel_values=pixel_values, decoder_input_ids=decoder_input_ids)
        # 打印输出 logits 的形状
        print("Shape of logits:", outputs.logits.shape)
        # 打印 logits 的前几个值
        print("First values of logits:", outputs.logits[0, :3, :3])

    # 比较输出 logits 的前几个值与预期值的接近程度，设定容差值为 1e-4
    # 在 Linux 上：tensor([[-18.5262, 1.5087, -15.7051]])
    # 在 Mac 上：tensor([[-19.4976, 0.8515, -17.1873]])
    try:
        assert torch.allclose(outputs.logits[0, :3, :3], torch.tensor([[-18.5262, 1.5087, -15.7051]]), atol=1e-4)
        print("Looks ok!")
    # 如果比较不通过，打印提示信息
    except Exception:
        print("logits don't match let's try to generate")

    # 验证自回归解码过程
    print("Testing generation...")
    # 构建模型参数字典
    model_kwargs = {"bbox": bbox, "pixel_values": pixel_values}
    # 调用模型进行生成，指定最大新增 token 数量为 20
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)

    # 打印生成的结果文本，跳过特殊 token 后解码成字符串
    print("Generated:", tokenizer.batch_decode(outputs, skip_special_tokens=True))

    # 使用原始输入数据进行自回归解码
    print("Testing generation with original inputs...")
    # 下载指定的模型输入数据文件
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="input_ids_udop.pt", repo_type="dataset")
    # 从文件加载预训练模型的输入标识符
    input_ids = torch.load(filepath)
    # 使用hf_hub_download函数下载指定仓库和文件名的内容，并更新filepath变量
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="bbox_udop.pt", repo_type="dataset")
    # 加载保存在filepath中的包围框数据
    bbox = torch.load(filepath)
    # 根据模型名称确定要加载的像素值文件名
    pixel_values_filename = "pixel_values_udop_512.pt" if "512" in model_name else "pixel_values_udop_224.pt"
    # 使用hf_hub_download函数下载指定仓库和文件名的内容，并更新filepath变量
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename=pixel_values_filename, repo_type="dataset")
    # 加载保存在filepath中的像素值数据
    pixel_values = torch.load(filepath)

    # 打印解码后的输入标识符，跳过特殊标记
    print("Decoded input ids:", tokenizer.decode(input_ids[0], skip_special_tokens=True))
    # 打印包围框的形状
    print("Bbox shape:", bbox.shape)

    # 准备模型参数，包括bbox和pixel_values
    model_kwargs = {"bbox": bbox, "pixel_values": pixel_values}
    # 使用模型生成文本输出，限制最大新生成的标记数量为20
    outputs = model.generate(input_ids=input_ids, **model_kwargs, max_new_tokens=20)
    # 解码模型生成的输出文本，跳过特殊标记，并取第一个文本结果
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # 打印生成的文本
    print("Generated:", generated_text)

    # 如果指定了PyTorch模型保存文件夹路径，则保存模型和分词器的预训练状态
    if pytorch_dump_folder_path is not None:
        model.save_pretrained(pytorch_dump_folder_path)
        tokenizer.save_pretrained(pytorch_dump_folder_path)

    # 如果设置了push_to_hub标志，则将模型和处理器推送到Hub上指定的仓库
    if push_to_hub:
        model.push_to_hub(f"microsoft/{model_name}")
        processor.push_to_hub(f"microsoft/{model_name}")
        # 重要提示：要将快速分词器文件保存在Hub上的仓库中，请执行以下操作：
        # 参见https://discuss.huggingface.co/t/convert-slow-xlmrobertatokenizer-to-fast-one/20876
if __name__ == "__main__":
    # 如果脚本直接执行而非被导入，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="udop-large",
        type=str,
        choices=["udop-large", "udop-large-512", "udop-large-512-300k"],
        help=("Name of the UDOP model you'd like to convert."),
    )
    # 添加一个必需的参数，用于指定要转换的 UDOP 模型的名称，提供了默认值和可选项

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加一个参数，用于指定输出的 PyTorch 模型目录的路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加一个参数，指定是否将转换后的模型推送到 🤗 hub

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，将解析后的参数传递给函数进行 UDOP 模型的转换
    convert_udop_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```