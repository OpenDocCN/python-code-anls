# `.\models\blip\convert_blip_original_pytorch_to_hf.py`

```py
# 定义一个装饰器，用于告知解释器在调用 convert_blip_checkpoint 函数时不需要进行梯度计算
@torch.no_grad()
# 定义函数 convert_blip_checkpoint，用于将 BLIP 模型的检查点转换为 transformers 模型的权重
def convert_blip_checkpoint(pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    # 如果提供了配置文件路径，则加载 BLIP 模型的配置信息
    if config_path is not None:
        config = BlipConfig.from_pretrained(config_path)
    else:
        # 如果没有提供配置，则使用默认配置创建 BlipConfig 对象
        config = BlipConfig(projection_dim=512, text_config={}, vision_config={})

    # 创建用于生成文本的 BLIP 模型对象，并设置为评估模式
    hf_model = BlipForConditionalGeneration(config).eval()

    # 指定预训练模型的 URL 地址
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"

    # 使用指定的 URL 地址加载预训练模型，并设置为评估模式
    pt_model = blip_decoder(pretrained=model_url, image_size=384, vit="base")
    pt_model = pt_model.eval()

    # 复制修改后的状态字典，并为每个键重新命名
    modified_state_dict = pt_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    # 将修改后的模型状态字典加载到 hf_model 中
    hf_model.load_state_dict(modified_state_dict)

    # 加载演示图像，并指定图像大小和设备
    image_size = 384
    image = load_demo_image(image_size=image_size, device="cpu")

    # 使用 Google BERT tokenizer 创建 tokenizer 对象
    tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

    # 使用 tokenizer 处理输入文本，生成 input_ids
    input_ids = tokenizer(["a picture of"]).input_ids

    # 使用 hf_model 生成文本输出
    out = hf_model.generate(image, input_ids)

    # 断言生成的文本输出符合预期的 token 序列
    assert out[0].tolist() == [30522, 1037, 3861, 1997, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]

    # 使用 hf_model 生成文本输出（不带额外的 input_ids）
    out = hf_model.generate(image)

    # 断言生成的文本输出符合预期的 token 序列
    assert out[0].tolist() == [30522, 1037, 2450, 3564, 2006, 1996, 3509, 2007, 2014, 3899, 102]

    # 如果指定了 pytorch_dump_folder_path，则保存 hf_model 的预训练参数
    if pytorch_dump_folder_path is not None:
        hf_model.save_pretrained(pytorch_dump_folder_path)

    # 指定用于 VQA 模型的预训练模型 URL 地址
    model_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"
    )

    # 使用指定的 URL 地址加载 VQA 预训练模型，并设置为评估模式
    vqa_model = blip_vqa(pretrained=model_url, image_size=image_size, vit="base")
    vqa_model.eval()

    # 复制修改后的状态字典，并为每个键重新命名
    modified_state_dict = vqa_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    # 创建用于 VQA 的 BLIP 模型对象
    hf_vqa_model = BlipForQuestionAnswering(config)

    # 将修改后的模型状态字典加载到 hf_vqa_model 中
    hf_vqa_model.load_state_dict(modified_state_dict)

    # 指定 VQA 问题
    question = ["How many dogs are in this image?"]
    # 使用 tokenizer 处理 VQA 问题，生成 question_input_ids
    question_input_ids = tokenizer(question, return_tensors="pt").input_ids

    # 使用 hf_vqa_model 生成 VQA 回答
    answer = hf_vqa_model.generate(question_input_ids, image)

    # 打印解码后的 VQA 回答文本
    print(tokenizer.decode(answer[0]))

    # 断言解码后的 VQA 回答文本符合预期结果
    assert tokenizer.decode(answer[0]) == "[UNK] 1 [SEP]"

    # 如果指定了 pytorch_dump_folder_path，则保存 hf_vqa_model 的预训练参数
    if pytorch_dump_folder_path is not None:
        hf_vqa_model.save_pretrained(pytorch_dump_folder_path + "_vqa")

    # 指定用于 ITM 模型的预训练模型 URL 地址
    model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth"

    # 使用指定的 URL 地址加载 ITM 预训练模型，并设置为评估模式
    itm_model = blip_itm(pretrained=model_url, image_size=image_size, vit="base")
    itm_model.eval()

    # 复制修改后的状态字典，并为每个键重新命名
    modified_state_dict = itm_model.state_dict()
    for key in modified_state_dict.copy():
        value = modified_state_dict.pop(key)
        renamed_key = rename_key(key)
        modified_state_dict[renamed_key] = value

    # 创建用于图像文本检索的 BLIP 模型对象
    hf_itm_model = BlipForImageTextRetrieval(config)

    # 将修改后的模型状态字典加载到 hf_itm_model 中
    hf_itm_model.load_state_dict(modified_state_dict)

    # 指定图像文本检索的问题
    question = ["A picture of a woman with a dog sitting in a beach"]
    # 使用tokenizer对问题进行编码，返回PyTorch张量格式的输入ID，进行填充和截断以适应最大长度为35
    question_input_ids = tokenizer(
        question,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=35,
    ).input_ids

    # 载入预训练模型的修改后状态字典，用于更新模型参数
    hf_itm_model.load_state_dict(modified_state_dict)
    # 将模型设置为评估模式，不启用训练相关的模块（如Dropout）
    hf_itm_model.eval()

    # 使用修改后的模型进行推理，生成图片和问题输入对应的输出，使用itm头部
    out_itm = hf_itm_model(question_input_ids, image, use_itm_head=True)
    # 使用修改后的模型进行推理，生成图片和问题输入对应的输出，不使用itm头部
    out = hf_itm_model(question_input_ids, image, use_itm_head=False)

    # 断言输出中的第一个元素等于预期的值 0.2110687494277954
    assert out[0].item() == 0.2110687494277954
    # 断言输出中softmax后在第二维度上第一列的值等于预期的值 0.45698845386505127
    assert torch.nn.functional.softmax(out_itm[0], dim=1)[:, 1].item() == 0.45698845386505127

    # 如果给定了PyTorch模型保存的文件夹路径，则将修改后的模型保存在指定路径+"_itm"处
    if pytorch_dump_folder_path is not None:
        hf_itm_model.save_pretrained(pytorch_dump_folder_path + "_itm")
# 如果当前脚本作为主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个命令行参数，用于指定输出的 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个命令行参数，用于指定要转换的模型的 hf config.json 文件路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用 convert_blip_checkpoint 函数，传入命令行参数中的 checkpoint_path、pytorch_dump_folder_path 和 config_path
    convert_blip_checkpoint(args.checkpoint_path, args.pytorch_dump_folder_path, args.config_path)
```