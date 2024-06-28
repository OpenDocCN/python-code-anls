# `.\models\llava_next\convert_llava_next_weights_to_hf.py`

```
# 导入必要的模块和库
import argparse  # 解析命令行参数的库
import glob  # 匹配文件路径名的模式扩展库
import json  # 处理 JSON 格式数据的库
from pathlib import Path  # 处理文件路径的对象模块

import requests  # 发送 HTTP 请求的库
import torch  # PyTorch 深度学习库
from accelerate import init_empty_weights  # 初始化空的模型权重的加速库函数
from huggingface_hub import hf_hub_download, snapshot_download  # 从Hugging Face Hub下载模型和快照的函数
from PIL import Image  # Python Imaging Library，处理图像的库
from safetensors import safe_open  # 安全地打开张量数据的库函数

from transformers import (  # 导入 Hugging Face Transformers 库中的相关模块和类
    AddedToken,
    AutoConfig,
    AutoTokenizer,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
)

# 将需要修改的键值映射关系定义为常量
KEYS_TO_MODIFY_MAPPING = {
    "model.vision_tower.": "",  # 替换模型视觉塔相关的键
    "model.mm_projector": "multi_modal_projector",  # 替换多模态投影器的键
    "model": "model.model",  # 替换模型的键
    "vision_model.model": "vision_model",  # 替换视觉模型的键
    "lm_head": "language_model.lm_head",  # 替换语言模型头部的键
    "model.model": "language_model.model",  # 替换模型的键
    "multi_modal_projector.0": "multi_modal_projector.linear_1",  # 替换多模态投影器的第一层线性层键
    "multi_modal_projector.2": "multi_modal_projector.linear_2",  # 替换多模态投影器的第二层线性层键
    "language_model.model.image_newline": "image_newline",  # 替换语言模型中的图像换行键
}


# 加载原始状态字典的函数
def load_original_state_dict(model_id):
    # 从指定的模型 ID 下载并解压快照，只允许安全张量文件格式
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    # 创建一个空的原始状态字典
    original_state_dict = {}
    # 遍历所有解压后的文件
    for path in glob.glob(f"{directory_path}/*"):
        # 如果文件是安全张量文件
        if path.endswith(".safetensors"):
            # 安全地打开文件并使用 PyTorch 框架读取
            with safe_open(path, framework="pt", device="cpu") as f:
                # 遍历文件中的每个键和对应的张量
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    # 返回完整的原始状态字典
    return original_state_dict


# 将状态字典转换为适合 Hugging Face 的格式的函数
def convert_state_dict_to_hf(state_dict):
    # 创建一个新的状态字典
    new_state_dict = {}
    # 遍历原始状态字典中的每个键值对
    for key, value in state_dict.items():
        # 如果键以 ".inv_freq" 结尾，则跳过
        if key.endswith(".inv_freq"):
            continue
        # 遍历预定义的需要修改的键值映射关系
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            # 如果需要修改的键值映射关系存在于当前键中
            if key_to_modify in key:
                # 替换当前键中的相应部分为新的键
                key = key.replace(key_to_modify, new_key)

        # 将当前处理后的键值对加入新的状态字典，并将值转换为 float16 类型
        new_state_dict[key] = value.to(torch.float16)

    # 返回转换后的新状态字典
    return new_state_dict


# 加载图像的函数
def load_image():
    # 图像的 URL 地址
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    # 从指定的 URL 获取图像数据，以流的方式读取
    image = Image.open(requests.get(url, stream=True).raw)
    # 返回读取的图像数据
    return image
def convert_llava_to_hf(model_id, pytorch_dump_folder_path, push_to_hub=False):
    # 使用指定的 model_id 从 HF Hub 下载模型配置文件 config.json
    filepath = hf_hub_download(repo_id=model_id, filename="config.json", repo_type="model")
    
    # 打开并读取 JSON 文件内容
    with open(filepath) as f:
        data = json.load(f)
        print(data)

    # 根据 model_id 不同设置对应的 text_model_id 和 image_token_index
    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        text_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        image_token_index = 32000
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        text_model_id = "lmsys/vicuna-7b-v1.5"
        image_token_index = 32000
    elif model_id == "liuhaotian/llava-v1.6-vicuna-13b":
        text_model_id = "lmsys/vicuna-13b-v1.5"
        image_token_index = 32000
    elif model_id == "liuhaotian/llava-v1.6-34b":
        text_model_id = "NousResearch/Nous-Hermes-2-Yi-34B"
        image_token_index = 64000
    
    # 从模型配置文件中获取 vision_model_id
    vision_model_id = data["mm_vision_tower"]

    # 设置默认的 torch 数据类型为 torch.float16
    torch.set_default_dtype(torch.float16)
    
    # 使用 text_model_id 创建 AutoConfig 对象
    text_config = AutoConfig.from_pretrained(text_model_id)

    # 根据 model_id 确定是否使用 fast tokenizer
    use_fast = False if model_id == "liuhaotian/llava-v1.6-34b" else True
    tokenizer = AutoTokenizer.from_pretrained(text_model_id, use_fast=use_fast)
    
    # 添加特殊的 "<image>" token 到 tokenizer
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        # 对于 Mistral-7B 模型，添加 "<pad>" 作为 padding token
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # 使用 vision_model_id 创建 LlavaNextImageProcessor 对象
    image_processor = LlavaNextImageProcessor.from_pretrained(vision_model_id)
    
    # 创建 LlavaNextProcessor 对象，传入 tokenizer 和 image_processor
    processor = LlavaNextProcessor(tokenizer=tokenizer, image_processor=image_processor)

    # 构建 LlavaNextConfig 对象，包括 text_config、image_grid_pinpoints 等参数
    config = LlavaNextConfig(
        text_config=text_config.to_dict(),
        image_grid_pinpoints=image_processor.image_grid_pinpoints,
        use_image_newline_parameter=True,
        image_token_index=image_token_index,
    )

    # 初始化空的权重，并创建 LlavaNextForConditionalGeneration 模型
    with init_empty_weights():
        model = LlavaNextForConditionalGeneration(config)

    # 加载原始状态字典
    state_dict = load_original_state_dict(model_id)
    state_dict = convert_state_dict_to_hf(state_dict)
    
    # 加载转换后的状态字典到模型中
    model.load_state_dict(state_dict, assign=True)
    
    # 设置模型为评估模式
    model.eval()

    # 获取模型中预扩展的 embeddings
    pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
    
    # 计算 embeddings 的均值 mu
    mu = torch.mean(pre_expansion_embeddings, dim=0).float()
    n = pre_expansion_embeddings.size()[0]
    
    # 计算 embeddings 的协方差矩阵 sigma
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    
    # 创建多变量正态分布对象 dist
    dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

    # 添加一个 "<image>" token 以调整模型大小
    # 为了性能原因，将模型的填充形状设为 64
    pad_shape = 64
    vocab_size = config.text_config.vocab_size
    
    if model_id == "liuhaotian/llava-v1.6-34b":
        # 对于该模型，有 3 个额外的 token，即 "<|startoftext|>", "<|endoftext|>" 和 "<image>"
        num_tokens = vocab_size + 3
    else:
        # 对于其他模型，有 2 个额外的 token，即 "<image>" 和 "<pad>"
        num_tokens = vocab_size + 2
    # 调整模型的词嵌入大小，使其能容纳给定的词汇量，并且将其填充到指定的形状
    model.resize_token_embeddings(num_tokens, pad_to_multiple_of=pad_shape)

    # 使用分布采样填充词嵌入权重的未初始化部分
    model.language_model.model.embed_tokens.weight.data[vocab_size:] = torch.stack(
        tuple(
            (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[vocab_size:].shape[0]))
        ),
        dim=0,
    )

    # 使用分布采样填充语言模型头部的未初始化部分
    model.language_model.lm_head.weight.data[vocab_size:] = torch.stack(
        tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[vocab_size:].shape[0]))),
        dim=0,
    )

    # 设置模型计算设备为 CUDA 第二块GPU
    device = "cuda:2"
    model.to(device)

    # 准备输入数据
    image = load_image()
    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        # 根据模型ID选择相应的提示文本
        prompt = "[INST] <image>\nWhat is shown in this image? [/INST]"
    elif model_id in ["liuhaotian/llava-v1.6-vicuna-7b", "liuhaotian/llava-v1.6-vicuna-13b"]:
        prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? ASSISTANT:"
    elif model_id == "liuhaotian/llava-v1.6-34b":
        prompt = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat is shown in this image?<|im_end|><|im_start|>assistant\n"
    # 使用处理器对图像和提示文本进行处理，返回PyTorch张量格式的输入数据
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    # 验证输入数据
    # 下载并加载原始像素值数据文件
    filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_pixel_values.pt", repo_type="dataset")
    original_pixel_values = torch.load(filepath, map_location="cpu")
    # 断言原始像素值与输入数据中的像素值相近（使用半精度浮点数比较）
    assert torch.allclose(original_pixel_values, inputs.pixel_values.half())

    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        # 下载并加载原始输入ID数据文件
        filepath = hf_hub_download(repo_id="nielsr/test-image", filename="llava_1_6_input_ids.pt", repo_type="dataset")
        original_input_ids = torch.load(filepath, map_location="cpu")
        # 将原始输入中的特殊标记 -200 替换为图像标记索引
        original_input_ids[original_input_ids == -200] = image_token_index
        # 解码并打印处理后的输入ID数据（排除特殊标记 -200）
        print(tokenizer.decode([id for id in original_input_ids.tolist()[0] if id != -200]))
        # 断言处理后的输入ID与模型输入ID相同
        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

    elif model_id == "liuhaotian/llava-v1.6-34b":
        # 下载并加载特定模型版本的原始输入ID数据文件
        filepath = hf_hub_download(
            repo_id="nielsr/test-image", filename="llava_1_6_34b_input_ids.pt", repo_type="dataset"
        )
        original_input_ids = torch.load(filepath, map_location="cpu")
        # 将原始输入中的特殊标记 -200 替换为图像标记索引
        original_input_ids[original_input_ids == -200] = image_token_index
        # 断言处理后的输入ID与模型输入ID相同
        assert original_input_ids[0].tolist() == inputs.input_ids[0].tolist()

    # 断言图像尺寸与输入数据中的图像尺寸相同
    image_sizes = torch.tensor([[899, 1024]])
    assert image_sizes[0].tolist() == inputs.image_sizes[0].tolist()

    # 执行单次前向传播验证
    print("Single forward pass")
    # 进入推断模式，此模式下不会进行梯度计算
    with torch.inference_mode():
        # 将输入数据移到指定设备上
        inputs = inputs.to(device)
        # 使用模型进行推断，获取输出结果
        outputs = model(**inputs)
        # 打印输出 logits 的形状
        print("Shape of logits:", outputs.logits.shape)
        # 打印 logits 的前几个值
        print("First values of logits:", outputs.logits[0, :3, :3])

        # 根据不同的模型 ID 设置预期的输出切片
        if model_id == "liuhaotian/llava-v1.6-mistral-7b":
            expected_slice = torch.tensor(
                [[-4.8555, -4.6992, -0.1996], [-10.5703, -10.7344, -2.7246], [-7.0391, -7.3672, -0.2634]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
            expected_slice = torch.tensor(
                [[1.4883, 0.9976, -0.6992], [-9.7031, -5.7031, -1.5557], [-5.1328, -5.5586, 8.8281]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "liuhaotian/llava-v1.6-vicuna-13b":
            expected_slice = torch.tensor(
                [[-0.9614, 7.3125, 0.2106], [-7.2695, -8.5469, 3.6211], [-6.3750, -8.1875, 5.4688]],
                dtype=torch.float32,
                device=device,
            )
        elif model_id == "liuhaotian/llava-v1.6-34b":
            expected_slice = torch.tensor(
                [[-9.0859, -9.1406, 5.9453], [-5.9570, -5.9766, 2.2754], [-5.7305, -5.7539, 4.0000]],
                dtype=torch.float32,
                device=device,
            )
        else:
            # 如果模型 ID 不在预期范围内，抛出异常
            raise ValueError(f"Model {model_id} not supported")

        # 断言实际输出的 logits 切片与预期的非常接近
        assert torch.allclose(outputs.logits[0, :3, :3], expected_slice, atol=1e-4)
        # 打印确认 logits 正确
        print("Logits are ok!")

    # 验证生成过程
    output_ids = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
    )

    # 解码生成的文本并去除特殊标记
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 打印生成的文本
    print("Generated text:", repr(generated_text))

    # 根据模型 ID 验证生成的文本是否符合预期
    if model_id == "liuhaotian/llava-v1.6-mistral-7b":
        expected_text = '[INST]  \nWhat is shown in this image? [/INST] The image appears to be a radar chart, which is a type of multi-dimensional plot that displays data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular radar chart, there are several axes labeled with different metrics or benchmarks, such as "MMM-Vet," "MMM-Bench," "LLaVA-Bench," "SLED-Bench," "'
    elif model_id == "liuhaotian/llava-v1.6-vicuna-7b":
        expected_text = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER:  \nWhat is shown in this image? ASSISTANT: The image appears to be a graphical representation of a benchmarking study comparing the performance of various models or systems. It\'s a scatter plot with a circular layout, where each point represents a different model or system, and the axes represent different metrics or dimensions of comparison.\n\nThe metrics are likely related to machine learning or artificial intelligence performance, as indicated by the terms like "BLIP-2," "Instruct BLIP," "POE," "QWA," "V"""
    elif model_id == "liuhaotian/llava-v1.6-vicuna-13b":
        expected_text = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:  \nWhat is shown in this image? ASSISTANT: The image appears to be a radar chart, also known as a spider chart or star chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular radar chart, there are several variables represented:\n\n- MM-Vet\n- LLa-Va-Bench\n- SEED-Bench\n- MM"
    elif model_id == "liuhaotian/llava-v1.6-34b":
        expected_text = "<|im_start|> system\nAnswer the questions. <|im_start|> user\n\nWhat is shown in this image? <|im_start|> assistant\nThe image appears to be a radar chart, also known as a spider chart, which is a graphical method of displaying multivariate data in the form of a two-dimensional chart of three or more quantitative variables represented on axes starting from the same point.\n\nIn this particular chart, there are several datasets represented by different colors and labeled with various acronyms such as MM-Vet, LLaVA-Bench, SEED-Bench, MM-Bench-CN, MM-"
    else:
        raise ValueError(f"Model {model_id} not supported")

    # 确保生成的文本与预期文本一致
    assert generated_text == expected_text
    # 打印确认信息
    print("Generated text is ok!")

    # 验证批量生成
    print("Batched generation...")
    # 指定图像 URL
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 下载并打开图像
    cats_image = Image.open(requests.get(url, stream=True).raw)

    # 处理器接收图像和文本输入，并进行填充和张量化处理
    inputs = processor(
        images=[image, cats_image],  # 图像列表
        text=[prompt, "[INST] <image>\nHow many cats are there? [/INST]"],  # 文本列表
        padding=True,  # 是否填充
        return_tensors="pt",  # 返回 PyTorch 张量
    ).to(device)

    # 打印每个输入项的形状
    for k, v in inputs.items():
        print(k, v.shape)

    # 打印图像尺寸信息
    print("Image sizes:", inputs.image_sizes)

    # 确保图像尺寸相同，以确保批量生成正常工作
    inputs.image_sizes[1] = inputs.image_sizes[0]

    # 再次确认批量生成正在进行
    print("Batched generation...")
    # 使用模型生成输出序列，接收输入参数，并指定最大新增标记数为20，启用缓存
    output_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        use_cache=True,
    )

    # 使用分词器批量解码生成的输出标识符序列，跳过特殊标记并返回文本输出列表
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    # 打印生成的文本输出列表
    print(outputs)

    # 如果指定了 PyTorch 模型导出路径
    if pytorch_dump_folder_path is not None:
        # 打印保存模型和处理器的消息，并创建必要的文件夹（如果不存在）
        print(f"Saving model and processor for {model_id} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)
        # 将处理器保存到指定路径
        processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub
    if push_to_hub:
        # 从模型 ID 中提取仓库 ID
        repo_id = model_id.split("/")[-1]
        # 推送模型到 Hub，命名规则为 llava-hf/{repo_id}-hf
        model.push_to_hub(f"llava-hf/{repo_id}-hf")
        # 推送处理器到 Hub，命名规则为 llava-hf/{repo_id}-hf
        processor.push_to_hub(f"llava-hf/{repo_id}-hf")
# 如果这个脚本被直接运行，执行以下操作
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加一个命令行参数，用于指定模型的Hub位置以进行转换
    parser.add_argument(
        "--model_id",
        help="Hub location of the model to convert",
        default="liuhaotian/llava-v1.6-mistral-7b",
        choices=[
            "liuhaotian/llava-v1.6-mistral-7b",
            "liuhaotian/llava-v1.6-vicuna-7b",
            "liuhaotian/llava-v1.6-vicuna-13b",
            "liuhaotian/llava-v1.6-34b",
        ],
        required=False,
    )
    
    # 添加一个命令行参数，用于指定输出的PyTorch模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    
    # 添加一个命令行参数，设置为True表示是否将转换后的模型推送到🤗 hub
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_llava_to_hf，传递解析后的命令行参数作为参数
    convert_llava_to_hf(args.model_id, args.pytorch_dump_folder_path, args.push_to_hub)
```