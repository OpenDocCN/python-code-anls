# `.\transformers\models\blip_2\convert_blip_2_original_to_pytorch.py`

```py
# 设置脚本编码格式为 UTF-8
# 版权声明及许可协议
"""
Convert BLIP-2 checkpoints from the original repository.
从原始存储库中转换 BLIP-2 检查点。

URL: https://github.com/salesforce/LAVIS/tree/main/projects/blip2
"""

# 导入模块
import argparse  # 导入参数解析模块

import requests  # 导入请求模块
import torch  # 导入 PyTorch 模块

# 安装 salesforce-lavis 库：pip3 install salesforce-lavis
# 我实际上安装了一个稍微修改过的版本：pip3 install -U git+https://github.com/nielsrogge/LAVIS.git@blip2_float32
# 以确保我们可以比较浮点32位的原始和 HF 实现
from lavis.models import load_model_and_preprocess  # 导入 lavis.models 模块中的 load_model_and_preprocess 函数
from PIL import Image  # 导入 PIL 模块中的 Image 类

from transformers import (  # 导入 transformers 模块中的相关类和函数
    AutoTokenizer,  # 导入 AutoTokenizer 类
    Blip2Config,  # 导入 Blip2Config 类
    Blip2ForConditionalGeneration,  # 导入 Blip2ForConditionalGeneration 类
    Blip2Processor,  # 导入 Blip2Processor 类
    Blip2VisionConfig,  # 导入 Blip2VisionConfig 类
    BlipImageProcessor,  # 导入 BlipImageProcessor 类
    OPTConfig,  # 导入 OPTConfig 类
    T5Config,  # 导入 T5Config 类
    set_seed,  # 导入 set_seed 函数
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD  # 导入常量 OPENAI_CLIP_MEAN 和 OPENAI_CLIP_STD

# 加载演示图片函数
def load_demo_image():
    # 图片 URL
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
    # 从 URL 获取图片，并转换为 RGB 格式
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    return image


# 创建重命名键函数
# 在这里列出所有要重命名的键（左边是原始名称，右边是我们的名称）
def create_rename_keys(config):
    # 初始化重命名键列表
    rename_keys = []
    # 格式化关闭

    # 视觉编码器
    rename_keys.append(("visual_encoder.cls_token", "vision_model.embeddings.class_embedding"))  # 添加键值对，将 "visual_encoder.cls_token" 重命名为 "vision_model.embeddings.class_embedding"
    rename_keys.append(("visual_encoder.pos_embed", "vision_model.embeddings.position_embedding"))  # 添加键值对，将 "visual_encoder.pos_embed" 重命名为 "vision_model.embeddings.position_embedding"
    rename_keys.append(("visual_encoder.patch_embed.proj.weight", "vision_model.embeddings.patch_embedding.weight"))  # 添加键值对，将 "visual_encoder.patch_embed.proj.weight" 重命名为 "vision_model.embeddings.patch_embedding.weight"
    rename_keys.append(("visual_encoder.patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.bias"))  # 添加键值对，将 "visual_encoder.patch_embed.proj.bias" 重命名为 "vision_model.embeddings.patch_embedding.bias"
    rename_keys.append(("ln_vision.weight", "vision_model.post_layernorm.weight"))  # 添加键值对，将 "ln_vision.weight" 重命名为 "vision_model.post_layernorm.weight"
    rename_keys.append(("ln_vision.bias", "vision_model.post_layernorm.bias"))  # 添加键值对，将 "ln_vision.bias" 重命名为 "vision_model.post_layernorm.bias"
    # 遍历视觉编码器（visual_encoder）中的隐藏层次数，进行参数重命名
    for i in range(config.vision_config.num_hidden_layers):
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的第一个规范化层（norm1）的权重键
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.weight", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的第一个规范化层（norm1）的偏置键
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的第二个规范化层（norm2）的权重键
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.weight", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的第二个规范化层（norm2）的偏置键
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的自注意力机制（attn）的权重键
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.qkv.weight", f"vision_model.encoder.layers.{i}.self_attn.qkv.weight"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的自注意力机制（attn）的投影权重键
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.weight", f"vision_model.encoder.layers.{i}.self_attn.projection.weight",))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的自注意力机制（attn）的投影偏置键
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.bias", f"vision_model.encoder.layers.{i}.self_attn.projection.bias"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的多层感知机（mlp）的第一层权重键
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.weight", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的多层感知机（mlp）的第一层偏置键
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的多层感知机（mlp）的第二层权重键
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.weight", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        # 添加重命名后的键值对，替换视觉编码器中第 i 个块的多层感知机（mlp）的第二层偏置键
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))
    
    # QFormer 模型参数重命名
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.weight", "qformer.layernorm.weight"))
    # 添加重命名后的键值对，替换 QFormer 模型的嵌入层规范化层的权重键
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.bias", "qformer.layernorm.bias"))
    # 添加重命名后的键值对，替换 QFormer 模型的嵌入层规范化层的偏置键
    
    # 返回参数重命名后的键值对列表
    return rename_keys
# 定义一个函数用于重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 在字典中添加新键，并赋予旧键对应的值
    dct[new] = val


# 定义一个函数用于读取模型状态字典中的 q 和 v 偏置信息
def read_in_q_v_bias(state_dict, config):
    # 遍历视觉编码器的隐藏层
    for i in range(config.vision_config.num_hidden_layers):
        # 读取原始的 q 和 v 偏置
        q_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.v_bias")

        # 合并 q 和 v 偏置，构成 qkv 偏置，并设置到状态字典中
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias


# 定义一个函数用于获取 BLIP2 模型的配置信息
def get_blip2_config(model_name, eos_token_id):
    # 根据模型名确定图像尺寸
    image_size = 364 if "coco" in model_name else 224
    # 获取视觉配置信息
    vision_config = Blip2VisionConfig(image_size=image_size).to_dict()

    # 确保模型具有正确的 bos_token_id 和 eos_token_id（对生成很重要）
    if "opt-2.7b" in model_name:
        # 如果模型名包含 "opt-2.7b"，则使用相应的 OPTConfig，并设置 eos_token_id
        text_config = OPTConfig.from_pretrained("facebook/opt-2.7b", eos_token_id=eos_token_id).to_dict()
    elif "opt-6.7b" in model_name:
        # 如果模型名包含 "opt-6.7b"，则使用相应的 OPTConfig，并设置 eos_token_id
        text_config = OPTConfig.from_pretrained("facebook/opt-6.7b", eos_token_id=eos_token_id).to_dict()
    elif "t5-xl" in model_name:
        # 如果模型名包含 "t5-xl"，则使用相应的 T5Config，并设置 bos_token_id
        text_config = T5Config.from_pretrained("google/flan-t5-xl", dense_act_fn="gelu", bos_token_id=1).to_dict()
    elif "t5-xxl" in model_name:
        # 如果模型名包含 "t5-xxl"，则使用相应的 T5Config，并设置 bos_token_id
        text_config = T5Config.from_pretrained("google/flan-t5-xxl", dense_act_fn="gelu", bos_token_id=1).to_dict()

    # 构建 BLIP2 模型的配置信息
    config = Blip2Config(vision_config=vision_config, text_config=text_config)

    return config, image_size


# 定义一个装饰器函数，用于禁用梯度计算的上下文
@torch.no_grad()
# 定义一个函数，用于将 BLIP2 模型的检查点转换为 Transformers 模型设计
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    # 根据模型名选择合适的分词器
    tokenizer = (
        AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        if "opt" in model_name
        else AutoTokenizer.from_pretrained("google/flan-t5-xl")
    )
    # 获取 EOS 符号的 token ID
    eos_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    # 获取 BLIP2 模型的配置信息和图像尺寸
    config, image_size = get_blip2_config(model_name, eos_token_id=eos_token_id)

    # 创建 BLIP2 模型并设置为评估模式
    hf_model = Blip2ForConditionalGeneration(config).eval()

    # 定义模型名到原始模型名称和类型的映射关系
    model_name_to_original = {
        "blip2-opt-2.7b": ("blip2_opt", "pretrain_opt2.7b"),
        "blip2-opt-6.7b": ("blip2_opt", "pretrain_opt6.7b"),
        "blip2-opt-2.7b-coco": ("blip2_opt", "caption_coco_opt2.7b"),
        "blip2-opt-6.7b-coco": ("blip2_opt", "caption_coco_opt6.7b"),
        "blip2-flan-t5-xl": ("blip2_t5", "pretrain_flant5xl"),
        "blip2-flan-t5-xl-coco": ("blip2_t5", "caption_coco_flant5xl"),
        "blip2-flan-t5-xxl": ("blip2_t5", "pretrain_flant5xxl"),
    }

    # 获取模型名对应的原始模型名称和类型
    name, type = model_name_to_original[model_name]

    # 注意：此脚本在两个 GPU 上测试，因为模型以 float32 进行比较，需要相当多的内存。
    # 因此，在单独的设备上加载两个模型以进行比较是最简单的方法
    # 检查是否有可用的 CUDA 设备，根据情况选择使用 "cuda:0" 或 "cpu"
    hf_model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 检查是否有可用的 CUDA 设备，根据情况选择使用 "cuda:1" 或 "cpu"
    lavis_device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # 加载原始模型
    print("Loading original model...")
    # 调用加载模型和预处理的函数，返回原始模型、可视化处理器和其他信息
    original_model, vis_processors, _ = load_model_and_preprocess(
        name=name, model_type=type, is_eval=True, device=lavis_device
    )
    # 设置原始模型为评估模式
    original_model.eval()
    print("Done!")

    # 更新状态字典的键
    state_dict = original_model.state_dict()
    # 创建重命名键的列表
    rename_keys = create_rename_keys(config)
    # 遍历重命名键列表，对状态字典进行键重命名
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 一些键可以进行有效的重命名
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        # 替换键名中的部分字符串
        if key.startswith("Qformer.bert"):
            key = key.replace("Qformer.bert", "qformer")
        if "attention.self" in key:
            key = key.replace("self", "attention")
        if "opt_proj" in key:
            key = key.replace("opt_proj", "language_projection")
        if "t5_proj" in key:
            key = key.replace("t5_proj", "language_projection")
        if key.startswith("opt"):
            key = key.replace("opt", "language")
        if key.startswith("t5"):
            key = key.replace("t5", "language")
        state_dict[key] = val

    # 读取 qv 偏置
    read_in_q_v_bias(state_dict, config)

    # 使用 state_dict 加载模型的参数，允许部分参数不匹配
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    # 断言确保没有缺失的键
    assert len(missing_keys) == 0
    # 断言确保没有意外的键
    assert unexpected_keys == ["qformer.embeddings.position_ids"]

    # 加载演示图像
    image = load_demo_image()
    # 对原始像素值进行处理，并转换为模型所需的格式
    original_pixel_values = vis_processors["eval"](image).unsqueeze(0).to(lavis_device)
    # 使用 tokenizer 对文本进行编码，并将其移至指定设备
    input_ids = tokenizer(["\n"], return_tensors="pt").input_ids.to(hf_model_device)

    # 创建图像处理器
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
    )
    # 创建处理器，将图像和文本转换为模型输入
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    # 对图像进行处理，并将其移至指定设备
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(hf_model_device)

    # 断言确保处理器创建了完全相同的像素值
    assert torch.allclose(pixel_values, original_pixel_values.to(pixel_values.device))

    # 将原始模型和 Hugging Face 模型移至指定设备
    original_model.to(lavis_device)
    hf_model.to(hf_model_device)
    # 使用 torch.no_grad() 上下文管理器，确保不会跟踪梯度
    with torch.no_grad():
        # 如果模型名包含 "opt"，则使用原始模型进行推理
        if "opt" in model_name:
            original_logits = original_model({"image": original_pixel_values, "text_input": [""]}).logits
            logits = hf_model(pixel_values, input_ids).logits
        # 否则，使用原始模型进行推理并计算损失
        else:
            original_logits = original_model(
                {"image": original_pixel_values, "text_input": ["\n"], "text_output": ["\n"]}
            ).logits
            labels = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
            logits = hf_model(pixel_values, input_ids, labels=labels).logits

    # 断言确保原始模型和 Hugging Face 模型的输出形状相同
    assert original_logits.shape == logits.shape
    # 打印原始logits的前3行3列的值
    print("First values of original logits:", original_logits[0, :3, :3])
    # 打印HF（Hugging Face）模型的logits的前3行3列的值
    print("First values of HF logits:", logits[0, :3, :3])

    # 断言原始logits和HF模型的logits的值是否在一定范围内接近
    assert torch.allclose(original_logits.to(logits.device), logits, atol=1e-4)
    # 打印确认信息
    print("Looks ok!")

    # 打印生成标题的提示
    print("Generating a caption...")
    # 设置提示文本
    prompt = "Question: what object is in this image? Answer:"
    # 使用分词器将提示转换为输入tensor，并移到HF模型所在的设备上
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(hf_model_device)

    # 设置随机种子
    set_seed(42)

    # 使用原始模型生成标题
    original_outputs = original_model.generate(
        {"image": original_pixel_values, "prompt": prompt}, use_nucleus_sampling=True
    )
    # 使用HF模型生成标题
    outputs = hf_model.generate(
        pixel_values,
        input_ids,
        do_sample=True,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        temperature=1,
    )
    # 解码生成的文本并去除特殊token
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    # 去除生成文本中的空格
    output_text = [text.strip() for text in output_text]
    # 打印原始生成的文本
    print("Original generation:", original_outputs)
    # 打印HF生成的文本
    print("HF generation:", output_text)

    # 如果指定了pytorch_dump_folder_path，则保存处理器和HF模型
    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到Hub
    if push_to_hub:
        # 将处理器推送到Hub
        processor.push_to_hub(f"nielsr/{model_name}")
        # 将HF模型推送到Hub
        hf_model.push_to_hub(f"nielsr/{model_name}")
# 如果当前脚本被当作主程序执行，则执行以下代码块
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 定义模型名称的可选项列表
    choices = [
        "blip2-opt-2.7b",
        "blip2-opt-6.7b",
        "blip2-opt-2.7b-coco",
        "blip2-opt-6.7b-coco",
        "blip2-flan-t5-xl",
        "blip2-flan-t5-xl-coco",
        "blip2-flan-t5-xxl",
    ]
    # 向参数解析器添加模型名称参数，包括默认值、可选值、类型和帮助信息
    parser.add_argument(
        "--model_name",
        default="blip2-opt-2.7b",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    # 向参数解析器添加输出 PyTorch 模型文件夹路径的参数，默认为 None
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 向参数解析器添加是否推送模型到 Hub 的参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    # 解析命令行参数并将其存储在 args 变量中
    args = parser.parse_args()

    # 调用函数 convert_blip2_checkpoint，传入参数 model_name、pytorch_dump_folder_path 和 push_to_hub
    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```