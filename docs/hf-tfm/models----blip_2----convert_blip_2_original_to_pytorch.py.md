# `.\models\blip_2\convert_blip_2_original_to_pytorch.py`

```
# 设置脚本的编码格式为 UTF-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert BLIP-2 checkpoints from the original repository.

URL: https://github.com/salesforce/LAVIS/tree/main/projects/blip2
"""

# 导入必要的库
import argparse  # 用于解析命令行参数

import requests  # 用于发送 HTTP 请求
import torch  # PyTorch 深度学习框架

# 安装了一个修改过的版本：pip3 install -U git+https://github.com/nielsrogge/LAVIS.git@blip2
# 以确保可以比较原版和 HF 实现的 float32 实现
from lavis.models import load_model_and_preprocess  # 导入 LAVIS 模型相关函数
from PIL import Image  # Python Imaging Library，用于图像处理

# 导入 Transformers 库中的相关模块和类
from transformers import (
    AutoTokenizer,  # 自动加载 Tokenizer
    Blip2Config,  # BLIP-2 模型的配置类
    Blip2ForConditionalGeneration,  # 用于条件生成的 BLIP-2 模型
    Blip2Processor,  # BLIP-2 处理器
    Blip2VisionConfig,  # BLIP-2 视觉配置类
    BlipImageProcessor,  # BLIP-2 图像处理器
    OPTConfig,  # OpenAI 的配置类
    T5Config,  # T5 模型的配置类
    set_seed,  # 设置随机种子的函数
)
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD  # OpenAI-CLIP 的均值和标准差常数

# 加载演示图像的函数
def load_demo_image():
    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png"
    # 从指定 URL 下载图像并转换为 RGB 模式的 PIL 图像对象
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    return image


# 这里列出所有需要重命名的键值对（原始名称在左边，我们的名称在右边）
def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # 视觉编码器
    rename_keys.append(("visual_encoder.cls_token", "vision_model.embeddings.class_embedding"))
    rename_keys.append(("visual_encoder.pos_embed", "vision_model.embeddings.position_embedding"))
    rename_keys.append(("visual_encoder.patch_embed.proj.weight", "vision_model.embeddings.patch_embedding.weight"))
    rename_keys.append(("visual_encoder.patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.bias"))
    rename_keys.append(("ln_vision.weight", "vision_model.post_layernorm.weight"))
    rename_keys.append(("ln_vision.bias", "vision_model.post_layernorm.bias"))
    # 遍历从配置中获取的视觉模型的隐藏层数量
    for i in range(config.vision_config.num_hidden_layers):
        # 将视觉编码器中第 i 层的权重重命名为视觉模型编码器中对应层的权重
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.weight", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.weight", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.qkv.weight", f"vision_model.encoder.layers.{i}.self_attn.qkv.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.weight", f"vision_model.encoder.layers.{i}.self_attn.projection.weight",))
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.bias", f"vision_model.encoder.layers.{i}.self_attn.projection.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.weight", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.weight", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))

    # 将 QFormer 模型中的 LayerNorm 权重重命名为 QFormer 模型的 layernorm 权重
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.weight", "qformer.layernorm.weight"))
    # 将 QFormer 模型中的 LayerNorm 偏置重命名为 QFormer 模型的 layernorm 偏置
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.bias", "qformer.layernorm.bias"))

    # 返回所有重命名键的列表
    return rename_keys
# 重命名字典中的键，将旧键的值弹出并存储在变量 val 中，然后将该值与新键 new 关联起来
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 从状态字典中读取视觉编码器每个隐藏层的 q 和 v 偏置
def read_in_q_v_bias(state_dict, config):
    for i in range(config.vision_config.num_hidden_layers):
        # 读取原始的 q 和 v 偏置
        q_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.v_bias")

        # 将 q 偏置和 v 偏置连接起来，并在 v 偏置后添加与其形状相同的零张量，构成 qkv 偏置
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        # 设置新的偏置值到状态字典中
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias


# 根据模型名称和 EOS 令牌 ID 获取 Blip2 模型的配置信息
def get_blip2_config(model_name, eos_token_id):
    # 根据模型名称确定图像尺寸
    image_size = 364 if "coco" in model_name else 224
    # 从 Blip2VisionConfig 对象中获取图像尺寸配置并转换为字典形式
    vision_config = Blip2VisionConfig(image_size=image_size).to_dict()

    # 确保模型具有正确的 bos_token_id 和 eos_token_id 设置（生成时很重要）
    if "opt-2.7b" in model_name:
        # 从预训练模型 facebook/opt-2.7b 加载配置信息并转换为字典形式
        text_config = OPTConfig.from_pretrained("facebook/opt-2.7b", eos_token_id=eos_token_id).to_dict()
    elif "opt-6.7b" in model_name:
        # 从预训练模型 facebook/opt-6.7b 加载配置信息并转换为字典形式
        text_config = OPTConfig.from_pretrained("facebook/opt-6.7b", eos_token_id=eos_token_id).to_dict()
    elif "t5-xl" in model_name:
        # 从预训练模型 google/flan-t5-xl 加载配置信息并转换为字典形式，设置 dense_act_fn="gelu" 和 bos_token_id=1
        text_config = T5Config.from_pretrained("google/flan-t5-xl", dense_act_fn="gelu", bos_token_id=1).to_dict()
    elif "t5-xxl" in model_name:
        # 从预训练模型 google/flan-t5-xxl 加载配置信息并转换为字典形式，设置 dense_act_fn="gelu" 和 bos_token_id=1
        text_config = T5Config.from_pretrained("google/flan-t5-xxl", dense_act_fn="gelu", bos_token_id=1).to_dict()

    # 构建 Blip2Config 对象，将视觉配置和文本配置整合到一起
    config = Blip2Config(vision_config=vision_config, text_config=text_config)

    return config, image_size


# 使用 torch.no_grad 装饰器定义的函数，将 Blip2 模型的权重转换到 Transformers 设计
@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    复制/粘贴/调整模型权重到 Transformers 设计。
    """
    # 根据模型名称选择合适的分词器
    tokenizer = (
        AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        if "opt" in model_name
        else AutoTokenizer.from_pretrained("google/flan-t5-xl")
    )
    # 获取 EOS 令牌的 ID
    eos_token_id = tokenizer("\n", add_special_tokens=False).input_ids[0]
    # 获取 Blip2 模型的配置信息和图像尺寸
    config, image_size = get_blip2_config(model_name, eos_token_id=eos_token_id)

    # 创建并评估 Blip2ForConditionalGeneration 模型
    hf_model = Blip2ForConditionalGeneration(config).eval()

    # 根据模型名称映射到原始模型名和类型
    model_name_to_original = {
        "blip2-opt-2.7b": ("blip2_opt", "pretrain_opt2.7b"),
        "blip2-opt-6.7b": ("blip2_opt", "pretrain_opt6.7b"),
        "blip2-opt-2.7b-coco": ("blip2_opt", "caption_coco_opt2.7b"),
        "blip2-opt-6.7b-coco": ("blip2_opt", "caption_coco_opt6.7b"),
        "blip2-flan-t5-xl": ("blip2_t5", "pretrain_flant5xl"),
        "blip2-flan-t5-xl-coco": ("blip2_t5", "caption_coco_flant5xl"),
        "blip2-flan-t5-xxl": ("blip2_t5", "pretrain_flant5xxl"),
    }

    # 注意: 此脚本在两个 GPU 上测试过，因为模型在 float32 下比较，需要相当多的内存。
    # 因此在单独的设备上加载两者以便比较是最简单的方式。
    # 如果CUDA可用，指定hf_model_device为cuda:0，否则为cpu
    hf_model_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 如果CUDA可用，指定lavis_device为cuda:1，否则为cpu
    lavis_device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # 加载原始模型
    print("Loading original model...")
    # 调用函数加载模型及预处理器，设置为评估模式，使用lavis_device指定的设备
    original_model, vis_processors, _ = load_model_and_preprocess(
        name=name, model_type=type, is_eval=True, device=lavis_device
    )
    # 将模型设置为评估模式
    original_model.eval()
    print("Done!")

    # 更新模型状态字典中的键名
    state_dict = original_model.state_dict()
    # 创建重命名映射表
    rename_keys = create_rename_keys(config)
    # 对状态字典中的每个键应用重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 一些键可以高效地重命名
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        # 根据特定规则重命名键
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
        # 将修改后的键值对重新添加到状态字典中
        state_dict[key] = val

    # 读取q_v偏置项
    read_in_q_v_bias(state_dict, config)

    # 加载状态字典到hf_model中，允许部分匹配
    missing_keys, unexpected_keys = hf_model.load_state_dict(state_dict, strict=False)
    # 断言确保没有缺失的键
    assert len(missing_keys) == 0
    # 断言确保出现的键是 ["qformer.embeddings.position_ids"]
    assert unexpected_keys == ["qformer.embeddings.position_ids"]

    # 加载演示图片
    image = load_demo_image()
    # 对演示图片进行处理得到原始像素值，并将其扩展为一维张量，放置在lavis_device上
    original_pixel_values = vis_processors["eval"](image).unsqueeze(0).to(lavis_device)
    # 使用tokenizer处理文本，返回输入的ids张量，并将其放置在hf_model_device上
    input_ids = tokenizer(["\n"], return_tensors="pt").input_ids.to(hf_model_device)

    # 创建图像处理器，指定图像尺寸、均值和标准差
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
    )
    # 创建Blip2Processor，结合图像处理器和tokenizer
    processor = Blip2Processor(image_processor=image_processor, tokenizer=tokenizer)
    # 使用processor处理图片，返回像素值，并将其放置在hf_model_device上
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(hf_model_device)

    # 确保processor创建的像素值与原始像素值完全相同
    assert torch.allclose(pixel_values, original_pixel_values.to(pixel_values.device))

    # 将原始模型和hf_model移动到指定设备上
    original_model.to(lavis_device)
    hf_model.to(hf_model_device)
    # 使用torch.no_grad()上下文管理器，确保在推理阶段不计算梯度
    with torch.no_grad():
        if "opt" in model_name:
            # 如果模型名中包含'opt'，使用原始模型生成logits
            original_logits = original_model({"image": original_pixel_values, "text_input": [""]}).logits
            # 使用hf_model生成logits
            logits = hf_model(pixel_values, input_ids).logits
        else:
            # 否则，使用原始模型生成logits，并使用输入ids进行标签掩码
            original_logits = original_model(
                {"image": original_pixel_values, "text_input": ["\n"], "text_output": ["\n"]}
            ).logits
            labels = input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100)
            logits = hf_model(pixel_values, input_ids, labels=labels).logits

    # 断言确保原始logits和生成logits的形状相同
    assert original_logits.shape == logits.shape
    
if __name__ == "__main__":
    # 如果脚本被直接执行而非被导入，则执行以下代码块

    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()

    # 定义可选的模型名称列表
    choices = [
        "blip2-opt-2.7b",
        "blip2-opt-6.7b",
        "blip2-opt-2.7b-coco",
        "blip2-opt-6.7b-coco",
        "blip2-flan-t5-xl",
        "blip2-flan-t5-xl-coco",
        "blip2-flan-t5-xxl",
    ]

    # 向参数解析器添加命令行参数：模型名称
    parser.add_argument(
        "--model_name",
        default="blip2-opt-2.7b",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )

    # 向参数解析器添加命令行参数：PyTorch 模型输出文件夹路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")

    # 向参数解析器添加命令行参数：是否推送模型到 Hub
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    # 解析命令行参数，并将其存储在 args 对象中
    args = parser.parse_args()

    # 调用函数 convert_blip2_checkpoint，传递解析后的参数
    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```