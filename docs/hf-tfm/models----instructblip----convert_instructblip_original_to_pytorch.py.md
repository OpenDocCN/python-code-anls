# `.\models\instructblip\convert_instructblip_original_to_pytorch.py`

```
# coding=utf-8
# 设置文件编码为UTF-8，确保支持中文和其他非ASCII字符

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据Apache许可证2.0版授权，详细内容可访问指定URL查看

# you may not use this file except in compliance with the License.
# 您除非遵守许可证，否则不能使用本文件

# You may obtain a copy of the License at
# 您可以在上述URL获取许可证的副本

# http://www.apache.org/licenses/LICENSE-2.0
# 许可证URL地址

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发，没有任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查阅许可证，了解权限和限制

"""
Convert InstructBLIP checkpoints from the original repository.

URL: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
"""
# 脚本说明，用于从原始存储库转换InstructBLIP检查点，附带原始存储库URL链接

import argparse
# 导入命令行参数解析模块

import requests
# 导入处理HTTP请求的模块

import torch
# 导入PyTorch深度学习库

# pip3 install salesforce-lavis
# 安装salesforce-lavis库的说明注释

# I'm actually installing a slightly modified version: pip3 install git+https://github.com/nielsrogge/LAVIS.git@fix_lavis_float32 (there's also the fix_lavis branch)
# also note: to convert Vicuna checkpoints, we had to include /home/niels/python_projects/checkpoints/FastChat/vicuna-7b in lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml
# same for Vicuna-13b
from lavis.models import load_model_and_preprocess
# 从lavis.models模块导入load_model_and_preprocess函数

from PIL import Image
# 导入Python Imaging Library模块中的Image类

from transformers import (
    AutoTokenizer,
    BlipImageProcessor,
    InstructBlipConfig,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
    LlamaConfig,
    LlamaTokenizerFast,
    T5Config,
    T5TokenizerFast,
)
# 从transformers库导入多个类和配置，用于自然语言处理和模型处理

from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
# 导入常量定义，用于处理OpenAI的相关内容

def load_demo_image():
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    # 指定图片的URL地址

    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    # 通过HTTP请求获取图片流，并转换为RGB格式的图像对象

    return image
    # 返回加载的图像对象

# here we list all keys to be renamed (original name on the left, our name on the right)
# 此处列出需要重命名的所有键（原始名称在左侧，我们的名称在右侧）

def create_rename_keys(config):
    rename_keys = []
    # 创建一个空列表，用于存储重命名的键值对

    # fmt: off
    # 关闭代码格式化功能，用于防止对后续部分的自动格式化

    # vision encoder
    rename_keys.append(("visual_encoder.cls_token", "vision_model.embeddings.class_embedding"))
    # 将键 "visual_encoder.cls_token" 重命名为 "vision_model.embeddings.class_embedding"

    rename_keys.append(("visual_encoder.pos_embed", "vision_model.embeddings.position_embedding"))
    # 将键 "visual_encoder.pos_embed" 重命名为 "vision_model.embeddings.position_embedding"

    rename_keys.append(("visual_encoder.patch_embed.proj.weight", "vision_model.embeddings.patch_embedding.weight"))
    # 将键 "visual_encoder.patch_embed.proj.weight" 重命名为 "vision_model.embeddings.patch_embedding.weight"

    rename_keys.append(("visual_encoder.patch_embed.proj.bias", "vision_model.embeddings.patch_embedding.bias"))
    # 将键 "visual_encoder.patch_embed.proj.bias" 重命名为 "vision_model.embeddings.patch_embedding.bias"

    rename_keys.append(("ln_vision.weight", "vision_model.post_layernorm.weight"))
    # 将键 "ln_vision.weight" 重命名为 "vision_model.post_layernorm.weight"

    rename_keys.append(("ln_vision.bias", "vision_model.post_layernorm.bias"))
    # 将键 "ln_vision.bias" 重命名为 "vision_model.post_layernorm.bias"
    # 遍历 vision_config 中的隐藏层数量次数，生成需要重命名的键值对列表
    for i in range(config.vision_config.num_hidden_layers):
        # 重命名 visual_encoder.blocks 中第 i 层的 norm1.weight 到 vision_model.encoder.layers 中对应层的 layer_norm1.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.weight", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        # 重命名 visual_encoder.blocks 中第 i 层的 norm1.bias 到 vision_model.encoder.layers 中对应层的 layer_norm1.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        # 重命名 visual_encoder.blocks 中第 i 层的 norm2.weight 到 vision_model.encoder.layers 中对应层的 layer_norm2.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.weight", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        # 重命名 visual_encoder.blocks 中第 i 层的 norm2.bias 到 vision_model.encoder.layers 中对应层的 layer_norm2.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        # 重命名 visual_encoder.blocks 中第 i 层的 attn.qkv.weight 到 vision_model.encoder.layers 中对应层的 self_attn.qkv.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.qkv.weight", f"vision_model.encoder.layers.{i}.self_attn.qkv.weight"))
        # 重命名 visual_encoder.blocks 中第 i 层的 attn.proj.weight 到 vision_model.encoder.layers 中对应层的 self_attn.projection.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.weight", f"vision_model.encoder.layers.{i}.self_attn.projection.weight",))
        # 重命名 visual_encoder.blocks 中第 i 层的 attn.proj.bias 到 vision_model.encoder.layers 中对应层的 self_attn.projection.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.bias", f"vision_model.encoder.layers.{i}.self_attn.projection.bias"))
        # 重命名 visual_encoder.blocks 中第 i 层的 mlp.fc1.weight 到 vision_model.encoder.layers 中对应层的 mlp.fc1.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.weight", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        # 重命名 visual_encoder.blocks 中第 i 层的 mlp.fc1.bias 到 vision_model.encoder.layers 中对应层的 mlp.fc1.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        # 重命名 visual_encoder.blocks 中第 i 层的 mlp.fc2.weight 到 vision_model.encoder.layers 中对应层的 mlp.fc2.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.weight", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        # 重命名 visual_encoder.blocks 中第 i 层的 mlp.fc2.bias 到 vision_model.encoder.layers 中对应层的 mlp.fc2.bias
    
    # QFormer 的特定重命名
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.weight", "qformer.embeddings.layernorm.weight"))
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.bias", "qformer.embeddings.layernorm.bias"))
    
    # 返回所有生成的重命名键值对列表
    return rename_keys
# 从字典中删除旧键，并将其对应的值赋给变量 val
def rename_key(dct, old, new):
    val = dct.pop(old)
    # 将旧键的值添加到字典中作为新键的值
    dct[new] = val


# 从状态字典中读取 Q、V 偏置，并设置到新的位置
def read_in_q_v_bias(state_dict, config):
    # 遍历视觉模型的隐藏层次数
    for i in range(config.vision_config.num_hidden_layers):
        # 从状态字典中移除原始 Q 偏置和 V 偏置
        q_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.v_bias")

        # 创建新的 QKV 偏置，按照特定顺序连接 Q 偏置、零张量、V 偏置
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        # 将新的 QKV 偏置设置到状态字典中的相应位置
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias


# 根据模型名称获取相应的配置信息和图像大小
def get_blip2_config(model_name):
    # 根据模型名称确定图像大小
    image_size = 364 if "coco" in model_name else 224
    # 获取视觉配置信息并转换为字典格式
    vision_config = InstructBlipVisionConfig(image_size=image_size).to_dict()

    # 根据模型名称选择文本配置信息
    if "t5-xl" in model_name:
        text_config = T5Config.from_pretrained("google/flan-t5-xl", dense_act_fn="gelu", bos_token_id=1).to_dict()
    elif "t5-xxl" in model_name:
        text_config = T5Config.from_pretrained("google/flan-t5-xxl", dense_act_fn="gelu", bos_token_id=1).to_dict()
    elif "vicuna-7b" in model_name:
        text_config = LlamaConfig.from_pretrained("decapoda-research/llama-7b-hf", vocab_size=32001).to_dict()
    elif "vicuna-13b" in model_name:
        text_config = LlamaConfig.from_pretrained("decapoda-research/llama-13b-hf", vocab_size=32001).to_dict()
    else:
        # 若模型名称不受支持，则引发值错误
        raise ValueError("Model name not supported")

    # Q-Former 模型配置信息，包含特殊的 "[DEC]" 标记，词汇大小为 30522 + 1
    qformer_config = InstructBlipQFormerConfig(vocab_size=30523).to_dict()
    # 构建并返回包含所有配置信息的 InstructBlipConfig 对象
    config = InstructBlipConfig(vision_config=vision_config, text_config=text_config, qformer_config=qformer_config)

    return config, image_size


# 使用 Torch 不计梯度上下文管理器，将 BLIP2 模型权重转换为 Transformers 设计
@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    # 使用 AutoTokenizer 从预训练模型中加载 Q-Former 令牌器，并添加特殊标记 "[DEC]"
    qformer_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", truncation_side="left")
    qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    # 根据模型名称选择相应的分词器
    if "t5" in model_name:
        tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xl", truncation_side="left")
    elif "vicuna" in model_name:
        # 如果模型名称包含"vicuna"
        
        # 使用快速加载LLAMA tokenizer，设定左截断、BOS和UNK特殊token
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "huggyllama/llama-7b", truncation_side="left", bos_token="</s>", unk_token="</s>"
        )
        
        # 添加[PAD]特殊token到tokenizer
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 获取BLIP2模型配置和图像尺寸
    config, image_size = get_blip2_config(model_name)
    
    # 加载BLIP2模型进行推理
    hf_model = InstructBlipForConditionalGeneration(config).eval()

    # 将模型名称映射到原始模型名称和类型
    model_name_to_original = {
        "instructblip-vicuna-7b": ("blip2_vicuna_instruct", "vicuna7b"),
        "instructblip-vicuna-13b": ("blip2_vicuna_instruct", "vicuna13b"),
        "instructblip-flan-t5-xl": ("blip2_t5_instruct", "flant5xl"),
        "instructblip-flan-t5-xxl": ("blip2_t5_instruct", "flant5xxl"),
    }

    # 根据模型名称获取原始模型名称和类型
    name, type = model_name_to_original[model_name]

    # 加载原始模型
    print("Loading original model...")
    # 检查GPU是否可用，选择合适的设备
    hf_model_device = "cuda:1" if torch.cuda.is_available() else "cpu"
    lavis_device = "cuda:2" if torch.cuda.is_available() else "cpu"
    # 加载模型及预处理器
    original_model, vis_processors, _ = load_model_and_preprocess(
        name=name, model_type=type, is_eval=True, device=lavis_device
    )
    original_model.eval()
    print("Done!")

    # 更新state dict中的键名
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 一些键名可以进行有效的重命名
    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if key.startswith("Qformer.bert"):
            key = key.replace("Qformer.bert", "qformer")
        if "attention.self" in key:
            key = key.replace("self", "attention")
        if "llm_proj" in key:
            key = key.replace("llm_proj", "language_projection")
        if "t5_proj" in key:
            key = key.replace("t5_proj", "language_projection")
        if key.startswith("llm_model"):
            key = key.replace("llm_model", "language_model")
        if key.startswith("t5"):
            key = key.replace("t5", "language")
        state_dict[key] = val

    # 读取qv biases
    read_in_q_v_bias(state_dict, config)

    # 注意: 默认情况下，权重以torch.float32加载
    # 使用state_dict加载模型权重
    hf_model.load_state_dict(state_dict, strict=True)

    # 加载演示图像
    image = load_demo_image()
    # 设定提示语句
    prompt = "What is unusual about this image?"

    # 创建图像处理器
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
    )
    # 创建一个 InstructBlipProcessor 实例，用于处理图像和文本数据
    processor = InstructBlipProcessor(
        image_processor=image_processor,  # 图像处理器对象
        tokenizer=tokenizer,  # 文本 tokenizer 对象
        qformer_tokenizer=qformer_tokenizer,  # qformer tokenizer 对象
    )
    
    # 使用 processor 处理图像和文本数据，将结果转移到指定的 hf_model_device 上
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(hf_model_device)

    # 确保 processor 创建的像素值与原始像素值完全相同
    original_pixel_values = vis_processors["eval"](image).unsqueeze(0).to(lavis_device)
    pixel_values = inputs.pixel_values
    assert torch.allclose(original_pixel_values.to(pixel_values.device), pixel_values)

    # 将 original_model 和 hf_model 移动到指定的设备上
    original_model.to(lavis_device)
    hf_model.to(hf_model_device)
    
    # 使用 torch.no_grad() 上下文管理器，避免计算梯度
    with torch.no_grad():
        # 根据 model_name 的不同选择不同的计算方式
        if "vicuna" in model_name:
            # 使用 original_model 和 hf_model 分别计算 logits
            original_logits = original_model({"image": original_pixel_values, "text_input": [prompt]}).logits
            logits = hf_model(**inputs).logits
        else:
            # 使用 original_model 和 hf_model 分别计算 logits，并为 HF 模型提供额外的文本输出信息
            original_logits = original_model(
                {"image": original_pixel_values, "text_input": [prompt], "text_output": ["\n"]}
            ).logits
            # 生成用于 HF 模型的标签输入 ids
            label_input_ids = tokenizer("\n", return_tensors="pt").input_ids.to(hf_model_device)
            # 将标签中的 pad_token_id 替换为 -100
            labels = label_input_ids.masked_fill(label_input_ids == tokenizer.pad_token_id, -100)
            logits = hf_model(**inputs, labels=labels).logits

    # 打印 original_logits 和 logits 的前几个值
    print("First values of original logits:", original_logits[0, :3, :3])
    print("First values of HF logits:", logits[0, :3, :3])

    # 断言 original_logits 和 logits 的形状相同
    assert original_logits.shape == logits.shape
    # 根据 model_name 的不同设置允许的误差范围
    atol = 1e-4 if "vicuna" in model_name else 1e-5
    # 断言 original_logits 和 logits 的数值在指定的误差范围内相等
    assert torch.allclose(original_logits.to(logits.device), logits, atol=atol)
    print("Looks ok!")

    # 使用 original_model 生成文本输出
    print("Generating with original model...")
    original_outputs = original_model.generate({"image": original_pixel_values, "prompt": prompt}, num_beams=5)

    # 将 HF 模型生成的输出文本
    print("Generating with HF model...")
    outputs = hf_model.generate(
        **inputs,
        do_sample=False,
        num_beams=5,
        max_length=256,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1,
    )
    # 如果 model_name 包含 "vicuna"，将输出 id 为 0 的位置替换为 2（eos_token_id）
    if "vicuna" in model_name:
        outputs[outputs == 0] = 2
    print("Original generation:", original_outputs)
    
    # 使用 processor 批量解码 HF 模型的输出，跳过特殊 token
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    # 去除输出文本中的首尾空格
    output_text = [text.strip() for text in output_text]
    print("HF generation:", output_text)

    # 如果指定了 pytorch_dump_folder_path，保存 processor 和 hf_model 的预训练参数
    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)

    # 如果指定了 push_to_hub，将 processor 和 hf_model 推送到指定的 Hub 路径
    if push_to_hub:
        processor.push_to_hub(f"Salesforce/{model_name}")
        hf_model.push_to_hub(f"Salesforce/{model_name}")
if __name__ == "__main__":
    # 如果脚本作为主程序运行，则执行以下代码块

    # 创建参数解析器对象
    parser = argparse.ArgumentParser()

    # 定义模型名称的选项列表
    choices = [
        "instructblip-vicuna-7b",
        "instructblip-vicuna-13b",
        "instructblip-flan-t5-xl",
        "instructblip-flan-t5-xxl",
    ]

    # 添加模型名称参数
    parser.add_argument(
        "--model_name",
        default="instructblip-flan-t5-xl",  # 默认模型名称为 instructblip-flan-t5-xl
        choices=choices,  # 可选的模型名称列表
        type=str,
        help="Path to hf config.json of model to convert",  # 参数帮助信息
    )

    # 添加 PyTorch 模型输出文件夹路径参数
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,  # 默认值为 None
        type=str,
        help="Path to the output PyTorch model.",  # 参数帮助信息
    )

    # 添加是否推送到 Hub 的标志参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",  # 如果设置，则为 True，否则为 False
        help="Whether to push the model and processor to the hub after converting",  # 参数帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_blip2_checkpoint，传递解析后的参数
    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```