# `.\models\instructblip\convert_instructblip_original_to_pytorch.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本使用此文件，除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
"""
从原始存储库转换 InstructBLIP 检查点。

URL: https://github.com/salesforce/LAVIS/tree/main/projects/instructblip
"""

import argparse

import requests
import torch

# 安装 salesforce-lavis 库
# 实际上我安装了一个稍微修改过的版本：pip3 install git+https://github.com/nielsrogge/LAVIS.git@fix_lavis_float32（还有 fix_lavis 分支）
# 还要注意：要转换 Vicuna 检查点，我们必须在 lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml 中包含 /home/niels/python_projects/checkpoints/FastChat/vicuna-7b
# 对于 Vicuna-13b 也是一样的
from lavis.models import load_model_and_preprocess
from PIL import Image

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
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD


def load_demo_image():
    # 加载演示图片
    url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
    # 从 URL 获取图片并转换为 RGB 格式
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    return image


# 这里列出所有要重命名的键（原始名称在左侧，我们的名称在右侧）
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
    # 遍历视觉编码器中的隐藏层，生成需要重命名的键值对列表
    for i in range(config.vision_config.num_hidden_layers):
        # 重命名视觉编码器中第 i 层的 norm1.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.weight", f"vision_model.encoder.layers.{i}.layer_norm1.weight"))
        # 重命名视觉编码器中第 i 层的 norm1.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.norm1.bias", f"vision_model.encoder.layers.{i}.layer_norm1.bias"))
        # 重命名视觉编码器中第 i 层的 norm2.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.weight", f"vision_model.encoder.layers.{i}.layer_norm2.weight"))
        # 重命名视觉编码器中第 i 层的 norm2.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.norm2.bias", f"vision_model.encoder.layers.{i}.layer_norm2.bias"))
        # 重命名视觉编码器中第 i 层的 attn.qkv.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.qkv.weight", f"vision_model.encoder.layers.{i}.self_attn.qkv.weight"))
        # 重命名视觉编码器中第 i 层的 attn.proj.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.weight", f"vision_model.encoder.layers.{i}.self_attn.projection.weight",))
        # 重命名视觉编码器中第 i 层的 attn.proj.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.attn.proj.bias", f"vision_model.encoder.layers.{i}.self_attn.projection.bias"))
        # 重命名视觉编码器中第 i 层的 mlp.fc1.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.weight", f"vision_model.encoder.layers.{i}.mlp.fc1.weight"))
        # 重命名视觉编码器中第 i 层的 mlp.fc1.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc1.bias", f"vision_model.encoder.layers.{i}.mlp.fc1.bias"))
        # 重命名视觉编码器中第 i 层的 mlp.fc2.weight
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.weight", f"vision_model.encoder.layers.{i}.mlp.fc2.weight"))
        # 重命名视觉编码器中第 i 层的 mlp.fc2.bias
        rename_keys.append((f"visual_encoder.blocks.{i}.mlp.fc2.bias", f"vision_model.encoder.layers.{i}.mlp.fc2.bias"))

    # QFormer
    # 重命名 QFormer 的 bert.embeddings.LayerNorm.weight
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.weight", "qformer.embeddings.layernorm.weight"))
    # 重命名 QFormer 的 bert.embeddings.LayerNorm.bias
    rename_keys.append(("Qformer.bert.embeddings.LayerNorm.bias", "qformer.embeddings.layernorm.bias"))

    # 返回重命名后的键值对列表
    return rename_keys
# 重命名字典中的键值对
def rename_key(dct, old, new):
    # 弹出旧键对应的值
    val = dct.pop(old)
    # 将值与新键组成新的键值对
    dct[new] = val


# 读取 Q、V 偏置项并设置到状态字典中
def read_in_q_v_bias(state_dict, config):
    # 遍历隐藏层
    for i in range(config.vision_config.num_hidden_layers):
        # 读取原始的 Q 和 V 偏置项
        q_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"visual_encoder.blocks.{i}.attn.v_bias")

        # 组合 QKV 偏置项
        qkv_bias = torch.cat((q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias))
        # 设置偏置项到状态字典中
        state_dict[f"vision_model.encoder.layers.{i}.self_attn.qkv.bias"] = qkv_bias


# 获取 Blip2 模型的配置信息
def get_blip2_config(model_name):
    # 根据模型名确定图像大小
    image_size = 364 if "coco" in model_name else 224
    # 获取视觉配置
    vision_config = InstructBlipVisionConfig(image_size=image_size).to_dict()

    # 确保模型设置了正确的 bos_token_id 和 eos_token_id（对生成很重要）
    if "t5-xl" in model_name:
        text_config = T5Config.from_pretrained("google/flan-t5-xl", dense_act_fn="gelu", bos_token_id=1).to_dict()
    elif "t5-xxl" in model_name:
        text_config = T5Config.from_pretrained("google/flan-t5-xxl", dense_act_fn="gelu", bos_token_id=1).to_dict()
    elif "vicuna-7b" in model_name:
        text_config = LlamaConfig.from_pretrained("decapoda-research/llama-7b-hf", vocab_size=32001).to_dict()
    elif "vicuna-13b" in model_name:
        text_config = LlamaConfig.from_pretrained("decapoda-research/llama-13b-hf", vocab_size=32001).to_dict()
    else:
        raise ValueError("Model name not supported")

    # 对于 Q-Former 模型，作者在词汇表中添加了一个特殊的 "[DEC]" 标记，因此词汇表大小为 30522 + 1
    qformer_config = InstructBlipQFormerConfig(vocab_size=30523).to_dict()
    # 组合配置信息
    config = InstructBlipConfig(vision_config=vision_config, text_config=text_config, qformer_config=qformer_config)

    return config, image_size


# 将 Blip2 模型的权重转换为 Transformers 设计
@torch.no_grad()
def convert_blip2_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to Transformers design.
    """
    # 使用 AutoTokenizer 加载 Q-Former 模型的 tokenizer，并添加特殊标记
    qformer_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
    qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})

    # 如果模型名中包含 "t5"，则使用 T5TokenizerFast 加载 tokenizer
    if "t5" in model_name:
        tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-xl", truncation_side="left")
    # 如果模型名称中包含"vicuna"，则执行以下操作
    elif "vicuna" in model_name:
        # 在原始实现中使用以下代码:
        # 从预训练模型"huggyllama/llama-7b"中创建LLamaTokenizer对象，不使用快速模式，截断在左侧
        # tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=False, truncation_side="left")
        # 添加特殊标记"[PAD]"
        # tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # 添加特殊标记"</s>"
        # tokenizer.add_special_tokens({"bos_token": "</s>"})
        # 添加特殊标记"</s>"
        # tokenizer.add_special_tokens({"eos_token": "</s>"})
        # 添加特殊标记"</s>"
        # tokenizer.add_special_tokens({"unk_token": "</s>"})
        # 从预训练模型"huggyllama/llama-7b"中创建LlamaTokenizerFast对象，截断在左侧，设置起始标记和未知标记
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "huggyllama/llama-7b", truncation_side="left", bos_token="</s>", unk_token="</s>"
        )
        # 添加特殊标记"[PAD]"
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 获取模型配置和图像大小
    config, image_size = get_blip2_config(model_name)
    # 创建InstructBlipForConditionalGeneration模型并设置为评估模式
    hf_model = InstructBlipForConditionalGeneration(config).eval()

    # 模型名称到原始模型的映射
    model_name_to_original = {
        "instructblip-vicuna-7b": ("blip2_vicuna_instruct", "vicuna7b"),
        "instructblip-vicuna-13b": ("blip2_vicuna_instruct", "vicuna13b"),
        "instructblip-flan-t5-xl": ("blip2_t5_instruct", "flant5xl"),
        "instructblip-flan-t5-xxl": ("blip2_t5_instruct", "flant5xxl"),
    }

    # 获取模型名称对应的原始模型名称和类型
    name, type = model_name_to_original[model_name]

    # 加载原始模型
    print("Loading original model...")
    # 根据CUDA是否可用选择设备
    hf_model_device = "cuda:1" if torch.cuda.is_available() else "cpu"
    lavis_device = "cuda:2" if torch.cuda.is_available() else "cpu"
    original_model, vis_processors, _ = load_model_and_preprocess(
        name=name, model_type=type, is_eval=True, device=lavis_device
    )
    original_model.eval()
    print("Done!")

    # 更新状态字典键
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 一些键可以高效地重命名
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

    # 读取qv偏置
    read_in_q_v_bias(state_dict, config)

    # 注意: 权重默认以torch.float32加载
    hf_model.load_state_dict(state_dict, strict=True)

    # 加载演示图像
    image = load_demo_image()
    prompt = "What is unusual about this image?"

    # 创建图像处理器
    image_processor = BlipImageProcessor(
        size={"height": image_size, "width": image_size}, image_mean=OPENAI_CLIP_MEAN, image_std=OPENAI_CLIP_STD
    )
    # 创建一个 InstructBlipProcessor 处理器对象，传入图像处理器、分词器和 qformer 分词器
    processor = InstructBlipProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        qformer_tokenizer=qformer_tokenizer,
    )
    # 使用处理器处理输入数据，将其转换为 PyTorch 张量，并移到指定设备上
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(hf_model_device)

    # 确保处理器创建的像素值与原始像素值完全相同
    original_pixel_values = vis_processors["eval"](image).unsqueeze(0).to(lavis_device)
    pixel_values = inputs.pixel_values
    assert torch.allclose(original_pixel_values.to(pixel_values.device), pixel_values)

    # 将原始模型和 HF 模型移动到指定设备上
    original_model.to(lavis_device)
    hf_model.to(hf_model_device)
    with torch.no_grad():
        # 根据模型名称选择不同的逻辑处理
        if "vicuna" in model_name:
            original_logits = original_model({"image": original_pixel_values, "text_input": [prompt]}).logits
            logits = hf_model(**inputs).logits
        else:
            original_logits = original_model(
                {"image": original_pixel_values, "text_input": [prompt], "text_output": ["\n"]}
            ).logits
            label_input_ids = tokenizer("\n", return_tensors="pt").input_ids.to(hf_model_device)
            labels = label_input_ids.masked_fill(label_input_ids == tokenizer.pad_token_id, -100)
            logits = hf_model(**inputs, labels=labels).logits

    # 打印原始 logits 和 HF logits 的前几个值
    print("First values of original logits:", original_logits[0, :3, :3])
    print("First values of HF logits:", logits[0, :3, :3])

    # 断言原始 logits 和 HF logits 的形状相同，并进行数值比较
    assert original_logits.shape == logits.shape
    atol = 1e-4 if "vicuna" in model_name else 1e-5
    assert torch.allclose(original_logits.to(logits.device), logits, atol=atol)
    print("Looks ok!")

    # 使用原始模型生成文本
    print("Generating with original model...")
    original_outputs = original_model.generate({"image": original_pixel_values, "prompt": prompt}, num_beams=5)

    # 使用 HF 模型生成文本
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
    if "vicuna" in model_name:
        # 将输出 id 0 转换为 2（eos_token_id）
        outputs[outputs == 0] = 2
    print("Original generation:", original_outputs)
    # 对 HF 生成的文本进行解码和处理
    output_text = processor.batch_decode(outputs, skip_special_tokens=True)
    output_text = [text.strip() for text in output_text]
    print("HF generation:", output_text)

    # 如果指定了 PyTorch 模型保存路径，则保存处理器和 HF 模型
    if pytorch_dump_folder_path is not None:
        processor.save_pretrained(pytorch_dump_folder_path)
        hf_model.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 Hub，则将处理器和 HF 模型推送到指定 Hub
    if push_to_hub:
        processor.push_to_hub(f"Salesforce/{model_name}")
        hf_model.push_to_hub(f"Salesforce/{model_name}")
# 如果当前脚本被直接执行，则执行以下代码
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 定义可选的模型名称列表
    choices = [
        "instructblip-vicuna-7b",
        "instructblip-vicuna-13b",
        "instructblip-flan-t5-xl",
        "instructblip-flan-t5-xxl",
    ]
    # 添加模型名称参数，包括默认值、可选值、类型和帮助信息
    parser.add_argument(
        "--model_name",
        default="instructblip-flan-t5-xl",
        choices=choices,
        type=str,
        help="Path to hf config.json of model to convert",
    )
    # 添加输出 PyTorch 模型路径参数，默认为 None
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加是否推送到 Hub 的参数
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数，将 BLIP 模型转换为检查点
    convert_blip2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```