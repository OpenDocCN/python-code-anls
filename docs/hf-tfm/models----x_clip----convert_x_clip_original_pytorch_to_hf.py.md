# `.\models\x_clip\convert_x_clip_original_pytorch_to_hf.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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

import argparse  # 导入命令行参数解析模块

import gdown  # 导入用于从Google Drive下载文件的模块
import numpy as np  # 导入用于数值计算的numpy库
import torch  # 导入PyTorch深度学习库
from huggingface_hub import hf_hub_download  # 导入从Hugging Face Hub下载模型的函数

from transformers import (  # 导入transformers库中的各类对象
    CLIPTokenizer,  # CLIP模型的分词器
    CLIPTokenizerFast,  # 加速版本的CLIP分词器
    VideoMAEImageProcessor,  # 视频和图像处理器
    XCLIPConfig,  # XCLIP模型的配置类
    XCLIPModel,  # XCLIP模型
    XCLIPProcessor,  # XCLIP处理器
    XCLIPTextConfig,  # XCLIP文本配置类
    XCLIPVisionConfig,  # XCLIP视觉配置类
)


def get_xclip_config(model_name, num_frames):
    text_config = XCLIPTextConfig()  # 创建一个XCLIP文本配置对象

    # 从模型名称中提取patch大小
    start_idx = model_name.find("patch")
    patch_size = int(model_name[start_idx + len("patch"): start_idx + len("patch") + 2])
    vision_config = XCLIPVisionConfig(patch_size=patch_size, num_frames=num_frames)  # 创建一个XCLIP视觉配置对象

    if "large" in model_name:
        # 如果模型名称中包含"large"，设置大模型的文本和视觉配置
        text_config.hidden_size = 768
        text_config.intermediate_size = 3072
        text_config.num_attention_heads = 12

        vision_config.hidden_size = 1024
        vision_config.intermediate_size = 4096
        vision_config.num_attention_heads = 16
        vision_config.num_hidden_layers = 24
        vision_config.mit_hidden_size = 768
        vision_config.mit_intermediate_size = 3072

    if model_name == "xclip-large-patch14-16-frames":
        # 如果模型名称是"xclip-large-patch14-16-frames"，设置特定的图片尺寸
        vision_config.image_size = 336

    config = XCLIPConfig.from_text_vision_configs(text_config, vision_config)  # 通过文本和视觉配置创建XCLIP模型的配置对象

    if "large" in model_name:
        config.projection_dim = 768  # 如果模型名称中包含"large"，设置投影维度为768

    return config  # 返回配置对象


def rename_key(name):
    # 文本编码器
    if name == "token_embedding.weight":
        name = name.replace("token_embedding.weight", "text_model.embeddings.token_embedding.weight")
    if name == "positional_embedding":
        name = name.replace("positional_embedding", "text_model.embeddings.position_embedding.weight")
    if "ln_1" in name:
        name = name.replace("ln_1", "layer_norm1")
    if "ln_2" in name:
        name = name.replace("ln_2", "layer_norm2")
    if "c_fc" in name:
        name = name.replace("c_fc", "fc1")
    if "c_proj" in name:
        name = name.replace("c_proj", "fc2")
    if name.startswith("transformer.resblocks"):
        name = name.replace("transformer.resblocks", "text_model.encoder.layers")
    if "attn.out_proj" in name and "message" not in name:
        name = name.replace("attn.out_proj", "self_attn.out_proj")
    if "ln_final" in name:
        name = name.replace("ln_final", "text_model.final_layer_norm")
    # 视觉编码器
    # 如果变量 name 等于 "visual.class_embedding"，则替换为 "vision_model.embeddings.class_embedding"
    if name == "visual.class_embedding":
        name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
    
    # 如果变量 name 等于 "visual.positional_embedding"，则替换为 "vision_model.embeddings.position_embedding.weight"
    if name == "visual.positional_embedding":
        name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
    
    # 如果变量 name 以 "visual.transformer.resblocks" 开头，则替换为 "vision_model.encoder.layers"
    if name.startswith("visual.transformer.resblocks"):
        name = name.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
    
    # 如果变量 name 中包含 "visual.conv1"，则替换为 "vision_model.embeddings.patch_embedding"
    if "visual.conv1" in name:
        name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
    
    # 如果变量 name 中包含 "visual.ln_pre"，则替换为 "vision_model.pre_layernorm"
    if "visual.ln_pre" in name:
        name = name.replace("visual.ln_pre", "vision_model.pre_layernorm")
    
    # 如果变量 name 中包含 "visual.ln_post"，则替换为 "vision_model.post_layernorm"
    if "visual.ln_post" in name:
        name = name.replace("visual.ln_post", "vision_model.post_layernorm")
    
    # 如果变量 name 中包含 "visual.proj"，则替换为 "visual_projection.weight"
    if "visual.proj" in name:
        name = name.replace("visual.proj", "visual_projection.weight")
    
    # 如果变量 name 中包含 "text_projection"，则替换为 "text_projection.weight"
    if "text_projection" in name:
        name = name.replace("text_projection", "text_projection.weight")
    
    # 如果变量 name 中包含 "prompts_visual_proj"，则替换为 "prompts_visual_projection"
    if "prompts_visual_proj" in name:
        name = name.replace("prompts_visual_proj", "prompts_visual_projection")
    
    # 如果变量 name 中包含 "prompts_visual_ln"，则替换为 "prompts_visual_layernorm"
    if "prompts_visual_ln" in name:
        name = name.replace("prompts_visual_ln", "prompts_visual_layernorm")
    
    # 如果变量 name 等于 "mit.positional_embedding"，则替换 "positional" 为 "position"
    if name == "mit.positional_embedding":
        name = name.replace("positional", "position")
    
    # 如果变量 name 以 "mit.resblocks" 开头，则替换为 "mit.encoder.layers"
    if name.startswith("mit.resblocks"):
        name = name.replace("mit.resblocks", "mit.encoder.layers")
    
    # 如果变量 name 以 "prompts_generator.norm" 开头，则替换为 "prompts_generator.layernorm"
    if name.startswith("prompts_generator.norm"):
        name = name.replace("prompts_generator.norm", "prompts_generator.layernorm")
    
    # 返回处理后的 name 变量
    return name
# 简单返回给定的原始状态字典，没有进行任何转换操作
def convert_state_dict(orig_state_dict, config):
    return orig_state_dict



# 准备视频数据，根据帧数选择对应的视频文件进行下载和加载
def prepare_video(num_frames):
    # 根据帧数选择对应的视频文件名
    if num_frames == 8:
        filename = "eating_spaghetti_8_frames.npy"
    elif num_frames == 16:
        filename = "eating_spaghetti.npy"
    elif num_frames == 32:
        filename = "eating_spaghetti_32_frames.npy"
    # 使用指定的repo_id和文件名从指定仓库类型（dataset）下载文件
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video",
        filename=filename,
        repo_type="dataset",
    )
    # 加载numpy数组中的视频数据
    video = np.load(file)
    # 将视频数据转换为列表形式并返回
    return list(video)



# 这是一个尚未实现的函数声明，用于将XClip模型的检查点转换为PyTorch格式
def convert_xclip_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    pass
    model_to_url = {
        # 定义一个字典，将模型名称映射到其对应的预训练模型下载地址
        # fully supervised kinetics-400 checkpoints
        "xclip-base-patch32": "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_32_8.pth",
        "xclip-base-patch32-16-frames": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_32_16.pth"
        ),
        "xclip-base-patch16": "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_16_8.pth",
        "xclip-base-patch16-16-frames": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k400_16_16.pth"
        ),
        "xclip-large-patch14": "https://drive.google.com/u/0/uc?id=1NUOImq0o5DlQTST17iIP3vG7DgmHQuCx&amp;export=download&amp;confirm=t&amp;uuid=b26caedc-88e2-473e-830a-9d158b653cdb",
        "xclip-large-patch14-16-frames": "https://drive.google.com/u/0/uc?id=1FOYgnJc097OJ4lGwtRCCydQyVPJEOH7d&amp;export=download&amp;confirm=t&amp;uuid=538fa810-e671-4050-b385-9a623f89804f",
        # fully supervised kinetics-600 checkpoints
        "xclip-base-patch16-kinetics-600": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k600_16_8.pth"
        ),
        "xclip-base-patch16-kinetics-600-16-frames": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/k600_16_16.pth"
        ),
        "xclip-large-patch14-kinetics-600": "https://drive.google.com/u/0/uc?id=1FV8C1INuM91sLAN4ImjzePLIlpMSihwV&amp;export=download&amp;confirm=t&amp;uuid=141d4977-4a65-44ae-864f-4b0c19f838be",
        # few shot
        "xclip-base-patch16-hmdb-2-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_2.pth"
        ),
        "xclip-base-patch16-hmdb-4-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_4.pth"
        ),
        "xclip-base-patch16-hmdb-8-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_8.pth"
        ),
        "xclip-base-patch16-hmdb-16-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_hmdb_16.pth"
        ),
        "xclip-base-patch16-ucf-2-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_2.pth"
        ),
        "xclip-base-patch16-ucf-4-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_4.pth"
        ),
        "xclip-base-patch16-ucf-8-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_8.pth"
        ),
        "xclip-base-patch16-ucf-16-shot": (
            "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/few_ucf_16.pth"
        ),
        # zero shot
        "xclip-base-patch16-zero-shot": "https://github.com/nbl97/X-CLIP_Model_Zoo/releases/download/v1.0/zero.pth",
    }

    # 根据给定的模型名称获取相应的预训练模型下载地址
    checkpoint_url = model_to_url[model_name]
    
    # 默认帧数设置为8帧，如果模型名称中包含"16-frames"，则设置为16帧
    num_frames = 8
    if "16-frames" in model_name:
        num_frames = 16
    # 如果模型名称中包含 "shot"，设定帧数为32
    elif "shot" in model_name:
        num_frames = 32

    # 根据模型名称获取对应的配置信息
    config = get_xclip_config(model_name, num_frames)
    # 创建 XCLIPModel 模型对象
    model = XCLIPModel(config)
    # 将模型设置为评估模式
    model.eval()

    # 如果 checkpoint_url 中包含 "drive"
    if "drive" in checkpoint_url:
        # 设置输出文件名为 "pytorch_model.bin"
        output = "pytorch_model.bin"
        # 使用 gdown 下载 checkpoint_url 对应的文件到 output
        gdown.cached_download(checkpoint_url, output, quiet=False)
        # 从下载的文件中加载模型状态字典到 state_dict，并指定在 CPU 上加载
        state_dict = torch.load(output, map_location="cpu")["model"]
    else:
        # 从 checkpoint_url 加载预训练模型状态字典到 state_dict
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]

    # 转换加载的状态字典以匹配当前配置
    state_dict = convert_state_dict(state_dict, config)

    # 创建 XCLIPModel 模型对象
    model = XCLIPModel(config)
    # 根据加载的状态字典加载模型参数，允许缺少键，严格性设置为 False
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 断言确保缺少的键为指定的列表
    assert missing_keys == ["text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"]
    # 将模型设置为评估模式
    model.eval()

    # 根据模型名称选择图片处理的尺寸
    size = 336 if model_name == "xclip-large-patch14-16-frames" else 224
    # 创建视频多模态自动编码器图像处理器对象，指定图片尺寸
    image_processor = VideoMAEImageProcessor(size=size)
    # 从预训练模型加载 CLIPTokenizer 对象
    slow_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # 从预训练模型加载 CLIPTokenizerFast 对象
    fast_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    # 创建 XCLIPProcessor 处理器对象，指定图像处理器和快速分词器
    processor = XCLIPProcessor(image_processor=image_processor, tokenizer=fast_tokenizer)

    # 准备视频数据，获取输入参数
    video = prepare_video(num_frames)
    # 使用处理器处理文本和视频输入数据，返回 PyTorch 张量格式，进行填充
    inputs = processor(
        text=["playing sports", "eating spaghetti", "go shopping"], videos=video, return_tensors="pt", padding=True
    )

    # 打印像素值的形状信息
    print("Shape of pixel values:", inputs.pixel_values.shape)

    # 禁用梯度计算
    with torch.no_grad():
        # 使用模型进行推断，获取输出
        outputs = model(**inputs)

    # 验证输出结果
    logits_per_video = outputs.logits_per_video
    # 对 logits 进行 softmax 处理得到概率
    probs = logits_per_video.softmax(dim=1)
    # 打印概率值
    print("Probs:", probs)

    # 根据模型名称选择预期的概率张量
    if model_name == "xclip-base-patch32":
        expected_probs = torch.tensor([[0.0019, 0.9951, 0.0030]])
    elif model_name == "xclip-base-patch32-16-frames":
        expected_probs = torch.tensor([[7.0999e-04, 9.9883e-01, 4.5580e-04]])
    elif model_name == "xclip-base-patch16":
        expected_probs = torch.tensor([[0.0083, 0.9681, 0.0236]])
    elif model_name == "xclip-base-patch16-16-frames":
        expected_probs = torch.tensor([[7.6937e-04, 9.9728e-01, 1.9473e-03]])
    elif model_name == "xclip-large-patch14":
        expected_probs = torch.tensor([[0.0062, 0.9864, 0.0075]])
    elif model_name == "xclip-large-patch14-16-frames":
        expected_probs = torch.tensor([[3.3877e-04, 9.9937e-01, 2.8888e-04]])
    elif model_name == "xclip-base-patch16-kinetics-600":
        expected_probs = torch.tensor([[0.0555, 0.8914, 0.0531]])
    elif model_name == "xclip-base-patch16-kinetics-600-16-frames":
        expected_probs = torch.tensor([[3.8554e-04, 9.9929e-01, 3.2754e-04]])
    elif model_name == "xclip-large-patch14-kinetics-600":
        expected_probs = torch.tensor([[0.0036, 0.9920, 0.0045]])
    elif model_name == "xclip-base-patch16-hmdb-2-shot":
        expected_probs = torch.tensor([[7.1890e-06, 9.9994e-01, 5.6559e-05]])
    # 根据模型名称选择预期的概率张量
    elif model_name == "xclip-base-patch16-hmdb-4-shot":
        expected_probs = torch.tensor([[1.0320e-05, 9.9993e-01, 6.2435e-05]])
    elif model_name == "xclip-base-patch16-hmdb-8-shot":
        expected_probs = torch.tensor([[4.1377e-06, 9.9990e-01, 9.8386e-05]])
    elif model_name == "xclip-base-patch16-hmdb-16-shot":
        expected_probs = torch.tensor([[4.1347e-05, 9.9962e-01, 3.3411e-04]])
    elif model_name == "xclip-base-patch16-ucf-2-shot":
        expected_probs = torch.tensor([[8.5857e-05, 9.9928e-01, 6.3291e-04]])
    elif model_name == "xclip-base-patch16-ucf-4-shot":
        expected_probs = torch.tensor([[8.5857e-05, 9.9928e-01, 6.3291e-04]])
    elif model_name == "xclip-base-patch16-ucf-8-shot":
        expected_probs = torch.tensor([[0.0027, 0.9904, 0.0070]])
    elif model_name == "xclip-base-patch16-ucf-16-shot":
        expected_probs = torch.tensor([[9.8219e-04, 9.9593e-01, 3.0863e-03]])
    # zero shot
    elif model_name == "xclip-base-patch16-zero-shot":
        expected_probs = torch.tensor([[3.5082e-04, 9.9785e-01, 1.7966e-03]])
    else:
        raise ValueError(f"Model name {model_name} not supported")

    # 使用assert语句检查模型输出的概率值与预期概率张量的接近程度
    assert torch.allclose(probs, expected_probs, atol=1e-3)
    # 输出确认信息
    print("Looks ok!")

    # 如果指定了PyTorch模型保存路径，则保存模型
    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到hub，则推送模型、processor和slow tokenizer文件到指定的组织
    if push_to_hub:
        print("Pushing model, processor and slow tokenizer files to the hub...")
        model.push_to_hub(model_name, organization="nielsr")
        processor.push_to_hub(model_name, organization="nielsr")
        slow_tokenizer.push_to_hub(model_name, organization="nielsr")
if __name__ == "__main__":
    # 如果作为主程序执行，进入主程序逻辑

    parser = argparse.ArgumentParser()
    # 创建参数解析器对象

    # Required parameters
    parser.add_argument(
        "--model_name",
        default="xclip-base-patch32",
        type=str,
        help="Name of the model.",
    )
    # 添加模型名称参数，默认为"xclip-base-patch32"，类型为字符串，用于指定模型名称

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    # 添加PyTorch模型输出目录路径参数，默认为None，类型为字符串，用于指定PyTorch模型的输出目录路径

    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    # 添加是否推送到🤗 hub的参数，使用store_true来标记是否推送模型到hub

    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数进行模型转换，传入命令行解析后的参数
    convert_xclip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```