# `.\transformers\models\x_clip\convert_x_clip_original_pytorch_to_hf.py`

```
# 设置编码为UTF-8
# 版权声明
#
# 根据 Apache 许可证 2.0 版本 ("License") 授权
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按 "AS IS" 基础分发软件
# 没有任何形式的担保或条件，无论明示还是暗示
# 查看许可证以获取特定语言下的权限和限制

# 导入必要的库
import argparse
import gdown
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    CLIPTokenizer,
    CLIPTokenizerFast,
    VideoMAEImageProcessor,
    XCLIPConfig,
    XCLIPModel,
    XCLIPProcessor,
    XCLIPTextConfig,
    XCLIPVisionConfig,
)

# 获取 XCLIP 模型配置
def get_xclip_config(model_name, num_frames):
    text_config = XCLIPTextConfig()
    
    # 从模型名称中获取 patch 大小
    start_idx = model_name.find("patch")
    patch_size = int(model_name[start_idx + len("patch") : start_idx + len("patch") + 2])
    vision_config = XCLIPVisionConfig(patch_size=patch_size, num_frames=num_frames)
    
    if "large" in model_name:
        # 设置大型模型的文本和视觉配置
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
        vision_config.image_size = 336
    
    # 创建 XCLIP 配置对象
    config = XCLIPConfig.from_text_vision_configs(text_config, vision_config)
    
    if "large" in model_name:
        # 对于大型模型，设置投影维度
        config.projection_dim = 768
    
    return config

# 重命名模型中的键名
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
    # 根据输入的 name 字符串进行一系列替换操作，将其转换为对应的模型参数名
    def convert_name(name):
        # 如果 name 等于 "visual.class_embedding"，将其替换为 "vision_model.embeddings.class_embedding"
        if name == "visual.class_embedding":
            name = name.replace("visual.class_embedding", "vision_model.embeddings.class_embedding")
        # 如果 name 等于 "visual.positional_embedding"，将其替换为 "vision_model.embeddings.position_embedding.weight"
        if name == "visual.positional_embedding":
            name = name.replace("visual.positional_embedding", "vision_model.embeddings.position_embedding.weight")
        # 如果 name 以 "visual.transformer.resblocks" 开头，将其替换为 "vision_model.encoder.layers"
        if name.startswith("visual.transformer.resblocks"):
            name = name.replace("visual.transformer.resblocks", "vision_model.encoder.layers")
        # 如果 name 包含 "visual.conv1"，将其替换为 "vision_model.embeddings.patch_embedding"
        if "visual.conv1" in name:
            name = name.replace("visual.conv1", "vision_model.embeddings.patch_embedding")
        # 如果 name 包含 "visual.ln_pre"，将其替换为 "vision_model.pre_layernorm"
        if "visual.ln_pre" in name:
            name = name.replace("visual.ln_pre", "vision_model.pre_layernorm")
        # 如果 name 包含 "visual.ln_post"，将其替换为 "vision_model.post_layernorm"
        if "visual.ln_post" in name:
            name = name.replace("visual.ln_post", "vision_model.post_layernorm")
        # 如果 name 包含 "visual.proj"，将其替换为 "visual_projection.weight"
        if "visual.proj" in name:
            name = name.replace("visual.proj", "visual_projection.weight")
        # 如果 name 包含 "text_projection"，将其替换为 "text_projection.weight"
        if "text_projection" in name:
            name = name.replace("text_projection", "text_projection.weight")
        # 如果 name 包含 "prompts_visual_proj"，将其替换为 "prompts_visual_projection"
        if "prompts_visual_proj" in name:
            name = name.replace("prompts_visual_proj", "prompts_visual_projection")
        # 如果 name 包含 "prompts_visual_ln"，将其替换为 "prompts_visual_layernorm"
        if "prompts_visual_ln" in name:
            name = name.replace("prompts_visual_ln", "prompts_visual_layernorm")
        # 如果 name 等于 "mit.positional_embedding"，将其替换为 "mit.position_embedding"
        if name == "mit.positional_embedding":
            name = name.replace("positional", "position")
        # 如果 name 以 "mit.resblocks" 开头，将其替换为 "mit.encoder.layers"
        if name.startswith("mit.resblocks"):
            name = name.replace("mit.resblocks", "mit.encoder.layers")
        # 如果 name 以 "prompts_generator.norm" 开头，将其替换为 "prompts_generator.layernorm"
        if name.startswith("prompts_generator.norm"):
            name = name.replace("prompts_generator.norm", "prompts_generator.layernorm")
        # 返回转换后的 name
        return name
# 根据原始状态字典和配置信息转换状态字典
def convert_state_dict(orig_state_dict, config):
    return orig_state_dict

# 准备视频数据
def prepare_video(num_frames):
    # 根据帧数选择对应的视频文件名
    if num_frames == 8:
        filename = "eating_spaghetti_8_frames.npy"
    elif num_frames == 16:
        filename = "eating_spaghetti.npy"
    elif num_frames == 32:
        filename = "eating_spaghetti_32_frames.npy"
    # 从数据仓库下载对应的视频文件
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video",
        filename=filename,
        repo_type="dataset",
    )
    # 从文件加载视频数据
    video = np.load(file)
    # 将视频数据转换为列表并返回
    return list(video)

# 转换 XClip 模型检查点
def convert_xclip_checkpoint(model_name, pytorch_dump_folder_path=None, push_to_hub=False):
    model_to_url = {
        # 定义模型名称到预训练权重文件的URL映射
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

    # 获取指定模型名称对应的预训练权重文件的URL
    checkpoint_url = model_to_url[model_name]
    # 设置默认帧数为8，如果模型名称包含"16-frames"，则将帧数设为16
    num_frames = 8
    if "16-frames" in model_name:
        num_frames = 16
    # 如果模型名称中包含"shot"，则设置帧数为32
    elif "shot" in model_name:
        num_frames = 32

    # 根据模型名称和帧数获取配置信息
    config = get_xclip_config(model_name, num_frames)
    # 创建XCLIP模型
    model = XCLIPModel(config)
    # 设置模型为评估模式
    model.eval()

    # 如果checkpoint_url中包含"drive"，则下载特定文件名的模型参数
    if "drive" in checkpoint_url:
        output = "pytorch_model.bin"
        gdown.cached_download(checkpoint_url, output, quiet=False)
        state_dict = torch.load(output, map_location="cpu")["model"]
    else:
        # 从指定URL加载模型参数
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)["model"]

    # 转换模型参数并更新配置信息
    state_dict = convert_state_dict(state_dict, config)

    # 加载模型参数到XCLIP模型中
    model = XCLIPModel(config)
    # 加载模型参数，允许缺失和不匹配的键
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # 确保模型参数中不缺失指定的键
    assert missing_keys == ["text_model.embeddings.position_ids", "vision_model.embeddings.position_ids"]
    # 设置模型为评估模式
    model.eval()

    # 根据模型名称设置图像的尺寸
    size = 336 if model_name == "xclip-large-patch14-16-frames" else 224
    # 创建视频处理器对象
    image_processor = VideoMAEImageProcessor(size=size)
    # 初始化慢速tokenizer
    slow_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    # 初始化快速tokenizer
    fast_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    # 创建XCLIP处理器对象
    processor = XCLIPProcessor(image_processor=image_processor, tokenizer=fast_tokenizer)

    # 准备视频数据
    video = prepare_video(num_frames)
    # 处理输入数据
    inputs = processor(
        text=["playing sports", "eating spaghetti", "go shopping"], videos=video, return_tensors="pt", padding=True
    )

    # 打印像素值的形状
    print("Shape of pixel values:", inputs.pixel_values.shape)

    # 禁用梯度计算
    with torch.no_grad():
        outputs = model(**inputs)

    # 检查输出结果
    logits_per_video = outputs.logits_per_video
    probs = logits_per_video.softmax(dim=1)
    print("Probs:", probs)
    
    # 根据模型名称设置期望的概率输出
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
    # 根据模型名称选择相应的预期概率
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
    # 如果是零样本情况
    elif model_name == "xclip-base-patch16-zero-shot":
        expected_probs = torch.tensor([[3.5082e-04, 9.9785e-01, 1.7966e-03]])
    else:
        raise ValueError(f"Model name {model_name} not supported")
    # 确保模型输出的概率与预期概率在给定的容差范围内匹配
    assert torch.allclose(probs, expected_probs, atol=1e-3)
    # 打印提示信息
    print("Looks ok!")

    # 如果有 PyTorch 模型保存路径
    if pytorch_dump_folder_path is not None:
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        # 将模型保存到指定路径
        model.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 Hub
    if push_to_hub:
        # 打印提示信息
        print("Pushing model, processor and slow tokenizer files to the hub...")
        # 推送模型、处理器和慢速分词器文件到 Hub
        model.push_to_hub(model_name, organization="nielsr")
        processor.push_to_hub(model_name, organization="nielsr")
        slow_tokenizer.push_to_hub(model_name, organization="nielsr")
# 如果该模块是主程序
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--model_name", # 模型名称
        default="xclip-base-patch32", # 默认值为"xclip-base-patch32"
        type=str,  # 参数类型为字符串
        help="Name of the model.",  # 参数的帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",  # PyTorch模型输出目录的路径
        default=None,  # 默认值为None
        type=str,  # 参数类型为字符串
        help="Path to the output PyTorch model directory."  # 参数的帮助信息
    )
    parser.add_argument(
        "--push_to_hub",  # 推送到🤗 hub
        action="store_true",  # 如果设置了该参数，则为True；否则为False
        help="Whether or not to push the converted model to the 🤗 hub."  # 参数的帮助信息
    )

    # 解析参数
    args = parser.parse_args()
    # 调用函数将xclip检查点转换为PyTorch模型
    convert_xclip_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
```