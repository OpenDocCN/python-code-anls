# `.\models\timesformer\convert_timesformer_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明及许可信息，指明代码使用的许可协议和版权归属
# 导入转换 TimeSformer 检查点所需的库和模块

import argparse  # 导入用于解析命令行参数的库
import json  # 导入处理 JSON 格式数据的库

import gdown  # 导入用于从 Google Drive 下载文件的库
import numpy as np  # 导入处理数值和数组的库
import torch  # 导入 PyTorch 深度学习框架
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载资源的函数

from transformers import TimesformerConfig, TimesformerForVideoClassification, VideoMAEImageProcessor  # 导入 TimeSformer 模型所需的配置、模型和处理器类


def get_timesformer_config(model_name):
    config = TimesformerConfig()  # 创建一个 TimeSformer 的配置对象

    if "large" in model_name:
        config.num_frames = 96  # 如果模型名包含 'large'，设置帧数为 96

    if "hr" in model_name:
        config.num_frames = 16  # 如果模型名包含 'hr'，设置帧数为 16
        config.image_size = 448  # 同时设置图像尺寸为 448

    repo_id = "huggingface/label-files"
    if "k400" in model_name:
        config.num_labels = 400  # 如果模型名包含 'k400'，设置标签数为 400
        filename = "kinetics400-id2label.json"  # 设置要下载的文件名为 kinetics400-id2label.json
    elif "k600" in model_name:
        config.num_labels = 600  # 如果模型名包含 'k600'，设置标签数为 600
        filename = "kinetics600-id2label.json"  # 设置要下载的文件名为 kinetics600-id2label.json
    elif "ssv2" in model_name:
        config.num_labels = 174  # 如果模型名包含 'ssv2'，设置标签数为 174
        filename = "something-something-v2-id2label.json"  # 设置要下载的文件名为 something-something-v2-id2label.json
    else:
        raise ValueError("Model name should either contain 'k400', 'k600' or 'ssv2'.")  # 如果模型名不符合预期，则引发错误
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))  # 从 Hugging Face Hub 下载并加载 JSON 格式的标签映射数据
    id2label = {int(k): v for k, v in id2label.items()}  # 将标签映射数据中的键转换为整数类型
    config.id2label = id2label  # 将加载的标签映射数据设置为配置对象的 id2label 属性
    config.label2id = {v: k for k, v in id2label.items()}  # 创建反向映射，从标签到 ID 的映射

    return config  # 返回配置对象


def rename_key(name):
    if "encoder." in name:
        name = name.replace("encoder.", "")  # 替换模型参数名中的 'encoder.' 为 ''
    if "cls_token" in name:
        name = name.replace("cls_token", "timesformer.embeddings.cls_token")  # 替换模型参数名中的 'cls_token' 为 'timesformer.embeddings.cls_token'
    if "pos_embed" in name:
        name = name.replace("pos_embed", "timesformer.embeddings.position_embeddings")  # 替换模型参数名中的 'pos_embed' 为 'timesformer.embeddings.position_embeddings'
    if "time_embed" in name:
        name = name.replace("time_embed", "timesformer.embeddings.time_embeddings")  # 替换模型参数名中的 'time_embed' 为 'timesformer.embeddings.time_embeddings'
    if "patch_embed.proj" in name:
        name = name.replace("patch_embed.proj", "timesformer.embeddings.patch_embeddings.projection")  # 替换模型参数名中的 'patch_embed.proj' 为 'timesformer.embeddings.patch_embeddings.projection'
    if "patch_embed.norm" in name:
        name = name.replace("patch_embed.norm", "timesformer.embeddings.norm")  # 替换模型参数名中的 'patch_embed.norm' 为 'timesformer.embeddings.norm'
    if "blocks" in name:
        name = name.replace("blocks", "timesformer.encoder.layer")  # 替换模型参数名中的 'blocks' 为 'timesformer.encoder.layer'
    if "attn.proj" in name:
        name = name.replace("attn.proj", "attention.output.dense")  # 替换模型参数名中的 'attn.proj' 为 'attention.output.dense'
    if "attn" in name and "bias" not in name and "temporal" not in name:
        name = name.replace("attn", "attention.self")  # 替换模型参数名中的 'attn' 为 'attention.self'，排除包含 'bias' 和 'temporal' 的情况
    if "attn" in name and "temporal" not in name:
        name = name.replace("attn", "attention.attention")  # 替换模型参数名中的 'attn' 为 'attention.attention'，排除包含 'temporal' 的情况
    # 检查字符串 "temporal_norm1" 是否在变量 name 中
    if "temporal_norm1" in name:
        # 如果是，则将字符串 "temporal_norm1" 替换为 "temporal_layernorm"
        name = name.replace("temporal_norm1", "temporal_layernorm")

    # 检查字符串 "temporal_attn.proj" 是否在变量 name 中
    if "temporal_attn.proj" in name:
        # 如果是，则将字符串 "temporal_attn" 替换为 "temporal_attention.output.dense"
        name = name.replace("temporal_attn", "temporal_attention.output.dense")

    # 检查字符串 "temporal_fc" 是否在变量 name 中
    if "temporal_fc" in name:
        # 如果是，则将字符串 "temporal_fc" 替换为 "temporal_dense"
        name = name.replace("temporal_fc", "temporal_dense")

    # 检查字符串 "norm1" 是否在变量 name 中，并且字符串中不包含 "temporal"
    if "norm1" in name and "temporal" not in name:
        # 如果是，则将字符串 "norm1" 替换为 "layernorm_before"
        name = name.replace("norm1", "layernorm_before")

    # 检查字符串 "norm2" 是否在变量 name 中
    if "norm2" in name:
        # 如果是，则将字符串 "norm2" 替换为 "layernorm_after"
        name = name.replace("norm2", "layernorm_after")

    # 检查字符串 "mlp.fc1" 是否在变量 name 中
    if "mlp.fc1" in name:
        # 如果是，则将字符串 "mlp.fc1" 替换为 "intermediate.dense"
        name = name.replace("mlp.fc1", "intermediate.dense")

    # 检查字符串 "mlp.fc2" 是否在变量 name 中
    if "mlp.fc2" in name:
        # 如果是，则将字符串 "mlp.fc2" 替换为 "output.dense"
        name = name.replace("mlp.fc2", "output.dense")

    # 检查字符串 "norm.weight" 是否在变量 name 中，并且字符串中不包含 "fc" 和 "temporal"
    if "norm.weight" in name and "fc" not in name and "temporal" not in name:
        # 如果是，则将字符串 "norm.weight" 替换为 "timesformer.layernorm.weight"
        name = name.replace("norm.weight", "timesformer.layernorm.weight")

    # 检查字符串 "norm.bias" 是否在变量 name 中，并且字符串中不包含 "fc" 和 "temporal"
    if "norm.bias" in name and "fc" not in name and "temporal" not in name:
        # 如果是，则将字符串 "norm.bias" 替换为 "timesformer.layernorm.bias"
        name = name.replace("norm.bias", "timesformer.layernorm.bias")

    # 检查字符串 "head" 是否在变量 name 中
    if "head" in name:
        # 如果是，则将字符串 "head" 替换为 "classifier"
        name = name.replace("head", "classifier")

    # 返回替换后的变量 name
    return name
# 根据给定的原始状态字典和配置，转换模型的状态字典
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的键（需要复制，因为后续会修改原始字典）
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值
        val = orig_state_dict.pop(key)

        # 如果键以"model."开头，则去除该前缀
        if key.startswith("model."):
            key = key.replace("model.", "")

        # 如果键包含"qkv"，则根据不同情况重新命名键
        if "qkv" in key:
            key_split = key.split(".")
            layer_num = int(key_split[1])
            prefix = "timesformer.encoder.layer."
            # 根据键名中是否包含"temporal"决定后缀
            if "temporal" in key:
                postfix = ".temporal_attention.attention.qkv."
            else:
                postfix = ".attention.attention.qkv."
            # 根据键名中是否包含"weight"决定修改状态字典中的键和对应的值
            if "weight" in key:
                orig_state_dict[f"{prefix}{layer_num}{postfix}weight"] = val
            else:
                orig_state_dict[f"{prefix}{layer_num}{postfix}bias"] = val
        else:
            # 否则，对键进行重命名
            orig_state_dict[rename_key(key)] = val

    # 返回转换后的原始状态字典
    return orig_state_dict


# 我们将在一个吃意大利面条的视频上验证我们的结果
# 使用的帧索引: [164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    # 从指定的数据集仓库下载名为"eating_spaghetti.npy"的文件
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    # 加载视频数据并转换为列表返回
    video = np.load(file)
    return list(video)


def convert_timesformer_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    # 获取特定模型名称的配置信息
    config = get_timesformer_config(model_name)

    # 使用配置创建一个 TimesformerForVideoClassification 模型
    model = TimesformerForVideoClassification(config)

    # 下载托管在 Google Drive 上的原始检查点文件
    output = "pytorch_model.bin"
    gdown.cached_download(checkpoint_url, output, quiet=False)
    # 加载检查点文件，根据文件中的键名不同进行适配
    files = torch.load(output, map_location="cpu")
    if "model" in files:
        state_dict = files["model"]
    elif "module" in files:
        state_dict = files["module"]
    else:
        state_dict = files["model_state"]
    # 转换加载的状态字典到新的状态字典格式
    new_state_dict = convert_state_dict(state_dict, config)

    # 加载模型的新状态字典
    model.load_state_dict(new_state_dict)
    # 设置模型为评估模式
    model.eval()

    # 在基本输入上验证模型
    # 创建一个图像处理器对象，用于视频处理
    image_processor = VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    # 准备视频数据
    video = prepare_video()
    # 使用图像处理器处理前8帧视频，并返回PyTorch张量格式的输入
    inputs = image_processor(video[:8], return_tensors="pt")

    # 使用模型进行推理，获取输出结果
    outputs = model(**inputs)
    logits = outputs.logits

    # 定义一组模型名称列表，包含不同版本和分辨率的预训练检查点
    model_names = [
        # Kinetics-400 数据集检查点（hr = 使用448px高分辨率输入而非224px）
        "timesformer-base-finetuned-k400",
        "timesformer-large-finetuned-k400",
        "timesformer-hr-finetuned-k400",
        # Kinetics-600 数据集检查点（hr = 使用448px高分辨率输入而非224px）
        "timesformer-base-finetuned-k600",
        "timesformer-large-finetuned-k600",
        "timesformer-hr-finetuned-k600",
        # Something-Something-v2 数据集检查点（hr = 使用448px高分辨率输入而非224px）
        "timesformer-base-finetuned-ssv2",
        "timesformer-large-finetuned-ssv2",
        "timesformer-hr-finetuned-ssv2",
    ]

    # 注意：logits使用了图像均值和标准差 [0.5, 0.5, 0.5] 和 [0.5, 0.5, 0.5] 进行了测试
    # 根据模型名称设置预期的输出形状和预期的输出值
    if model_name == "timesformer-base-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.3016, -0.7713, -0.4205])
    elif model_name == "timesformer-base-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([-0.7267, -0.7466, 3.2404])
    elif model_name == "timesformer-base-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-0.9059, 0.6433, -3.1457])
    elif model_name == "timesformer-large-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-large-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-large-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0, 0, 0])
    elif model_name == "timesformer-hr-finetuned-k400":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.9617, -3.7311, -3.7708])
    elif model_name == "timesformer-hr-finetuned-k600":
        expected_shape = torch.Size([1, 600])
        expected_slice = torch.tensor([2.5273, 0.7127, 1.8848])
    elif model_name == "timesformer-hr-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([-3.6756, -0.7513, 0.7180])
    else:
        raise ValueError(f"Model name not supported. Should be one of {model_names}")

    # 验证模型输出的形状是否与预期一致
    assert logits.shape == expected_shape
    # 验证模型输出的前三个元素是否与预期的数值接近
    assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    # 打印确认信息
    print("Logits ok!")

    # 如果指定了 PyTorch 模型保存路径，则保存模型和图像处理器
    if pytorch_dump_folder_path is not None:
        print(f"Saving model and image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)

    # 如果需要推送到 hub
    if push_to_hub:
        # 打印推送到 hub 的消息
        print("Pushing to the hub...")
        # 将模型推送到指定路径下的 hub
        model.push_to_hub(f"fcakyon/{model_name}")
if __name__ == "__main__":
    # 如果作为主程序执行，则开始解析命令行参数
    parser = argparse.ArgumentParser()
    
    # 添加必需的参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=17yvuYp9L4mn-HpIcK5Zo6K3UoOy1kA5l&export=download",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    
    parser.add_argument(
        "--model_name", 
        default="timesformer-base-finetuned-k400", 
        type=str, 
        help="Name of the model."
    )
    
    parser.add_argument(
        "--push_to_hub", 
        action="store_true", 
        help="Whether or not to push the converted model to the 🤗 hub."
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用函数 convert_timesformer_checkpoint，传入解析得到的参数
    convert_timesformer_checkpoint(
        args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub
    )
```