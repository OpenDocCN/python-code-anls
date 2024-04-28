# `.\transformers\models\videomae\convert_videomae_to_pytorch.py`

```
# 设置脚本的字符编码为 utf-8
# 版权声明
# 2022 年版权所有 The HuggingFace Inc. 团队
#
# 根据 Apache 许可证 2.0 版（"许可证"）的规定
# 除非符合许可证的规定，否则您不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则该软件是基于"原样"分发的
# 没有任何形式的明示或暗示的担保或条件
# 有关特定的语言版本的权限和限制，请参见许可证

# 导入所需的库
import argparse  # 导入命令行参数解析模块 argparse
import json  # 导入 json 模块用于处理 JSON 格式的数据

import gdown  # 导入 gdown 模块进行 Google Drive 文件的下载
import numpy as np  # 导入 numpy 库
import torch  # 导入 PyTorch 库
from huggingface_hub import hf_hub_download  # 从 huggingface_hub 模块中导入 hf_hub_download 函数

# 从 transformers 库中导入以下模型相关组件
from transformers import (
    VideoMAEConfig,  # 导入 VideoMAEConfig 类
    VideoMAEForPreTraining,  # 导入 VideoMAEForPreTraining 类
    VideoMAEForVideoClassification,  # 导入 VideoMAEForVideoClassification 类
    VideoMAEImageProcessor,  # 导入 VideoMAEImageProcessor 类
)


# 根据模型名称获取 VideoMAEConfig 配置信息
def get_videomae_config(model_name):
    # 创建 VideoMAEConfig 对象
    config = VideoMAEConfig()

    # 根据模型名称设置架构配置
    set_architecture_configs(model_name, config)

    # 如果模型名称中不包含 'finetuned'，则将 use_mean_pooling 设为 False
    if "finetuned" not in model_name:
        config.use_mean_pooling = False

    # 如果模型名称中包含 'finetuned'，处理标签信息
    if "finetuned" in model_name:
        repo_id = "huggingface/label-files"
        # 根据模型名称设定标签数和文件名
        if "kinetics" in model_name:
            config.num_labels = 400
            filename = "kinetics400-id2label.json"
        elif "ssv2" in model_name:
            config.num_labels = 174
            filename = "something-something-v2-id2label.json"
        else:
            raise ValueError("Model name should either contain 'kinetics' or 'ssv2' in case it's fine-tuned.")
        # 加载标签数据
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    return config


# 根据模型名称设置架构配置信息
def set_architecture_configs(model_name, config):
    # 根据模型名称设置不同的架构参数
    if "small" in model_name:
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 16
        config.decoder_num_hidden_layers = 12
        config.decoder_num_attention_heads = 3
        config.decoder_hidden_size = 192
        config.decoder_intermediate_size = 768
    elif "large" in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        config.decoder_num_hidden_layers = 12
        config.decoder_num_attention_heads = 8
        config.decoder_hidden_size = 512
        config.decoder_intermediate_size = 2048
    # 如果模型名称包含"huge"关键词，则设置配置参数为大型模型的数值
    elif "huge" in model_name:
        # 设置隐藏层大小为1280
        config.hidden_size = 1280
        # 设置中间层大小为5120
        config.intermediate_size = 5120
        # 设置隐藏层数为32
        config.num_hidden_layers = 32
        # 设置注意力头数为16
        config.num_attention_heads = 16
        # 设置解码器隐藏层数为12
        config.decoder_num_hidden_layers = 12
        # 设置解码器注意力头数为8
        config.decoder_num_attention_heads = 8
        # 设置解码器隐藏层大小为640
        config.decoder_hidden_size = 640
        # 设置解码器中间层大小为2560
        config.decoder_intermediate_size = 2560
    # 如果模型名称不包含"base"关键词，则触发值错误异常，要求模型名称包含"small", "base", "large", 或 "huge"
    elif "base" not in model_name:
        raise ValueError('Model name should include either "small", "base", "large", or "huge"')
# 重命名给定的参数名，根据不同的规则进行替换
def rename_key(name):
    # 如果参数名中包含"encoder."
    if "encoder." in name:
        # 将"encoder."替换为空字符串
        name = name.replace("encoder.", "")
    # 如果参数名中包含"cls_token"
    if "cls_token" in name:
        # 将"cls_token"替换为"videomae.embeddings.cls_token"
        name = name.replace("cls_token", "videomae.embeddings.cls_token")
    # ...
    # 其他条件下的替换规则同上
    # ...
    # 最后将修改后的参数名返回
    return name


# 根据给定的原始状态和配置转换状态字典
def convert_state_dict(orig_state_dict, config):
    # 遍历原始状态字典的键集合的拷贝，以便在迭代时可以安全地修改原始字典
    for key in orig_state_dict.copy().keys():
        # 弹出当前键对应的值，并赋值给变量 val
        val = orig_state_dict.pop(key)
    
        # 检查当前键是否以 "encoder." 开头，如果是，则去除该前缀
        if key.startswith("encoder."):
            key = key.replace("encoder.", "")
    
        # 检查当前键是否包含 "qkv"，如果是，则进一步处理
        if "qkv" in key:
            # 使用 "." 分割键名，以便提取层号等信息
            key_split = key.split(".")
    
            # 检查键名是否以 "decoder.blocks" 开头，根据不同情况设置不同的维度和前缀
            if key.startswith("decoder.blocks"):
                dim = config.decoder_hidden_size
                layer_num = int(key_split[2])
                prefix = "decoder.decoder_layers."
    
                # 如果键名中包含 "weight"，则按照特定规则重命名，并设置对应的值
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
            else:
                dim = config.hidden_size
                layer_num = int(key_split[1])
                prefix = "videomae.encoder.layer."
    
                # 如果键名中包含 "weight"，则按照特定规则重命名，并设置对应的值
                if "weight" in key:
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.query.weight"] = val[:dim, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.key.weight"] = val[dim : dim * 2, :]
                    orig_state_dict[f"{prefix}{layer_num}.attention.attention.value.weight"] = val[-dim:, :]
        else:
            # 如果不符合上述条件，则调用函数 rename_key() 重命名键，并设置对应的值
            orig_state_dict[rename_key(key)] = val
    
    # 返回处理后的原始状态字典
    return orig_state_dict
# 下面的代码准备在吃意大利面视频上验证我们的结果
# 使用的帧索引：[164 168 172 176 181 185 189 193 198 202 206 210 215 219 223 227]
def prepare_video():
    # 从数据集中下载视频
    file = hf_hub_download(
        repo_id="hf-internal-testing/spaghetti-video", filename="eating_spaghetti.npy", repo_type="dataset"
    )
    # 加载视频数据
    video = np.load(file)
    return list(video)


def convert_videomae_checkpoint(checkpoint_url, pytorch_dump_folder_path, model_name, push_to_hub):
    # 根据模型名获取Videomae的配置
    config = get_videomae_config(model_name)

    if "finetuned" in model_name:
        # 如果模型名中包含finetuned，则使用VideoMAEForVideoClassification
        model = VideoMAEForVideoClassification(config)
    else:
        # 否则使用VideoMAEForPreTraining
        model = VideoMAEForPreTraining(config)

    # 下载托管在Google Drive上的原始检查点
    output = "pytorch_model.bin"
    gdown.cached_download(checkpoint_url, output, quiet=False)
    # 加载预训练模型权重
    files = torch.load(output, map_location="cpu")
    if "model" in files:
        state_dict = files["model"]
    else:
        state_dict = files["module"]
    # 转换模型权重成适配当前配置的格式
    new_state_dict = convert_state_dict(state_dict, config)

    model.load_state_dict(new_state_dict)
    model.eval()

    # 在基本输入上验证模型
    image_processor = VideoMAEImageProcessor(image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5])
    video = prepare_video()
    inputs = image_processor(video, return_tensors="pt")

    if "finetuned" not in model_name:
        # 如果模型名中不包含finetuned，下载bool-masked-pos文件
        local_path = hf_hub_download(repo_id="hf-internal-testing/bool-masked-pos", filename="bool_masked_pos.pt")
        inputs["bool_masked_pos"] = torch.load(local_path)

    outputs = model(**inputs)
    logits = outputs.logits

    model_names = [
        "videomae-small-finetuned-kinetics",
        "videomae-small-finetuned-ssv2",
        # Kinetics-400 checkpoints (short = pretrained only for 800 epochs instead of 1600)
        "videomae-base-short",
        "videomae-base-short-finetuned-kinetics",
        "videomae-base",
        "videomae-base-finetuned-kinetics",
        "videomae-large",
        "videomae-large-finetuned-kinetics",
        "videomae-huge-finetuned-kinetics",
        # Something-Something-v2 checkpoints (short = pretrained only for 800 epochs instead of 2400)
        "videomae-base-short-ssv2",
        "videomae-base-short-finetuned-ssv2",
        "videomae-base-ssv2",
        "videomae-base-finetuned-ssv2",
    ]

    # 注意：logits与image_mean和image_std都等于[0.5, 0.5, 0.5]和[0.5, 0.5, 0.5]时进行了测试
    if model_name == "videomae-small-finetuned-kinetics":
        expected_shape = torch.Size([1, 400])
        expected_slice = torch.tensor([-0.9291, -0.4061, -0.9307])
    elif model_name == "videomae-small-finetuned-ssv2":
        expected_shape = torch.Size([1, 174])
        expected_slice = torch.tensor([0.2671, -0.4689, -0.8235])
    elif model_name == "videomae-base":
        expected_shape = torch.Size([1, 1408, 1536])
        expected_slice = torch.tensor([[0.7739, 0.7968, 0.7089], [0.6701, 0.7487, 0.6209], [0.4287, 0.5158, 0.4773]])
    elif model_name == "videomae-base-short":
        # 设置预期输出形状为 [1, 1408, 1536]
        expected_shape = torch.Size([1, 1408, 1536])
        # 设置预期输出切片为指定的张量
        expected_slice = torch.tensor([[0.7994, 0.9612, 0.8508], [0.7401, 0.8958, 0.8302], [0.5862, 0.7468, 0.7325]])
        # 对于这个模型，我们验证了标准像素损失和非标准化目标的损失
        expected_loss = torch.tensor([0.5142]) if config.norm_pix_loss else torch.tensor([0.6469])
    elif model_name == "videomae-large":
        # 设置预期输出形状为 [1, 1408, 1536]
        expected_shape = torch.Size([1, 1408, 1536])
        # 设置预期输出切片为指定的张量
        expected_slice = torch.tensor([[0.7149, 0.7997, 0.6966], [0.6768, 0.7869, 0.6948], [0.5139, 0.6221, 0.5605]])
    # ... 其他模型的设置预期输出形状和切片
    else:
        # 如果模型名称不支持，则引发值错误
        raise ValueError(f"Model name not supported. Should be one of {model_names}")
    
    # 验证 logits
    assert logits.shape == expected_shape
    if "finetuned" in model_name:
        # 如果模型是微调的，则通过 allclose 函数验证指定切片是否接近预期值
        assert torch.allclose(logits[0, :3], expected_slice, atol=1e-4)
    else:
        # 如果模型不是微调的，打印 logits 的部分内容，并通过 allclose 函数验证其切片是否接近预期值
        print("Logits:", logits[0, :3, :3])
        assert torch.allclose(logits[0, :3, :3], expected_slice, atol=1e-4)
    print("Logits ok!")

    # 验证损失，如果适用
    if model_name == "videomae-base-short":
        loss = outputs.loss
        # 通过 allclose 函数验证损失是否接近预期值
        assert torch.allclose(loss, expected_loss, atol=1e-4)
        print("Loss ok!")

    if pytorch_dump_folder_path is not None:
        # 如果存在 pytorch_dump_folder_path，则保存模型和图像处理器
        print(f"Saving model and image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)
        model.save_pretrained(pytorch_dump_folder_path)
    # 如果需要推送到hub，则执行以下操作
    if push_to_hub:
        # 打印提示信息，表示正在推送到hub
        print("Pushing to the hub...")
        # 将模型推送到hub，指定模型名称和组织
        model.push_to_hub(model_name, organization="nielsr")
# 如果当前脚本被作为主程序运行
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    
    # 添加必选参数
    parser.add_argument(
        "--checkpoint_url",
        default="https://drive.google.com/u/1/uc?id=1tEhLyskjb755TJ65ptsrafUG2llSwQE1&amp;export=download&amp;confirm=t&amp;uuid=aa3276eb-fb7e-482a-adec-dc7171df14c4",
        type=str,
        help=(
            "URL of the original PyTorch checkpoint (on Google Drive) you'd like to convert. Should be a direct"
            " download link."
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="/Users/nielsrogge/Documents/VideoMAE/Test",
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--model_name", default="videomae-base", type=str, help="Name of the model.")
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    # 解析传入的参数
    args = parser.parse_args()
    # 调用函数来转换 VideoMAE 检查点
    convert_videomae_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.model_name, args.push_to_hub)
```