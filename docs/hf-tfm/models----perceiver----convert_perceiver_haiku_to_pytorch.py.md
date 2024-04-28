# `.\transformers\models\perceiver\convert_perceiver_haiku_to_pytorch.py`

```py
# 设置文件编码为utf-8
# 版权声明，告知文件使用者此文件的版权信息
# 授权协议的说明，告知文件使用者可以根据Apache许可证第2.0版的规定使用本文件
# 可以从指定链接获取许可证的副本
# 根据适用法律或书面同意，本程序仅按“原样”分发
# 没有任何明示或暗示的保证或条件，包括但不限于特定目的的保证或条件
# 请查看许可证以了解限制及特定语言规定
# 本程序将Haiku实现的Perceiver检查点转换为相应的数据结构
import argparse
import json
import pickle
from pathlib import Path

# 导入haiku模块作为hk
import haiku as hk
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

# 从transformers模块中导入所需的类和函数
from transformers import (
    PerceiverConfig,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
    PerceiverForMaskedLM,
    PerceiverForMultimodalAutoencoding,
    PerceiverForOpticalFlow,
    PerceiverImageProcessor,
    PerceiverTokenizer,
)
from transformers.utils import logging

# 设置日志verbosity级别为info
logging.set_verbosity_info()
# 获取或创建指定名称的Logger
logger = logging.get_logger(__name__)

# 准备图像数据来验证结果
# 从指定URL获取图片数据
# 使用PIL库打开图片数据并返回
def prepare_img():
    url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im

# 重命名模型检查点的参数键值对
@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, architecture="MLM"):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """

    # 以FlatMapping数据结构加载参数
    with open(pickle_file, "rb") as f:
        checkpoint = pickle.loads(f.read())

    state = None
    if isinstance(checkpoint, dict) and architecture in [
        "image_classification",
        "image_classification_fourier",
        "image_classification_conv",
    ]:
        # 图像分类检查点也包含批量归一化状态（running_mean和running_var）
        params = checkpoint["params"]
        state = checkpoint["state"]
    else:
        params = checkpoint

    # 转换为初始状态字典
    state_dict = {}
    for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
        for param_name, param in parameters.items():
            state_dict[scope_name + "/" + param_name] = param

    if state is not None:
        # 添加状态变量
        for scope_name, parameters in hk.data_structures.to_mutable_dict(state).items():
            for param_name, param in parameters.items():
                state_dict[scope_name + "/" + param_name] = param

    # 重命名键值对
    rename_keys(state_dict, architecture=architecture)

    # 加载HuggingFace模型
    config = PerceiverConfig()
``` 
    # 初始化变量subsampling为None
    subsampling = None
    # 设定repo_id为"huggingface/label-files"
    repo_id = "huggingface/label-files"
    # 根据架构设置模型配置和模型
    if architecture == "MLM":
        # 设定config.qk_channels为8 * 32
        config.qk_channels = 8 * 32
        # 设定config.v_channels为1280
        config.v_channels = 1280
        # 创建PerceiverForMaskedLM模型
        model = PerceiverForMaskedLM(config)
    # 如果架构中包含"image_classification"
    elif "image_classification" in architecture:
        # 设置模型配置中的参数
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        config.qk_channels = None
        config.v_channels = None
        # 设置标签
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        # 从数据集下载标签数据并加载id2label字典
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        # 根据具体的图片分类架构选择不同的模型
        if architecture == "image_classification":
            config.image_size = 224
            model = PerceiverForImageClassificationLearned(config)
        elif architecture == "image_classification_fourier":
            config.d_model = 261
            model = PerceiverForImageClassificationFourier(config)
        elif architecture == "image_classification_conv":
            config.d_model = 322
            model = PerceiverForImageClassificationConvProcessing(config)
        else:
            # 如果架构不支持，则抛出异常
            raise ValueError(f"Architecture {architecture} not supported")
    # 如果架构为"optical_flow"
    elif architecture == "optical_flow":
        # 设置模型配置中的参数
        config.num_latents = 2048
        config.d_latents = 512
        config.d_model = 322
        config.num_blocks = 1
        config.num_self_attends_per_block = 24
        config.num_self_attention_heads = 16
        config.num_cross_attention_heads = 1
        # 创建PerceiverForOpticalFlow模型
        model = PerceiverForOpticalFlow(config)
    # 如果选择的架构是"multimodal_autoencoding"
    elif architecture == "multimodal_autoencoding":
        # 配置模型的潜变量数量为输入图片的像素数
        config.num_latents = 28 * 28 * 1
        # 配置模型的潜变量维度
        config.d_latents = 512
        # 配置模型的维度
        config.d_model = 704
        # 配置模型的块数量
        config.num_blocks = 1
        # 配置每个块中的自注意力模块数
        config.num_self_attends_per_block = 8
        # 配置自注意力模块中的头数
        config.num_self_attention_heads = 8
        # 配置跨注意力模块中的头数
        config.num_cross_attention_heads = 1
        # 配置标签数量
        config.num_labels = 700
        # 定义虚拟输入+子采样（因为每次前向传递仅对图像数据的一部分和音频数据进行操作）
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        nchunks = 128
        image_chunk_size = np.prod((16, 224, 224)) // nchunks
        audio_chunk_size = audio.shape[1] // config.samples_per_patch // nchunks
        # 处理第一个块
        chunk_idx = 0
        subsampling = {
            "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            "label": None,
        }
        # 创建PerceiverForMultimodalAutoencoding模型
        model = PerceiverForMultimodalAutoencoding(config)
        # 设置标签
        filename = "kinetics700-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        # 抛出错误，指定的架构不受支持
        raise ValueError(f"Architecture {architecture} not supported")
    # 模型设置为评估模式
    model.eval()

    # 加载权重
    model.load_state_dict(state_dict)

    # 准备虚拟输入
    input_mask = None
    if architecture == "MLM":
        tokenizer = PerceiverTokenizer.from_pretrained("/Users/NielsRogge/Documents/Perceiver/Tokenizer files")
        text = "This is an incomplete sentence where some words are missing."
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")
        # 掩盖" missing.". 注意，如果被遮蔽的块以空格开头，模型的性能会更好
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs = encoding.input_ids
        input_mask = encoding.attention_mask
    elif architecture in ["image_classification", "image_classification_fourier", "image_classification_conv"]:
        # 创建PerceiverImageProcessor对象用于图像处理
        image_processor = PerceiverImageProcessor()
        image = prepare_img()
        encoding = image_processor(image, return_tensors="pt")
        inputs = encoding.pixel_values
    elif architecture == "optical_flow":
        # 初始化光流预测的虚拟输入
        inputs = torch.randn(1, 2, 27, 368, 496)
    elif architecture == "multimodal_autoencoding":
        # 初始化多模式自编码的虚拟输入
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        inputs = {"image": images, "audio": audio, "label": torch.zeros((images.shape[0], 700))}

    # 进行前向传播
    if architecture == "multimodal_autoencoding":
        # 使用模型进行前向传播，传入输入数据、注意力掩码和子采样输出点
        outputs = model(inputs=inputs, attention_mask=input_mask, subsampled_output_points=subsampling)
    else:
        # 使用模型对输入进行预测并生成输出
        outputs = model(inputs=inputs, attention_mask=input_mask)
    # 获取模型输出的logits
    logits = outputs.logits

    # 验证logits
    if not isinstance(logits, dict):
        # 如果logits不是字典，则打印logits的形状
        print("Shape of logits:", logits.shape)
    else:
        # 如果logits是字典，则遍历字典并打印每个modality的logits形状
        for k, v in logits.items():
            print(f"Shape of logits of modality {k}", v.shape)

    if architecture == "MLM":
        # 如果模型架构为MLM，则进行以下操作
        # 创建期望的tensor slice
        expected_slice = torch.tensor(
            [[-11.8336, -11.6850, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.8440, -12.6410, -12.8646]]
        )
        # 验证logits的特定部分是否接近期望的slice，并打印结果
        assert torch.allclose(logits[0, :3, :3], expected_slice)
        # 获取logits中特定位置的预测并打印
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        expected_list = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        # 验证模型生成的masked_tokens_predictions是否与期望的列表相等
        assert masked_tokens_predictions == expected_list
        print("Greedy predictions:")
        print(masked_tokens_predictions)
        print()
        print("Predicted string:")
        # 打印由tokenizer解码后的预测字符串
        print(tokenizer.decode(masked_tokens_predictions))

    elif architecture in ["image_classification", "image_classification_fourier", "image_classification_conv"]:
        # 如果模型架构为图像分类的其中一种，则打印预测的类别
        print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])

    # 最后，保存文件
    # 如果路径不存在，则创建路径
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印正在保存模型文件的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本是主程序入口点
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    
    # 定义必需的命令行参数
    # 参数名：--pickle_file
    # 类型：字符串
    # 默认值：None
    # 是否必填：是
    # 帮助信息：转换的 Perceiver 检查点的本地 pickle 文件路径
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.",
    )
    
    # 参数名：--pytorch_dump_folder_path
    # 类型：字符串 

    # 默认值：None
    # 是否必填：是
    # 帮助信息：输出 PyTorch 模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )
    
    # 参数名：--architecture
    # 类型：字符串
    # 默认值："MLM"
    # 是否必填：否
    # 帮助信息：模型架构，可以是"MLM"、"image_classification"、"image_classification_fourier"、"optical_flow"或"multimodal_autoencoding"之一
    parser.add_argument(
        "--architecture",
        default="MLM",
        type=str,
        help="""
        Architecture, provided as a string. One of 'MLM', 'image_classification', image_classification_fourier',
        image_classification_fourier', 'optical_flow' or 'multimodal_autoencoding'.
        """,
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 convert_perceiver_checkpoint 函数，传入解析后的参数
    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path, args.architecture)
```