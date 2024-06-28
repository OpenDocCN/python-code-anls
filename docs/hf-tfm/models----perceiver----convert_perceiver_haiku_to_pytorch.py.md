# `.\models\perceiver\convert_perceiver_haiku_to_pytorch.py`

```
# 设置编码格式为 UTF-8
# 版权声明，声明代码版权及使用许可
# 根据 Apache 许可证版本 2.0 使用本文件，详见指定链接
# 除非适用法律要求或书面同意，本软件是基于"原样"提供的，无任何明示或暗示的保证或条件
# 请参阅许可证，了解详细的法律条款
"""将 Haiku 实现的 Perceiver 检查点转换为 PyTorch 模型。"""


import argparse  # 导入用于解析命令行参数的模块
import json  # 导入处理 JSON 数据的模块
import pickle  # 导入序列化和反序列化 Python 对象的模块
from pathlib import Path  # 导入处理路径的模块

import haiku as hk  # 导入 Haiku 深度学习库
import numpy as np  # 导入处理数组和矩阵的数学库
import requests  # 导入处理 HTTP 请求的库
import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 导入从 Hugging Face Hub 下载模型的函数
from PIL import Image  # 导入处理图像的 Python 库

from transformers import (  # 导入 Transformers 库中的多个模型和工具类
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
from transformers.utils import logging  # 导入 Transformers 中的日志模块


logging.set_verbosity_info()  # 设置日志记录详细程度为信息级别
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


def prepare_img():
    # 我们将使用一张狗的图像来验证我们的结果
    url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
    im = Image.open(requests.get(url, stream=True).raw)  # 从 URL 加载图像并打开
    return im


def rename_keys(state_dict, architecture):
@torch.no_grad()  # 使用装饰器声明不需要梯度的上下文管理器
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, architecture="MLM"):
    """
    将模型的权重复制/粘贴/调整为我们的 Perceiver 结构。
    """

    # 将参数作为 FlatMapping 数据结构加载
    with open(pickle_file, "rb") as f:
        checkpoint = pickle.loads(f.read())

    state = None
    if isinstance(checkpoint, dict) and architecture in [
        "image_classification",
        "image_classification_fourier",
        "image_classification_conv",
    ]:
        # 图像分类 Conv 检查点还包含批归一化状态 (running_mean 和 running_var)
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

    # 重命名键名
    rename_keys(state_dict, architecture=architecture)

    # 加载 HuggingFace 模型
    config = PerceiverConfig()
    # 初始化 subsampling 变量为 None
    subsampling = None
    # 设置 repo_id 变量为 "huggingface/label-files"
    repo_id = "huggingface/label-files"
    # 根据不同的架构设置模型配置和实例化不同的 Perceiver 模型
    if architecture == "MLM":
        # 针对 MLM 架构设置特定的配置参数
        config.qk_channels = 8 * 32
        config.v_channels = 1280
        # 实例化一个 PerceiverForMaskedLM 模型
        model = PerceiverForMaskedLM(config)
    elif "image_classification" in architecture:
        # 针对图像分类相关架构设置特定的配置参数
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        # 重置 config 中的 qk_channels 和 v_channels 为 None
        config.qk_channels = None
        config.v_channels = None
        # 设置 num_labels 为 1000，并加载对应的类别标签映射文件
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        # 从指定的 repo_id 中下载并读取 id2label 映射
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        # 将 id2label 字典的键值转换为整数类型
        id2label = {int(k): v for k, v in id2label.items()}
        # 设置模型配置的 id2label 和 label2id 属性
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        if architecture == "image_classification":
            # 针对 image_classification 架构设置图像尺寸为 224，并实例化 PerceiverForImageClassificationLearned 模型
            config.image_size = 224
            model = PerceiverForImageClassificationLearned(config)
        elif architecture == "image_classification_fourier":
            # 针对 image_classification_fourier 架构设置特定的 d_model，并实例化 PerceiverForImageClassificationFourier 模型
            config.d_model = 261
            model = PerceiverForImageClassificationFourier(config)
        elif architecture == "image_classification_conv":
            # 针对 image_classification_conv 架构设置特定的 d_model，并实例化 PerceiverForImageClassificationConvProcessing 模型
            config.d_model = 322
            model = PerceiverForImageClassificationConvProcessing(config)
        else:
            # 如果架构不在预期的架构列表中，抛出异常
            raise ValueError(f"Architecture {architecture} not supported")
    elif architecture == "optical_flow":
        # 针对 optical_flow 架构设置特定的配置参数，并实例化 PerceiverForOpticalFlow 模型
        config.num_latents = 2048
        config.d_latents = 512
        config.d_model = 322
        config.num_blocks = 1
        config.num_self_attends_per_block = 24
        config.num_self_attention_heads = 16
        config.num_cross_attention_heads = 1
        model = PerceiverForOpticalFlow(config)
    # 如果架构是多模态自编码
    elif architecture == "multimodal_autoencoding":
        # 设置编码器的输入大小为图像的像素数
        config.num_latents = 28 * 28 * 1
        # 设置潜在空间向量的维度
        config.d_latents = 512
        # 设置模型的维度
        config.d_model = 704
        # 设置模型的块数
        config.num_blocks = 1
        # 每个块的自注意力层数
        config.num_self_attends_per_block = 8
        # 自注意力头数
        config.num_self_attention_heads = 8
        # 交叉注意力头数
        config.num_cross_attention_heads = 1
        # 标签数
        config.num_labels = 700
        
        # 定义虚拟输入和子采样（因为每次前向传播只处理图像和音频数据的一部分）
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        nchunks = 128
        # 图像块大小
        image_chunk_size = np.prod((16, 224, 224)) // nchunks
        # 音频块大小
        audio_chunk_size = audio.shape[1] // config.samples_per_patch // nchunks
        
        # 处理第一个块
        chunk_idx = 0
        # 设置子采样字典，包含图像和音频的索引
        subsampling = {
            "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            "label": None,
        }
        
        # 创建多模态自编码器模型
        model = PerceiverForMultimodalAutoencoding(config)
        
        # 设置标签
        filename = "kinetics700-id2label.json"
        # 从数据集库中下载标签文件
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        # 将标签字典中的键转换为整数类型
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        # 设置标签到ID的映射
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        # 抛出异常，指出不支持的架构类型
        raise ValueError(f"Architecture {architecture} not supported")
    
    # 将模型设置为评估模式
    model.eval()
    
    # 加载模型权重
    model.load_state_dict(state_dict)
    
    # 准备虚拟输入
    input_mask = None
    if architecture == "MLM":
        # 从预训练的分词器创建分词器对象
        tokenizer = PerceiverTokenizer.from_pretrained("/Users/NielsRogge/Documents/Perceiver/Tokenizer files")
        # 文本输入，包含一部分单词缺失的不完整句子
        text = "This is an incomplete sentence where some words are missing."
        # 对文本进行编码，填充到最大长度，并返回PyTorch张量
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")
        # 掩码掉 " missing." 部分的词。模型更好地表现需要掩码的部分以空格开头。
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs = encoding.input_ids
        input_mask = encoding.attention_mask
    elif architecture in ["image_classification", "image_classification_fourier", "image_classification_conv"]:
        # 创建图像处理器对象
        image_processor = PerceiverImageProcessor()
        # 准备图像数据
        image = prepare_img()
        # 对图像进行编码，返回PyTorch张量
        encoding = image_processor(image, return_tensors="pt")
        inputs = encoding.pixel_values
    elif architecture == "optical_flow":
        # 生成随机张量作为输入
        inputs = torch.randn(1, 2, 27, 368, 496)
    elif architecture == "multimodal_autoencoding":
        # 使用虚拟数据设置输入为图像、音频和标签
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        inputs = {"image": images, "audio": audio, "label": torch.zeros((images.shape[0], 700))}
    
    # 执行前向传播
    if architecture == "multimodal_autoencoding":
        # 使用模型进行前向传播，传入输入数据、注意力掩码和子采样输出点
        outputs = model(inputs=inputs, attention_mask=input_mask, subsampled_output_points=subsampling)
    else:
        # 使用模型进行推理，获取模型输出
        outputs = model(inputs=inputs, attention_mask=input_mask)
    # 获取模型输出中的 logits
    logits = outputs.logits

    # 验证 logits
    if not isinstance(logits, dict):
        # 如果 logits 不是字典，打印其形状
        print("Shape of logits:", logits.shape)
    else:
        # 如果 logits 是字典，逐个打印每个模态的 logits 形状
        for k, v in logits.items():
            print(f"Shape of logits of modality {k}", v.shape)

    if architecture == "MLM":
        # 对于 Masked Language Model (MLM) 架构
        expected_slice = torch.tensor(
            [[-11.8336, -11.6850, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.8440, -12.6410, -12.8646]]
        )
        # 断言切片部分的 logits 与预期的张量接近
        assert torch.allclose(logits[0, :3, :3], expected_slice)
        # 获取被掩码的标记的预测值，并转换为列表
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        # 预期的列表
        expected_list = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        # 断言掩码标记的预测值与预期列表相等
        assert masked_tokens_predictions == expected_list
        # 打印贪婪预测结果
        print("Greedy predictions:")
        print(masked_tokens_predictions)
        print()
        # 打印预测的字符串
        print("Predicted string:")
        print(tokenizer.decode(masked_tokens_predictions))

    elif architecture in ["image_classification", "image_classification_fourier", "image_classification_conv"]:
        # 对于图像分类等架构，打印预测的类别
        print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])

    # 最后，保存文件
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 将模型保存到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本被直接执行（而不是被导入到其他模块中），则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    # Required parameters
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.",
    )
    # 添加一个必需的参数：指向本地 Perceiver 检查点 pickle 文件的路径

    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )
    # 添加一个必需的参数：指向输出 PyTorch 模型目录的路径，作为一个字符串提供

    parser.add_argument(
        "--architecture",
        default="MLM",
        type=str,
        help="""
        Architecture, provided as a string. One of 'MLM', 'image_classification', image_classification_fourier',
        image_classification_fourier', 'optical_flow' or 'multimodal_autoencoding'.
        """,
    )
    # 添加一个可选参数：模型的架构类型，作为字符串提供。可选项包括 'MLM', 'image_classification',
    # 'image_classification_fourier', 'optical_flow' 或 'multimodal_autoencoding'

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在 args 变量中

    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path, args.architecture)
    # 调用函数 convert_perceiver_checkpoint，传递命令行参数中的 pickle_file、pytorch_dump_folder_path 和 architecture
```