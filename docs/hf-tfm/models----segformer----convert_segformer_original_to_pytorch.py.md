# `.\models\segformer\convert_segformer_original_to_pytorch.py`

```py
# 设置文件编码为UTF-8，确保可以处理各种字符集
# 版权声明，指明代码版权归HuggingFace Inc.团队所有
#
# 根据Apache许可证2.0版，除非符合许可证的规定，否则不得使用本文件
# 可在以下网址获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件按"现状"提供，没有任何形式的担保或条件
# 有关详细信息，请参阅许可证中的特定语言
"""将SegFormer模型检查点转换为不同任务模型的格式。"""


import argparse  # 导入解析命令行参数的模块
import json  # 导入处理JSON格式的模块
from collections import OrderedDict  # 导入有序字典，用于保持键的插入顺序
from pathlib import Path  # 导入处理文件路径的模块

import requests  # 导入处理HTTP请求的模块
import torch  # 导入PyTorch深度学习框架
from huggingface_hub import hf_hub_download  # 导入从Hugging Face Hub下载模型的函数
from PIL import Image  # 导入处理图像的PIL库

from transformers import (  # 导入Transformers库中的SegFormer相关组件
    SegformerConfig,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from transformers.utils import logging  # 导入Transformers库的日志模块

# 设置日志级别为信息级别
logging.set_verbosity_info()
# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


def rename_keys(state_dict, encoder_only=False):
    # 创建一个新的有序字典，用于存储重命名后的模型状态字典
    new_state_dict = OrderedDict()
    # 遍历输入的状态字典的每个键值对
    for key, value in state_dict.items():
        # 如果只需要编码器部分并且键名不是以 "head" 开头，则添加前缀 "segformer.encoder."
        if encoder_only and not key.startswith("head"):
            key = "segformer.encoder." + key
        # 如果键名以 "backbone" 开头，则替换为 "segformer.encoder"
        if key.startswith("backbone"):
            key = key.replace("backbone", "segformer.encoder")
        # 如果键名包含 "patch_embed"，例如替换 "patch_embed1" 为 "patch_embeddings.0"
        if "patch_embed" in key:
            idx = key[key.find("patch_embed") + len("patch_embed")]
            key = key.replace(f"patch_embed{idx}", f"patch_embeddings.{int(idx)-1}")
        # 如果键名包含 "norm"，替换为 "layer_norm"
        if "norm" in key:
            key = key.replace("norm", "layer_norm")
        # 如果键名包含 "segformer.encoder.layer_norm"，例如替换 "layer_norm1" 为 "layer_norm.0"
        if "segformer.encoder.layer_norm" in key:
            idx = key[key.find("segformer.encoder.layer_norm") + len("segformer.encoder.layer_norm")]
            key = key.replace(f"layer_norm{idx}", f"layer_norm.{int(idx)-1}")
        # 如果键名是 "layer_norm1"，替换为 "layer_norm_1"
        if "layer_norm1" in key:
            key = key.replace("layer_norm1", "layer_norm_1")
        # 如果键名是 "layer_norm2"，替换为 "layer_norm_2"
        if "layer_norm2" in key:
            key = key.replace("layer_norm2", "layer_norm_2")
        # 如果键名包含 "block"，例如替换 "block1" 为 "block.0"
        if "block" in key:
            idx = key[key.find("block") + len("block")]
            key = key.replace(f"block{idx}", f"block.{int(idx)-1}")
        # 如果键名包含 "attn.q"，替换为 "attention.self.query"
        if "attn.q" in key:
            key = key.replace("attn.q", "attention.self.query")
        # 如果键名包含 "attn.proj"，替换为 "attention.output.dense"
        if "attn.proj" in key:
            key = key.replace("attn.proj", "attention.output.dense")
        # 如果键名包含 "attn"，替换为 "attention.self"
        if "attn" in key:
            key = key.replace("attn", "attention.self")
        # 如果键名包含 "fc1"，替换为 "dense1"
        if "fc1" in key:
            key = key.replace("fc1", "dense1")
        # 如果键名包含 "fc2"，替换为 "dense2"
        if "fc2" in key:
            key = key.replace("fc2", "dense2")
        # 如果键名包含 "linear_pred"，替换为 "classifier"
        if "linear_pred" in key:
            key = key.replace("linear_pred", "classifier")
        # 如果键名包含 "linear_fuse"，替换为 "linear_fuse" 和 "batch_norm"
        if "linear_fuse" in key:
            key = key.replace("linear_fuse.conv", "linear_fuse")
            key = key.replace("linear_fuse.bn", "batch_norm")
        # 如果键名包含 "linear_c"，例如替换 "linear_c4" 为 "linear_c.3"
        if "linear_c" in key:
            idx = key[key.find("linear_c") + len("linear_c")]
            key = key.replace(f"linear_c{idx}", f"linear_c.{int(idx)-1}")
        # 如果键名以 "head" 开头，替换为 "classifier"
        if key.startswith("head"):
            key = key.replace("head", "classifier")
        # 将处理后的键值对加入新的状态字典
        new_state_dict[key] = value

    # 返回转换后的新状态字典
    return new_state_dict
# 从给定的状态字典中读取键和值的权重和偏置，对于每个编码器块：
def read_in_k_v(state_dict, config):
    # 遍历每个编码器块：
    for i in range(config.num_encoder_blocks):
        for j in range(config.depths[i]):
            # 从状态字典中弹出键和值的权重和偏置
            kv_weight = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.weight")
            kv_bias = state_dict.pop(f"segformer.encoder.block.{i}.{j}.attention.self.kv.bias")
            # 将键和值的权重添加回状态字典，根据隐藏大小进行切片
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.weight"] = kv_weight[
                : config.hidden_sizes[i], :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.key.bias"] = kv_bias[: config.hidden_sizes[i]]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.weight"] = kv_weight[
                config.hidden_sizes[i] :, :
            ]
            state_dict[f"segformer.encoder.block.{i}.{j}.attention.self.value.bias"] = kv_bias[
                config.hidden_sizes[i] :
            ]


# 我们将在一个COCO图像上验证我们的结果
def prepare_img():
    # 定义图像的URL并加载为Image对象
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    return image


@torch.no_grad()
def convert_segformer_checkpoint(model_name, checkpoint_path, pytorch_dump_folder_path):
    """
    将模型的权重复制/粘贴/调整为我们的SegFormer结构。
    """

    # 加载默认的SegFormer配置
    config = SegformerConfig()
    encoder_only = False

    # 根据model_name设置属性
    repo_id = "huggingface/label-files"
    if "segformer" in model_name:
        size = model_name[len("segformer.") : len("segformer.") + 2]
        if "ade" in model_name:
            config.num_labels = 150
            filename = "ade20k-id2label.json"
            expected_shape = (1, 150, 128, 128)
        elif "city" in model_name:
            config.num_labels = 19
            filename = "cityscapes-id2label.json"
            expected_shape = (1, 19, 128, 128)
        else:
            raise ValueError(f"Model {model_name} not supported")
    elif "mit" in model_name:
        encoder_only = True
        size = model_name[4:6]
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        expected_shape = (1, 1000)
    else:
        raise ValueError(f"Model {model_name} not supported")

    # 设置配置的属性
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    if size == "b0":
        pass
    elif size == "b1":
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 256
    elif size == "b2":
        # 设置模型配置：隐藏层尺寸、解码器隐藏层尺寸和深度
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 4, 6, 3]
    elif size == "b3":
        # 设置模型配置：隐藏层尺寸、解码器隐藏层尺寸和深度
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 4, 18, 3]
    elif size == "b4":
        # 设置模型配置：隐藏层尺寸、解码器隐藏层尺寸和深度
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 8, 27, 3]
    elif size == "b5":
        # 设置模型配置：隐藏层尺寸、解码器隐藏层尺寸和深度
        config.hidden_sizes = [64, 128, 320, 512]
        config.decoder_hidden_size = 768
        config.depths = [3, 6, 40, 3]
    else:
        # 如果尺寸不支持，则引发值错误
        raise ValueError(f"Size {size} not supported")

    # 加载图像处理器（仅进行大小调整和标准化）
    image_processor = SegformerImageProcessor(
        image_scale=(512, 512), keep_ratio=False, align=False, do_random_crop=False
    )

    # 准备图像
    image = prepare_img()
    # 使用图像处理器处理图像并返回张量
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    # 记录信息：转换模型名称
    logger.info(f"Converting model {model_name}...")

    # 加载原始状态字典
    if encoder_only:
        # 如果仅加载编码器，则只加载模型参数
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    else:
        # 否则加载完整的状态字典
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]

    # 重命名键名
    state_dict = rename_keys(state_dict, encoder_only=encoder_only)
    if not encoder_only:
        # 如果不是仅加载编码器，则删除解码头部分的权重和偏置
        del state_dict["decode_head.conv_seg.weight"]
        del state_dict["decode_head.conv_seg.bias"]

    # 键和值矩阵需要特殊处理
    read_in_k_v(state_dict, config)

    # 创建HuggingFace模型并加载状态字典
    if encoder_only:
        # 如果仅加载编码器，则设置最后阶段不重塑
        config.reshape_last_stage = False
        model = SegformerForImageClassification(config)
    else:
        # 否则创建用于语义分割的Segformer模型
        model = SegformerForSemanticSegmentation(config)
    model.load_state_dict(state_dict)
    model.eval()

    # 进行前向传播
    outputs = model(pixel_values)
    logits = outputs.logits

    # 根据模型名称设置预期切片
    # ADE20k检查点
    if model_name == "segformer.b0.512x512.ade.160k":
        # 设置预期切片张量
        expected_slice = torch.tensor(
            [
                [[-4.6310, -5.5232, -6.2356], [-5.1921, -6.1444, -6.5996], [-5.4424, -6.2790, -6.7574]],
                [[-12.1391, -13.3122, -13.9554], [-12.8732, -13.9352, -14.3563], [-12.9438, -13.8226, -14.2513]],
                [[-12.5134, -13.4686, -14.4915], [-12.8669, -14.4343, -14.7758], [-13.2523, -14.5819, -15.0694]],
            ]
        )
    elif model_name == "segformer.b1.512x512.ade.160k":
        # 设置预期切片张量
        expected_slice = torch.tensor(
            [
                [[-7.5820, -8.7231, -8.3215], [-8.0600, -10.3529, -10.0304], [-7.5208, -9.4103, -9.6239]],
                [[-12.6918, -13.8994, -13.7137], [-13.3196, -15.7523, -15.4789], [-12.9343, -14.8757, -14.9689]],
                [[-11.1911, -11.9421, -11.3243], [-11.3342, -13.6839, -13.3581], [-10.3909, -12.1832, -12.4858]],
            ]
        )
    # 如果模型名称为 "segformer.b2.512x512.ade.160k"，设置预期的张量切片
    elif model_name == "segformer.b2.512x512.ade.160k":
        # 使用 torch.tensor 创建张量，包含三个子列表，每个子列表又包含三个子列表，每个子列表包含三个浮点数
        expected_slice = torch.tensor(
            [
                [[-11.8173, -14.3850, -16.3128], [-14.5648, -16.5804, -18.6568], [-14.7223, -15.7387, -18.4218]],
                [[-15.7290, -17.9171, -19.4423], [-18.3105, -19.9448, -21.4661], [-17.9296, -18.6497, -20.7910]],
                [[-15.0783, -17.0336, -18.2789], [-16.8771, -18.6870, -20.1612], [-16.2454, -17.1426, -19.5055]],
            ]
        )
    
    # 如果模型名称为 "segformer.b3.512x512.ade.160k"，设置预期的张量切片
    elif model_name == "segformer.b3.512x512.ade.160k":
        # 使用 torch.tensor 创建张量，包含三个子列表，每个子列表又包含三个子列表，每个子列表包含三个浮点数
        expected_slice = torch.tensor(
            [
                [[-9.0878, -10.2081, -10.1891], [-9.3144, -10.7941, -10.9843], [-9.2294, -10.3855, -10.5704]],
                [[-12.2316, -13.9068, -13.6102], [-12.9161, -14.3702, -14.3235], [-12.5233, -13.7174, -13.7932]],
                [[-14.6275, -15.2490, -14.9727], [-14.3400, -15.9687, -16.2827], [-14.1484, -15.4033, -15.8937]],
            ]
        )
    
    # 如果模型名称为 "segformer.b4.512x512.ade.160k"，设置预期的张量切片
    elif model_name == "segformer.b4.512x512.ade.160k":
        # 使用 torch.tensor 创建张量，包含三个子列表，每个子列表又包含三个子列表，每个子列表包含三个浮点数
        expected_slice = torch.tensor(
            [
                [[-12.3144, -13.2447, -14.0802], [-13.3614, -14.5816, -15.6117], [-13.3340, -14.4433, -16.2219]],
                [[-19.2781, -20.4128, -20.7506], [-20.6153, -21.6566, -22.0998], [-19.9800, -21.0430, -22.1494]],
                [[-18.8739, -19.7804, -21.1834], [-20.1233, -21.6765, -23.2944], [-20.0315, -21.2641, -23.6944]],
            ]
        )
    
    # 如果模型名称为 "segformer.b5.640x640.ade.160k"，设置预期的张量切片
    elif model_name == "segformer.b5.640x640.ade.160k":
        # 使用 torch.tensor 创建张量，包含三个子列表，每个子列表又包含三个子列表，每个子列表包含三个浮点数
        expected_slice = torch.tensor(
            [
                [[-9.5524, -12.0835, -11.7348], [-10.5229, -13.6446, -14.5662], [-9.5842, -12.8851, -13.9414]],
                [[-15.3432, -17.5323, -17.0818], [-16.3330, -18.9255, -19.2101], [-15.1340, -17.7848, -18.3971]],
                [[-12.6072, -14.9486, -14.6631], [-13.7629, -17.0907, -17.7745], [-12.7899, -16.1695, -17.1671]],
            ]
        )
    
    # 如果模型名称为 "segformer.b0.1024x1024.city.160k"，设置预期的张量切片
    elif model_name == "segformer.b0.1024x1024.city.160k":
        # 使用 torch.tensor 创建张量，包含三个子列表，每个子列表又包含三个子列表，每个子列表包含三个浮点数
        expected_slice = torch.tensor(
            [
                [[-11.9295, -13.4057, -14.8106], [-13.3431, -14.8179, -15.3781], [-14.2836, -15.5942, -16.1588]],
                [[-11.4906, -12.8067, -13.6564], [-13.1189, -14.0500, -14.1543], [-13.8748, -14.5136, -14.8789]],
                [[0.5374, 0.1067, -0.4742], [0.1141, -0.2255, -0.7099], [-0.3000, -0.5924, -1.3105]],
            ]
        )
    
    # 如果模型名称为 "segformer.b0.512x1024.city.160k"，设置预期的张量切片
    elif model_name == "segformer.b0.512x1024.city.160k":
        # 使用 torch.tensor 创建张量，包含三个子列表，每个子列表又包含三个子列表，每个子列表包含三个浮点数
        expected_slice = torch.tensor(
            [
                [[-7.8217, -9.8767, -10.1717], [-9.4438, -10.9058, -11.4047], [-9.7939, -12.3495, -12.1079]],
                [[-7.1514, -9.5336, -10.0860], [-9.7776, -11.6822, -11.8439], [-10.1411, -12.7655, -12.8972]],
                [[0.3021, 0.0805, -0.2310], [-0.0328, -0.1605, -0.2714], [-0.1408, -0.5477, -0.6976]],
            ]
        )
    elif model_name == "segformer.b0.640x1280.city.160k":
        expected_slice = torch.tensor(
            [
                # 三维张量，表示预期的切片数据，每个元素都是一个三元素列表
                [
                    [-1.1372e01, -1.2787e01, -1.3477e01],  # 第一个子列表
                    [-1.2536e01, -1.4194e01, -1.4409e01],  # 第二个子列表
                    [-1.3217e01, -1.4888e01, -1.5327e01],  # 第三个子列表
                ],
                [
                    [-1.4791e01, -1.7122e01, -1.8277e01],  # 第四个子列表
                    [-1.7163e01, -1.9192e01, -1.9533e01],  # 第五个子列表
                    [-1.7897e01, -1.9991e01, -2.0315e01],  # 第六个子列表
                ],
                [
                    [7.6723e-01, 4.1921e-01, -7.7878e-02],  # 第七个子列表
                    [4.7772e-01, 9.5557e-03, -2.8082e-01],  # 第八个子列表
                    [3.6032e-01, -2.4826e-01, -5.1168e-01],  # 第九个子列表
                ],
            ]
        )
    elif model_name == "segformer.b0.768x768.city.160k":
        expected_slice = torch.tensor(
            [
                # 三维张量，表示预期的切片数据，每个元素都是一个三元素列表
                [[-9.4959, -11.3087, -11.7479], [-11.0025, -12.6540, -12.3319], [-11.4064, -13.0487, -12.9905]],
                [[-9.8905, -11.3084, -12.0854], [-11.1726, -12.7698, -12.9583], [-11.5985, -13.3278, -14.1774]],
                [[0.2213, 0.0192, -0.2466], [-0.1731, -0.4213, -0.4874], [-0.3126, -0.6541, -1.1389]],
            ]
        )
    elif model_name == "segformer.b1.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                # 三维张量，表示预期的切片数据，每个元素都是一个三元素列表
                [[-13.5748, -13.9111, -12.6500], [-14.3500, -15.3683, -14.2328], [-14.7532, -16.0424, -15.6087]],
                [[-17.1651, -15.8725, -12.9653], [-17.2580, -17.3718, -14.8223], [-16.6058, -16.8783, -16.7452]],
                [[-3.6456, -3.0209, -1.4203], [-3.0797, -3.1959, -2.0000], [-1.8757, -1.9217, -1.6997]],
            ]
        )
    elif model_name == "segformer.b2.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                # 三维张量，表示预期的切片数据，每个元素都是一个三元素列表
                [[-16.0976, -16.4856, -17.3962], [-16.6234, -19.0342, -19.7685], [-16.0900, -18.0661, -19.1180]],
                [[-18.4750, -18.8488, -19.5074], [-19.4030, -22.1570, -22.5977], [-19.1191, -20.8486, -22.3783]],
                [[-4.5178, -5.5037, -6.5109], [-5.0884, -7.2174, -8.0334], [-4.4156, -5.8117, -7.2970]],
            ]
        )
    elif model_name == "segformer.b3.1024x1024.city.160k":
        expected_slice = torch.tensor(
            [
                # 三维张量，表示预期的切片数据，每个元素都是一个三元素列表
                [[-14.2081, -14.4732, -14.1977], [-14.5867, -16.4423, -16.6356], [-13.4441, -14.9685, -16.8696]],
                [[-14.4576, -14.7073, -15.0451], [-15.0816, -17.6237, -17.9873], [-14.4213, -16.0199, -18.5992]],
                [[-4.7349, -4.9588, -5.0966], [-4.3210, -6.9325, -7.2591], [-3.4312, -4.7484, -7.1917]],
            ]
        )
    # 如果模型名称为 "segformer.b4.1024x1024.city.160k"，设置预期的切片张量
    expected_slice = torch.tensor(
        [
            [[-11.7737, -11.9526, -11.3273], [-13.6692, -14.4574, -13.8878], [-13.8937, -14.6924, -15.9345]],
            [[-14.6706, -14.5330, -14.1306], [-16.1502, -16.8180, -16.4269], [-16.8338, -17.8939, -20.1746]],
            [[1.0491, 0.8289, 1.0310], [1.1044, 0.5219, 0.8055], [1.0899, 0.6926, 0.5590]],
        ]
    )
elif model_name == "segformer.b5.1024x1024.city.160k":
    # 如果模型名称为 "segformer.b5.1024x1024.city.160k"，设置另一个预期的切片张量
    expected_slice = torch.tensor(
        [
            [[-12.5641, -13.4777, -13.0684], [-13.9587, -15.8983, -16.6557], [-13.3109, -15.7350, -16.3141]],
            [[-14.7074, -15.4352, -14.5944], [-16.6353, -18.1663, -18.6120], [-15.1702, -18.0329, -18.1547]],
            [[-1.7990, -2.0951, -1.7784], [-2.6397, -3.8245, -3.9686], [-1.5264, -2.8126, -2.9316]],
        ]
    )
else:
    # 如果模型名称不匹配上述任何一个条件，则根据预测的类别索引获取预测的类别，并打印出来
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

# 验证 logits 的形状是否符合预期的形状
if not encoder_only:
    assert logits.shape == expected_shape
    # 检查 logits 的前一个切片是否与预期的切片接近，允许的绝对误差为 1e-2
    assert torch.allclose(logits[0, :3, :3, :3], expected_slice, atol=1e-2)

# 最后，保存模型和图像处理器到指定路径，并记录日志
logger.info(f"Saving PyTorch model and image processor to {pytorch_dump_folder_path}...")
Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
model.save_pretrained(pytorch_dump_folder_path)
image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果这个模块被直接执行（而不是被导入到其他模块中），则执行以下代码
    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    parser.add_argument(
        "--model_name",
        default="segformer.b0.512x512.ade.160k",
        type=str,
        help="Name of the model you'd like to convert.",
    )
    # 添加一个参数选项：模型的名称，默认为"segformer.b0.512x512.ade.160k"

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, help="Path to the original PyTorch checkpoint (.pth file)."
    )
    # 添加一个参数选项：原始 PyTorch checkpoint 文件的路径

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 添加一个参数选项：输出 PyTorch 模型的文件夹路径

    args = parser.parse_args()
    # 解析命令行参数并存储到 args 对象中

    convert_segformer_checkpoint(args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path)
    # 调用函数 convert_segformer_checkpoint，传入解析得到的参数作为函数的输入
```