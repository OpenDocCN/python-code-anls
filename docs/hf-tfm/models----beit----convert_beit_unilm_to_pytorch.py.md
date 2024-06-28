# `.\models\beit\convert_beit_unilm_to_pytorch.py`

```py
# 设置编码格式为 UTF-8

# 版权声明和许可证信息
# 版权所有 2021 年的 HuggingFace Inc. 团队。
# 根据 Apache 许可证 2.0 版本进行许可；
# 除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则分发的软件
# 基于“原样”分发，不提供任何形式的保证或条件。
# 请查阅许可证了解具体的法律条文和限制。

"""从 unilm 代码库转换 BEiT 检查点。"""

import argparse  # 导入命令行参数解析模块
import json  # 导入 JSON 操作模块
from pathlib import Path  # 导入路径操作模块

import requests  # 导入 HTTP 请求模块
import torch  # 导入 PyTorch 深度学习框架
from datasets import load_dataset  # 导入数据集加载模块
from huggingface_hub import hf_hub_download  # 导入 HuggingFace Hub 模型下载工具
from PIL import Image  # 导入图像处理库 PIL

from transformers import (  # 导入 transformers 库中的多个类
    BeitConfig,  # BEiT 模型配置类
    BeitForImageClassification,  # 用于图像分类的 BEiT 模型类
    BeitForMaskedImageModeling,  # 用于图像修复的 BEiT 模型类
    BeitForSemanticSegmentation,  # 用于语义分割的 BEiT 模型类
    BeitImageProcessor,  # BEiT 模型的图像处理器类
)
from transformers.image_utils import PILImageResampling  # 导入图像重采样函数
from transformers.utils import logging  # 导入 transformers 的日志记录模块

logging.set_verbosity_info()  # 设置日志记录级别为信息
logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象


# 这里列出所有需要重命名的键（左边是原始名称，右边是我们的名称）
def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    prefix = "backbone." if is_semantic else ""  # 如果是语义模型，则前缀为 "backbone."
    
    rename_keys = []  # 创建一个空列表用于存储重命名键值对
    for i in range(config.num_hidden_layers):
        # 编码器层：输出投影、两个前馈神经网络和两个层归一化
        rename_keys.append((f"{prefix}blocks.{i}.norm1.weight", f"beit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm1.bias", f"beit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.weight", f"beit.encoder.layer.{i}.attention.output.dense.weight")
        )
        rename_keys.append(
            (f"{prefix}blocks.{i}.attn.proj.bias", f"beit.encoder.layer.{i}.attention.output.dense.bias")
        )
        rename_keys.append((f"{prefix}blocks.{i}.norm2.weight", f"beit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.norm2.bias", f"beit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.weight", f"beit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc1.bias", f"beit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.weight", f"beit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"{prefix}blocks.{i}.mlp.fc2.bias", f"beit.encoder.layer.{i}.output.dense.bias"))

    # 投影层 + 位置嵌入
    # 将以下键值对添加到 rename_keys 列表中，用于重命名模型中的特定参数路径
    rename_keys.extend(
        [
            # 将 "{prefix}cls_token" 改为 "beit.embeddings.cls_token"
            (f"{prefix}cls_token", "beit.embeddings.cls_token"),
            # 将 "{prefix}patch_embed.proj.weight" 改为 "beit.embeddings.patch_embeddings.projection.weight"
            (f"{prefix}patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),
            # 将 "{prefix}patch_embed.proj.bias" 改为 "beit.embeddings.patch_embeddings.projection.bias"
            (f"{prefix}patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),
        ]
    )
    
    if has_lm_head:
        # 如果模型包含语言模型头部，则添加以下键值对到 rename_keys 列表中
        rename_keys.extend(
            [
                # 将 "mask_token" 改为 "beit.embeddings.mask_token"
                ("mask_token", "beit.embeddings.mask_token"),
                # 将 "rel_pos_bias.relative_position_bias_table" 改为 "beit.encoder.relative_position_bias.relative_position_bias_table"
                ("rel_pos_bias.relative_position_bias_table", "beit.encoder.relative_position_bias.relative_position_bias_table"),
                # 将 "rel_pos_bias.relative_position_index" 改为 "beit.encoder.relative_position_bias.relative_position_index"
                ("rel_pos_bias.relative_position_index", "beit.encoder.relative_position_bias.relative_position_index"),
                # 将 "norm.weight" 改为 "layernorm.weight"
                ("norm.weight", "layernorm.weight"),
                # 将 "norm.bias" 改为 "layernorm.bias"
                ("norm.bias", "layernorm.bias"),
            ]
        )
    elif is_semantic:
        # 如果模型是语义分割模型，则添加以下键值对到 rename_keys 列表中
        rename_keys.extend(
            [
                # 将 "decode_head.conv_seg.weight" 改为 "decode_head.classifier.weight"
                ("decode_head.conv_seg.weight", "decode_head.classifier.weight"),
                # 将 "decode_head.conv_seg.bias" 改为 "decode_head.classifier.bias"
                ("decode_head.conv_seg.bias", "decode_head.classifier.bias"),
                # 将 "auxiliary_head.conv_seg.weight" 改为 "auxiliary_head.classifier.weight"
                ("auxiliary_head.conv_seg.weight", "auxiliary_head.classifier.weight"),
                # 将 "auxiliary_head.conv_seg.bias" 改为 "auxiliary_head.classifier.bias"
                ("auxiliary_head.conv_seg.bias", "auxiliary_head.classifier.bias"),
            ]
        )
    else:
        # 如果以上条件都不满足，则添加以下键值对到 rename_keys 列表中
        rename_keys.extend(
            [
                # 将 "fc_norm.weight" 改为 "beit.pooler.layernorm.weight"
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),
                # 将 "fc_norm.bias" 改为 "beit.pooler.layernorm.bias"
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),
                # 将 "head.weight" 改为 "classifier.weight"
                ("head.weight", "classifier.weight"),
                # 将 "head.bias" 改为 "classifier.bias"
                ("head.bias", "classifier.bias"),
            ]
        )
    
    return rename_keys
# 将每个编码器层的矩阵拆分为查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    # 遍历每个隐藏层
    for i in range(config.num_hidden_layers):
        # 如果是语义模型，则使用特定的前缀
        prefix = "backbone." if is_semantic else ""

        # 从状态字典中弹出查询、键、值的权重矩阵
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        # 将查询矩阵权重和偏置添加到 BEiT 模型的状态字典中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        # 将键矩阵权重添加到 BEiT 模型的状态字典中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        # 将值矩阵权重和偏置添加到 BEiT 模型的状态字典中
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # 弹出并重命名 gamma_1 和 gamma_2 为 lambda_1 和 lambda_2，以防止在 .from_pretrained 方法中被重命名
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")
        state_dict[f"beit.encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"beit.encoder.layer.{i}.lambda_2"] = gamma_2

        # 如果模型没有语言模型头部，则处理相对位置偏置表和索引
        if not has_lm_head:
            # 每个层级都有自己的相对位置偏置表和索引
            table = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_bias_table")
            index = state_dict.pop(f"{prefix}blocks.{i}.attn.relative_position_index")
            # 将相对位置偏置表和索引添加到 BEiT 模型的状态字典中
            state_dict[
                f"beit.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_bias_table"
            ] = table
            state_dict[
                f"beit.encoder.layer.{i}.attention.attention.relative_position_bias.relative_position_index"
            ] = index


# 重命名状态字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用 requests 获取图片流，并打开为 PIL 图像对象
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 使用无梯度计算上下文环境，将检查点转换为 BEiT 结构
@torch.no_grad()
def convert_beit_checkpoint(checkpoint_url, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重到我们的 BEiT 结构。
    """

    # 定义默认的 BEiT 配置
    config = BeitConfig()
    has_lm_head = False
    is_semantic = False
    repo_id = "huggingface/label-files"
    # 根据 URL 设置配置参数
    if checkpoint_url[-9:-4] == "pt22k":
        # 使用共享的相对位置偏置表和遮蔽标记
        config.use_shared_relative_position_bias = True
        config.use_mask_token = True
        has_lm_head = True
    elif checkpoint_url[-9:-4] == "ft22k":
        # 对ImageNet-22k进行中间微调
        config.use_relative_position_bias = True
        config.num_labels = 21841
        filename = "imagenet-22k-id2label.json"
        # 从指定的HF Hub下载数据集文件，加载ID到标签的映射关系
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        # 该数据集包含21843个标签，但模型只有21841个，因此删除不需要的类别
        del id2label[9205]
        del id2label[15027]
        config.id2label = id2label
        # 构建标签到ID的反向映射
        config.label2id = {v: k for k, v in id2label.items()}
    elif checkpoint_url[-8:-4] == "to1k":
        # 对ImageNet-1k进行微调
        config.use_relative_position_bias = True
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        # 从指定的HF Hub下载数据集文件，加载ID到标签的映射关系
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        # 构建标签到ID的反向映射
        config.label2id = {v: k for k, v in id2label.items()}
        # 根据URL中的尺寸信息设置图像大小
        if "384" in checkpoint_url:
            config.image_size = 384
        if "512" in checkpoint_url:
            config.image_size = 512
    elif "ade20k" in checkpoint_url:
        # 对ADE20K数据集进行微调
        config.use_relative_position_bias = True
        config.num_labels = 150
        filename = "ade20k-id2label.json"
        # 从指定的HF Hub下载数据集文件，加载ID到标签的映射关系
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        # 构建标签到ID的反向映射
        config.label2id = {v: k for k, v in id2label.items()}
        # 设置图像大小为640，并标记为语义分割任务
        config.image_size = 640
        is_semantic = True
    else:
        raise ValueError("Checkpoint not supported, URL should either end with 'pt22k', 'ft22k', 'to1k' or 'ade20k'")

    # 架构的尺寸设置
    if "base" in checkpoint_url:
        pass
    elif "large" in checkpoint_url:
        # 设置大型模型的隐藏层大小、中间层大小、隐藏层层数和注意力头数
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
        # 如果是ADE20K数据集，设置特定的图像大小和输出索引
        if "ade20k" in checkpoint_url:
            config.image_size = 640
            config.out_indices = [7, 11, 15, 23]
    else:
        raise ValueError("Should either find 'base' or 'large' in checkpoint URL")

    # 加载原始模型的state_dict，并移除/重命名部分键
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)
    # 如果不是ADE20K数据集，只加载"model"部分，否则加载"state_dict"部分
    state_dict = state_dict["model"] if "ade20k" not in checkpoint_url else state_dict["state_dict"]

    # 创建重命名键列表并应用到state_dict
    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取QKV（查询、键、值）的信息并应用到state_dict
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head, is_semantic=is_semantic)
    # 如果是语义分割模型
    if is_semantic:
        # 对状态字典中的键添加前缀
        for key, val in state_dict.copy().items():
            val = state_dict.pop(key)
            if key.startswith("backbone.fpn"):
                key = key.replace("backbone.fpn", "fpn")
            state_dict[key] = val

    # 加载 HuggingFace 模型
    if checkpoint_url[-9:-4] == "pt22k":
        # 根据 URL 后缀选择合适的模型：MaskedImageModeling
        model = BeitForMaskedImageModeling(config)
    elif "ade20k" in checkpoint_url:
        # 如果 URL 中包含 "ade20k"，选择语义分割模型
        model = BeitForSemanticSegmentation(config)
    else:
        # 默认选择图像分类模型
        model = BeitForImageClassification(config)
    model.eval()  # 将模型设置为评估模式
    model.load_state_dict(state_dict)  # 加载状态字典到模型中

    # 根据是否是语义分割选择图像处理器和图像
    if is_semantic:
        # 创建语义分割图像处理器
        image_processor = BeitImageProcessor(size=config.image_size, do_center_crop=False)
        # 加载测试数据集中的图像
        ds = load_dataset("hf-internal-testing/fixtures_ade20k", split="test")
        image = Image.open(ds[0]["file"])
    else:
        # 创建图像处理器，设置图像大小和重采样方式
        image_processor = BeitImageProcessor(
            size=config.image_size, resample=PILImageResampling.BILINEAR, do_center_crop=False
        )
        # 准备图像
        image = prepare_img()

    # 对图像进行编码，返回编码结果
    encoding = image_processor(images=image, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    # 使用模型进行推理，得到输出
    outputs = model(pixel_values)
    logits = outputs.logits

    # 验证输出 logits 的形状是否符合预期
    expected_shape = torch.Size([1, 1000])  # 默认情况下 logits 形状为 [1, 1000]
    if checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k"):
        expected_shape = torch.Size([1, 196, 8192])  # 特定模型的预期 logits 形状
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k"):
        expected_shape = torch.Size([1, 196, 8192])  # 特定模型的预期 logits 形状
    elif checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k_ft22k"):
        expected_shape = torch.Size([1, 21841])  # 特定模型的预期 logits 形状
        expected_logits = torch.tensor([2.2288, 2.4671, 0.7395])  # 预期的 logits 值
        expected_class_idx = 2397  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k_ft22k"):
        expected_shape = torch.Size([1, 21841])  # 特定模型的预期 logits 形状
        expected_logits = torch.tensor([1.6881, -0.2787, 0.5901])  # 预期的 logits 值
        expected_class_idx = 2396  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k_ft1k"):
        expected_logits = torch.tensor([0.1241, 0.0798, -0.6569])  # 预期的 logits 值
        expected_class_idx = 285  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_base_patch16_224_pt22k_ft22kto1k"):
        expected_logits = torch.tensor([-1.2385, -1.0987, -1.0108])  # 预期的 logits 值
        expected_class_idx = 281  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_base_patch16_384_pt22k_ft22kto1k"):
        expected_logits = torch.tensor([-1.5303, -0.9484, -0.3147])  # 预期的 logits 值
        expected_class_idx = 761  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k_ft1k"):
        expected_logits = torch.tensor([0.4610, -0.0928, 0.2086])  # 预期的 logits 值
        expected_class_idx = 761  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_large_patch16_224_pt22k_ft22kto1k"):
        expected_logits = torch.tensor([-0.4804, 0.6257, -0.1837])  # 预期的 logits 值
        expected_class_idx = 761  # 预期的类别索引
    elif checkpoint_url[:-4].endswith("beit_large_patch16_384_pt22k_ft22kto1k"):
        # 设置预期的模型输出日志和类别索引，用于后续验证
        expected_logits = torch.tensor([[-0.5122, 0.5117, -0.2113]])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith("beit_large_patch16_512_pt22k_ft22kto1k"):
        # 设置预期的模型输出日志和类别索引，用于后续验证
        expected_logits = torch.tensor([-0.3062, 0.7261, 0.4852])
        expected_class_idx = 761
    elif checkpoint_url[:-4].endswith("beit_base_patch16_640_pt22k_ft22ktoade20k"):
        # 设置预期的模型输出形状和日志，用于后续验证
        expected_shape = (1, 150, 160, 160)
        expected_logits = torch.tensor(
            [
                [[-4.9225, -2.3954, -3.0522], [-2.8822, -1.0046, -1.7561], [-2.9549, -1.3228, -2.1347]],
                [[-5.8168, -3.4129, -4.0778], [-3.8651, -2.2214, -3.0277], [-3.8356, -2.4643, -3.3535]],
                [[-0.0078, 3.9952, 4.0754], [2.9856, 4.6944, 5.0035], [3.2413, 4.7813, 4.9969]],
            ]
        )
    elif checkpoint_url[:-4].endswith("beit_large_patch16_640_pt22k_ft22ktoade20k"):
        # 设置预期的模型输出形状和日志，用于后续验证
        expected_shape = (1, 150, 160, 160)
        expected_logits = torch.tensor(
            [
                [[-4.3305, -2.3049, -3.0161], [-2.9591, -1.5305, -2.2251], [-3.4198, -1.8004, -2.9062]],
                [[-5.8922, -3.7435, -4.3978], [-4.2063, -2.7872, -3.4755], [-4.2791, -3.1874, -4.1681]],
                [[0.9895, 4.3467, 4.7663], [4.2476, 5.6830, 6.1518], [4.5550, 6.2495, 6.5154]],
            ]
        )
    else:
        # 如果不是支持的模型类型，则引发错误
        raise ValueError("Can't verify logits as model is not supported")

    if logits.shape != expected_shape:
        # 检查模型输出的形状是否符合预期
        raise ValueError(f"Shape of logits not as expected. {logits.shape=}, {expected_shape=}")
    if not has_lm_head:
        if is_semantic:
            # 如果是语义任务，检查模型输出的前几个元素是否与预期的日志值接近
            if not torch.allclose(logits[0, :3, :3, :3], expected_logits, atol=1e-3):
                raise ValueError("First elements of logits not as expected")
        else:
            # 如果不是语义任务，打印预测的类别索引并检查模型输出的前几个元素是否与预期的日志值接近
            print("Predicted class idx:", logits.argmax(-1).item())

            if not torch.allclose(logits[0, :3], expected_logits, atol=1e-3):
                raise ValueError("First elements of logits not as expected")
            if logits.argmax(-1).item() != expected_class_idx:
                raise ValueError("Predicted class index not as expected")

    # 创建保存模型的文件夹（如果不存在）
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
if __name__ == "__main__":
    # 如果当前脚本被直接执行，则执行以下代码块

    parser = argparse.ArgumentParser()
    # 创建一个参数解析器对象

    parser.add_argument(
        "--checkpoint_url",
        default="https://conversationhub.blob.core.windows.net/beit-share-public/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    # 添加名为--checkpoint_url的命令行参数，设置默认值和类型，并提供帮助信息

    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 添加名为--pytorch_dump_folder_path的命令行参数，设置默认值和类型，并提供帮助信息

    args = parser.parse_args()
    # 解析命令行参数，并将其存储在args对象中

    convert_beit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path)
    # 调用convert_beit_checkpoint函数，传递命令行参数中的checkpoint_url和pytorch_dump_folder_path作为参数
```