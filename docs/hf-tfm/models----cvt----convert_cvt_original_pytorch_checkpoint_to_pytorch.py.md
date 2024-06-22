# `.\models\cvt\convert_cvt_original_pytorch_checkpoint_to_pytorch.py`

```py
# 设置代码文件的编码格式为 utf-8
# 版权声明，声明该代码文件遵循 Apache License Version 2.0
# 获得 Apache License 的详细内容可通过指定 URL 获取
# 在适用法律下或书面约定的情况下，分发的软件是基于"按原样"的基础分发的，没有任何担保或条件，无论是明示或暗示的
# 查看许可证中的内容，以了解对特定语言的权利和限制
"""转换自原存储库的 CvT 模型检查点。

URL: https://github.com/microsoft/CvT"""

# 导入必要的库
import argparse  # 用于解析命令行参数
import json  # 用于操作 JSON 数据
from collections import OrderedDict  # 从 collections 模块中导入 OrderedDict 数据结构

import torch  # 导入 PyTorch 深度学习库
from huggingface_hub import cached_download, hf_hub_url  # 从 huggingface_hub 库中导入 cached_download 和 hf_hub_url 函数

from transformers import AutoImageProcessor, CvtConfig, CvtForImageClassification  # 从 transformers 库导入 AutoImageProcessor、CvtConfig 和 CvtForImageClassification 类


def embeddings(idx):
    """
    该函数帮助重命名嵌入层权重。

    Args:
        idx: 原始模型中的阶段编号
    """
    # 创建一个空列表用于保存不同嵌入层权重的对应关系
    embed = []
    # 添加嵌入层权重的对应关系
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.weight",
            f"stage{idx}.patch_embed.proj.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.projection.bias",
            f"stage{idx}.patch_embed.proj.bias",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.weight",
            f"stage{idx}.patch_embed.norm.weight",
        )
    )
    embed.append(
        (
            f"cvt.encoder.stages.{idx}.embedding.convolution_embeddings.normalization.bias",
            f"stage{idx}.patch_embed.norm.bias",
        )
    )
    return embed


def attention(idx, cnt):
    """
    该函数帮助重命名注意力机制块层的权重。

    Args:
        idx: 原始模型中的阶段编号
        cnt: 每个阶段中块的计数
    """
    # 创建一个空列表用于保存不同注意力机制块层权重的对应关系
    attention_weights = []
    # 添加注意力机制块层权重的对应关系
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.conv.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.weight",
        )
    )
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.bias",
        )
    )
```  
    # 添加注意力权重信息，包括查询的正常化均值
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_mean",
        )
    )
    # 添加注意力权重信息，包括查询的正常化方差
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.running_var",
        )
    )
    # 添加注意力权重信息，包括查询的正常化追踪批次
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_query.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_q.bn.num_batches_tracked",
        )
    )
    # 添加注意力权重信息，包括键的正常化卷积层权重
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.conv.weight",
        )
    )
    # 添加注意力权重信息，包括键的正常化权重
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.weight",
        )
    )
    # 添加注意力权重信息，包括键的正常化偏置
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.bias",
        )
    )
    # 添加注意力权重信息，包括键的正常化均值
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_mean",
        )
    )
    # 添加注意力权重信息，包括键的正常化方差
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.running_var",
        )
    )
    # 添加注意力权重信息，包括键���正常化追踪批次
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_key.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_k.bn.num_batches_tracked",
        )
    )
    # 添加注意力权重信息，包括值的卷积层权重
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.convolution.weight",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.conv.weight",
        )
    )
    # 将注意力权重的参数对应的路径添加到注意力权重列表中
    attention_weights.append(
        (
            # 注意力权重参数的路径：注意力权重的投影值卷积层权重的权重参数路径
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.weight",
            # 对应的模型结构的路径：对应的注意力模块的投影值卷积层的批归一化层的权重参数路径
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.weight",
        )
    )
    # 重复上述过程，添加注意力权重的偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.bias",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.bias",
        )
    )
    # 添加注意力权重的批归一化层的运行时均值路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_mean",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_mean",
        )
    )
    # 添加注意力权重的批归一化层的运行时方差路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.running_var",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.running_var",
        )
    )
    # 添加注意力权重的批归一化层的跟踪批次数路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.convolution_projection_value.convolution_projection.normalization.num_batches_tracked",
            f"stage{idx}.blocks.{cnt}.attn.conv_proj_v.bn.num_batches_tracked",
        )
    )
    # 重复上述过程，添加注意力权重的查询投影层权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_q.weight",
        )
    )
    # 重复上述过程，添加注意力权重的查询投影层偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_query.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_q.bias",
        )
    )
    # 重复上述过程，添加注意力权重的键投影层权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_k.weight",
        )
    )
    # 重复上述过程，添加注意力权重的键投影层偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_key.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_k.bias",
        )
    )
    # 重复上述过程，添加注意力权重的值投影层权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj_v.weight",
        )
    )
    # 重复上述过程，添加注意力权重的值投影层偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.attention.projection_value.bias",
            f"stage{idx}.blocks.{cnt}.attn.proj_v.bias",
        )
    )
    # 重复上述过程，添加注意力权重的输出层权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.weight",
            f"stage{idx}.blocks.{cnt}.attn.proj.weight",
        )
    )
    # 将多个注意力权重参数对应的路径添加到 attention_weights 列表中
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.attention.output.dense.bias",  # 第一个参数：注意力输出密集层的偏置项路径
            f"stage{idx}.blocks.{cnt}.attn.proj.bias",  # 第二个参数：对应的注意力 projection 层的偏置项路径
        )
    )
    # 同上，添加注意力输出的权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.weight",  # 第一个参数：中间层的权重路径
            f"stage{idx}.blocks.{cnt}.mlp.fc1.weight"  # 第二个参数：对应的 MLP 第一层的权重路径
        )
    )
    # 同上，添加中间层的偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.intermediate.dense.bias",  # 第一个参数：中间层的偏置项路径
            f"stage{idx}.blocks.{cnt}.mlp.fc1.bias"  # 第二个参数：对应的 MLP 第一层的偏置项路径
        )
    )
    # 同上，添加输出层的权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.weight",  # 第一个参数：输出密集层的权重路径
            f"stage{idx}.blocks.{cnt}.mlp.fc2.weight"  # 第二个参数：对应的 MLP 第二层的权重路径
        )
    )
    # 同上，添加输出层的偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.output.dense.bias",  # 第一个参数：输出密集层的偏置项路径
            f"stage{idx}.blocks.{cnt}.mlp.fc2.bias"  # 第二个参数：对应的 MLP 第二层的偏置项路径
        )
    )
    # 同上，添加 layernorm_before 层的权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.weight",  # 第一个参数：layernorm_before 层的权重路径
            f"stage{idx}.blocks.{cnt}.norm1.weight"  # 第二个参数：对应的 norm1 层的权重路径
        )
    )
    # 同上，添加 layernorm_before 层的偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_before.bias",  # 第一个参数：layernorm_before 层的偏置项路径
            f"stage{idx}.blocks.{cnt}.norm1.bias"  # 第二个参数：对应的 norm1 层的偏置项路径
        )
    )
    # 同上，添加 layernorm_after 层的权重路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.weight",  # 第一个参数：layernorm_after 层的权重路径
            f"stage{idx}.blocks.{cnt}.norm2.weight"  # 第二个参数：对应的 norm2 层的权重路径
        )
    )
    # 同上，添加 layernorm_after 层的偏置项路径
    attention_weights.append(
        (
            f"cvt.encoder.stages.{idx}.layers.{cnt}.layernorm_after.bias",  # 第一个参数：layernorm_after 层的偏置项路径
            f"stage{idx}.blocks.{cnt}.norm2.bias"  # 第二个参数：对应的 norm2 层的偏置项路径
        )
    )
    # 返回添加了多个注意力权重参数路径的 attention_weights 列表
    return attention_weights
def cls_token(idx):
    """
    Function helps in renaming cls_token weights
    """
    # 初始化一个空列表用于存储权重重命名的元组
    token = []
    # 添加元组到列表中，将原始权重名称替换为新的名称
    token.append((f"cvt.encoder.stages.{idx}.cls_token", "stage2.cls_token"))
    # 返回存储重命名元组的列表
    return token


def final():
    """
    Function helps in renaming final classification layer
    """
    # 初始化一个空列表用于存储权重重命名的元组
    head = []
    # 添加元组到列表中，将原始权重名称替换为新的名称
    head.append(("layernorm.weight", "norm.weight"))
    head.append(("layernorm.bias", "norm.bias"))
    head.append(("classifier.weight", "head.weight"))
    head.append(("classifier.bias", "head.bias"))
    # 返回存储重命名元组的列表
    return head


def convert_cvt_checkpoint(cvt_model, image_size, cvt_file_name, pytorch_dump_folder):
    """
    Fucntion to convert the microsoft cvt checkpoint to huggingface checkpoint
    """
    # 定义 ImageNet 类别文件名和类别数量
    img_labels_file = "imagenet-1k-id2label.json"
    num_labels = 1000

    # 设置 Hugging Face Hub 仓库 ID
    repo_id = "huggingface/label-files"
    # 加载 ImageNet 类别文件，并转换为 ID 到类别名称的映射字典
    id2label = json.load(open(cached_download(hf_hub_url(repo_id, img_labels_file, repo_type="dataset")), "r"))
    id2label = {int(k): v for k, v in id2label.items()}

    # 将类别映射字典重新赋值
    id2label = id2label
    # 构建类别名称到 ID 的映射字典
    label2id = {v: k for k, v in id2label.items()}

    # 初始化 CVT 模型配置
    config = config = CvtConfig(num_labels=num_labels, id2label=id2label, label2id=label2id)

    # 根据 CVT 模型深度设置不同的配置
    if cvt_model.rsplit("/", 1)[-1][4:6] == "13":
        config.depth = [1, 2, 10]
    elif cvt_model.rsplit("/", 1)[-1][4:6] == "21":
        config.depth = [1, 4, 16]
    else:
        config.depth = [2, 2, 20]
        config.num_heads = [3, 12, 16]
        config.embed_dim = [192, 768, 1024]

    # 初始化 CVT 图像分类模型和图像处理器
    model = CvtForImageClassification(config)
    image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-base-224-22k-1k")
    image_processor.size["shortest_edge"] = image_size
    # 加载 CVT 模型的原始权重
    original_weights = torch.load(cvt_file_name, map_location=torch.device("cpu"))

    # 初始化存储 Hugging Face 模型权重的有序字典
    huggingface_weights = OrderedDict()
    list_of_state_dict = []

    # 遍历 CVT 模型的层和注意力头，重命名权重并存储到列表中
    for idx in range(len(config.depth)):
        if config.cls_token[idx]:
            list_of_state_dict = list_of_state_dict + cls_token(idx)
        list_of_state_dict = list_of_state_dict + embeddings(idx)
        for cnt in range(config.depth[idx]):
            list_of_state_dict = list_of_state_dict + attention(idx, cnt)

    # 添加最终分类层的权重重命名到列表中
    list_of_state_dict = list_of_state_dict + final()
    
    # 打印权重重命名信息
    for gg in list_of_state_dict:
        print(gg)
    
    # 将原始权重中的对应权重拷贝到 Hugging Face 模型的有序字典中
    for i in range(len(list_of_state_dict)):
        huggingface_weights[list_of_state_dict[i][0]] = original_weights[list_of_state_dict[i][1]]

    # 加载 Hugging Face 模型权重
    model.load_state_dict(huggingface_weights)
    # 保存 Hugging Face 模型和图像处理器
    model.save_pretrained(pytorch_dump_folder)
    image_processor.save_pretrained(pytorch_dump_folder)


# 下载权重文件：https://1drv.ms/u/s!AhIXJn_J-blW9RzF3rMW7SsLHa8h?e=blQ0Al

if __name__ == "__main__":
    # 初始化参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个名为 "cvt_model" 的命令行参数，设置默认值为 "cvt-w24"，类型为字符串，提供关于 cvt 模型的转换
    parser.add_argument(
        "--cvt_model",
        default="cvt-w24",
        type=str,
        help="Name of the cvt model you'd like to convert.",
    )
    # 添加一个名为 "image_size" 的命令行参数，设置默认值为 384，类型为整数，提供输入图片的大小
    parser.add_argument(
        "--image_size",
        default=384,
        type=int,
        help="Input Image Size",
    )
    # 添加一个名为 "cvt_file_name" 的命令行参数，设置默认值为 "cvtmodels\CvT-w24-384x384-IN-22k.pth"，类型为字符串，提供 cvt 模型文件的名称
    parser.add_argument(
        "--cvt_file_name",
        default=r"cvtmodels\CvT-w24-384x384-IN-22k.pth",
        type=str,
        help="Input Image Size",
    )
    # 添加一个名为 "pytorch_dump_folder_path" 的命令行参数，设置默认值为 None，类型为字符串，提供输出 PyTorch 模型目录的路径
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，将 cvt 模型转换为 PyTorch 模型
    convert_cvt_checkpoint(args.cvt_model, args.image_size, args.cvt_file_name, args.pytorch_dump_folder_path)
```