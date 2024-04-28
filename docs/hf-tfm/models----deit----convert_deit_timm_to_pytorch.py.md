# `.\models\deit\convert_deit_timm_to_pytorch.py`

```
# 导入模块
import argparse  # 解析命令行参数的模块
import json  # 处理 JSON 数据的模块
from pathlib import Path  # 处理文件路径的模块
import requests  # 发送 HTTP 请求的模块
import timm  # PyTorch 图像模型库
import torch  # PyTorch 深度学习库
from huggingface_hub import hf_hub_download  # 从 Hugging Face Hub 下载模型
from PIL import Image  # Python 图像处理库

# 导入 Transformers 库的相关模块
from transformers import DeiTConfig, DeiTForImageClassificationWithTeacher, DeiTImageProcessor
from transformers.utils import logging  # Transformers 工具模块中的日志模块

# 设置日志的详细程度为 info
logging.set_verbosity_info()

# 获取日志记录器
logger = logging.get_logger(__name__)


# 定义函数，用于创建需要重命名的键值对列表
# 参数 config: DeiT 模型配置对象
# 参数 base_model: 是否是基础模型，默认为 False
def create_rename_keys(config, base_model=False):
    # 创建空的重命名键值对列表
    rename_keys = []
    # 遍历 DeiT 模型的隐藏层数
    for i in range(config.num_hidden_layers):
        # 对于每一层，将原始的键名映射到新的键名，涉及到的是模型的各个部分的权重和偏置
        rename_keys.append((f"blocks.{i}.norm1.weight", f"deit.encoder.layer.{i}.layernorm_before.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"deit.encoder.layer.{i}.layernorm_before.bias"))
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"deit.encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"deit.encoder.layer.{i}.attention.output.dense.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"deit.encoder.layer.{i}.layernorm_after.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"deit.encoder.layer.{i}.layernorm_after.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"deit.encoder.layer.{i}.intermediate.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"deit.encoder.layer.{i}.intermediate.dense.bias"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"deit.encoder.layer.{i}.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"deit.encoder.layer.{i}.output.dense.bias"))

    # 投影层 + 位置编码
    # 将一些特殊的键名映射到新的键名，这些键名涉及到模型的投影层和位置编码
    rename_keys.extend(
        [
            ("cls_token", "deit.embeddings.cls_token"),
            ("dist_token", "deit.embeddings.distillation_token"),
            ("patch_embed.proj.weight", "deit.embeddings.patch_embeddings.projection.weight"),
            ("patch_embed.proj.bias", "deit.embeddings.patch_embeddings.projection.bias"),
            ("pos_embed", "deit.embeddings.position_embeddings"),
        ]
    )
    if base_model:
        # 如果是基础模型，进行下列键的重命名
        rename_keys.extend(
            [
                ("norm.weight", "layernorm.weight"),  # 将 "norm.weight" 重命名为 "layernorm.weight"
                ("norm.bias", "layernorm.bias"),  # 将 "norm.bias" 重命名为 "layernorm.bias"
                ("pre_logits.fc.weight", "pooler.dense.weight"),  # 将 "pre_logits.fc.weight" 重命名为 "pooler.dense.weight"
                ("pre_logits.fc.bias", "pooler.dense.bias"),  # 将 "pre_logits.fc.bias" 重命名为 "pooler.dense.bias"
            ]
        )

        # 如果只有基础模型，应该从所有以 "deit" 开头的键中去除 "deit"
        rename_keys = [(pair[0], pair[1][4:]) if pair[1].startswith("deit") else pair for pair in rename_keys]
        # 对于以 "deit" 开头的键，去除前缀 "deit"
    else:
        # 如果不是基础模型，进行下列键的重命名
        rename_keys.extend(
            [
                ("norm.weight", "deit.layernorm.weight"),  # 将 "norm.weight" 重命名为 "deit.layernorm.weight"
                ("norm.bias", "deit.layernorm.bias"),  # 将 "norm.bias" 重命名为 "deit.layernorm.bias"
                ("head.weight", "cls_classifier.weight"),  # 将 "head.weight" 重命名为 "cls_classifier.weight"
                ("head.bias", "cls_classifier.bias"),  # 将 "head.bias" 重命名为 "cls_classifier.bias"
                ("head_dist.weight", "distillation_classifier.weight"),  # 将 "head_dist.weight" 重命名为 "distillation_classifier.weight"
                ("head_dist.bias", "distillation_classifier.bias"),  # 将 "head_dist.bias" 重命名为 "distillation_classifier.bias"
            ]
        )

    return rename_keys
# 将每个编码器层的矩阵分割成查询（queries）、键（keys）和值（values）
def read_in_q_k_v(state_dict, config, base_model=False):
    # 遍历编码器层
    for i in range(config.num_hidden_layers):
        # 如果是基础模型，设置前缀为空字符串，否则设置为 "deit."
        if base_model:
            prefix = ""
        else:
            prefix = "deit."
        # 读取输入投影层（在 timm 中，这是一个矩阵 + 偏置）
        # 读取输入投影层权重和偏置
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # 将查询、键和值（按顺序）添加到状态字典中
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[
            : config.hidden_size, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[
            -config.hidden_size :, :
        ]
        state_dict[f"{prefix}encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


# 重命名字典中的键
def rename_key(dct, old, new):
    # 弹出旧键，并将其值赋给新键
    val = dct.pop(old)
    dct[new] = val


# 我们将在一张可爱猫咪的图片上验证我们的结果
def prepare_img():
    # 图片链接
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 获取图片并打开
    im = Image.open(requests.get(url, stream=True).raw)
    return im


# 在无梯度更新的上下文中，将 DeiT 检查点转换为 PyTorch 模型
@torch.no_grad()
def convert_deit_checkpoint(deit_name, pytorch_dump_folder_path):
    """
    复制/粘贴/调整模型的权重以符合我们的 DeiT 结构。
    """

    # 定义默认的 DeiT 配置
    config = DeiTConfig()
    # 所有的 DeiT 模型都有微调的头
    base_model = False
    # 数据集（在 ImageNet 2012 上微调）、patch_size 和 image_size
    config.num_labels = 1000
    repo_id = "huggingface/label-files"
    filename = "imagenet-1k-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    config.patch_size = int(deit_name[-6:-4])
    config.image_size = int(deit_name[-3:])
    # 架构的大小
    if deit_name[9:].startswith("tiny"):
        config.hidden_size = 192
        config.intermediate_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 3
    elif deit_name[9:].startswith("small"):
        config.hidden_size = 384
        config.intermediate_size = 1536
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    # 如果模型名称去除前9个字符后以"base"开头，则执行下一步，否则进入下一个条件分支
    if deit_name[9:].startswith("base"):
        pass
    # 如果模型名称去除前4个字符后以"large"开头，则执行以下步骤，设置一些配置参数为大型模型的值
    elif deit_name[4:].startswith("large"):
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # 从timm库中加载原始模型
    timm_model = timm.create_model(deit_name, pretrained=True)
    # 设置模型为评估模式
    timm_model.eval()

    # 加载原始模型的state_dict，并删除/重命名一些键
    state_dict = timm_model.state_dict()
    # 创建要重命名的键的列表
    rename_keys = create_rename_keys(config, base_model)
    # 遍历重命名键列表，并执行相应的重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    # 读取query、key和value的权重
    read_in_q_k_v(state_dict, config, base_model)

    # 加载HuggingFace模型
    # 创建DeiTForImageClassificationWithTeacher模型的实例，并设置为评估模式
    model = DeiTForImageClassificationWithTeacher(config).eval()
    # 加载模型的state_dict
    model.load_state_dict(state_dict)

    # 检查在由DeiTImageProcessor准备的图像上的模型输出
    # 计算图像的大小以保持与224像素的图像相同的比例
    size = int(
        (256 / 224) * config.image_size
    )  # 为了保持与224像素图像相同的比例，参见https://github.com/facebookresearch/deit/blob/ab5715372db8c6cad5740714b2216d55aeae052e/datasets.py#L103
    # 创建DeiTImageProcessor的实例，指定大小和裁剪尺寸
    image_processor = DeiTImageProcessor(size=size, crop_size=config.image_size)
    # 对准备好的图像进行编码，返回PyTorch张量
    encoding = image_processor(images=prepare_img(), return_tensors="pt")
    # 提取像素值
    pixel_values = encoding["pixel_values"]
    # 使用模型对图像进行推断
    outputs = model(pixel_values)

    # 对于timm模型也进行推断，用于与HuggingFace模型的输出进行比较
    timm_logits = timm_model(pixel_values)
    # 断言timm模型的输出形状与HuggingFace模型的logits形状相同
    assert timm_logits.shape == outputs.logits.shape
    # 断言timm模型的输出与HuggingFace模型的logits在一定的误差范围内相等
    assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)

    # 创建文件夹以保存PyTorch模型和图像处理器
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印要保存的模型名称和路径
    print(f"Saving model {deit_name} to {pytorch_dump_folder_path}")
    # 保存模型到指定路径
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印要保存的图像处理器的路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器到指定路径
    image_processor.save_pretrained(pytorch_dump_folder_path)
# 检查程序是否在主程序中执行
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加必需的参数
    parser.add_argument(
        "--deit_name", # 参数名
        default="vit_deit_base_distilled_patch16_224", # 默认值
        type=str, # 参数类型为字符串
        help="Name of the DeiT timm model you'd like to convert." # 参数的帮助信息
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", # 参数名
        default=None, # 默认值
        type=str, # 参数类型为字符串
        help="Path to the output PyTorch model directory." # 参数的帮助信息
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 调用函数，转换 DeiT 模型的检查点
    convert_deit_checkpoint(args.deit_name, args.pytorch_dump_folder_path)
```