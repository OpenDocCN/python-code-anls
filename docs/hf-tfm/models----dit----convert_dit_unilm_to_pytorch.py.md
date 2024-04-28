# `.\models\dit\convert_dit_unilm_to_pytorch.py`

```py
# 设置文件编码为 utf-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求或书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据“原样”分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
"""从 unilm 存储库转换 DiT 检查点。"""

# 导入所需的库
import argparse
import json
from pathlib import Path
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import BeitConfig, BeitForImageClassification, BeitForMaskedImageModeling, BeitImageProcessor
from transformers.image_utils import PILImageResampling
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取 logger
logger = logging.get_logger(__name__)

# 创建要重命名的所有键列表（原始名称在左侧，我们的名称在右侧）
def create_rename_keys(config, has_lm_head=False, is_semantic=False):
    # 如果是语义模型，则添加前缀
    prefix = "backbone." if is_semantic else ""

    rename_keys = []
    for i in range(config.num_hidden_layers):
        # 编码器层：输出投影、2 个前馈神经网络和 2 个层归一化
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
    # 扩展重命名键列表，将原始键和新键对应关系添加到列表中
    rename_keys.extend(
        [
            (f"{prefix}cls_token", "beit.embeddings.cls_token"),
            (f"{prefix}patch_embed.proj.weight", "beit.embeddings.patch_embeddings.projection.weight"),
            (f"{prefix}patch_embed.proj.bias", "beit.embeddings.patch_embeddings.projection.bias"),
            (f"{prefix}pos_embed", "beit.embeddings.position_embeddings"),
        ]
    )

    # 如果模型有语言模型头
    if has_lm_head:
        # 添加 mask token 和 layernorm 的重命名键对应关系到列表中
        rename_keys.extend(
            [
                ("mask_token", "beit.embeddings.mask_token"),
                ("norm.weight", "layernorm.weight"),
                ("norm.bias", "layernorm.bias"),
            ]
        )
    else:
        # 添加 layernorm 和分类头的重命名键对应关系到列表中
        rename_keys.extend(
            [
                ("fc_norm.weight", "beit.pooler.layernorm.weight"),
                ("fc_norm.bias", "beit.pooler.layernorm.bias"),
                ("head.weight", "classifier.weight"),
                ("head.bias", "classifier.bias"),
            ]
        )

    # 返回重命名键列表
    return rename_keys
# we split up the matrix of each encoder layer into queries, keys and values
# 将每个编码器层的矩阵分割成查询、键和数值

def read_in_q_k_v(state_dict, config, has_lm_head=False, is_semantic=False):
    # 循环遍历编码器层的数量
    for i in range(config.num_hidden_layers):
        prefix = "backbone." if is_semantic else ""
        # queries, keys and values
        # 获取查询、键和数值
        in_proj_weight = state_dict.pop(f"{prefix}blocks.{i}.attn.qkv.weight")
        q_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.q_bias")
        v_bias = state_dict.pop(f"{prefix}blocks.{i}.attn.v_bias")

        # 更新状态字典，将查询权重和偏置存入对应的位置
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.query.bias"] = q_bias
        # 更新状态字典，将键权重存入对应位置
        state_dict[f"beit.encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[config.hidden_size : config.hidden_size * 2, :]
        # 更新状态字典，将数值权重和偏置存入对应位置
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"beit.encoder.layer.{i}.attention.attention.value.bias"] = v_bias

        # gamma_1 and gamma_2
        # we call them lambda because otherwise they are renamed when using .from_pretrained
        # 更新状态字典，将gamma_1和gamma_2赋值给lambda_1和lambda_2
        gamma_1 = state_dict.pop(f"{prefix}blocks.{i}.gamma_1")
        gamma_2 = state_dict.pop(f"{prefix}blocks.{i}.gamma_2")
        state_dict[f"beit.encoder.layer.{i}.lambda_1"] = gamma_1
        state_dict[f"beit.encoder.layer.{i}.lambda_2"] = gamma_2


# 重命名字典中的键
def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
# 我们将在一张可爱猫咪的图片上验证结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 使用requests库获取图片的字节流，并用Image库打开图片
    im = Image.open(requests.get(url, stream=True).raw)
    # 返回图片对象
    return im


# 对检查点进行转换
@torch.no_grad()
def convert_dit_checkpoint(checkpoint_url, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our BEiT structure.
    """

    # 定义默认的BEiT配置
    # 根据检查点URL判断是否有lm_head
    has_lm_head = False if "rvlcdip" in checkpoint_url else True
    # 定义BEiT配置
    config = BeitConfig(use_absolute_position_embeddings=True, use_mask_token=has_lm_head)

    # size of the architecture
    # 根据检查点URL的不同设置BEiT的隐藏层大小、中间层大小等参数
    if "large" in checkpoint_url or "dit-l" in checkpoint_url:
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 16

    # labels
    # 根据检查点URL初始化labels相关参数
    if "rvlcdip" in checkpoint_url:
        config.num_labels = 16
        # 从hub下载rvlcdip标签文件并载入
        repo_id = "huggingface/label-files"
        filename = "rvlcdip-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}

    # load state_dict of original model, remove and rename some keys
    # 载入原始模型的状态字典，并移除/重命名一些键，将权重调整到BEiT结构中
    # 从指定 URL 加载模型的状态字典，并且只加载模型的参数
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["model"]

    # 创建重命名键列表
    rename_keys = create_rename_keys(config, has_lm_head=has_lm_head)

    # 遍历重命名键列表，执行重命名操作
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    # 读取 Q、K、V 的参数
    read_in_q_k_v(state_dict, config, has_lm_head=has_lm_head)

    # 根据是否含有语言模型头来选择加载 huggingface 模型
    model = BeitForMaskedImageModeling(config) if has_lm_head else BeitForImageClassification(config)
    # 设定模型状态为评估模式
    model.eval()
    # 加载模型的状态字典
    model.load_state_dict(state_dict)

    # 创建 BeitImageProcessor 对象
    image_processor = BeitImageProcessor(
        size=config.image_size, resample=PILImageResampling.BILINEAR, do_center_crop=False
    )
    # 准备图片
    image = prepare_img()

    # 对图片进行编码
    encoding = image_processor(images=image, return_tensors="pt")
    # 获取像素值
    pixel_values = encoding["pixel_values"]

    # 将像素值输入到模型中，获取输出结果
    outputs = model(pixel_values)
    # 获取逻辑回归结果
    logits = outputs.logits

    # 验证逻辑回归结果的形状
    expected_shape = [1, 16] if "rvlcdip" in checkpoint_url else [1, 196, 8192]
    assert logits.shape == torch.Size(expected_shape), "Shape of logits not as expected"

    # 创建文件夹，用于保存模型文件
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    # 打印保存模型的文件夹路径
    print(f"Saving model to {pytorch_dump_folder_path}")
    # 保存模型权重参数
    model.save_pretrained(pytorch_dump_folder_path)
    # 打印保存图像处理器的文件夹路径
    print(f"Saving image processor to {pytorch_dump_folder_path}")
    # 保存图像处理器参数
    image_processor.save_pretrained(pytorch_dump_folder_path)

    # 如果需要将模型推送到 Hub 上
    if push_to_hub:
        # 根据模型是否含有语言模型头选择模型名称
        if has_lm_head:
            model_name = "dit-base" if "base" in checkpoint_url else "dit-large"
        else:
            model_name = "dit-base-finetuned-rvlcdip" if "dit-b" in checkpoint_url else "dit-large-finetuned-rvlcdip"
        # 将图像处理器推送到 Hub 上
        image_processor.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add image processor",
            use_temp_dir=True,
        )
        # 将模型推送到 Hub 上
        model.push_to_hub(
            repo_path_or_name=Path(pytorch_dump_folder_path, model_name),
            organization="nielsr",
            commit_message="Add model",
            use_temp_dir=True,
        )
# 如果该脚本是作为主程序运行
if __name__ == "__main__":
    # 创建命令行解析器
    parser = argparse.ArgumentParser()

    # 添加命令行参数，指定checkpoint_url，默认为给定的URL，类型为字符串，提供帮助信息
    parser.add_argument(
        "--checkpoint_url",
        default="https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    # 添加命令行参数，指定pytorch_dump_folder_path，默认为None，类型为字符串，提供帮助信息
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    # 添加命令行参数，指定push_to_hub，如果存在则为True
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 调用convert_dit_checkpoint函数并传入参数
    convert_dit_checkpoint(args.checkpoint_url, args.pytorch_dump_folder_path, args.push_to_hub)
```