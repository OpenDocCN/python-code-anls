# `.\models\flava\convert_flava_original_pytorch_to_hf.py`

```
# 导入命令行参数解析库
import argparse
# 导入操作系统功能模块
import os

# 导入PyTorch库
import torch

# 导入transformers库中的FlavaConfig和FlavaForPreTraining类
from transformers import FlavaConfig, FlavaForPreTraining
# 导入convert_dalle_to_flava_codebook模块中的convert_dalle_checkpoint函数
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint


# 定义函数：计算模型参数数量
def count_parameters(state_dict):
    # 对模型参数进行求和计算，但跳过名称中包含"encoder.embeddings"的部分
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


# 定义函数：升级模型状态字典
def upgrade_state_dict(state_dict, codebook_state_dict):
    # 初始化升级后的状态字典
    upgrade = {}

    # 遍历原始状态字典中的键值对
    for key, value in state_dict.items():
        # 如果键名中包含"text_encoder.embeddings"或"image_encoder.embeddings"，则跳过处理
        if "text_encoder.embeddings" in key or "image_encoder.embeddings" in key:
            continue

        # 替换键名中的特定子串，以适配新模型结构
        key = key.replace("heads.cmd.mim_head.cls.predictions", "mmm_image_head")
        key = key.replace("heads.cmd.mlm_head.cls.predictions", "mmm_text_head")
        key = key.replace("heads.cmd.itm_head.cls", "itm_head")
        key = key.replace("heads.cmd.itm_head.pooler", "itm_head.pooler")
        key = key.replace("heads.cmd.clip_head.logit_scale", "flava.logit_scale")
        key = key.replace("heads.fairseq_mlm.cls.predictions", "mlm_head")
        key = key.replace("heads.imagenet.mim_head.cls.predictions", "mim_head")
        key = key.replace("mm_text_projection", "flava.text_to_mm_projection")
        key = key.replace("mm_image_projection", "flava.image_to_mm_projection")
        key = key.replace("image_encoder.module", "flava.image_model")
        key = key.replace("text_encoder.module", "flava.text_model")
        key = key.replace("mm_encoder.module.encoder.cls_token", "flava.multimodal_model.cls_token")
        key = key.replace("mm_encoder.module", "flava.multimodal_model")
        key = key.replace("text_projection", "flava.text_projection")
        key = key.replace("image_projection", "flava.image_projection")

        # 将处理后的键值对应存入升级后的状态字典
        upgrade[key] = value.float()

    # 将代码簿状态字典中的键值对应存入升级后的状态字典，前缀为"image_codebook."
    for key, value in codebook_state_dict.items():
        upgrade[f"image_codebook.{key}"] = value

    return upgrade


# 定义函数：转换FLAVA模型的检查点
@torch.no_grad()
def convert_flava_checkpoint(checkpoint_path, codebook_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    将模型权重复制/粘贴/调整为transformers设计。
    """
    # 如果提供了配置文件路径，则从预训练配置文件中加载配置，否则使用默认配置
    if config_path is not None:
        config = FlavaConfig.from_pretrained(config_path)
    else:
        config = FlavaConfig()

    # 创建一个FlavaForPreTraining模型，并设置为评估模式
    hf_model = FlavaForPreTraining(config).eval()
    # 调用函数 `convert_dalle_checkpoint`，将 `codebook_path` 转换为 DALL-E 模型的状态字典
    codebook_state_dict = convert_dalle_checkpoint(codebook_path, None, save_checkpoint=False)

    # 检查 `checkpoint_path` 是否存在
    if os.path.exists(checkpoint_path):
        # 如果存在，则从本地加载 PyTorch 模型状态字典
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        # 如果不存在，则从指定的 URL 加载 PyTorch 模型状态字典
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")

    # 升级模型的状态字典 `state_dict`，结合 `codebook_state_dict`
    hf_state_dict = upgrade_state_dict(state_dict, codebook_state_dict)

    # 将升级后的状态字典加载到 `hf_model` 中
    hf_model.load_state_dict(hf_state_dict)

    # 获取 `hf_model` 的当前状态字典
    hf_state_dict = hf_model.state_dict()

    # 计算 `hf_model` 中可训练参数的总数
    hf_count = count_parameters(hf_state_dict)

    # 计算总共的模型参数数目，包括 `state_dict` 和 `codebook_state_dict`
    state_dict_count = count_parameters(state_dict) + count_parameters(codebook_state_dict)

    # 使用断言确保 `hf_count` 与 `state_dict_count` 之间的参数数量非常接近，允许误差为 1e-3
    assert torch.allclose(hf_count, state_dict_count, atol=1e-3)

    # 将 `hf_model` 的模型权重保存到指定的 PyTorch 转储文件夹路径中
    hf_model.save_pretrained(pytorch_dump_folder_path)
# 如果当前脚本作为主程序运行，则执行以下代码块
if __name__ == "__main__":
    # 创建参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加命令行参数，用于指定输出的 PyTorch 模型的路径
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加命令行参数，用于指定 flava checkpoint 的路径
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    # 添加命令行参数，用于指定 flava codebook checkpoint 的路径
    parser.add_argument("--codebook_path", default=None, type=str, help="Path to flava codebook checkpoint")
    # 添加命令行参数，用于指定待转换模型的 hf config.json 文件的路径
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_flava_checkpoint，传入命令行参数中指定的路径信息
    convert_flava_checkpoint(args.checkpoint_path, args.codebook_path, args.pytorch_dump_folder_path, args.config_path)
```