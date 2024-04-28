# `.\models\flava\convert_flava_original_pytorch_to_hf.py`

```
# 设置 Python 文件编码格式为 utf-8
# 版权声明
# 2022 年版权由 Meta Platforms 作者和 HuggingFace 团队所有。保留所有权利。
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求或经书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面协议同意，否则根据"AS IS"的基础分发软件，
# 没有任何种类的明示或暗示担保或条件
# 详见许可证 关于特定语言的授权权限和限制
import argparse  # 导入 argparse 模块，用于命令行参数解析
import os  # 导入 os 模块，用于与操作系统交互

import torch  # 导入 torch 模块，用于深度学习

from transformers import FlavaConfig, FlavaForPreTraining  # 从 transformers 模块中导入 FlavaConfig 和 FlavaForPreTraining 类
from transformers.models.flava.convert_dalle_to_flava_codebook import convert_dalle_checkpoint  # 导入 convert_dalle_checkpoint 函数


def count_parameters(state_dict):
    # 统计参数数量
    # 在原始 FLAVA 中，encoder.embeddings 被复制了两次
    return sum(param.float().sum() if "encoder.embeddings" not in key else 0 for key, param in state_dict.items())


def upgrade_state_dict(state_dict, codebook_state_dict):
    upgrade = {}  # 创建空字典 upgrade

    for key, value in state_dict.items():
        if "text_encoder.embeddings" in key or "image_encoder.embeddings" in key:
            continue

        # 更新模型状态字典的键
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

        upgrade[key] = value.float()

    for key, value in codebook_state_dict.items():
        upgrade[f"image_codebook.{key}"] = value

    return upgrade


@torch.no_grad()
def convert_flava_checkpoint(checkpoint_path, codebook_path, pytorch_dump_folder_path, config_path=None):
    """
    Copy/paste/tweak model's weights to transformers design.
    复制/粘贴/微调模型权重到 transformers 设计中。
    """
    if config_path is not None:
        config = FlavaConfig.from_pretrained(config_path)  # 如果配置路径不为 None，则从预训练配置加载配置信息
    else:
        config = FlavaConfig()  # 否则使用默认 FLAVA 配置

    hf_model = FlavaForPreTraining(config).eval()  # 创建一个 FLAVA 预训练模型对象，并设置为 eval 模式
    # 将 codebook 文件路径转换为指定格式的 state_dict
    codebook_state_dict = convert_dalle_checkpoint(codebook_path, None, save_checkpoint=False)
    
    # 检查是否存在指定的 checkpoint 文件路径
    if os.path.exists(checkpoint_path):
        # 如果存在，从指定路径加载 state_dict
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    else:
        # 如果不存在，从指定路径下载 state_dict
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")
    
    # 更新 hf_state_dict，将 codebook_state_dict 与 state_dict 结合
    hf_state_dict = upgrade_state_dict(state_dict, codebook_state_dict)
    
    # 使用 hf_state_dict 更新 hf_model 的状态
    hf_model.load_state_dict(hf_state_dict)
    
    # 重新获取 hf_model 的 state_dict
    hf_state_dict = hf_model.state_dict()
    
    # 计算 hf_model 的参数数量
    hf_count = count_parameters(hf_state_dict)
    
    # 计算 state_dict 和 codebook_state_dict 的参数总数
    state_dict_count = count_parameters(state_dict) + count_parameters(codebook_state_dict)
    
    # 检查 hf_count 和 state_dict_count 是否接近（绝对误差不超过 1e-3）
    assert torch.allclose(hf_count, state_dict_count, atol=1e-3)
    
    # 保存 hf_model 的训练参数到指定路径
    hf_model.save_pretrained(pytorch_dump_folder_path)
# 如果运行的脚本是主程序，而不是被导入的模块
if __name__ == "__main__":
    # 创建一个参数解析器
    parser = argparse.ArgumentParser()
    # 添加一个用于接收 PyTorch 模型输出路径的参数
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    # 添加一个用于接收 flava 检查点路径的参数
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to flava checkpoint")
    # 添加一个用于接收 flava 代码本检查点路径的参数
    parser.add_argument("--codebook_path", default=None, type=str, help="Path to flava codebook checkpoint")
    # 添加一个用于接收将要转换的模型的 hf 配置文件路径的参数
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    # 解析命令行参数
    args = parser.parse_args()

    # 调用函数 convert_flava_checkpoint，并传入命令行参数所得的路径
    convert_flava_checkpoint(args.checkpoint_path, args.codebook_path, args.pytorch_dump_folder_path, args.config_path)
```