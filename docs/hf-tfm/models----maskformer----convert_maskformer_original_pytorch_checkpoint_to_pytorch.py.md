# `.\models\maskformer\convert_maskformer_original_pytorch_checkpoint_to_pytorch.py`

```py
# coding=utf-8
# Copyright 2022 Meta Platforms, Inc. and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys  # 导入系统模块
from argparse import ArgumentParser  # 导入命令行参数解析模块
from dataclasses import dataclass  # 导入数据类装饰器
from pathlib import Path  # 导入处理路径的模块
from pprint import pformat  # 导入格式化输出模块
from typing import Any, Dict, Iterator, List, Set, Tuple  # 导入类型提示模块

import requests  # 导入处理 HTTP 请求的模块
import torch  # 导入 PyTorch 深度学习框架
import torchvision.transforms as T  # 导入图像转换模块
from detectron2.checkpoint import DetectionCheckpointer  # 导入检查点模块
from detectron2.config import get_cfg  # 导入配置获取函数
from detectron2.data import MetadataCatalog  # 导入元数据目录模块
from detectron2.projects.deeplab import add_deeplab_config  # 导入 DeepLab 配置模块
from PIL import Image  # 导入 Python 图像处理库
from torch import Tensor, nn  # 导入张量和神经网络模块

# 导入 MaskFormer 相关模块
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerForInstanceSegmentationOutput,
    MaskFormerModel,
    MaskFormerModelOutput,
)
from transformers.utils import logging  # 导入日志模块

StateDict = Dict[str, Tensor]  # 定义状态字典类型别名

logging.set_verbosity_info()  # 设置日志输出详细程度为信息级别
logger = logging.get_logger()  # 获取日志记录器对象

torch.manual_seed(0)  # 设置随机种子以确保实验结果可复现


class TrackedStateDict:
    def __init__(self, to_track: Dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): The dictionary we wish to track
        """
        self.to_track = to_track  # 初始化要跟踪的字典
        self._seen: Set[str] = set()  # 初始化一个集合，用于记录已经访问的键名

    def __getitem__(self, key: str) -> Any:
        return self.to_track[key]  # 返回指定键名对应的值

    def __setitem__(self, key: str, item: Any):
        self._seen.add(key)  # 将访问过的键名添加到集合中
        self.to_track[key] = item  # 更新字典中指定键名的值

    def diff(self) -> List[str]:
        """This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            List[str]: List of keys not yet updated
        """
        return set(self.to_track.keys()) - self._seen  # 返回未更新的键名列表

    def copy(self) -> Dict:
        # proxy the call to the internal dictionary
        return self.to_track.copy()  # 返回字典的浅拷贝


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # 定义图像的 URL
    img_data = requests.get(url, stream=True).raw  # 从 URL 获取图像数据
    im = Image.open(img_data)  # 打开图像数据
    return im  # 返回图像对象


@dataclass
class Args:
    """Fake command line arguments needed by maskformer/detectron implementation"""

    config_file: str  # 命令行参数类的属性：配置文件路径
# 从文件和命令行参数中加载配置信息
def setup_cfg(args: Args):
    # 获取一个新的配置对象
    cfg = get_cfg()
    # 添加 DeepLab 配置到配置对象
    add_deeplab_config(cfg)
    # 添加 MaskFormer 配置到配置对象
    add_mask_former_config(cfg)
    # 从配置文件中加载更多配置到当前配置对象
    cfg.merge_from_file(args.config_file)
    # 冻结配置对象，防止后续修改
    cfg.freeze()
    # 返回配置对象
    return cfg


class OriginalMaskFormerConfigToOursConverter:
    def __call__(self, original_config: object) -> MaskFormerConfig:
        # 获取原始配置对象的模型部分
        model = original_config.MODEL
        # 获取模型中的 MASK_FORMER 部分
        mask_former = model.MASK_FORMER
        # 获取模型中的 SWIN 部分
        swin = model.SWIN

        # 从元数据目录中获取测试数据集的类别信息
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])
        # 创建从类别 ID 到类别名称的映射字典
        id2label = dict(enumerate(dataset_catalog.stuff_classes))
        # 创建从类别名称到类别 ID 的映射字典
        label2id = {label: idx for idx, label in id2label.items()}

        # 创建 MaskFormerConfig 对象，并填充其属性值
        config: MaskFormerConfig = MaskFormerConfig(
            fpn_feature_size=model.SEM_SEG_HEAD.CONVS_DIM,
            mask_feature_size=model.SEM_SEG_HEAD.MASK_DIM,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            no_object_weight=mask_former.NO_OBJECT_WEIGHT,
            num_queries=mask_former.NUM_OBJECT_QUERIES,
            backbone_config={
                "pretrain_img_size": swin.PRETRAIN_IMG_SIZE,
                "image_size": swin.PRETRAIN_IMG_SIZE,
                "in_channels": 3,
                "patch_size": swin.PATCH_SIZE,
                "embed_dim": swin.EMBED_DIM,
                "depths": swin.DEPTHS,
                "num_heads": swin.NUM_HEADS,
                "window_size": swin.WINDOW_SIZE,
                "drop_path_rate": swin.DROP_PATH_RATE,
                "model_type": "swin",
            },
            dice_weight=mask_former.DICE_WEIGHT,
            ce_weight=1.0,
            mask_weight=mask_former.MASK_WEIGHT,
            decoder_config={
                "model_type": "detr",
                "max_position_embeddings": 1024,
                "encoder_layers": 6,
                "encoder_ffn_dim": 2048,
                "encoder_attention_heads": 8,
                "decoder_layers": mask_former.DEC_LAYERS,
                "decoder_ffn_dim": mask_former.DIM_FEEDFORWARD,
                "decoder_attention_heads": mask_former.NHEADS,
                "encoder_layerdrop": 0.0,
                "decoder_layerdrop": 0.0,
                "d_model": mask_former.HIDDEN_DIM,
                "dropout": mask_former.DROPOUT,
                "attention_dropout": 0.0,
                "activation_dropout": 0.0,
                "init_std": 0.02,
                "init_xavier_std": 1.0,
                "scale_embedding": False,
                "auxiliary_loss": False,
                "dilation": False,
                # 默认的预训练配置数值
            },
            id2label=id2label,
            label2id=label2id,
        )

        # 返回配置对象
        return config


class OriginalMaskFormerConfigToImageProcessorConverter:
    # 等待实现的类，用于将原始的 MaskFormer 配置转换为图像处理器配置
    pass
    # 定义一个特殊方法，使得对象可以被调用，并返回一个 MaskFormerImageProcessor 实例
    def __call__(self, original_config: object) -> MaskFormerImageProcessor:
        # 从配置中获取模型对象
        model = original_config.MODEL
        # 从配置中获取输入设置
        model_input = original_config.INPUT
        # 获取测试数据集的元数据目录
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])

        # 返回一个 MaskFormerImageProcessor 实例，并传入以下参数：
        return MaskFormerImageProcessor(
            # 计算并转换像素均值为列表形式
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            # 计算并转换像素标准差为列表形式
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            # 设置测试图像的最小尺寸
            size=model_input.MIN_SIZE_TEST,
            # 设置测试图像的最大尺寸
            max_size=model_input.MAX_SIZE_TEST,
            # 设置语义分割头部的类别数目
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            # 设置忽略索引，通常用于标注中的背景类别
            ignore_index=dataset_catalog.ignore_label,
            # 设置尺寸可分割性，通常为模型要求的倍数，这里为32，适用于 Swin 模型
            size_divisibility=32,
        )
# 定义一个类用于将原始模型的检查点转换为新模型的检查点
class OriginalMaskFormerCheckpointToOursConverter:
    # 初始化方法，接收原始模型和配置对象作为参数
    def __init__(self, original_model: nn.Module, config: MaskFormerConfig):
        self.original_model = original_model  # 存储原始模型
        self.config = config  # 存储配置对象

    # 弹出并重命名所有给定键对应的值，并将其添加到目标状态字典中
    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    # 替换像素模块的特定部分，并根据配置更新相应的目标状态字典
    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "pixel_level_module.decoder"  # 目标状态字典的前缀
        src_prefix: str = "sem_seg_head.pixel_decoder"  # 源状态字典的前缀

        # 使用给定配置更新背景模型
        self.replace_backbone(dst_state_dict, src_state_dict, self.config)

        # 定义一个函数用于为卷积层重命名键
        def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
            return [
                (f"{detectron_conv}.weight", f"{mine_conv}.0.weight"),
                (f"{detectron_conv}.norm.weight", f"{mine_conv}.1.weight"),
                (f"{detectron_conv}.norm.bias", f"{mine_conv}.1.bias"),
            ]

        # 添加用于转换的特定键对，如掩码特征的权重和偏置
        renamed_keys = [
            (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_projection.weight"),
            (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_projection.bias"),
        ]
        
        # 添加用于转换的卷积层的键对，例如特征金字塔网络（FPN）的stem层
        renamed_keys.extend(rename_keys_for_conv(f"{src_prefix}.layer_4", f"{dst_prefix}.fpn.stem"))

        # 循环添加FPN的各层，根据配置参数确定层数
        for src_i, dst_i in zip(range(3, 0, -1), range(0, 3)):
            renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.adapter_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.proj")
            )
            renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.layer_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.block")
            )

        # 调用pop_all方法，将所有重命名的键对应的值从源状态字典中弹出，并添加到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 定义一个方法，用于重命名 DETR 解码器的状态字典中的键
    def rename_keys_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的键前缀
        dst_prefix: str = "transformer_module.decoder"
        # 源状态字典的键前缀
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        
        # not sure why we are not popping direcetly here!
        # 不确定为什么这里没有直接弹出（删除）！
        
        # 在下面列出需要重命名的所有键（左侧为原始名称，右侧为我们的名称）
        rename_keys = []
        
        # 循环遍历解码器配置中的每一层
        for i in range(self.config.decoder_config.decoder_layers):
            # 添加重命名规则：自注意力机制的输出投影权重
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.self_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.weight",
                )
            )
            # 添加重命名规则：自注意力机制的输出投影偏置
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.self_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.self_attn.out_proj.bias",
                )
            )
            # 添加重命名规则：多头注意力机制的输出投影权重
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.multihead_attn.out_proj.weight",
                    f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.weight",
                )
            )
            # 添加重命名规则：多头注意力机制的输出投影偏置
            rename_keys.append(
                (
                    f"{src_prefix}.layers.{i}.multihead_attn.out_proj.bias",
                    f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.bias",
                )
            )
            # 添加重命名规则：线性层1的权重
            rename_keys.append((f"{src_prefix}.layers.{i}.linear1.weight", f"{dst_prefix}.layers.{i}.fc1.weight"))
            # 添加重命名规则：线性层1的偏置
            rename_keys.append((f"{src_prefix}.layers.{i}.linear1.bias", f"{dst_prefix}.layers.{i}.fc1.bias"))
            # 添加重命名规则：线性层2的权重
            rename_keys.append((f"{src_prefix}.layers.{i}.linear2.weight", f"{dst_prefix}.layers.{i}.fc2.weight"))
            # 添加重命名规则：线性层2的偏置
            rename_keys.append((f"{src_prefix}.layers.{i}.linear2.bias", f"{dst_prefix}.layers.{i}.fc2.bias"))
            # 添加重命名规则：层归一化1的权重
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm1.weight", f"{dst_prefix}.layers.{i}.self_attn_layer_norm.weight")
            )
            # 添加重命名规则：层归一化1的偏置
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm1.bias", f"{dst_prefix}.layers.{i}.self_attn_layer_norm.bias")
            )
            # 添加重命名规则：层归一化2的权重
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm2.weight", f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.weight")
            )
            # 添加重命名规则：层归一化2的偏置
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm2.bias", f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.bias")
            )
            # 添加重命名规则：层归一化3的权重
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm3.weight", f"{dst_prefix}.layers.{i}.final_layer_norm.weight")
            )
            # 添加重命名规则：层归一化3的偏置
            rename_keys.append(
                (f"{src_prefix}.layers.{i}.norm3.bias", f"{dst_prefix}.layers.{i}.final_layer_norm.bias")
            )

        # 返回包含所有重命名规则的列表
        return rename_keys
    # 定义一个方法用于替换 DETR 解码器中的权重和偏置
    def replace_q_k_v_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 设置目标状态字典中的键前缀
        dst_prefix: str = "transformer_module.decoder"
        # 设置源状态字典中的键前缀
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        # 循环遍历解码器层数量次数
        for i in range(self.config.decoder_config.decoder_layers):
            # 从源状态字典中弹出自注意力层的输入投影层的权重和偏置
            in_proj_weight = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_weight")
            in_proj_bias = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_bias")
            # 将自注意力层的查询、键和值（按顺序）添加到目标状态字典中
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
            # 从源状态字典中读取跨注意力层的输入投影层的权重和偏置
            in_proj_weight_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_weight")
            in_proj_bias_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_bias")
            # 将跨注意力层的查询、键和值（按顺序）添加到目标状态字典中
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
    # 用于替换`detr`模型的解码器部分的权重和偏置
    def replace_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标模型权重前缀
        dst_prefix: str = "transformer_module.decoder"
        # 源模型权重前缀
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        
        # 重命名两个模型权重的键名列表
        renamed_keys = self.rename_keys_in_detr_decoder(dst_state_dict, src_state_dict)
        
        # 添加更多的键名映射，例如层归一化的权重和偏置
        renamed_keys.extend(
            [
                (f"{src_prefix}.norm.weight", f"{dst_prefix}.layernorm.weight"),
                (f"{src_prefix}.norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        # 根据映射关系从源模型中移除对应的键值对
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

        # 替换`detr`模型解码器的query、key和value权重
        self.replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)

    # 替换`transformer_module`中的权重和偏置
    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标模型权重前缀
        dst_prefix: str = "transformer_module"
        # 源模型权重前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 调用`replace_detr_decoder`函数，替换解码器部分的权重和偏置
        self.replace_detr_decoder(dst_state_dict, src_state_dict)

        # 重命名`transformer_module`中的特定权重和偏置
        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.input_proj.weight", f"{dst_prefix}.input_projection.weight"),
            (f"{src_prefix}.input_proj.bias", f"{dst_prefix}.input_projection.bias"),
        ]

        # 根据映射关系从源模型中移除对应的键值对
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # 替换实例分割模块中的权重和偏置
    def replace_instance_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 注意：我们的情况中没有前缀，因此我们在后续处理中移除了键名中的“.”
        dst_prefix: str = ""
        # 源模型权重前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 定义要重命名的键名映射列表
        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        # 循环处理MLP层，构建映射列表
        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    (f"{src_prefix}.mask_embed.layers.{i}.weight", f"{dst_prefix}mask_embedder.{i}.0.weight"),
                    (f"{src_prefix}.mask_embed.layers.{i}.bias", f"{dst_prefix}mask_embedder.{i}.0.bias"),
                ]
            )
        
        # 记录日志，显示替换的键名映射列表
        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        
        # 根据映射关系从源模型中移除对应的键值对
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # 执行模型权重的转换
    def convert(self, mask_former: MaskFormerModel) -> MaskFormerModel:
        # 创建目标模型状态字典，基于输入模型的状态字典
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        # 获取原始模型的状态字典
        src_state_dict = self.original_model.state_dict()

        # 替换像素模块中的权重和偏置
        self.replace_pixel_module(dst_state_dict, src_state_dict)
        
        # 替换`transformer_module`中的权重和偏置
        self.replace_transformer_module(dst_state_dict, src_state_dict)

        # 记录未匹配的键名差异
        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        # 记录未复制的源模型键名列表
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        # 日志记录：操作完成
        logger.info("🙌 Done")

        # 使用更新后的目标状态字典加载模型权重
        mask_former.load_state_dict(dst_state_dict)

        # 返回更新后的模型
        return mask_former
    # 将给定的实例分割模型转换为另一种实例分割模型类型，并返回转换后的模型
    def convert_instance_segmentation(
        self, mask_former: MaskFormerForInstanceSegmentation
    ) -> MaskFormerForInstanceSegmentation:
        # 创建目标模型的状态字典，复制输入模型的状态字典
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        # 获取原始模型的状态字典
        src_state_dict = self.original_model.state_dict()

        # 用原始模型的状态字典替换目标模型中的实例分割模块
        self.replace_instance_segmentation_module(dst_state_dict, src_state_dict)

        # 将更新后的状态字典加载到输入的实例分割模型中
        mask_former.load_state_dict(dst_state_dict)

        # 返回更新后的实例分割模型
        return mask_former

    @staticmethod
    # 返回一个迭代器，该迭代器生成一系列元组，每个元组包含一个配置文件路径、一个检查点文件路径和一个配置目录路径
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        # 获取检查点目录下所有的.pkl文件路径列表
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

        # 遍历每个检查点文件路径
        for checkpoint in checkpoints:
            # 记录信息：转换正在处理的检查点文件名（不带扩展名）
            logger.info(f"💪 Converting {checkpoint.stem}")
            # 查找与当前检查点文件关联的配置文件路径
            config: Path = config_dir / checkpoint.parents[0].stem / "swin" / f"{checkpoint.stem}.yaml"

            # 返回当前配置文件路径、检查点文件路径和配置目录路径的元组
            yield config, checkpoint
def test(original_model, our_model: MaskFormerForInstanceSegmentation, image_processor: MaskFormerImageProcessor):
    # 使用torch.no_grad()上下文管理器，关闭梯度计算以加快推断速度
    with torch.no_grad():
        # 将原始模型和我们的模型设为评估模式
        original_model = original_model.eval()
        our_model = our_model.eval()

        # 准备图像数据
        im = prepare_img()

        # 图像转换的组合操作，包括调整大小、转换为Tensor、归一化
        tr = T.Compose(
            [
                T.Resize((384, 384)),  # 调整图像大小为384x384
                T.ToTensor(),  # 转换为Tensor
                T.Normalize(  # 归一化操作
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ],
        )

        # 对输入图像应用转换操作，并扩展维度以匹配模型的输入要求
        x = tr(im).unsqueeze(0)

        # 使用原始模型的backbone提取特征
        original_model_backbone_features = original_model.backbone(x.clone())

        # 使用我们的模型进行推断，同时请求输出隐藏状态
        our_model_output: MaskFormerModelOutput = our_model.model(x.clone(), output_hidden_states=True)

        # 对比原始模型和我们的模型的backbone特征是否接近
        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=1e-3
            ), "The backbone features are not the same."

        # 使用原始模型的语义分割头部进行像素解码
        original_model_pixel_out = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        # 对比原始模型和我们的模型的像素解码器的最后隐藏状态是否接近
        assert torch.allclose(
            original_model_pixel_out[0], our_model_output.pixel_decoder_last_hidden_state, atol=1e-4
        ), "The pixel decoder feature are not the same"

        # 测试完整模型的输出
        original_model_out = original_model([{"image": x.squeeze(0)}])

        # 获取原始模型的语义分割结果
        original_segmentation = original_model_out[0]["sem_seg"]

        # 使用我们的模型进行推断，并后处理分割结果
        our_model_out: MaskFormerForInstanceSegmentationOutput = our_model(x)

        our_segmentation = image_processor.post_process_segmentation(our_model_out, target_size=(384, 384))

        # 对比原始模型和我们的模型的语义分割结果是否接近
        assert torch.allclose(
            original_segmentation, our_segmentation, atol=1e-3
        ), "The segmentation image is not the same."

        # 记录测试通过的信息
        logger.info("✅ Test passed!")


def get_name(checkpoint_file: Path):
    # 从检查点文件名中提取模型名称
    model_name_raw: str = checkpoint_file.stem
    # model_name_raw 的格式类似于 maskformer_panoptic_swin_base_IN21k_384_bs64_554k
    parent_name: str = checkpoint_file.parents[0].stem
    backbone = "swin"
    dataset = ""
    
    # 根据父文件夹名称确定数据集类型
    if "coco" in parent_name:
        dataset = "coco"
    elif "ade" in parent_name:
        dataset = "ade"
    else:
        raise ValueError(f"{parent_name} must be wrong since we didn't find 'coco' or 'ade' in it ")

    # 支持的backbone类型列表
    backbone_types = ["tiny", "small", "base", "large"]

    # 从模型名称中匹配backbone类型
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0]

    # 组合最终的模型名称
    model_name = f"maskformer-{backbone}-{backbone_type}-{dataset}"

    return model_name


if __name__ == "__main__":
    # 命令行解析器，用于转换原始的MaskFormers模型到我们的实现
    parser = ArgumentParser(
        description="Command line to convert the original maskformers (with swin backbone) to our implementations."
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.pkl"
        ),
    )
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.yaml"
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
    parser.add_argument(
        "--maskformer_dir",
        required=True,
        type=Path,
        help=(
            "A path to MaskFormer's original implementation directory. You can download from here:"
            " https://github.com/facebookresearch/MaskFormer"
        ),
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 将命令行参数转换为对应的变量
    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.pytorch_dump_folder_path
    maskformer_dir: Path = args.maskformer_dir

    # 将 MaskFormer 的父目录添加到系统路径中
    sys.path.append(str(maskformer_dir.parent))
    
    # 导入所需的模块和类
    from MaskFormer.mask_former import add_mask_former_config
    from MaskFormer.mask_former.mask_former_model import MaskFormer as OriginalMaskFormer

    # 如果保存模型的目录不存在，则创建它及其父目录
    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    # 循环遍历原始 MaskFormer 的配置文件和检查点文件
    for config_file, checkpoint_file in OriginalMaskFormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        ):
            # 创建一个用于处理原始掩模形状配置到图像处理器转换的实例，并调用其方法
            image_processor = OriginalMaskFormerConfigToImageProcessorConverter()(setup_cfg(Args(config_file=config_file)))

        # 使用给定的配置文件设置配置对象
        original_config = setup_cfg(Args(config_file=config_file))

        # 根据原始配置创建原始掩模形状对象的参数
        mask_former_kwargs = OriginalMaskFormer.from_config(original_config)

        # 创建原始掩模形状模型的实例并设置为评估模式
        original_model = OriginalMaskFormer(**mask_former_kwargs).eval()

        # 加载预训练检查点文件到原始模型
        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        # 将原始配置转换为我们的掩模形状配置对象
        config: MaskFormerConfig = OriginalMaskFormerConfigToOursConverter()(original_config)

        # 创建我们的掩模形状模型的实例并设置为评估模式
        mask_former = MaskFormerModel(config=config).eval()

        # 创建用于将原始掩模形状检查点转换为我们的形式的转换器
        converter = OriginalMaskFormerCheckpointToOursConverter(original_model, config)

        # 将原始模型转换为我们的掩模形状模型
        maskformer = converter.convert(mask_former)

        # 创建用于实例分割的掩模形状模型的实例并设置为评估模式
        mask_former_for_instance_segmentation = MaskFormerForInstanceSegmentation(config=config).eval()

        # 设置实例分割模型的形状模型
        mask_former_for_instance_segmentation.model = mask_former

        # 将实例分割模型转换为我们的形式
        mask_former_for_instance_segmentation = converter.convert_instance_segmentation(
            mask_former_for_instance_segmentation
        )

        # 运行测试函数，传入原始模型、实例分割模型和图像处理器
        test(original_model, mask_former_for_instance_segmentation, image_processor)

        # 获取检查点文件的名称
        model_name = get_name(checkpoint_file)

        # 记录保存操作信息
        logger.info(f"🪄 Saving {model_name}")

        # 保存图像处理器预训练模型到指定目录
        image_processor.save_pretrained(save_directory / model_name)

        # 保存实例分割模型到指定目录
        mask_former_for_instance_segmentation.save_pretrained(save_directory / model_name)

        # 将图像处理器推送到 Hub 上
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add model",
            use_temp_dir=True,
        )

        # 将实例分割模型推送到 Hub 上
        mask_former_for_instance_segmentation.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add model",
            use_temp_dir=True,
        )
```