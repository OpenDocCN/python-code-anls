# `.\models\oneformer\convert_to_hf_oneformer.py`

```py
# coding=utf-8
# 声明文件编码格式为 UTF-8

# 版权声明和许可证信息
# Copyright 2022 SHI Labs and The HuggingFace Inc. team. All rights reserved.
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

"""Convert OneFormer checkpoints from the original repository. URL: https://github.com/SHI-Labs/OneFormer"""
# 文件描述：从原始存储库转换 OneFormer 检查点的功能

import os
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple

import requests
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor, nn

# 尝试导入依赖库（detectron2），如果导入失败则忽略
try:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config
except ImportError:
    pass

# 导入 OneFormer 相关模块和类
from transformers import CLIPTokenizer, DinatConfig, SwinConfig
from transformers.models.oneformer.image_processing_oneformer import OneFormerImageProcessor
from transformers.models.oneformer.modeling_oneformer import (
    OneFormerConfig,
    OneFormerForUniversalSegmentation,
    OneFormerForUniversalSegmentationOutput,
    OneFormerModel,
    OneFormerModelOutput,
)
from transformers.models.oneformer.processing_oneformer import OneFormerProcessor
from transformers.utils import logging

# 定义 StateDict 类型别名
StateDict = Dict[str, Tensor]

# 设置日志的详细程度为信息级别
logging.set_verbosity_info()
logger = logging.get_logger()

# 设定随机数种子
torch.manual_seed(0)


class TrackedStateDict:
    def __init__(self, to_track: Dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): The dictionary we wish to track
        """
        self.to_track = to_track
        self._seen: Set[str] = set()

    def __getitem__(self, key: str) -> Any:
        return self.to_track[key]

    def __setitem__(self, key: str, item: Any):
        self._seen.add(key)
        self.to_track[key] = item

    def diff(self) -> List[str]:
        """This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            List[str]: List of keys not yet updated
        """
        return set(self.to_track.keys()) - self._seen

    def copy(self) -> Dict:
        # proxy the call to the internal dictionary
        return self.to_track.copy()


# 准备用于验证结果的图像
def prepare_img():
    # 定义一个 URL 变量，指向图像文件的网络地址
    url = "https://praeclarumjj3.github.io/files/coco.jpeg"
    # 使用 requests 库发起 GET 请求获取图像数据，设置 stream=True 以获取原始字节流
    img_data = requests.get(url, stream=True).raw
    # 使用 Image.open() 方法打开图像数据流，返回一个图像对象
    im = Image.open(img_data)
    # 返回打开的图像对象
    return im
# 定义一个数据类，用于存储模型配置文件路径等命令行参数
@dataclass
class Args:
    """Fake command line arguments needed by oneformer/detectron2 implementation"""

    config_file: str


# 配置模型的函数，从指定的配置文件和命令行参数加载配置
def setup_cfg(args: Args):
    # 获取一个空的配置对象
    cfg = get_cfg()
    # 添加 Deeplab 配置到配置对象
    add_deeplab_config(cfg)
    # 添加通用配置到配置对象
    add_common_config(cfg)
    # 添加 OneFormer 特定配置到配置对象
    add_oneformer_config(cfg)
    # 添加 Swin 模型配置到配置对象
    add_swin_config(cfg)
    # 添加 Dinat 模型配置到配置对象
    add_dinat_config(cfg)
    # 从指定的配置文件中合并配置到配置对象
    cfg.merge_from_file(args.config_file)
    # 冻结配置，防止进一步修改
    cfg.freeze()
    # 返回配置对象
    return cfg


# 将原始 OneFormer 配置转换为我们自己的处理器配置的类
class OriginalOneFormerConfigToOursConverter:

# 将原始 OneFormer 配置转换为处理器配置的类
class OriginalOneFormerConfigToProcessorConverter:

    # 将原始配置对象转换为 OneFormerProcessor 实例的调用方法
    def __call__(self, original_config: object, model_repo: str) -> OneFormerProcessor:
        # 获取原始模型和输入配置
        model = original_config.MODEL
        model_input = original_config.INPUT
        # 获取元数据目录中指定测试数据集的信息
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST_PANOPTIC[0])

        # 根据模型仓库名称选择类别信息文件
        if "ade20k" in model_repo:
            class_info_file = "ade20k_panoptic.json"
        elif "coco" in model_repo:
            class_info_file = "coco_panoptic.json"
        elif "cityscapes" in model_repo:
            class_info_file = "cityscapes_panoptic.json"
        else:
            raise ValueError("Invalid Dataset!")

        # 创建 OneFormerImageProcessor 实例，设置图像处理参数和类别信息文件
        image_processor = OneFormerImageProcessor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_index=dataset_catalog.ignore_label,
            class_info_file=class_info_file,
        )

        # 从模型仓库加载 CLIPTokenizer 实例
        tokenizer = CLIPTokenizer.from_pretrained(model_repo)

        # 返回一个 OneFormerProcessor 实例，包含图像处理器、分词器及相关配置
        return OneFormerProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            task_seq_length=original_config.INPUT.TASK_SEQ_LEN,
            max_seq_length=original_config.INPUT.MAX_SEQ_LEN,
        )


# 将原始 OneFormer 检查点转换为我们自己的检查点转换器的类
class OriginalOneFormerCheckpointToOursConverter:

    # 初始化函数，接受原始模型和 OneFormer 配置对象作为参数
    def __init__(self, original_model: nn.Module, config: OneFormerConfig):
        self.original_model = original_model
        self.config = config

    # 弹出所有重命名的键到目标状态字典中
    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    # Swin Backbone
    # Dinat Backbone
    # Backbone + Pixel Decoder
    # Transformer Decoder
    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的键前缀
        dst_prefix: str = "transformer_module.decoder.layers"
        # 源状态字典的键前缀
        src_prefix: str = "sem_seg_head.predictor"
        
        # 遍历每个解码器层
        for i in range(self.config.decoder_layers - 1):
            # 从源状态字典中弹出自注意力层的输入投影层权重和偏置
            in_proj_weight = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight"
            )
            in_proj_bias = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias"
            )
            
            # 将查询、键和值（按顺序）添加到目标状态字典
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    def replace_task_mlp(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的键前缀
        dst_prefix: str = "task_encoder"
        # 源状态字典的键前缀
        src_prefix: str = "task_mlp"

        # 定义用于重命名权重和偏置的函数
        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = []

        # 遍历两个MLP层
        for i in range(2):
            # 扩展重命名键列表，将源状态字典中的对应键映射到目标状态字典
            renamed_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.layers.{i}", f"{dst_prefix}.task_mlp.layers.{i}.0")
            )

        # 调用方法，从两个状态字典中移除所有重命名的键
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    def replace_text_projector(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的键前缀
        dst_prefix: str = "text_mapper.text_projector"
        # 源状态字典的键前缀
        src_prefix: str = "text_projector"

        # 定义用于重命名权重和偏置的函数
        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = []

        # 根据文本编码器配置中的投影层数量，重命名权重和偏置的键
        for i in range(self.config.text_encoder_config["text_encoder_proj_layers"]):
            renamed_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.layers.{i}", f"{dst_prefix}.{i}.0"))

        # 调用方法，从两个状态字典中移除所有重命名的键
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 定义一个方法，用于将源状态字典中的文本编码器部分映射到目标状态字典中
    def replace_text_mapper(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典中文本编码器的前缀
        dst_prefix: str = "text_mapper.text_encoder"
        # 源状态字典中文本编码器的前缀
        src_prefix: str = "text_encoder"

        # 调用内部方法，将源状态字典中的投影器映射到目标状态字典中
        self.replace_text_projector(dst_state_dict, src_state_dict)

        # 定义一个内部函数，用于重命名权重和偏置的键
        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        # 定义一个内部函数，用于重命名注意力机制相关的键
        def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
            # 初始化注意力机制相关的键
            attn_keys = [
                (f"{src_prefix}.in_proj_bias", f"{dst_prefix}.in_proj_bias"),
                (f"{src_prefix}.in_proj_weight", f"{dst_prefix}.in_proj_weight"),
            ]
            # 扩展注意力机制中的权重和偏置键
            attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.out_proj", f"{dst_prefix}.out_proj"))

            return attn_keys

        # 定义一个内部函数，用于重命名层级的键
        def rename_keys_for_layer(src_prefix: str, dst_prefix: str):
            # 初始化层级的键列表
            resblock_keys = []

            # 扩展层级键列表，包括多层感知机的权重和偏置
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.mlp.c_fc", f"{dst_prefix}.mlp.fc1"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.mlp.c_proj", f"{dst_prefix}.mlp.fc2"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_1", f"{dst_prefix}.layer_norm1"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_2", f"{dst_prefix}.layer_norm2"))
            resblock_keys.extend(rename_keys_for_attn(f"{src_prefix}.attn", f"{dst_prefix}.self_attn"))

            return resblock_keys

        # 初始化已重命名的键列表，直接包含特定的重命名键
        renamed_keys = [
            ("prompt_ctx.weight", "text_mapper.prompt_ctx.weight"),
        ]

        # 扩展已重命名的键列表，包括位置嵌入和令牌嵌入的权重
        renamed_keys.extend(
            [
                (f"{src_prefix}.positional_embedding", f"{dst_prefix}.positional_embedding"),
                (f"{src_prefix}.token_embedding.weight", f"{dst_prefix}.token_embedding.weight"),
            ]
        )

        # 扩展已重命名的键列表，包括最终层级的层归一化和前缀的权重和偏置
        renamed_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_final", f"{dst_prefix}.ln_final"))

        # 循环遍历文本编码器配置中的所有层，重命名每个层级的键
        for i in range(self.config.text_encoder_config["text_encoder_num_layers"]):
            renamed_keys.extend(
                rename_keys_for_layer(
                    f"{src_prefix}.transformer.resblocks.{i}", f"{dst_prefix}.transformer.layers.{i}"
                )
            )

        # 调用对象方法，从目标状态字典和源状态字典中弹出所有已重命名的键
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 将给定的模型转换为特定格式的模型对象
    def convert(self, oneformer: OneFormerModel, is_swin: bool) -> OneFormerModel:
        # 创建目标模型状态字典的跟踪对象，复制输入模型的状态字典
        dst_state_dict = TrackedStateDict(oneformer.state_dict())
        # 获取原始模型的状态字典
        src_state_dict = self.original_model.state_dict()

        # 替换目标模型的像素模块，使用原始模型的对应部分
        self.replace_pixel_module(dst_state_dict, src_state_dict, is_swin)
        # 替换目标模型的变换模块，使用原始模型的对应部分
        self.replace_transformer_module(dst_state_dict, src_state_dict)
        # 替换目标模型的任务 MLP，使用原始模型的对应部分
        self.replace_task_mlp(dst_state_dict, src_state_dict)
        
        # 如果配置为训练模式，则替换目标模型的文本映射器，使用原始模型的对应部分
        if self.config.is_training:
            self.replace_text_mapper(dst_state_dict, src_state_dict)

        # 记录目标模型状态字典中未复制的键
        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        # 记录原始模型状态字典中未被复制的键
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        # 输出完成信息
        logger.info("🙌 Done")

        # 将更新后的状态字典加载到输入的模型对象中
        oneformer.load_state_dict(dst_state_dict)

        # 返回更新后的模型对象
        return oneformer

    @staticmethod
    # 使用指定的目录查找检查点文件和配置文件，返回迭代器
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        # 获取所有以 .pth 结尾的检查点文件列表
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pth")

        # 遍历每个检查点文件
        for checkpoint in checkpoints:
            # 记录正在转换的检查点文件信息
            logger.info(f"💪 Converting {checkpoint.stem}")
            # 查找关联的配置文件，根据检查点文件名生成配置文件路径
            config: Path = config_dir / f"{checkpoint.stem}.yaml"

            # 返回配置文件路径、检查点文件路径的迭代器
            yield config, checkpoint
# 对语义分割模型输出进行后处理，将输出调整到指定的目标大小
def post_process_sem_seg_output(outputs: OneFormerForUniversalSegmentationOutput, target_size: Tuple[int, int]):
    # 获取类别查询的逻辑回归输出，形状为 [BATCH, QUERIES, CLASSES + 1]
    class_queries_logits = outputs.class_queries_logits
    # 获取掩码查询的逻辑回归输出，形状为 [BATCH, QUERIES, HEIGHT, WIDTH]
    masks_queries_logits = outputs.masks_queries_logits
    if target_size is not None:
        # 如果指定了目标大小，则通过双线性插值调整掩码查询的逻辑回归输出尺寸
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    # 去除掉空类别 `[..., :-1]`，得到掩码类别概率，形状为 [BATCH, QUERIES, CLASSES]
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    # 将掩码查询的逻辑回归输出通过 sigmoid 函数转换为概率，形状为 [BATCH, QUERIES, HEIGHT, WIDTH]
    masks_probs = masks_queries_logits.sigmoid()
    # 使用 Einstein Summation 计算语义分割结果，形状为 [BATCH, CLASSES, HEIGHT, WIDTH]
    # 其中 masks_classes 是掩码类别概率，masks_probs 是掩码概率
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

    return segmentation


def test(
    original_model,
    our_model: OneFormerForUniversalSegmentation,
    processor: OneFormerProcessor,
    model_repo: str,
):
    # 内部函数，用于对文本进行预处理，将文本列表转换为模型输入的张量
    def _preprocess_text(text_list=None, max_length=77):
        if text_list is None:
            raise ValueError("tokens cannot be None.")

        # 使用 tokenizer 对文本列表进行编码处理，进行填充和截断以匹配模型输入要求
        tokens = tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        # 遍历生成每个文本的张量输入
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        # 将列表转换为张量，并按第一维拼接，形成最终的输入张量
        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs
    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算，以加快推理速度
    with torch.no_grad():
        # 使用 CLIPTokenizer 从预训练模型库加载 tokenizer
        tokenizer = CLIPTokenizer.from_pretrained(model_repo)
        # 将原始模型和我们的模型设置为评估模式
        original_model = original_model.eval()
        our_model = our_model.eval()

        # 准备图像数据
        im = prepare_img()

        # 定义图像预处理的转换操作序列
        tr = T.Compose(
            [
                # 调整图像大小为 (640, 640)
                T.Resize((640, 640)),
                # 将图像转换为张量
                T.ToTensor(),
                # 标准化图像张量
                T.Normalize(
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ],
        )

        # 对图像进行预处理并增加一个维度，以符合模型的输入要求
        x = tr(im).unsqueeze(0)

        # 定义任务的输入文本
        task_input = ["the task is semantic"]
        # 对任务文本进行预处理，确保长度不超过处理器的最大序列长度
        task_token = _preprocess_text(task_input, max_length=processor.task_seq_length)

        # 提取原始模型的骨干网络特征
        original_model_backbone_features = original_model.backbone(x.clone())

        # 使用我们的模型进行推理，并要求输出隐藏状态
        our_model_output: OneFormerModelOutput = our_model.model(x.clone(), task_token, output_hidden_states=True)

        # 检查原始模型和我们的模型的骨干特征是否相似
        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=3e-3
            ), "The backbone features are not the same."

        # 提取原始模型的语义分割头部解码器特征
        mask_features, _, multi_scale_features, _, _ = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        # 收集所有的原始像素解码器特征
        original_pixel_decoder_features = []
        original_pixel_decoder_features.append(mask_features)
        for i in range(len(multi_scale_features)):
            original_pixel_decoder_features.append(multi_scale_features[i])

        # 检查原始模型和我们的模型的像素解码器特征是否相似
        for original_model_feature, our_model_feature in zip(
            original_pixel_decoder_features, our_model_output.pixel_decoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=3e-4
            ), "The pixel decoder feature are not the same"

        # 定义完整的图像转换操作序列
        tr_complete = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ],
        )

        # 对图像进行完整的预处理并转换为整型张量
        y = (tr_complete(im) * 255.0).to(torch.int).float()

        # 测试完整模型的输出
        original_model_out = original_model([{"image": y.clone(), "task": "The task is semantic"}])

        # 提取原始模型的语义分割结果
        original_segmentation = original_model_out[0]["sem_seg"]

        # 使用我们的模型进行推理，并对语义分割结果进行后处理
        our_model_out: OneFormerForUniversalSegmentationOutput = our_model(
            x.clone(), task_token, output_hidden_states=True
        )

        our_segmentation = post_process_sem_seg_output(our_model_out, target_size=(640, 640))[0]

        # 检查原始模型和我们的模型的语义分割结果是否相似
        assert torch.allclose(
            original_segmentation, our_segmentation, atol=1e-3
        ), "The segmentation image is not the same."

        # 记录测试通过的消息
        logger.info("✅ Test passed!")
def get_name(checkpoint_file: Path):
    # 从文件路径中获取模型名称（不含扩展名）
    model_name_raw: str = checkpoint_file.stem

    # 根据模型名称判断使用的骨干网络（backbone）
    backbone = "swin" if "swin" in model_name_raw else "dinat"

    # 初始化数据集名称为空字符串
    dataset = ""
    
    # 根据模型名称确定数据集类型
    if "coco" in model_name_raw:
        dataset = "coco"
    elif "ade20k" in model_name_raw:
        dataset = "ade20k"
    elif "cityscapes" in model_name_raw:
        dataset = "cityscapes"
    else:
        # 如果模型名称不包含预期的数据集类型，则抛出值错误异常
        raise ValueError(
            f"{model_name_raw} must be wrong since we didn't find 'coco' or 'ade20k' or 'cityscapes' in it "
        )

    # 支持的骨干网络类型列表
    backbone_types = ["tiny", "large"]

    # 使用过滤器找到模型名称中包含的骨干网络类型
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0]

    # 构建最终的模型名称
    model_name = f"oneformer_{dataset}_{backbone}_{backbone_type}"

    return model_name


if __name__ == "__main__":
    # 创建参数解析器，描述用途是转换原始 OneFormer 模型（使用 swin 骨干网络）为 Transformers 实现的命令行工具
    parser = ArgumentParser(
        description=(
            "Command line to convert the original oneformer models (with swin backbone) to transformers"
            " implementation."
        )
    )

    # 添加命令行参数：模型检查点目录的路径
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.pth; where <CONFIG_NAME> name must follow the"
            " following nomenclature: oneformer_<DATASET_NAME>_<BACKBONE>_<BACKBONE_TYPE>"
        ),
    )
    
    # 添加命令行参数：模型配置文件目录的路径
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.yaml; where <CONFIG_NAME> name must follow the"
            " following nomenclature: oneformer_<DATASET_NAME>_<BACKBONE>_<BACKBONE_TYPE>"
        ),
    )
    
    # 添加命令行参数：输出 PyTorch 模型的文件夹路径（必需参数）
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
    
    # 添加命令行参数：原始 OneFormer 实现目录的路径（必需参数）
    parser.add_argument(
        "--oneformer_dir",
        required=True,
        type=Path,
        help=(
            "A path to OneFormer's original implementation directory. You can download from here: "
            "https://github.com/SHI-Labs/OneFormer"
        ),
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 初始化各参数为对应的路径对象
    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.pytorch_dump_folder_path
    oneformer_dir: Path = args.oneformer_dir

    # 如果输出路径不存在，则创建
    if not save_directory.exists():
        save_directory.mkdir(parents=True)
    # 遍历 OriginalOneFormerCheckpointToOursConverter 类的 using_dirs 方法返回的迭代器，
    # 该方法根据给定的 checkpoints_dir 和 config_dir 返回配置文件和检查点文件的元组
    for config_file, checkpoint_file in OriginalOneFormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        # 创建 OriginalOneFormerConfigToProcessorConverter 的实例，将配置文件转换为处理器对象
        processor = OriginalOneFormerConfigToProcessorConverter()(
            setup_cfg(Args(config_file=config_file)), os.path.join("shi-labs", config_file.stem)
        )

        # 根据配置文件创建原始配置对象
        original_config = setup_cfg(Args(config_file=config_file))

        # 根据原始配置对象获取 OneFormer 模型的关键字参数
        oneformer_kwargs = OriginalOneFormer.from_config(original_config)

        # 创建原始的 OneFormer 模型，并设置为评估模式
        original_model = OriginalOneFormer(**oneformer_kwargs).eval()

        # 加载检查点文件到原始模型中
        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        # 检查 config_file.stem 是否包含 "swin"，用于判断是否为 Swin 模型
        is_swin = "swin" in config_file.stem

        # 使用 OriginalOneFormerConfigToOursConverter 将原始配置转换为我们的配置对象
        config: OneFormerConfig = OriginalOneFormerConfigToOursConverter()(original_config, is_swin)

        # 创建 OneFormerModel 对象，并设置为评估模式
        oneformer = OneFormerModel(config=config).eval()

        # 使用 OriginalOneFormerCheckpointToOursConverter 将原始模型和配置转换为我们的 OneFormer 模型
        converter = OriginalOneFormerCheckpointToOursConverter(original_model, config)
        oneformer = converter.convert(oneformer, is_swin)

        # 创建用于通用分割的 OneFormerForUniversalSegmentation 对象，并设置为评估模式
        oneformer_for_universal_segmentation = OneFormerForUniversalSegmentation(config=config).eval()

        # 将转换后的 OneFormer 模型设置为通用分割模型的属性
        oneformer_for_universal_segmentation.model = oneformer

        # 执行测试函数，测试原始模型和转换后的通用分割模型在处理器和路径下的表现
        test(
            original_model,
            oneformer_for_universal_segmentation,
            processor,
            os.path.join("shi-labs", config_file.stem),
        )

        # 获取模型名称，用于保存和日志记录
        model_name = get_name(checkpoint_file)

        # 记录信息，表明正在保存模型
        logger.info(f"🪄 Saving {model_name}")

        # 将处理器和通用分割模型保存到指定的目录下
        processor.save_pretrained(save_directory / model_name)
        oneformer_for_universal_segmentation.save_pretrained(save_directory / model_name)

        # 将处理器和通用分割模型推送到指定的 Hub 仓库
        processor.push_to_hub(
            repo_id=os.path.join("shi-labs", config_file.stem),
            commit_message="Add configs",
            use_temp_dir=True,
        )
        oneformer_for_universal_segmentation.push_to_hub(
            repo_id=os.path.join("shi-labs", config_file.stem),
            commit_message="Add model",
            use_temp_dir=True,
        )
```