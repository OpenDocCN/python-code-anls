# `.\transformers\models\maskformer\convert_maskformer_original_pytorch_checkpoint_to_pytorch.py`

```
# 设置 Python 文件的编码为 UTF-8
# 版权声明：该代码版权归 Meta Platforms, Inc. 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获得许可证副本：
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件根据“原样”分发，
# 没有任何担保或条件，无论是明示的还是暗示的
# 有关许可证的详细信息，请参阅许可证
# 导入 sys 模块，用于与 Python 解释器进行交互
import sys
# 从 argparse 模块导入 ArgumentParser 类，用于解析命令行参数
from argparse import ArgumentParser
# 从 dataclasses 模块导入 dataclass 装饰器，用于创建不可变数据类
from dataclasses import dataclass
# 从 pathlib 模块导入 Path 类，用于处理文件路径
from pathlib import Path
# 从 pprint 模块导入 pformat 函数，用于格式化输出 Python 对象
from pprint import pformat
# 从 typing 模块导入 Any、Dict、Iterator、List、Set、Tuple 等类型
from typing import Any, Dict, Iterator, List, Set, Tuple
# 导入 requests 模块，用于发送 HTTP 请求
import requests
# 导入 torch 模块，用于构建深度学习模型
import torch
# 从 torchvision.transforms 模块导入 T 模块，用于图像转换
import torchvision.transforms as T
# 从 detectron2.checkpoint 模块导入 DetectionCheckpointer 类，用于加载检查点
from detectron2.checkpoint import DetectionCheckpointer
# 从 detectron2.config 模块导入 get_cfg 函数，用于获取配置对象
from detectron2.config import get_cfg
# 从 detectron2.data 模块导入 MetadataCatalog 类，用于管理元数据
from detectron2.data import MetadataCatalog
# 从 detectron2.projects.deeplab 模块导入 add_deeplab_config 函数，用于添加 DeepLab 配置
from detectron2.projects.deeplab import add_deeplab_config
# 从 PIL 模块导入 Image 类，用于图像处理
from PIL import Image
# 从 torch 模块导入 Tensor、nn 等类，用于构建神经网络
from torch import Tensor, nn
# 从 transformers.models.maskformer.feature_extraction_maskformer 模块导入 MaskFormerImageProcessor 类，用于特征提取
from transformers.models.maskformer.feature_extraction_maskformer import MaskFormerImageProcessor
# 从 transformers.models.maskformer.modeling_maskformer 模块导入 MaskFormerConfig、MaskFormerForInstanceSegmentation、MaskFormerForInstanceSegmentationOutput、MaskFormerModel、MaskFormerModelOutput 类，用于 MaskFormer 模型
from transformers.models.maskformer.modeling_maskformer import (
    MaskFormerConfig,
    MaskFormerForInstanceSegmentation,
    MaskFormerForInstanceSegmentationOutput,
    MaskFormerModel,
    MaskFormerModelOutput,
)
# 从 transformers.utils 模块导入 logging 函数，用于记录日志
from transformers.utils import logging

# 定义 StateDict 类型
StateDict = Dict[str, Tensor]

# 设置日志记录级别为信息
logging.set_verbosity_info()
# 获取记录器对象
logger = logging.get_logger()
# 设置随机种子为 0
torch.manual_seed(0)


# 定义 TrackedStateDict 类
class TrackedStateDict:
    def __init__(self, to_track: Dict):
        """This class "tracks" a python dictionary by keeping track of which item is accessed.

        Args:
            to_track (Dict): The dictionary we wish to track
        """
        # 初始化对象属性
        self.to_track = to_track
        self._seen: Set[str] = set()

    # 获取字典中指定键的值
    def __getitem__(self, key: str) -> Any:
        return self.to_track[key]

    # 设置字典中指定键的值
    def __setitem__(self, key: str, item: Any):
        self._seen.add(key)
        self.to_track[key] = item

    # 返回字典中未被访问的键的列表
    def diff(self) -> List[str]:
        """This method returns a set difference between the keys in the tracked state dict and the one we have access so far.
        This is an effective method to check if we have update all the keys

        Returns:
            List[str]: List of keys not yet updated
        """
        return set(self.to_track.keys()) - self._seen

    # 返回字典的副本
    def copy(self) -> Dict:
        # 通过调用内部字典的 copy 方法来获取副本
        return self.to_track.copy()


# 定义一个函数，用于准备图像数据
def prepare_img():
    # 图片 URL 地址
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 发送 HTTP 请求获取图像数据
    img_data = requests.get(url, stream=True).raw
    # 打开图像
    im = Image.open(img_data)
    # 返回图像对象
    return im


# 定义一个命令行参数类
@dataclass
class Args:
    """Fake command line arguments needed by maskformer/detectron implementation"""

    # 配置文件路径
    config_file: str
# 定义一个函数用于设置配置参数，接受一个参数对象 Args
def setup_cfg(args: Args):
    # 获取一个空的配置对象
    cfg = get_cfg()
    # 添加 DeepLab 的配置到 cfg
    add_deeplab_config(cfg)
    # 添加 MaskFormer 的配置到 cfg
    add_mask_former_config(cfg)
    # 从文件中加载配置，并合并命令行参数
    cfg.merge_from_file(args.config_file)
    # 冻结配置，使其不可更改
    cfg.freeze()
    # 返回配置对象
    return cfg


# 定义一个类，用于将原始的 MaskFormer 配置转换为我们所需的配置
class OriginalMaskFormerConfigToOursConverter:
    # 实现类的调用方法，接受一个原始配置对象，返回我们需要的 MaskFormerConfig 对象
    def __call__(self, original_config: object) -> MaskFormerConfig:
        # 从原始配置中获取模型对象
        model = original_config.MODEL
        # 从模型对象中获取 MaskFormer 对象
        mask_former = model.MASK_FORMER
        # 从模型对象中获取 Swin Transformer 对象
        swin = model.SWIN

        # 从元数据目录中获取数据集类别信息
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])
        # 构建类别 id 到标签名称的字典
        id2label = dict(enumerate(dataset_catalog.stuff_classes))
        # 构建标签名称到类别 id 的字典
        label2id = {label: idx for idx, label in id2label.items()}

        # 构建 MaskFormerConfig 对象
        config: MaskFormerConfig = MaskFormerConfig(
            # FPN 特征的维度大小
            fpn_feature_size=model.SEM_SEG_HEAD.CONVS_DIM,
            # Mask 特征的维度大小
            mask_feature_size=model.SEM_SEG_HEAD.MASK_DIM,
            # 类别数量
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            # 无目标权重
            no_object_weight=mask_former.NO_OBJECT_WEIGHT,
            # 查询对象数量
            num_queries=mask_former.NUM_OBJECT_QUERIES,
            # 骨干网络配置
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
            # Dice 损失权重
            dice_weight=mask_former.DICE_WEIGHT,
            # 交叉熵损失权重
            ce_weight=1.0,
            # Mask 损失权重
            mask_weight=mask_former.MASK_WEIGHT,
            # 解码器配置
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
                # 默认预训练配置值
            },
            # 类别 id 到标签名称的映射字典
            id2label=id2label,
            # 标签名称到类别 id 的映射字典
            label2id=label2id,
        )

        # 返回 MaskFormerConfig 对象
        return config


class OriginalMaskFormerConfigToImageProcessorConverter:
    # 待实现的类，用于将原始的 MaskFormer 配置转换为图像处理器的配置
```  
    # __call__ 方法是一个特殊的实例方法，允许类实例像函数一样被调用
    def __call__(self, original_config: object) -> MaskFormerImageProcessor:
        # 从 original_config 对象中获取 MODEL 配置
        model = original_config.MODEL
        # 从 original_config 对象中获取 INPUT 配置
        model_input = original_config.INPUT
        # 从 MetadataCatalog 获取测试数据集的相关信息
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST[0])
    
        # 返回一个 MaskFormerImageProcessor 的实例，并使用以下参数进行配置:
        return MaskFormerImageProcessor(
            # 将模型的像素均值从 0-255 的范围转换到 0-1 的范围
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            # 将模型的像素标准差从 0-255 的范围转换到 0-1 的范围
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            # 设置测试时的最小输入尺寸
            size=model_input.MIN_SIZE_TEST,
            # 设置测试时的最大输入尺寸
            max_size=model_input.MAX_SIZE_TEST,
            # 设置语义分割的类别数
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            # 设置需要忽略的标签
            ignore_index=dataset_catalog.ignore_label,
            # 设置输入尺寸的对齐值为 32，这是 Swin Transformer 模型的要求
            size_divisibility=32,
        )
class OriginalMaskFormerCheckpointToOursConverter:
    # 定义 OriginalMaskFormerCheckpointToOursConverter 类
    def __init__(self, original_model: nn.Module, config: MaskFormerConfig):
        # 初始化函数，接受原始模型和配置参数
        self.original_model = original_model
        # 将原始模型保存到实例变量中
        self.config = config
        # 将配置参数保存到实例变量中

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        # 定义 pop_all 方法，接受重命名键值对列表、目标状态字典和源状态字典作为参数
        for src_key, dst_key in renamed_keys:
            # 遍历重命名键值对列表
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)
            # 将源状态字典的键值对弹出并添加到目标状态字典中

    def replace_pixel_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 定义 replace_pixel_module 方法，接受目标状态字典和源状态字典作为参数
        dst_prefix: str = "pixel_level_module.decoder"
        # 设置目标状态字典前缀
        src_prefix: str = "sem_seg_head.pixel_decoder"
        # 设置源状态字典前缀

        self.replace_backbone(dst_state_dict, src_state_dict, self.config)
        # 调用 replace_backbone 方法，替换backbone部分的模型参数

        def rename_keys_for_conv(detectron_conv: str, mine_conv: str):
            # 定义内部函数 rename_keys_for_conv，接受Detectron和自定义卷积层前缀作为参数
            return [
                (f"{detectron_conv}.weight", f"{mine_conv}.0.weight"),
                # 返回一组重命名键值对
                (f"{detectron_conv}.norm.weight", f"{mine_conv}.1.weight"),
                (f"{detectron_conv}.norm.bias", f"{mine_conv}.1.bias"),
            ]

        renamed_keys = [
            (f"{src_prefix}.mask_features.weight", f"{dst_prefix}.mask_projection.weight"),
            # 定义一组重命名键值对
            (f"{src_prefix}.mask_features.bias", f"{dst_prefix}.mask_projection.bias"),
            # 定义一组重命名键值对
        ]

        renamed_keys.extend(rename_keys_for_conv(f"{src_prefix}.layer_4", f"{dst_prefix}.fpn.stem"))
        # 扩展重命名键值对列表，用于重命名convolution层

        for src_i, dst_i in zip(range(3, 0, -1), range(0, 3)):
            # 遍历循环，定义src_i和dst_i
            renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.adapter_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.proj")
            )
            # 扩展重命名键值对列表，用于重命名adapter和projection层
            renamed_keys.extend(
                rename_keys_for_conv(f"{src_prefix}.layer_{src_i}", f"{dst_prefix}.fpn.layers.{dst_i}.block")
            )
            # 扩展重命名键值对列表，用于重命名layer和block层

        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        # 调用pop_all方法，应用所有重命名键值对到目标状态字典和源状态字典
        # 重命名传入的状态字典中的键值对，将源状态字典中的指定前缀改为目标状态字典中的指定前缀
        def rename_keys_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
            # 设置目标状态字典的前缀
            dst_prefix: str = "transformer_module.decoder"
            # 设置源状态字典的前缀
            src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
            # not sure why we are not popping direcetly here!
            # 这里列出了所有需要重命名的键值对（原始名称在左边，我们的名称在右边）
            rename_keys = []
            # 根据解码器层数进行循环，获取各层的参数
            for i in range(self.config.decoder_config.decoder_layers):
                # 获取self-attention层中的参数，并修改名称
                rename_keys.append(
                    (
                        f"{src_prefix}.layers.{i}.self_attn.out_proj.weight",
                        f"{dst_prefix}.layers.{i}.self_attn.out_proj.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"{src_prefix}.layers.{i}.self_attn.out_proj.bias",
                        f"{dst_prefix}.layers.{i}.self_attn.out_proj.bias",
                    )
                )
                # 获取multi-head attention层中的参数，并修改名称
                rename_keys.append(
                    (
                        f"{src_prefix}.layers.{i}.multihead_attn.out_proj.weight",
                        f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"{src_prefix}.layers.{i}.multihead_attn.out_proj.bias",
                        f"{dst_prefix}.layers.{i}.encoder_attn.out_proj.bias",
                    )
                )
                # 获取线性层1的参数，并修改名称
                rename_keys.append((f"{src_prefix}.layers.{i}.linear1.weight", f"{dst_prefix}.layers.{i}.fc1.weight"))
                rename_keys.append((f"{src_prefix}.layers.{i}.linear1.bias", f"{dst_prefix}.layers.{i}.fc1.bias"))
                # 获取线性层2的参数，并修改名称
                rename_keys.append((f"{src_prefix}.layers.{i}.linear2.weight", f"{dst_prefix}.layers.{i}.fc2.weight"))
                rename_keys.append((f"{src_prefix}.layers.{i}.linear2.bias", f"{dst_prefix}.layers.{i}.fc2.bias"))
                # 获取layernorm层的参数，并修改名称
                rename_keys.append(
                    (f"{src_prefix}.layers.{i}.norm1.weight", f"{dst_prefix}.layers.{i}.self_attn_layer_norm.weight")
                )
                rename_keys.append(
                    (f"{src_prefix}.layers.{i}.norm1.bias", f"{dst_prefix}.layers.{i}.self_attn_layer_norm.bias")
                )
                rename_keys.append(
                    (f"{src_prefix}.layers.{i}.norm2.weight", f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.weight")
                )
                rename_keys.append(
                    (f"{src_prefix}.layers.{i}.norm2.bias", f"{dst_prefix}.layers.{i}.encoder_attn_layer_norm.bias")
                )
                rename_keys.append(
                    (f"{src_prefix}.layers.{i}.norm3.weight", f"{dst_prefix}.layers.{i}.final_layer_norm.weight")
                )
                rename_keys.append(
                    (f"{src_prefix}.layers.{i}.norm3.bias", f"{dst_prefix}.layers.{i}.final_layer_norm.bias")
                )

            # 返回重命名后的键值对列表
            return rename_keys
    # 替换 DETR 模型的解码器中的权重和偏置
    def replace_q_k_v_in_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 声明目标和源的状态字典前缀
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        # 遍历解码器层，根据层数替换权重和偏置
        for i in range(self.config.decoder_config.decoder_layers):
            # 读取自注意力机制的输入投影层的权重和偏置
            in_proj_weight = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_weight")
            in_proj_bias = src_state_dict.pop(f"{src_prefix}.layers.{i}.self_attn.in_proj_bias")
            # 将查询、键和值（按顺序）添加到目标状态字典中
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
            # 读取交叉注意力机制的输入投影层的权重和偏置
            in_proj_weight_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_weight")
            in_proj_bias_cross_attn = src_state_dict.pop(f"{src_prefix}.layers.{i}.multihead_attn.in_proj_bias")
            # 将查询、键和值（按顺序）添加到目标状态字典中
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
            dst_state_dict[f"{dst_prefix}.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]
    # 替换 DETR 解码器的权重
    def replace_detr_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标权重的前缀
        dst_prefix: str = "transformer_module.decoder"
        # 源权重的前缀
        src_prefix: str = "sem_seg_head.predictor.transformer.decoder"
        
        # 重命名 DETR 解码器中的权重
        renamed_keys = self.rename_keys_in_detr_decoder(dst_state_dict, src_state_dict)
        
        # 添加更多的映射关系
        renamed_keys.extend(
            [
                (f"{src_prefix}.norm.weight", f"{dst_prefix}.layernorm.weight"),
                (f"{src_prefix}.norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        # 移除所有指定的映射关系
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

        # 替换 DETR 解码器中的 Q、K、V
        self.replace_q_k_v_in_detr_decoder(dst_state_dict, src_state_dict)

    # 替换 Transformer 模块的权重
    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标权重的前缀
        dst_prefix: str = "transformer_module"
        # 源权重的前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 调用替换 DETR 解码器的函数
        self.replace_detr_decoder(dst_state_dict, src_state_dict)

        # 定义需要重命名的键值对
        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.input_proj.weight", f"{dst_prefix}.input_projection.weight"),
            (f"{src_prefix}.input_proj.bias", f"{dst_prefix}.input_projection.bias"),
        ]

        # 移除所有指定的映射关系
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # 替换实例分割模块的权重
    def replace_instance_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 在这种情况下我们没有前缀，所以后面的键中要去掉 "."
        dst_prefix: str = ""
        src_prefix: str = "sem_seg_head.predictor"

        # 定义需要重命名的键值对
        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    (f"{src_prefix}.mask_embed.layers.{i}.weight", f"{dst_prefix}mask_embedder.{i}.0.weight"),
                    (f"{src_prefix}.mask_embed.layers.{i}.bias", f"{dst_prefix}mask_embedder.{i}.0.bias"),
                ]
            )
        # 输出日志信��
        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        # 移除所有指定的映射关系
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # 转换模型
    def convert(self, mask_former: MaskFormerModel) -> MaskFormerModel:
        # 创建目标模型权重状态字典
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        # 获取原始模型的权重状态字典
        src_state_dict = self.original_model.state_dict()

        # 替换像素模块的权重
        self.replace_pixel_module(dst_state_dict, src_state_dict)
        # 替换 Transformer 模块的权重
        self.replace_transformer_module(dst_state_dict, src_state_dict)

        # 输出日志信息
        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        logger.info("🙌 Done")

        # 加载目标模型的权重
        mask_former.load_state_dict(dst_state_dict)

        return mask_former
    # 将实例分割模型转换为另一个实例分割模型
    def convert_instance_segmentation(
        self, mask_former: MaskFormerForInstanceSegmentation
    ) -> MaskFormerForInstanceSegmentation:
        # 创建目标状态字典，复制原始模型的状态字典
        dst_state_dict = TrackedStateDict(mask_former.state_dict())
        # 获取原始模型的状态字典
        src_state_dict = self.original_model.state_dict()

        # 替换实例分割模型的模块
        self.replace_instance_segmentation_module(dst_state_dict, src_state_dict)

        # 加载目标状态字典到实例分割模型
        mask_former.load_state_dict(dst_state_dict)

        # 返回转换后的实例分割模型
        return mask_former

    # 静态方法，用于遍历检查点目录和配置目录，返回迭代器
    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        # 获取检查点目录下所有.pkl文件的路径列表
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

        # 遍历检查点列表
        for checkpoint in checkpoints:
            # 打印信息，表示正在转换该检查点
            logger.info(f"💪 Converting {checkpoint.stem}")
            # 查找关联的配置文件
            config: Path = config_dir / checkpoint.parents[0].stem / "swin" / f"{checkpoint.stem}.yaml"

            # 返回配置文件路径和检查点路径的元组
            yield config, checkpoint
# 定义一个测试函数，用于测试原始模型和我们的模型
def test(original_model, our_model: MaskFormerForInstanceSegmentation, image_processor: MaskFormerImageProcessor):
    # 禁用梯度计算
    with torch.no_grad():
        # 将原始模型和我们的模型设置为评估模式
        original_model = original_model.eval()
        our_model = our_model.eval()

        # 准备图像数据
        im = prepare_img()

        # 图像预处理操作
        tr = T.Compose(
            [
                T.Resize((384, 384)),
                T.ToTensor(),
                T.Normalize(
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ],
        )

        # 对图像进行预处理操作
        x = tr(im).unsqueeze(0)

        # 获取原始模型的骨干特征
        original_model_backbone_features = original_model.backbone(x.clone())

        # 获取我们模型的输出，包括隐藏状态
        our_model_output: MaskFormerModelOutput = our_model.model(x.clone(), output_hidden_states=True)

        # 检查原始模型和我们模型的骨干特征是否相似
        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=1e-3
            ), "The backbone features are not the same."

        # 获取原始模型的像素输出
        original_model_pixel_out = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        # 检查原始模型和我们模型的像素输出是否相似
        assert torch.allclose(
            original_model_pixel_out[0], our_model_output.pixel_decoder_last_hidden_state, atol=1e-4
        ), "The pixel decoder feature are not the same"

        # 测试完整模型
        original_model_out = original_model([{"image": x.squeeze(0)}])

        # 获取原始模型的分割结果
        original_segmentation = original_model_out[0]["sem_seg"]

        # 获取我们模型的分割结果
        our_model_out: MaskFormerForInstanceSegmentationOutput = our_model(x)

        # 对我们模型的分割结果进行后处理
        our_segmentation = image_processor.post_process_segmentation(our_model_out, target_size=(384, 384))

        # 检查原始模型和我们模型的分割结果是否相似
        assert torch.allclose(
            original_segmentation, our_segmentation, atol=1e-3
        ), "The segmentation image is not the same."

        # 输出测试通过信息
        logger.info("✅ Test passed!")


# 获取模型名称函数
def get_name(checkpoint_file: Path):
    # 获取模型文件名
    model_name_raw: str = checkpoint_file.stem
    # 父目录名称
    parent_name: str = checkpoint_file.parents[0].stem
    backbone = "swin"
    dataset = ""
    # 根据父目录名称确定数据集类型
    if "coco" in parent_name:
        dataset = "coco"
    elif "ade" in parent_name:
        dataset = "ade"
    else:
        raise ValueError(f"{parent_name} must be wrong since we didn't find 'coco' or 'ade' in it ")

    # 定义骨干类型列表
    backbone_types = ["tiny", "small", "base", "large"]

    # 获取模型名称中的骨干类型
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0]

    # 组合模型名称
    model_name = f"maskformer-{backbone}-{backbone_type}-{dataset}"

    return model_name


# 主函数入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = ArgumentParser(
        description="Command line to convert the original maskformers (with swin backbone) to our implementations."
    )
    # 添加命令行参数，用于指定模型检查点的目录
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.pkl"
        ),
    )
    # 添加命令行参数，用于指定模型配置文件的目录
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.yaml"
        ),
    )
    # 添加命令行参数，用于指定输出 PyTorch 模型的文件夹路径
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
    # 添加命令行参数，用于指定 MaskFormer 原始实现的目录路径
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

    # 将命令行参数转换为 Path 类型
    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.pytorch_dump_folder_path
    maskformer_dir: Path = args.maskformer_dir
    # 将 MaskFormer 目录的父目录添加到系统路径中
    sys.path.append(str(maskformer_dir.parent))
    # 导入所需的模块
    from MaskFormer.mask_former import add_mask_former_config
    from MaskFormer.mask_former.mask_former_model import MaskFormer as OriginalMaskFormer

    # 如果输出目录不存在，则创建
    if not save_directory.exists():
        save_directory.mkdir(parents=True)

    # 遍历原始 MaskFormer 检查点和配置文件，转换为当前模型所需的格式
    for config_file, checkpoint_file in OriginalMaskFormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
        ):
        # 使用 OriginalMaskFormerConfigToImageProcessorConverter 转换器将配置文件转换为图像处理器
        image_processor = OriginalMaskFormerConfigToImageProcessorConverter()(setup_cfg(Args(config_file=config_file)))

        # 根据配置文件设置原始配置
        original_config = setup_cfg(Args(config_file=config_file))
        # 从原始配置中获取 MaskFormer 的参数
        mask_former_kwargs = OriginalMaskFormer.from_config(original_config)

        # 根据 MaskFormer 的参数创建原始 MaskFormer 模型
        original_model = OriginalMaskFormer(**mask_former_kwargs).eval()

        # 加载检查点文件到原始模型
        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        # 将原始配置转换为 MaskFormerConfig 类型的配置
        config: MaskFormerConfig = OriginalMaskFormerConfigToOursConverter()(original_config)

        # 根据配置创建 MaskFormerModel 模型
        mask_former = MaskFormerModel(config=config).eval()

        # 创建原始 MaskFormer 模型到我们的 MaskFormer 模型的转换器
        converter = OriginalMaskFormerCheckpointToOursConverter(original_model, config)

        # 将原始 MaskFormer 模型转换为我们的 MaskFormer 模型
        maskformer = converter.convert(mask_former)

        # 创建用于实例分割的 MaskFormerForInstanceSegmentation 模型
        mask_former_for_instance_segmentation = MaskFormerForInstanceSegmentation(config=config).eval()

        # 将 mask_former 设置为 mask_former_for_instance_segmentation 的模型
        mask_former_for_instance_segmentation.model = mask_former
        # 将 mask_former_for_instance_segmentation 转换为我们的实例分割模型
        mask_former_for_instance_segmentation = converter.convert_instance_segmentation(
            mask_former_for_instance_segmentation
        )

        # 测试原始模型和实例分割模型
        test(original_model, mask_former_for_instance_segmentation, image_processor)

        # 获取模型名称
        model_name = get_name(checkpoint_file)
        logger.info(f"🪄 Saving {model_name}")

        # 保存图像处理器和实例分割模型到指定目录
        image_processor.save_pretrained(save_directory / model_name)
        mask_former_for_instance_segmentation.save_pretrained(save_directory / model_name)

        # 将图像处理器推送到 Hub
        image_processor.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add model",
            use_temp_dir=True,
        )
        # 将实例分割模型推送到 Hub
        mask_former_for_instance_segmentation.push_to_hub(
            repo_path_or_name=save_directory / model_name,
            commit_message="Add model",
            use_temp_dir=True,
        )
```