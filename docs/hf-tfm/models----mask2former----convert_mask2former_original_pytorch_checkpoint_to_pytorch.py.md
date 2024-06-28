# `.\models\mask2former\convert_mask2former_original_pytorch_checkpoint_to_pytorch.py`

```
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

# 导入必要的库和模块
import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterator, List, Set, Tuple

import requests
import torch
import torchvision.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from huggingface_hub import hf_hub_download
from PIL import Image
from torch import Tensor, nn

# 导入 transformers 相关模块和类
from transformers import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerImageProcessor,
    Mask2FormerModel,
    SwinConfig,
)
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerForUniversalSegmentationOutput,
    Mask2FormerModelOutput,
)
from transformers.utils import logging

# 设置日志级别为 info
logging.set_verbosity_info()
# 获取日志记录器
logger = logging.get_logger()

# 设定随机数种子为 0
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


# 准备一个图片数据，用于后续验证结果
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # 通过 URL 获取图片数据流
    img_data = requests.get(url, stream=True).raw
    # 打开并返回图像对象
    im = Image.open(img_data)
    return im


@dataclass
class Args:
    """Fake command line arguments needed by mask2former/detectron implementation"""

    config_file: str
# 从参数 `args` 中获取配置，加载配置文件和命令行参数
def setup_cfg(args: Args):
    # 调用 `get_cfg()` 函数创建配置对象 `cfg`
    cfg = get_cfg()
    # 添加 DeepLab 相关的配置到 `cfg` 中
    add_deeplab_config(cfg)
    # 添加 MaskFormer2 相关的配置到 `cfg` 中
    add_maskformer2_config(cfg)
    # 从指定的配置文件 `args.config_file` 中合并配置到 `cfg` 中
    cfg.merge_from_file(args.config_file)
    # 冻结配置，防止修改
    cfg.freeze()
    # 返回配置对象 `cfg`
    return cfg


# 将原始 Mask2Former 配置转换为我们定义的 ImageProcessor
class OriginalMask2FormerConfigToOursConverter:
# 将原始 Mask2Former 配置转换为我们定义的 ImageProcessor
class OriginalMask2FormerConfigToImageProcessorConverter:
    # 将原始配置对象转换为 Mask2FormerImageProcessor 实例
    def __call__(self, original_config: object) -> Mask2FormerImageProcessor:
        # 获取原始配置中的模型和输入信息
        model = original_config.MODEL
        model_input = original_config.INPUT

        # 返回一个 Mask2FormerImageProcessor 实例，使用标准化后的像素均值和标准差，以及其他相关配置
        return Mask2FormerImageProcessor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
            size=model_input.MIN_SIZE_TEST,
            max_size=model_input.MAX_SIZE_TEST,
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
            ignore_index=model.SEM_SEG_HEAD.IGNORE_VALUE,
            size_divisibility=32,
        )


# 将原始 Mask2Former 检查点转换为我们定义的检查点
class OriginalMask2FormerCheckpointToOursConverter:
    # 初始化转换器，接收原始模型和配置
    def __init__(self, original_model: nn.Module, config: Mask2FormerConfig):
        self.original_model = original_model
        self.config = config

    # 从源状态字典中弹出所有指定的重命名键，将其添加到目标状态字典中
    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    # 替换 MaskFormer Swin Transformer 的骨干部分
    def replace_maskformer_swin_backbone(
        self, dst_state_dict: StateDict, src_state_dict: StateDict, config: Mask2FormerConfig
    ):
        # 声明目标前缀和源前缀
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"

        # 重命名键列表在 `dst_state_dict` 和 `src_state_dict` 之间进行转换
        renamed_keys = self.rename_keys_in_masked_attention_decoder(dst_state_dict, src_state_dict)

        # 添加更多的重命名键
        renamed_keys.extend(
            [
                (f"{src_prefix}.decoder_norm.weight", f"{dst_prefix}.layernorm.weight"),
                (f"{src_prefix}.decoder_norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        mlp_len = 3
        # 遍历 MLP 层，并添加相应的重命名键
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.weight",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight",
                    ),
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.bias",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias",
                    ),
                ]
            )

        # 弹出所有的重命名键，并添加到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 将 Transformer 解码器的自注意力层的权重和偏置从源状态字典中弹出并添加到目标状态字典中
    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典中的键前缀
        dst_prefix: str = "transformer_module.decoder.layers"
        # 源状态字典中的键前缀
        src_prefix: str = "sem_seg_head.predictor"
        
        # 遍历 Transformer 解码器的每一层
        for i in range(self.config.decoder_layers - 1):
            # 读取自注意力层的输入投影层的权重和偏置
            in_proj_weight = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight"
            )
            in_proj_bias = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias"
            )
            
            # 将查询、键和值的投影权重和偏置添加到目标状态字典中
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 替换 Transformer 模块的键名称，并将其从源状态字典移动到目标状态字典中
    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典中的键前缀
        dst_prefix: str = "transformer_module"
        # 源状态字典中的键前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 调用替换掩蔽注意力解码器的方法
        self.replace_masked_attention_decoder(dst_state_dict, src_state_dict)

        # 定义要重命名的键对
        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.query_feat.weight", f"{dst_prefix}.queries_features.weight"),
            (f"{src_prefix}.level_embed.weight", f"{dst_prefix}.level_embed.weight"),
        ]

        # 从源状态字典中移除所有重命名的键，并将它们添加到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        
        # 调用替换 Transformer 解码器中查询、键、值的投影权重和偏置的方法
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    # 替换通用分割模块的键名称，并将其从源状态字典移动到目标状态字典中
    def replace_universal_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典中的键前缀（空字符串表示直接替换）
        dst_prefix: str = ""
        # 源状态字典中的键前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 定义要重命名的键对
        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        # 记录日志，指示正在替换的键
        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        
        # 从源状态字典中移除所有重命名的键，并将它们添加到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 将传入的 mask2former 对象的状态字典转换为可追踪的状态字典对象
    dst_state_dict = TrackedStateDict(mask2former.state_dict())
    # 获取原始模型的状态字典
    src_state_dict = self.original_model.state_dict()

    # 替换目标模型中的像素模块
    self.replace_pixel_module(dst_state_dict, src_state_dict)
    # 替换目标模型中的 Transformer 模块
    self.replace_transformer_module(dst_state_dict, src_state_dict)

    # 记录并输出未复制成功的键的信息
    logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
    # 记录并输出未复制的键的信息
    logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
    # 输出转换完成的信息
    logger.info("🙌 Done")

    # 从追踪的状态字典中选取需要追踪的键，构成新的状态字典
    state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
    # 加载新的状态字典到 mask2former 对象中
    mask2former.load_state_dict(state_dict)
    # 返回更新后的 mask2former 对象
    return mask2former

def convert_universal_segmentation(
    self, mask2former: Mask2FormerForUniversalSegmentation
) -> Mask2FormerForUniversalSegmentation:
    # 将传入的 mask2former 对象的状态字典转换为可追踪的状态字典对象
    dst_state_dict = TrackedStateDict(mask2former.state_dict())
    # 获取原始模型的状态字典
    src_state_dict = self.original_model.state_dict()

    # 替换通用分割模块
    self.replace_universal_segmentation_module(dst_state_dict, src_state_dict)

    # 从追踪的状态字典中选取需要追踪的键，构成新的状态字典
    state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
    # 加载新的状态字典到 mask2former 对象中
    mask2former.load_state_dict(state_dict)

    # 返回更新后的 mask2former 对象
    return mask2former

@staticmethod
def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
    # 获取 checkpoints_dir 目录下所有后缀为 .pkl 的文件路径列表
    checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")

    # 遍历每个 checkpoint 路径
    for checkpoint in checkpoints:
        # 输出正在转换的信息及其文件名（不带后缀）
        logger.info(f"💪 Converting {checkpoint.stem}")

        # 查找关联的配置文件

        # 数据集名称，例如 'coco'
        dataset_name = checkpoint.parents[2].stem
        # 如果数据集名称为 "ade"，则替换为 "ade20k"
        if dataset_name == "ade":
            dataset_name = dataset_name.replace("ade", "ade20k")

        # 分割任务类型，例如 'instance-segmentation'
        segmentation_task = checkpoint.parents[1].stem

        # 与 checkpoint 相关联的配置文件名
        config_file_name = f"{checkpoint.parents[0].stem}.yaml"

        # 构建配置文件的完整路径
        config: Path = config_dir / dataset_name / segmentation_task / "swin" / config_file_name
        # 返回配置文件路径和相应的 checkpoint 路径的迭代器
        yield config, checkpoint
# 定义一个测试函数，用于比较原始模型和我们的模型的性能
def test(
    original_model,  # 原始模型
    our_model: Mask2FormerForUniversalSegmentation,  # 我们的模型，特定类型为 Mask2FormerForUniversalSegmentation
    image_processor: Mask2FormerImageProcessor,  # 图像处理器，用于准备图像数据
    tolerance: float,  # 容忍度，用于比较数值时的误差允许范围
):
    with torch.no_grad():  # 使用 torch.no_grad() 禁用梯度计算
        original_model = original_model.eval()  # 将原始模型设置为评估模式
        our_model = our_model.eval()  # 将我们的模型设置为评估模式

        im = prepare_img()  # 准备图像数据
        x = image_processor(images=im, return_tensors="pt")["pixel_values"]  # 使用图像处理器处理图像并返回像素值张量

        original_model_backbone_features = original_model.backbone(x.clone())  # 提取原始模型的骨干特征
        our_model_output: Mask2FormerModelOutput = our_model.model(x.clone(), output_hidden_states=True)  # 使用我们的模型，获取输出并包括隐藏状态

        # 测试骨干特征
        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=tolerance
            ), "The backbone features are not the same."

        # 测试像素解码器
        mask_features, _, multi_scale_features = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )

        for original_model_feature, our_model_feature in zip(
            multi_scale_features, our_model_output.pixel_decoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=tolerance
            ), "The pixel decoder feature are not the same"

        # 测试完整模型
        tr_complete = T.Compose(
            [T.Resize((384, 384)), T.ToTensor()],
        )
        y = (tr_complete(im) * 255.0).to(torch.int).float()  # 转换图像数据到指定类型和范围

        # 修改原始的 Mask2Former 代码以返回掩码和类别 logits
        original_class_logits, original_mask_logits = original_model([{"image": y.clone().squeeze(0)}])

        our_model_out: Mask2FormerForUniversalSegmentationOutput = our_model(x.clone())
        our_mask_logits = our_model_out.masks_queries_logits  # 获取我们模型的掩码 logits
        our_class_logits = our_model_out.class_queries_logits  # 获取我们模型的类别 logits

        assert original_mask_logits.shape == our_mask_logits.shape, "Output masks shapes are not matching."
        assert original_class_logits.shape == our_class_logits.shape, "Output class logits shapes are not matching."
        assert torch.allclose(
            original_class_logits, our_class_logits, atol=tolerance
        ), "The class logits are not the same."
        assert torch.allclose(
            original_mask_logits, our_mask_logits, atol=tolerance
        ), "The predicted masks are not the same."

        logger.info("✅ Test passed!")  # 记录测试通过信息


# 定义一个函数，用于从检查点文件路径中获取模型名称
def get_model_name(checkpoint_file: Path):
    # model_name_raw 是检查点文件路径的父目录名
    model_name_raw: str = checkpoint_file.parents[0].stem

    # segmentation_task_name 必须是以下之一：instance-segmentation、panoptic-segmentation、semantic-segmentation
    segmentation_task_name: str = checkpoint_file.parents[1].stem
    # 检查分割任务名称是否在预定义的列表中，如果不在则引发值错误异常
    if segmentation_task_name not in ["instance-segmentation", "panoptic-segmentation", "semantic-segmentation"]:
        raise ValueError(
            f"{segmentation_task_name} must be wrong since acceptable values are: instance-segmentation,"
            " panoptic-segmentation, semantic-segmentation."
        )

    # 提取数据集名称，应为以下之一：`coco`, `ade`, `cityscapes`, `mapillary-vistas`
    dataset_name: str = checkpoint_file.parents[2].stem
    if dataset_name not in ["coco", "ade", "cityscapes", "mapillary-vistas"]:
        raise ValueError(
            f"{dataset_name} must be wrong since we didn't find 'coco' or 'ade' or 'cityscapes' or 'mapillary-vistas'"
            " in it "
        )

    # 设置模型的骨干网络类型为 "swin"
    backbone = "swin"

    # 定义可接受的骨干网络类型列表
    backbone_types = ["tiny", "small", "base_IN21k", "base", "large"]

    # 从模型名称中筛选出存在于骨干网络类型列表中的类型，并用连字符替换下划线
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0].replace("_", "-")

    # 组装模型名称，格式为 "mask2former-{backbone}-{backbone_type}-{dataset_name}-{segmentation_task_name.split('-')[0]}"
    model_name = f"mask2former-{backbone}-{backbone_type}-{dataset_name}-{segmentation_task_name.split('-')[0]}"

    # 返回生成的模型名称
    return model_name
if __name__ == "__main__":
    # 创建命令行解析器对象，设置描述信息
    parser = ArgumentParser(
        description="Command line to convert the original mask2formers (with swin backbone) to our implementations."
    )

    # 添加命令行参数 --checkpoints_dir，类型为 Path，用于指定模型检查点所在的目录路径
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " <DIR_NAME>/<DATASET_NAME>/<SEGMENTATION_TASK_NAME>/<CONFIG_NAME>.pkl"
        ),
    )

    # 添加命令行参数 --configs_dir，类型为 Path，用于指定模型配置文件所在的目录路径
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<SEGMENTATION_TASK_NAME>/<CONFIG_NAME>.yaml"
        ),
    )

    # 添加命令行参数 --mask2former_dir，类型为 Path，必选参数，用于指定 Mask2Former 的原始实现代码所在的目录路径
    parser.add_argument(
        "--mask2former_dir",
        required=True,
        type=Path,
        help=(
            "A path to Mask2Former's original implementation directory. You can download from here:"
            " https://github.com/facebookresearch/Mask2Former"
        ),
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 将命令行参数赋值给相应变量
    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    mask2former_dir: Path = args.mask2former_dir

    # 将 Mask2Former 的父目录添加到系统路径中，以便导入原始 Mask2Former 的配置和模型
    sys.path.append(str(mask2former_dir.parent))

    # 从原始源代码仓库导入原始 Mask2Former 的配置和模型
    from Mask2Former.mask2former.config import add_maskformer2_config
    from Mask2Former.mask2former.maskformer_model import MaskFormer as OriginalMask2Former

    # 使用循环处理每对配置文件和检查点文件，转换成我们的实现格式
    for config_file, checkpoint_file in OriginalMask2FormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
        ):
            # 从检查点文件获取模型名称
            model_name = get_model_name(checkpoint_file)
            # 使用原始配置文件创建图像处理器，并进行设置
            image_processor = OriginalMask2FormerConfigToImageProcessorConverter()(
                setup_cfg(Args(config_file=config_file))
            )
            # 设置图像处理器的尺寸为固定值
            image_processor.size = {"height": 384, "width": 384}

            # 使用原始配置文件创建配置对象
            original_config = setup_cfg(Args(config_file=config_file))
            # 从原始配置创建 Mask2Former 模型的参数
            mask2former_kwargs = OriginalMask2Former.from_config(original_config)
            # 创建并初始化原始 Mask2Former 模型
            original_model = OriginalMask2Former(**mask2former_kwargs).eval()

            # 加载模型的检查点
            DetectionCheckpointer(original_model).load(str(checkpoint_file))

            # 将原始配置转换为我们的 Mask2Former 配置对象
            config: Mask2FormerConfig = OriginalMask2FormerConfigToOursConverter()(original_config)
            # 创建并初始化我们的 Mask2Former 模型
            mask2former = Mask2FormerModel(config=config).eval()

            # 将原始 Mask2Former 模型和配置转换为我们的模型和配置
            converter = OriginalMask2FormerCheckpointToOursConverter(original_model, config)
            mask2former = converter.convert(mask2former)

            # 创建用于通用分割的 Mask2FormerForUniversalSegmentation 模型并初始化
            mask2former_for_segmentation = Mask2FormerForUniversalSegmentation(config=config).eval()
            # 将我们的 Mask2Former 模型应用于通用分割模型
            mask2former_for_segmentation.model = mask2former

            # 将通用分割模型从原始格式转换为我们的格式
            mask2former_for_segmentation = converter.convert_universal_segmentation(mask2former_for_segmentation)

            # 设置容差阈值
            tolerance = 3e-1
            # 需要高容差的模型列表
            high_tolerance_models = [
                "mask2former-swin-base-IN21k-coco-instance",
                "mask2former-swin-base-coco-instance",
                "mask2former-swin-small-cityscapes-semantic",
            ]

            # 如果模型名称在高容差模型列表中，则设置更高的容差阈值
            if model_name in high_tolerance_models:
                tolerance = 3e-1

            # 记录当前正在测试的模型名称
            logger.info(f"🪄 Testing {model_name}...")
            # 执行测试，评估模型性能
            test(original_model, mask2former_for_segmentation, image_processor, tolerance)
            # 记录当前正在推送的模型名称
            logger.info(f"🪄 Pushing {model_name} to hub...")

            # 将图像处理器推送至模型中心
            image_processor.push_to_hub(model_name)
            # 将通用分割模型推送至模型中心
            mask2former_for_segmentation.push_to_hub(model_name)
```