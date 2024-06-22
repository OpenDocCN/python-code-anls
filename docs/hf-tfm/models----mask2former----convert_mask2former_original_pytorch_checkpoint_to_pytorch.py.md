# `.\transformers\models\mask2former\convert_mask2former_original_pytorch_checkpoint_to_pytorch.py`

```
# 导入必要的模块和类
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

# 定义一些类型别名
StateDict = Dict[str, Tensor]

# 设置日志输出级别为info
logging.set_verbosity_info()
logger = logging.get_logger()

# 设置随机种子
torch.manual_seed(0)

# 定义一个TrackedStateDict类，用于跟踪字典的访问情况
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

# 定义一个prepare_img函数，用于准备一张猫咪图像
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    img_data = requests.get(url, stream=True).raw
    im = Image.open(img_data)
    return im

# 定义一个Args类，用于存储命令行参数
@dataclass
class Args:
    """Fake command line arguments needed by mask2former/detectron implementation"""
    config_file: str
def setup_cfg(args: Args):
    # 从文件和命令行参数加载配置
    cfg = get_cfg()
    add_deeplab_config(cfg)  # 添加 DeepLab 配置
    add_maskformer2_config(cfg)  # 添加 Maskformer2 配置
    cfg.merge_from_file(args.config_file)  # 从文件中合并配置
    cfg.freeze()  # 冻结配置，防止修改
    return cfg


class OriginalMask2FormerConfigToOursConverter:
    # 将原始 Mask2Former 配置转换为我们的配置
    class OriginalMask2FormerConfigToImageProcessorConverter:
        def __call__(self, original_config: object) -> Mask2FormerImageProcessor:
            # 提取模型和输入配置
            model = original_config.MODEL
            model_input = original_config.INPUT

            return Mask2FormerImageProcessor(
                # 设置图像均值为模型像素均值的标准化列表
                image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),
                # 设置图像标准差为模型像素标准差的标准化列表
                image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),
                # 设置测试时的最小尺寸
                size=model_input.MIN_SIZE_TEST,
                # 设置测试时的最大尺寸
                max_size=model_input.MAX_SIZE_TEST,
                # 设置类别数目
                num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,
                # 设置忽略值
                ignore_index=model.SEM_SEG_HEAD.IGNORE_VALUE,
                # 设置尺寸可分割性
                size_divisibility=32,
            )


class OriginalMask2FormerCheckpointToOursConverter:
    def __init__(self, original_model: nn.Module, config: Mask2FormerConfig):
        self.original_model = original_model
        self.config = config

    # 将所有键从源状态字典中弹出并插入目标状态字典中，同时进行重命名
    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)

    # 替换 Maskformer Swin 骨干部分
    # 替换 Transformer 解码器部分
    def replace_masked_attention_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        dst_prefix: str = "transformer_module.decoder"
        src_prefix: str = "sem_seg_head.predictor"

        renamed_keys = self.rename_keys_in_masked_attention_decoder(dst_state_dict, src_state_dict)

        # 添加更多键值对
        renamed_keys.extend(
            [
                # 替换归一化层权重
                (f"{src_prefix}.decoder_norm.weight", f"{dst_prefix}.layernorm.weight"),
                # 替换归一化层偏置
                (f"{src_prefix}.decoder_norm.bias", f"{dst_prefix}.layernorm.bias"),
            ]
        )

        mlp_len = 3
        for i in range(mlp_len):
            renamed_keys.extend(
                [
                    # 替换掩码嵌入层权重
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.weight",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.weight",
                    ),
                    # 替换掩码嵌入层偏置
                    (
                        f"{src_prefix}.mask_embed.layers.{i}.bias",
                        f"{dst_prefix}.mask_predictor.mask_embedder.{i}.0.bias",
                    ),
                ]
            )

        # 弹出源状态字典中的键并插入目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 用于将源状态字典中的解码器自注意力层的权重和偏置替换为目标状态字典中的对应项
    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典中解码器自注意力层的前缀
        dst_prefix: str = "transformer_module.decoder.layers"
        # 源状态字典中语义分割头预测器的前缀
        src_prefix: str = "sem_seg_head.predictor"
        # 对解码器的每一层进行迭代
        for i in range(self.config.decoder_layers - 1):
            # 从源状态字典中弹出解码器自注意力层输入投影层的权重和偏置
            in_proj_weight = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight"
            )
            in_proj_bias = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias"
            )
            # 接下来，按顺序添加查询、键和值到状态字典
            # 查询权重
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            # 查询偏置
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
            # 键权重
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            # 键偏置
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            # 值权重
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            # 值偏置
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 用于替换状态字典中的转换器模块
    def replace_transformer_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典中转换器模块的前缀
        dst_prefix: str = "transformer_module"
        # 源状态字典中语义分割头预测器的前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 替换掩码注意力解码器
        self.replace_masked_attention_decoder(dst_state_dict, src_state_dict)

        # 重命名的键对
        renamed_keys = [
            (f"{src_prefix}.query_embed.weight", f"{dst_prefix}.queries_embedder.weight"),
            (f"{src_prefix}.query_feat.weight", f"{dst_prefix}.queries_features.weight"),
            (f"{src_prefix}.level_embed.weight", f"{dst_prefix}.level_embed.weight"),
        ]

        # 从状态字典中弹出所有的键对，并替换相关键
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
        # 替换解码器自注意力层的键
        self.replace_keys_qkv_transformer_decoder(dst_state_dict, src_state_dict)

    # 用于替换通用分割模块
    def replace_universal_segmentation_module(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的前缀
        dst_prefix: str = ""
        # 源状态字典中语义分割头预测器的前缀
        src_prefix: str = "sem_seg_head.predictor"

        # 重命名的键对
        renamed_keys = [
            (f"{src_prefix}.class_embed.weight", f"{dst_prefix}class_predictor.weight"),
            (f"{src_prefix}.class_embed.bias", f"{dst_prefix}class_predictor.bias"),
        ]

        # 记录日志，显示将要替换的键对
        logger.info(f"Replacing keys {pformat(renamed_keys)}")
        # 从状态字典中弹出所有的键对，并替换相关键
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 将输入的 Mask2FormerModel 转换为 Mask2FormerModel 类型
    def convert(self, mask2former: Mask2FormerModel) -> Mask2FormerModel:
        # 创建目标状态字典并拷贝输入模型的状态字典
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()
    
        # 替换像素模块
        self.replace_pixel_module(dst_state_dict, src_state_dict)
        # 替换变换器模块
        self.replace_transformer_module(dst_state_dict, src_state_dict)
    
        # 打印缺失的键值对
        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        # 打印未拷贝的键值
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        # 输出完成信息
        logger.info("🙌 Done")
    
        # 根据需要追踪的键值对创建状态字典
        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        # 加载新的状态字典到模型
        mask2former.load_state_dict(state_dict)
        return mask2former
    
    # 将输入的 Mask2FormerForUniversalSegmentation 转换为 Mask2FormerForUniversalSegmentation 类型
    def convert_universal_segmentation(
        self, mask2former: Mask2FormerForUniversalSegmentation
    ) -> Mask2FormerForUniversalSegmentation:
        # 创建目标状态字典并拷贝输入模型的状态字典
        dst_state_dict = TrackedStateDict(mask2former.state_dict())
        src_state_dict = self.original_model.state_dict()
    
        # 替换通用分割模块
        self.replace_universal_segmentation_module(dst_state_dict, src_state_dict)
    
        # 根据需要追踪的键值对创建状态字典
        state_dict = {key: dst_state_dict[key] for key in dst_state_dict.to_track.keys()}
        # 加载新的状态字典到模型
        mask2former.load_state_dict(state_dict)
    
        return mask2former
    
    # 静态方法，根据检查点和配置目录生成路径信息的迭代器
    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        # 获取所有检查点文件的路径列表
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pkl")
    
        # 遍历每个检查点文件
        for checkpoint in checkpoints:
            logger.info(f"💪 Converting {checkpoint.stem}")
            # 查找关联的配置文件
    
            # 数据集名称，例如 'coco'
            dataset_name = checkpoint.parents[2].stem
            if dataset_name == "ade":
                dataset_name = dataset_name.replace("ade", "ade20k")
    
            # 任务类型，例如 'instance-segmentation'
            segmentation_task = checkpoint.parents[1].stem
    
            # 与检查点对应的配置文件名
            config_file_name = f"{checkpoint.parents[0].stem}.yaml"
    
            # 配置文件路径
            config: Path = config_dir / dataset_name / segmentation_task / "swin" / config_file_name
            yield config, checkpoint
# 测试两个模型是否在给定容差下输出相同结果
def test(
    original_model,  # 原始模型
    our_model: Mask2FormerForUniversalSegmentation,  # 我们的模型
    image_processor: Mask2FormerImageProcessor,  # 图像处理器
    tolerance: float,  # 容差值
):
    # 禁用梯度计算
    with torch.no_grad():
        # 将原始模型和我们的模型设置为评估模式
        original_model = original_model.eval()
        our_model = our_model.eval()

        # 准备图像数据
        im = prepare_img()
        x = image_processor(images=im, return_tensors="pt")["pixel_values"]

        # 获取原始模型的主干特征
        original_model_backbone_features = original_model.backbone(x.clone())
        # 获取我们的模型的输出，包括隐藏状态
        our_model_output: Mask2FormerModelOutput = our_model.model(x.clone(), output_hidden_states=True)

        # 测试主干
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
        y = (tr_complete(im) * 255.0).to(torch.int).float()

        # 修改原始 Mask2Former 代码以返回掩码和类别 logits
        original_class_logits, original_mask_logits = original_model([{"image": y.clone().squeeze(0)}])

        # 获取我们模型的输出
        our_model_out: Mask2FormerForUniversalSegmentationOutput = our_model(x.clone())
        our_mask_logits = our_model_out.masks_queries_logits
        our_class_logits = our_model_out.class_queries_logits

        # 断言原始模型和我们的模型输出形状相同
        assert original_mask_logits.shape == our_mask_logits.shape, "Output masks shapes are not matching."
        assert original_class_logits.shape == our_class_logits.shape, "Output class logits shapes are not matching."
        # 断言类别 logits 和预测的掩码相同
        assert torch.allclose(
            original_class_logits, our_class_logits, atol=tolerance
        ), "The class logits are not the same."
        assert torch.allclose(
            original_mask_logits, our_mask_logits, atol=tolerance
        ), "The predicted masks are not the same."

        # 记录测试通过信息
        logger.info("✅ Test passed!")


# 从检查点文件中获取模型名称
def get_model_name(checkpoint_file: Path):
    # model_name_raw 是形如 maskformer2_swin_small_bs16_50ep 的字符串
    model_name_raw: str = checkpoint_file.parents[0].stem

    # `segmentation_task_type` 必须是以下之一: `instance-segmentation`, `panoptic-segmentation`, `semantic-segmentation`
    segmentation_task_name: str = checkpoint_file.parents[1].stem
    # 检查segmentation_task_name是否在指定的三种分割任务名称之一，否则抛出数值错误异常
    if segmentation_task_name not in ["instance-segmentation", "panoptic-segmentation", "semantic-segmentation"]:
        raise ValueError(
            f"{segmentation_task_name} must be wrong since acceptable values are: instance-segmentation,"
            " panoptic-segmentation, semantic-segmentation."
        )

    # 从checkpoint_file父目录的父目录中获取数据集名称，必须是"coco", "ade", "cityscapes", "mapillary-vistas"之一，否则抛出数值错误异常
    dataset_name: str = checkpoint_file.parents[2].stem
    if dataset_name not in ["coco", "ade", "cityscapes", "mapillary-vistas"]:
        raise ValueError(
            f"{dataset_name} must be wrong since we didn't find 'coco' or 'ade' or 'cityscapes' or 'mapillary-vistas'"
            " in it "
        )

    # 设置backbone为"swin"，定义backbone_types列表和当前模型的backbone类型
    backbone = "swin"
    backbone_types = ["tiny", "small", "base_IN21k", "base", "large"]
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0].replace("_", "-")

    # 根据backbone、backbone_type、dataset_name和segmentation_task_name组合成模型名称
    model_name = f"mask2former-{backbone}-{backbone_type}-{dataset_name}-{segmentation_task_name.split('-')[0]}"

    # 返回构建好的模型名称
    return model_name
# 当该脚本作为主程序运行时执行以下代码
if __name__ == "__main__":
    # 创建一个ArgumentParser对象，用于解析命令行参数
    parser = ArgumentParser(
        # 设置程序的描述信息
        description="Command line to convert the original mask2formers (with swin backbone) to our implementations."
    )

    # 添加一个参数，指定包含模型checkpoint的目录
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " <DIR_NAME>/<DATASET_NAME>/<SEGMENTATION_TASK_NAME>/<CONFIG_NAME>.pkl"
        ),
    )
    # 添加一个参数，指定包含模型配置文件的目录
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<SEGMENTATION_TASK_NAME>/<CONFIG_NAME>.yaml"
        ),
    )
    # 添加一个必需参数，指定Mask2Former的原始实现目录
    parser.add_argument(
        "--mask2former_dir",
        required=True,
        type=Path,
        help=(
            "A path to Mask2Former's original implementation directory. You can download from here:"
            " https://github.com/facebookresearch/Mask2Former"
        ),
    )

    # 解析命令行参数，获取结果
    args = parser.parse_args()

    # 从解析的参数中获取各个目录的路径
    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    mask2former_dir: Path = args.mask2former_dir
    # 将Mask2Former原始实现目录的父目录添加到系统路径中
    sys.path.append(str(mask2former_dir.parent))
    # 从Mask2Former的原始源代码中导入配置和模型类
    from Mask2Former.mask2former.config import add_maskformer2_config
    from Mask2Former.mask2former.maskformer_model import MaskFormer as OriginalMask2Former

    # 遍历checkpoints_dir和config_dir中的文件，并转换为我们自己的实现
    for config_file, checkpoint_file in OriginalMask2FormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        # 获取模型名称
        model_name = get_model_name(checkpoint_file)
        # 创建图像处理器对象，并将原始的配置文件转换成图像处理器的配置
        image_processor = OriginalMask2FormerConfigToImageProcessorConverter()(
            setup_cfg(Args(config_file=config_file))
        )
        # 设置图像处理器的尺寸为384x384
        image_processor.size = {"height": 384, "width": 384}

        # 根据配置文件创建原始的Mask2Former模型
        original_config = setup_cfg(Args(config_file=config_file))
        mask2former_kwargs = OriginalMask2Former.from_config(original_config)
        original_model = OriginalMask2Former(**mask2former_kwargs).eval()

        # 加载checkpoint文件中的模型参数到原始模型中
        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        # 将原始模型的配置转换成我们的配置
        config: Mask2FormerConfig = OriginalMask2FormerConfigToOursConverter()(original_config)
        # 创建我们的Mask2Former模型
        mask2former = Mask2FormerModel(config=config).eval()

        # 将原始模型的参数转换成我们的模型
        converter = OriginalMask2FormerCheckpointToOursConverter(original_model, config)
        mask2former = converter.convert(mask2former)

        # 创建用于通用分割的Mask2Former模型
        mask2former_for_segmentation = Mask2FormerForUniversalSegmentation(config=config).eval()
        # 将我们的模型设置为通用分割模型的子模型
        mask2former_for_segmentation.model = mask2former

        # 将通用分割模型的参数转换成我们的模型的参数
        mask2former_for_segmentation = converter.convert_universal_segmentation(mask2former_for_segmentation)

        # 设置容差值
        tolerance = 3e-1
        # 高容差的模型列表
        high_tolerance_models = [
            "mask2former-swin-base-IN21k-coco-instance",
            "mask2former-swin-base-coco-instance",
            "mask2former-swin-small-cityscapes-semantic",
        ]

        if model_name in high_tolerance_models:
            # 如果模型在高容差模型列表中，则将容差值设置为3e-1
            tolerance = 3e-1

        # 记录日志，测试模型
        logger.info(f"🪄 Testing {model_name}...")
        test(original_model, mask2former_for_segmentation, image_processor, tolerance)
        # 记录日志，将模型推送到hub
        logger.info(f"🪄 Pushing {model_name} to hub...")

        # 将图像处理器对象上传到hub
        image_processor.push_to_hub(model_name)
        # 将通用分割模型上传到hub
        mask2former_for_segmentation.push_to_hub(model_name)
```