# `.\transformers\models\oneformer\convert_to_hf_oneformer.py`

```
# 定义文件编码格式
# 版权声明
#
# 导入模块
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


# 尝试导入 detectron2 框架相关模块，如未安装则忽略
try:
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detectron2.projects.deeplab import add_deeplab_config
except ImportError:
    pass

# 导入一些来自 Hugging Face 的 transformer 模块
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


# 设定状态字典的类型
StateDict = Dict[str, Tensor]

# 设定日志级别
logging.set_verbosity_info()
# 初始化日志记录器
logger = logging.get_logger()

# 设定随机数种子
torch.manual_seed(0)


# 定义一个类用于跟踪状态字典
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
    # 定义一个URL变量，存储图片的网络地址
    url = "https://praeclarumjj3.github.io/files/coco.jpeg"
    # 通过请求获取图片数据，使用流模式
    img_data = requests.get(url, stream=True).raw
    # 通过获取的图片数据创建一个图像对象
    im = Image.open(img_data)
    # 返回图像对象
    return im
from dataclasses import dataclass  # 导入dataclass装饰器用于创建数据类
from detectron2.config import get_cfg  # 导入get_cfg函数用于获取Detectron2配置
from detectron2.modeling import MetadataCatalog  # 导入MetadataCatalog用于获取元数据
from typing import List, Tuple  # 导入类型提示用于函数参数和返回值的类型声明
import torch  # 导入torch用于深度学习模型操作
from transformers import CLIPTokenizer  # 导入CLIPTokenizer用于处理文本输入
from .oneformer_processor import OneFormerProcessor  # 导入OneFormerProcessor类
from .oneformer_image_processor import OneFormerImageProcessor  # 导入OneFormerImageProcessor类
from .utils import StateDict  # 导入StateDict类型提示用于模型权重字典操作

@dataclass
class Args:
    """模拟一个需要的命令行参数的数据类，用于传递配置文件路径"""

    config_file: str  # 配置文件路径字符串

def setup_cfg(args: Args):
    # 从文件和命令行参数加载配置
    cfg = get_cfg()  # 获取Detectron2配置对象
    # 添加DeepLab相关配置
    add_deeplab_config(cfg)
    # 添加通用配置
    add_common_config(cfg)
    # 添加OneFormer相关配置
    add_oneformer_config(cfg)
    # 添加Swin相关配置
    add_swin_config(cfg)
    # 添加Dinat相关配置
    add_dinat_config(cfg)
    cfg.merge_from_file(args.config_file)  # 从文件中加载配置覆盖默认配置
    cfg.freeze()  # 冻结配置，防止意外修改
    return cfg  # 返回配置对象

class OriginalOneFormerConfigToOursConverter:
    pass  # 原始配置到我们配置的转换器类，暂无实现

class OriginalOneFormerConfigToProcessorConverter:
    def __call__(self, original_config: object, model_repo: str) -> OneFormerProcessor:
        # 提取原始模型的相关信息和数据集元数据
        model = original_config.MODEL  # 提取模型信息
        model_input = original_config.INPUT  # 提取模型输入信息
        dataset_catalog = MetadataCatalog.get(original_config.DATASETS.TEST_PANOPTIC[0])  # 获取数据集元数据

        # 根据模型所在的模型库设置类别信息文件
        if "ade20k" in model_repo:
            class_info_file = "ade20k_panoptic.json"
        elif "coco" in model_repo:
            class_info_file = "coco_panoptic.json"
        elif "cityscapes" in model_repo:
            class_info_file = "cityscapes_panoptic.json"
        else:
            raise ValueError("Invalid Dataset!")  # 抛出数值错误异常

        # 创建OneFormer图像处理器和CLIPTokenizer对象
        image_processor = OneFormerImageProcessor(
            image_mean=(torch.tensor(model.PIXEL_MEAN) / 255).tolist(),  # 图像均值
            image_std=(torch.tensor(model.PIXEL_STD) / 255).tolist(),  # 图像标准差
            size=model_input.MIN_SIZE_TEST,  # 最小图像尺寸
            max_size=model_input.MAX_SIZE_TEST,  # 最大图像尺寸
            num_labels=model.SEM_SEG_HEAD.NUM_CLASSES,  # 分割类别数
            ignore_index=dataset_catalog.ignore_label,  # 忽略的索引
            class_info_file=class_info_file,  # 类别信息文件
        )

        tokenizer = CLIPTokenizer.from_pretrained(model_repo)  # 使用模型库创建CLIPTokenizer对象

        return OneFormerProcessor(
            image_processor=image_processor,  # 图像处理器
            tokenizer=tokenizer,  # 分词器
            task_seq_length=original_config.INPUT.TASK_SEQ_LEN,  # 任务序列长度
            max_seq_length=original_config.INPUT.MAX_SEQ_LEN,  # 最大序列长度
        )

class OriginalOneFormerCheckpointToOursConverter:
    def __init__(self, original_model: nn.Module, config: OneFormerConfig):
        self.original_model = original_model  # 原始模型
        self.config = config  # 配置对象

    def pop_all(self, renamed_keys: List[Tuple[str, str]], dst_state_dict: StateDict, src_state_dict: StateDict):
        # 从源状态字典中弹出所有指定的键值对到目标状态字典
        for src_key, dst_key in renamed_keys:
            dst_state_dict[dst_key] = src_state_dict.pop(src_key)  # 弹出并添加键值对

    # Swin Backbone
    # Dinat Backbone
    # Backbone + Pixel Decoder
    # Transformer Decoder
    # 替换解码器中的自注意力模块的参数
    def replace_keys_qkv_transformer_decoder(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的前缀
        dst_prefix: str = "transformer_module.decoder.layers"
        # 源状态字典的前缀
        src_prefix: str = "sem_seg_head.predictor"
        # 对解码器中每一层进行迭代
        for i in range(self.config.decoder_layers - 1):
            # 读取自注意力层的输入投影层的权重和偏置
            in_proj_weight = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_weight"
            )
            in_proj_bias = src_state_dict.pop(
                f"{src_prefix}.transformer_self_attention_layers.{i}.self_attn.in_proj_bias"
            )
            # 接下来，按顺序将查询、键和值添加到状态字典中
            # 查询投影层的权重
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
            # 查询投影层的偏置
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.q_proj.bias"] = in_proj_bias[:256]
            # 键投影层的权重
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
            # 键投影层的偏置
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.k_proj.bias"] = in_proj_bias[256:512]
            # 值投影层的权重
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
            # 值投影层的偏置
            dst_state_dict[f"{dst_prefix}.{i}.self_attn.self_attn.v_proj.bias"] = in_proj_bias[-256:]

    # 替换任务MLP的参数
    def replace_task_mlp(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的前缀
        dst_prefix: str = "task_encoder"
        # 源状态字典的前缀
        src_prefix: str = "task_mlp"

        # 重命名权重和偏置的辅助函数
        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = []

        # 对任务MLP的每一层进行迭代
        for i in range(2):
            # 将权重和偏置的键重命名，并添加到重命名键列表中
            renamed_keys.extend(
                rename_keys_for_weight_bias(f"{src_prefix}.layers.{i}", f"{dst_prefix}.task_mlp.layers.{i}.0")
            )

        # 从源状态字典中移除所有重命名的键，并将其添加到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)

    # 替换文本投影器的参数
    def replace_text_projector(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 目标状态字典的前缀
        dst_prefix: str = "text_mapper.text_projector"
        # 源状态字典的前缀
        src_prefix: str = "text_projector"

        # 重命名权重和偏置的辅助函数
        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        renamed_keys = []

        # 对文本投影器的每一层进行迭代
        for i in range(self.config.text_encoder_config["text_encoder_proj_layers"]):
            # 将权重和偏置的键重命名，并添加到重命名键列表中
            renamed_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.layers.{i}", f"{dst_prefix}.{i}.0"))

        # 从源状态字典中移除所有重命名的键，并将其添加到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 替换文本映射器的权重和偏置参数，将源状态字典中的参数替换到目标状态字典中
    def replace_text_mapper(self, dst_state_dict: StateDict, src_state_dict: StateDict):
        # 设置目标状态字典中的权重和偏置参数的前缀
        dst_prefix: str = "text_mapper.text_encoder"
        # 设置源状态字典中的权重和偏置参数的前缀
        src_prefix: str = "text_encoder"

        # 调用内部函数替换文本投影器的权重和偏置参数
        self.replace_text_projector(dst_state_dict, src_state_dict)

        # 定义用于权重和偏置参数重命名的函数，针对注意力机制
        def rename_keys_for_weight_bias(src_prefix: str, dst_prefix: str):
            return [
                (f"{src_prefix}.weight", f"{dst_prefix}.weight"),
                (f"{src_prefix}.bias", f"{dst_prefix}.bias"),
            ]

        # 定义用于注意力机制的权重和偏置参数重命名的函数
        def rename_keys_for_attn(src_prefix: str, dst_prefix: str):
            # 定义注意力机制的参数键列表
            attn_keys = [
                (f"{src_prefix}.in_proj_bias", f"{dst_prefix}.in_proj_bias"),
                (f"{src_prefix}.in_proj_weight", f"{dst_prefix}.in_proj_weight"),
            ]
            # 将注意力机制的输出投影参数也加入到参数键列表中
            attn_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.out_proj", f"{dst_prefix}.out_proj"))

            return attn_keys

        # 定义用于层的权重和偏置参数重命名的函数
        def rename_keys_for_layer(src_prefix: str, dst_prefix: str):
            # 定义残差块的参数键列表
            resblock_keys = []

            # 将残差块中的 MLP 层的参数加入到参数键列表中
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.mlp.c_fc", f"{dst_prefix}.mlp.fc1"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.mlp.c_proj", f"{dst_prefix}.mlp.fc2"))
            # 将残差块中的 Layer Normalization 层的参数加入到参数键列表中
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_1", f"{dst_prefix}.layer_norm1"))
            resblock_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_2", f"{dst_prefix}.layer_norm2"))
            # 将残差块中的注意力机制的参数加入到参数键列表中
            resblock_keys.extend(rename_keys_for_attn(f"{src_prefix}.attn", f"{dst_prefix}.self_attn"))

            return resblock_keys

        # 定义用于所有参数的重命名键列表
        renamed_keys = [
            ("prompt_ctx.weight", "text_mapper.prompt_ctx.weight"),
        ]

        # 将其他需要重命名的参数键加入到重命名键列表中
        renamed_keys.extend(
            [
                (f"{src_prefix}.positional_embedding", f"{dst_prefix}.positional_embedding"),
                (f"{src_prefix}.token_embedding.weight", f"{dst_prefix}.token_embedding.weight"),
            ]
        )

        # 将最终的 Layer Normalization 层的参数键加入到重命名键列表中
        renamed_keys.extend(rename_keys_for_weight_bias(f"{src_prefix}.ln_final", f"{dst_prefix}.ln_final"))

        # 遍历所有层，并将各层的参数键加入到重命名键列表中
        for i in range(self.config.text_encoder_config["text_encoder_num_layers"]):
            renamed_keys.extend(
                rename_keys_for_layer(
                    f"{src_prefix}.transformer.resblocks.{i}", f"{dst_prefix}.transformer.layers.{i}"
                )
            )

        # 将所有需要替换的参数从源状态字典中弹出，并更新到目标状态字典中
        self.pop_all(renamed_keys, dst_state_dict, src_state_dict)
    # 将传入的模型参数从 OneFormerModel 转换为 OneFormerModel，并返回转换后的模型参数
    def convert(self, oneformer: OneFormerModel, is_swin: bool) -> OneFormerModel:
        # 创建目标模型参数的跟踪状态字典，初始化为传入模型参数的状态字典
        dst_state_dict = TrackedStateDict(oneformer.state_dict())
        # 创建原始模型参数的状态字典，使用 self.original_model 的状态字典初始化
        src_state_dict = self.original_model.state_dict()

        # 用原始模型参数的状态字典替换目标模型参数的像素模块
        self.replace_pixel_module(dst_state_dict, src_state_dict, is_swin)
        # 用原始模型参数的状态字典替换目标模型参数的变换器模块
        self.replace_transformer_module(dst_state_dict, src_state_dict)
        # 用原始模型参数的状态字典替换目标模型参数的任务 MLP
        self.replace_task_mlp(dst_state_dict, src_state_dict)
        # 如果配置为训练模式，用原始模型参数的状态字典替换目标模型参数的文本映射器
        if self.config.is_training:
            self.replace_text_mapper(dst_state_dict, src_state_dict)

        # 记录未成功替换的键和值的差异
        logger.info(f"Missed keys are {pformat(dst_state_dict.diff())}")
        # 记录未成功复制的键
        logger.info(f"Not copied keys are {pformat(src_state_dict.keys())}")
        # 输出“完成”的信息
        logger.info("🙌 Done")

        # 加载目标模型参数到传入的模型参数中
        oneformer.load_state_dict(dst_state_dict)

        # 返回转换后的模型参数
        return oneformer

    # 静态方法：使用给定的检查点目录和配置文件目录，生成检查点文件和配置文件的迭代器
    @staticmethod
    def using_dirs(checkpoints_dir: Path, config_dir: Path) -> Iterator[Tuple[object, Path, Path]]:
        # 获取检查点目录中所有.pth文件的路径列表
        checkpoints: List[Path] = checkpoints_dir.glob("**/*.pth")

        # 遍历检查点文件路径列表
        for checkpoint in checkpoints:
            # 输出“转换”信息和当前检查点文件的文件名
            logger.info(f"💪 Converting {checkpoint.stem}")
            # 查找与当前检查点文件相关联的配置文件
            config: Path = config_dir / f"{checkpoint.stem}.yaml"

            # 返回配置文件、检查点文件的元组
            yield config, checkpoint
# 对语义分割的输出进行后处理，调整输出尺寸为指定的大小
def post_process_sem_seg_output(outputs: OneFormerForUniversalSegmentationOutput, target_size: Tuple[int, int]):
    # 获取类别查询的逻辑值，形状为[BATCH, QUERIES, CLASSES + 1]
    class_queries_logits = outputs.class_queries_logits
    # 获取掩码查询的逻辑值，形状为[BATCH, QUERIES, HEIGHT, WIDTH]
    masks_queries_logits = outputs.masks_queries_logits
    if target_size is not None:
        # 若指定了目标尺寸，则对掩码查询的逻辑值进行插值调整尺寸
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    # 删除最后一个空类[..., :-1]
    masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
    # 掩码概率的形状为[BATCH, QUERIES, HEIGHT, WIDTH]
    masks_probs = masks_queries_logits.sigmoid()
    # 现在我们希望对查询求和，
    # $ out_{c,h,w} =  \sum_q p_{q,c} * m_{q,h,w} $
    # 其中 $ softmax(p) \in R^{q, c} $ 是掩码类别
    # 而 $ sigmoid(m) \in R^{q, h, w}$ 是掩码概率
    # b(atch)q(uery)c(lasses), b(atch)q(uery)h(eight)w(idth)
    segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)

    return segmentation

# 测试函数，用于测试模型
def test(
    original_model,
    our_model: OneFormerForUniversalSegmentation,
    processor: OneFormerProcessor,
    model_repo: str,
):
    # 文本预处理函数，用于处理文本列表并返回token输入
    def _preprocess_text(text_list=None, max_length=77):
        if text_list is None:
            raise ValueError("tokens cannot be None.")
        
        # 对文本列表进行处理，获取token
        tokens = tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        for attn_mask, input_id in zip(attention_masks, input_ids):
            # 生成token输入
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs
    # 使用 PyTorch 的 no_grad() 模式进行推理
    with torch.no_grad():
        # 加载 CLIP 分词器
        tokenizer = CLIPTokenizer.from_pretrained(model_repo)
        # 设置原始模型和我们自己的模型为评估模式
        original_model = original_model.eval()
        our_model = our_model.eval()
    
        # 准备图像
        im = prepare_img()
    
        # 定义图像预处理转换器
        tr = T.Compose(
            [
                # 调整图像大小为 640x640
                T.Resize((640, 640)),
                # 将图像转换为张量
                T.ToTensor(),
                # 对图像进行归一化处理
                T.Normalize(
                    mean=torch.tensor([123.675, 116.280, 103.530]) / 255.0,
                    std=torch.tensor([58.395, 57.120, 57.375]) / 255.0,
                ),
            ]
        )
    
        # 对图像应用预处理转换器
        x = tr(im).unsqueeze(0)
    
        # 定义任务输入
        task_input = ["the task is semantic"]
        # 对任务输入进行预处理
        task_token = _preprocess_text(task_input, max_length=processor.task_seq_length)
    
        # 获取原始模型的 backbone 特征
        original_model_backbone_features = original_model.backbone(x.clone())
    
        # 获取我们自己的模型的输出
        our_model_output: OneFormerModelOutput = our_model.model(x.clone(), task_token, output_hidden_states=True)
    
        # 比较原始模型和我们自己的模型的 backbone 特征是否相同
        for original_model_feature, our_model_feature in zip(
            original_model_backbone_features.values(), our_model_output.encoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=3e-3
            ), "The backbone features are not the same."
    
        # 从原始模型中获取 pixel decoder 特征
        mask_features, _, multi_scale_features, _, _ = original_model.sem_seg_head.pixel_decoder.forward_features(
            original_model_backbone_features
        )
        original_pixel_decoder_features = []
        original_pixel_decoder_features.append(mask_features)
        for i in range(len(multi_scale_features)):
            original_pixel_decoder_features.append(multi_scale_features[i])
    
        # 比较原始模型和我们自己的模型的 pixel decoder 特征是否相同
        for original_model_feature, our_model_feature in zip(
            original_pixel_decoder_features, our_model_output.pixel_decoder_hidden_states
        ):
            assert torch.allclose(
                original_model_feature, our_model_feature, atol=3e-4
            ), "The pixel decoder feature are not the same"
    
        # 定义完整的图像预处理转换器
        tr_complete = T.Compose(
            [
                # 调整图像大小为 640x640
                T.Resize((640, 640)),
                # 将图像转换为张量
                T.ToTensor(),
            ]
        )
    
        # 对图像应用完整的预处理转换器
        y = (tr_complete(im) * 255.0).to(torch.int).float()
    
        # 使用原始模型进行语义分割
        original_model_out = original_model([{"image": y.clone(), "task": "The task is semantic"}])
        original_segmentation = original_model_out[0]["sem_seg"]
    
        # 使用我们自己的模型进行语义分割
        our_model_out: OneFormerForUniversalSegmentationOutput = our_model(
            x.clone(), task_token, output_hidden_states=True
        )
        our_segmentation = post_process_sem_seg_output(our_model_out, target_size=(640, 640))[0]
    
        # 比较原始模型和我们自己的模型的语义分割结果是否相同
        assert torch.allclose(
            original_segmentation, our_segmentation, atol=1e-3
        ), "The segmentation image is not the same."
    
        # 打印测试通过的消息
        logger.info("✅ Test passed!")
# 根据检查点文件的路径获取模型的名称
def get_name(checkpoint_file: Path):
    # 从检查点文件名中获取模型原始名称
    model_name_raw: str = checkpoint_file.stem

    # 判断模型使用的骨干网络类型是 Swin 或者 Dino
    backbone = "swin" if "swin" in model_name_raw else "dinat"
    
    dataset = ""
    # 根据模型名包含的关键词确定数据集类型
    if "coco" in model_name_raw:
        dataset = "coco"
    elif "ade20k" in model_name_raw:
        dataset = "ade20k"
    elif "cityscapes" in model_name_raw:
        dataset = "cityscapes"
    else:
        raise ValueError(
            f"{model_name_raw} must be wrong since we didn't find 'coco' or 'ade20k' or 'cityscapes' in it "
        )

    # 定义可能的骨干网络类型
    backbone_types = ["tiny", "large"]

    # 根据模型名中的关键词确定具体的骨干网络类型
    backbone_type = list(filter(lambda x: x in model_name_raw, backbone_types))[0]

    # 根据获得的信息组合模型名称
    model_name = f"oneformer_{dataset}_{backbone}_{backbone_type}"

    return model_name

# 主程序入口
if __name__ == "__main__":
    # 创建参数解析器
    parser = ArgumentParser(
        description=(
            "Command line to convert the original oneformer models (with swin backbone) to transformers"
            " implementation."
        )
    )

    # 添加命令行参数
    parser.add_argument(
        "--checkpoints_dir",
        type=Path,
        help=(
            "A directory containing the model's checkpoints. The directory has to have the following structure:"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.pth; where <CONFIG_NAME> name must follow the"
            " following nomenclature nomenclature: oneformer_<DATASET_NAME>_<BACKBONE>_<BACKBONE_TYPE>"
        ),
    )
    parser.add_argument(
        "--configs_dir",
        type=Path,
        help=(
            "A directory containing the model's configs, see detectron2 doc. The directory has to have the following"
            " structure: <DIR_NAME>/<DATASET_NAME>/<CONFIG_NAME>.yaml; where <CONFIG_NAME> name must follow the"
            " following nomenclature nomenclature: oneformer_<DATASET_NAME>_<BACKBONE>_<BACKBONE_TYPE>"
        ),
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=Path,
        help="Path to the folder to output PyTorch models.",
    )
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

    # 获取各个参数的值
    checkpoints_dir: Path = args.checkpoints_dir
    config_dir: Path = args.configs_dir
    save_directory: Path = args.pytorch_dump_folder_path
    oneformer_dir: Path = args.oneformer_dir

    # 如果输出目录不存在，则创建
    if not save_directory.exists():
        save_directory.mkdir(parents=True)
    # 遍历 OriginalOneFormerCheckpointToOursConverter 类的 using_dirs 方法返回的每对配置文件和检查点文件
    for config_file, checkpoint_file in OriginalOneFormerCheckpointToOursConverter.using_dirs(
        checkpoints_dir, config_dir
    ):
        # 创建 OriginalOneFormerConfigToProcessorConverter 实例，根据配置文件创建处理器
        processor = OriginalOneFormerConfigToProcessorConverter()(
            setup_cfg(Args(config_file=config_file)), os.path.join("shi-labs", config_file.stem)
        )

        # 根据配置文件创建原始配置
        original_config = setup_cfg(Args(config_file=config_file))
        
        # 从原始配置中创建 OneFormer 模型的参数
        oneformer_kwargs = OriginalOneFormer.from_config(original_config)

        # 创建 OriginalOneFormer 模型的实例，并设置为评估模式
        original_model = OriginalOneFormer(**oneformer_kwargs).eval()

        # 加载检查点文件到 OriginalOneFormer 模型中
        DetectionCheckpointer(original_model).load(str(checkpoint_file))

        # 检查 config_file.stem 是否包含 "swin"，并赋值给 is_swin
        is_swin = "swin" in config_file.stem

        # 使用 OriginalOneFormerConfigToOursConverter 将原始配置转换为我们的配置
        config: OneFormerConfig = OriginalOneFormerConfigToOursConverter()(original_config, is_swin)

        # 创建 OneFormerModel 模型的实例，并设置为评估模式
        oneformer = OneFormerModel(config=config).eval()

        # 创建 OriginalOneFormerCheckpointToOursConverter 的实例，同时传入原始模型和配置
        converter = OriginalOneFormerCheckpointToOursConverter(original_model, config)

        # 将原始模型转换为我们的模型
        oneformer = converter.convert(oneformer, is_swin)

        # 创建 OneFormerForUniversalSegmentation 模型的实例，并设置为评估模式
        oneformer_for_universal_segmentation = OneFormerForUniversalSegmentation(config=config).eval()

        # 设置 OneFormerForUniversalSegmentation 的模型为转换后的 OneFormer 模型
        oneformer_for_universal_segmentation.model = oneformer

        # 测试 OriginalOneFormer 和转换后的模型
        test(
            original_model,
            oneformer_for_universal_segmentation,
            processor,
            os.path.join("shi-labs", config_file.stem),
        )

        # 获取模型名称
        model_name = get_name(checkpoint_file)
        logger.info(f"🪄 Saving {model_name}")

        # 保存处理器预训练模型到指定目录下
        processor.save_pretrained(save_directory / model_name)
        # 保存 OneFormerForUniversalSegmentation 模型到指定目录下
        oneformer_for_universal_segmentation.save_pretrained(save_directory / model_name)

        # 推送处理器到指定仓库
        processor.push_to_hub(
            repo_id=os.path.join("shi-labs", config_file.stem),
            commit_message="Add configs",
            use_temp_dir=True,
        )
        # 推送 OneFormerForUniversalSegmentation 模型到指定仓库
        oneformer_for_universal_segmentation.push_to_hub(
            repo_id=os.path.join("shi-labs", config_file.stem),
            commit_message="Add model",
            use_temp_dir=True,
        )
```