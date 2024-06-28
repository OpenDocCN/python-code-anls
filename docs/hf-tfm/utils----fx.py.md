# `.\utils\fx.py`

```
# 导入 Python 内置模块和第三方库
import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
# 导入类型提示相关的模块和类
from typing import Any, Callable, Dict, List, Optional, Type, Union

# 导入 PyTorch 库
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy

# 导入 Transformers 相关模块和类
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from ..pytorch_utils import is_torch_greater_or_equal_than_2_0
from ..utils import (
    ENV_VARS_TRUE_VALUES,
    TORCH_FX_REQUIRED_VERSION,
    get_torch_version,
    is_peft_available,
    is_torch_fx_available,
)

# 如果 peft 可用，则导入 PeftModel 类
if is_peft_available():
    from peft import PeftModel

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 检查是否在调试模式下
_IS_IN_DEBUG_MODE = os.environ.get("FX_DEBUG_MODE", "").upper() in ENV_VARS_TRUE_VALUES


def _generate_supported_model_class_names(
    model_name: Type[PretrainedConfig],
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    # 定义任务映射字典，将任务名称映射到模型名称字典
    task_mapping = {
        "default": MODEL_MAPPING_NAMES,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING_NAMES,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "ctc": MODEL_FOR_CTC_MAPPING_NAMES,
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        "backbone": MODEL_FOR_BACKBONE_MAPPING_NAMES,
        "image-feature-extraction": MODEL_FOR_IMAGE_MAPPING_NAMES,
    }
    
    # 如果 supported_tasks 参数为 None，则使用所有任务的键作为支持的任务列表
    if supported_tasks is None:
        supported_tasks = task_mapping.keys()
    
    # 如果 supported_tasks 是字符串，则转换为包含该字符串的列表
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]
    
    # 初始化空列表，用于存储模型类名称
    model_class_names = []
    
    # 遍历每个支持的任务
    for task in supported_tasks:
        # 获取任务对应的模型名称的类名，如果找不到则设为 None
        class_name = task_mapping[task].get(model_name, None)
        # 如果找到了类名，则将其添加到模型类名称列表中
        if class_name:
            model_class_names.append(class_name)
    
    # 返回所有找到的模型类名称列表
    return model_class_names
# 正常支持的模型名称和任务列表，用于模型选择和加载
_REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS = [
    "altclip",              # 替代版本的CLIP模型
    "albert",               # ALBERT模型
    "bart",                 # BART模型
    "bert",                 # BERT模型
    "blenderbot",           # BlenderBot模型
    "blenderbot-small",     # 小型BlenderBot模型
    "bloom",                # Bloom模型
    "clip",                 # CLIP模型
    "convnext",             # ConvNext模型
    "deberta",              # DeBERTa模型
    "deberta-v2",           # DeBERTa-v2模型
    "dinov2",               # DINOv2模型
    "distilbert",           # DistilBERT模型
    "donut-swin",           # Donut-Swin模型
    "electra",              # Electra模型
    "gpt2",                 # GPT-2模型
    "gpt_neo",              # GPT-Neo模型
    "gptj",                 # GPT-J模型
    "hubert",               # Hubert模型
    "layoutlm",             # LayoutLM模型
    "llama",                # LLaMA模型
    "cohere",               # Cohere模型
    "lxmert",               # LXMERT模型
    "m2m_100",              # M2M-100模型
    "marian",               # Marian模型
    "mbart",                # mBART模型
    "megatron-bert",        # Megatron-BERT模型
    "mobilebert",           # MobileBERT模型
    "mt5",                  # MT5模型
    "nezha",                # NeZha模型
    "opt",                  # Opt模型
    "pegasus",              # Pegasus模型
    "plbart",               # PLBART模型
    "resnet",               # ResNet模型
    "roberta",              # RoBERTa模型
    "segformer",            # Segformer模型
    "speech_to_text",       # 语音转文本模型
    "speech_to_text_2",     # 语音转文本模型的另一版本
    "swin",                 # Swin模型
    "t5",                   # T5模型
    "trocr",                # TrOCR模型
    "vit",                  # ViT模型
    "xglm",                 # XGLM模型
    "wav2vec2",             # Wav2Vec 2.0模型
    #    "xlnet",             # 暂时未支持XLNet模型
]

# 支持KV缓存的特殊模型列表
_FX_SUPPORTED_MODELS_WITH_KV_CACHE = ["llama", "opt"]

# 初始化空的正常支持模型列表
_REGULAR_SUPPORTED_MODELS = []

# 遍历正常支持的模型名称和任务列表
for item in _REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS:
    # 如果列表项是字典，则生成支持的模型类名并扩展到正常支持模型列表
    if isinstance(item, dict):
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_class_names(**item))
    else:
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_class_names(item))

# 特殊支持的模型列表，包含特定类名的模型
_SPECIAL_SUPPORTED_MODELS = [
    "CLIPTextModel",                    # CLIP文本模型
    "CLIPTextModelWithProjection",      # 带投影的CLIP文本模型
    "CLIPVisionModel",                  # CLIP视觉模型
    "CLIPVisionModelWithProjection",    # 带投影的CLIP视觉模型
    "AltCLIPTextModel",                 # 替代版本的CLIP文本模型
    "AltCLIPVisionModel",               # 替代版本的CLIP视觉模型
    "GitVisionModel",                   # Git视觉模型
    "GPT2DoubleHeadsModel",             # GPT-2双头模型
    "Speech2Text2Decoder",              # 语音转文本2解码器
    "TrOCRDecoder",                     # TrOCR解码器
    "PeftModelForCausalLM",             # 用于因果语言建模的Peft模型
    "PeftModelForSeq2SeqLM",            # 用于序列到序列语言建模的Peft模型
    # TODO: 添加对它们的支持，这应该很容易做到（存在小的阻碍问题）。
    # XLNetForQuestionAnswering,       # 问答XLNet模型，暂未支持
]

# 所有支持的模型列表，由正常支持和特殊支持模型列表组成，按字母顺序排序并去重
_SUPPORTED_MODELS = tuple(sorted(set(_REGULAR_SUPPORTED_MODELS + _SPECIAL_SUPPORTED_MODELS)))


def torch_nn_embedding(self, input):
    # 创建一个与输入形状相同，但最后一个维度与权重张量相同的空张量，设备为"meta"，类型与权重相同
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta", dtype=self.weight.dtype)


def torch_nn_functional_embedding(
    input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
):
    # 创建一个与输入形状相同，但最后一个维度与权重张量相同的空张量，设备为"meta"，类型与权重相同
    return torch.empty(*input.shape, weight.shape[-1], device="meta", dtype=weight.dtype)


def torch_nn_layernorm(self, input):
    # 返回输入本身，表示不应用 LayerNorm 操作
    return input


def torch_nn_groupnorm(self, input):
    # 返回输入本身，表示不应用 GroupNorm 操作
    return input


def torch_nn_linear(self, input):
    # 创建一个与输入形状的前几维相同，但最后一维是输出特征数的空张量，设备为"meta"
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")


def torch_relu(x):
    # 返回输入本身，表示不应用 ReLU 激活函数
    return x


def torch_nn_relu(self, x):
    # 返回输入本身，表示不应用 ReLU 激活函数
    return x


def torch_nn_functional_relu(x, inplace=False):
    # 如果不支持原地操作，则抛出异常
    if not inplace:
        raise ValueError("Don't support in-place functional.relu for MetaTensor analysis")
    return x


def torch_where(condition, x, y):
    # 使用加法模拟 torch.where 的行为，返回条件、x 和 y 的广播张量
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")


def torch_abs(input, *, out=None):
    # 如果要求原地操作，则抛出异常
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    return input


def torch_arange(*args, **kwargs):
    # 计算输入参数的数量
    n = len(args)
    step = 1
    # 检查参数 n 的数量，根据不同情况设置起始、结束和步长
    if n == 1:
        # 当只有一个参数时，设定起始为0，结束为第一个参数值
        start = 0
        end = args[0]
    elif n == 2:
        # 当有两个参数时，分别设置起始和结束为两个参数的值
        start, end = args
    else:
        # 当有三个参数时，设置起始、结束和步长分别为三个参数的值
        start, end, step = args
    
    # 检查起始、结束和步长是否为浮点数，若是则转换为整数
    if isinstance(start, float):
        start = int(start)
    if isinstance(end, float):
        # 修正：应该是 end 而不是 start
        end = int(end)
    if isinstance(step, float):
        step = int(step)
    
    # 获取关键字参数中的步长，若未提供则使用之前设置的步长值
    step = kwargs.get("step", step)
    # 获取关键字参数中的数据类型，用于创建 tensor
    dtype = kwargs.get("dtype")
    # 返回一个空的 torch tensor，形状为((end - start) // step)，指定数据类型和设备为 "meta"
    return torch.empty((end - start) // step, dtype=dtype, device="meta")
# 创建一个函数 torch_full，用于生成一个指定维度和值的张量
def torch_full(*args, **kwargs):
    # 将位置参数转换为列表
    args = list(args)
    # 如果第二个参数是 torch.Tensor 类型且设备是 "meta"，则将其设为 1（任意值）
    if isinstance(args[1], torch.Tensor) and args[1].device == torch.device("meta"):
        args[1] = 1  # 任意值。
    # 复制关键字参数到新的字典，去除 "device" 参数
    kwargs_without_device = dict(kwargs)
    kwargs_without_device.pop("device", None)
    # 调用 torch.full 函数，生成张量并返回
    return torch.full(*args, **kwargs_without_device)


# 创建一个函数 torch_cat，用于沿指定维度对张量进行拼接
def torch_cat(tensors, dim=None, axis=None, *, out=None):
    # 如果 dim 和 axis 都未指定，则将 dim 设置为 0
    if dim is None and axis is None:
        dim = 0
    # 如果 dim 未指定但 axis 指定了，则将 dim 设置为 axis
    if dim is None and axis is not None:
        dim = axis
    # 如果 dim 是负数，将其转换为正数索引
    if dim < 0:
        dim = tensors[0].dim() + dim
    # 获取所有张量的形状
    shapes = [t.shape for t in tensors]
    # 获取第一个张量的形状
    shape = list(shapes[0])
    # 计算拼接后张量在指定维度上的总长度
    concatenated_dim = sum(shape[dim] for shape in shapes)
    # 构建最终的张量形状
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim + 1 :]
    # 返回一个新的空张量，形状为 final_shape，设备为 "meta"
    return torch.empty(final_shape, device="meta")


# 创建一个函数 torch_stack，用于沿新的维度对张量序列进行堆叠
def torch_stack(tensors, dim=None, axis=None, *, out=None):
    # 如果 dim 和 axis 都未指定，则将 dim 设置为 0
    if dim is None and axis is None:
        dim = 0
    # 如果 dim 未指定但 axis 指定了，则将 dim 设置为 axis
    if dim is None and axis is not None:
        dim = axis
    # 如果 dim 是负数，将其转换为正数索引
    if dim < 0:
        dim = tensors[0].dim() + 1 + dim
    # 获取第一个张量的形状
    shape = list(tensors[0].shape)
    # 在指定维度上插入新的维度，长度为张量序列的长度
    shape.insert(dim, len(tensors))
    # 返回一个新的空张量，形状为 shape，设备为 "meta"
    return torch.empty(shape, device="meta")


# 创建一个函数 torch_add，用于对两个张量进行加法操作
def torch_add(input, other, *, alpha=1, out=None):
    # 如果 input 不是 torch.Tensor 类型，则返回一个与 other 相同形状的空张量，设备为 "meta"
    if not isinstance(input, torch.Tensor):
        return torch.empty_like(other, device="meta")
    # 如果 other 不是 torch.Tensor 类型，则返回一个与 input 相同形状的空张量，设备为 "meta"
    if not isinstance(other, torch.Tensor):
        return torch.empty_like(input, device="meta")
    # 计算两个张量的最大维度
    max_length = max(input.dim(), other.dim())
    # 将 input 和 other 扩展为相同的维度
    input_shape = list(input.shape) + [1] * (max_length - input.dim())
    other_shape = list(other.shape) + [1] * (max_length - other.dim())
    shape = []
    for i in range(max_length):
        shape.append(max(input_shape[i], other_shape[i]))
    # 返回一个新的空张量，形状为 shape，设备为 "meta"
    return torch.empty(shape, device="meta")


# 创建一个函数 torch_mul，用于对两个张量进行乘法操作，实际上调用了 torch_add 函数
def torch_mul(input, other, *, out=None):
    return torch_add(input, other, out=out)


# 创建一个函数 torch_tensor_mul，用于对两个张量进行乘法操作，实际上调用了 torch_mul 函数
def torch_tensor_mul(self, other):
    return torch_mul(self, other)


# 创建一个函数 torch_matmul，用于对两个张量进行矩阵乘法操作
def torch_matmul(input, other, *, out=None):
    # 获取 input 和 other 的维度
    d1 = input.dim()
    d2 = other.dim()
    shape = None
    # 根据不同的维度情况进行判断和处理
    if d1 == 1 and d2 == 1:
        shape = None
    elif d1 == 2 and d2 == 2:
        shape = (input.size(0), other.size(1))
    elif d1 == 1 and d2 == 2:
        shape = (other.size(1),)
    elif d1 == 2 and d1 == 1:  # 应为 d2 == 1
        shape = (input.size(0),)
    else:
        max_length = max(input.dim(), other.dim())
        shape1 = list(input.shape)
        shape2 = list(other.shape)
        if d1 == 1:
            shape1 = [1] + shape1
        if d2 == 1:
            shape2.append(1)
        shape1 = [-1] * (max_length - d1) + list(input.shape)
        shape2 = [-1] * (max_length - d2) + list(other.shape)
        shape = []
        for i in range(max_length):
            shape.append(max(shape1[i], shape2[i]))
        shape[-2] = shape1[-2]
        shape[-1] = shape2[-1]
        if d1 == 1:
            shape.pop(-2)
        if d2 == 1:
            shape.pop(-1)
    # 如果 shape 为 None，则返回一个标量张量 0.0，设备为 "meta"
    if shape is None:
        return torch.tensor(0.0, device="meta")
    # 返回一个新的空张量，形状为 shape，设备为 "meta"
    return torch.empty(*shape, device="meta")
def torch_bmm(input, mat2, *, out=None):
    # 如果指定了输出张量out，抛出值错误异常，不支持原地操作
    if out is not None:
        raise ValueError("Don't support in-place bmm for MetaTensor analysis")
    # 获取输入张量input的批大小、行数n、列数m
    batch_size, n, m = input.shape
    # 获取mat2张量的最后两个维度的大小，即第二个维度的行数和第三个维度的列数
    _, _, p = mat2.shape
    # 返回一个空的元数据设备张量，形状为(batch_size, n, p)
    return torch.empty(batch_size, n, p, device="meta")


def torch_baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    # 如果指定了输出张量out，抛出值错误异常，不支持原地操作
    if out is not None:
        raise ValueError("Don't support in-place baddbmm for MetaTensor analysis")
    # 调用torch_bmm函数计算batch1和batch2的批次矩阵乘积，返回结果
    return torch_bmm(batch1, batch2)


def torch_tensor_baddbmm(self, batch1, batch2, *, beta=1, alpha=1, out=None):
    # 调用torch_baddbmm函数计算self张量和batch1、batch2的批次矩阵乘积，返回结果
    return torch_baddbmm(self, batch1, batch2, beta=beta, alpha=alpha, out=out)


def torch_einsum(equation, *operands):
    # TODO: infer shape without performing the computation, this might be quite hard.
    # 创建与操作数具有相同形状的空张量列表，设备为CPU
    concrete_operands = (torch.empty_like(operand, device="cpu") for operand in operands)
    # 执行爱因斯坦求和符号运算，返回结果张量并设备为元数据设备
    return torch.einsum(equation, *concrete_operands).to("meta")


def torch_tensor_repeat(self, *sizes):
    # 获取self张量的形状列表
    shape = list(self.shape)
    # 根据sizes参数修改形状列表中的每个维度大小
    for i, x in enumerate(sizes):
        shape[i] *= x
    # 返回一个空的元数据设备张量，形状由修改后的形状列表确定
    return torch.empty(shape, device="meta")


def torch_repeat_interleave(*args, dim=None, output_size=None):
    # 获取参数数量
    num_args = len(args)
    # 如果参数数量为1
    if num_args == 1:
        # 如果output_size不为None，则创建一个形状为[output_size]的列表
        shape = [output_size if output_size is not None else args[0].sum()]
    else:
        # 否则创建一个形状与第一个参数args[0]相同的列表
        shape = list(args[0].shape)
        # 如果未指定维度dim
        if dim is None:
            # 如果参数数量大于2，则将dim设置为args[2]
            if num_args > 2:
                dim = args[2]
            else:
                # 否则将shape变为包含总和的列表，并将dim设置为0
                shape = [sum(shape)]
                dim = 0
        # 获取重复次数repeats
        repeats = args[1]
        # 如果repeats是整数或者元素数量为1
        if isinstance(repeats, int) or torch.numel(repeats) == 1:
            # 将shape[dim]乘以repeats的整数值
            shape[dim] *= int(repeats)
        else:
            # 否则将shape[dim]设置为output_size或者repeats的总和
            shape[dim] = output_size if output_size is not None else repeats.sum()
    # 返回一个空的元数据设备张量，形状由shape确定
    return torch.empty(*shape, device="meta")


def torch_index_select(input, dim, index, *, out=None):
    # 获取input张量的形状列表
    shape = list(input.shape)
    # 修改形状列表中指定维度dim的大小为索引index的长度
    shape[dim] = len(index)
    # 返回一个空的元数据设备张量，形状由修改后的形状列表确定
    return torch.empty(*shape, device="meta")


def torch_tensor_index_select(self, dim, index):
    # 调用torch_index_select函数从self张量中选择指定维度dim的索引index，返回结果
    return torch_index_select(self, dim, index)


def torch_gather(input, dim, index, *, sparse_grad=False, out=None):
    # 获取input张量的形状列表
    shape = list(input.shape)
    # 修改形状列表中指定维度dim的大小为索引index指定维度的大小
    shape[dim] = index.shape[dim]
    # 返回一个空的元数据设备张量，形状由修改后的形状列表确定
    return torch.empty(*shape, device="meta")


def torch_tensor_gather(self, dim, index):
    # 调用torch_gather函数从self张量中收集指定维度dim的索引index，返回结果
    return torch_gather(self, dim, index)


def torch_roll(input, shifts, dims=None):
    # 返回未修改的输入张量input
    return input


def torch_flip(input, dims):
    # 返回未修改的输入张量input
    return input


def torch_tensor_flip(self, dims):
    # 返回未修改的self张量
    return self


def torch_nn_conv1d(self, input):
    # 获取输入input的最后一个维度的大小
    l_in = input.shape[-1]
    # 初始化形状为None
    shape = None
    # 获取卷积的填充方式
    padding = self.padding
    # 如果填充为"valid"，则将padding设为(0, 0)
    if padding == "valid":
        padding = (0, 0)
    # 如果填充为"same"
    if padding == "same":
        # 将shape设置为输入input的形状列表
        shape = list(input.shape)
    # 如果shape仍为None
    if shape is None:
        # 将shape设置为输入input的形状列表
        shape = list(input.shape)
        # 计算输出的长度l_out，并向下取整
        l_out = math.floor(
            (l_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        # 修改shape中倒数第二个维度的大小为输出通道数self.out_channels
        shape[-1] = l_out
    # 修改shape中倒数第三个维度的大小为输出通道数self.out_channels
    shape[-2] = self.out_channels
    # 返回一个空的元数据设备张量，形状由修改后的shape确定
    return torch.empty(shape, device="meta")
# 定义一个类方法用于进行二维卷积操作
def torch_nn_conv2d(self, input):
    # 获取输入张量的高度和宽度
    h_in, w_in = input.shape[-2:]
    # 初始化形状变量为 None
    shape = None
    # 获取填充参数
    padding = self.padding
    # 如果填充方式是 "valid"，则将 padding 设置为 (0, 0)
    if padding == "valid":
        padding = (0, 0)
    # 如果填充方式是 "same"，则复制输入张量的形状
    if padding == "same":
        shape = list(input.shape)
    # 如果形状仍为 None，则根据卷积参数计算输出形状
    if shape is None:
        shape = list(input.shape)
        h_out = math.floor(
            (h_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        w_out = math.floor(
            (w_in + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )
        shape[-2:] = [h_out, w_out]
    # 设置输出张量的通道数维度为 self.out_channels
    shape[-3] = self.out_channels
    # 返回一个空的元数据张量，其形状由计算得出
    return torch.empty(shape, device="meta")


# 定义一个函数用于从输入张量中挤压维度为1的维度
def torch_squeeze(input, dim=None):
    # 获取输入张量的形状
    shape = list(input.shape)
    # 如果指定了维度 dim
    if dim is not None:
        # 将负数索引转换为正数索引
        if dim < 0:
            dim = input.dim() + dim
        # 如果指定维度的大小为1，则从形状中删除该维度
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        # 如果未指定维度 dim，则遍历形状，删除所有大小为1的维度
        new_shape = []
        for dim_value in shape:
            if dim_value == 1:
                continue
            new_shape.append(dim_value)
        shape = new_shape
    # 返回一个空的元数据张量，其形状为经过挤压操作后的形状
    return torch.empty(shape, device="meta")


# 定义一个类方法，用于对类的实例进行挤压操作
def torch_tensor_squeeze(self, dim=None):
    # 调用全局的挤压函数 torch_squeeze 对类实例进行操作
    return torch_squeeze(self, dim)


# 定义一个函数用于在指定维度 dim 上对输入张量进行展开操作
def torch_unsqueeze(input, dim):
    # 获取输入张量的形状
    shape = list(input.shape)
    # 将负数索引转换为正数索引
    if dim < 0:
        dim = input.dim() + 1 + dim
    # 在指定维度 dim 处插入一个大小为1的维度
    shape.insert(dim, 1)
    # 返回一个空的元数据张量，其形状为经过展开操作后的形状
    return torch.empty(shape, device="meta")


# 定义一个类方法，用于对类的实例进行展开操作
def torch_tensor_unsqueeze(self, dim):
    # 调用全局的展开函数 torch_unsqueeze 对类实例进行操作
    return torch_unsqueeze(self, dim)


# 定义一个函数用于计算输入张量的连续唯一值，并保持顺序
def torch_unique_consecutive(input, **kwargs):
    # 调用 PyTorch 的 torch.unique_consecutive 函数对输入张量进行操作
    output = torch.unique_consecutive(torch.zeros_like(input, device="cpu"), **kwargs)
    # 如果输出是张量，则将其转移到设备 "meta"
    if isinstance(output, torch.Tensor):
        return output.to("meta")
    else:
        # 否则，将其元组化，并使用映射函数将其中的张量转移到设备 "meta"
        return tuple(map(output, lambda x: x.to("meta")))


# 定义一个函数用于为输入张量的每个元素创建一个指定长度的独热编码
def torch_nn_functional_one_hot(tensor, num_classes=-1):
    # 如果未指定 num_classes，则抛出错误，不支持自动推断 num_classes
    if num_classes < 0:
        raise ValueError("Don't support automatic num_classes inference for MetaTensor analysis")
    # 计算输出张量的形状，将最后一个维度扩展为 num_classes
    shape = list(tensor.shape) + [num_classes]
    # 返回一个空的元数据张量，其形状为计算得出的形状
    return torch.empty(shape, device="meta")


# 定义一个函数用于实现缩放点积注意力机制
def torch_nn_functional_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    # 获取查询张量的目标长度和值张量的头部维度
    target_length = query.shape[-2]
    head_dim = value.shape[-1]
    # 返回一个空的元数据张量，其形状为 (query.shape[:-2], target_length, head_dim)
    return torch.empty((*query.shape[:-2], target_length, head_dim), device="meta")


# 定义一个类方法用于计算均方误差损失
def torch_nn_mseloss(self, input, target):
    # 如果损失函数的减少方式为 "none"，则输出的形状与目标形状相同
    if self.reduction == "none":
        shape = target.shape
    else:
        # 否则，输出的形状为 (1,)
        shape = (1,)
    # 返回一个空的元数据张量，其形状为计算得出的形状
    return torch.empty(shape, device="meta")


# 定义一个类方法用于计算交叉熵损失
def torch_nn_crossentropyloss(self, input, target):
    # 如果损失函数的减少方式为 "none"，则输出的形状与目标形状相同
    if self.reduction == "none":
        shape = target.shape
    else:
        # 否则，输出的形状为 (1,)
        shape = (1,)
    # 返回一个空的元数据张量，其形状为计算得出的形状
    return torch.empty(shape, device="meta")


# 定义一个类方法用于计算带有 logits 的二元交叉熵损失
def torch_nn_bcewithlogitsloss(self, input, target):
    # 如果损失函数的减少方式为 "none"，则输出的形状与目标形状相同
    if self.reduction == "none":
        shape = target.shape
    else:
        # 否则，输出的形状为 (1,)
        shape = (1,)
    # 返回一个空的元数据张量，其形状为计算得出的形状
    return torch.empty(shape, device="meta")


# 定义一个函数操作符，用于获取张量 a 中的 b 元素
def operator_getitem(a, b):
    # 省略函数实现，没有实际执行体
    pass
    # 定义函数 to_concrete，用于将输入的张量 t 转换为具体的张量
    def to_concrete(t):
        # 如果 t 是 torch.Tensor 类型
        if isinstance(t, torch.Tensor):
            # 创建一个与 t 相同形状的全为1的张量，存储在 CPU 上
            concrete = torch.ones_like(t, device="cpu")
            # 如果 concrete 的数据类型是浮点数或者整数32位，则将其转换为64位整数
            if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
                concrete = concrete.to(torch.int64)
            return concrete
        # 如果 t 不是 torch.Tensor 类型，直接返回 t
        return t

    # 如果 a 是 torch.Tensor 类型
    if isinstance(a, torch.Tensor):
        # TODO: 推断形状而不执行计算。
        # 如果 b 是元组类型，对元组中的每个元素应用 to_concrete 函数
        if isinstance(b, tuple):
            b = tuple(map(to_concrete, b))
        else:
            # 否则，将 b 应用 to_concrete 函数
            b = to_concrete(b)
        # 返回使用 a 形状创建的空张量的子集，转换为 "meta" 类型
        return operator.getitem(torch.empty_like(a, device="cpu"), b).to("meta")
    # 如果 a 不是 torch.Tensor 类型，返回 a 的子集
    return operator.getitem(a, b)
# 定义一个字典，用于存储手动指定的函数覆盖映射，将特定的 Torch 函数映射到自定义函数
_MANUAL_META_OVERRIDES: Dict[Callable, Callable] = {
    torch.nn.Embedding: torch_nn_embedding,  # 将 torch.nn.Embedding 映射到 torch_nn_embedding 函数
    torch.nn.functional.embedding: torch_nn_functional_embedding,  # 将 torch.nn.functional.embedding 映射到 torch_nn_functional_embedding 函数
    torch.nn.LayerNorm: torch_nn_layernorm,  # 将 torch.nn.LayerNorm 映射到 torch_nn_layernorm 函数
    torch.nn.GroupNorm: torch_nn_groupnorm,  # 将 torch.nn.GroupNorm 映射到 torch_nn_groupnorm 函数
    torch.nn.Linear: torch_nn_linear,  # 将 torch.nn.Linear 映射到 torch_nn_linear 函数
    torch.relu: torch_relu,  # 将 torch.relu 映射到 torch_relu 函数
    torch.nn.functional.relu: torch_nn_functional_relu,  # 将 torch.nn.functional.relu 映射到 torch_nn_functional_relu 函数
    torch.nn.ReLU: torch_nn_relu,  # 将 torch.nn.ReLU 映射到 torch_nn_relu 函数
    torch.where: torch_where,  # 将 torch.where 映射到 torch_where 函数
    torch.abs: torch_abs,  # 将 torch.abs 映射到 torch_abs 函数
    torch.arange: torch_arange,  # 将 torch.arange 映射到 torch_arange 函数
    torch.full: torch_full,  # 将 torch.full 映射到 torch_full 函数
    torch.cat: torch_cat,  # 将 torch.cat 映射到 torch_cat 函数
    torch.stack: torch_stack,  # 将 torch.stack 映射到 torch_stack 函数
    torch.add: torch_add,  # 将 torch.add 映射到 torch_add 函数
    torch.mul: torch_mul,  # 将 torch.mul 映射到 torch_mul 函数
    torch.Tensor.mul: torch_tensor_mul,  # 将 torch.Tensor.mul 映射到 torch_tensor_mul 函数
    torch.matmul: torch_matmul,  # 将 torch.matmul 映射到 torch_matmul 函数
    torch.bmm: torch_bmm,  # 将 torch.bmm 映射到 torch_bmm 函数
    torch.baddbmm: torch_baddbmm,  # 将 torch.baddbmm 映射到 torch_baddbmm 函数
    torch.Tensor.baddbmm: torch_tensor_baddbmm,  # 将 torch.Tensor.baddbmm 映射到 torch_tensor_baddbmm 函数
    torch.einsum: torch_einsum,  # 将 torch.einsum 映射到 torch_einsum 函数
    torch.Tensor.repeat: torch_tensor_repeat,  # 将 torch.Tensor.repeat 映射到 torch_tensor_repeat 函数
    torch.repeat_interleave: torch_repeat_interleave,  # 将 torch.repeat_interleave 映射到 torch_repeat_interleave 函数
    torch.roll: torch_roll,  # 将 torch.roll 映射到 torch_roll 函数
    torch.flip: torch_flip,  # 将 torch.flip 映射到 torch_flip 函数
    torch.Tensor.flip: torch_tensor_flip,  # 将 torch.Tensor.flip 映射到 torch_tensor_flip 函数
    torch.index_select: torch_index_select,  # 将 torch.index_select 映射到 torch_index_select 函数
    torch.Tensor.index_select: torch_tensor_index_select,  # 将 torch.Tensor.index_select 映射到 torch_tensor_index_select 函数
    torch.gather: torch_gather,  # 将 torch.gather 映射到 torch_gather 函数
    torch.Tensor.gather: torch_tensor_gather,  # 将 torch.Tensor.gather 映射到 torch_tensor_gather 函数
    torch.nn.Conv1d: torch_nn_conv1d,  # 将 torch.nn.Conv1d 映射到 torch_nn_conv1d 函数
    torch.nn.Conv2d: torch_nn_conv2d,  # 将 torch.nn.Conv2d 映射到 torch_nn_conv2d 函数
    torch.squeeze: torch_squeeze,  # 将 torch.squeeze 映射到 torch_squeeze 函数
    torch.Tensor.squeeze: torch_tensor_squeeze,  # 将 torch.Tensor.squeeze 映射到 torch_tensor_squeeze 函数
    torch.unsqueeze: torch_unsqueeze,  # 将 torch.unsqueeze 映射到 torch_unsqueeze 函数
    torch.Tensor.unsqueeze: torch_tensor_unsqueeze,  # 将 torch.Tensor.unsqueeze 映射到 torch_tensor_unsqueeze 函数
    torch.unique_consecutive: torch_unique_consecutive,  # 将 torch.unique_consecutive 映射到 torch_unique_consecutive 函数
    torch.nn.functional.one_hot: torch_nn_functional_one_hot,  # 将 torch.nn.functional.one_hot 映射到 torch_nn_functional_one_hot 函数
    torch.nn.MSELoss: torch_nn_mseloss,  # 将 torch.nn.MSELoss 映射到 torch_nn_mseloss 函数
    torch.nn.CrossEntropyLoss: torch_nn_crossentropyloss,  # 将 torch.nn.CrossEntropyLoss 映射到 torch_nn_crossentropyloss 函数
    torch.nn.BCEWithLogitsLoss: torch_nn_bcewithlogitsloss,  # 将 torch.nn.BCEWithLogitsLoss 映射到 torch_nn_bcewithlogitsloss 函数
    operator.getitem: operator_getitem,  # 将 operator.getitem 映射到 operator_getitem 函数
}

# 如果 Torch 的版本大于等于 2.0，将 torch.nn.functional.scaled_dot_product_attention 映射到 torch_nn_functional_scaled_dot_product_attention 函数
if is_torch_greater_or_equal_than_2_0:
    _MANUAL_META_OVERRIDES[
        torch.nn.functional.scaled_dot_product_attention
    ] = torch_nn_functional_scaled_dot_product_attention


class HFProxy(Proxy):
    """
    Proxy that uses metadata to handle data-dependent control-flow.
    """

    def install_metadata(self, metadata):
        self._metadata = metadata

    @property
    def shape(self):
        # 使用追踪器创建一个代理对象，调用方法为 "size"，参数为自身 (self)，返回创建的代理对象
        return self.tracer.create_proxy("call_method", "size", (self,), {})

    @property
    def device(self):
        # 用于跟踪设备使用情况的 Hack。在元张量传播期间，将这些值替换为常量 'meta'
        return MetaDeviceAttribute(self, "device")

    def __len__(self):
        # 如果存在 _metadata 属性且不为 None，则返回 _metadata 的长度，否则调用父类的 __len__ 方法返回长度
        if hasattr(self, "_metadata") and self._metadata is not None:
            return len(self._metadata)
        return super().__len__()

    def __bool__(self):
        # 如果存在 _metadata 属性且不为 None，则返回 _metadata，否则调用父类的 __bool__ 方法返回布尔值
        if hasattr(self, "_metadata") and self._metadata is not None:
            return self._metadata
        return super().__bool__()
    # 当访问不存在的属性时被调用，这里检查属性名称是否为"_metadata"
    def __getattr__(self, k):
        # 如果属性名称为"_metadata"，直接返回其属性值
        if k == "_metadata":
            return self.__getattribute__(k)
        # 如果不是"_metadata"，创建并返回一个HFAttribute对象，用于处理属性访问
        # 注意：如果这是一个方法调用，它还未添加到图形中，我们会优化为方法调用
        return HFAttribute(self, k)

    # 当使用self[key] = value时调用，创建一个"call_function"代理对象来追踪此操作
    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_function", operator.setitem, (self, indices, values), {})

    # 当使用key in self语句检查成员资格时调用
    def __contains__(self, key):
        # 检查是否存在"_metadata"属性且不为None，如果是，则检查key是否在"_metadata"中
        if hasattr(self, "_metadata") and self._metadata is not None:
            return key in self._metadata
        # 否则，委托给超类的__contains__方法来处理key的成员资格检查
        return super().__contains__(key)
class HFAttribute(HFProxy):
    # HFAttribute 类继承自 HFProxy 类
    def __init__(self, root, attr: str):
        # 初始化方法，接受 root 和 attr 参数
        self.root = root
        self.attr = attr
        self.tracer = root.tracer  # 将 root 的 tracer 赋给实例变量 tracer
        self._node = None  # 初始化 _node 为 None

        # 如果 root 对象有 _metadata 属性，则安装对应 attr 的元数据
        if hasattr(self.root, "_metadata"):
            self.install_metadata(getattr(self.root._metadata, attr))

    @property
    def node(self):
        # node 属性，延迟加载节点，大多数情况下只有方法调用，不依赖于 getitem 调用
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", builtins.getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        # 实例可调用，创建代理对象，调用方法为 call_method
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(HFAttribute):
    # MetaDeviceAttribute 类继承自 HFAttribute 类
    pass


def _proxies_to_metas(v):
    """Returns the underlying metadata for HFProxies, and behaves like the identity for the others."""
    # 将 HFProxies 的基础元数据返回，并对其他对象行为像是返回自身
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        # 对于 torch.fx.Proxy 类型的对象，确保其有元数据，否则引发 RuntimeError
        if not (isinstance(v, HFProxy) and hasattr(v, "_metadata")):
            raise RuntimeError(f"No metadata was found for {v}")
        return v._metadata
    return v


def _gen_constructor_wrapper(target):
    # 生成构造函数的包装器，用于包装目标函数 target
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            # 检查参数中是否有 Proxy 对象
            if isinstance(v, Proxy):
                nonlocal proxy
                proxy = v

        # 对 args 和 kwargs 进行映射，检查是否有 Proxy 对象存在
        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        # 如果存在 Proxy 对象，则通过 tracer 创建代理对象，调用方式为 call_function
        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


def _generate_random_int(low: int = 10, high: int = 20, forbidden_values: Optional[List[int]] = None):
    # 生成指定范围内的随机整数，可以排除 forbidden_values 中的值
    if forbidden_values is None:
        forbidden_values = []
    value = random.randint(low, high)
    while value in forbidden_values:
        value = random.randint(low, high)
    return value


class HFTracer(Tracer):
    """
    Tracer that is able to symbolically trace models from the library. To do that, it uses the HFProxy instead of the
    regular PyTorch torch.fx.Proxy.
    """
    # HFTracer 类，用于符号化跟踪库中的模型，使用 HFProxy 而非常规的 PyTorch torch.fx.Proxy

    # 用于代理访问缓冲区值的功能标志
    proxy_buffer_attributes: bool = True
    allow_insert_stateless_mods: bool = True
    _TORCH_METHODS_TO_PATCH = [
        "arange",
        "zeros",
        "ones",
        "full",
        "full_like",
        "eye",
        "empty",
        "tensor",
        "clamp",
        "finfo",
    ]
    # 支持的架构类型，包括 PreTrainedModel 和 PeftModel（如果可用）
    supported_archs = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    # 初始化方法，用于设置自动包装的模块和函数
    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        # 调用父类的初始化方法
        super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)

        # 检查是否有可用的 Torch FX 版本
        if not is_torch_fx_available():
            # 如果没有可用的 Torch FX 版本，则抛出 ImportError 异常
            raise ImportError(
                f"Found an incompatible version of torch. Found version {get_torch_version()}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

    # 生成虚拟输入的方法
    def _generate_dummy_input(
        self, model: PreTrainedModel, input_name: str, shape: List[int], input_names: List[str]
    ):
        # 已被 PyTorch 1.13 替换为 .getattr 方法

    # 用于获取模块属性的方法
    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        # 如果禁用了模块获取属性，则直接返回属性值
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:
            # 内部函数，用于获取属性的代理
            def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
                for n, p in collection_to_search:
                    if attr_val is p:
                        # 如果尚未缓存属性的代理，则创建新的代理
                        if n not in parameter_proxy_cache:
                            kwargs = {}
                            # 如果支持参数形状常量，则创建参数代理
                            if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                                kwargs["proxy_factory_fn"] = (
                                    None
                                    if not self.param_shapes_constant
                                    else lambda node: ParameterProxy(self, node, n, attr_val)
                                )
                            # 使用代理工厂函数创建代理
                            val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                            parameter_proxy_cache[n] = val_proxy
                        return parameter_proxy_cache[n]
                return None

            # 如果属性值是 torch.nn.Parameter 类型，则尝试获取参数代理
            if isinstance(attr_val, torch.nn.Parameter):
                maybe_parameter_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_parameters(), parameter_proxy_cache
                )
                if maybe_parameter_proxy is not None:
                    return maybe_parameter_proxy

            # 如果启用了代理缓冲属性，并且属性值是 torch.Tensor 类型，则尝试获取缓冲区代理
            if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
                maybe_buffer_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_buffers(), parameter_proxy_cache
                )
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

            # 如果未找到代理，直接返回属性值
            return attr_val

    # PyTorch 1.13+ 版本所需的方法，用于获取属性
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        return self._module_getattr(attr, attr_val, parameter_proxy_cache)

    # 调用模块的方法，设置原始前向方法并调用父类方法
    def call_module(self, m, forward, args, kwargs):
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    # 返回 HFProxy 实例的方法
    def proxy(self, node):
        return HFProxy(node, self)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any]],
        concrete_args: Optional[Dict[str, Any]] = None,
        dummy_inputs: Optional[Dict[str, Any]] = None,
        complete_concrete_args_with_inputs_not_in_dummy_inputs: bool = True,
    ):
        """
        Trace method for tracing through the module hierarchy starting from `root`.

        Args:
            root (Union[torch.nn.Module, Callable[..., Any]]): The root module or callable to start tracing from.
            concrete_args (Optional[Dict[str, Any]]): Concrete arguments for the traced function.
            dummy_inputs (Optional[Dict[str, Any]]): Dummy inputs for the traced function.
            complete_concrete_args_with_inputs_not_in_dummy_inputs (bool):
                Flag indicating whether to complete concrete arguments with inputs not in dummy inputs.

        Returns:
            None
        """

    def _stateless_mod_instanciation_depends_on_proxies(self, mod: nn.Module) -> bool:
        """
        Check if the module's instantiation depends on Proxies.

        Args:
            mod (nn.Module): The module to check.

        Returns:
            bool: True if the module was instantiated with Proxies, otherwise False.
        """
        return any(isinstance(attr, Proxy) for attr in mod.__dict__.values())

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Try to insert a module that was not declared as a submodule.

        Args:
            mod (nn.Module): The module to insert.

        Returns:
            str: Path where the module was inserted as a submodule, or an empty string if insertion failed.
        """

        # If one of the module attributes is a Proxy, its instantiation is input-dependent.
        if self._stateless_mod_instanciation_depends_on_proxies(mod):
            return ""

        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        already_inserted = False

        # Check if the module is already inserted at the computed path
        while hasattr(self.root, path):
            if getattr(self.root, path) is mod:
                already_inserted = True
                break
            path = f"{mod_name}_{idx}"
            idx += 1

        # Insert the module if it's not already present
        if not already_inserted:
            self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: nn.Module) -> str:
        """
        Find the qualified name of `mod` in the Module hierarchy of `root`.

        Args:
            mod (nn.Module): The module to retrieve the qualified name for.

        Returns:
            str: Qualified path of the module in the Module hierarchy of `root`.
        """

        try:
            return super().path_of_module(mod)
        except NameError as e:
            # Handle case where `mod` is not directly found in `root`'s modules
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                return path
            raise e

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        Check if a module is a leaf module in the module hierarchy.

        Args:
            m (torch.nn.Module): The module to check.
            module_qualified_name (str): Qualified name of the module in the hierarchy.

        Returns:
            bool: True if the module is a leaf module, otherwise False.
        """

        # Check if module instantiation depends on Proxies and delegate to superclass method
        return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and super().is_leaf_module(
            m, module_qualified_name
        )

    @compatibility(is_backward_compatible=True)
    # Decorator indicating backward compatibility
    def keys(self, obj: "Proxy") -> Any:
        """Called when a proxy object has the keys() method called.
        当代理对象调用keys()方法时调用此函数。
        This is what happens when ** is called on a proxy.
        当代理对象上调用**运算符时会发生这种情况。
        This should return an iterator if ** is supposed to work in
        your custom tracer.
        如果希望在自定义的追踪器中**运算符正常工作，此方法应返回一个迭代器。
        """
        # Create an HFAttribute object for the 'keys' attribute of the proxy object
        attribute = HFAttribute(obj, "keys")()
        # Check if the target of the proxy object is '**kwargs'
        if obj.node.target == "**kwargs":
            # Return the metadata of the attribute if the target is '**kwargs'
            return attribute._metadata
        # Otherwise, return the attribute itself
        return attribute
# 获取模型的 forward 方法的参数签名
sig = inspect.signature(model.forward)

# 检查输入参数列表是否都在模型的参数签名中
if not (set(input_names) <= set(sig.parameters.keys())):
    # 如果有未在参数签名中的输入参数，生成格式化的错误信息并抛出 ValueError 异常
    formatted_input_names = input_names[0] if len(input_names) == 1 else ", ".join(input_names)
    formatted_allowed_input_names = ", ".join(sig.parameters.keys())
    raise ValueError(
        f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
        f" {formatted_allowed_input_names}"
    )

# 返回模型 forward 方法中除了输入参数之外的参数名和默认值构成的字典
return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
```