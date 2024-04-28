# `.\transformers\utils\fx.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本授权使用此文件
# 除非符合许可证规定，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入模块
import builtins
import collections
import functools
import inspect
import math
import operator
import os
import random
import warnings
from typing import Any, Callable, Dict, List, Optional, Type, Union

# 导入 torch 模块
import torch
from torch import nn
from torch.fx import Graph, GraphModule, Proxy, Tracer
from torch.fx._compatibility import compatibility
from torch.fx.proxy import ParameterProxy

# 导入 HuggingFace 模块
from .. import PretrainedConfig, PreTrainedModel, logging
from ..models.auto import get_values
from ..models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
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
from ..utils import (
    ENV_VARS_TRUE_VALUES,
    TORCH_FX_REQUIRED_VERSION,
    get_torch_version,
    is_peft_available,
    is_torch_fx_available,
)

# 如果 peft 可用，则导入 PeftModel
if is_peft_available():
    from peft import PeftModel

# 获取 logger
logger = logging.get_logger(__name__)
# 检查是否处于调试模式
_IS_IN_DEBUG_MODE = os.environ.get("FX_DEBUG_MODE", "").upper() in ENV_VARS_TRUE_VALUES

# 定义函数 _generate_supported_model_class_names
def _generate_supported_model_class_names(
    model_name: Type[PretrainedConfig],
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    task_mapping = {
        "default": MODEL_MAPPING_NAMES,  # 默认任务映射到模型名称
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING_NAMES,  # 预训练任务映射到模型名称
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,  # 下一个句子预测任务映射到模型名称
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING_NAMES,  # 掩码语言建模任务映射到模型名称
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,  # 因果语言建模任务映射到模型名称
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,  # 序列到序列因果语言建模任务映射到模型名称
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,  # 语音序列到序列任务映射到模型名称
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,  # 多项选择任务映射到模型名称
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,  # 文档问答任务映射到模型名称
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,  # 问题回答任务映射到模型名称
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,  # 序列分类任务映射到模型名称
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,  # 标记分类任务映射到模型名称
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,  # 掩码图像建模任务映射到模型名称
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,  # 图像分类任务映射到模型名称
        "zero-shot-image-classification": MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES,  # 零样本图像分类任务映射到模型名称
        "ctc": MODEL_FOR_CTC_MAPPING_NAMES,  # 连续文本分类任务映射到模型名称
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,  # 音频分类任务映射到模型名称
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,  # 语义分割任务映射到模型名称
        "backbone": MODEL_FOR_BACKBONE_MAPPING_NAMES,  # 骨干模型任务映射到模型名称
    }

    if supported_tasks is None:
        supported_tasks = task_mapping.keys()  # 如果支持的任务为空，则设置为所有任务
    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]  # 如果支持的任务是字符串，则转换为列表

    model_class_names = []
    for task in supported_tasks:
        class_name = task_mapping[task].get(model_name, None)  # 获取任务对应的模型名称
        if class_name:
            model_class_names.append(class_name)  # 如果模型名称存在，则添加到列表中

    return model_class_names  # 返回模型名称列表
# 定义支持的常规模型名称和任务列表
_REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS = [
    "altclip",
    "albert",
    "bart",
    "bert",
    "blenderbot",
    "blenderbot-small",
    "bloom",
    "clip",
    "convnext",
    "deberta",
    "deberta-v2",
    "dinov2",
    "distilbert",
    "donut-swin",
    "electra",
    "gpt2",
    "gpt_neo",
    "gptj",
    "hubert",
    "layoutlm",
    "llama",
    "lxmert",
    "m2m_100",
    "marian",
    "mbart",
    "megatron-bert",
    "mobilebert",
    "mt5",
    "nezha",
    "opt",
    "pegasus",
    "plbart",
    "resnet",
    "roberta",
    "segformer",
    "speech_to_text",
    "speech_to_text_2",
    "swin",
    "t5",
    "trocr",
    "vit",
    "xglm",
    "wav2vec2",
    #    "xlnet",
]

# 定义支持的具有键值缓存的 FX 模型列表
_FX_SUPPORTED_MODELS_WITH_KV_CACHE = ["llama", "opt"]

# 初始化常规支持的模型列表
_REGULAR_SUPPORTED_MODELS = []
for item in _REGULAR_SUPPORTED_MODEL_NAMES_AND_TASKS:
    if isinstance(item, dict):
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_class_names(**item))
    else:
        _REGULAR_SUPPORTED_MODELS.extend(_generate_supported_model_class_names(item))

# 特殊支持的模型列表
_SPECIAL_SUPPORTED_MODELS = [
    "CLIPTextModel",
    "CLIPTextModelWithProjection",
    "CLIPVisionModel",
    "CLIPVisionModelWithProjection",
    "AltCLIPTextModel",
    "AltCLIPVisionModel",
    "GitVisionModel",
    "GPT2DoubleHeadsModel",
    "Speech2Text2Decoder",
    "TrOCRDecoder",
    "PeftModelForCausalLM",
    "PeftModelForSeq2SeqLM",
    # TODO: add support for them as it should be quite easy to do so (small blocking issues).
    # XLNetForQuestionAnswering,
]

# 合并所有支持的模型列表
_SUPPORTED_MODELS = tuple(sorted(set(_REGULAR_SUPPORTED_MODELS + _SPECIAL_SUPPORTED_MODELS)))

# 定义 torch_nn_embedding 函数
def torch_nn_embedding(self, input):
    return torch.empty(*input.shape, self.weight.shape[-1], device="meta", dtype=self.weight.dtype)

# 定义 torch_nn_functional_embedding 函数
def torch_nn_functional_embedding(
    input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False
):
    return torch.empty(*input.shape, weight.shape[-1], device="meta", dtype=weight.dtype)

# 定义 torch_nn_layernorm 函数
def torch_nn_layernorm(self, input):
    return input

# 定义 torch_nn_groupnorm 函数
def torch_nn_groupnorm(self, input):
    return input

# 定义 torch_nn_linear 函数
def torch_nn_linear(self, input):
    return torch.empty(input.shape[:-1] + (self.out_features,), device="meta")

# 定义 torch_relu 函数
def torch_relu(x):
    return x

# 定义 torch_nn_relu 函数
def torch_nn_relu(self, x):
    return x

# 定义 torch_nn_functional_relu 函数
def torch_nn_functional_relu(x, inplace=False):
    if not inplace:
        raise ValueError("Don't support in-place functional.relu for MetaTensor analysis")
    return x

# 定义 torch_where 函数
def torch_where(condition, x, y):
    # torch.where 返回条件、x 和 y 的广播张量，通过使用加法来模拟
    return condition.to(device="meta") + x.to(device="meta") + y.to(device="meta")

# 定义 torch_abs 函数
def torch_abs(input, *, out=None):
    if out is not None:
        raise ValueError("Don't support in-place abs for MetaTensor analysis")
    return input

# 定义 torch_arange 函数
def torch_arange(*args, **kwargs):
    n = len(args)
    step = 1
    if n == 1:
        start = 0
        end = args[0]
    # 如果参数个数为2，则将参数分别赋值给start和end
    elif n == 2:
        start, end = args
    # 如果参数个数不为2，则将参数分别赋值给start、end和step
    else:
        start, end, step = args
    # 如果start是浮点数，则转换为整数
    if isinstance(start, float):
        start = int(start)
    # 如果end是浮点数，则转换为整数
    if isinstance(end, float):
        start = int(end)
    # 如果step是浮点数，则转换为整数
    if isinstance(step, float):
        step = int(step)
    # 获取kwargs中的step值，如果不存在则使用默认值
    step = kwargs.get("step", step)
    # 获取kwargs中的dtype值，如果不存在则使用默认值
    dtype = kwargs.get("dtype")
    # 返回一个torch张量，形状为(end - start) // step，数据类型为dtype，设备为"meta"
    return torch.empty((end - start) // step, dtype=dtype, device="meta")
# 定义一个函数，返回一个指定形状和值的张量
def torch_full(*args, **kwargs):
    # 将参数转换为列表
    args = list(args)
    # 如果第二个参数是 torch.Tensor 类型且设备是 "meta"，则将其值设为 1
    if isinstance(args[1], torch.Tensor) and args[1].device == torch.device("meta"):
        args[1] = 1  # 任意值
    # 复制 kwargs，并移除键为 "device" 的项
    kwargs_without_device = dict(kwargs)
    kwargs_without_device.pop("device", None)
    # 返回 torch.full 函数的结果
    return torch.full(*args, **kwargs_without_device)


# 定义一个函数，沿指定维度拼接张量
def torch_cat(tensors, dim=None, axis=None, *, out=None):
    # 如果 dim 和 axis 都为 None，则将 dim 设为 0
    if dim is None and axis is None:
        dim = 0
    # 如果 dim 为 None 而 axis 不为 None，则将 dim 设为 axis
    if dim is None and axis is not None:
        dim = axis
    # 如果 dim 是负数，则根据第一个张量的维度计算出正数的 dim
    if dim < 0:
        dim = tensors[0].dim() + dim
    # 计算拼接后的张量形状
    shapes = [t.shape for t in tensors]
    shape = list(shapes[0])
    concatenated_dim = sum(shape[dim] for shape in shapes)
    final_shape = shape[:dim] + [concatenated_dim] + shape[dim + 1 :]
    # 返回一个指定形状的空张量
    return torch.empty(final_shape, device="meta")


# 定义一个函数，沿指定维度堆叠张量
def torch_stack(tensors, dim=None, axis=None, *, out=None):
    # 如果 dim 和 axis 都为 None，则将 dim 设为 0
    if dim is None and axis is None:
        dim = 0
    # 如果 dim 为 None 而 axis 不为 None，则将 dim 设为 axis
    if dim is None and axis is not None:
        dim = axis
    # 如果 dim 是负数，则根据第一个张量的维度计算出正数的 dim
    if dim < 0:
        dim = tensors[0].dim() + 1 + dim
    # 计算堆叠后的张量形状
    shape = list(tensors[0].shape)
    shape.insert(dim, len(tensors))
    # 返回一个指定形状的空张量
    return torch.empty(shape, device="meta")


# 定义一个函数，对两个张量进行加法操作
def torch_add(input, other, *, alpha=1, out=None):
    # 如果 input 不是 torch.Tensor 类型，则返回一个与 other 相同形状的空张量
    if not isinstance(input, torch.Tensor):
        return torch.empty_like(other, device="meta")
    # 如果 other 不是 torch.Tensor 类型，则返回一个与 input 相同形状的空张量
    if not isinstance(other, torch.Tensor):
        return torch.empty_like(input, device="meta")
    # 计算两个张量的最大维度，并扩展维度
    max_length = max(input.dim(), other.dim())
    input_shape = list(input.shape) + [1] * (max_length - input.dim())
    other_shape = list(other.shape) + [1] * (max_length - other.dim())
    shape = []
    for i in range(max_length):
        shape.append(max(input_shape[i], other_shape[i]))
    # 返回一个指定形状的空张量
    return torch.empty(shape, device="meta")


# 定义一个函数，对两个张量进行乘法操作
def torch_mul(input, other, *, out=None):
    # 调用 torch_add 函数进行乘法操作
    return torch_add(input, other, out=out)


# 定义一个方法，用于对两个张量进行乘法操作
def torch_tensor_mul(self, other):
    # 调用 torch_mul 函数进行乘法操作
    return torch_mul(self, other)


# 定义一个函数，对两个张量进行矩阵乘法操作
def torch_matmul(input, other, *, out=None):
    # 获取两个张量的维度
    d1 = input.dim()
    d2 = other.dim()
    shape = None
    # 根据不同情况计算矩阵乘法后的形状
    if d1 == 1 and d2 == 1:
        shape = None
    elif d1 == 2 and d2 == 2:
        shape = (input.size(0), other.size(1))
    elif d1 == 1 and d2 == 2:
        shape = (other.size(1),)
    elif d1 == 2 and d1 == 1:
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
    # 如果形状为 None，则返回一个值为 0.0 的张量
    if shape is None:
        return torch.tensor(0.0, device="meta")
    # 返回一个指定形状的空张量
    return torch.empty(*shape, device="meta")
# 执行 torch_bmm 操作，计算两个矩阵的批量乘积
def torch_bmm(input, mat2, *, out=None):
    # 如果指定了输出张量，则抛出数值错误
    if out is not None:
        raise ValueError("Don't support in-place bmm for MetaTensor analysis")
    # 获取输入张量的形状信息
    batch_size, n, m = input.shape
    _, _, p = mat2.shape
    # 返回一个新的空张量，用于存储结果，设备为 "meta"
    return torch.empty(batch_size, n, p, device="meta")


# 执行 torch_baddbmm 操作，计算两个批量矩阵相加
def torch_baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None):
    # 如果指定了输出张量，则抛出数值错误
    if out is not None:
        raise ValueError("Don't support in-place baddbmm for MetaTensor analysis")
    # 调用 torch_bmm 函数计算两个矩阵的批量乘积
    return torch_bmm(batch1, batch2)


# 对张量进行重复操作
def torch_tensor_baddbmm(self, batch1, batch2, *, beta=1, alpha=1, out=None):
    return torch_baddbmm(self, batch1, batch2, beta=beta, alpha=alpha, out=out)


# 执行 torch_einsum 操作，根据方程式计算张量的乘积
def torch_einsum(equation, *operands):
    # TODO: 推断形状而不执行计算，这可能会很困难。
    # 创建一个与操作数形状相同的空张量，设备为 "cpu"
    concrete_operands = (torch.empty_like(operand, device="cpu") for operand in operands)
    # 执行 einsum 操作并将结果转移到设备 "meta"
    return torch.einsum(equation, *concrete_operands).to("meta")


# 对张量进行重复操作
def torch_tensor_repeat(self, *sizes):
    shape = list(self.shape)
    for i, x in enumerate(sizes):
        shape[i] *= x
    # 返回一个新的空张量，形状为计算后的 shape，设备为 "meta"
    return torch.empty(shape, device="meta")


# 执行 torch_repeat_interleave 操作，重复插入张量的元素
def torch_repeat_interleave(*args, dim=None, output_size=None):
    num_args = len(args)
    if num_args == 1:
        shape = [output_size if output_size is not None else args[0].sum()]
    else:
        shape = list(args[0].shape)
        if dim is None:
            if num_args > 2:
                dim = args[2]
            else:
                shape = [sum(shape)]
                dim = 0
        repeats = args[1]
        if isinstance(repeats, int) or torch.numel(repeats) == 1:
            shape[dim] *= int(repeats)
        else:
            shape[dim] = output_size if output_size is not None else repeats.sum()
    # 返回一个新的空张量，形状为计算后的 shape，设备为 "meta"
    return torch.empty(*shape, device="meta")


# 执行 torch_index_select 操作，根据索引从输入张量中选择元素
def torch_index_select(input, dim, index, *, out=None):
    shape = list(input.shape)
    shape[dim] = len(index)
    # 返回一个新的空张量，形状为计算后的 shape，设备为 "meta"
    return torch.empty(*shape, device="meta")


# 对张量进行索引选择操作
def torch_tensor_index_select(self, dim, index):
    return torch_index_select(self, dim, index)


# 执行 torch_gather 操作，根据索引从输入张量中收集元素
def torch_gather(input, dim, index, *, sparse_grad=False, out=None):
    shape = list(input.shape)
    shape[dim] = index.shape[dim]
    # 返回一个新的空张量，形状为计算后的 shape，设备为 "meta"
    return torch.empty(*shape, device="meta")


# 对张量进行收集操作
def torch_tensor_gather(self, dim, index):
    return torch_gather(self, dim, index)


# 执行 torch_roll 操作，按指定维度滚动张量
def torch_roll(input, shifts, dims=None):
    return input


# 执行 torch_flip 操作，按指定维度翻转张量
def torch_flip(input, dims):
    return input


# 对张量进行翻转操作
def torch_tensor_flip(self, dims):
    return self


# 执行 torch_nn_conv1d 操作，对输入进行一维卷积操作
def torch_nn_conv1d(self, input):
    l_in = input.shape[-1]
    shape = None
    padding = self.padding
    if padding == "valid":
        padding = (0, 0)
    if padding == "same":
        shape = list(input.shape)
    if shape is None:
        shape = list(input.shape)
        l_out = math.floor(
            (l_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        shape[-1] = l_out
    shape[-2] = self.out_channels
    # 返回一个新的空张量，形状为计算后的 shape，设备为 "meta"
    return torch.empty(shape, device="meta")
def torch_nn_conv2d(self, input):
    # 获取输入张量的高度和宽度
    h_in, w_in = input.shape[-2:]
    shape = None
    padding = self.padding
    # 如果填充方式为"valid"，则将padding设为(0, 0)
    if padding == "valid":
        padding = (0, 0)
    # 如果填充方式为"same"，则将shape设为输入张量的形状
    if padding == "same":
        shape = list(input.shape)
    # 如果shape为None，则根据公式计算输出张量的高度和宽度
    if shape is None:
        shape = list(input.shape)
        h_out = math.floor(
            (h_in + 2 * padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1
        )
        w_out = math.floor(
            (w_in + 2 * padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1
        )
        shape[-2:] = [h_out, w_out]
    shape[-3] = self.out_channels
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def torch_squeeze(input, dim=None):
    # 获取输入张量的形状
    shape = list(input.shape)
    # 如果指定了维度dim
    if dim is not None:
        # 处理负数索引
        if dim < 0:
            dim = input.dim() + dim
        # 如果指定维度的大小为1，则删除该维度
        if shape[dim] == 1:
            shape.pop(dim)
    else:
        new_shape = []
        # 遍历形状，将大小为1的维度删除
        for dim_value in shape:
            if dim_value == 1:
                continue
            new_shape.append(dim_value)
        shape = new_shape
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def torch_tensor_squeeze(self, dim=None):
    # 调用torch_squeeze函数
    return torch_squeeze(self, dim)


def torch_unsqueeze(input, dim):
    # 获取输入张量的形状
    shape = list(input.shape)
    # 处理负数索引
    if dim < 0:
        dim = input.dim() + 1 + dim
    # 在指定维度前插入大小为1的维度
    shape.insert(dim, 1)
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def torch_tensor_unsqueeze(self, dim):
    # 调用torch_unsqueeze函数
    return torch_unsqueeze(self, dim)


def torch_unique_consecutive(input, **kwargs):
    # 调用torch.unique_consecutive函数
    output = torch.unique_consecutive(torch.zeros_like(input, device="cpu"), **kwargs)
    if isinstance(output, torch.Tensor):
        return output.to("meta")
    else:
        return tuple(map(output, lambda x: x.to("meta")))


def torch_nn_functional_one_hot(tensor, num_classes=-1):
    # 如果num_classes小于0，则抛出异常
    if num_classes < 0:
        raise ValueError("Don't support automatic num_classes inference for MetaTensor analysis")
    # 构建新的形状，增加一个维度表示类别数
    shape = list(tensor.shape) + [num_classes]
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def torch_nn_functional_scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    # 获取目标长度和头维度
    target_length = query.shape[-2]
    head_dim = value.shape[-1]
    # 返回一个形状为(*query.shape[:-2], target_length, head_dim)的空张量，设备为"meta"
    return torch.empty((*query.shape[:-2], target_length, head_dim), device="meta")


def torch_nn_mseloss(self, input, target):
    # 根据reduction属性确定返回张量的形状
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def torch_nn_crossentropyloss(self, input, target):
    # 根据reduction属性确定返回张量的形状
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def torch_nn_bcewithlogitsloss(self, input, target):
    # 根据reduction属性确定返回张量的形状
    if self.reduction == "none":
        shape = target.shape
    else:
        shape = (1,)
    # 返回一个形状为shape的空张量，设备为"meta"
    return torch.empty(shape, device="meta")


def operator_getitem(a, b):
    # 将输入转换为具体的值，如果是 torch.Tensor 类型，则创建一个与输入相同形状的全为1的张量
    def to_concrete(t):
        if isinstance(t, torch.Tensor):
            concrete = torch.ones_like(t, device="cpu")
            # 如果具体值的数据类型是浮点数或整数，则将其转换为 torch.int64 类型
            if concrete.dtype in [torch.float16, torch.float32, torch.float64, torch.int32]:
                concrete = concrete.to(torch.int64)
            return concrete
        return t

    if isinstance(a, torch.Tensor):
        # 如果 a 是 torch.Tensor 类型
        # TODO: 推断形状而不执行计算。
        if isinstance(b, tuple):
            # 如果 b 是元组类型，则对元组中的每个元素应用 to_concrete 函数
            b = tuple(map(to_concrete, b))
        else:
            # 否则，将 b 转换为具体值
            b = to_concrete(b)
        # 返回 a 张量的指定元素，转换为 "meta" 设备
        return operator.getitem(torch.empty_like(a, device="cpu"), b).to("meta")
    # 如果 a 不是 torch.Tensor 类型，则返回 a 的指定元素
    return operator.getitem(a, b)
# 定义一个字典，用于手动覆盖元数据
_MANUAL_META_OVERRIDES: Dict[Callable, Callable] = {
    torch.nn.Embedding: torch_nn_embedding,  # 将 torch.nn.Embedding 映射到 torch_nn_embedding
    torch.nn.functional.embedding: torch_nn_functional_embedding,  # 将 torch.nn.functional.embedding 映射到 torch_nn_functional_embedding
    ...
}

# 创建一个 HFProxy 类，继承自 Proxy 类
class HFProxy(Proxy):
    """
    Proxy that uses metadata to handle data-dependent control-flow.
    """

    # 安装元数据的方法
    def install_metadata(self, metadata):
        self._metadata = metadata

    # 返回代理对象的形状属性
    @property
    def shape(self):
        return self.tracer.create_proxy("call_method", "size", (self,), {})

    # 返回代理对象的设备属性
    @property
    def device(self):
        # 用于跟踪设备使用情况的 hack，在元数据传播期间，将这些值替换为常量 'meta'
        return MetaDeviceAttribute(self, "device")

    # 重写 len 方法
    def __len__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return len(self._metadata)
        return super().__len__()

    # 重写 bool 方法
    def __bool__(self):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return self._metadata
        return super().__bool__()
    # 当对象的属性被访问时调用，如果属性是_metadata，则直接返回该属性的值
    def __getattr__(self, k):
        if k == "_metadata":
            return self.__getattribute__(k)
        # 注意：如果这是一个方法调用，尚未添加到图形中，我们会优化为方法调用
        return HFAttribute(self, k)

    # 当对象的索引被设置时调用，创建一个代理对象来表示调用函数operator.setitem
    def __setitem__(self, indices, values):
        return self.tracer.create_proxy("call_function", operator.setitem, (self, indices, values), {})

    # 当对象被检查是否包含某个键时调用，如果对象有_metadata属性且不为None，则检查键是否在_metadata中
    def __contains__(self, key):
        if hasattr(self, "_metadata") and self._metadata is not None:
            return key in self._metadata
        # 否则调用父类的__contains__方法
        return super().__contains__(key)
class HFAttribute(HFProxy):
    # HFAttribute 类继承自 HFProxy 类
    def __init__(self, root, attr: str):
        # 初始化方法，接受 root 和 attr 两个参数
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

        if hasattr(self.root, "_metadata"):
            # 如果 root 对象有 _metadata 属性，则调用 install_metadata 方法
            self.install_metadata(getattr(self.root._metadata, attr))

    @property
    def node(self):
        # 对属性进行惰性加载，大多数情况下只是方法调用，不依赖于 getitem 调用
        if self._node is None:
            self._node = self.tracer.create_proxy("call_function", builtins.getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        # 调用对象时创建代理
        return self.tracer.create_proxy("call_method", self.attr, (self.root,) + args, kwargs)


class MetaDeviceAttribute(HFAttribute):
    # MetaDeviceAttribute 类继承自 HFAttribute 类
    pass


def _proxies_to_metas(v):
    """Returns the underlying metadata for HFProxies, and behaves like the identity for the others."""
    # 返回 HFProxies 的基础元数据，对于其他对象则返回自身
    if isinstance(v, MetaDeviceAttribute):
        return "meta"
    if isinstance(v, torch.fx.Proxy):
        if not (isinstance(v, HFProxy) and hasattr(v, "_metadata")):
            raise RuntimeError(f"No metadata was found for {v}")
        return v._metadata
    return v


def _gen_constructor_wrapper(target):
    # 生成构造函数包装器
    @functools.wraps(target)
    def wrapper(*args, **kwargs):
        proxy = None

        def check_has_proxy(v):
            if isinstance(v, Proxy):
                nonlocal proxy
                proxy = v

        torch.fx.node.map_aggregate(args, check_has_proxy)
        torch.fx.node.map_aggregate(kwargs, check_has_proxy)

        if proxy is not None:
            return proxy.tracer.create_proxy("call_function", target, args, kwargs)
        else:
            return target(*args, **kwargs)

    return wrapper, target


def _generate_random_int(low: int = 10, high: int = 20, forbidden_values: Optional[List[int]] = None):
    # 生成指定范围内的随机整数，排除指定的值
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
    # 能够从库中符号化跟踪模型的跟踪器，使用 HFProxy 而不是常规的 PyTorch torch.fx.Proxy
    # 特性标志，用于代理对缓冲区值的访问
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
    supported_archs = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
    # 初始化函数，设置自动包装的模块和函数
    def __init__(self, autowrap_modules=(math,), autowrap_functions=()):
        # 调用父类的初始化函数，传入自动包装的模块和函数
        super().__init__(autowrap_modules=autowrap_modules, autowrap_functions=autowrap_functions)

        # 检查是否存在兼容的 torch 版本
        if not is_torch_fx_available():
            # 抛出导入错误，显示不兼容的 torch 版本
            raise ImportError(
                f"Found an incompatible version of torch. Found version {get_torch_version()}, but only version "
                f"{TORCH_FX_REQUIRED_VERSION} is supported."
            )

    # 生成虚拟输入数据
    def _generate_dummy_input(
        self, model: PreTrainedModel, input_name: str, shape: List[int]
    # 从 PyTorch 1.13 开始被 .getattr 替换
    def _module_getattr(self, attr, attr_val, parameter_proxy_cache):
        if getattr(self, "_disable_module_getattr", False):
            return attr_val
        else:

            # 获取属性值的代理
            def maybe_get_proxy_for_attr(attr_val, collection_to_search, parameter_proxy_cache):
                for n, p in collection_to_search:
                    if attr_val is p:
                        if n not in parameter_proxy_cache:
                            kwargs = {}
                            if "proxy_factory_fn" in inspect.signature(self.create_proxy).parameters:
                                kwargs["proxy_factory_fn"] = (
                                    None
                                    if not self.param_shapes_constant
                                    else lambda node: ParameterProxy(self, node, n, attr_val)
                                )
                            val_proxy = self.create_proxy("get_attr", n, (), {}, **kwargs)  # type: ignore[arg-type]
                            parameter_proxy_cache[n] = val_proxy
                        return parameter_proxy_cache[n]
                return None

            # 如果属性值是 torch.nn.Parameter 类型
            if isinstance(attr_val, torch.nn.Parameter):
                maybe_parameter_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_parameters(), parameter_proxy_cache
                )
                if maybe_parameter_proxy is not None:
                    return maybe_parameter_proxy

            # 如果启用了代理缓冲属性，并且属性值是 torch.Tensor 类型
            if self.proxy_buffer_attributes and isinstance(attr_val, torch.Tensor):
                maybe_buffer_proxy = maybe_get_proxy_for_attr(
                    attr_val, self.root.named_buffers(), parameter_proxy_cache
                )
                if maybe_buffer_proxy is not None:
                    return maybe_buffer_proxy

            return attr_val

    # 用于 PyTorch 1.13+ 版本
    def getattr(self, attr: str, attr_val: Any, parameter_proxy_cache: Dict[str, Any]):
        return self._module_getattr(attr, attr_val, parameter_proxy_cache)

    # 调用模块的函数
    def call_module(self, m, forward, args, kwargs):
        # 保存原始的 forward 函数
        self.orig_forward = forward
        return super().call_module(m, forward, args, kwargs)

    # 创建代理
    def proxy(self, node):
        return HFProxy(node, self)
    def trace(
        self,
        root: Union[torch.nn.Module, Callable[..., Any],
        concrete_args: Optional[Dict[str, Any]] = None,
        dummy_inputs: Optional[Dict[str, Any]] = None,
        complete_concrete_args_with_inputs_not_in_dummy_inputs: bool = True,
    # 定义一个方法用于跟踪模块的执行过程，包括根模块、具体参数、虚拟输入等
    def _stateless_mod_instanciation_depends_on_proxies(self, mod: nn.Module) -> bool:
        """
        Whether the module was instantiated with Proxies. If that is the case, such module cannot be a leaf module
        because its attributes are input-dependent.
        """
        # 判断模块是否使用了代理进行实例化，如果是，则模块不是叶子模块，因为其属性依赖于输入
        return any(isinstance(attr, Proxy) for attr in mod.__dict__.values())

    def _insert_module_as_submodule(self, mod: nn.Module) -> str:
        """
        Helper method which tries to insert a module that was not declared as submodule.
        """
        # 如果模块的属性中包含代理对象，则模块的实例化依赖于输入
        # 无法插入这样的模块，应该通过跟踪来处理
        if self._stateless_mod_instanciation_depends_on_proxies(mod):
            return ""
        idx = 0
        mod_name = mod.__class__.__name__.lower()
        path = f"{mod_name}_{idx}"
        already_inserted = False
        while hasattr(self.root, path):
            if getattr(self.root, path) is mod:
                already_inserted = True
                break
            path = f"{mod_name}_{idx}"
            idx += 1

        # 不需要添加多个相同模块的实例
        if not already_inserted:
            self.root.add_module(path, mod)
        return path

    def path_of_module(self, mod: nn.Module) -> str:
        """
        Helper method to find the qualified name of `mod` in the Module hierarchy of `root`. For example, if `root` has
        a submodule named `foo`, which has a submodule named `bar`, passing `bar` into this function will return the
        string "foo.bar".

        Args:
            mod (str): The `Module` to retrieve the qualified name for.
        """
        try:
            return super().path_of_module(mod)
        except NameError as e:
            if self.allow_insert_stateless_mods and len(list(mod.parameters())) == 0 and len(list(mod.buffers())) == 0:
                path = self._insert_module_as_submodule(mod)
                return path
            raise e

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (not self._stateless_mod_instanciation_depends_on_proxies(m)) and super().is_leaf_module(
            m, module_qualified_name
        )

    @compatibility(is_backward_compatible=True)
    # 定义一个方法，用于处理代理对象调用 keys() 方法的情况
    def keys(self, obj: "Proxy") -> Any:
        """Called when a proxy object is has the keys() method called.
        This is what happens when ** is called on a proxy. This should return an iterator if ** is supposed to work in
        your custom tracer.
        """
        # 创建一个 HFAttribute 对象，用于处理 obj 对象的 keys 属性
        attribute = HFAttribute(obj, "keys")()
        # 如果 obj 对象的节点目标是 "**kwargs"，则返回 attribute 对象的元数据
        if obj.node.target == "**kwargs":
            return attribute._metadata
        # 否则返回 attribute 对象
        return attribute
def get_concrete_args(model: nn.Module, input_names: List[str]):
    # 获取模型 forward 方法的参数签名
    sig = inspect.signature(model.forward)

    # 检查输入参数是否在模型参数中
    if not (set(input_names) <= set(sig.parameters.keys())):
        formatted_input_names = input_names[0] if len(input_names) == 1 else ", ".join(input_names)
        formatted_allowed_input_names = ", ".join(sig.parameters.keys())
        # 如果输入参数不在模型参数中，则抛出数值错误
        raise ValueError(
            f"The model does not have input(s) named: {formatted_input_names}, expected a subset of the following:"
            f" {formatted_allowed_input_names}"
        )

    # 返回模型参数中不在输入参数中的参数名和默认值的字典
    return {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}


def is_model_supported(model: PreTrainedModel):
    # 检查模型是否在支持的模型列表中
    return model.__class__.__name__ in _SUPPORTED_MODELS


def check_if_model_is_supported(model: PreTrainedModel):
    # 如果模型不在支持的模型列表中，则抛出未实现错误
    if not is_model_supported(model):
        supported_model_names = ", ".join(_SUPPORTED_MODELS)
        raise NotImplementedError(
            f"Model {model.__class__.__name__} is not supported yet, supported models: {supported_model_names}"
        )


def symbolic_trace(
    model: PreTrainedModel,
    input_names: Optional[List[str]] = None,
    disable_check: bool = False,
    tracer_cls: Type[HFTracer] = HFTracer,
) -> GraphModule:
    """
    Performs symbolic tracing on the model.

    Args:
        model ([`PretrainedModel`]):
            The model to trace.
        input_names (`List[str]`, *optional*):
            The names of the inputs of the traced model. If unset, model.dummy_inputs.keys() are used instead.
        disable_check (`bool`, *optional*, defaults to `False`):
            If `True`, no check is done before trying to trace the model, this is mostly usesul for debugging purposes.
        tracer_cls (`Type[HFTracer]`, *optional*, defaults to `HFTracer`):
            The tracer class to use for instantiating the tracer. If unset, `HFTracer` is used instead.

    Returns:
        `torch.fx.GraphModule`: A GraphModule constructed by recording operations seen while tracing the model.

    Example:

        ```python
        from transformers.utils.fx import symbolic_trace

        traced_model = symbolic_trace(model, input_names=["input_ids", "attention_mask", "token_type_ids"])
        ```py
    """
    # 如果未指定输入参数名，则使用模型的虚拟输入��数名
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    input_names = list(input_names)
    concrete_args = get_concrete_args(model, input_names)

    # 如果禁用检查，则跳过模型支持性检查
    if not disable_check:
        check_if_model_is_supported(model)

    # 进行追踪
    tracer = tracer_cls()
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    traced.config = model.config
    # 必须将模型类存储为属性，以允许模型反序列化，其中使用追踪，因此需要模型类
    traced.class_for_deserialization = model.__class__
    traced.device = model.device

    return traced
```