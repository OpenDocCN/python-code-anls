# `.\pytorch\benchmarks\dynamo\common.py`

```py
# 指定当前脚本使用 Python 3 运行
#!/usr/bin/env python3

# 导入必要的模块和类
from __future__ import annotations

import abc  # 导入抽象基类模块
import argparse  # 解析命令行参数的模块
import collections  # 提供额外的数据结构
import contextlib  # 提供上下文管理工具
import copy  # 提供对象的复制操作
import csv  # 读写 CSV 文件的模块
import dataclasses  # 提供用于数据类的装饰器
import functools  # 提供高阶函数：部分应用和函数包装
import importlib  # 提供实现模块的导入
import itertools  # 提供用于高效循环的工具
import logging  # 提供灵活的日志记录系统
import os  # 提供与操作系统交互的功能
import shutil  # 提供高级文件操作
import signal  # 提供与信号交互的功能
import subprocess  # 提供生成子进程的功能
import sys  # 提供与 Python 解释器交互的功能
import time  # 提供时间相关的功能
import weakref  # 提供弱引用对象的支持
from contextlib import contextmanager  # 上下文管理的装饰器
from pathlib import Path  # 提供处理文件路径的类
from typing import (  # 提供类型提示
    Any,
    Callable,
    Generator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TYPE_CHECKING,
)
from typing_extensions import Self  # 提供扩展的类型提示支持
from unittest.mock import MagicMock  # 提供创建模拟对象的支持

import numpy as np  # 数组处理和数学函数库
import pandas as pd  # 数据分析工具
import psutil  # 提供系统进程和系统利用率相关的功能
from scipy.stats import gmean, ttest_ind  # 科学计算库，提供统计函数
from tqdm.auto import tqdm, trange  # 进度条显示工具

import torch  # 深度学习框架 PyTorch
import torch._dynamo  # PyTorch 内部动态计算图库
import torch._dynamo.utils  # PyTorch 动态计算图工具函数
import torch._export  # PyTorch 导出工具
import torch.distributed  # PyTorch 分布式训练模块
import torch.multiprocessing as mp  # PyTorch 多进程支持
from torch._C import _has_cuda as HAS_CUDA, _has_xpu as HAS_XPU  # 检查是否有 CUDA 和 XPU 支持
from torch._dynamo.profiler import fx_insert_profiling, Profiler  # 动态计算图分析工具
from torch._dynamo.testing import (  # 动态计算图测试工具
    dummy_fx_compile,
    format_speedup,
    reset_rng_state,
    same,
)

try:
    from torch._dynamo.utils import (  # PyTorch 动态计算图工具函数
        clone_inputs,
        graph_break_reasons,
        maybe_enable_compiled_autograd,
    )
    from torch._inductor.utils import fresh_inductor_cache  # 导入 Inductor 缓存刷新工具函数
except ImportError:
    from _dynamo.utils import (  # 备用导入：PyTorch 动态计算图工具函数
        clone_inputs,
        graph_break_reasons,
        maybe_enable_compiled_autograd,
    )

import torch._functorch.config  # Functorch 配置模块
from torch._functorch.aot_autograd import set_model_name  # Functorch 自动微分模型命名工具
from torch._inductor import config as inductor_config, metrics  # Inductor 配置和度量模块
from torch._subclasses.fake_tensor import FakeTensorMode  # PyTorch 的假张量模式
from torch.utils import _pytree as pytree  # PyTorch 工具中的树形结构操作
from torch.utils._pytree import tree_map, tree_map_only  # 树形结构映射操作

try:
    import torch_xla  # PyTorch XLA 支持
    import torch_xla.core.xla_model as xm  # XLA 模型接口

    # 解决反向问题 https://github.com/pytorch/xla/issues/4174 的问题
    torch_xla._XLAC._init_computation_client()
except ImportError:
    # 如果未安装 torch_xla，忽略错误
    pass

if TYPE_CHECKING:
    from torch.onnx._internal.fx import diagnostics  # 导入 FX 模块的诊断工具

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象

# 主要关注 TF32 的配置
torch.backends.cuda.matmul.allow_tf32 = True

# 抑制 torch.profiler 的日志输出
os.environ["KINETO_LOG_LEVEL"] = "5"

current_name = ""  # 当前模型名称
current_device = ""  # 当前设备
current_onnx_compiler = ""  # 当前的 ONNX 编译器
current_batch_size = None  # 当前批处理大小
output_filename = None  # 输出文件名
disable_output = False  # 禁用输出标志

MAX_DOWNLOAD_ATTEMPTS = 5  # 最大下载尝试次数

# CI 配置信息
class CI(NamedTuple):
    backend: str  # 后端选择：aot_eager 或者 inductor
    training: bool  # 是否进行训练
    dynamic: bool = False  # 是否动态批次，默认为否
    device: str = "cuda"  # 使用的设备，默认为 CUDA

# CI 跳过优化器的模型列表
CI_SKIP_OPTIMIZER = {
    # TIMM 模型
    "convmixer_768_32",  # 精度问题
    "hrnet_w18",  # FX 模块的堆栈问题
    # HF 模型
    "pnasnet5large",  # FX 模块的堆栈问题
    "MobileBertForMaskedLM",  # FX 模块的堆栈问题
    "MobileBertForQuestionAnswering",  # FX 模块的堆栈问题
    "PegasusForConditionalGeneration",  # OOM 问题
}

# CI 只跳过动态批次的模型列表
CI_SKIP_DYNAMIC_BATCH_ONLY = {
    "sam",
    # "sam" model, referring to https://github.com/mindee/doctr/blob/f2114758d529ed8d3d0030581638f0520b6b98d8/doctr/models/detection/core.py#L89
    # This entry is related to a specific model iteration over a dynamic batch causing issues with Dynamo.
    # Graphbreak point for debugging purposes.
    "doctr_det_predictor",
    # "doctr_det_predictor" model
    "dlrm",
    # "dlrm" model
    "pyhpc_isoneutral_mixing",
    # "pyhpc_isoneutral_mixing" model
    "pyhpc_equation_of_state",
    # "pyhpc_equation_of_state" model
    "pyhpc_turbulent_kinetic_energy",
    # "pyhpc_turbulent_kinetic_energy" model
    "detectron2_fcos_r_50_fpn",
    # "detectron2_fcos_r_50_fpn" model
    "detectron2_fasterrcnn_r_101_c4",
    # "detectron2_fasterrcnn_r_101_c4" model
    "detectron2_fasterrcnn_r_101_dc5",
    # "detectron2_fasterrcnn_r_101_dc5" model
    "detectron2_fasterrcnn_r_101_fpn",
    # "detectron2_fasterrcnn_r_101_fpn" model
    "detectron2_fasterrcnn_r_50_c4",
    # "detectron2_fasterrcnn_r_50_c4" model
    "detectron2_fasterrcnn_r_50_dc5",
    # "detectron2_fasterrcnn_r_50_dc5" model
    "detectron2_fasterrcnn_r_50_fpn",
    # "detectron2_fasterrcnn_r_50_fpn" model
    "hf_T5_generate",
    # "hf_T5_generate" model
# 这些模型在使用 eager Adam 优化器时存在精度问题，因此在运行完整基准测试时我们使用 SGD
# 参考：https://github.com/pytorch/pytorch/issues/115966
BENCHMARK_USE_SGD = {
    # TorchBench
    "BERT_pytorch",  # BERT 模型
    "LearningToPaint",  # 学习绘画模型
    "alexnet",  # AlexNet 模型
    "dcgan",  # DCGAN 模型
    "demucs",  # Demucs 模型
    "densenet121",  # DenseNet121 模型
    "dlrm",  # DLRM 模型
    "fastNLP_Bert",  # fastNLP Bert 模型
    "mobilenet_v2",  # MobileNet V2 模型
    "phlippe_densenet",  # Philippe DenseNet 模型
    "phlippe_resnet",  # Philippe ResNet 模型
    "pytorch_stargan",  # PyTorch StarGAN 模型
    "resnet18",  # ResNet18 模型
    "shufflenet_v2_x1_0",  # ShuffleNet V2 x1.0 模型
    "speech_transformer",  # 语音 Transformer 模型
    "squeezenet1_1",  # SqueezeNet1.1 模型
    "stable_diffusion_text_encoder",  # 稳定扩散文本编码器模型
    "timm_efficientdet",  # TIMM EfficientDet 模型
    "timm_nfnet",  # TIMM NFNet 模型
    "timm_regnet",  # TIMM RegNet 模型
    "timm_vision_transformer",  # TIMM Vision Transformer 模型
    "timm_vovnet",  # TIMM VoVNet 模型
    "vgg16",  # VGG16 模型
    "hf_T5",  # HF T5 模型（动态失败 https://github.com/pytorch/pytorch/issues/115968）
    # HF
    "AlbertForMaskedLM",  # Albert For MaskedLM 模型
    "BartForCausalLM",  # Bart For CausalLM 模型
    "BartForConditionalGeneration",  # Bart For Conditional Generation 模型
    "BlenderbotSmallForCausalLM",  # Blenderbot Small For CausalLM 模型
    "BlenderbotSmallForConditionalGeneration",  # Blenderbot Small For Conditional Generation 模型
    "DebertaV2ForQuestionAnswering",  # DebertaV2 For Question Answering 模型（eager OOM）
    "ElectraForCausalLM",  # Electra For CausalLM 模型
    "M2M100ForConditionalGeneration",  # M2M100 For Conditional Generation 模型
    "MBartForCausalLM",  # MBart For CausalLM 模型
    "MBartForConditionalGeneration",  # MBart For Conditional Generation 模型
    "OPTForCausalLM",  # OPT For CausalLM 模型
    "PLBartForCausalLM",  # PLBart For CausalLM 模型
    "PLBartForConditionalGeneration",  # PLBart For Conditional Generation 模型
    "PegasusForCausalLM",  # Pegasus For CausalLM 模型
    "Speech2Text2ForCausalLM",  # Speech2Text2 For CausalLM 模型
    "TrOCRForCausalLM",  # TrOCR For CausalLM 模型
    "XGLMForCausalLM",  # XGLM For CausalLM 模型
    # TIMM
    "adv_inception_v3",  # adv_inception_v3 模型
    "botnet26t_256",  # botnet26t_256 模型
    "cait_m36_384",  # cait_m36_384 模型（OOM）
    "coat_lite_mini",  # coat_lite_mini 模型
    "convit_base",  # convit_base 模型
    "dpn107",  # dpn107 模型
    "fbnetv3_b",  # fbnetv3_b 模型
    "gernet_l",  # gernet_l 模型
    "lcnet_050",  # lcnet_050 模型
    "mixnet_l",  # mixnet_l 模型
    "res2net101_26w_4s",  # res2net101_26w_4s 模型
    "res2net50_14w_8s",  # res2net50_14w_8s 模型
    "res2next50",  # res2next50 模型
    "resnest101e",  # resnest101e 模型
    "sebotnet33ts_256",  # sebotnet33ts_256 模型
    "swsl_resnext101_32x16d",  # swsl_resnext101_32x16d 模型
    "tf_efficientnet_b0",  # tf_efficientnet_b0 模型
    "ghostnet_100",  # ghostnet_100 模型
    "gmixer_24_224",  # gmixer_24_224 模型
    "tinynet_a",  # tinynet_a 模型
}

# 这些模型在 CI 中由于 Adam 优化器状态的额外内存导致内存溢出（OOM），因此我们在 CI 中使用 SGD
CI_USE_SGD = {
    "torchrec_dlrm",  # torchrec_dlrm 模型
    "demucs",  # demucs 模型
    "detectron2_fasterrcnn_r_101_c4",  # detectron2_fasterrcnn_r_101_c4 模型
    "detectron2_fasterrcnn_r_101_dc5",  # detectron2_fasterrcnn_r_101_dc5 模型
    "detectron2_fasterrcnn_r_101_fpn",  # detectron2_fasterrcnn_r_101_fpn 模型
    "detectron2_fasterrcnn_r_50_c4",  # detectron2_fasterrcnn_r_50_c4 模型
    "detectron2_fasterrcnn_r_50_dc5",  # detectron2_fasterrcnn_r_50_dc5 模型
    "detectron2_fasterrcnn_r_50_fpn",  # detectron2_fasterrcnn_r_50_fpn 模型
    "detectron2_maskrcnn_r_101_c4",  # detectron2_maskrcnn_r_101_c4 模型
    "detectron2_maskrcnn_r_101_fpn",  # detectron2_maskrcnn_r_101_fpn 模型
    "detectron2_maskrcnn_r_50_c4",  # detectron2_maskrcnn_r_50_c4 模型
    "detectron2_maskrcnn_r_50_fpn",  # detectron2_maskrcnn_r_50_fpn 模型
    "hf_T5_base",  # HF T5 base 模型
    "hf_clip",  # HF Clip 模型
    "llama_v2_7b_16h",  # llama_v2_7b_16h 模型
    "mobilenet_v2_quantized_qat",  # mobilenet_v2_quantized_qat 模型
    "phi_1_5 resnet50_quantized_qat",  # phi_1_5 resnet50_quantized_qat 模型
    "BlenderbotForCausalLM",  # Blenderbot For CausalLM 模型
    "cait_m36_384",  # cait_m36_384 模型
    "DALLE2_pytorch",  # DALLE2 PyTorch 模型
    "moco",  # moco 模型
    "timm_efficientdet",  # TIMM EfficientDet 模型
    "ghostnet_100",  # ghostnet_100 模型
    "regnety_002",  # regnety_002 模型
    "poolformer_m36",  # poolformer_m36 模型
    "inception_v3",  # inception_v3 模型
    "tinynet_a",  # tinynet_a 模型
    "selecsls42b",  # selecsls42b 模型
    "mobilevit_s",  # mobilevit_s 模型
    "pytorch_CycleGAN_and_pix2pix",  # PyTorch CycleGAN and pix2pix 模型
    "vision_maskrcnn",  #
# 在 CI 运行中捕获 TORCH_COMPILE_DEBUG 日志，并在结果状态匹配时保存（例如，用于上传）。
CI_PRESERVE_COMPILE_DEBUG = {
    # 例如：
    # "mnasnet1_0": ["fail_accuracy"],
}


def model_specified_by_path(path_and_class_str):
    # 检查路径和类名字符串中是否包含冒号，用于判断模型是否通过路径指定
    return ":" in path_and_class_str


def load_model_from_path(path_and_class_str):
    # 解析路径和类名字符串，生成配置字典
    configs = {}
    for kvstr in path_and_class_str.split(","):
        k, v = kvstr.split(":")
        configs[k] = v

    # 检查配置中是否包含必需的"path"和"class"字段，若缺失则引发运行时错误
    for name in ["path", "class"]:
        if name not in configs:
            raise RuntimeError(
                "Invalid --only arguments. Check help message for the correct format"
            )

    # 获取模型路径和类名
    path = configs["path"]
    class_name = configs["class"]

    # 如果路径不是绝对路径，则引发运行时错误，建议使用绝对路径以避免由于工作目录变化而导致的问题
    if path[:1] != "/":
        raise RuntimeError(
            "Use absolute path since dynamo may change the current working directory which makes using relative path tricky"
        )

    # 使用指定路径加载模块并执行，获取模型类
    spec = importlib.util.spec_from_file_location("module_name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取模型类并确保其为 torch.nn.Module 的子类
    model_class = getattr(module, class_name)
    assert issubclass(model_class, torch.nn.Module)

    # 创建模型实例并确保其具有 "get_example_inputs" 方法
    model = model_class()
    assert hasattr(model, "get_example_inputs")

    # 获取模型的示例输入并返回模型及其输入
    inputs = model.get_example_inputs()
    return model, inputs


def output_csv(filename, headers, row):
    global disable_output
    # 如果全局变量 disable_output 为 True，则不进行输出操作
    if disable_output:
        return

    # 如果文件已存在，则读取现有内容到 lines 列表中，否则初始化 lines 为包含头部的列表
    if os.path.exists(filename):
        with open(filename) as fd:
            lines = list(csv.reader(fd)) or [[]]
            # 如果提供了 headers 并且长度大于 lines 的第一个元素（即头部），则更新头部
            if headers and len(headers) > len(lines[0]):
                lines[0] = headers
            else:
                headers = lines[0]
    else:
        lines = [headers]

    # 将当前行数据格式化为包含字符串形式的列表，追加到 lines 中
    lines.append([(f"{x:.6f}" if isinstance(x, float) else x) for x in row])

    # 将更新后的 lines 写入到指定的 CSV 文件中，每行结尾用 "\n" 分隔
    with open(filename, "w") as fd:
        writer = csv.writer(fd, lineterminator="\n")
        for line in lines:
            # 写入每行数据，并用 "0" 填充以确保与头部长度一致
            writer.writerow(list(line) + ["0"] * (len(headers) - len(line)))


def nothing(f):
    # 简单的函数装饰器，直接返回传入的函数 f
    return f


@functools.lru_cache(None)
def patch_torch_manual_seed():
    """Make torch manual seed deterministic. Helps with accuracy testing."""

    def deterministic_torch_manual_seed(*args, **kwargs):
        # 导入 Torch 的默认生成器并设置种子为 1337，如果支持 CUDA 或 XPU 则也设置其种子
        from torch._C import default_generator

        seed = 1337
        if HAS_CUDA:
            import torch.cuda

            if not torch.cuda._is_in_bad_fork():
                torch.cuda.manual_seed_all(seed)
        if HAS_XPU:
            import torch.xpu

            if not torch.xpu._is_in_bad_fork():
                torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    # 替换 Torch 的手动种子设置函数为 deterministic_torch_manual_seed 函数
    torch.manual_seed = deterministic_torch_manual_seed


def empty_gpu_cache(device):
    """
    Explicitly empty gpu cache to avoid OOM in subsequent run.
    """
    # 如果设备不是 "cuda" 或 "xpu" 中的一种，记录警告信息并返回
    if device not in ["cuda", "xpu"]:
        log.warning(
            "Trying to call the empty_gpu_cache for device: %s, which is not in list [cuda, xpu]",
            device,
        )
        # 返回，不执行后续的 GPU 缓存清理操作
        return
    
    # 如果设备是 "cuda"，执行清空 CUDA GPU 缓存的操作
    if device == "cuda":
        torch.cuda.empty_cache()
    # 如果设备是 "xpu"，执行清空 XPU（假设）缓存的操作
    elif device == "xpu":
        torch.xpu.empty_cache()
# 定义一个空函数 synchronize()，暂无实际功能
def synchronize():
    pass


# 汇总并去重图表中的断点，基于断点原因字符串进行排序。注意，此函数仅尽力减少日志信息，可能会因为去重而漏掉一些图表断点。
# 在需要时可以进一步优化此函数。
def summarize_graph_break(filename):
    # 构建日志文件名，去除末尾的 '.csv' 后缀，并加上 '_graph_breaks.csv'
    log_file = f"{filename.rstrip('.csv')}_graph_breaks.csv"
    
    # 如果日志文件存在，则读取为 DataFrame 对象 df
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
        
        # 根据 'reason' 列对 DataFrame 进行排序，并去除重复项
        df = df.sort_values("reason").drop_duplicates(subset="reason")
        
        # 针对多张量 SGD 进行特殊处理，因为其原因并非完全相同
        multi_tensor_sgd_row = df.loc[df["reason"].str.contains("_multi_tensor_sgd")]
        if len(multi_tensor_sgd_row):
            # 从 DataFrame 中去除所有包含 '_multi_tensor_sgd' 的行
            df = df[~df["reason"].str.contains("_multi_tensor_sgd")]
            # 将单行 multi_tensor_sgd_row 添加回 DataFrame 中
            df = pd.concat([df, pd.DataFrame([multi_tensor_sgd_row.iloc[0]])], axis=0)
        
        # 将处理后的 DataFrame 写入新的文件，去除 '.csv' 后缀并加上 '_deduped.csv'
        df.to_csv(f"{log_file.rstrip('.csv')}_deduped.csv", index=False)


# 打印文件的摘要信息，可选择是否打印 DataFrame
def print_summary(filename, print_dataframe=False):
    # 如果文件名不存在或者文件不存在，则直接返回
    if not (filename and os.path.exists(filename)):
        return
    
    # 从 CSV 文件中读取数据为 DataFrame 对象 data
    data = pd.read_csv(filename)
    
    # 如果数据中包含 'tag' 列
    if "tag" in data.columns:
        # 遍历数据中唯一的 tag
        for tag in data.tag.unique():
            # 如果 tag 为 "0.0000"，则跳过（通常表示运行失败）
            if tag == "0.0000":
                continue
            # 打印特定 tag 的摘要信息
            print(f"\nSummary for tag={tag}:")
            print_summary_table(data[data.tag == tag], print_dataframe=print_dataframe)
    else:
        # 打印整个数据集的摘要信息
        print_summary_table(data, print_dataframe=print_dataframe)
    
    # 汇总并去重图表中的断点
    summarize_graph_break(filename)


# 打印 DataFrame 的摘要信息表格，可选择是否打印 DataFrame
def print_summary_table(data, print_dataframe=False):
    # 如果需要打印 DataFrame，则设置 pandas 的显示选项
    if print_dataframe:
        pd.options.display.max_rows = 1000
        pd.options.display.max_columns = 1000
        pd.options.display.width = 2000
        print(data)
    
    # 计算列名的最大宽度
    width = max(map(len, data.columns))
    
    # 遍历 DataFrame 的每一列
    for col in data.columns:
        try:
            # 如果列名在指定的集合中，则跳过
            if col in ("dev", "name", "batch_size", "tag"):
                continue
            # 如果列名属于特定的浮点数列，则按特定格式打印
            elif col in ("pct_ops", "pct_time"):
                print(col.ljust(width), f"{data[col].mean():.3%}")
            elif col in ("graphs", "graph_calls", "captured_ops", "total_ops"):
                print(col.ljust(width), f"{data[col].mean():.3f}")
            elif col in ("compilation_latency"):
                print(col.ljust(width), f"mean={data[col].mean():.3f} seconds")
            elif col in ("compression_ratio"):
                print(col.ljust(width), f"mean={data[col].mean():.3f}x")
            # 如果列名是 'accuracy'，则计算通过率并打印
            elif col in ("accuracy"):
                pass_rate = (data[col] == "pass").mean()
                print(col.ljust(width), f"pass_rate={100*pass_rate:.2f}%")
            else:
                # 否则，对数据进行几何平均和平均值的打印
                cdata = data[col]
                print(
                    col.ljust(width),
                    f"gmean={gmean(cdata):.2f}x mean={cdata.mean():.3f}x",
                )
        except Exception as e:
            pass


# 检查张量是否在 XLA 上运行的函数
    # 定义一个名为 visit 的函数，接受一个参数 x，类型为 torch.Tensor
    def visit(x: torch.Tensor):
        # 声明 result 变量为 nonlocal，即在 visit 函数外定义的 result 变量
        nonlocal result
        # 如果 x 的设备类型是 "xla"（即 Tensor 对象在 XLA 设备上）
        if x.device.type == "xla":
            # 将 result 设置为 True
            result = True

    # 初始化 result 变量为 False
    result = False
    # 对 tensors 中的每个 torch.Tensor 对象调用 tree_map_only 函数，并将 visit 函数作为处理函数传入
    tree_map_only(torch.Tensor, visit, tensors)
    # 返回最终的 result 变量值，表示是否存在 "xla" 设备上的 Tensor 对象
    return result
def timed(
    model,
    model_iter_fn,
    example_inputs,
    times=1,
    return_result=False,
    collect_outputs=False,
):
    # 检测是否在使用XLA加速
    use_xla = tensor_is_on_xla(example_inputs)
    # 同步设备状态
    synchronize()

    # 如果正在使用XLA加速，则标记当前步骤，并等待设备操作完成
    if use_xla:
        xm.mark_step()
        xm.wait_device_ops()

    # 初始化总时间记录
    time_total = 0
    # 不收集输出以正确测量时间
    for _ in range(times):
        # 在每次迭代中调用此函数以重置随机数生成器的状态
        # 不包括 reset_rng_state() 以正确测量时间
        reset_rng_state(use_xla)
        # 记录迭代开始时间
        t_iter_begin = time.perf_counter()
        # 调用模型迭代函数，获取结果
        result = model_iter_fn(model, example_inputs, collect_outputs=collect_outputs)

        # 如果正在使用XLA加速，则标记当前步骤
        # 对于基线的torchxla模型运行，我们需要标记步骤以发送累积的图形进行编译
        # 对于使用dynamo/torchxla桥接的模型，在训练情况下，我们需要标记步骤以发送优化器图形进行编译
        if use_xla:
            xm.mark_step()
        # 记录迭代结束时间
        t_iter_end = time.perf_counter()
        # 累加迭代所用时间
        time_total += t_iter_end - t_iter_begin

    # 记录起始时间 t_0
    t_0 = time.perf_counter()
    # 如果正在使用XLA加速，则等待设备操作完成
    if use_xla:
        xm.wait_device_ops()
    # 同步设备状态
    synchronize()
    # 记录结束时间 t_1
    t_1 = time.perf_counter()
    # 累加整体时间
    time_total += t_1 - t_0
    # 如果需要返回结果，则返回总时间和结果；否则只返回总时间
    return (time_total, result) if return_result else time_total


def _normalize_bench_inputs(example_inputs) -> Tuple[Tuple[Any], Mapping[str, Any]]:
    # 对于huggingface基准测试，example_inputs格式为字典，类似 `model(**example_inputs)` 使用
    # 对于其他基准测试，example_inputs格式为元组，类似 `model(*example_inputs)` 使用
    if isinstance(example_inputs, dict):
        return (), example_inputs
    else:
        return tuple(example_inputs), {}


def _register_dataclass_output_as_pytree(example_outputs) -> None:
    # 对于huggingface基准测试，某些例子输出格式为数据类，pytree无法处理。因此在此注册pytree实现
    example_outputs_flat = pytree.tree_leaves(example_outputs)
    output_dataclass_types = [
        type(out) for out in example_outputs_flat if dataclasses.is_dataclass(type(out))
    ]
    for output_type in output_dataclass_types:
        from torch._export.utils import register_dataclass_as_pytree_node

        # 注册数据类作为pytree节点
        register_dataclass_as_pytree_node(
            output_type,
            serialized_type_name=f"{output_type.__module__}.{output_type.__name__}",
        )


class Stats:
    # 默认使用Counter的默认字典集合totals
    totals = collections.defaultdict(collections.Counter)

    @classmethod
    # 重置计数器的静态方法，用于聚合并重置计数器的值
    def reset_counters(cls):
        # 遍历 torch._dynamo.utils.counters 中的每一项，更新到 cls.totals 中对应项
        for k, v in torch._dynamo.utils.counters.items():
            cls.totals[k].update(v)
        # 获取 torch._dynamo.utils.counters 中 "frames" 的 "ok" 值
        ok = torch._dynamo.utils.counters["frames"]["ok"]
        # 获取 torch._dynamo.utils.counters 中 "frames" 的 "total" 值
        total = torch._dynamo.utils.counters["frames"]["total"]
        # 清空 torch._dynamo.utils.counters 中的所有计数器
        torch._dynamo.utils.counters.clear()
        # 返回 "ok" 和 "total" 值
        return ok, total
    
    # 打印摘要信息的静态方法
    @classmethod
    def print_summary(cls):
        # 遍历并排序 cls.totals 中的每一项
        for k, v in sorted(cls.totals.items()):
            # 将每个项的前50个最常见的条目格式化为字符串列表
            lines = "\n  ".join(map(str, v.most_common(50)))
            # 打印每个项目的统计摘要
            print(f"STATS {k}\n  {lines}")
    
    # 返回一次性编译摘要信息的静态方法
    @classmethod
    def aot_summary(cls):
        # 返回 "aot_autograd" 中 "total" 和 "ok" 的值作为列表
        return [cls.totals["aot_autograd"]["total"], cls.totals["aot_autograd"]["ok"]]
# 测试 TorchDynamo 的运算符/模型覆盖率，并记录统计信息，主要用于检查正确性
def coverage_experiment(args, model_iter_fn, model, example_inputs):
    # 创建一个性能分析器对象
    profiler = Profiler()
    # 通过 TorchDynamo 运行模型迭代函数，返回一个冻结的模型迭代函数
    frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)
    # 使用性能分析器进行性能分析
    with profiler.prof:
        frozen_model_iter_fn(model, example_inputs)
    # 获取覆盖率分析结果
    coverage_result = profiler.results()
    # 将结果输出到 coverage.csv 文件中
    output_csv(
        output_filename,
        (
            "dev",
            "name",
            "batch_size",
            "graphs",
            "graph_calls",
            "captured_ops",
            "total_ops",
            "pct_ops",
            "pct_time",
        ),
        [
            current_device,
            current_name,
            current_batch_size,
        ]
        + coverage_result.tocsv(),
    )
    # 返回覆盖率分析结果
    return coverage_result


# 测量使用 TRT 推理后端的快速度量
def speedup_experiment_fx2trt(args, model_iter_fn, model, example_inputs):
    """
    Measure speedups over eager using the trt inference backend. TRT backend is based fx graph
    generated by torch._dynamo.
    Writes to ./speedups_fx2trt.csv
    """
    # 调用 speedup_experiment 函数进行实验
    return speedup_experiment(args, model_iter_fn, model, example_inputs)


# 重新编译使用性能分析器的实验
def recompile_profiler_experiment(args, model_iter_fn, model, example_inputs):
    # 创建一个编译器性能分析器对象
    prof = torch._dynamo.utils.CompilerProfiler()
    # 优化模型迭代函数，返回优化后的模型迭代函数
    opt_model_iter_fn = torch._dynamo.optimize(prof, nopython=args.nopython)(
        model_iter_fn
    )
    # 使用优化后的模型迭代函数进行模型推断
    opt_model_iter_fn(model, example_inputs)
    # 输出编译器分析结果到 output_filename 文件中
    output_csv(
        output_filename, ["model", "profiler report"], [current_name, prof.report()]
    )
    # 获取编译器性能分析的指标
    met = prof.get_metrics()
    # 获取守卫失败的数量
    guard_failures = len(met["guard_failures"])
    # 返回守卫失败的数量列表
    return [guard_failures]


# 随机化输入数据
def randomize_input(inputs):
    if isinstance(inputs, (list, tuple)):
        # 如果输入是列表或元组，则递归随机化每个元素
        return type(inputs)([randomize_input(x) for x in inputs])
    elif isinstance(inputs, torch.Tensor):
        if inputs.dtype in (torch.float32, torch.float64):
            # 如果输入是浮点类型张量，随机生成与输入相同形状的随机数张量
            torch._dynamo.utils.counters["randomize_input"]["times"] += 1
            return torch.randn_like(inputs)
        elif inputs.dtype == torch.int64:
            # 如果输入是 int64 类型张量，则返回原始张量（不随机化）
            return inputs
        else:
            # 抛出运行时错误，说明不支持当前类型的张量随机化
            raise RuntimeError(
                f"randomize_input need support tensor of type {inputs.dtype}"
            )
    else:
        # 抛出运行时错误，说明无法处理当前类型的输入
        raise RuntimeError(
            f"randomize_input can not handle input of type {type(inputs)}"
        )


# 如果 args.trace_on_xla 为 True，则在 XLA 中标记当前步骤
def maybe_mark_step(args):
    if args.trace_on_xla:
        xm.mark_step()


# 测量使用 TRT 推理后端的快速度量
def speedup_experiment(args, model_iter_fn, model, example_inputs, **kwargs):
    """
    Measure speedups over eager.

    Writes to ./speedups.csv
    """
    timings = np.zeros((args.repeat, 2), np.float64)
    # 创建一个大小为 (args.repeat, 2) 的零数组，用于存储每次运行的时间统计

    should_randomize_input = args.randomize_input
    # 根据参数 args.randomize_input 确定是否需要对输入进行随机化

    import contextlib
    # 导入 contextlib 模块，用于创建上下文管理器

    from torch._inductor.utils import maybe_profile
    # 导入 maybe_profile 函数，用于条件性地进行性能分析

    @contextlib.contextmanager
    def maybe_mark_profile(*args, **kwargs):
        # 定义一个上下文管理器 maybe_mark_profile，用于可能的性能分析标记
        prof: torch.profiler.profile = kwargs.pop("p", None)
        mark = kwargs.pop("mark", None)
        if prof:
            with torch.profiler.record_function(mark):
                yield
        else:
            yield

    times = args.iterations_per_run
    # 设置每次运行的迭代次数为 args.iterations_per_run

    # 当在 XLA 上进行跟踪时，需要增加容差，因为 XLA 在图的大小变化时会导致数值不稳定性
    tolerance = args.xla_tolerance if args.trace_on_xla else 1e-4
    torch._dynamo.config.repro_tolerance = tolerance
    # 设置 Torch XLA 的容差值为 tolerance

    with maybe_profile(args.export_profiler_trace) as p:
        # 如果设置了 args.export_profiler_trace，则使用 maybe_profile 进行性能分析
        if args.export_aot_inductor:
            frozen_model_iter_fn = export_aot_inductor(
                model, example_inputs, args.devices[0]
            )
        else:
            frozen_model_iter_fn = torch._dynamo.run(model_iter_fn)
        # 根据参数 args.export_aot_inductor，导出或运行静态编译的模型迭代函数

        for rep in trange(args.repeat, desc="running benchmark"):
            # 对于每次重复的实验，使用 tqdm 进行进度显示
            inputs = (
                randomize_input(copy.deepcopy(example_inputs))
                if should_randomize_input
                else example_inputs
            )
            # 根据 should_randomize_input 决定是否对输入进行随机化，并生成输入数据

            maybe_mark_step(args)
            # 可能调用 mark_step 函数，用于执行计算，确保随机化输入后第一次调用的性能惩罚最小化

            with maybe_mark_profile(p=p, mark="expected"):
                # 如果设置了性能分析 p，则在该块中记录函数执行时间为 "expected"
                timings[rep, 0], expected_output = timed(
                    model,
                    model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )
                # 测量模型在期望情况下的执行时间，并获取期望的输出结果

            maybe_mark_step(args)
            # 再次调用 mark_step 函数，确保比较公平地进行性能比较

            with maybe_mark_profile(p=p, mark="actual"), maybe_enable_compiled_autograd(
                args.compiled_autograd,
                fullgraph=args.nopython,
                dynamic=args.dynamic_shapes,
            ):
                # 如果设置了性能分析 p，则在该块中记录函数执行时间为 "actual"，并根据参数条件启用编译自动微分
                timings[rep, 1], actual_output = timed(
                    model,
                    frozen_model_iter_fn,
                    inputs,
                    return_result=True,
                    times=times,
                    collect_outputs=args.collect_outputs,
                )
                # 测量模型在实际情况下的执行时间，并获取实际的输出结果
    # 如果设置了导出性能分析器跟踪数据的参数
    if args.export_profiler_trace:
        # 构造跟踪文件名，包括模型名称和可能的进程等级
        name = args.profiler_trace_name + "_" + model.name
        if hasattr(args, "rank"):
            # 如果参数中有进程等级信息，则加入文件名中
            name += f"_rank_{args.rank}"
        # 添加文件扩展名
        name += ".json"
        # 构造完整的文件路径
        name = os.path.join(torch._dynamo.config.base_dir, name)
        # 导出 Chrome 浏览器跟踪文件
        p.export_chrome_trace(name)

    # 计算时间中位数
    median = np.median(timings, axis=0)
    # 计算加速比
    speedup = median[0] / median[1]

    # 如果设置了输出原始度量参数的参数
    if args.dump_raw_metrics:
        # 保存原始时间数据到文件
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    # 设置第一行的列标题和字段值
    first_headers = ["dev", "name", "batch_size"]
    first_fields = [current_device, current_name, current_batch_size]

    # 如果传递了额外的关键字参数中含有'tag'
    if "tag" in kwargs:
        # 添加'tag'到列标题和字段值
        first_headers.append("tag")
        first_fields.append(kwargs["tag"])

    # 设置完整的列标题
    headers = first_headers + ["speedup", "abs_latency"]

    # 设置完整的行数据
    row = first_fields + [float(speedup), median[1] * 1000]

    # 设置消息字符串，用于显示加速比
    msg = f"{speedup:.3f}x"

    # 如果指定了基准参数
    if args.baseline:
        # 添加基准相关的列标题
        headers.extend(
            [
                "baseline",
                "speedup_vs_baseline",
            ]
        )
        # 从 CSV 文件中读取基准数据
        df = pd.read_csv(args.baseline)
        try:
            # 查找当前模型的基准加速比
            baseline_speedup = df[df["name"] == current_name]["speedup"].item()
            # 添加基准加速比和相对基准的加速比
            row.extend([baseline_speedup, speedup / baseline_speedup])
            # 更新显示消息
            msg = f"{baseline_speedup:.3f}x -> {speedup:.3f}x [{speedup / baseline_speedup:.3f}x]"
        except (KeyError, ZeroDivisionError):
            # 处理找不到基准数据或除以零的情况
            row.extend(
                [
                    0.0,
                    0.0,
                ]
            )

    # 如果传递了'compilation_latency'参数
    if "compilation_latency" in kwargs:
        # 添加编译延迟、压缩比率、Eager 模式峰值内存、Dynamo 模式峰值内存等列标题
        headers += [
            "compilation_latency",
            "compression_ratio",
            "eager_peak_mem",
            "dynamo_peak_mem",
        ]
        # 添加相应的行数据
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])
        row.append(kwargs["eager_peak_mem"])
        row.append(kwargs["dynamo_peak_mem"])

    # 如果传递了'cache_lookup_latency'参数
    if "cache_lookup_latency" in kwargs:
        # 添加缓存查找延迟列标题
        headers.append("cache_lookup_latency")
        # 添加相应的行数据
        row.append(kwargs["cache_lookup_latency"])

    # 如果传递了'dynamo_stats'参数
    if "dynamo_stats" in kwargs:
        # 遍历所有动态统计数据，并将其作为列标题和行数据添加到表格中
        for k, v in kwargs["dynamo_stats"].items():
            headers.append(k)
            row.append(v)

    # 输出 CSV 文件
    output_csv(
        output_filename,
        headers,
        row,
    )

    # 获取 Torch 动态工具库中的编译时间数据
    headers, data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)

    # 断言输出文件名包含'.csv'扩展名
    assert (
        output_filename.find(".csv") > 0
    ), f"expected output_filename to be a .csv, but got {output_filename}"

    # 输出编译度量指标的 CSV 文件
    output_csv(
        output_filename[:-4] + "_compilation_metrics.csv",
        first_headers + headers,
        first_fields + data,
    )

    # 返回消息字符串
    return msg
# 用于执行动态形状的实验，测量模型在不同输入下的运行时间表现
def speedup_experiment_ds(args, model_iter_fn, model, example_inputs):
    """
    Run dynamic shapes benchmarks.

    Requires dynamic shape compatible models, which provide a list of example inputs.

    Warms up using the first input example and then iterates the inputs,
    measuring (and expecting minimal) variance between the runtime for different examples.
    """

    # 初始化一个用于存储时间的数组，大小为 (重复次数, 输入示例数量, 2)，数据类型为 float64
    timings = np.zeros((args.repeat, len(example_inputs), 2), np.float64)

    # 如果重复次数大于5，打印警告信息，建议将 --repeat 设置为小于该值
    if args.repeat > 5:
        print(
            f"\ndynamic shapes experiments are slow, consider setting --repeat less than {args.repeat}\n"
        )

    # 定义预热次数
    nwarmup = 4

    # 对于每一次重复实验
    for rep in range(args.repeat):
        # 每次实验前重置 torch._dynamo 状态
        torch._dynamo.reset()
        
        # 对模型迭代函数进行优化
        optimized_model_iter_fn = optimize_ctx(model_iter_fn)
        
        # 对第一个示例输入进行预热
        for _ in range(nwarmup):
            optimized_model_iter_fn(model, example_inputs[0])

        # 对于每个输入示例及其索引
        for input_idx, inputs in enumerate(example_inputs):
            # 测量模型在原始迭代函数下的运行时间
            timings[rep, input_idx, 0] = timed(
                model, model_iter_fn, inputs, return_result=False
            )
            # 测量模型在优化后的迭代函数下的运行时间
            timings[rep, input_idx, 1] = timed(
                model, optimized_model_iter_fn, inputs, return_result=False
            )

    # 计算所有重复实验中每个输入示例的中位数
    medians = np.median(timings, axis=0)

    # 计算每个输入示例的加速比
    speedups = list(medians[:, 0] / medians[:, 1])

    # 计算加速比的平均值、中位数和方差
    speedups_mean = np.mean(speedups)
    speedups_median = np.median(speedups)
    speedups_var = np.var(speedups)

    # 获取所有输入示例的形状
    shapes = [x[0].shape for x in example_inputs]
    
    # 从形状列表中提取唯一的形状键值
    shape_keys = sorted(set(shapes))
    
    # 根据形状，将加速比分组成字典
    shape_speedups = {
        shape: [
            it[1] for it in filter(lambda it: it[0] == shape, zip(shapes, speedups))
        ]
        for shape in shape_keys
    }

    # 构建输出字符串，包括加速比的统计信息和按形状分类的加速比信息
    output_str = (
        f"mean: {speedups_mean:.3f}, median: {speedups_median:.3f}, var: {speedups_var:.3f}"
        + "\nSpeedups by shape: "
        + "\n".join(
            [
                f"{shape}: "
                + ", ".join([f"{speedup: .3g}" for speedup in shape_speedups[shape]])
                for shape in shape_keys
            ]
        )
    )

    # 输出到 CSV 文件，包括设备、名称、批次大小以及加速比的统计信息
    output_csv(
        output_filename,
        ("dev", "name", "batch_size", "speedup mean", "speedup median", "speedup var"),
        [
            current_device,
            current_name,
            current_batch_size,
            speedups_mean,
            speedups_median,
            speedups_var,
        ],
    )

    # 返回生成的输出字符串
    return output_str
    # 尝试执行以下代码块
    try:
        # 如果 iobinding 不为 None，则执行以下操作
        if iobinding is not None:
            
            # 定义一个新的同步函数 new_synchronize
            def new_synchronize():
                # 调用 iobinding 的 synchronize_inputs 方法，同步输入
                iobinding.synchronize_inputs()
                # 调用 iobinding 的 synchronize_outputs 方法，同步输出
                iobinding.synchronize_outputs()
            
            # 将 synchronize 函数指向新定义的 new_synchronize 函数
            synchronize = new_synchronize
        
        # 使用生成器语法，返回一个迭代对象
        yield
    
    # 无论是否发生异常，最终都会执行以下代码块
    finally:
        # 将 synchronize 函数恢复为之前的 prev_synchrnoize 函数（可能是之前定义的同步函数）
        synchronize = prev_synchrnoize
# 定义一个函数，用于在 ONNX 模型上进行性能加速实验
def speedup_experiment_onnx(
    args,  # 函数的参数，包含实验的各种配置和选项
    model_iter_fn,  # 函数，用于迭代模型
    onnx_model: OnnxModel,  # ONNX 模型对象
    model,  # 原始模型对象
    example_inputs,  # 示例输入数据
    **kwargs,  # 其他关键字参数
):
    """
    Measure speedups over eager.

    This function is responsible for the following:
        1. Creating iobinding with OnnxModel if device is CUDA, which is essential for perf measurement.
        2. Running ORT with OnnxModel.

    Writes to ./{output_filename}, which should be
        `Path(self.output_dir) / f"{self.compiler}_{suite}_{self.dtype}_{self.mode}_{self.device}_{self.testing}.csv".

    TODO(bowbao): Record export time and export peak memory usage.
    """

    # 初始化一个二维数组来存储时间测量结果
    timings = np.zeros((args.repeat, 2), np.float64)
    # 初始化一个布尔值表示输入数据是否正确
    is_correct = True
    # 是否随机化输入数据的标志
    should_randomize_input = args.randomize_input
    # 每次运行的迭代次数
    times = args.iterations_per_run

    # 定义一个内部函数，用于创建与 ONNX 模型绑定的输入函数和输出
    def create_onnx_input_binded_fn(onnx_model: OnnxModel, pt_inputs, example_outputs):
        # 目标是将 IO 绑定的创建移出计时器函数外
        iobinding, outputs = onnx_model.create_iobinding(pt_inputs, example_outputs)

        def onnxrt_model_iter_fn(model, inputs, collect_outputs=True):
            # 使用 IO 绑定运行 ONNX 模型
            onnx_model.run_with_iobinding(iobinding, outputs)
            if collect_outputs:
                return outputs

        return onnxrt_model_iter_fn, iobinding

    # 定义一个内部函数，用于创建 ONNX 模型的函数
    def create_onnx_fn(onnx_model: OnnxModel, pt_inputs):
        # 注意：通过将 I/O 适配部分移到外部，使性能比较更加公平
        # 1. 预先适配 `pt_inputs` 到 `onnx_inputs`
        # 2. 不需要适配输出到 `pt_outputs`。输出比较不是性能测量的一部分。
        onnx_inputs = onnx_model.adapt_pt_inputs_to_onnx(pt_inputs)

        def onnxrt_model_iter_fn(model, inputs, collect_outputs=True):
            # 使用 ONNX 输入运行 ONNX 模型
            return onnx_model.run_with_onnx_inputs(onnx_inputs)

        return onnxrt_model_iter_fn

    # 定义一个函数，用于在 ONNX 模型上进行计时操作
    def timed_onnx(model, onnx_model: OnnxModel, inputs):
        # 如果当前设备是 CPU 或者 ONNX 模型是 CPU 的话，创建 ONNX 函数，没有 IO 绑定
        if current_device == "cpu" or onnx_model.is_cpu():
            onnxrt_model_iter_fn = create_onnx_fn(onnx_model, inputs)
            iobinding = None
        else:
            # 否则，创建与输入绑定的 ONNX 函数
            onnxrt_model_iter_fn, iobinding = create_onnx_input_binded_fn(
                onnx_model, inputs, expected_output
            )
        # 使用覆盖同步 ONNX IO 绑定的上下文管理器
        with override_synchronize_with_onnx_iobinding(iobinding):
            return timed(
                model,
                onnxrt_model_iter_fn,
                inputs,
                return_result=True,
                times=times,
                collect_outputs=args.collect_outputs,
            )

    # 插入 ONNX 的预热
    inputs = (
        randomize_input(copy.deepcopy(example_inputs))
        if should_randomize_input
        else example_inputs
    )
    # 进行模型的时间计量，并获取预期输出
    _, expected_output = timed(
        model,
        model_iter_fn,
        inputs,
        return_result=True,
        times=times,
        collect_outputs=args.collect_outputs,
    )
    # 进行两次 ONNX 计时
    for _ in range(2):
        timed_onnx(model, onnx_model, inputs)
    # 复制执行 args.repeat 次循环
    for rep in range(args.repeat):
        # 如果应随机化输入，则复制并随机化 example_inputs
        inputs = (
            randomize_input(copy.deepcopy(example_inputs))
            if should_randomize_input
            else example_inputs
        )
        
        # 如果有多个 CUDA 设备
        if torch.cuda.device_count() > 1:
            # 手动设置正确的 torch.cuda.current_device，确保 torch.cuda.synchronize() 正常工作
            # 当存在多个 CUDA 设备时，第一个用于 PyTorch eager 模式，第二个用于 ONNX ORT 模式
            torch.cuda.set_device(0)
        
        # 计时模型执行时间，并返回预期的输出
        timings[rep, 0], expected_output = timed(
            model,
            model_iter_fn,
            inputs,
            return_result=True,
            times=times,
            collect_outputs=args.collect_outputs,
        )
        
        # 如果有多个 CUDA 设备
        if torch.cuda.device_count() > 1:
            # 手动设置正确的 torch.cuda.current_device，确保 torch.cuda.synchronize() 正常工作
            # 当存在多个 CUDA 设备时，第一个用于 PyTorch eager 模式，第二个用于 ONNX ORT 模式
            torch.cuda.set_device(1)
        
        # 计时 ONNX 模型执行时间，并返回实际输出
        timings[rep, 1], actual_output = timed_onnx(model, onnx_model, inputs)

    # 对 timings 的两列进行 t-检验，并获取 p 值
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    
    # 计算 timings 的中位数
    median = np.median(timings, axis=0)
    
    # 计算加速比
    speedup = median[0] / median[1]
    
    # 如果设置了输出原始指标数据的选项
    if args.dump_raw_metrics:
        # 保存 timings 到指定文件名
        np.save(
            f"{output_filename[:-4]}-raw_timings-{current_name}-{current_device}.npy",
            timings,
        )

    # 设置 CSV 文件的表头
    headers = ["dev", "name", "batch_size", "speedup", "abs_latency"]
    
    # 设置 CSV 文件的一行数据
    row = [
        current_device,
        current_name,
        current_batch_size,
        float(speedup),
        median[1] * 1000,
    ]
    
    # 如果 kwargs 中包含编译延迟数据
    if "compilation_latency" in kwargs:
        # 更新表头和数据行，添加编译延迟和压缩比数据
        headers = headers + ["compilation_latency", "compression_ratio"]
        row.append(kwargs["compilation_latency"])
        row.append(kwargs["compression_ratio"])

    # 输出 CSV 文件
    output_csv(
        output_filename,
        headers,
        row,
    )
    
    # 获取编译时间统计数据
    headers, data = torch._dynamo.utils.compile_times(repr="csv", aggregate=True)
    
    # 断言输出文件名包含 ".csv"
    assert (
        output_filename.find(".csv") > 0
    ), f"expected output_filename to be a .csv, but got {output_filename}"
    
    # 输出编译统计数据到 CSV 文件
    output_csv(
        output_filename[:-4] + "_compilation_metrics.csv",
        ["dev", "name", "batch_size"] + headers,
        [current_device, current_name, current_batch_size] + data,
    )
    
    # 返回加速比的格式化字符串
    return format_speedup(speedup, pvalue, is_correct=is_correct)
# 测量 TorchDynamo 的开销，通过仅使用 eager+FX 运行，报告与仅使用 eager 的速度提升/减慢情况
def overhead_experiment(*args, model_iter_fn):
    # 返回调用 speedup_experiment 函数的结果，其参数包括所有传入的位置参数和 model_iter_fn 函数
    return speedup_experiment(*args, model_iter_fn)


# 打印 TorchScript 的图形表示
def print_fx(gm, example_inputs):
    print(gm.graph)
    return gm


# 打印 Aten 操作的 TorchScript 表示
def print_aten_ops(gm, example_inputs):
    from functorch.compile import aot_module

    # 定义一个用于打印 TorchScript 图形表示的函数 trace_printer
    def trace_printer(gm, _):
        print(gm.graph)
        return gm

    # 使用 aot_module 函数编译 gm 模型，使用 trace_printer 作为编译前后的处理器
    return aot_module(gm, fw_compiler=trace_printer, bw_compiler=trace_printer)


# 执行所有基准实验的通用测量代码
def baselines(models, model_iter_fn, example_inputs, args):
    # 将 models 转换为列表
    models = list(models)
    # 遍历 models 列表的索引和元素
    for idx, (name, model) in enumerate(models):
        # 如果是第一个模型
        if idx == 0:
            # 对第一个模型执行 model_iter_fn 函数，记录结果到 result0
            result0 = model_iter_fn(model, example_inputs)
        # 对于非第一个模型且模型非空的情况
        elif model is not None:
            try:
                # 尝试执行 model_iter_fn 函数，记录结果到 result
                result = model_iter_fn(model, example_inputs)
                # 比较结果是否与第一个模型相同
                if same(result0, result):
                    continue
                # 如果结果不同，打印模型名和错误信息
                print(name, "is INCORRECT")
            except Exception:
                # 捕获异常，记录日志中的错误信息
                log.exception("error checking %s", name)
            # 将当前模型置为 (name, None)
            models[idx] = (name, None)
    # 初始化一个重复次数为 args.repeat、模型数量为 len(models) 的零矩阵 timings
    timings = np.zeros((args.repeat, len(models)), np.float64)
    timings.fill(1.0e10)
    # 对于重复次数 rep 中的每次重复
    for rep in range(args.repeat):
        # 对于 models 中的每个模型的索引和元素
        for idx, (name, model) in enumerate(models):
            # 如果模型不为空
            if model is not None:
                try:
                    # 计时执行 model_iter_fn 函数，记录结果到 timings 中
                    timings[rep, idx] = timed(model, model_iter_fn, example_inputs)
                except Exception:
                    # 捕获异常，继续下一个模型的计时
                    pass
    # 计算 timings[:, 0] 和 timings[:, i] 之间的 t 检验的 p 值，存储在 pvalue 中
    pvalue = [
        ttest_ind(timings[:, 0], timings[:, i]).pvalue
        for i in range(1, timings.shape[1])
    ]
    # 计算 timings 的中位数，存储在 median 中
    median = np.median(timings, axis=0)
    # 计算速度提升比例，以第一个模型的中位数除以其他模型的中位数
    speedup = median[0] / median[1:]
    # 对于 models[1:] 中的每个模型的索引和元素
    for idx, (name, model) in enumerate(models[1:]):
        # 如果模型为空，将对应速度提升设置为 0.0
        if model is None:
            speedup[idx] = 0.0
    # 格式化速度提升、p 值和模型是否存在的信息，形成结果字符串
    result = " ".join(
        [
            format_speedup(s, p, m is not None)
            for s, p, m in zip(speedup, pvalue, [m for n, m in models[1:]])
        ]
    )
    # 输出 CSV 文件，包括当前设备、模型名称和速度提升数据
    output_csv(
        output_filename,
        ("dev", "name", "batch_size") + tuple(n for n, m in models[1:]),
        [current_device, current_name, current_batch_size]
        + [f"{x:.4f}" for x in speedup],
    )
    # 返回结果字符串
    return result


# 执行 XLA 加速的实验
def xla(args, model_iter_fn, model, example_inputs):
    # 使用 xm.xla_device 创建 XLA 设备
    xla_dev = xm.xla_device(devkind=current_device)
    # 将模型 model 深度复制到 CPU，再移到 xla_dev 上
    model_xla = copy.deepcopy(model).to("cpu").to(device=xla_dev)
    # 使用 tree_map_only 将 example_inputs 中的张量转移到 xla_dev
    example_inputs_xla = tree_map_only(
        torch.Tensor, lambda x: x.to("cpu").to(device=xla_dev), example_inputs
    )
    # 对模型和模型在 XLA 设备上的执行进行三次热身
    for _ in range(3):  # warmup
        timed(model, model_iter_fn, example_inputs)
        timed(model_xla, model_iter_fn, example_inputs_xla)
    # 初始化一个重复次数为 args.repeat、结果列数为 2 的零矩阵 timings
    timings = np.zeros((args.repeat, 2), np.float64)
    timings.fill(1.0e10)
    # 对于指定的重复次数，执行以下操作
    for rep in range(args.repeat):
        # 测量未经 XLA 优化和经 XLA 优化的模型执行时间，并记录到 timings 数组中
        timings[rep, 0] = timed(model, model_iter_fn, example_inputs)
        timings[rep, 1] = timed(model_xla, model_iter_fn, example_inputs_xla)

    # 计算 timings 数组中两组数据的 t 检验 p 值
    pvalue = ttest_ind(timings[:, 0], timings[:, 1]).pvalue
    # 计算 timings 数组中每列（即不同执行方式下的时间）的中位数
    time_baseline, time_xla = np.median(timings, axis=0)
    # 计算 XLA 优化相对于基准执行时间的加速比
    speedup = time_baseline / time_xla
    # 将结果输出到指定的 CSV 文件中
    output_csv(
        output_filename,
        ("dev", "name", "batch_size", "speedup", "time_baseline", "time_xla"),
        [
            current_device,
            current_name,
            current_batch_size,
            speedup,
            time_baseline,
            time_xla,
        ],
    )
    # 返回格式化后的加速比及其统计显著性的字符串表示
    return format_speedup(speedup, pvalue)
def try_script(model, example_inputs):
    # 尝试对模型进行 Torch 脚本编译
    try:
        return torch.jit.script(model)
    except Exception:
        return None


class AOTInductorModelCache:
    # 缓存静态字典
    cache = dict()

    @classmethod
    def load(cls, model, example_inputs, device):
        import torch._inductor
        import torch.export._trace

        # 使用弱引用作为键来引用模型
        key = weakref.ref(model)
        if key not in cls.cache:
            # 规范化输入参数
            example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
            with torch.no_grad():
                # 深拷贝模型的调用结果以避免意外的副作用
                example_outputs = copy.deepcopy(model)(*example_args, **example_kwargs)

            # 注册数据类输出作为 pytree
            if pytree._is_namedtuple_instance(example_outputs):
                typ = type(example_outputs)
                pytree._register_namedtuple(
                    typ,
                    serialized_type_name=f"{typ.__module__}.{typ.__name__}",
                )
            else:
                _register_dataclass_output_as_pytree(example_outputs)

            # 导出模型到 Torch IR
            gm = torch.export._trace._export_to_torch_ir(
                model,
                example_args,
                example_kwargs,
            )
            with torch.no_grad():
                # 编译为 AOT 模型
                so_path = torch._inductor.aot_compile(
                    gm, example_args, example_kwargs
                )  # type: ignore[arg-type]

            # 加载 AOT 模型并存入缓存
            cls.cache[key] = torch._export.aot_load(so_path, device)

        return cls.cache[key]


def export(model, example_inputs):
    # 规范化输入参数
    example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
    # 调用模型获取输出
    example_outputs = model(*example_args, **example_kwargs)
    # 注册数据类输出作为 pytree
    _register_dataclass_output_as_pytree(example_outputs)

    # 导出模型
    ep = torch.export.export(model, example_args, example_kwargs)

    def opt_export(_, example_inputs):
        # 优化导出函数，重新规范化输入参数并调用导出函数
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return ep(*example_args, **example_kwargs)

    return opt_export


def export_aot_inductor(model, example_inputs, device):
    # 加载优化过的 AOT 模型
    optimized = AOTInductorModelCache.load(model, example_inputs, device)

    def opt_aot_inductor(_, example_inputs, collect_outputs=False):
        # 优化 AOT 模型导出函数，重新规范化输入参数并调用优化过的模型
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        return optimized(*example_args, **example_kwargs)

    return opt_aot_inductor


def download_retry_decorator(download_fn):
    """
    Decorator function for applying retry logic to a download function.

    The wrapped function will be called up to 5 times and raises an exception if the function fails each time.
    """
    # 下载函数的重试装饰器
    # 被装饰的函数最多重试 5 次，在每次失败后引发异常
    @functools.wraps(download_fn)
    def wrapper(self, *args, **kwargs) -> Any:
        tries = 0
        total_allowed_tries = MAX_DOWNLOAD_ATTEMPTS
        while tries <= total_allowed_tries:
            try:
                # 尝试执行下载函数，传入参数和关键字参数
                model = download_fn(self, *args, **kwargs)
                # 如果下载成功，则返回模型
                return model
            except Exception as e:
                tries += 1
                if tries <= total_allowed_tries:
                    # 计算等待时间，等待时间随尝试次数线性增加
                    wait = tries * 30
                    # 打印错误信息和重试次数，并等待一段时间后重试
                    print(
                        f"Failed to load model: {e}. Trying again ({tries}/{total_allowed_tries}) after {wait}s"
                    )
                    time.sleep(wait)
                else:
                    # 如果达到最大重试次数仍未成功，则抛出运行时错误
                    raise RuntimeError(
                        f"Failed to load model '{args}' with following error(s): {str(e)}."
                    )
        # 返回装饰后的函数
    return wrapper
    # 定义一个抽象基类 OnnxModel，继承自 abc.ABC
    class OnnxModel(abc.ABC):
        # 定义一个类变量，将 Torch 的数据类型映射到 NumPy 的数据类型
        TORCH_TO_NUMPY_DTYPE = {
            torch.float16: np.float16,
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.uint8: np.uint8,
            torch.int8: np.int8,
            torch.int16: np.int16,
            torch.int32: np.int32,
            torch.int64: np.longlong,
            torch.bool: np.bool_,
        }

        # 类变量 _COMPILER_NAME，暂时未赋值
        _COMPILER_NAME: str

        # 初始化方法，用于导出 ONNX 模型
        def __init__(
            self,
            output_directory,
            model,
            example_inputs,
            dynamic_shapes: bool,
            copy_before_export: bool = False,
            use_experimental_patch: bool = False,
        ):
            """The abstract class for exporting ONNX model.

            Args:
                output_directory: 输出路径
                model: 模型
                example_inputs: 用于导出的示例输入
                dynamic_shapes (bool): 是否导出具有动态形状的模型
                copy_before_export (bool,): 在导出之前是否复制模型。默认为 False.
                use_experimental_patch (bool): 是否应用 torch_onnx 补丁，使用 torch.export 和 onnx ir 导出。默认为 False.
            """
            model_name = current_name  # 获取当前模型名称，但未在示例中定义
            self.copy_before_export = copy_before_export  # 设置是否在导出前复制模型的标志
            self.use_experimental_patch = use_experimental_patch  # 设置是否使用实验性补丁的标志
            # 注意：torch_onnx 补丁使用 OnnxModelFromTorchScript 导出 ONNX 模型。
            if self.use_experimental_patch:
                self._COMPILER_NAME = "torch_onnx_patch"  # 如果使用实验性补丁，则设置编译器名称
            # 生成 ONNX 模型的目录路径，使用指定的输出目录、编译器名称和模型名称
            self.model_dir = self._generate_onnx_model_directory(
                output_directory, self._COMPILER_NAME, model_name
            )
            # 设置导出的 ONNX 模型文件路径
            self.model_path = str(
                self.model_dir / f"{model_name}_{self._COMPILER_NAME}.onnx"
            )

        # 内部方法，确定深拷贝目标设备
        def _determine_deepcopy_target_device(self):
            # 如果当前设备为 CPU，则目标设备也是 CPU
            if current_device == "cpu":
                target_device = "cpu"
            else:
                # 如果有多个 CUDA 设备，则选择第二个 CUDA 设备以避免内存不足
                if torch.cuda.device_count() > 1:
                    target_device = "cuda:1"
                else:
                    target_device = "cuda"  # 否则选择第一个 CUDA 设备
            return target_device

        # 方法，深拷贝模型和输入到指定设备
        def deepcopy_model_and_inputs_to_device(self, model, example_inputs, target_device):
            # 深拷贝模型以避免修改基线模型
            model_device = next(model.parameters()).device
            model.to("cpu")  # 将模型移动到 CPU
            model_copy = copy.deepcopy(model).to(target_device)  # 对模型进行深拷贝并移动到目标设备
            model.to(model_device)  # 将模型恢复到原始设备

            # 将示例输入移动到目标设备上
            target_device_example_inputs = tree_map_only(
                torch.Tensor, lambda x: x.to(device=target_device), example_inputs
            )

            return model_copy, target_device_example_inputs

        # 类方法，生成导出 ONNX 模型的目录
        @classmethod
        def _generate_onnx_model_directory(
            cls, output_directory: str, compiler_name: str, model_name: str
        ):
    ) -> Path:
        # 构建模型路径，包括输出目录、模型名称和编译器名称
        model_path = Path(
            output_directory,
            ".onnx_models",
            model_name,
            compiler_name,
        )
        # 如果模型路径已经存在且是一个目录，则递归删除该目录
        if model_path.exists() and model_path.is_dir():
            shutil.rmtree(model_path)
        # 创建模型路径，包括其所有必要的父目录，如果不存在则创建
        model_path.mkdir(parents=True, exist_ok=True)
        # 返回创建的模型路径
        return model_path

    @abc.abstractmethod
    def format_pt_inputs(self, pt_inputs: Any) -> Sequence[torch.Tensor]:
        # 抽象方法：格式化 PyTorch 输入数据，返回一个张量序列
        ...

    @abc.abstractmethod
    def format_pt_outputs(self, pt_outputs: Any) -> Sequence[torch.Tensor]:
        # 抽象方法：格式化 PyTorch 输出数据，返回一个张量序列
        ...

    def adapt_pt_inputs_to_onnx(self, pt_inputs) -> Mapping[str, np.ndarray]:
        # 将 PyTorch 输入数据适配为 ONNX 输入格式的方法
        pt_inputs = self.format_pt_inputs(pt_inputs)
        return {
            ort_input.name: pt_input.cpu().numpy()
            for ort_input, pt_input in zip(self.onnx_session.get_inputs(), pt_inputs)
        }

    def adapt_onnx_outputs_to_pt(self, onnx_outputs: List[np.ndarray]) -> Any:
        # 将 ONNX 输出数据适配为 PyTorch 格式的方法
        pt_outputs = [
            torch.from_numpy(onnx_output).to(current_device)
            for onnx_output in onnx_outputs
        ]
        if len(pt_outputs) == 1:
            return pt_outputs[0]
        return pt_outputs

    def _init_ort_session(self, model_path: str):
        # 初始化 ONNX 运行时会话的私有方法
        import onnxruntime

        if current_device == "cpu":
            ort_providers = ["CPUExecutionProvider"]
        else:
            # 使用另一块 GPU 运行 ORT 以减少内存溢出的风险
            cuda_provider_options = {
                "device_id": 1 if torch.cuda.device_count() > 1 else 0,
            }
            ort_providers = [("CUDAExecutionProvider", cuda_provider_options)]
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3  # 设置日志输出级别为 Error

        # 创建 ONNX 推理会话对象
        ort_session = onnxruntime.InferenceSession(
            self.model_path,
            providers=ort_providers,
            sess_options=session_options,
        )
        return ort_session

    def is_cpu(self) -> bool:
        # 判断当前 ONNX 会话是否使用 CPU 执行提供程序
        return self.onnx_session.get_providers()[0] == "CPUExecutionProvider"

    def cpu(self) -> Self:
        # 设置 ONNX 会话的执行提供程序为 CPU
        self.onnx_session.set_providers(["CPUExecutionProvider"])
        return self

    def create_outputs(self, *example_outputs):
        # 创建与给定示例输出张量相同形状的空张量元组
        return tuple(torch.empty_like(x) for x in example_outputs)
    # 创建输入输出绑定并返回
    def create_iobinding(self, pt_inputs, example_outputs):
        # 格式化 PyTorch 输入数据
        pt_inputs = self.format_pt_inputs(pt_inputs)
        # 格式化示例输出数据
        example_outputs = self.format_pt_outputs(example_outputs)

        # 创建 ONNX 的输入输出绑定对象
        iobinding = self.onnx_session.io_binding()
        # 对每个输入张量进行连续化处理
        args = [arg.contiguous() for arg in pt_inputs]
        # 遍历 ONNX 会话的输入和连续化处理后的输入数据
        for ort_input, arg in zip(self.onnx_session.get_inputs(), args):
            # 如果有多个 CUDA 设备，将数据移到第二个 CUDA 设备以减少内存占用
            if torch.cuda.device_count() > 1:
                arg = arg.detach().to("cuda:1")
            device = arg.device
            # 绑定输入数据到 ONNX 的输入张量
            iobinding.bind_input(
                ort_input.name,
                device.type,
                device.index or 0,
                self.TORCH_TO_NUMPY_DTYPE[arg.dtype],
                arg.size(),
                arg.data_ptr(),
            )

        # 创建输出数据
        outputs = self.create_outputs(*example_outputs)
        # 遍历 ONNX 会话的输出和创建的输出数据
        for ort_output, output in zip(self.onnx_session.get_outputs(), outputs):
            # 如果有多个 CUDA 设备，将数据移到第二个 CUDA 设备以减少内存占用
            if torch.cuda.device_count() > 1:
                output = output.detach().to("cuda:1")
            device = output.device
            # 绑定输出数据到 ONNX 的输出张量
            iobinding.bind_output(
                ort_output.name,
                device.type,
                device.index or 0,
                self.TORCH_TO_NUMPY_DTYPE[output.dtype],
                output.size(),
                output.data_ptr(),
            )
        # 返回输入输出绑定对象和输出数据
        return iobinding, outputs

    # 使用输入输出绑定运行 ONNX 会话，并返回输出数据
    def run_with_iobinding(self, iobinding, outputs):
        # 使用输入输出绑定对象运行 ONNX 会话
        self.onnx_session.run_with_iobinding(iobinding)
        # 返回输出数据
        return outputs

    # 使用 ONNX 输入数据运行 ONNX 会话，并返回结果
    def run_with_onnx_inputs(self, onnx_inputs):
        return self.onnx_session.run(None, onnx_inputs)

    # 将 NumPy 张量保存为 ONNX 的序列化格式文件
    @classmethod
    def save_tensor_data(cls, numpy_tensor, output_path):
        from onnx import numpy_helper

        # 将 NumPy 张量转换为 ONNX 的序列化格式
        proto_tensor = numpy_helper.from_array(numpy_tensor)
        # 将序列化数据写入指定路径的文件
        with open(output_path, "wb") as f:
            f.write(proto_tensor.SerializeToString())

    # 运行 ONNX 会话并序列化输入输出数据，保存为文件
    def run_and_serialize_inputs_outputs(self, pt_inputs):
        # 创建存储测试数据的目录
        test_data_dir = self.model_dir / "test_data_set_0"
        test_data_dir.mkdir(parents=True, exist_ok=True)

        # 将 PyTorch 输入数据适配为 ONNX 输入数据格式
        onnx_inputs = self.adapt_pt_inputs_to_onnx(pt_inputs)
        # 对每个 ONNX 输入数据保存为对应的输入文件
        for i, onnx_input in enumerate(onnx_inputs.values()):
            self.save_tensor_data(onnx_input, str(test_data_dir / f"input_{i}.pb"))

        # 使用 ONNX 输入数据运行 ONNX 会话，获取输出数据
        onnx_outputs = self.run_with_onnx_inputs(onnx_inputs)

        # 对每个 ONNX 输出数据保存为对应的输出文件
        for i, onnx_output in enumerate(onnx_outputs):
            self.save_tensor_data(onnx_output, str(test_data_dir / f"output_{i}.pb"))

        # 将 ONNX 输出数据适配为 PyTorch 格式并返回
        return self.adapt_onnx_outputs_to_pt(onnx_outputs)
    def run(self, pt_inputs):
        # NOTE: For CUDA performance testing, use `run_with_iobinding` to exclude memory
        # copying overhead for inputs/outputs between cpu and gpu.
        # Otherwise perf number is inaccurate.
        # 将 PyTorch 格式的输入适配为 ONNX 格式的输入
        onnx_inputs = self.adapt_pt_inputs_to_onnx(pt_inputs)
        # 使用 ONNX 格式的输入运行推理，并获取输出
        onnx_outputs = self.run_with_onnx_inputs(onnx_inputs)
        # 将 ONNX 格式的输出适配为 PyTorch 格式的输出
        return self.adapt_onnx_outputs_to_pt(onnx_outputs)
# 定义一个继承自 OnnxModel 的类，用于基于 TorchScript 导出 ONNX 模型
class OnnxModelFromTorchScript(OnnxModel):
    """TorchScript based onnx export. `torch.onnx.export`

    TODO(bowbao):
    * large model export failed.
          Onnx Model is larger than 2GB, but exporter makes decision based pt model size, which is
          smaller than 2GB.
    * OOM on slightly larger model.
          Both pt model and ort inference session are on gpu. Attempt has been made to move ORT to
          cuda:1, however ORT perf drop significantly.
          For now running everything with batch_size 1 set in launch script.
    """

    _COMPILER_NAME = "torchscript"  # 设置编译器名称为 "torchscript"

    def __init__(
        self, output_directory, model, example_inputs, dynamic_shapes: bool, **kwargs
    ):
        if dynamic_shapes:
            raise NotImplementedError("NYI dynamic shapes for OnnxModelFromTorchScript")  # 如果 dynamic_shapes 为 True，抛出未实现的异常
        super().__init__(
            output_directory, model, example_inputs, dynamic_shapes, **kwargs
        )
        # 调用父类构造函数初始化
        self._export(
            model,
            example_inputs,
            self.model_path,
            opset_version=17,
            do_constant_folding=False,
            verbose=False,
        )
        # 初始化 ONNX 会话
        self.onnx_session = self._init_ort_session(self.model_path)

    def _export(self, model, example_inputs, output_path: str, /, **kwargs) -> None:
        if self.copy_before_export:
            # 如果设置了复制模型在导出前，深拷贝模型和输入以避免对基线模型的修改
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        # 对于 huggingface 模型进行特殊处理（仅针对 kwargs）
        if isinstance(example_inputs, dict):

            class WrapperModel(torch.nn.Module):
                def __init__(self, model, keys):
                    super().__init__()
                    self.model = model
                    self.keys = keys

                def forward(self, *args):
                    return self.model(**dict(zip(self.keys, args)))

            model = WrapperModel(model, list(example_inputs.keys()))

        if self.use_experimental_patch:
            import torch_onnx

            torch_onnx.patch_torch(error_report=True, profile=True)  # 开启 Torch 的 ONNX 补丁，包含错误报告和性能分析
        else:
            # 确保补丁未生效
            try:
                import torch_onnx

                torch_onnx.unpatch_torch()  # 尝试取消 Torch 的 ONNX 补丁
            except ImportError:
                pass

        torch.onnx.export(
            model,
            self.format_pt_inputs(example_inputs),  # 格式化 PyTorch 输入
            output_path,
            **kwargs,  # 导出 ONNX 模型
        )
    # 将输入的 PyTorch 输入格式化为标准形式，以便模型接受
    def format_pt_inputs(self, pt_inputs):
        # 对于 HuggingFace 基准测试，pt_inputs 被格式化为字典，类似 `model(**pt_inputs)` 的用法。
        # 对于其他基准测试，pt_inputs 被格式化为元组，类似 `model(*pt_inputs)` 的用法。
        
        # 如果 pt_inputs 是字典，则将其转换为值的列表
        if isinstance(pt_inputs, dict):
            pt_inputs = list(pt_inputs.values())
        
        # 如果 pt_inputs 是单个的 torch.Tensor，则将其转换为包含该 tensor 的元组
        if isinstance(pt_inputs, torch.Tensor):
            pt_inputs = (pt_inputs,)
        
        # 返回一个元组，其中每个元素都是连续的（contiguous）torch.Tensor
        return tuple(arg.contiguous() for arg in pt_inputs)

    # 将 PyTorch 输出格式化为标准形式
    def format_pt_outputs(self, pt_outputs):
        # 如果 pt_outputs 是单个的 torch.Tensor，则将其转换为包含该 tensor 的元组
        if isinstance(pt_outputs, torch.Tensor):
            pt_outputs = (pt_outputs,)
        
        # 使用 pytree 将 pt_outputs 展平为一个列表
        pt_outputs = pytree.tree_leaves(pt_outputs)

        # HuggingFace 模型输出的特殊处理
        try:
            from transformers import modeling_outputs
        except ImportError:
            pass
        else:
            # 定义一个函数，用于将 modeling_outputs.ModelOutput 转换为元组
            def _to_tuple(x):
                if isinstance(x, modeling_outputs.ModelOutput):
                    return x.to_tuple()
                return x
            
            # 使用 pytree 应用 _to_tuple 函数到 pt_outputs 的每个元素
            pt_outputs = pytree.tree_map(_to_tuple, pt_outputs)
            # 再次使用 pytree 将 pt_outputs 展平为一个列表
            pt_outputs = pytree.tree_leaves(pt_outputs)
        
        # 返回格式化后的 pt_outputs
        return pt_outputs
    ) -> torch.onnx.ONNXProgram:
        # 如果设置了复制模型选项，深拷贝模型和输入以避免对基准模型的修改
        if self.copy_before_export:
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        # 规范化示例输入，分解成参数和关键字参数
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)
        
        # 配置导出选项，包括是否启用动态形状
        options = torch.onnx.ExportOptions(dynamic_shapes=self._dynamic_shapes)
        
        # 使用 torch.onnx.dynamo_export 导出模型，得到 ONNXProgram 对象
        onnx_program = torch.onnx.dynamo_export(
            model, *example_args, **example_kwargs, export_options=options
        )

        # 将导出的 ONNXProgram 对象保存到指定路径
        onnx_program.save(output_path)
        
        # 返回导出的 ONNXProgram 对象
        return onnx_program

    def format_pt_inputs(self, pt_inputs):
        # 规范化 PyTorch 输入，以适配到 ONNX 输入格式
        pt_args, pt_kwargs = _normalize_bench_inputs(pt_inputs)
        return self._onnx_program.adapt_torch_inputs_to_onnx(*pt_args, **pt_kwargs)

    def format_pt_outputs(self, pt_outputs):
        # 适配 PyTorch 输出到 ONNX 输出格式
        return self._onnx_program.adapt_torch_outputs_to_onnx(pt_outputs)
    ) -> torch.onnx.ONNXProgram:
        # 如果设置了复制模型选项，则在导出前深度复制模型，避免修改基准模型。
        model, example_inputs = self.deepcopy_model_and_inputs_to_device(
            model, example_inputs, self._determine_deepcopy_target_device()
        )

        # 规范化示例输入，返回标准化后的参数和关键字参数
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)

        # 设置导出选项，包括是否支持动态形状
        options = torch.onnx.ExportOptions(dynamic_shapes=self._dynamic_shapes)

        # 使用 torch.onnx.dynamo_export 导出模型为 ONNX 程序
        onnx_program = torch.onnx.dynamo_export(
            model, *example_args, **example_kwargs, export_options=options
        )

        # 在导出后应用 AOT 内联处理
        # 要求 onnx >= 1.15
        import onnx
        import onnx.inliner

        # 解决内联器不支持超过 2GB 模型的问题
        # 首先将模型保存到磁盘，分离外部数据，然后加载时不包含外部数据以便内联处理
        model_proto = onnx_program.model_proto
        onnx.save_model(model_proto, output_path, save_as_external_data=True)
        model_proto = onnx.load(output_path, load_external_data=False)
        model_proto = onnx.inliner.inline_local_functions(model_proto)

        # 将处理后的模型重新保存到磁盘
        onnx.save_model(model_proto, output_path)

        # 返回导出的 ONNX 程序对象
        return onnx_program
# 定义一个继承自 OnnxModelFromDynamo 的类 OnnxModelFromDynamoAotOptimize，
# 用于实现 Dynamo 和 Fx 的基于 AOT 优化的导出。使用了 torch.onnx.dynamo_export。
class OnnxModelFromDynamoAotOptimize(OnnxModelFromDynamo):
    """Dynamo and Fx based export, with AOT optimize post export. `torch.onnx.dynamo_export`."""

    # 定义了编译器的名称为 "dynamo_aot_optimize"
    _COMPILER_NAME = "dynamo_aot_optimize"

    # 定义了一个方法 _export，用于导出模型到 ONNX 格式，并返回 ONNXProgram 对象。
    def _export(
        self, model, example_inputs, output_path: str
    ) -> torch.onnx.ONNXProgram:
        # 如果设置了 copy_before_export 标志为 True，则在导出之前深拷贝模型，以避免修改基准模型。
        if self.copy_before_export:
            model, example_inputs = self.deepcopy_model_and_inputs_to_device(
                model, example_inputs, self._determine_deepcopy_target_device()
            )

        # 规范化示例输入，将其转换为 args 和 kwargs
        example_args, example_kwargs = _normalize_bench_inputs(example_inputs)

        # 设置导出选项，包括是否支持动态形状
        options = torch.onnx.ExportOptions(dynamic_shapes=self._dynamic_shapes)

        # 使用 torch.onnx.dynamo_export 导出模型，得到导出的输出结果
        export_output = torch.onnx.dynamo_export(
            model, *example_args, **example_kwargs, export_options=options
        )

        # 导入必要的库
        import onnx
        from onnxscript.rewriter.onnxruntime import rewrite

        # 重写模型协议，以便进一步处理
        model_proto = rewrite(export_output.model_proto)

        # 将模型协议保存到指定路径的 ONNX 文件中
        onnx.save_model(
            model_proto,
            output_path,
            save_as_external_data=True,  # 将数据保存为外部文件
            all_tensors_to_one_file=True,  # 将所有张量保存到一个文件中
        )

        # 返回导出的输出结果
        return export_output


class _OnnxPatch:
    # 定义了一个私有类 _OnnxPatch，用于执行 ONNX 模型的修补操作
    @classmethod
    def patch_non_tensor_outputs(cls, correct_result, new_result, fp64_outputs):
        """Patch non-tensor outputs to make them comparable with the correct result.

        ONNX model always returns a flat tuple of tensors, but the PyTorch model outputs
        `correct_result` and `fp64_outputs` can be arbitrary types. This function normalizes
        the outputs to make them comparable with the ONNX model output.
        """
        try:
            from transformers import modeling_outputs
        except ImportError:
            # 如果 transformers 模块导入失败，则标记为没有 transformers
            has_transformers = False
        else:
            # 如果成功导入 transformers 模块，则标记为有 transformers
            has_transformers = True

        if has_transformers and isinstance(
            correct_result, modeling_outputs.ModelOutput
        ):
            # 如果有 transformers 模块且 correct_result 是 ModelOutput 类型的实例
            # 转换 correct_result 和 fp64_outputs 到 tuple 形式
            correct_result = correct_result.to_tuple()
            fp64_outputs = fp64_outputs.to_tuple() if fp64_outputs is not None else None
        elif type(correct_result).__name__ in (
            "MaskedLMOutput",
            "Seq2SeqLMOutput",
            "CausalLMOutputWithCrossAttentions",
            "LongformerMaskedLMOutput",
            "Instances",
            "SquashedNormal",
            "Boxes",
            "Normal",
            "TanhTransform",
            "Foo",
            "Variable",
        ):
            # 如果 correct_result 的类型名称匹配指定的字符串
            # 从 correct_result 和 fp64_outputs 的 __dict__ 中筛选非空值组成列表
            correct_result = [
                value
                for key in correct_result.__dict__.keys()
                if (value := getattr(correct_result, key)) is not None
            ]
            fp64_outputs = (
                [
                    value
                    for key in fp64_outputs.__dict__.keys()
                    if (value := getattr(fp64_outputs, key)) is not None
                ]
                if fp64_outputs is not None
                else None
            )

        # 展开嵌套的张量元组，例如 past_key_values
        correct_result = pytree.tree_leaves(correct_result)
        # 在相同设备上处理结果，确保所有张量在同一设备上
        devices = [x.device for x in correct_result if isinstance(x, torch.Tensor)]
        assert devices and all(
            x == devices[0] for x in devices
        ), "All tensors must be on same device!"
        device = devices[0]
        # 将 new_result 中的张量移到指定设备，其它保持不变
        new_result = pytree.tree_leaves(new_result)
        new_result = pytree.tree_map(
            lambda x: x.to(device=device) if isinstance(x, torch.Tensor) else x,
            new_result,
        )
        # 展开 fp64_outputs 中的元素
        fp64_outputs = pytree.tree_leaves(fp64_outputs)

        return correct_result, new_result, fp64_outputs
@dataclasses.dataclass
class OnnxExportErrorRow:
    # 数据类，用于表示导出ONNX时出现的错误信息的行
    device: str
    model_name: str
    batch_size: int
    rule_id: Optional[str] = None
    rule_name: Optional[str] = None
    diagnostic_level: Optional[str] = None
    diagnostic_message: Optional[str] = None
    exception_type_name: Optional[str] = None
    exception_message: Optional[str] = None

    def __post_init__(self):
        # 确保 rule_id、rule_name、diagnostic_level 和 diagnostic_message 四者中至少有三者非空，
        # 或者 exception_type_name 非空
        assert (
            self.rule_id is not None
            and self.rule_name is not None
            and self.diagnostic_level is not None
            and self.diagnostic_message is not None
        ) or self.exception_type_name, (
            "Either rule_id, rule_name, diagnostic_level and diagnostic_message "
            "must be set or exception_type_name must be set"
        )

    @property
    def headers(self) -> List[str]:
        # 返回数据类的所有字段名组成的列表
        return [field.name for field in dataclasses.fields(self)]

    @property
    def row(self) -> List[str]:
        # 返回数据类的所有字段值组成的列表
        return [getattr(self, field.name) for field in dataclasses.fields(self)]


class OnnxExportErrorParser:
    def __init__(self, device: str, model_name: str, batch_size: int):
        # 初始化函数，设定设备、模型名称和批处理大小
        self.device = device
        self.model_name = model_name
        self.batch_size = batch_size

    def _qualified_exception_class_name(self, exception: Exception) -> str:
        # 返回异常对象的完全限定类名字符串
        if exception.__class__.__module__ == "builtins":
            return exception.__class__.__name__
        return f"{exception.__class__.__module__}.{exception.__class__.__name__}"

    def parse_diagnostic_context(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
    ) -> Generator[OnnxExportErrorRow, Any, Any]:
        from torch.onnx._internal.fx import diagnostics

        # 解析诊断上下文，生成导出ONNX时出现错误信息行的生成器
        for diagnostic in diagnostic_context.diagnostics:
            if diagnostic.level >= diagnostics.levels.ERROR:
                yield OnnxExportErrorRow(
                    device=self.device,
                    model_name=self.model_name,
                    batch_size=self.batch_size,
                    rule_id=diagnostic.rule.id,
                    rule_name=diagnostic.rule.name,
                    diagnostic_level=diagnostic.level.name,
                    diagnostic_message=diagnostic.message,
                )

    def parse_exception(self, exception: Exception) -> OnnxExportErrorRow:
        # 解析异常对象，返回表示导出ONNX时出现的异常信息行
        return OnnxExportErrorRow(
            device=self.device,
            model_name=self.model_name,
            batch_size=self.batch_size,
            exception_type_name=self._qualified_exception_class_name(exception),
            exception_message=str(exception),
        )


@dataclasses.dataclass
class OnnxContext:
    # 数据类，用于表示ONNX上下文信息
    onnx_model: Optional[OnnxModel] = None


def optimize_onnx_ctx(
    output_directory: str,
    onnx_model_cls: Type[OnnxModel],
    run_n_iterations: Callable,
    dynamic_shapes: bool = False,
    copy_before_export: bool = False,
    use_experimental_patch: bool = False,
) -> Callable:
    # 优化ONNX上下文的函数，返回一个可调用对象
    # 创建一个新的 OnnxContext 对象，用于管理 ONNX 运行环境
    context = OnnxContext()
    
    # 初始化一个标志，表示测试数据尚未导出
    test_data_dumped = False
    
    # 将当前的 OnnxContext 对象绑定到函数 run_n_iterations_onnx 上，使其成为其属性
    run_n_iterations_onnx.context = context
    
    # 返回函数 run_n_iterations_onnx，该函数的作用是：
    #   1. 导出并缓存模型。
    #   2. 为 ONNX 创建输入输出绑定。
    #   3. 使用 ONNX 运行 n 次迭代。
    # 缓存的模型存储在 'context' 下的返回的可调用对象中。
    return run_n_iterations_onnx
# 从文件中读取批处理大小
def read_batch_size_from_file(args, filename, model_name):
    batch_size = None  # 初始化批处理大小为None
    if os.path.exists("benchmarks"):  # 检查是否存在目录"benchmarks"
        filename = os.path.join("benchmarks", filename)  # 如果存在，将文件名与目录名拼接
    assert os.path.exists(filename), filename  # 断言文件确实存在，否则抛出 AssertionError
    with open(filename) as f:  # 打开文件
        lines = f.readlines()  # 读取文件所有行
        lines = [i.split(",") for i in lines if len(i.strip()) > 0]  # 去除空行，并将每行按逗号分隔为列表
        for val in lines:  # 遍历每个逗号分隔的值对
            cur_name, b = val  # 将当前模型名和批处理大小赋值给 cur_name 和 b
            if model_name == cur_name:  # 如果当前模型名与所需模型名相符
                batch_size = int(b)  # 将批处理大小转换为整数并赋值给 batch_size
    if batch_size is None:  # 如果批处理大小仍为None
        log.warning("Could not find batch size for %s", model_name)  # 记录警告信息，指示找不到指定模型的批处理大小
    elif batch_size == -1:  # 如果批处理大小为-1
        raise RuntimeError(  # 抛出运行时错误
            f"Batch size is unset for {model_name} in {args.batch_size_file}"
        )
    print(f"batch size: {batch_size}")  # 打印批处理大小信息
    return batch_size  # 返回批处理大小


class TimeOutException(Exception):
    pass


def alarm_handler(signum, frame):
    raise TimeOutException  # 闹钟信号处理函数，引发超时异常


def exit_after(s):
    """
    Decorator to raise TimeoutException if the fn is taking more than s seconds
    to run.
    """
    def outer(fn):
        def inner(*args, **kwargs):
            signal.signal(signal.SIGALRM, alarm_handler)  # 设置信号处理函数为 alarm_handler
            signal.alarm(s)  # 设置闹钟信号超时时间为 s 秒
            try:
                result = fn(*args, **kwargs)  # 执行函数 fn
            finally:
                signal.alarm(0)  # 关闭闹钟信号
            return result
        return inner
    return outer


def get_peak_memory():
    return torch.cuda.max_memory_allocated() / 10**9  # 获取当前 CUDA 设备上的最大内存占用量（GB为单位）


def null_experiment(args, model_iter_fn, model, example_inputs):
    """
    A no-op experiment useful for making sure TorchBenchark alone works properly.
    """
    return []  # 返回空列表，用于模拟没有操作的实验


def cast_to(dtype, model, inputs):
    # 将模型和输入转换为指定的数据类型
    if dtype == torch.float16:  # 如果指定类型为 torch.float16
        model = model.half()  # 将模型转换为半精度浮点数类型
    else:
        model = model.to(dtype)  # 将模型转换为指定的数据类型

    inputs = tree_map(
        lambda x: x.to(dtype)  # 将输入中的浮点数类型张量转换为指定的数据类型
        if isinstance(x, torch.Tensor) and x.is_floating_point()  # 如果 x 是浮点数类型的张量
        else x,
        inputs,
    )
    return model, inputs  # 返回转换后的模型和输入


def cast_to_bf16(model, inputs):
    return cast_to(torch.bfloat16, model, inputs)  # 将模型和输入转换为 torch.bfloat16 类型


def cast_to_fp16(model, inputs):
    return cast_to(torch.float16, model, inputs)  # 将模型和输入转换为 torch.float16 类型


def cast_to_fp64(model, inputs):
    return cast_to(torch.float64, model, inputs)  # 将模型和输入转换为 torch.float64 类型


def cast_to_fp32(model, inputs):
    return cast_to(torch.float32, model, inputs)  # 将模型和输入转换为 torch.float32 类型


class DummyGradScaler:
    def scale(self, loss):
        return loss  # 返回未经任何操作的损失值


def get_dynamo_stats():
    # TODO: consider deepcopy'ing the entire counters struct and
    # adding a helper to do subtraction on it
    pass  # 获取 Dynamo 统计信息的函数，目前未实现详细功能
    # 返回一个包含统计信息的 Counter 对象，统计项如下：
    return collections.Counter(
        {
            # 统计捕获的函数调用数
            "calls_captured": torch._dynamo.utils.counters["stats"]["calls_captured"],
            # 统计唯一图的数量
            "unique_graphs": torch._dynamo.utils.counters["stats"]["unique_graphs"],
            # 统计所有图中的断点总数
            "graph_breaks": sum(torch._dynamo.utils.counters["graph_break"].values()),
            # 注意事项：加号会移除零计数项，统计非零计数的唯一断点数
            "unique_graph_breaks": len(+torch._dynamo.utils.counters["graph_break"]),
            # 统计编译的自动微分捕获数
            "autograd_captures": torch._dynamo.utils.counters["compiled_autograd"]["captures"],
            # 统计编译的自动微分编译数
            "autograd_compiles": torch._dynamo.utils.counters["compiled_autograd"]["compiles"],
            # 统计 CUDA 图算法器跳过的数量
            "cudagraph_skips": torch._dynamo.utils.counters["inductor"]["cudagraph_skips"],
        }
    )
@contextmanager
def maybe_init_distributed(should_init_distributed, rank, world_size, port="6789"):
    try:
        # 如果需要初始化分布式，设置当前 CUDA 设备
        if should_init_distributed:
            torch.cuda.set_device(rank)
            # 设置主地址和端口号
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = port
            # 初始化分布式进程组
            torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        # 执行代码块
        yield
    finally:
        # 如果需要初始化分布式，销毁分布式进程组
        if should_init_distributed:
            torch.distributed.destroy_process_group()


@contextmanager
def maybe_snapshot_memory(should_snapshot_memory, suffix):
    # 启用内存快照工具进行深度内存分析：
    # https://pytorch.org/blog/understanding-gpu-memory-1/
    try:
        # 如果需要快照内存
        if should_snapshot_memory:
            # 记录内存历史记录
            torch.cuda.memory._record_memory_history(max_entries=100000)
        # 执行代码块
        yield
    finally:
        # 如果需要快照内存
        if should_snapshot_memory:
            try:
                # 尝试保存内存快照
                torch.cuda.memory._dump_snapshot(
                    os.path.join(
                        torch._dynamo.config.base_dir,
                        f"{output_filename.rstrip('.csv')}_{suffix}.pickle",
                    )
                )
            except Exception as e:
                # 如果保存快照失败，记录错误日志
                logging.error("Failed to save memory snapshot, %s", e)

            # 停止记录内存历史记录
            torch.cuda.memory._record_memory_history(enabled=None)


class BenchmarkRunner:
    def __init__(self):
        self.model_iter_fn = None  # 模型迭代函数
        self.grad_scaler = DummyGradScaler()  # 梯度缩放器
        self.autocast = contextlib.nullcontext  # 自动类型转换上下文管理器
        self.autocast_arg = {}  # 自动类型转换参数
        self.optimizer = None  # 优化器
        self._args = None  # 内部参数
    def setup_amp(self, current_device=None):
        if self.args.only in self.fp32_only_models:
            return

        devices = [current_device] if current_device else self.args.devices
        if self.args.amp:
            # 如果设置了自动混合精度（AMP），则执行以下操作
            # AMP 训练可能导致损失值很小，可能会导致梯度值为零。为了解决这个问题，
            # PyTorch 引入了 GradScaler。GradScaler 是一个状态结构，用于缩放损失值以防止下溢。
            # 在训练开始时损失值通常较大（因此不需要缩放），随着网络的改进，损失值倾向于变小（需要缩放）。
            # GradScaler 管理所有这些微调，检查梯度是否变为 inf，并丢弃这样的批次。

            # 由于我们不运行长迭代，init_scale 的默认值 65536 会将所有梯度变为 inf。
            # 因此，我们只使用 init_scale 为 2.0 用于基准测试目的。

            # 禁用 Gradscaler 的原因是：
            #  1）基准设置运行 fwd-bwd 的 2 次迭代。因此不实用。
            #  2）当前设置共享 grad_scaler 用于 eager 和 dynamo 模型，这是不好的，
            #     因为 Gradscaler 有状态，并且可以调整 eager 和 dynamo 运行之间的缩放因子，
            #     使得准确性检查更加困难。
            # self.grad_scaler = torch.amp.GradScaler(device="cuda", init_scale=2.0)

            # 设置自动混合精度（AMP）的自动转换上下文
            self.autocast = functools.partial(
                torch.amp.autocast, device_type=devices[0]
            )
            
            # 如果指定了自动混合精度的数据类型，则根据参数设置 amp_dtype
            if self.args.amp_dtype:
                amp_dtype = (
                    torch.float16
                    if self.args.amp_dtype == "float16"
                    else torch.bfloat16
                )
                self.autocast_arg["dtype"] = amp_dtype
    # 初始化优化器函数，根据设备和名称选择适当的优化器并设置参数
    def init_optimizer(self, name, device, params):
        # 如果设备是 "cuda"，且处于训练模式下，并且优化器名称不在 CI_SKIP_OPTIMIZER 中
        if device == "cuda" and self.args.training and name not in CI_SKIP_OPTIMIZER:
            # 如果满足条件，使用 SGD 优化器
            if (name in CI_USE_SGD and self.args.ci) or name in BENCHMARK_USE_SGD:
                # 创建 SGD 优化器对象，设置学习率为 0.01，并启用 foreach 参数
                self.optimizer = torch.optim.SGD(params, lr=0.01, foreach=True)
                # 禁用 multi_tensor_sgd 以进行基准测试，因为编译此优化器没有显著的性能提升（约1%），
                # 这是因为它只是一个单个 foreach 操作，增加了编译时间。
                # 在自动调整和虚拟张量缓存生效后，我们可以启用此功能，因为编译时间影响将会降低。
                # 虚拟张量缓存: https://github.com/pytorch/pytorch/pull/113873
                # 自动调整: https://github.com/pytorch/pytorch/issues/117447
                self.optimizer.step = torch._dynamo.disable(self.optimizer.step)
            else:
                # 否则，创建 Adam 优化器对象，设置学习率为 0.01，并启用 capturable 和 foreach 参数
                self.optimizer = torch.optim.Adam(
                    params, lr=0.01, capturable=True, foreach=True
                )
        else:
            # 如果不满足上述条件，将优化器设为 None
            self.optimizer = None

    # 获取参数对象 _args 的方法
    @property
    def args(self):
        return self._args

    # 设置参数对象 _args 的方法
    @args.setter
    def args(self, args):
        self._args = args

    # 返回空集合的方法，用于标记要跳过的模型
    @property
    def skip_models(self):
        return set()

    # 返回空集合的方法，用于标记要在 cuda 设备上跳过的模型
    @property
    def skip_models_for_cuda(self):
        return set()

    # 返回空集合的方法，用于标记要在 cpu 设备上跳过的模型
    @property
    def skip_models_for_cpu(self):
        return set()

    # 返回空集合的方法，用于标记要在冻结模型时跳过的模型
    @property
    def skip_models_for_freezing(self):
        return set()

    # 返回空集合的方法，用于标记要跳过的慢速模型
    @property
    def slow_models(self):
        return set()

    # 返回空集合的方法，用于标记要跳过的非常慢速模型
    @property
    def very_slow_models(self):
        return set()

    # 返回空集合的方法，用于标记要跳过的非确定性模型
    @property
    def non_deterministic_models(self):
        return set()

    # 返回空集合的方法，用于标记只允许单精度浮点数的模型
    @property
    def fp32_only_models(self):
        return set()

    # 返回空集合的方法，用于标记要强制使用 AMP 对 FP16/BF16 模型进行训练的模型
    @property
    def force_amp_for_fp16_bf16_models(self):
        return set()

    # 返回空集合的方法，用于标记要强制使用 FP16 对 BF16 模型进行训练的模型
    @property
    def force_fp16_for_bf16_models(self):
        return set()

    # 返回空集合的方法，用于标记不适合训练的模型
    @property
    def skip_not_suitable_for_training_models(self):
        return set()

    # 返回空集合的方法，用于标记因 Torchinductor 失败而跳过的模型
    @property
    def failing_torchinductor_models(self):
        return set()

    # 返回空集合的方法，用于标记因 FX2TRT 失败而跳过的模型
    @property
    def failing_fx2trt_models(self):
        return set()

    # 返回空集合的方法，用于标记在大型模型仪表板上跳过精度检查的模型
    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        return set()

    # 返回空集合的方法，用于标记在急切非确定性模式下跳过精度检查的模型
    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        return set()

    # 返回空集合的方法，用于标记要跳过多进程模型
    @property
    def skip_multiprocess_models(self):
        return set()

    # 返回空集合的方法，用于标记因控制流问题而跳过的模型
    @property
    def skip_models_due_to_control_flow(self):
        return set()

    # 返回空集合的方法，用于标记在 NN 模块上要保护的模型
    @property
    def guard_on_nn_module_models(self):
        return set()

    # 返回空集合的方法，用于标记内置 NN 模块中要内联的模型
    @property
    def inline_inbuilt_nn_modules_models(self):
        return set()

    # 获取容忍度和余弦标志的方法，需要在子类中实现
    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        raise NotImplementedError

    # 返回布尔值 equal_nan 的方法，根据 args 中的 float32 属性决定是否设置为 False
    @property
    def equal_nan(self):
        equal_nan = True
        if self.args.float32:
            equal_nan = False
        return equal_nan
    def iter_models(self, args):
        """
        Generator function that yields models loaded on specified devices.
        
        Args:
            args: Command-line arguments containing device information and batch size.

        Yields:
            Loaded model instances for each model name on each specified device.
        """
        for model_name in self.iter_model_names(args):
            for device in args.devices:
                try:
                    yield self.load_model(
                        device,
                        model_name,
                        batch_size=args.batch_size,
                    )
                except NotImplementedError:
                    continue  # bad benchmark implementation

    def deepcopy_model(self, model):
        """
        Create a deep copy of the given model.

        Args:
            model: The model to be copied.

        Returns:
            A deep copy of the model.
        """
        return copy.deepcopy(model)

    def cast_based_on_args(self, model, example_inputs):
        """
        Casts the model and example inputs to the desired precision based on arguments.

        Args:
            model: The model to cast.
            example_inputs: Example inputs for the model.

        Returns:
            Tuple containing the casted model and example inputs.
        """
        if self.args.float32 or self.args.only in self.fp32_only_models:
            if not self.args.float32:
                log.warning("Model %s supports float32 only", self.args.only)
            model, example_inputs = cast_to_fp32(model, example_inputs)
        elif self.args.float16:
            if self.args.only in self.force_amp_for_fp16_bf16_models:
                log.warning(
                    "Model %s does not support float16, running with amp instead",
                    self.args.only,
                )
                self.args.amp = True
                self.setup_amp()
            else:
                model, example_inputs = cast_to_fp16(model, example_inputs)
        elif self.args.bfloat16:
            if self.args.only in self.force_amp_for_fp16_bf16_models:
                log.warning(
                    "Model %s does not support bfloat16, running with amp instead",
                    self.args.only,
                )
                self.args.amp = True
                self.setup_amp()
            elif self.args.only in self.force_fp16_for_bf16_models:
                log.warning(
                    "Model %s does not support bfloat16, running with float16 instead",
                    self.args.only,
                )
                model, example_inputs = cast_to_fp16(model, example_inputs)
            else:
                model, example_inputs = cast_to_bf16(model, example_inputs)

        return model, example_inputs

    def validate_model(self, model, example_inputs):
        """
        Validates the model by running it eagerly with example inputs.

        Args:
            model: The model to validate.
            example_inputs: Example inputs for the model.

        Raises:
            RuntimeError: If eager run fails.
        """
        model = self.deepcopy_model(model)
        example_inputs = clone_inputs(example_inputs)
        model, example_inputs = self.cast_based_on_args(model, example_inputs)
        try:
            self.model_iter_fn(model, example_inputs)
        except Exception as e:
            raise RuntimeError("Eager run failed") from e

    def maybe_cast(self, model, example_inputs):
        """
        Conditionally casts the model and example inputs based on arguments.

        Args:
            model: The model to cast.
            example_inputs: Example inputs for the model.

        Returns:
            Tuple containing the casted model and example inputs.
        """
        model, example_inputs = self.cast_based_on_args(model, example_inputs)
        return model, example_inputs
    # 根据给定的批次大小和衰减因子计算新的批次大小
    def decay_batch_exp(self, batch_size, factor=0.5, divisor=2):
        out_batch_size = batch_size * factor
        # 如果新批次大小大于指定的除数，则将其调整为最接近的可以整除的值
        if out_batch_size > divisor:
            out_batch_size = (out_batch_size + 1) // divisor * divisor
        else:
            # 否则，将新批次大小设为原始批次大小减一
            out_batch_size = batch_size - 1
        return max(0, int(out_batch_size))

    # 在给定设备上寻找适合的批次大小，用于加载模型
    def batch_size_finder(self, device, model_name, initial_batch_size=1024):
        batch_size = initial_batch_size
        while batch_size >= 1:
            # 清空当前设备上的 GPU 缓存
            empty_gpu_cache(current_device)
            try:
                # 加载模型并获取相关信息
                device, name, model, example_inputs, _ = self.load_model(
                    device,
                    model_name,
                    batch_size,
                )
                # 执行模型迭代函数以验证批次大小是否适用
                self.model_iter_fn(model, example_inputs)
                return batch_size
            except RuntimeError as e:
                error_str = str(e)
                # 如果异常信息包含 "channels_last"，表示不支持的通道顺序，结束尝试
                if "channels_last" in error_str:
                    break
            # 根据指数衰减函数计算新的批次大小
            batch_size = self.decay_batch_exp(batch_size)
        return 1

    # 运行模型指定次数的迭代，并返回输出结果
    def run_n_iterations(self, mod, inputs):
        n = self.args.iterations
        for _ in range(n - 1):
            # 执行模型迭代函数，不收集输出结果
            self.model_iter_fn(mod, inputs, collect_outputs=False)
        # 执行模型迭代函数，收集输出结果
        return self.model_iter_fn(mod, inputs, collect_outputs=True)

    # 禁用 PyTorch 动态图功能，用于优化器操作
    @torch._disable_dynamo(recursive=True)
    def optimizer_zero_grad(self, mod):
        if self.optimizer is not None:
            # 如果存在优化器，调用其 zero_grad 方法清空梯度
            self.optimizer.zero_grad(True)
        else:
            # 否则，调用模型的 zero_grad 方法清空梯度
            mod.zero_grad(True)

    # 执行优化器的一步优化操作
    def optimizer_step(self):
        if self.optimizer is not None:
            # 如果存在优化器，执行其 step 方法进行一步优化
            self.optimizer.step()

    # 根据总长度获取分区的起始和结束索引
    def get_benchmark_indices(self, length):
        start = self._args.partition_id * (length // self._args.total_partitions)
        end = (
            (self._args.partition_id + 1) * (length // self._args.total_partitions)
            if self._args.partition_id < self._args.total_partitions - 1
            else length
        )
        return start, end
    def get_fsdp_auto_wrap_policy(self, model_name: str):
        # 导入所需模块
        from diffusers.models.transformer_2d import Transformer2DModel
        from torchbenchmark.models.nanogpt.model import Block
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        from transformers.models.t5.modeling_t5 import T5Block
        from transformers.models.whisper.modeling_whisper import WhisperEncoderLayer

        from torch.distributed.fsdp.wrap import (
            ModuleWrapPolicy,
            size_based_auto_wrap_policy,
        )

        # 手动配置的模型包装策略字典
        MODEL_FSDP_WRAP = {
            "stable_diffusion_unet": (Transformer2DModel,),
            "hf_T5": (T5Block,),
            "hf_T5_base": (T5Block,),
            "hf_T5_large": (T5Block,),
            "hf_Whisper": (WhisperEncoderLayer,),
            "llama_v2_7b_16h": (LlamaDecoderLayer,),
            "nanogpt": (Block,),
        }

        if model_name not in MODEL_FSDP_WRAP:
            # 如果模型名不在预定义的包装策略中，则使用基于模块大小的自动包装策略
            return functools.partial(
                size_based_auto_wrap_policy, recurse=True, min_num_params=int(1e5)
            )

        # 根据模型名返回对应的模块包装策略
        return ModuleWrapPolicy(MODEL_FSDP_WRAP[model_name])

    def deepcopy_and_maybe_parallelize(self, model):
        # 深度复制模型
        model = self.deepcopy_model(model)
        if self.args.ddp:
            assert (
                torch.distributed.is_available()
            ), "Can't use DDP without a distributed enabled build"
            from torch.nn.parallel import DistributedDataParallel as DDP

            # 如果启用了分布式数据并行（DDP），使用DDP并行化模型
            model = DDP(model, find_unused_parameters=True)
        elif self.args.fsdp:
            assert (
                torch.distributed.is_available()
            ), "Can't use FSDP without a distributed enabled build"
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                MixedPrecision,
            )

            # 根据参数设置混合精度策略
            if self.args.float16:
                dtype = torch.float16
            elif self.args.bfloat16:
                dtype = torch.bfloat16
            else:
                dtype = torch.float32

            # 配置混合精度策略
            mp_policy = MixedPrecision(
                param_dtype=dtype,
                # 梯度传输精度
                reduce_dtype=dtype,
                # 缓冲区精度
                buffer_dtype=dtype,
            )

            # 使用完全分片数据并行（FSDP）并行化模型
            model = FSDP(
                model,
                use_orig_params=True,
                device_id=torch.cuda.current_device()
                if self.args.devices[-1] == "cuda"
                else None,
                mixed_precision=mp_policy,
                limit_all_gathers=True,
                auto_wrap_policy=self.get_fsdp_auto_wrap_policy(self.args.only),
            )
        return model

    def check_accuracy(
        self, name, model, example_inputs, optimize_ctx, experiment, tag
    ):
        # 省略检查模型准确性的函数实现，需要进一步完善
    def check_tolerance(
        self, name, model, example_inputs, optimize_ctx, base_device="cpu"
    ):
        # 检查模型推理结果的容差，以确定性能优化是否有效
        logging.info("Checking tolerance for %s...", name)
        # 执行模型推理，并记录优化上下文
        self.run_performance_test(name, model, example_inputs, optimize_ctx, experiment)

    def run_performance_test(
        self, name, model, example_inputs, optimize_ctx, experiment, tag=None
    ):
        # 运行性能测试，评估模型在给定示例输入下的推理速度
        logging.info("Running performance test for %s...", name)
        # 最小化模型图以提高性能
        self.minify_model(name, model, example_inputs, optimize_ctx, experiment, tag)

    def minify_model(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        tag,
    ):
        # 最小化模型，减小其图形表示以优化性能
        logging.info("Minifying %s...", name)
        # 设置环境变量，启用调试和动态分析
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCHDYNAMO_REPRO_AFTER"] = "dynamo"
        os.environ["TORCHDYNAMO_REPRO_LEVEL"] = "4"

        # 检查模型准确性
        self.check_accuracy(name, model, example_inputs, optimize_ctx, experiment, tag)

        # 确定复现脚本的输出目录
        if self.args.output_directory:
            repro_dir = self.args.output_directory
        else:
            repro_dir = torch._dynamo.config.base_dir

        try:
            # 尝试将复现脚本移动到指定目录
            shutil.move("repro.py", f"{repro_dir}/{name}_repro.py")
        except OSError as e:
            # 处理移动失败的情况
            logging.error("Could not find repro script for model %s", name)
        else:
            # 记录复现脚本保存的位置
            logging.info(
                "Repro script for model %s with minified graph saved to %s",
                name,
                repro_dir,
            )

    def maybe_preserve_compile_debug(self, name, status):
        # 如果模型名称在指定的保留列表中，并且状态也在其中，则可能保留编译调试信息
        if (
            name in CI_PRESERVE_COMPILE_DEBUG
            and status in CI_PRESERVE_COMPILE_DEBUG[name]
        ):
            # 获取调试目录，并尝试将其移动到测试目录下的调试子目录
            src_dir = torch._dynamo.utils.get_debug_dir()
            if os.path.isdir(src_dir):
                dbg_dir = os.path.join(
                    os.getcwd(), "test", "debug", "torch_compile_debug"
                )
                dst_dir = os.path.join(dbg_dir, os.path.basename(src_dir))
                try:
                    # 创建目标调试目录并移动调试目录
                    os.makedirs(dbg_dir, exist_ok=True)
                    os.rename(src_dir, dst_dir)
                    log.warning("Moved %s to %s", src_dir, dst_dir)
                except OSError:
                    log.exception("Failed to preserve %s", src_dir)

    def run_one_model(
        self,
        name,
        model,
        example_inputs,
        optimize_ctx,
        experiment,
        explain=False,
        tag=None,
    ):
        # 运行单个模型的推理
        # 如果需要，解释推理结果
        logging.info("Running %s...", name)
        # 检查模型的容差性
        self.check_tolerance(name, model, example_inputs, optimize_ctx, base_device="cpu")
        ):
            # 根据训练模式设置消息模式为 "train" 或 "eval"
            mode = "train" if self.args.training else "eval"
            # 构建消息字符串，包括当前设备信息、模式、当前名称和可能的标签
            msg = f"{current_device:4} {mode:5} {current_name:34} "
            if tag:
                msg += f" {tag:26}"
            # 打印消息，并刷新输出缓冲区
            print(msg, flush=True)

            # 获取当前 Dynamo 统计信息的起始状态
            start_stats = get_dynamo_stats()

            # 如果设置了 --accuracy 参数，执行模型准确性检查
            if self.args.accuracy:
                # 调用检查准确性的方法，返回状态信息
                status = self.check_accuracy(
                    name, model, example_inputs, optimize_ctx, experiment, tag
                )
                # 打印状态信息
                print(status)
                # 如果准确性检查失败并且设置了 --minify 参数，缩小模型
                if status == "fail_accuracy" and self.args.minify:
                    self.minify_model(
                        name, model, example_inputs, optimize_ctx, experiment, tag
                    )
            # 如果设置了 --tolerance 参数，执行模型容差检查
            elif self.args.tolerance:
                status = self.check_tolerance(name, model, example_inputs, optimize_ctx)
                print(status)
            # 如果设置了 --performance 参数，执行性能测试
            elif self.args.performance:
                status = self.run_performance_test(
                    name, model, example_inputs, optimize_ctx, experiment, tag
                )
                print(status)
            
            # 清空当前设备的 GPU 缓存
            empty_gpu_cache(current_device)

            # 可能保留编译调试信息
            self.maybe_preserve_compile_debug(name, status)

            # 如果设置了 --timing 参数，打印时间报告和统计信息
            if self.args.timing:
                from torch._dynamo.utils import op_count, print_time_report
                from torch.utils._stats import simple_call_counter

                # 打印时间报告
                print_time_report()
                # 构建统计信息字符串
                stats = "STATS: "
                stats = stats + " | ".join(
                    itertools.chain(
                        [f"call_* op count: {op_count}"],
                        (f"{key}:{value}" for key, value in simple_call_counter.items()),
                    )
                )
                # 打印统计信息
                print(stats)
            
            # 获取当前 Dynamo 统计信息
            stats = get_dynamo_stats()
            # 计算统计信息的变化量
            stats.subtract(start_stats)

            # 如果需要解释，打印生成的图形数量及相关信息
            if explain:
                print(
                    f"Dynamo produced {stats['unique_graphs']} graphs "
                    f"covering {stats['calls_captured']} ops with "
                    f"{stats['graph_breaks']} graph breaks ({stats['unique_graph_breaks']} unique)"
                )

            # 如果需要解释或者设置了 --log_graph_breaks 或 --print_graph_breaks 参数，写入图形中断原因到文件
            if explain or self.args.log_graph_breaks or self.args.print_graph_breaks:
                # 构建输出文件名
                filename = f"{output_filename.rstrip('.csv')}_graph_breaks.csv"

                # 定义函数，用于将字符串添加双引号
                def add_double_quotes(x):
                    # 由于原因可能包含逗号，因此添加双引号
                    return f'"{x}"'

                # 遍历图形中断原因列表，将原因和用户堆栈写入 CSV 文件
                for graph_break in graph_break_reasons:
                    reason = add_double_quotes(graph_break.reason)
                    user_stack = add_double_quotes(
                        ", ".join([str(x) for x in graph_break.user_stack])
                    )
                    output_csv(
                        filename,
                        ["model", "reason", "user_stack"],
                        [current_name, reason, user_stack],
                    )

            # 如果设置了 --stats 参数，打印统计摘要信息
            if self.args.stats:
                Stats.print_summary()
# 返回函数对象的文档字符串作为帮助信息
def help(fn):
    return fn.__doc__


# 默认的 diff_branch 值，用于判断是否指定了不同的分支
diff_branch_default = "DIFF-BRANCH-DEFAULT"


# 判断是否指定了不同的分支
def should_diff_branch(args):
    return args.diff_branch != diff_branch_default


# 解析命令行参数的函数
def parse_args(args=None):
    # 创建一个命令行参数解析器
    parser = argparse.ArgumentParser()

    # 添加命令行选项
    parser.add_argument(
        "--filter", "-k", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude", "-x", action="append", help="filter benchmarks with regexp"
    )
    parser.add_argument(
        "--exclude-exact", action="append", help="filter benchmarks with exact match"
    )
    parser.add_argument(
        "--total-partitions",
        type=int,
        default=1,
        choices=range(1, 10),
        help="Total number of partitions we want to divide the benchmark suite into",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        default=0,
        help="ID of the benchmark suite partition to be run. Used to divide CI tasks",
    )
    parser.add_argument(
        "--devices", "--device", "-d", action="append", help="cpu or cuda"
    )
    parser.add_argument("--device-index", help="CUDA device index")
    parser.add_argument(
        "--repeat", "-n", type=int, default=30, help="number of timing runs"
    )
    iterations_per_run_help = """
        Run this may iterations for each time measurement. This is mainly used for
        XLA training. We want to run multiple iterations per measurement so the
        tracing and computation for different iteartions can overlap with each
        other. This makes sure we have an accurate xla baseline.
    """
    parser.add_argument(
        "--iterations-per-run", type=int, default=1, help=iterations_per_run_help
    )
    parser.add_argument(
        "--randomize-input",
        action="store_true",
        help="Whether to randomize the input values. Dimensions will be kept the same.",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        help="number of threads to use for eager and inductor",
    )
    parser.add_argument(
        "--nopython", action="store_true", help="Turn graph breaks into errors"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="run models that are in the global SKIP list",
    )
    parser.add_argument(
        "--prims-nvfuser", action="store_true", help="user prims + nvfuser backend"
    )
    parser.add_argument(
        "--dump-raw-metrics",
        action="store_true",
        help="dump raw timing metrics from speedup experiment",
    )
    parser.add_argument(
        "--log-operator-inputs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help="use channels last format",
    )
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, help="batch size for benchmarking"
    )

    # 返回解析后的命令行参数对象
    return parser.parse_args(args)
    # 添加一个命令行参数 --iterations，类型为整数，默认值为2，用于指定运行的迭代次数
    parser.add_argument(
        "--iterations", type=int, default=2, help="how many iterations to run"
    )

    # 添加一个命令行参数 --batch-size-file，类型为字符串，用于指定从文件中加载的批量大小的字符串
    parser.add_argument(
        "--batch-size-file", type=str, help="String to load batch size from"
    )

    # 添加一个命令行参数 --cosine，如果设置了这个参数则使用余弦相似度作为选项
    parser.add_argument("--cosine", action="store_true", help="use cosine similarity")

    # 添加一个命令行参数 --freezing，如果设置了这个参数则开启冻结，默认为关闭
    parser.add_argument(
        "--freezing", action="store_true", help="turn on freezing", default=False
    )

    # 添加一个命令行参数 --inductor-config 或者 -c，允许多次出现，用于指定在 torch._inductor.config 中的键值对配置
    parser.add_argument(
        "--inductor-config",
        "-c",
        action="append",
        help="key=value in torch._inductor.config",
    )

    # 添加一个命令行参数 --ci，如果设置了这个参数则表示是一个 CI 运行
    parser.add_argument(
        "--ci", action="store_true", help="Flag to tell that its a CI run"
    )

    # 添加一个命令行参数 --dashboard，如果设置了这个参数则表示是一个 Dashboard 运行
    parser.add_argument(
        "--dashboard", action="store_true", help="Flag to tell that its a Dashboard run"
    )

    # 添加一个命令行参数 --skip-fp64-check，如果设置了这个参数则跳过使用 fp64 进行的精度检查
    parser.add_argument(
        "--skip-fp64-check", action="store_true", help="skip accuracy check using fp64"
    )

    # 添加一个命令行参数 --fast 或者 -f，如果设置了这个参数则跳过慢速的基准测试
    parser.add_argument(
        "--fast", "-f", action="store_true", help="skip slow benchmarks"
    )

    # 添加一个命令行参数 --only，用于指定只运行 torchbench 中的一个模型，或者指定模型文件路径和类名
    parser.add_argument(
        "--only",
        help="""Run just one model from torchbench. Or
        specify the path and class name of the model in format like:
        --only=path:<MODEL_FILE_PATH>,class:<CLASS_NAME>

        Due to the fact that dynamo changes current working directory,
        the path should be an absolute path.

        The class should have a method get_example_inputs to return the inputs
        for the model. An example looks like
        ```
        class LinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

            def get_example_inputs(self):
                return (torch.randn(2, 10),)
        ```py
    """,
    )

    # 添加一个命令行参数 --multiprocess，如果设置了这个参数则根据设备数量创建多个进程（分布式场景下使用）
    parser.add_argument(
        "--multiprocess",
        action="store_true",
        help="Create n processes based on the number of devices (distributed use case).",
    )

    # 添加一个命令行参数 --ddp，如果设置了这个参数则在运行模型前使用 DDP 进行封装，并使用 dynamo 的 DDPOptimizer（默认图形断开）
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Wraps model in DDP before running it, and uses dynamo DDPOptimizer (graph breaks) by default.",
    )

    # 添加一个命令行参数 --fsdp，如果设置了这个参数则在运行模型前使用 FSDP 进行封装，不递归封装，主要用于检查 dynamo UnspecNNModule 的兼容性
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help="""Wraps model in FSDP before running it.
        Doesn't recursively wrap, mainly useful for checking dynamo UnspecNNModule compatibility
    """,
    )

    # 添加一个命令行参数 --optimize-ddp-mode，用于指定 DDP 优化模式的参数，默认为 ddp_optimizer
    parser.add_argument(
        "--optimize-ddp-mode",
        type=str,
        default="ddp_optimizer",
        help="Specify the DDP optimization mode -- the value of torch._dynamo.config.optimize_ddp.",
    )

    # 添加一个命令行参数 --distributed-master-port，用于指定 torch.distributed 绑定的端口号，默认为 6789
    parser.add_argument(
        "--distributed-master-port",
        default="6789",
        help="Port to bind for for torch.distributed.  Use the default unless it's conflicting with another user",
    )
    parser.add_argument(
        "--dynamic-shapes",
        action="store_true",
        help="Runs a dynamic shapes version of the benchmark, if available.",
    )
    # 添加命令行参数 "--dynamic-shapes"，如果存在则运行动态形状版本的基准测试

    parser.add_argument(
        "--propagate-real-tensors",
        action="store_true",
        help="Capture as much data dependent as you can by unsoundly propagating real tensors",
    )
    # 添加命令行参数 "--propagate-real-tensors"，捕获尽可能多的数据依赖，通过不安全地传播真实张量

    parser.add_argument(
        "--dynamic-batch-only",
        action="store_true",
        help="Only assume batch dimension is dynamic.  Implies --dynamic-shapes",
    )
    # 添加命令行参数 "--dynamic-batch-only"，仅假设批次维度是动态的。隐含了 "--dynamic-shapes"

    parser.add_argument(
        "--specialize-int", action="store_true", help="Run with specialize_int=True."
    )
    # 添加命令行参数 "--specialize-int"，使用 specialize_int=True 运行

    parser.add_argument(
        "--use-eval-mode",
        action="store_true",
        help="sets model.eval() to reduce randomness",
    )
    # 添加命令行参数 "--use-eval-mode"，设置 model.eval() 以减少随机性

    parser.add_argument(
        "--skip-accuracy-check",
        action="store_true",
        help="keeps running even when accuracy fails",
    )
    # 添加命令行参数 "--skip-accuracy-check"，即使准确性失败也继续运行

    parser.add_argument(
        "--generate-aot-autograd-stats",
        action="store_true",
        help="Generates AOT Autograd stats like how mnay graphs are sent to AOT",
    )
    # 添加命令行参数 "--generate-aot-autograd-stats"，生成 AOT 自动求导统计信息，例如发送给 AOT 的图的数量

    parser.add_argument(
        "--inductor-settings",
        action="store_true",
        help="Use same settings as --inductor for baseline comparisons",
    )
    # 添加命令行参数 "--inductor-settings"，用于基准比较的设置与 "--inductor" 相同

    parser.add_argument(
        "--suppress-errors",
        action="store_true",
        help="Suppress errors instead of raising them",
    )
    # 添加命令行参数 "--suppress-errors"，抑制错误而不是引发它们

    parser.add_argument(
        "--output",
        help="Overrides the output filename",
    )
    # 添加命令行参数 "--output"，覆盖输出文件名

    parser.add_argument(
        "--output-directory",
        help="Overrides the directory to place output files.",
    )
    # 添加命令行参数 "--output-directory"，覆盖放置输出文件的目录

    parser.add_argument(
        "--disable-output",
        action="store_true",
        help="Disable writing of output files, e.g., for warm-up runs",
    )
    # 添加命令行参数 "--disable-output"，禁用输出文件的写入，例如用于预热运行

    parser.add_argument(
        "--baseline",
        help="Compare with a prior --output",
    )
    # 添加命令行参数 "--baseline"，与先前的 "--output" 进行比较

    parser.add_argument(
        "--part",
        default=None,
        help="Specify the part of the model to run.",
    )
    # 添加命令行参数 "--part"，指定要运行的模型部分

    parser.add_argument(
        "--export-profiler-trace",
        action="store_true",
        help="exports trace of kineto profiler",
    )
    # 添加命令行参数 "--export-profiler-trace"，导出 Kineto 分析器的跟踪信息

    parser.add_argument(
        "--profiler-trace-name",
        "--profiler_trace_name",
        help="Overwrites exported trace name",
    )
    # 添加命令行参数 "--profiler-trace-name" 或 "--profiler_trace_name"，覆盖导出的跟踪名称

    parser.add_argument(
        "--diff-branch",
        default=diff_branch_default,
        help="delta current branch against given branch.",
    )
    # 添加命令行参数 "--diff-branch"，将当前分支与给定分支进行比较

    parser.add_argument(
        "--tag", default=None, help="Specify a tag to be included in csv files."
    )
    # 添加命令行参数 "--tag"，指定要包含在 CSV 文件中的标签

    parser.add_argument(
        "--explain",
        action="store_true",
        help="print some graph/op statistics during the run, similar to .explain()",
    )
    # 添加命令行参数 "--explain"，在运行期间打印一些图形/操作统计信息，类似于 .explain()

    parser.add_argument(
        "--stats",
        action="store_true",
        help="print graph counter stats",
    )
    # 添加命令行参数 "--stats"，打印图形计数统计信息
    # 添加一个命令行参数，用于启用使用热运行测量峰值内存，以减少自动调优噪声
    parser.add_argument(
        "--use-warm-peak-memory",
        "--use_warm_peak_memory",
        action="store_true",
        help="Measure peak memory using a warm run to reduce autotuning noise",
    )

    # 添加一个命令行参数，用于打印额外的内存统计信息
    parser.add_argument(
        "--print-memory",
        action="store_true",
        help="print extra memory statistics",
    )

    # 添加一个命令行参数，用于打印编译延迟时间
    parser.add_argument(
        "--print-compilation-time",
        action="store_true",
        help="print compilation latency",
    )

    # 添加一个命令行参数，用于打印用于计算精度的数据帧摘要
    parser.add_argument(
        "--print-dataframe-summary",
        action="store_true",
        help="print dataframe result used for calculating accuracy",
    )

    # 添加一个命令行参数，用于禁用 Inductor 的 cudagraphs
    parser.add_argument(
        "--disable-cudagraphs",
        action="store_true",
        help="Disables cudagraphs for Inductor",
    )

    # 添加一个命令行参数，用于禁用 Inductor 的 split reductions
    parser.add_argument(
        "--disable-split-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )

    # 添加一个命令行参数，用于禁用 Inductor 的 persistent reductions
    parser.add_argument(
        "--disable-persistent-reductions",
        action="store_true",
        help="Disables split reductions for Inductor",
    )

    # 添加一个命令行参数，用于向 Triton 提供 Inductor 可被 16 整除的提示
    parser.add_argument(
        "--disable-divisible-by-16",
        action="store_true",
        help="Disables divisible by 16 hint to Triton for Inductor",
    )

    # 添加一个命令行参数，用于指定 Inductor 运行的 torch.compile 模式参数
    parser.add_argument(
        "--inductor-compile-mode",
        default=None,
        help="torch.compile mode argument for inductor runs.",
    )

    # 添加一个命令行参数，用于显示图断点时显示警告
    parser.add_argument(
        "--print-graph-breaks",
        action="store_true",
        help="Show a warning whenever graph break",
    )

    # 添加一个命令行参数，用于在文件中记录图断点
    parser.add_argument(
        "--log-graph-breaks",
        action="store_true",
        help="log graph breaks in a file",
    )

    # 添加一个命令行参数，用于在 XLA 上追踪模型或在 eager 设备上追踪模型
    parser.add_argument(
        "--trace-on-xla",
        action="store_true",
        help="Whether to trace the model on XLA or on eager device",
    )

    # 添加一个命令行参数，用于设置 XLA 的容差阈值以通过正确性检查
    parser.add_argument(
        "--xla-tolerance",
        type=float,
        default=1e-2,
        help="XLA needs a loose tolerance to pass the correctness check",
    )

    # 添加一个命令行参数，用于收集训练输出。如果要验证梯度的数值正确性，则设置为 true。但这可能会导致时间测量不准确
    parser.add_argument(
        "--collect-outputs",
        action="store_true",
        help="""Whether to collect outputs for training. Set this to true if we
        want to verify the numerical correctness of graidents. But that may
        cause time measurement not accurate""",
    )

    # 添加一个命令行参数，用于启用 HF 模型的激活检查点
    parser.add_argument(
        "--enable-activation-checkpointing",
        action="store_true",
        help="Enables activation checkpointing for HF models",
    )

    # 添加一个命令行参数，用于发出阶段定时信息
    parser.add_argument("--timing", action="store_true", help="Emits phase timing")

    # 添加一个命令行参数，用于在每次模型运行之间打印 n/k 模型消息
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print n/k models message between each model run.",
    )

    # 添加一个命令行参数，用于设置用于基准测试的超时时间（秒）
    parser.add_argument(
        "--timeout",
        type=int,
        default=2000,
        help="timeout (second) for benchmarking.",
    )
    # 添加一个名为 --per_process_memory_fraction 的命令行参数，用于设置每个进程的 GPU 内存分配比例，默认为 1。
    parser.add_argument(
        "--per_process_memory_fraction",
        type=float,
        default=1,
        help="Set per-process GPU memory fraction (limit) for reducing usable size and reproducing OOMs",
    )

    # 添加一个名为 --no-translation-validation 的命令行参数，如果设置了该选项，则禁用翻译验证，适用于精确构建。
    parser.add_argument(
        "--no-translation-validation",
        action="store_true",
        help="Disable translation validation for accuracy builds.",
    )

    # 添加一个名为 --minify 的命令行参数，如果设置了该选项，则启用代码缩减，当故障低于容差时保存每个模型的复现脚本。
    parser.add_argument(
        "--minify",
        action="store_true",
        help="Enable minification when failure is below tolerance. Save repro script for each model.",
    )

    # 添加一个名为 --compiled-autograd 的命令行参数，如果设置了该选项，则启用编译的自动求导功能用于编译基准测试。
    parser.add_argument(
        "--compiled-autograd",
        action="store_true",
        help="Enables compiled autograd on compiled benchmark",
    )

    # 添加一个名为 --profile_dynamo_cache_lookup 或 --profile-dynamo-cache-lookup 的命令行参数，如果设置了该选项，则对 TorchDynamo 缓存查找进行分析。
    parser.add_argument(
        "--profile_dynamo_cache_lookup",
        "--profile-dynamo-cache-lookup",
        action="store_true",
        help="profiles TorchDynamo cache lookup",
    )

    # 添加一个名为 --snapshot-memory 或 --snapshot_memory 的命令行参数，如果设置了该选项，则启用内存快照工具进行深入内存分析。
    parser.add_argument(
        "--snapshot-memory",
        "--snapshot_memory",
        action="store_true",
        help="Enables Memory Snapshot tool for memory deep dives: https://pytorch.org/blog/understanding-gpu-memory-1/",
    )

    # 创建一个互斥的参数组 group_latency，包含 --cold-start-latency 和 --warm-start-latency 两个选项，用于控制模型冷启动和热启动的行为。
    group_latency = parser.add_mutually_exclusive_group()
    group_latency.add_argument(
        "--cold-start-latency",
        "--cold_start_latency",
        action="store_true",
        help="Use a fresh triton cachedir when running each model, to force cold-start compile.",
    )
    group_latency.add_argument(
        "--warm-start-latency",
        "--warm_start_latency",
        action="store_true",
        help="Run model(s) twice and preseve caches in between to enable a 'warm start' on the 2nd run",
    )

    # 创建一个互斥的参数组 group_fuser，包含 --nvfuser 和 --nnc 两个选项，用于控制 GPU 的融合器设置。
    group_fuser = parser.add_mutually_exclusive_group()
    # --nvfuser 已经成为默认选项，保留此选项以避免破坏脚本。
    group_fuser.add_argument("--nvfuser", action="store_true", help=argparse.SUPPRESS)
    group_fuser.add_argument("--nnc", action="store_true", help="enable NNC for GPUs")

    # 创建一个互斥的参数组 group_prec，包含 --float16、--bfloat16、--float32 和 --amp 四个选项，用于控制模型的精度设置。
    group_prec = parser.add_mutually_exclusive_group()
    group_prec.add_argument("--float16", action="store_true", help="cast model to fp16")
    group_prec.add_argument(
        "--bfloat16", action="store_true", help="cast model to bf16"
    )
    group_prec.add_argument("--float32", action="store_true", help="cast model to fp32")
    group_prec.add_argument(
        "--amp", action="store_true", help="use automatic mixed precision"
    )
    
    # 添加一个名为 --amp-dtype 的命令行参数，允许用户选择在自动混合精度中使用的数据类型，可选的值为 "bfloat16" 或 "float16"。
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        help="the data type used with automatic mixed precision",
    )

    # 创建一个互斥的参数组 group_printout，包含 --verbose/-v 和 --quiet/-q 两个选项，用于控制调试输出的详细程度。
    group_printout = parser.add_mutually_exclusive_group()
    group_printout.add_argument(
        "--verbose", "-v", action="store_true", help="enable verbose debug printouts"
    )
    group_printout.add_argument(
        "--quiet", "-q", action="store_true", help="suppress debug printouts"
    )

    # 创建一个互斥的参数组 group，用于将上述互斥组组合到一个整体的参数解析器中。
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--coverage", action="store_true", help="(default) " + help(coverage_experiment)
    )
    # 添加一个命令行参数 "--coverage"，如果存在则设置为 True，表示进行覆盖率实验，带有默认帮助信息和覆盖率实验的帮助信息
    group.add_argument(
        "--overhead", action="store_true", help=help(overhead_experiment)
    )
    # 添加一个命令行参数 "--overhead"，如果存在则设置为 True，带有 overhead_experiment 函数返回的帮助信息
    group.add_argument(
        "--speedup-dynamo-ts",
        action="store_true",
        help="TorchDynamo frontend with torchscript backend",
    )
    # 添加一个命令行参数 "--speedup-dynamo-ts"，如果存在则设置为 True，表示使用 TorchDynamo 前端和 torchscript 后端，带有指定的帮助信息
    group.add_argument(
        "--speedup-fx2trt", action="store_true", help=help(speedup_experiment_fx2trt)
    )
    # 添加一个命令行参数 "--speedup-fx2trt"，如果存在则设置为 True，带有 speedup_experiment_fx2trt 函数返回的帮助信息
    group.add_argument(
        "--speedup-fx2trt-fp16",
        action="store_true",
        help=help(speedup_experiment_fx2trt),
    )
    # 添加一个命令行参数 "--speedup-fx2trt-fp16"，如果存在则设置为 True，带有 speedup_experiment_fx2trt 函数返回的帮助信息
    group.add_argument(
        "--print-fx",
        action="store_true",
        help="Print fx traces captured from model",
    )
    # 添加一个命令行参数 "--print-fx"，如果存在则设置为 True，用于打印从模型捕获的 fx 跟踪，带有指定的帮助信息
    group.add_argument(
        "--print-aten-ops",
        action="store_true",
        help="Print traces of aten ops captured by AOT autograd",
    )
    # 添加一个命令行参数 "--print-aten-ops"，如果存在则设置为 True，用于打印由 AOT autograd 捕获的 aten 操作的跟踪，带有指定的帮助信息
    group.add_argument(
        "--inductor",
        action="store_true",
        help="Measure speedup with TorchInductor",
    )
    # 添加一个命令行参数 "--inductor"，如果存在则设置为 True，用于测量使用 TorchInductor 的加速效果，带有指定的帮助信息
    group.add_argument(
        "--quantization",
        choices=[
            "int8dynamic",
            "int8weightonly",
            "int4weightonly",
            "autoquant",
            "noquant",
        ],
        default=None,
        help="Measure speedup of torchao quantization with TorchInductor baseline",
    )
    # 添加一个命令行参数 "--quantization"，用于选择量化模式，带有指定的选项列表和默认值，带有指定的帮助信息
    group.add_argument(
        "--export",
        action="store_true",
        help="Measure pass rate with export",
    )
    # 添加一个命令行参数 "--export"，如果存在则设置为 True，用于测量导出的通过率，带有指定的帮助信息
    group.add_argument(
        "--export-aot-inductor",
        action="store_true",
        help="Measure pass rate with Export+AOTInductor",
    )
    # 添加一个命令行参数 "--export-aot-inductor"，如果存在则设置为 True，用于测量导出加 AOTInductor 的通过率，带有指定的帮助信息
    group.add_argument(
        "--xla", action="store_true", help="Compare TorchXLA to eager PyTorch"
    )
    # 添加一个命令行参数 "--xla"，如果存在则设置为 True，用于比较 TorchXLA 和 eager PyTorch，带有指定的帮助信息
    group.add_argument(
        "--torchscript-onnx",
        "--torchscript_onnx",
        action="store_true",
        help="Measure speedup with TorchScript ONNX, i.e. `torch.onnx.export`",
    )
    # 添加一个命令行参数 "--torchscript-onnx" 或 "--torchscript_onnx"，如果存在则设置为 True，用于测量使用 TorchScript ONNX 的加速效果，带有指定的帮助信息
    group.add_argument(
        "--torch-onnx-patch",
        "--torch_onnx_patch",
        action="store_true",
        help="Measure speedup with dynamo ONNX patch, i.e. `torch_onnx`",
    )
    # 添加一个命令行参数 "--torch-onnx-patch" 或 "--torch_onnx_patch"，如果存在则设置为 True，用于测量使用 dynamo ONNX 补丁的加速效果，带有指定的帮助信息
    group.add_argument(
        "--dynamo-onnx",
        "--dynamo_onnx",
        action="store_true",
        help="Measure speedup with Dynamo ONNX, i.e. `torch.onnx.dynamo_export`",
    )
    # 添加一个命令行参数 "--dynamo-onnx" 或 "--dynamo_onnx"，如果存在则设置为 True，用于测量使用 Dynamo ONNX 的加速效果，带有指定的帮助信息
    group.add_argument(
        "--dynamo-onnx-aot-inline",
        "--dynamo_onnx_aot_inline",
        action="store_true",
        help="Measure speedup with Dynamo ONNX AOT Inline, i.e. `torch.onnx.dynamo_export`",
    )
    # 添加一个命令行参数 "--dynamo-onnx-aot-inline" 或 "--dynamo_onnx_aot_inline"，如果存在则设置为 True，用于测量使用 Dynamo ONNX AOT Inline 的加速效果，带有指定的帮助信息
    group.add_argument(
        "--dynamo-onnx-aot-optimize",
        "--dynamo_onnx_aot_optimize",
        action="store_true",
        help="Measure speedup with Dynamo ONNX w/ ort fusions, i.e. `torch.onnx.dynamo_export`",
    )
    # 添加一个命令行参数 "--dynamo-onnx-aot-optimize" 或 "--dynamo_onnx_aot_optimize"，如果存在则设置为 True，用于测量使用 Dynamo ONNX 和 ort 融合的加速效果，带有指定的帮助信息
    group.add_argument(
        "--backend",
        choices=torch._dynamo.list_backends(exclude_tags=None),
        help="measure speedup with a given backend",
    )
    # 添加一个命令行参数 "--backend"，用于选择指定的后端，带有指定的选项列表和帮助信息
    # 添加一个名为 "--nothing" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    group.add_argument("--nothing", action="store_true", help=help(null_experiment))
    
    # 添加一个名为 "--log-conv-args" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    group.add_argument(
        "--log-conv-args",
        action="store_true",
        help="Dump convolution input/weight/bias's shape/stride/dtype and other options to json",
    )
    
    # 添加一个名为 "--recompile-profiler" 或 "--recompile_profiler" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    group.add_argument(
        "--recompile-profiler",
        "--recompile_profiler",
        action="store_true",
        help="Run the dynamo recompilation profiler on each model.",
    )
    
    # 添加一个名为 "--find-batch-sizes" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    group.add_argument(
        "--find-batch-sizes",
        action="store_true",
        help="finds the largest batch size that could fit on GPUs",
    )
    
    # 创建一个互斥的参数组 mode_group，至少需要选择其中一个参数
    mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # 添加一个名为 "--accuracy" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    mode_group.add_argument(
        "--accuracy",
        action="store_true",
        help="Checks accuracy with small batch size and eval mode",
    )
    
    # 添加一个名为 "--performance" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    mode_group.add_argument(
        "--performance", action="store_true", help="Measures performance speedup"
    )
    
    # 添加一个名为 "--tolerance" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    mode_group.add_argument(
        "--tolerance",
        action="store_true",
        help="extracts the tolerance for each model with small batch size and eval mode",
    )
    
    # 创建一个互斥的参数组 run_mode_group，至少需要选择其中一个参数
    run_mode_group = parser.add_mutually_exclusive_group(required=True)
    
    # 添加一个名为 "--training" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    run_mode_group.add_argument(
        "--training",
        action="store_true",
        help="Performs training",
    )
    
    # 添加一个名为 "--inference" 的命令行参数，如果指定则执行存储真值操作，并显示对应的帮助信息
    run_mode_group.add_argument(
        "--inference", action="store_true", help="Performs inference"
    )
    
    # 解析命令行参数并返回结果
    return parser.parse_args(args)
# 处理给定的运行排名、运行器、原始目录和参数，设置参数中的排名信息，初始化分布式环境并运行指定的运行器
def process_entry(rank, runner, original_dir, args):
    args.rank = rank
    # 根据可能的分布式初始化参数，设置分布式环境并运行指定的运行器
    with maybe_init_distributed(
        args.init_distributed,
        rank=rank,
        world_size=args.world_size,
        port=args.distributed_master_port,
    ):
        return run(runner, args, original_dir)


# 根据环境变量检查是否应该创建一个新的缓存上下文，若未设置缓存目录并且有冷启动或热启动延迟或 CI 标志，则创建一个新的感应器缓存
def maybe_fresh_cache(args):
    cache_dir_assigned = "TORCHINDUCTOR_CACHE_DIR" in os.environ
    if not cache_dir_assigned and (
        args.cold_start_latency or args.warm_start_latency or args.ci
    ):
        return fresh_inductor_cache()
    else:
        return contextlib.nullcontext()


# 主函数，根据给定的运行器和参数，设置工作目录、解析参数，处理基线路径，并在必要时检查分支差异
def main(runner, original_dir=None, args=None):
    if original_dir:
        os.chdir(original_dir)
    # 如果未提供参数，则解析命令行参数
    args = parse_args() if not args else parse_args(args)
    if args.baseline:
        # 将基线路径转换为绝对路径
        args.baseline = os.path.abspath(args.baseline)

    # 如果需要检查分支差异，则导入 Git 模块并进行相关检查
    if should_diff_branch(args):
        import git

        # 提前检查当前分支的状态，如果存在未提交的更改，则抛出错误
        repo = git.Repo()
        if repo.is_dirty():
            raise RuntimeError(
                "--diff-branch called on dirty branch. Commit, stash, or reset."
            )
        # 获取当前活动分支的名称
        main_branch = repo.active_branch.name
        # 如果当前分支与指定的 diff 分支相同，则抛出错误
        if main_branch == args.diff_branch:
            raise RuntimeError(
                f"--diff-branch: current branch is same as {args.diff_branch} branch, what are you diffing?"
            )
    # 尝试使用可能的缓存参数启动代码块
    with maybe_fresh_cache(args):
        # 如果只有一个进程并且需要多进程支持，则初始化分布式参数
        args.init_distributed = args.only and args.multiprocess
        if args.init_distributed:
            # 注意：在 CUDA 初始化之前不要查询设备数量；
            # 因为我们将覆盖 CUDA_VISIBLE_DEVICES，这可能导致问题
            device_count = torch.cuda.device_count()
            if device_count <= 1:
                log.warning(
                    "The use multiprocess flag is set but there are <= 1 devices available."
                )
            # 多进程路径
            args.world_size = device_count
            # 使用多进程方式启动子进程
            mp.spawn(
                process_entry, args=(runner, original_dir, args), nprocs=device_count
            )
        elif args.only and args.warm_start_latency:
            # 温启动模式。启用 FX 图缓存并在单独的进程中连续运行（确保在不同运行间保留导体缓存）。
            env = os.environ.copy()
            env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
            cmd = [sys.executable] + sys.argv
            cmd.remove("--warm-start-latency")

            print(f"Performing cold-start run for {args.only}")
            # 执行冷启动运行
            warmup_cmd = cmd + ["--repeat=1", "--disable-output"]
            subprocess.check_call(warmup_cmd, timeout=args.timeout, env=env)

            print(f"Performing warm-start run for {args.only}")
            # 执行温启动运行
            subprocess.check_call(cmd, timeout=args.timeout, env=env)
        else:
            # 单进程路径，仅使用主进程
            args.world_size = 1
            # 运行主进程入口函数
            process_entry(0, runner, original_dir, args)
# 打印状态信息到标准输出
def write_csv_when_exception(args, name: str, status: str, device=None):
    print(status)
    # 初始化占位批处理大小为零
    placeholder_batch_size = 0
    # 根据是否指定了设备，确定使用的设备列表
    devices = [device] if device is not None else args.devices
    # 根据参数指定的模式，选择不同的表头和行数据格式
    if args.accuracy:
        headers = ["dev", "name", "batch_size", "accuracy"]
        rows = [[device, name, placeholder_batch_size, status] for device in devices]
    elif args.performance:
        headers = ["dev", "name", "batch_size", "speedup", "abs_latency"]
        rows = [[device, name, placeholder_batch_size, 0.0, 0.0] for device in devices]
    else:
        headers = []
        rows = [[device, name, placeholder_batch_size, 0.0] for device in devices]

    # 遍历行数据，将每行写入输出 CSV 文件
    for row in rows:
        output_csv(output_filename, headers, row)


# 将解析后的参数对象传递给基准测试运行器对象
def run(runner, args, original_dir=None):
    runner.args = args

    # 如果未指定过滤器，则使用默认的正则表达式列表
    args.filter = args.filter or [r"."]
    # 如果未指定排除规则，则使用空的正则表达式列表
    args.exclude = args.exclude or [r"^$"]
    args.exclude_exact = args.exclude_exact or []

    # 如果启用感应器模式，则必须未指定后端，并将后端设置为"inductor"
    if args.inductor:
        assert args.backend is None
        args.backend = "inductor"
    # 如果启用量化，则必须未指定后端，并将后端设置为"torchao"
    if args.quantization:
        assert args.backend is None
        args.backend = "torchao"
    # 如果仅动态批处理，则设置动态形状为True，并修改默认静态假设设置
    if args.dynamic_batch_only:
        args.dynamic_shapes = True
        torch._dynamo.config.assume_static_by_default = True
    # 如果启用动态形状，根据是否仅动态批处理设置静态假设设置
    if args.dynamic_shapes:
        if not args.dynamic_batch_only:
            torch._dynamo.config.assume_static_by_default = False
    # 如果传播实际张量，则设置相关的动态配置选项
    if args.propagate_real_tensors:
        # TODO: 单独标志用于数据相关设置
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._functorch.config.fake_tensor_propagate_real_tensors = True
    # 如果启用整数专用化，则设置相关的动态配置选项
    if args.specialize_int:
        torch._dynamo.config.specialize_int = True
    # 如果在CI环境中运行，则根据精度模式减少迭代次数，并设置翻译验证默认开启
    if args.ci:
        if args.accuracy:
            args.repeat = min(args.repeat, 2)  # 在检查精度时运行较少的迭代次数
            torch.fx.experimental._config.translation_validation = True  # 在CI精度运行时默认开启翻译验证

        # 针对CI环境配置的部分函数
        ci = functools.partial(
            CI, args.backend, training=args.training, dynamic=args.dynamic_shapes
        )
    # 如果启用分布式数据并行，则确保处于训练模式，并根据优化DDP模式设置相关动态配置选项
    if args.ddp:
        assert args.training, "DDP benchmark requires --training mode"
        torch._dynamo.config.optimize_ddp = args.optimize_ddp_mode
        # 如果仅支持特定应用（如DLRM），则报错并退出
        if args.only == "dlrm":
            log.error(
                "DLRM+DDP is unsupported as it requires sharding the embedding layer separately from DDP"
            )
            return sys.exit(-1)
    # 如果设置了参数 --accuracy
    if args.accuracy:
        # 使用较小的批量大小。我们使用 >1 的批量大小来确保测试能够
        # 考虑作用于批量维度的 batch_norm 类型的运算符。
        # TODO - 检查批量大小为 2 时的失败情况
        if args.batch_size is None:
            # 根据 runner 的 suite_name 设置不同的批量大小
            if runner.suite_name == "huggingface":
                args.batch_size = 1
            elif runner.suite_name == "torchbench":
                args.batch_size = 4
            else:
                # 对于 TIMM 模型使用较大的批量大小以确保稳定的 batch_norm
                assert runner.suite_name == "timm_models"
                args.batch_size = 8

        # 去除随机性来源
        if runner.suite_name not in ("timm_models", "huggingface"):
            # TODO - 对于 timm_models 和 HF 模型使用训练模式，也将 Torchbench 模型移至训练模式。
            args.use_eval_mode = True
        inductor_config.fallback_random = True

        # 如果指定了 args.only，并且不在指定的模型集合内，则设置确定性算法
        if args.only is not None and args.only not in {
            "alexnet",
            "Background_Matting",
            "pytorch_CycleGAN_and_pix2pix",
            "pytorch_unet",
            "Super_SloMo",
            "vgg16",
            # https://github.com/pytorch/pytorch/issues/96724
            "Wav2Vec2ForCTC",
            "Wav2Vec2ForPreTraining",
            "sam",
            "sam_fast",
            "resnet50_quantized_qat",
            "mobilenet_v2_quantized_qat",
        }:
            # 一些模型不支持使用确定性算法
            torch.use_deterministic_algorithms(True)

        # 设置 CUBLAS 的工作空间配置
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # 设置 PyTorch 的 cudnn 模块参数以确保确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

        # 设置 PyTorch 的 mkldnn 模块参数以确保确定性
        torch.backends.mkldnn.deterministic = True

        # 调整 torch 手动种子时的随机性
        patch_torch_manual_seed()

        # 某些模型（如 yolov3）对于 n_gpus 上的批量大小有严格要求
        if "CUDA_VISIBLE_DEVICES" not in os.environ and not args.multiprocess:
            args.device_index = "0"

        # 更严格地禁用回退错误检查
        args.suppress_errors = False

    # 如果指定了参数 --device_index
    if args.device_index is not None:
        # 如果同时指定了 --device_index 和 --multiprocess，则报错退出
        if args.multiprocess:
            print("Cannot specify both --device_index and --multiprocess")
            return sys.exit(-1)
        # 设置 CUDA_VISIBLE_DEVICES 环境变量为指定的设备索引
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index

    # 如果指定了参数 --performance
    elif args.performance:
        # 确保我们在真实场景下进行测试
        args.use_eval_mode = False

    # 如果 args.partition_id 超出了有效的分区范围
    if args.partition_id > args.total_partitions or args.partition_id < 0:
        print("Invalid partition id")
        return sys.exit(-1)

    # 如果未指定 args.devices
    if not args.devices:
        # 如果 CUDA 可用，则使用 "cuda" 设备；否则使用 CPU
        if torch.cuda.is_available():
            args.devices = ["cuda"]
        else:
            log.warning("torch.cuda.is_available() == False, using CPU")
            args.devices = ["cpu"]
    # 检查是否指定了非CPU设备并且系统支持CUDA或XPU加速
    if args.devices != ["cpu"] and (HAS_CUDA or HAS_XPU):
        # 根据系统支持情况选择同步函数
        global synchronize
        synchronize = torch.cuda.synchronize if HAS_CUDA else torch.xpu.synchronize

    # 检查是否指定了仅CUDA设备并且第一个CUDA设备的总内存小于25GB
    if (
        args.devices == ["cuda"]
        and torch.cuda.get_device_properties(0).total_memory < 25 * 2**30
    ):
        # 在RTX 3090（24GB RAM）上遇到OOM错误，跳过指定模型以避免问题
        runner.skip_models.update(
            {
                # torchbench
                "hf_Longformer",
                "timm_nfnet",
                "timm_efficientdet",
            }
        )
        if args.training:
            runner.skip_models.add("hf_T5")

    # 如果指定了NNC，设置Torch的CPU和GPU融合优化选项
    if args.nnc:
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        torch._C._jit_set_texpr_fuser_enabled(True)
        torch._C._jit_set_nvfuser_enabled(False)

    # 如果指定了线程数，设置Torch的线程数
    if args.threads:
        torch.set_num_threads(args.threads)

    # 如果指定了verbose模式，设置Torch的日志级别为DEBUG
    if args.verbose:
        torch._logging.set_logs(dynamo=logging.DEBUG)

    # 如果指定了print_graph_breaks，设置Torch打印图中断的日志
    if args.print_graph_breaks:
        torch._logging.set_logs(graph_breaks=True)

    # 如果指定了quiet模式，设置Torch的日志级别为ERROR
    if args.quiet:
        torch._logging.set_logs(dynamo=logging.ERROR)

    # 根据命令行参数设置Torch Dynamo的错误抑制选项
    torch._dynamo.config.suppress_errors = args.suppress_errors

    # 根据训练模式设置runner的模型迭代函数和需要跳过的模型集合
    if args.training:
        runner.model_iter_fn = runner.forward_and_backward_pass
        runner.skip_models.update(runner.skip_not_suitable_for_training_models)
    else:
        runner.model_iter_fn = runner.forward_pass

    # 如果指定了fast模式，更新需要跳过的模型集合为slow_models
    if args.fast:
        runner.skip_models.update(runner.slow_models)

    # 根据设备参数更新需要跳过的模型集合，如果设备是CPU则使用very_slow_models和skip_models_for_cpu
    if args.devices == ["cpu"]:
        runner.skip_models.update(runner.very_slow_models)
        runner.skip_models.update(runner.skip_models_for_cpu)
    elif args.devices == ["cuda"]:
        runner.skip_models.update(runner.skip_models_for_cuda)

    # 如果不使用多进程，则更新需要跳过的模型集合为skip_multiprocess_models
    if not args.multiprocess:
        runner.skip_models.update(runner.skip_multiprocess_models)

    # 如果指定了freezing模式，更新需要跳过的模型集合为skip_models_for_freezing
    if args.freezing:
        runner.skip_models.update(runner.skip_models_for_freezing)

    # 如果指定了no_skip，清空需要跳过的模型集合
    if args.no_skip:
        runner.skip_models.clear()

    # 根据命令行参数设置experiment为null_experiment，并初始化一些全局变量
    experiment = null_experiment
    global current_name, current_device, current_batch_size, output_filename, disable_output, optimize_ctx, current_onnx_compiler
    optimize_ctx = contextlib.nullcontext()

    # 如果指定了disable_output，设置disable_output为True
    if args.disable_output:
        disable_output = True

    # 根据命令行参数设置优化上下文和experiment类型
    if args.overhead:
        # 如果指定了overhead，使用torch._dynamo.optimize进行优化，并设置相应的experiment和输出文件名
        optimize_ctx = torch._dynamo.optimize(dummy_fx_compile, nopython=args.nopython)
        experiment = speedup_experiment
        output_filename = "overheads.csv"
    elif args.inductor:
        # 如果指定了inductor，配置相关参数并设置optimize_ctx为inductor编译器
        inductor_config.debug = args.verbose
        if args.threads:
            inductor_config.cpp.threads = args.threads

        optimize_ctx = functools.partial(
            torch.compile,
            backend="inductor",
            fullgraph=args.nopython,
            mode=args.inductor_compile_mode,
        )
        experiment = speedup_experiment
        output_filename = "inductor.csv"
    # 如果命令行参数 args.export 存在，则设置优化上下文为 export
    elif args.export:
        optimize_ctx = export
        # 设置实验类型为 speedup_experiment
        experiment = speedup_experiment
        # 设置输出文件名为 "export.csv"
        output_filename = "export.csv"
    # 如果命令行参数 args.xla 存在
    elif args.xla:
        # 从参数中获取设备类型 dev
        (dev,) = args.devices
        # 设置环境变量 "PJRT_DEVICE" 根据设备类型选择 "GPU" 或 "CPU"
        os.environ["PJRT_DEVICE"] = {"cuda": "GPU", "cpu": "CPU"}[dev]
        # 设置 torch._dynamo.mark_dynamic 为 MagicMock 对象
        torch._dynamo.mark_dynamic = MagicMock()
        # 设置实验类型为 xla
        experiment = xla
        # 设置输出文件名为 "xla.csv"
        output_filename = "xla.csv"
    # 如果命令行参数 args.torchscript_onnx 存在
    elif args.torchscript_onnx:
        # 设置 optimize_ctx 为部分应用的 optimize_onnx_ctx 函数
        # 参数包括输出目录、OnnxModelFromTorchScript 类型等
        # 根据参数设置是否执行深拷贝
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromTorchScript,
            copy_before_export=args.performance,  # Accuarcy bench already did deepcopy
        )
        # 设置实验类型为 speedup_experiment_onnx
        experiment = speedup_experiment_onnx
        # 设置输出文件名为 "torchscript_onnx.csv"
        output_filename = "torchscript_onnx.csv"
        # 设置当前 ONNX 编译器为 "torchscript"
        current_onnx_compiler = "torchscript"
    # 如果命令行参数 args.torch_onnx_patch 存在
    elif args.torch_onnx_patch:
        # 设置 optimize_ctx 为部分应用的 optimize_onnx_ctx 函数
        # 参数包括输出目录、OnnxModelFromTorchScript 类型等
        # 根据参数设置是否执行深拷贝和使用实验性补丁
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromTorchScript,
            copy_before_export=args.performance,
            use_experimental_patch=True,
        )
        # 设置实验类型为 speedup_experiment_onnx
        experiment = speedup_experiment_onnx
        # 设置输出文件名为 "torch_onnx_patch.csv"
        output_filename = "torch_onnx_patch.csv"
        # 设置当前 ONNX 编译器为 "torch_onnx_patch"
        current_onnx_compiler = "torch_onnx_patch"
    # 如果命令行参数 args.dynamo_onnx 存在
    elif args.dynamo_onnx:
        # 设置 optimize_ctx 为部分应用的 optimize_onnx_ctx 函数
        # 参数包括输出目录、OnnxModelFromDynamo 类型、动态形状等
        # 根据参数设置是否执行深拷贝
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromDynamo,
            dynamic_shapes=args.dynamic_shapes,
            copy_before_export=args.performance,
        )
        # 设置实验类型为 speedup_experiment_onnx
        experiment = speedup_experiment_onnx
        # 设置输出文件名为 "dynamo_onnx.csv"
        output_filename = "dynamo_onnx.csv"
        # 设置当前 ONNX 编译器为 "dynamo"
        current_onnx_compiler = "dynamo"
    # 如果命令行参数 args.dynamo_onnx_aot_inline 存在
    elif args.dynamo_onnx_aot_inline:
        # 设置 optimize_ctx 为部分应用的 optimize_onnx_ctx 函数
        # 参数包括输出目录、OnnxModelFromDynamoAotInline 类型、动态形状等
        # 根据参数设置是否执行深拷贝
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromDynamoAotInline,
            dynamic_shapes=args.dynamic_shapes,
            copy_before_export=args.performance,
        )
        # 设置实验类型为 speedup_experiment_onnx
        experiment = speedup_experiment_onnx
        # 设置输出文件名为 "dynamo_onnx_aot_inline.csv"
        output_filename = "dynamo_onnx_aot_inline.csv"
        # 设置当前 ONNX 编译器为 "dynamo"
        current_onnx_compiler = "dynamo"
    # 如果命令行参数 args.dynamo_onnx_aot_optimize 存在
    elif args.dynamo_onnx_aot_optimize:
        # 设置 optimize_ctx 为部分应用的 optimize_onnx_ctx 函数
        # 参数包括输出目录、OnnxModelFromDynamoAotOptimize 类型、动态形状等
        # 根据参数设置是否执行深拷贝
        optimize_ctx = functools.partial(
            optimize_onnx_ctx,
            args.output_directory or ".",
            OnnxModelFromDynamoAotOptimize,
            dynamic_shapes=args.dynamic_shapes,
            copy_before_export=args.performance,
        )
        # 设置实验类型为 speedup_experiment_onnx
        experiment = speedup_experiment_onnx
        # 设置输出文件名为 "dynamo_onnx_aot_optimize.csv"
        output_filename = "dynamo_onnx_aot_optimize.csv"
        # 设置当前 ONNX 编译器为 "dynamo"
        current_onnx_compiler = "dynamo"
    # 如果命令行参数 args.speedup_dynamo_ts 存在
    elif args.speedup_dynamo_ts:
        # 设置 optimize_ctx 为 torch._dynamo.optimize 函数的调用
        # 参数包括编译器类型 "ts" 和是否启用 nopython
        optimize_ctx = torch._dynamo.optimize("ts", nopython=args.nopython)
        # 设置实验类型为 speedup_experiment
        experiment = speedup_experiment
        # 设置输出文件名为 "speedup_dynamo_ts.csv"
        output_filename = "speedup_dynamo_ts.csv"
    # 如果命令行参数 args.prims_nvfuser 存在
    elif args.prims_nvfuser:
        # 设置 optimize_ctx 为 torch._dynamo.optimize 函数的调用
        # 参数包括编译器类型 "prims_nvfuser" 和是否启用 nopython
        optimize_ctx = torch._dynamo.optimize("prims_nvfuser", nopython=args.nopython)
        # 设置实验类型为 speedup_experiment
        experiment = speedup_experiment
        # 设置 backend_str 为 "prims_nvfuser"
        backend_str = "prims_nvfuser"
        # 设置输出文件名为 "accuracy_aot_prims_nvfuser.csv"
        output_filename = f"accuracy_aot_{backend_str}.csv"
    elif args.print_fx:
        # 如果指定打印 fx，则调用 torch._dynamo.optimize 函数进行优化
        optimize_ctx = torch._dynamo.optimize(
            print_fx,
            nopython=args.nopython,
        )
    elif args.print_aten_ops:
        # 如果指定打印 aten 操作，则同样调用 torch._dynamo.optimize 函数进行优化
        optimize_ctx = torch._dynamo.optimize(
            print_aten_ops,
            nopython=args.nopython,
        )
    elif args.nothing:
        # 如果参数指定为 nothing，则不进行优化，设定实验为 speedup_experiment，输出文件名为 "nothing.csv"
        optimize_ctx = nothing
        experiment = speedup_experiment
        output_filename = "nothing.csv"
    elif args.backend or args.export_aot_inductor:
        if args.export_aot_inductor:
            # 如果指定导出 AOTInductor，则部分断言检查和设置函数调用
            assert not args.training, "AOTInductor only supports inference"
            optimize_ctx = functools.partial(
                export_aot_inductor, device=args.devices[0]
            )

            # AOTInductor 目前不支持控制流
            runner.skip_models.update(runner.skip_models_due_to_control_flow)
        elif args.backend == "torchao":
            # 如果指定后端为 "torchao"，则进行相关设置和优化上下文的设定
            assert "cuda" in args.devices, "Quantization requires CUDA device."
            assert args.bfloat16, "Quantization requires dtype bfloat16."
            try:
                from torchao_backend import setup_baseline, torchao_optimize_ctx
            except ImportError:
                from userbenchmark.dynamo.dynamobench.torchao_backend import (
                    setup_baseline,
                    torchao_optimize_ctx,
                )

            setup_baseline()
            # 设定 baseline_ctx 为编译后的模型迭代函数
            baseline_ctx = functools.partial(
                torch.compile,
                backend="inductor",
                fullgraph=args.nopython,
                mode=args.inductor_compile_mode,
            )
            runner.model_iter_fn = baseline_ctx(runner.model_iter_fn)
            # 设定 optimize_ctx 为 torchao_optimize_ctx 函数，参数为量化设置 args.quantization
            optimize_ctx = torchao_optimize_ctx(args.quantization)
        else:
            # 对于其他后端情况，使用 torch._dynamo.optimize 函数进行优化
            optimize_ctx = torch._dynamo.optimize(args.backend, nopython=args.nopython)
        experiment = speedup_experiment
        # 根据参数选择输出文件名，可能为 accuracy_xxx.csv、tolerance_xxx.csv 或 speedup_xxx.csv
        if args.accuracy:
            output_filename = f"accuracy_{args.backend}.csv"
        elif args.tolerance:
            output_filename = f"tolerance_{args.backend}.csv"
        else:
            output_filename = f"speedup_{args.backend}.csv"
    elif args.recompile_profiler:
        # 如果指定重新编译 profiler，则输出文件名设定为 "recompile_profiler_log.csv"，实验设定为 recompile_profiler_experiment
        output_filename = "recompile_profiler_log.csv"
        experiment = recompile_profiler_experiment
    else:
        # 默认情况下，使用 torch._dynamo.optimize 函数对 fx_insert_profiling 进行优化
        optimize_ctx = torch._dynamo.optimize(
            fx_insert_profiling, nopython=args.nopython
        )
        experiment = coverage_experiment
        output_filename = "coverage.csv"
    # 检查是否需要配置感应器（inductor），或者后端是"inductor"，或者需要导出AOT感应器
    if args.inductor or args.backend == "inductor" or args.export_aot_inductor:
        # 设置 Triton 配置中的 cudagraphs 标志，根据参数确定是否禁用 cudagraphs
        inductor_config.triton.cudagraphs = not args.disable_cudagraphs
        # 设置 Triton 配置中的 persistent_reductions 标志，根据参数确定是否禁用 persistent_reductions
        inductor_config.triton.persistent_reductions = (
            not args.disable_persistent_reductions
        )
        # 设置 Triton 配置中的 split_reductions 标志，根据参数确定是否禁用 split_reductions
        inductor_config.split_reductions = not args.disable_split_reductions
        # 设置 Triton 配置中的 divisible_by_16 标志，根据参数确定是否禁用 divisible_by_16
        inductor_config.triton.divisible_by_16 = not args.disable_divisible_by_16
        # 如果启用推断（inference）模式，设置冻结（freezing）标志
        if args.inference:
            inductor_config.freezing = args.freezing
        # 如果有感应器配置参数，逐个处理每个配置
        if args.inductor_config:
            for config in args.inductor_config:
                # 分割键值对配置
                key, value = config.split("=")
                # 获取感应器配置对象的属性类型
                typ = type(inductor_config.__getattr__(key))
                # 根据属性类型进行类型转换
                if issubclass(typ, bool):
                    assert value in ("0", "1", "True", "False")
                    value = value in ("1", "True")
                elif issubclass(typ, (str, int, float)):
                    value = typ(value)
                else:
                    raise NotImplementedError(typ)
                # 设置感应器配置对象的属性值
                inductor_config.__setattr__(key, value)

    # 设置运行器的自动混合精度（Automatic Mixed Precision，AMP）
    runner.setup_amp()

    # 如果指定了输出文件名参数
    if args.output:
        output_filename = args.output

    # 如果存在输出文件名
    if output_filename:
        # 如果指定了输出目录参数，则将输出文件名与输出目录合并
        if args.output_directory:
            output_filename = os.path.join(args.output_directory, output_filename)
        else:
            # 否则，将输出文件名与基础目录（base_dir）合并
            output_filename = os.path.join(
                torch._dynamo.config.base_dir, output_filename
            )

    # 如果需要查找批量大小并且指定了only参数
    if args.find_batch_sizes and args.only:
        # 遍历指定的设备列表
        for device in args.devices:
            # 使用运行器查找设备上only参数对应的批量大小
            batch_size = runner.batch_size_finder(device, args.only)
            # 打印only参数和其对应的批量大小
            print(args.only, batch_size)
            # 输出CSV文件，包含only参数和对应的批量大小
            output_csv(output_filename, [], [args.only, batch_size])
        # 返回结束函数执行
        return

    # 如果需要导出分析器追踪
    if args.export_profiler_trace:
        # 如果未指定分析器追踪名称
        if args.profiler_trace_name is None:
            # 根据后端参数设置分析器追踪名称
            if args.backend:
                args.profiler_trace_name = args.backend
            # 如果使用感应器，设置分析器追踪名称为"inductor"
            elif args.inductor:
                args.profiler_trace_name = "inductor"
            # 否则，设置默认分析器追踪名称为"profile"
            else:
                args.profiler_trace_name = "profile"
        else:
            # 否则，保持指定的分析器追踪名称不变
            args.profiler_trace_name = args.profiler_trace_name

    # 如果禁用翻译验证
    if args.no_translation_validation:
        # 覆盖'translation_validation'配置，设置为False
        torch.fx.experimental._config.translation_validation = False

    # 配置实验函数，部分应用参数和模型迭代器函数
    experiment = functools.partial(experiment, args, runner.model_iter_fn)
    # 检查是否设置了 `--only` 参数并且应当比较分支
    if args.only and should_diff_branch(args):
        # 导入 git 模块
        import git

        # 获取当前 git 仓库对象
        repo = git.Repo()
        # 获取当前活跃分支的名称
        main_branch = repo.active_branch.name
        try:
            # 向 `args` 中再次添加 `diff-branch` 参数，将覆盖之前的值
            call_args = (
                [sys.executable] + sys.argv + [f"--diff-branch={diff_branch_default}"]
            )
            # 运行主分支的命令
            subprocess.check_call(call_args + [f"--tag={main_branch}"])
            # 切换到比较分支
            repo.git.checkout(args.diff_branch)
            # 运行比较分支的命令
            subprocess.check_call(call_args + [f"--tag={args.diff_branch}"])
        finally:
            # 回到主分支
            repo.git.checkout(main_branch)
    else:
        # 清理旧的日志文件
        metrics.purge_old_log_files()
        # 如果指定了输出文件名且文件存在，则删除该文件
        if output_filename and os.path.exists(output_filename):
            os.unlink(output_filename)
        # 如果有原始目录，则切换回原始目录
        if original_dir:
            os.chdir(original_dir)
        # 获取模型名称列表
        model_names = list(runner.iter_model_names(args))
        # 获取模型数量
        nmodels = len(model_names)
        # 遍历模型名称列表
        for i, name in enumerate(model_names):
            # 当前模型名称
            current_name = name
            # 如果设置了 `--progress` 参数，则打印运行进度信息
            if args.progress:
                print(f"Running model {i+1}/{nmodels}", flush=True)

            try:
                # 设置超时时间为 `args.timeout` 的两倍，如果需要比较分支
                timeout = args.timeout
                if should_diff_branch(args):
                    timeout *= 2
                # 复制当前环境变量
                env = os.environ.copy()
                # 如果是持续集成环境且模型名称在 `CI_PRESERVE_COMPILE_DEBUG` 中，则设置 `TORCH_COMPILE_DEBUG` 为 `1`
                if args.ci and name in CI_PRESERVE_COMPILE_DEBUG:
                    env["TORCH_COMPILE_DEBUG"] = "1"
                # 执行子进程命令
                subprocess.check_call(
                    [sys.executable] + sys.argv + [f"--only={name}"],
                    timeout=timeout,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                # 处理超时异常，写入 CSV 文件以记录异常信息
                write_csv_when_exception(args, name, "timeout")
            except subprocess.CalledProcessError as e:
                # 处理调用子进程返回非零错误码的异常
                print("Run failed with return code: ", e.returncode, file=sys.stderr)
                print("Output: ", e.output, file=sys.stderr)
                print("Error: ", e.stderr, file=sys.stderr)
        # 打印总结信息到指定的输出文件中，如果需要打印数据框摘要
        print_summary(output_filename, print_dataframe=args.print_dataframe_summary)
# 记录操作符输入的函数，用于生成运行日志
def log_operator_inputs(model, example_inputs, model_iter_fn, name, args):
    # 根据参数决定模式是训练还是评估
    mode = "training" if args.training else "eval"
    # 根据输出路径和名称生成日志文件路径
    output = os.path.join(os.path.dirname(args.output), f"{name}_{mode}.txt")

    # 如果输出文件已经存在，则跳过当前操作
    if os.path.exists(output):
        print(f"Skipping {name}, {output} already exists")
        return

    # 输出当前运行的操作名称
    print(f"Running {name}")
    
    try:
        # 尝试导入运算符输入工具类
        from .microbenchmarks.operator_inp_utils import OperatorInputsMode
    except ImportError:
        # 如果导入失败，则尝试从全局中导入
        from microbenchmarks.operator_inp_utils import OperatorInputsMode

    # 创建运算符输入模式和虚拟张量模式对象
    operator_mode = OperatorInputsMode()
    fake_tensor_mode = FakeTensorMode()

    # 使用虚拟张量模式复制模型和示例输入
    with torch._subclasses.fake_tensor.FakeCopyMode(fake_tensor_mode):
        model_fake = copy.deepcopy(model)
        example_inputs_fake = copy.deepcopy(example_inputs)

    try:
        # 尝试使用虚拟张量模式和运算符输入模式执行模型迭代函数
        with fake_tensor_mode, operator_mode:
            model_iter_fn(model_fake, example_inputs_fake, collect_outputs=False)
    except Exception as e:
        # 如果使用虚拟张量模式失败，则尝试使用真实模式执行
        print(f"{name} failed to run with fake tensors, trying real. Exception: {e}")
        operator_mode = OperatorInputsMode()
        try:
            with operator_mode:
                model_iter_fn(model, example_inputs, collect_outputs=False)
        except Exception as e2:
            # 如果真实模式也失败，则抛出异常
            print(f"{name} failed to run with real. Exception: {e2}")
            raise

    # 将运行结果写入指定的输出文件中
    print(f"Writing output to {output}")
    operator_mode.log_to_file(output)


# 如果当前脚本是主程序，则抛出运行时错误，建议使用其他指定的脚本运行
if __name__ == "__main__":
    raise RuntimeError(
        f"You shouldn't run {sys.argv[0]} directly, instead try timm_model.py, torchbench.py or huggingface.py"
    )
```