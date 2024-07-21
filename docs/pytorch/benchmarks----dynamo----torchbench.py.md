# `.\pytorch\benchmarks\dynamo\torchbench.py`

```py
#!/usr/bin/env python3
import functools
import gc
import importlib
import logging
import os
import re
import sys
import warnings
from collections import namedtuple
from os.path import abspath, exists

import yaml

import torch

try:
    from .common import BenchmarkRunner, main
except ImportError:
    from common import BenchmarkRunner, main

from torch._dynamo.testing import collect_results, reduce_to_scalar_loss
from torch._dynamo.utils import clone_inputs

# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True
# 允许使用 tf32 数据类型进行 CUDA 矩阵乘法运算

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True
    # 如果环境变量中没有设置 TORCHINDUCTOR_FX_GRAPH_CACHE，则启用 FX 图形缓存

def _reassign_parameters(model):
    # torch_geometric models register parameter as tensors due to
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/dense/linear.py#L158-L168
    # Since it is unusual thing to do, we just reassign them to parameters
    # torch_geometric 模型将参数注册为张量，参考上述链接的实现细节
    # 由于这种做法比较不寻常，我们将它们重新分配为参数

    def state_dict_hook(module, destination, prefix, local_metadata):
        for name, param in module.named_parameters():
            if isinstance(destination[name], torch.Tensor) and not isinstance(
                destination[name], torch.nn.Parameter
            ):
                destination[name] = torch.nn.Parameter(destination[name])

    model._register_state_dict_hook(state_dict_hook)
    # 注册状态字典钩子，用于将非参数张量转换为参数

def setup_torchbench_cwd():
    original_dir = abspath(os.getcwd())
    # 获取当前工作目录的绝对路径

    os.environ["KALDI_ROOT"] = "/tmp"  # avoids some spam
    # 设置环境变量 KALDI_ROOT 避免某些不必要的输出

    for torchbench_dir in (
        "./torchbenchmark",
        "../torchbenchmark",
        "../torchbench",
        "../benchmark",
        "../../torchbenchmark",
        "../../torchbench",
        "../../benchmark",
    ):
        if exists(torchbench_dir):
            break
    # 在预定义目录中查找 torchbenchmark 文件夹的存在

    if exists(torchbench_dir):
        torchbench_dir = abspath(torchbench_dir)
        os.chdir(torchbench_dir)
        sys.path.append(torchbench_dir)
    # 如果找到了 torchbenchmark 目录，则切换到该目录并将其添加到系统路径中

    return original_dir
    # 返回最初的工作目录路径

@functools.lru_cache(maxsize=1)
def load_yaml_file():
    filename = "torchbench.yaml"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    # 构建 torchbench.yaml 文件的完整路径

    with open(filepath) as f:
        data = yaml.safe_load(f)
    # 使用安全加载方式读取 YAML 文件内容

    def flatten(lst):
        for item in lst:
            if isinstance(item, list):
                yield from flatten(item)
            else:
                yield item
    # 定义一个展开列表的生成器函数

    def maybe_list_to_set(obj):
        if isinstance(obj, dict):
            return {k: maybe_list_to_set(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return set(flatten(obj))
        return obj
    # 定义一个函数，将可能的列表转换为集合

    return maybe_list_to_set(data)
    # 返回处理后的数据集合

def process_hf_reformer_output(out):
    assert isinstance(out, list)
    # 断言输出 out 是一个列表
    # second output is unstable
    # 第二个输出不稳定
    return [elem for i, elem in enumerate(out) if i != 1]
    # 返回一个去除第二个元素的输出列表

def process_hf_whisper_output(out):
    out_ret = []
    # 初始化空列表 out_ret
    # 遍历列表 out 中的元素，同时获取元素的索引 i 和元素本身 elem
    for i, elem in enumerate(out):
        # 如果当前索引 i 等于 0
        if i == 0:
            # 断言 elem 是一个字典类型
            assert isinstance(elem, dict)
            # 将 elem 中除去键为 "logits" 的项组成新的字典，添加到 out_ret 列表中
            out_ret.append({k: v for k, v in elem.items() if k != "logits"})
        # 如果当前索引 i 不等于 1
        elif i != 1:
            # 直接将 elem 添加到 out_ret 列表中
            out_ret.append(elem)

    # 返回处理后的列表 out_ret
    return out_ret
# 创建一个字典，将不同的处理函数与各自的模型名称对应起来，用于处理训练模型的输出
process_train_model_output = {
    "hf_Reformer": process_hf_reformer_output,
    "hf_Whisper": process_hf_whisper_output,
}

# 定义一个继承自BenchmarkRunner的TorchBenchmarkRunner类
class TorchBenchmarkRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "torchbench"  # 设置性能基准测试套件名称为torchbench
        self.optimizer = None  # 初始化优化器为None

    @property
    def _config(self):
        return load_yaml_file()  # 加载并返回配置文件的内容

    @property
    def _skip(self):
        return self._config["skip"]  # 返回配置文件中的skip部分内容

    @property
    def _batch_size(self):
        return self._config["batch_size"]  # 返回配置文件中的batch_size值

    @property
    def _tolerance(self):
        return self._config["tolerance"]  # 返回配置文件中的tolerance值

    @property
    def _accuracy(self):
        return self._config["accuracy"]  # 返回配置文件中的accuracy值

    @property
    def skip_models(self):
        return self._skip["all"]  # 返回配置文件中skip部分的所有模型列表

    @property
    def skip_models_for_cpu(self):
        return self._skip["device"]["cpu"]  # 返回配置文件中skip部分CPU设备相关的模型列表

    @property
    def skip_models_for_cuda(self):
        return self._skip["device"]["cuda"]  # 返回配置文件中skip部分CUDA设备相关的模型列表

    @property
    def skip_models_for_freezing(self):
        return self._skip["freezing"]  # 返回配置文件中skip部分冻结相关的模型列表

    @property
    def slow_models(self):
        return self._config["slow"]  # 返回配置文件中标记为慢速的模型列表

    @property
    def very_slow_models(self):
        return self._config["very_slow"]  # 返回配置文件中标记为非常慢速的模型列表

    @property
    def non_deterministic_models(self):
        return self._config["non_deterministic"]  # 返回配置文件中标记为非确定性的模型列表

    @property
    def get_output_amp_train_process_func(self):
        return process_train_model_output  # 返回处理训练模型输出的函数字典

    @property
    def skip_not_suitable_for_training_models(self):
        return self._skip["test"]["training"]  # 返回配置文件中不适合训练的模型列表

    @property
    def failing_fx2trt_models(self):
        return self._config["trt_not_yet_working"]  # 返回配置文件中尚未支持TensorRT的模型列表

    @property
    def force_amp_for_fp16_bf16_models(self):
        return self._config["dtype"]["force_amp_for_fp16_bf16_models"]  # 返回配置文件中需要强制使用AMP的FP16/BF16模型列表

    @property
    def force_fp16_for_bf16_models(self):
        return self._config["dtype"]["force_fp16_for_bf16_models"]  # 返回配置文件中需要强制使用FP16的BF16模型列表

    @property
    def skip_accuracy_checks_large_models_dashboard(self):
        if self.args.dashboard or self.args.accuracy:
            return self._accuracy["skip"]["large_models"]  # 若启用了dashboard或者accuracy选项，返回配置文件中大模型跳过的准确性检查列表
        return set()

    @property
    def skip_accuracy_check_as_eager_non_deterministic(self):
        if self.args.accuracy and self.args.training:
            return self._accuracy["skip"]["eager_not_deterministic"]  # 若同时启用了accuracy和training选项，返回配置文件中急切非确定性跳过的准确性检查列表
        return set()

    @property
    def skip_multiprocess_models(self):
        return self._skip["multiprocess"]  # 返回配置文件中多进程模型跳过的列表

    @property
    def skip_models_due_to_control_flow(self):
        return self._skip["control_flow"]  # 返回配置文件中由于控制流问题跳过的模型列表

    @property
    def guard_on_nn_module_models(self):
        return {
            "vision_maskrcnn",
        }  # 返回一个包含特定模型名称的集合，用于保护NN模块模型
    # 返回一个包含多个预置模型名称的集合
    def inline_inbuilt_nn_modules_models(self):
        return {
            "basic_gnn_edgecnn",
            "drq",
            "hf_Reformer",
            "DALLE2_pytorch",
            "hf_BigBird",
            "detectron2_maskrcnn_r_50_fpn",
            "detectron2_maskrcnn_r_101_fpn",
            "vision_maskrcnn",
            "doctr_reco_predictor",
        }

    # 加载指定模型，并返回加载后的模型对象
    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
        extra_args=None,
    ):
        from torchbenchmark import _list_canary_model_paths, _list_model_paths

        # 获取所有模型的路径
        models = _list_model_paths()
        # 添加特定的“金丝雀模型”路径
        models += [
            f
            for f in _list_canary_model_paths()
            if os.path.basename(f) in self._config["canary_models"]
        ]
        # 按字母顺序排序模型路径列表
        models.sort()

        # 计算用于基准测试的模型索引范围
        start, end = self.get_benchmark_indices(len(models))
        # 迭代指定范围内的模型名称
        for index, model_path in enumerate(models):
            if index < start or index >= end:
                continue

            # 获取模型的基本名称
            model_name = os.path.basename(model_path)
            # 根据过滤条件跳过不需要加载的模型
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in args.exclude_exact
                or model_name in self.skip_models
            ):
                continue

            # 生成当前模型名称
            yield model_name

    # 迭代器函数，生成模型名称列表
    def iter_model_names(self, args):
        from torchbenchmark import _list_canary_model_paths, _list_model_paths

        # 获取所有模型的路径
        models = _list_model_paths()
        # 添加特定的“金丝雀模型”路径，并检查其是否在配置中指定
        models += [
            f
            for f in _list_canary_model_paths()
            if os.path.basename(f) in self._config["canary_models"]
        ]
        # 按字母顺序排序模型路径列表
        models.sort()

        # 计算用于基准测试的模型索引范围
        start, end = self.get_benchmark_indices(len(models))
        # 迭代指定范围内的模型名称
        for index, model_path in enumerate(models):
            if index < start or index >= end:
                continue

            # 获取模型的基本名称
            model_name = os.path.basename(model_path)
            # 根据过滤条件跳过不需要加载的模型
            if (
                not re.search("|".join(args.filter), model_name, re.I)
                or re.search("|".join(args.exclude), model_name, re.I)
                or model_name in args.exclude_exact
                or model_name in self.skip_models
            ):
                continue

            # 生成当前模型名称
            yield model_name

    # 根据名称和训练状态选择梯度处理方法
    def pick_grad(self, name, is_training):
        if is_training or name in ("maml",):
            return torch.enable_grad()
        else:
            return torch.no_grad()

    # 根据训练状态、当前设备和模型名称获取误差容差和余弦相似度标志
    def get_tolerance_and_cosine_flag(self, is_training, current_device, name):
        tolerance = 1e-4
        cosine = self.args.cosine
        # 如果使用 float16 或者自动混合精度训练，增加 torch allclose 的容差
        if self.args.float16 or self.args.amp:
            if name in self._tolerance["higher_fp16"]:
                return 1e-2, cosine
            return 1e-3, cosine

        # 如果使用 bfloat16，根据模型指定较高的容差
        if self.args.bfloat16:
            if name in self._tolerance["higher_bf16"]:
                return 1e-2, cosine

        # 对于训练状态且在 GPU 或 XPU 上，调整容差和余弦相似度标志
        if is_training and (current_device == "cuda" or current_device == "xpu"):
            tolerance = 1e-3
            if name in self._tolerance["cosine"]:
                cosine = True
            elif name in self._tolerance["higher"]:
                tolerance = 1e-3
            elif name in self._tolerance["even_higher"]:
                tolerance = 8 * 1e-2
        return tolerance, cosine

    # 计算模型预测的损失值
    def compute_loss(self, pred):
        return reduce_to_scalar_loss(pred)

    # 执行模型前向传播，支持输入为字典或列表
    def forward_pass(self, mod, inputs, collect_outputs=True):
        # 根据自动混合精度设置上下文
        with self.autocast(**self.autocast_arg):
            if isinstance(inputs, dict):
                return mod(**inputs)
            else:
                return mod(*inputs)
    # 定义一个方法，执行模型的前向和反向传播
    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        # 克隆输入以确保不改变原始输入数据
        cloned_inputs = clone_inputs(inputs)
        
        # 调用优化器的方法将模型参数的梯度清零
        self.optimizer_zero_grad(mod)
        
        # 使用自动混合精度加速（如果已启用），配置参数由self.autocast_arg提供
        with self.autocast(**self.autocast_arg):
            # 根据输入类型调用模型进行预测
            if isinstance(cloned_inputs, dict):
                pred = mod(**cloned_inputs)
            else:
                pred = mod(*cloned_inputs)
            
            # 计算模型预测产生的损失
            loss = self.compute_loss(pred)
        
        # 使用梯度缩放器对损失进行反向传播
        self.grad_scaler.scale(loss).backward()
        
        # 优化器执行一步更新模型参数
        self.optimizer_step()
        
        # 如果指定收集输出，则调用collect_results函数收集相关输出信息并返回
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        
        # 否则返回空值
        return None
# 定义主函数，用于执行 TorchBench 主要任务
def torchbench_main():
    # 设置当前工作目录为 TorchBench 的初始目录，并保存原始目录路径
    original_dir = setup_torchbench_cwd()
    # 设置日志记录的基本配置，设定日志级别为警告
    logging.basicConfig(level=logging.WARNING)
    # 忽略警告信息的输出
    warnings.filterwarnings("ignore")
    # 调用主函数，运行 TorchBenchmarkRunner 的实例，并传入原始目录路径
    main(TorchBenchmarkRunner(), original_dir)

# 如果当前脚本作为主程序运行，则执行 torchbench_main 函数
if __name__ == "__main__":
    torchbench_main()
```