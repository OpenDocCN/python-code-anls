# `.\pytorch\benchmarks\gpt_fast\generate.py`

```
import dataclasses  # 导入用于定义数据类的模块
import itertools  # 导入用于迭代操作的模块
import time  # 导入时间相关的模块
from typing import Optional, Tuple  # 导入类型提示相关的类和元组

from mixtral_moe_model import ConditionalFeedForward, Transformer as MixtralMoE  # 导入MixtralMoE模型相关的类
from mixtral_moe_quantize import (
    ConditionalFeedForwardInt8,
    WeightOnlyInt8QuantHandler as MixtralMoEWeightOnlyInt8QuantHandler,
)  # 导入MixtralMoE模型量化处理相关的类
from model import Transformer as LLaMA  # 导入LLaMA模型的Transformer类
from quantize import WeightOnlyInt8QuantHandler as LLaMAWeightOnlyInt8QuantHandler  # 导入LLaMA模型量化处理相关的类

import torch  # 导入PyTorch库
import torch._inductor.config  # 导入PyTorch的内部配置模块

torch._inductor.config.coordinate_descent_tuning = True  # 设置PyTorch的配置选项
torch._inductor.config.triton.unique_kernel_names = True  # 设置PyTorch的配置选项
torch._inductor.config.fx_graph_cache = True  # 实验性功能，用于减少编译时间，未来将默认启用
torch._inductor.config.assert_indirect_indexing = False  # 设置PyTorch的配置选项，禁用间接索引断言


@dataclasses.dataclass
class GPTModelConfig:
    name: str  # 模型名称
    module: type  # 模型类型
    mode: Optional[str]  # 可选的模式字符串
    quantizer: type  # 量化器类型
    token_per_sec: float  # 每秒处理的标记数
    memory_bandwidth: float  # 内存带宽
    compilation_time: float  # 编译时间


def device_sync(device):
    if "cuda" in device:  # 如果设备包含'cuda'，同步CUDA设备
        torch.cuda.synchronize(device)
    elif "cpu" in device:  # 如果设备包含'cpu'，不进行同步
        pass
    else:  # 对于不支持的设备类型，输出错误信息
        print(f"device={device} is not yet suppported")


def multinomial_sample_one_no_sync(
    probs_sort,
):  # 执行无CUDA同步的多项式抽样
    q = torch.empty_like(probs_sort).exponential_(1)  # 创建指数分布的随机张量
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)  # 计算概率比率的最大值索引


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)  # 对logits进行温度缩放

    if top_k is not None:  # 如果指定了top_k参数
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))  # 获取logits中最大的前top_k个值
        pivot = v.select(-1, -1).unsqueeze(-1)  # 选择最后一个最大值并扩展为张量
        logits = torch.where(logits < pivot, -float("Inf"), logits)  # 将低于最大值的logits设置为负无穷
    probs = torch.nn.functional.softmax(logits, dim=-1)  # 计算softmax概率
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[0, -1], temperature, top_k)  # 将logits转换为概率分布
    idx_next = multinomial_sample_one_no_sync(probs)  # 执行无同步的多项式抽样
    return idx_next, probs  # 返回抽样的索引和概率


@torch.compile(fullgraph=True)
def prefill(
    model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> torch.Tensor:
    # input_pos: [B, S]，输入位置编码是一个形状为[B, S]的张量
    logits = model(x, input_pos)  # 使用模型计算logits
    return sample(logits, **sampling_kwargs)[0]  # 返回抽样的索引


@torch.compile(fullgraph=True, mode="reduce-overhead")
def decode_one_token(
    model: torch.nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]，输入位置编码是一个形状为[B, 1]的张量
    assert input_pos.shape[-1] == 1  # 断言输入位置编码的形状的最后一个维度为1
    logits = model(x, input_pos)  # 使用模型计算logits
    return sample(logits, **sampling_kwargs)  # 返回抽样的索引和概率


def decode_n_tokens(
    model: torch.nn.Module,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    **sampling_kwargs,
):
    new_tokens, new_probs = [], []  # 初始化新标记列表和概率列表
    # 对于每一个需要生成的新 token，执行以下操作
    for i in range(num_new_tokens):
        # 使用指定的 SDP 后端（数学后端）创建一个注意力机制的内核
        with torch.nn.attention.sdpa_kernel(
            torch.nn.attention.SDPBackend.MATH
        ):  
            # 调用 decode_one_token 函数，生成下一个 token 和对应的概率
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
            # 更新输入位置
            input_pos += 1
            # 将生成的下一个 token 添加到新 token 列表中（深拷贝）
            new_tokens.append(next_token.clone())
            # 将生成的下一个 token 对应的概率添加到新概率列表中（深拷贝）
            new_probs.append(next_prob.clone())
            # 更新当前 token 为生成的下一个 token 的视图
            cur_token = next_token.view(1, -1)

    # 返回生成的所有新 token 和对应的概率列表
    return new_tokens, new_probs
# 使用 @torch.no_grad() 装饰器，确保在生成过程中不会计算梯度
@torch.no_grad()
# 生成文本序列的函数，接受模型、初始提示、最大新标记数量等参数，并返回生成的文本序列张量
def generate(
    model: torch.nn.Module, prompt: torch.Tensor, max_new_tokens: int, **sampling_kwargs
) -> torch.Tensor:
    # 获取初始提示张量的设备和数据类型
    device, dtype = prompt.device, prompt.dtype
    # 获取初始提示的长度
    T = prompt.size(0)
    # 计算生成文本的最终长度
    T_new = T + max_new_tokens
    # 确定模型能处理的最大序列长度
    max_seq_length = min(T_new, model.config.block_size)

    # 将模型缓存设置为当前生成任务的要求
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    # 创建一个形状符合最终生成文本的空张量，并将初始提示复制到其中
    empty = torch.empty(T_new, dtype=dtype, device=device)
    empty[:T] = prompt
    seq = empty
    # 为输入位置创建张量，表示当前已生成的序列长度
    input_pos = torch.arange(0, T, device=device)

    # 预测下一个标记，使用预填充函数，并将其放入序列中
    next_token = prefill(model, prompt.view(1, -1), input_pos, **sampling_kwargs)
    seq[T] = next_token

    # 更新输入位置，表示生成了一个新标记
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    # 解码生成指定数量的新标记，并将它们填充到序列的末尾
    generated_tokens, _ = decode_n_tokens(
        model, next_token.view(1, -1), input_pos, max_new_tokens - 1, **sampling_kwargs
    )
    seq[T + 1 :] = torch.cat(generated_tokens)
    return seq


# 加载模型的私有函数，接受 GPTModelConfig 对象和设备类型，默认为 "cuda"，精度为 torch.bfloat16
def _load_model(x: GPTModelConfig, device="cuda", precision=torch.bfloat16):
    # 使用 "meta" 设备上下文加载模型
    with torch.device("meta"):
        model = x.module.from_name(x.name)
    # 将模型转移到指定精度上
    model = model.to(dtype=precision)

    # 如果模型使用 int8 模式，则进行权重量化
    if x.mode == "int8":
        print("Using int8 weight-only quantization!")
        model = x.quantizer(model).convert_for_runtime()

    # 获取模型的状态字典，并根据需要替换成随机张量的参数
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = torch.nn.Parameter(
            torch.randn(v.shape, device=device).to(dtype=v.dtype),
            requires_grad=v.requires_grad,
        )
    # 加载更新后的状态字典到模型中，并设置为评估模式
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


# 计算模型大小的私有函数，仅计算激活的参数和缓冲区的大小
def _get_model_size(model):
    model_size = 0
    # 遍历模型的子模块，累加激活的参数和缓冲区的大小，排除嵌入层
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )

    # 如果模型配置中包含 num_experts 属性，修正模型大小，移除未激活的专家（适用于混合专家架构）
    if hasattr(model.config, "num_experts"):
        config = model.config
        for submodule in model.modules():
            if isinstance(
                submodule, (ConditionalFeedForward, ConditionalFeedForwardInt8)
            ):
                model_size -= (
                    sum(
                        [
                            p.numel() * p.dtype.itemsize
                            for p in itertools.chain(
                                submodule.parameters(), child.buffers()
                            )
                        ]
                    )
                    * (config.num_experts - config.num_activated_experts)
                    / config.num_experts
                )

    return model_size


# 运行实验的函数定义，接受 GPTModelConfig 对象和其他参数
def run_experiment(
    x: GPTModelConfig,
    # 定义一个整数变量，表示生成的样本数量，默认为5个
    num_samples: int = 5,
    # 定义一个整数变量，表示每个样本最大生成的新标记数，默认为200个
    max_new_tokens: int = 200,
    # 定义一个整数变量，表示生成文本时考虑的最高概率前几个标记，默认为200个
    top_k: int = 200,
    # 定义一个浮点数变量，表示生成文本时的温度参数，默认为0.8
    temperature: float = 0.8,
    # 定义一个字符串变量，表示运行模型的设备类型，默认为"cuda"
    device: str = "cuda",
# 定义函数 run_experiment，用于运行实验并评估模型性能
def run_experiment(x) -> None:
    # 打印模型加载信息
    print(f"Loading model {x.name}")
    # 记录开始加载模型的时间
    t0 = time.time()
    # 载入模型
    model = _load_model(x)
    # 同步设备（这里的 MKG 是一个注释或标记）
    device_sync(device=device)  # MKG
    # 打印加载模型所需时间
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # 定义一个预设的输入提示（prompt）
    prompt = torch.tensor(
        [1, 15043, 29892, 590, 1024, 338], device=device, dtype=torch.int32
    )
    # 获取输入提示的长度
    prompt_length = prompt.size(0)

    # 设置随机种子
    torch.manual_seed(1234)
    # 获取模型大小
    model_size = _get_model_size(model)

    # 初始化聚合指标字典
    aggregate_metrics = {"tokens_per_sec": [], "memory_bandwidth": []}
    # 设置起始值为 -1
    start = -1
    # 编译时间初始化为 None
    compilation_time = None

    # 进行循环，运行多个样本
    for i in range(start, num_samples):
        # 同步设备（这里的 MKG 是一个注释或标记）
        device_sync(device=device)  # MKG

        # 记录开始生成的时间
        t0 = time.perf_counter()
        # 生成输出结果 y
        y = generate(
            model, prompt, max_new_tokens, temperature=temperature, top_k=top_k
        )

        # 如果是第一次循环（i == -1），记录编译时间并跳过后续步骤
        if i == -1:
            compilation_time = time.perf_counter() - t0
            print(f"Compilation time: {compilation_time:.2f} seconds")
            continue

        # 同步设备（这里的 MKG 是一个注释或标记）
        device_sync(device=device)  # MKG
        # 计算生成 y 所需的时间
        t = time.perf_counter() - t0
        # 计算生成的令牌数量
        tokens_generated = y.size(0) - prompt_length
        # 计算每秒生成的令牌数
        tokens_sec = tokens_generated / t
        # 将每秒生成的令牌数添加到聚合指标中
        aggregate_metrics["tokens_per_sec"].append(tokens_sec)
        # 计算内存带宽，并添加到聚合指标中
        aggregate_metrics["memory_bandwidth"].append(model_size * tokens_sec / 1e9)

    # 计算平均每秒生成的令牌数
    token_per_sec = torch.mean(torch.tensor(aggregate_metrics["tokens_per_sec"])).item()
    # 计算平均内存带宽
    memory_bandwidth = torch.mean(
        torch.tensor(aggregate_metrics["memory_bandwidth"])
    ).item()
    # 打印平均每秒生成的令牌数
    print(f"Average tokens/sec: {token_per_sec:.2f} tokens/sec")
    # 打印实现的平均带宽
    print(f"Average bandwidth achieved: {memory_bandwidth:.02f} GB/s")
    # 打印所使用的内存
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
    # 返回每秒生成的令牌数、平均内存带宽和编译时间
    return token_per_sec, memory_bandwidth, compilation_time


# 定义函数 run_llama2_7b_bf16，用于在指定设备上运行 Llama-2-7b 模型的 bfloat16 实验
def run_llama2_7b_bf16(device: str = "cuda"):
    # 导入实验模块
    from benchmark import Experiment

    # 配置 Llama-2-7b 模型的参数
    model = GPTModelConfig(
        "Llama-2-7b-chat-hf",
        LLaMA,
        "bfloat16",
        LLaMAWeightOnlyInt8QuantHandler,
        94,
        1253,
        162,
    )
    # 运行实验并获取结果
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model)
    # 返回实验结果的列表
    return [
        Experiment(
            model.name,
            "token_per_sec",
            model.token_per_sec,
            f"{token_per_sec:.02f}",
            model.mode,
            device,
            True,
        ),
        Experiment(
            model.name,
            "memory_bandwidth(GB/s)",
            model.memory_bandwidth,
            f"{memory_bandwidth:.02f}",
            model.mode,
            device,
            True,
        ),
        Experiment(
            model.name,
            "compilation_time(s)",
            model.compilation_time,
            f"{compilation_time:.02f}",
            model.mode,
            device,
            True,
        ),
    ]


# 提示：token_per_sec 和 memory_bandwidth 的目标数值适用于 A100-40GB，与 typcial A100-80GB 不同。
# 定义一个函数，用于运行 LLAMA-2-7b 模型的 int8 版本的基准测试
def run_llama2_7b_int8(device: str = "cuda"):
    # 导入实验相关的模块
    from benchmark import Experiment

    # 创建一个 GPTModelConfig 实例，配置 LLAMA-2-7b 模型的参数
    model = GPTModelConfig(
        "Llama-2-7b-chat-hf",  # 模型名称
        LLaMA,  # 模型类型
        "int8",  # 模型精度
        LLaMAWeightOnlyInt8QuantHandler,  # 模型量化处理器
        144,  # token 每秒处理数
        957,  # 内存带宽（GB/s）
        172,  # 编译时间（秒）
    )

    # 运行基准测试并获取结果：token 每秒处理数、内存带宽、编译时间
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model)

    # 返回一个包含三个 Experiment 对象的列表，每个对象表示一个基准测试结果项
    return [
        Experiment(
            model.name,
            "token_per_sec",
            model.token_per_sec,
            f"{token_per_sec:.02f}",
            model.mode,
            device,
            True,
        ),
        Experiment(
            model.name,
            "memory_bandwidth(GB/s)",
            model.memory_bandwidth,
            f"{memory_bandwidth:.02f}",
            model.mode,
            device,
            True,
        ),
        Experiment(
            model.name,
            "compilation_time(s)",
            model.compilation_time,
            f"{compilation_time:.02f}",
            model.mode,
            device,
            True,
        ),
    ]


# token_per_sec and memory_bandwidth target numbers are for A100-40GB, which are different from the typical A100-80GB.
# 定义一个函数，用于运行 Mixtral-8x7B-v0.1 模型的 int8 版本的基准测试
def run_mixtral_8x7b_int8(device: str = "cuda"):
    # 导入实验相关的模块
    from benchmark import Experiment

    # 创建一个 GPTModelConfig 实例，配置 Mixtral-8x7B-v0.1 模型的参数
    # 注意：我们将原始的层数从 32 层减少到 16 层，以适应 CI 环境的内存限制。
    model = GPTModelConfig(
        "Mixtral-8x7B-v0.1",  # 模型名称
        MixtralMoE,  # 模型类型
        "int8",  # 模型精度
        MixtralMoEWeightOnlyInt8QuantHandler,  # 模型量化处理器
        175,  # token 每秒处理数
        1130,  # 内存带宽（GB/s）
        162,  # 编译时间（秒）
    )

    # 运行基准测试并获取结果：token 每秒处理数、内存带宽、编译时间
    token_per_sec, memory_bandwidth, compilation_time = run_experiment(model)

    # 返回一个包含三个 Experiment 对象的列表，每个对象表示一个基准测试结果项
    return [
        Experiment(
            model.name,
            "token_per_sec",
            model.token_per_sec,
            f"{token_per_sec:.02f}",
            model.mode,
            device,
            True,
        ),
        Experiment(
            model.name,
            "memory_bandwidth(GB/s)",
            model.memory_bandwidth,
            f"{memory_bandwidth:.02f}",
            model.mode,
            device,
            True,
        ),
        Experiment(
            model.name,
            "compilation_time(s)",
            model.compilation_time,
            f"{compilation_time:.02f}",
            model.mode,
            device,
            True,
        ),
    ]
```