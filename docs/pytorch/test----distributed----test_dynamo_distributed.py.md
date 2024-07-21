# `.\pytorch\test\distributed\test_dynamo_distributed.py`

```py
# 导入需要的模块和类
import contextlib  # 上下文管理工具模块
import copy  # 复制对象模块
import functools  # 函数装饰器模块
import random  # 随机数生成模块
import unittest  # 单元测试框架模块
from contextlib import contextmanager  # 上下文管理器类
from io import StringIO  # 字符串IO模块
from typing import List  # 类型提示模块中的列表类型
from unittest.mock import patch  # 单元测试模块中的模拟对象功能

import numpy as np  # 数值计算库numpy

import torch  # PyTorch深度学习库
import torch._dynamo  # PyTorch内部功能模块
import torch._dynamo.logging  # PyTorch内部日志模块
import torch._dynamo.test_case  # PyTorch内部测试用例模块
from torch import nn  # PyTorch神经网络模块
from torch._C import FileCheck  # PyTorch C++扩展中的文件检查工具
from torch._dynamo import config  # PyTorch内部配置模块
from torch._dynamo.backends.distributed import DDPOptimizer  # PyTorch分布式优化器模块
from torch._dynamo.comptime import comptime  # PyTorch编译时计算模块
from torch._dynamo.testing import collect_results  # PyTorch测试结果收集模块
from torch._dynamo.utils import same  # PyTorch工具函数模块
from torch._higher_order_ops.wrap import tag_activation_checkpoint  # PyTorch高阶操作模块
from torch.distributed._functional_collectives import _maybe_wrap_tensor  # PyTorch分布式函数集合模块
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # PyTorch全分片数据并行模块
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,  # PyTorch FSDP模块中的Lambda自动包装策略函数
    transformer_auto_wrap_policy,  # PyTorch FSDP模块中的Transformer自动包装策略函数
)
from torch.nn.parallel import DistributedDataParallel as DDP  # PyTorch分布式数据并行模块
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,  # 是否支持闪存注意力的平台支持标志
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,  # 是否支持内存效率注意力的平台支持标志
)
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,  # Dynamo分布式每个排名的初始化函数
    DynamoDistributedMultiProcTestCase,  # Dynamo多进程分布式测试用例类
    DynamoDistributedSingleProcTestCase,  # Dynamo单进程分布式测试用例类
    import_transformers_or_skip,  # 导入Transformers或跳过的函数
    requires_nccl,  # 需要NCCL库的装饰器函数
    skip_if_lt_x_gpu,  # 如果GPU数量小于x则跳过的装饰器函数
)
from torch.testing._internal.common_utils import requires_cuda  # 需要CUDA的装饰器函数
from torch.utils._triton import has_triton  # 是否有Triton加速的标志

# 重置随机数生成器状态的函数
def reset_rng_state():
    torch.manual_seed(1337)  # 设置PyTorch随机种子
    random.seed(1337)  # 设置Python随机种子
    np.random.seed(1337)  # 设置NumPy随机种子

# 初始化模型权重的函数
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # 使用Xavier均匀初始化线性层的权重
        m.bias.data.fill_(0.01)  # 将偏置项数据填充为0.01

# ToyModel类：一个简单的神经网络模型
class ToyModel(nn.Module):
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        super().__init__()
        self.ctx_manager = ctx_manager  # 上下文管理器对象
        # 定义网络结构
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )

    def forward(self, inputs):
        if self.ctx_manager is not None:
            with self.ctx_manager():  # 如果有上下文管理器，则在其下运行网络
                return self.net(inputs)
        else:
            return self.net(inputs)  # 否则直接运行网络

# 获取模型的函数
def get_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    m = ToyModel(  # 创建ToyModel实例
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)  # 将模型移动到指定设备
    m.apply(init_weights)  # 初始化模型的权重
    inputs = torch.rand(bsz, in_feat).to(device)  # 创建随机输入数据，并移动到指定设备
    outputs = m(inputs)  # 获取模型输出
    return m, inputs, outputs  # 返回模型、输入数据和输出数据的元组

# MutatingModel类：一个用于示例的神经网络模型
class MutatingModel(nn.Module):
    # 初始化函数，定义了神经网络的结构和初始状态
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        # 调用父类初始化方法
        super().__init__()
        # 设置上下文管理器
        self.ctx_manager = ctx_manager
        # 定义神经网络的层次结构，包括多个线性层和ReLU激活函数
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )
        # 设置初始状态为1
        self.state = 1

    # 前向传播函数，计算神经网络的输出
    def forward(self, inputs):
        # 将状态设置为2
        self.state = 2
        # 执行神经网络的前向传播，并乘以当前状态值
        return self.net(inputs) * self.state
# 创建一个可变模型并返回其实例、输入和输出
def get_mutating_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    # 创建 MutatingModel 实例，设定输入特征、隐藏特征、输出特征以及上下文管理器，并将其移到指定设备上
    m = MutatingModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    # 对模型应用初始化权重函数
    m.apply(init_weights)
    # 创建随机输入数据，并将其移到指定设备上
    inputs = torch.rand(bsz, in_feat).to(device)
    # 使用模型处理输入数据，生成输出结果
    outputs = m(inputs)
    # 返回模型实例、输入数据和输出结果
    return m, inputs, outputs


class ToyInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 创建包含两个线性层的列表，并通过 nn.Sequential 封装成序列模块
        self.layers = [nn.Linear(100, 100), nn.Linear(100, 100)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        # 前向传播函数，通过序列模块处理输入数据并返回结果
        return self.layers(inputs)


class ToyOuterModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        # 使用列表推导式创建两个 ToyInnerModel 实例，并将它们移到指定设备上
        self.layers = [ToyInnerModel().to(device) for _ in range(2)]
        # 使用 nn.Sequential 将两个 ToyInnerModel 实例和 ReLU 激活函数串联起来成为外部模型的层次结构
        self.layers = nn.Sequential(
            self.layers[0], nn.ReLU(), self.layers[1], nn.ReLU()
        )

    def forward(self, inputs):
        # 外部模型的前向传播函数，通过层次结构处理输入数据并返回结果
        return self.layers(inputs)


# 创建一个玩具模型用于激活检查点技术，并返回模型实例和输入数据
def get_toy_model_for_activation_checkpointing(device):
    m = ToyOuterModel(device).to(device)
    # 对模型应用初始化权重函数
    m.apply(init_weights)
    # 创建随机输入数据，并将其移到指定设备上
    inputs = torch.rand(100, 100).to(device)
    # 返回模型实例和输入数据
    return m, inputs


# 在图模型中查找第一个符合给定函数的节点并返回
def find_first_node(gm, func):
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    # 若未找到符合条件的节点，则返回 None
    return None


# 应用带有检查点技术的 FSDP 模型，并返回结果模型
def apply_fsdp_with_checkpointing(
    model, wrap_policy, checkpoint_policy, use_activation_checkpointing=True
):
    # 导入必要的检查点相关函数和类
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    # 对模型进行深拷贝，并用 FSDP 封装，指定自动包装策略和使用原始参数
    model = FSDP(
        copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
    )
    # 若启用激活检查点技术，则设置检查点包装器函数和检查点策略函数
    if use_activation_checkpointing:
        checkpoint_wrapper_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=checkpoint_policy,
        )
    # 返回应用了检查点技术的 FSDP 封装模型
    return model


# 创建一个自定义模型类，包含一个带有自定义权重的线性层
def get_custom_model(device):
    class MyCustomLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 声明一个带有随机初始权重的参数化线性层
            self.weight = nn.Parameter(torch.randn(512, 512))

        def forward(self, x):
            # 前向传播函数，执行矩阵乘法和条件判断，返回结果
            tmp = torch.mm(x, self.weight.t())
            return tmp + torch.where(tmp < 0.5, 0.3, 0.6)

    class MyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # 声明一个具有输入和输出尺寸的线性层
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            # 前向传播函数，应用线性层处理输入并返回结果
            return self.linear(x)
    # 定义一个名为 MyModule 的自定义模块，继承自 torch.nn.Module
    class MyModule(torch.nn.Module):
        # 初始化方法
        def __init__(self):
            # 调用父类的初始化方法
            super().__init__()
            # 定义一个模块列表 mods，包含三对自定义模块和 ReLU 激活函数
            mods = [
                (MyLinear(), torch.nn.ReLU()),  # 第一对模块和激活函数
                # 在自定义模块中间放置一个自定义线性层，确保在两者之前和之后都有激活函数
                (MyCustomLinear(), torch.nn.ReLU()),  # 中间的自定义线性层和激活函数
                (MyLinear(), torch.nn.ReLU()),  # 最后一对模块和激活函数
            ]
            # 使用 torch.nn.Sequential 将 mods 中的模块和激活函数按顺序连接起来
            self.seq = torch.nn.Sequential(*[x for items in mods for x in items])

        # 前向传播方法
        def forward(self, x, y):
            # 对特殊情况进行测试：当第0个桶（接近图输入的层）已满，需要触发一个新的桶，
            # 但没有参数的微不足道的操作可放入新的桶中。通过将这个“空桶”与前一个满的桶合并来优化此情况。
            return self.seq(x + y)  # 执行序列中的前向传播

    # 创建 MyModule 的实例 m，并将其移到指定的设备上
    m = MyModule().to(device)
    # 对模块应用初始化函数 init_weights
    m.apply(init_weights)
    # 创建一个随机输入张量，大小为 (512, 512)，并将其移到指定的设备上
    inputs = torch.rand((512, 512)).to(device)
    # 对输入进行测试：复制输入数据，生成一个输入元组
    inputs = (inputs, inputs)
    # 计算正确的输出，调用模块 m 的 __call__ 方法
    correct_outputs = m(*inputs)
    # 返回模块 m、输入 inputs 和正确的输出 correct_outputs
    return m, inputs, correct_outputs
# 定义一个函数，获取带有遮盖语言模型（Masked Language Model, MLM）功能的Hugging Face BERT模型
def get_hf_bert(rank):
    # 注意：如果在多进程测试中使用此函数，请在测试用例中使用 @import_transformers_or_skip 装饰器
    try:
        # 尝试导入所需的库和模块
        from transformers import AutoModelForMaskedLM, BertConfig
    except ImportError as e:
        # 如果导入失败，抛出跳过测试的异常，并附加导入错误信息
        raise unittest.SkipTest("Unable to import transformers") from e

    # 设置批量大小、最大长度、BERT配置和设备
    batch_size, max_length, config, device = 4, 512, BertConfig(), f"cuda:{rank}"
    # 从配置创建并移动到指定设备的MLM BERT模型
    model = AutoModelForMaskedLM.from_config(config).to(device)
    # 生成随机的输入ID和解码ID，并移动到指定设备
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
        device
    )
    # 构建模型输入字典
    inputs = {"input_ids": input_ids, "labels": decoder_ids}
    # 设置模型处于训练模式
    model.train()
    # 返回模型及其输入字典
    return model, inputs


# 模拟一个编译检查类，用于计数编译函数的调用次数
class CheckSplitsCompiler:
    def __init__(self):
        self.compiler_called = 0

    # 编译函数，简单地增加编译调用次数并返回输入的模型
    def compile_fn(self, gm, example_inputs):
        self.compiler_called += 1
        return gm


# 模拟的分布式数据并行（DDP）类，用于优化模型训练过程
class FakeDDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # 设置存储桶的最大字节数，以MB为单位
        bucket_cap_mb = 25
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

    # 上下文管理器，用于设置和恢复DDP优化过程中的活动模块
    @contextmanager
    def _inside_ddp_forward(self):
        DDP._active_ddp_module = self
        try:
            yield
        finally:
            DDP._active_ddp_module = None

    # 前向传播函数，包装在DDP优化环境中执行模型的前向传播
    def forward(self, *inputs, **kwargs):
        with self._inside_ddp_forward():
            return self.module.forward(*inputs, **kwargs)


# 运行Hugging Face BERT模型在DDP环境中的训练和优化，并进行结果的验证
def run_hf_bert_ddp(self, model, inputs, backend):
    # 重置随机数生成器状态
    reset_rng_state()
    # 使用模型处理输入数据，获取正确的输出
    correct_outputs = model(**inputs)
    # 获取正确输出的损失值
    correct_loss = correct_outputs.loss
    # 反向传播正确损失值
    correct_loss.backward()

    # 重置随机数生成器状态
    reset_rng_state()
    # 使用Dynamo库优化模型
    opt_model = torch._dynamo.optimize(backend)(model)
    # 使用优化后的模型处理输入数据，获取优化后的输出
    opt_outputs = opt_model(**inputs)
    # 获取优化后输出的损失值
    opt_loss = opt_outputs.loss
    # 反向传播优化后的损失值
    opt_loss.backward()

    # 提取输入数据的扁平化列表
    inputs_flat = [inputs[k] for k in inputs]
    # 收集正确和优化后的模型结果
    correct_results = collect_results(
        model, correct_outputs.logits, correct_loss, inputs_flat
    )
    opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
    # 使用断言验证正确和优化后的结果是否相同
    self.assertTrue(same(correct_results, opt_results))


# 测试类，用于验证FakeDDP在单进程环境中的表现
class TestFakeDistributedSingleProc(torch._dynamo.test_case.TestCase):
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor(self):
        # 获取Hugging Face BERT模型和输入数据
        model, inputs = get_hf_bert(0)
        # 包装模型在FakeDDP中
        model = FakeDDP(model)
        # 运行Hugging Face BERT模型在FakeDDP环境中的训练和优化
        run_hf_bert_ddp(self, model, inputs, "inductor")

    @patch.object(config, "optimize_ddp", True)
    # 定义测试方法 test_hf_bert_ddp_aot_eager，用于测试 HF BERT 模型在 AOT eager 模式下的运行
    def test_hf_bert_ddp_aot_eager(self):
        # 调用 get_hf_bert 函数获取 HF BERT 模型及其输入
        model, inputs = get_hf_bert(0)
        # 使用 FakeDDP 包装模型，模拟分布式数据并行环境
        model = FakeDDP(model)
        # 运行 HF BERT 模型在 DDP 策略下的 aot_eager 优化模式
        run_hf_bert_ddp(self, model, inputs, "aot_eager")

    # 定义测试方法 test_issue90375，用于验证在配置 optimize_ddp 为 True 时的特定问题
    @patch.object(config, "optimize_ddp", True)
    def test_issue90375(self):
        # 定义简单的模型类 Model，继承自 nn.Module
        class Model(nn.Module):
            # 定义模型的前向传播方法
            def forward(self):
                # 返回一个随机生成的张量乘以另一个随机生成的张量
                return torch.randn(3) * torch.randn(3)

        # 创建 Model 类的实例
        model = Model()
        # 使用 FakeDDP 包装模型，模拟分布式数据并行环境
        model = FakeDDP(model)

        # 使用 torch._dynamo.optimize("aot_eager") 对模型进行即时编译优化
        opt_model = torch._dynamo.optimize("aot_eager")(model)
        # 执行优化后的模型
        opt_model()

    # 定义测试方法 test_symbol_splitting，用于测试符号拆分的情况
    @patch.object(config, "optimize_ddp", True)
    def test_symbol_splitting(self):
        # 定义包含两个权重参数的模型类 Model，继承自 nn.Module
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            # 定义模型的前向传播方法
            def forward(self, x):
                # 将输入张量 x 进行复制拼接操作
                x = torch.cat([x, x])
                # 计算 x 与 weight1 的矩阵乘积
                y = x @ self.weight1
                # 计算 x 与 weight2 的矩阵乘积后加到 x 上
                z = x + y @ self.weight2
                return z

        # 创建 Model 类的实例
        model = Model()
        # 使用 FakeDDP 包装模型，模拟分布式数据并行环境
        model = FakeDDP(model)

        # 使用 torch.compile(dynamic=True) 对模型进行动态即时编译
        opt_model = torch.compile(dynamic=True)(model)
        # 执行优化后的模型，并传入指定形状的输入张量
        opt_model(torch.randn(20, 512))

    # 定义测试方法 test_call_method_forward，用于测试模型的前向传播方法调用
    @patch.object(config, "optimize_ddp", True)
    def test_call_method_forward(self):
        # 定义包含多层处理器的模型类 Model，继承自 nn.Module
        class Model(nn.Module):
            def __init__(
                self,
            ):
                super().__init__()
                layers = []
                for l in range(2):
                    # 每层包含 LayerNorm 和 MultiheadAttention 两个处理器
                    layer = nn.ModuleList(
                        [
                            nn.LayerNorm(96),
                            nn.MultiheadAttention(
                                embed_dim=96, num_heads=4, batch_first=True
                            ),
                        ]
                    )
                    layers.append(layer)
                self.layers = nn.ModuleList(layers)

            # 定义模型的前向传播方法，接受一个形状为 [Batch, Freq, Time, Feature] 的张量 x
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: [Batch, Freq, Time, Feature]
                B, F, T, H = x.shape
                for m in self.layers:
                    # 将输入张量 x 重塑为 [Batch*Freq, Time, Feature] 的形状
                    x = x.reshape(B * F, T, H)
                    # 对 x 应用 LayerNorm 处理器
                    x = m[0](x)
                    # 使用 MultiheadAttention 处理器对 x 进行处理，并返回处理结果及注意力分布
                    x, attn = m[1].forward(x, x, x)
                    # 将处理后的 x 重塑回 [Batch, Freq, Time, Feature] 的形状
                    x = x.reshape(B, F, T, H)
                return x

        # 创建 Model 类的实例
        model = Model()
        # 使用 FakeDDP 包装模型，模拟分布式数据并行环境
        model = FakeDDP(model)
        # 使用 torch.compile 对模型进行编译优化
        opt_model = torch.compile(model)
        # 执行优化后的模型，并传入指定形状的输入张量
        opt_model(torch.randn(2, 129, 100, 96))
# 如果这些测试失败了？检查并查看 TestFakeDistributedSingleProc 是否有单进程版本；
# 如果问题仅限于 Dynamo 分布式优化器，你应该可以在单进程中重现它！

# 使用 requires_nccl 装饰器确保测试依赖的 NCCL 库可用
@requires_nccl()
# TestMultiProc 类继承自 DynamoDistributedMultiProcTestCase，是多进程测试用例的基类
class TestMultiProc(DynamoDistributedMultiProcTestCase):
    """
    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """
    
    # 装饰器 skip_if_lt_x_gpu(2)：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 使用 patch.object 修改配置项，禁用 DDP 优化
    @patch.object(config, "optimize_ddp", False)
    # 测试方法：验证 DDP 基线、AOT（Ahead of Time）和 Eager 模式的多进程执行
    def test_ddp_baseline_aot_eager_multiprocess(self):
        # 在每个进程中初始化 Dynamo 分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 验证 optimize_ddp 是否为 False
            self.assertFalse(config.optimize_ddp)
            # 获取模型、输入和正确输出数据
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            # 使用 DDP 在指定设备上进行模型并行训练
            m = DDP(m, device_ids=[self.rank])
            # 在模型上应用 "aot_eager" 优化
            m = torch._dynamo.optimize("aot_eager")(m)
            # 对输入进行模型推断
            outputs = m(inputs)
            # 验证输出与正确输出是否一致
            self.assertTrue(same(correct_outputs, outputs))

    # 辅助方法：测试 HF BERT 模型在 DDP 模式下的 Inductor 逻辑
    def _test_hf_bert_ddp_inductor(self, static_graph):
        # 在每个进程中初始化 Dynamo 分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 获取 HF BERT 模型和输入数据
            model, inputs = get_hf_bert(self.rank)
            # 使用 DDP 在静态图模式下进行模型并行训练
            model = DDP(model, static_graph=static_graph)
            # 执行 HF BERT 模型的 DDP Inductor 测试
            run_hf_bert_ddp(self, model, inputs, "inductor")

    # 装饰器 skip_if_lt_x_gpu(2)：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器 import_transformers_or_skip()：导入 Transformers 库，否则跳过测试
    @import_transformers_or_skip()
    # 装饰器 unittest.skipIf()：如果没有 Triton 或 GPU 架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # 使用 patch.object 修改配置项，启用 DDP 优化
    @patch.object(config, "optimize_ddp", True)
    # 使用 patch.object 修改 Torch Inductor 配置项，启用随机模式回退
    @patch.object(torch._inductor.config, "fallback_random", True)
    # 测试方法：验证 HF BERT 模型在 DDP 模式下的 Inductor 逻辑
    def test_hf_bert_ddp_inductor(self):
        # 调用辅助方法进行测试，静态图模式为 False
        self._test_hf_bert_ddp_inductor(static_graph=False)

    # 装饰器 skip_if_lt_x_gpu(2)：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器 import_transformers_or_skip()：导入 Transformers 库，否则跳过测试
    @import_transformers_or_skip()
    # 装饰器 unittest.skipIf()：如果没有 Triton 或 GPU 架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # 使用 patch.object 修改配置项，启用 DDP 优化
    @patch.object(config, "optimize_ddp", True)
    # 使用 patch.object 修改 Torch Inductor 配置项，启用随机模式回退
    @patch.object(torch._inductor.config, "fallback_random", True)
    # 测试方法：验证 HF BERT 模型在静态图模式下的 DDP Inductor 逻辑
    def test_hf_bert_ddp_inductor_static_graph(self):
        # 调用辅助方法进行测试，静态图模式为 True
        self._test_hf_bert_ddp_inductor(static_graph=True)

    # 辅助方法：测试 HF BERT 模型在 DDP 模式下的 AOT（Ahead of Time）和 Eager 模式
    def _test_hf_bert_aot_eager(self, static_graph):
        # 在每个进程中初始化 Dynamo 分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 获取 HF BERT 模型和输入数据
            model, inputs = get_hf_bert(self.rank)
            # 使用 DDP 在静态图模式下进行模型并行训练
            model = DDP(model, static_graph=static_graph)
            # 执行 HF BERT 模型的 DDP AOT（Ahead of Time）和 Eager 模式测试
            run_hf_bert_ddp(self, model, inputs, "aot_eager")

    # 装饰器 skip_if_lt_x_gpu(2)：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器 import_transformers_or_skip()：导入 Transformers 库，否则跳过测试
    @import_transformers_or_skip()
    # 使用 patch.object 修改配置项，启用 DDP 优化
    @patch.object(config, "optimize_ddp", True)
    # 测试方法：验证 HF BERT 模型在 DDP 模式下的 AOT（Ahead of Time）和 Eager 模式测试
    def test_hf_bert_ddp_aot_eager(self):
        # 调用辅助方法进行测试，静态图模式为 False
        self._test_hf_bert_aot_eager(static_graph=False)

    # 装饰器 skip_if_lt_x_gpu(2)：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器 import_transformers_or_skip()：导入 Transformers 库，否则跳过测试
    @import_transformers_or_skip()
    # 使用 patch.object 修改配置项，启用 DDP 优化
    @patch.object(config, "optimize_ddp", True)
    # 测试方法：验证 HF BERT 模型在静态图模式下的 DDP AOT（Ahead of Time）和 Eager 模式测试
    def test_hf_bert_ddp_aot_eager_static_graph(self):
        # 调用辅助方法进行测试，静态图模式为 True
        self._test_hf_bert_aot_eager(static_graph=True)

    # 装饰器 skip_if_lt_x_gpu(2)：如果 GPU 数量小于 2 则跳过测试
    @skip_if_lt_x_gpu(2)
    # 装饰器 unittest.skipIf()：如果没有 Triton 或 GPU 架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # 使用 patch.object 修改配置项，禁用 DDP 优化
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_activation_checkpointing(self):
        # 导入所需的模块和类
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        # 定义一个简单的神经网络模型类
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 32)
                self.fc2 = torch.nn.Linear(32, 16)
                self.fc3 = torch.nn.Linear(16, 8)

            def forward(self, inp):
                return self.fc3(self.fc2(self.fc1(inp)))

        # 使用 _dynamo_dist_per_rank_init 进行分布式初始化
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 断言配置不优化分布式数据并行
            self.assertFalse(config.optimize_ddp)
            # 创建模型并将其放在 CUDA 设备上
            model = MyModel().to(device="cuda")

            # 对线性层应用激活检查点技术
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            # 定义用于检查线性层的函数
            check_fn = lambda submodule: isinstance(
                submodule, torch.nn.Linear
            )
            # 应用激活检查点技术到模型
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )

            # 将模型包装成分布式数据并行模型
            model = DDP(model)
            # 创建输入张量并将其放在 CUDA 设备上
            x = torch.randn(10, 64).cuda()
            # 获取正常输出
            correct_outputs = model(x)

            # 对模型进行优化编译
            opt_model = torch.compile(model)
            # 获取优化后模型的输出
            outputs = opt_model(x)
            # 断言优化后的输出与正常输出相同
            self.assertTrue(same(correct_outputs, outputs))

    @skip_if_lt_x_gpu(1)
    def test_fsdp_aot_eager(self):
        # 使用 _dynamo_dist_per_rank_init 进行分布式初始化
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 测试基本的 FSDP 包装（整个模型的外部包装）
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            # 使用 "aot_eager" 优化模式进行优化
            fsdp_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
            # 获取输出结果
            outputs = fsdp_m(inputs)
            # 断言优化后的输出与正确输出相同
            self.assertTrue(same(correct_outputs, outputs))

            # 测试递归包装，每个线性层周围嵌套的 FSDP
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            # 使用 "aot_eager" 优化模式进行优化
            fsdp_m = torch._dynamo.optimize("aot_eager")(fsdp_m)
            # 获取输出结果
            outputs = fsdp_m(inputs)
            # 断言优化后的输出与正确输出相同
            self.assertTrue(same(correct_outputs, outputs))

    @skip_if_lt_x_gpu(1)
    def test_fsdp_setattr(self):
        # 使用 _dynamo_dist_per_rank_init 初始化分布式环境，设置当前进程的排名和总进程数
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 测试基本的 FSDP 封装（将整个模型外部包装）
            # 获取可变模型 m、输入 inputs 和正确的输出 correct_outputs
            m, inputs, correct_outputs = get_mutating_model(f"cuda:{self.rank}")
            # 使用 FSDP 包装模型 m，保留原始参数
            fsdp_m = FSDP(m, use_orig_params=True)
            # 创建 Torch 动态编译器 Profiler 对象
            prof = torch._dynamo.utils.CompileProfiler()
            # 使用编译器 prof 编译 fsdp_m 模型，使用的后端为 prof，完整图设置为 False
            fsdp_m = torch.compile(fsdp_m, backend=prof, fullgraph=False)
            # 将输入 inputs 传递给 fsdp_m 模型进行推理，得到 outputs
            outputs = fsdp_m(inputs)
            # 断言 correct_outputs 与 outputs 相同
            self.assertTrue(same(correct_outputs, outputs))
            # 运行 Profiler 报告，检查是否包含特定字符串，验证性能和正确性
            FileCheck().check("Torchdynamo Profiler Report").check(
                "Graph Breaks"
            ).check_not(
                "setattr(FSDPManagedNNModuleVariable(MutatingModel), state, ...)"
            ).check_not(
                "setattr(FSDPManagedNNModuleVariable(FullyShardedDataParallel), _is_root, ...)"
            ).run(
                prof.report()
            )

    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_inductor(self):
        # 使用 _dynamo_dist_per_rank_init 初始化分布式环境，设置当前进程的排名和总进程数
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 测试基本的 FSDP 封装（将整个模型外部包装）
            # 获取模型 m、输入 inputs 和正确的输出 correct_outputs
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            # 使用 FSDP 包装模型 m，保留原始参数
            fsdp_m = FSDP(m, use_orig_params=True)
            # 使用 Torch 动态优化器 "inductor" 优化 fsdp_m 模型
            fsdp_m = torch._dynamo.optimize("inductor")(fsdp_m)
            # 将输入 inputs 传递给 fsdp_m 模型进行推理，得到 outputs
            outputs = fsdp_m(inputs)
            # 断言 correct_outputs 与 outputs 相同
            self.assertTrue(same(correct_outputs, outputs))

            # 测试递归封装，每个 Linear 层周围嵌套的 FSDP 封装
            # 获取模型 m、输入 inputs 和正确的输出 correct_outputs
            m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
            # 使用 FSDP 包装模型 m，配置自动封装策略为对 Linear 层进行封装
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            # 使用 Torch 动态优化器 "inductor" 优化 fsdp_m 模型
            fsdp_m = torch._dynamo.optimize("inductor")(fsdp_m)
            # 将输入 inputs 传递给 fsdp_m 模型进行推理，得到 outputs
            outputs = fsdp_m(inputs)
            # 断言 correct_outputs 与 outputs 相同
            self.assertTrue(same(correct_outputs, outputs))
    # 定义测试函数，用于测试FSDP激活检查点功能
    def test_fsdp_activation_checkpointing(self):
        # 使用上下文管理器初始化分布式环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 调用函数获取用于激活检查点的玩具模型和输入数据
            model, inputs = get_toy_model_for_activation_checkpointing(
                f"cuda:{self.rank}"
            )
            # 定义判断是否为内部模型的lambda函数
            is_inner = lambda module: isinstance(module, ToyInnerModel)  # noqa: E731
            # 使用functools.partial创建自动包装策略，使用is_inner作为参数
            wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_inner)
            # 应用FSDP并启用检查点功能到模型上
            model = apply_fsdp_with_checkpointing(model, wrap_policy, is_inner)
            # 对模型进行推理得到正确的输出
            correct_outputs = model(inputs)
            # 创建一个带有后端计数器的优化模型
            cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
            opt_model = torch._dynamo.optimize(cnt)(model)
            # 使用优化后的模型进行推理
            outputs = opt_model(inputs)
            # 断言正确的输出与优化后的输出相同
            self.assertTrue(same(correct_outputs, outputs))
            # 检查计数器中的帧数是否为2
            self.assertEqual(cnt.frame_count, 2)
            # 断言第一个图中存在激活检查点标记的节点
            self.assertTrue(
                find_first_node(cnt.graphs[0], tag_activation_checkpoint) is not None
            )

    # 导入transformers或跳过测试
    @import_transformers_or_skip()
    # 如果没有Triton或GPU架构较旧，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) 调查为什么hf_bert在Inductor+fsdp中断
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    # 如果平台支持闪存注意力或内存效率注意力，则跳过测试
    @unittest.skipIf(
        PLATFORM_SUPPORTS_FLASH_ATTENTION or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Inaccurate results with fused SDPA kernels",
    )
    # 定义测试函数 test_hf_bert_fsdp，用于测试基于 hf_bert 模型的 FSDP 功能
    def test_hf_bert_fsdp(self):
        
        # 定义内部函数 apply_fsdp，用于将模型应用 FSDP 包装策略
        def apply_fsdp(model, wrap_policy):
            # 深拷贝模型并应用 FSDP，使用原始参数
            model = FSDP(
                copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
            )
            return model

        # 使用 _dynamo_dist_per_rank_init 方法初始化分布式环境，适应当前 rank 和 world_size
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            
            # 遍历 wrap_policy 和 test_instance 的元组
            for wrap_policy, test_instance in (
                (None, "FSDP without recursive wrapping"),
            ):
                # 打印当前测试实例的信息
                print(f"Running hf_bert test for {test_instance}")
                
                # 调用 get_hf_bert 函数获取模型和输入数据
                model, inputs = get_hf_bert(self.rank)
                
                # 重置随机数生成器状态
                reset_rng_state()
                
                # 对模型应用 FSDP 包装策略，获取包含正确输出的 eager_model
                eager_model = apply_fsdp(model, wrap_policy)
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                
                # 计算正确的损失值的梯度
                correct_loss.backward()

                # 再次重置随机数生成器状态
                reset_rng_state()
                
                # 对模型应用 FSDP 包装策略，获取用于优化的 opt_model
                opt_model = apply_fsdp(model, wrap_policy)
                opt_model = torch._dynamo.optimize("inductor")(opt_model)
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                
                # 计算优化损失值的梯度
                opt_loss.backward()

                # 将输入数据展平为列表
                inputs_flat = [inputs[k] for k in inputs]
                
                # 收集 eager_model 的正确结果，包括 logits 和损失值
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                
                # 收集 opt_model 的结果，包括 logits 和损失值
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                
                # 使用 assertTrue 断言函数判断两种结果是否相同
                self.assertTrue(same(correct_results, opt_results))

    # 使用 import_transformers_or_skip 装饰器，导入 transformers 库，否则跳过测试
    @import_transformers_or_skip()
    
    # 如果没有 triton 或 GPU 架构太老，则跳过测试
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    
    # TODO(whc) 调查 cudagraphs 在 hf_bert 的 inductor+fsdp 中为什么会出问题
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    
    # 设置 fallback_random 参数为 True
    @patch.object(torch._inductor.config, "fallback_random", True)
    
    # 设置 guard_nn_modules 参数为 True
    @patch.object(torch._dynamo.config, "guard_nn_modules", True)
    # 定义一个测试函数，用于测试带有 FSDP 激活检查点的 HF Bert 模型
    def test_hf_bert_fsdp_activation_checkpointing(self):
        # 从 transformers 库中导入 BertLayer 类
        from transformers.models.bert.modeling_bert import BertLayer

        # 使用 _dynamo_dist_per_rank_init 初始化分布环境
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # 遍历包含 wrap_policy 和 test_instance 的元组
            for wrap_policy, test_instance in (
                (
                    functools.partial(
                        transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer,)
                    ),
                    "FSDP with recursive wrapping BertLayer instances",
                ),
            ):
                # 打印测试实例的名称
                print(
                    f"Running hf_bert_activation_checkpointing test for {test_instance}"
                )
                # 调用 get_hf_bert 函数获取模型和输入数据
                model, inputs = get_hf_bert(self.rank)
                # 定义检查函数，用于验证子模块是否为 BertLayer 类型
                check_fn = lambda submodule: isinstance(  # noqa: E731
                    submodule, BertLayer
                )
                # 重置随机数生成器的状态
                reset_rng_state()
                # 使用 apply_fsdp_with_checkpointing 应用 FSDP 并启用检查点的模型
                eager_model = apply_fsdp_with_checkpointing(
                    model, wrap_policy, check_fn
                )
                # 在 eager_model 上进行模型推断，获取正确的输出
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                # 再次重置随机数生成器的状态
                reset_rng_state()
                # 使用 apply_fsdp_with_checkpointing 应用 FSDP 并启用检查点的模型
                opt_model = apply_fsdp_with_checkpointing(model, wrap_policy, check_fn)
                # 通过 torch._dynamo.optimize("inductor") 进行优化处理
                opt_model = torch._dynamo.optimize("inductor")(opt_model)
                # 在 opt_model 上进行模型推断，获取优化后的输出
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                # 将输入数据扁平化为列表形式
                inputs_flat = [inputs[k] for k in inputs]
                # 收集 eager_model 的结果，包括 logits、loss 和扁平化后的输入数据
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                # 收集 opt_model 的结果，包括 logits、loss 和扁平化后的输入数据
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                # 使用 self.assertTrue 检查 correct_results 是否与 opt_results 相同
                self.assertTrue(same(correct_results, opt_results))
# 为了执行此测试，需要确保使用了 NCCL 库
# 确保 CUDA 可用
@requires_nccl()
@requires_cuda
class TestSingleProc(DynamoDistributedSingleProcTestCase):
    """
    测试单进程的测试框架，初始化分布式进程组。

    在这里测试简单的事物，因为它们更容易调试。
    对于需要在多个节点上运行的事物，请使用 TestMultiProc。
    """

    def get_model(
        self, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
    ):
        """
        创建一个模型并返回其实例、输入和正确的输出。

        参数:
        - bsz: 批量大小 (默认为 20)
        - in_feat: 输入特征数 (默认为 10)
        - hidden_feat: 隐藏层特征数 (默认为 5000)
        - out_feat: 输出特征数 (默认为 5)
        - ctx_manager: 上下文管理器 (默认为 None)

        返回:
        - m: 创建的 ToyModel 实例
        - inputs: 随机生成的输入张量
        - outputs: 模型对输入的输出
        """
        m = ToyModel(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            out_feat=out_feat,
            ctx_manager=ctx_manager,
        ).to(self.device)
        m.apply(init_weights)
        inputs = torch.rand(bsz, in_feat).to(self.device)
        outputs = m(inputs)
        return m, inputs, outputs

    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager(self):
        """
        测试基于 DDP 的基准模型，使用 aot_eager 优化。

        验证模型输出与预期输出是否相同。
        """
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch._dynamo.optimize("aot_eager")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_inductor(self):
        """
        测试基于 DDP 的基准模型，使用 inductor 优化。

        验证模型输出与预期输出是否相同。
        """
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch._dynamo.optimize("inductor")(ddp_m)
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split(self):
        """
        测试图分割优化的功能。

        确保 DDPOptimizer 根据桶大小和模型参数调用用户提供的编译器适当次数。
        验证优化后的模型输出与预期输出是否相同。
        """
        assert config.optimize_ddp

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

        # ensure compatibility with dynamo explain

        explain_out = torch._dynamo.explain(ddp_m)(inputs)
        break_reasons = explain_out.break_reasons
        self.assertEqual(len(break_reasons), 3)
        self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split_ctx_manager(self):
        """
        Ensures that we get the right number of splits and that the respective
        context managers' effects are applied to the computation.
        """

        for get_compiler in [
            lambda: CheckSplitsCompiler(),
            lambda: None,
        ]:
            for ctx_manager, output_test in [
                (
                    lambda: torch.autocast(
                        torch.device(self.device).type, torch.float16
                    ),
                    lambda out: self.assertEqual(out.dtype, torch.float16),
                ),  # 设置上下文管理器为自动混合精度模式，检查输出的数据类型是否为 torch.float16
                (torch.enable_grad, lambda out: self.assertTrue(out.requires_grad)),  # 启用梯度追踪，检查输出是否需要梯度
                (torch.no_grad, lambda out: self.assertTrue(not out.requires_grad)),  # 禁用梯度追踪，检查输出是否不需要梯度
            ]:
                m, inputs, correct_outputs = self.get_model(
                    out_feat=1000,
                    hidden_feat=1000,
                    in_feat=1000,
                    ctx_manager=ctx_manager,
                )
                # 输入 inputs 是 1000 * 1000 的 float32 矩阵（4字节）= 4MB
                # 隐藏层 hidden 是 1000 * 1000 的 float32 矩阵（4字节）= 4MB
                bucket_cap_mb = 3.5  # 4MB
                ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=bucket_cap_mb)

                compiler = get_compiler()

                @torch._dynamo.optimize(
                    compiler.compile_fn if compiler else "aot_eager"
                )
                def opt_fn(inputs):
                    return ddp_m(inputs)

                opt_outputs = opt_fn(inputs)
                self.assertTrue(same(correct_outputs, opt_outputs))
                if compiler:
                    self.assertEqual(compiler.compiler_called, 4)  # 断言编译器被调用了4次

                output_test(opt_outputs)  # 检查输出是否符合预期

                # ensure compatibility with dynamo explain

                explain_out = torch._dynamo.explain(ddp_m)(inputs)
                break_reasons = explain_out.break_reasons
                self.assertEqual(len(break_reasons), 4)  # 断言中断原因的数量为4
                self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))  # 断言所有中断原因中都包含"DDPOptimizer"

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor(self):
        assert config.optimize_ddp
        """
        Same as above, but using inductor backend.
        We observed issues with inductor/fx interface in the past.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch._dynamo.optimize("inductor")
        def opt_fn(inputs):
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))

    @torch._inductor.config.patch(
        {"layout_optimization": True, "keep_output_stride": False}
    )
    @patch.object(config, "optimize_ddp", True)
    def _test_graph_split_inductor_layout_optimizations_impl(self, context):
        # 断言配置中启用了 DDP 优化
        assert config.optimize_ddp
        # 定义通道维度为 512
        channel_dim = 512
        # 通道维度必须大于 64，以便电感器执行布局优化并使用 NHWC 格式

        class ToyModelConv(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义网络结构为一系列的卷积层，每层输出维度为 channel_dim，卷积核大小为 1，步长为 1，无偏置
                self.net = nn.Sequential(
                    *[
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                )

            def forward(self, inputs):
                # 前向传播函数
                return self.net(inputs)

        def get_model():
            # 创建 ToyModelConv 模型并移动到指定设备
            m = ToyModelConv().to(self.device)
            # 初始化模型权重
            m.apply(init_weights)
            # 创建随机输入数据并移动到指定设备
            inputs = torch.rand(2, channel_dim, channel_dim, 128).to(self.device)
            # 使用模型进行前向传播
            outputs = m(inputs)
            return m, inputs, outputs

        # 运行在给定上下文中的测试函数
        with context():
            # 获取模型及其输入、正确输出
            m, inputs, correct_outputs = get_model()
            # 使用 DDP 将模型包装，指定设备和桶容量
            ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

            @torch._dynamo.optimize("inductor")
            def opt_fn(inputs):
                # 运行优化函数
                return ddp_m(inputs)

            # 运行优化函数得到输出
            opt_outputs = opt_fn(inputs)
            # 断言优化后的输出与正确输出相同
            self.assertTrue(same(correct_outputs, opt_outputs))

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_layout_optimizations_training(self):
        # 测试图分离电感器布局优化训练过程
        self._test_graph_split_inductor_layout_optimizations_impl(
            contextlib.nullcontext
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_layout_optimizations_inference(self):
        # 测试图分离电感器布局优化推断过程
        self._test_graph_split_inductor_layout_optimizations_impl(torch.no_grad)

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_transpose(self):
        # 断言配置中是否开启了优化 DDP
        assert config.optimize_ddp

        B = 100  # 设置批次大小为 100
        N = 30   # 设置输入向量的维度为 30
        D = 50   # 设置输入矩阵的深度为 50
        K = 70   # 设置输出向量的维度为 70

        class Foo(nn.Module):
            def __init__(self):
                super().__init__()
                # 定义神经网络模块：线性层0，输入维度 N，输出维度 K
                self.linear0 = nn.Linear(N, K)
                # 定义神经网络模块：线性层1，输入维度 D * K，输出维度 2048
                self.linear1 = torch.nn.Linear(D * K, 2048)

            def forward(self, x):
                # 对输入张量 x 进行转置，交换维度 2 和 1
                xt = x.transpose(2, 1)
                # 将转置后的张量 xt 输入线性层 linear0，并展平成一维
                xt = self.linear0(xt).flatten(1)
                # 经过 linear1 线性层处理，得到最终输出
                return self.linear1(xt)

        # 创建 Foo 类的实例，并将其移动到指定设备上
        mod = Foo().to(self.device)

        # 编译模型为使用 "inductor" 后端的形式
        compiled_mod = torch.compile(mod, backend="inductor")
        # 对编译后的模型应用分布式数据并行，设备 ID 由 self.device_ids 指定
        ddp_compiled_mod = DDP(compiled_mod, device_ids=self.device_ids)

        # 生成一个随机张量 x，形状为 (B, N, D)，数据类型为 torch.float32，位于指定设备上
        x = torch.randn((B, N, D), dtype=torch.float32, device=self.device)
        # 断言模型 mod 和经过分布式数据并行处理的编译模型 ddp_compiled_mod 输出相同结果
        self.assertTrue(same(mod(x), ddp_compiled_mod(x)))

        # 生成一个随机张量 x_1，形状为 (B * 2, N, D)，数据类型为 torch.float32，位于指定设备上
        x_1 = torch.randn((B * 2, N, D), dtype=torch.float32, device=self.device)
        # 断言模型 mod 和经过分布式数据并行处理的编译模型 ddp_compiled_mod 输出相同结果
        self.assertTrue(same(mod(x_1), ddp_compiled_mod(x_1)))

        # 生成一个随机张量 x_2，形状为 (B * 3, N, D)，数据类型为 torch.float32，位于指定设备上
        x_2 = torch.randn((B * 3, N, D), dtype=torch.float32, device=self.device)
        # 断言模型 mod 和经过分布式数据并行处理的编译模型 ddp_compiled_mod 输出相同结果
        self.assertTrue(same(mod(x_2), ddp_compiled_mod(x_2)))

    @patch.object(config, "optimize_ddp", True)
    def test_no_split(self):
        """
        确保 DDPOptimizer 返回一个正确编译的模块，且不引入图分割。
        (基于模型参数适合桶容量的前提下)
        """
        # 从 self.get_model 方法获取模型 m、输入 inputs 和正确输出 correct_outputs
        m, inputs, correct_outputs = self.get_model(hidden_feat=5)
        # 使用 DDP 将模型 m 分布式并行，设备 ID 由 self.device_ids 指定，桶容量为 250 MB
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=250)
        # 创建一个 CheckSplitsCompiler 实例
        check_splits_compiler = CheckSplitsCompiler()

        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            # 执行分布式数据并行模型 ddp_m，并返回输出结果
            return ddp_m(inputs)

        # 使用输入 inputs 调用优化函数 opt_fn，获取其输出结果 opt_outputs
        opt_outputs = opt_fn(inputs)
        # 断言优化输出 correct_outputs 和 opt_outputs 相同
        self.assertTrue(same(correct_outputs, opt_outputs))
        # 断言编译器调用次数为 1
        self.assertEqual(check_splits_compiler.compiler_called, 1)

    @patch.object(config, "optimize_ddp", True)
    def test_aot_autograd(self):
        """
        明确检查 AotAutograd 编译器系列工作，
        因为它们要求示例输入在图分割之间传播。
        """
        # 从 self.get_model 方法获取模型 m、输入 inputs 和正确输出 correct_outputs
        m, inputs, correct_outputs = self.get_model()
        # 使用 DDP 将模型 m 分布式并行，设备 ID 由 self.device_ids 指定，桶容量为 25 MB
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch._dynamo.optimize("aot_eager")
        def opt_fn(inputs):
            # 执行分布式数据并行模型 ddp_m，并返回输出结果
            return ddp_m(inputs)

        # 使用输入 inputs 调用优化函数 opt_fn，获取其输出结果 opt_outputs
        opt_outputs = opt_fn(inputs)
        # 对 opt_outputs 求和并执行反向传播
        opt_outputs.sum().backward()
        # 断言优化输出 correct_outputs 和 opt_outputs 相同
        self.assertTrue(same(correct_outputs, opt_outputs))
    def test_custom_layer(self):
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """
        # 获取定制模型 m、输入 inputs 和正确输出 correct_outputs
        m, inputs, correct_outputs = get_custom_model(self.device)
        # 使用 DDP 对象包装模型 m，指定设备 ID 和桶容量
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=1)

        # 创建 CheckSplitsCompiler 实例
        check_splits_compiler = CheckSplitsCompiler()

        # 使用 torch._dynamo.optimize 优化器，传入编译函数进行优化
        @torch._dynamo.optimize(check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            return ddp_m(*inputs)

        # 运行优化后的函数，得到输出结果 opt_outputs
        opt_outputs = opt_fn(inputs)
        # 断言优化后的输出与正确输出相同
        self.assertTrue(same(correct_outputs, opt_outputs))
        # 断言编译器被调用的次数为 3
        self.assertEqual(check_splits_compiler.compiler_called, 3)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    def test_empty_graph_inductor(self):
        """
        Verifies ddp graph-split logic ignores parameters marked to ignore on DDP module.
        Hooks up graph-split optimizer manually so it can peek at internal state.
        """
        # 定义一个函数 fn，返回当前的世界大小
        def fn():
            get_world_size = torch.distributed.distributed_c10d.get_world_size()
            return (get_world_size,)

        # 使用 torch._dynamo.optimize 进行优化，传入"inductor"作为优化器
        opt_fn = torch._dynamo.optimize("inductor")(fn)
        res = None
        try:
            # 尝试运行优化后的函数，并获取第一个返回值
            res = opt_fn()[0]
        except Exception:
            pass
        # 断言 res 的值为 1
        self.assertEqual(res, 1)

    @patch.object(config, "optimize_ddp", False)
    def test_ignored_parameters(self):
        """
        Verifies ddp graph-split logic ignores parameters marked to ignore on DDP module.
        Hooks up graph-split optimizer manually so it can peek at internal state.
        """
        # 获取定制模型 m、输入 inputs 和正确输出 correct_outputs
        m, inputs, correct_outputs = get_custom_model(self.device)
        # 标记要忽略的参数列表
        parameters_to_ignore = ["seq.2.weight", "seq.4.linear.bias"]
        # 设置 DDP 模块中要忽略的参数
        DDP._set_params_and_buffers_to_ignore_for_model(m, parameters_to_ignore)
        # 使用 DDP 对象包装模型 m，指定设备 ID 和桶容量
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)
        # 获取要忽略的参数 ID 列表
        parameter_ids_to_ignore = [
            id(ddp_m.module.get_parameter(p)) for p in ddp_m.parameters_to_ignore
        ]

        # 创建 CheckSplitsCompiler 实例
        check_splits_compiler = CheckSplitsCompiler()
        # 创建 DDPOptimizer 实例，指定桶容量和后端编译函数
        ddp_optimizer = DDPOptimizer(
            bucket_bytes_cap=ddp_m.bucket_bytes_cap,
            backend_compile_fn=check_splits_compiler.compile_fn,
        )

        # 使用 torch._dynamo.optimize 进行优化，传入编译函数进行优化
        @torch._dynamo.optimize(ddp_optimizer.compile_fn)
        def opt_fn(inputs):
            return ddp_m(*inputs)

        # 运行优化后的函数，得到输出结果 opt_outputs
        opt_outputs = opt_fn(inputs)
        # 断言优化后的输出与正确输出相同
        self.assertTrue(same(correct_outputs, opt_outputs))
        # 断言编译器被调用的次数为 2
        self.assertEqual(check_splits_compiler.compiler_called, 2)
        # 遍历所有桶，确保其中的参数 ID 不在要忽略的参数 ID 列表中
        for b in ddp_optimizer.buckets:
            for p_id in b.param_ids:
                self.assertFalse(p_id in parameter_ids_to_ignore)

    @patch.object(config, "optimize_ddp", True)
    # 定义一个测试函数，用于测试高阶操作
    def test_higher_order_op(self):
        # 从torch.utils.checkpoint模块导入checkpoint函数
        from torch.utils.checkpoint import checkpoint

        # 设置变量N为1000
        N = 1000

        # 定义一个内部模块InnerModule，继承自torch.nn.Module类
        class InnerModule(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建一个线性层linear1，输入和输出维度均为N
                self.linear1 = torch.nn.Linear(N, N)
                # 创建一个线性层linear2，输入和输出维度均为N
                self.linear2 = torch.nn.Linear(N, N)

            # 前向传播函数
            def forward(self, x):
                # 对输入x应用linear1
                a = self.linear1(x)
                # 对上一步的结果a应用linear2
                a = self.linear2(a)
                return a

        # 定义一个模拟模块MockModule，继承自torch.nn.Module类
        class MockModule(torch.nn.Module):
            # 初始化函数
            def __init__(self):
                super().__init__()
                # 创建两个内部模块实例inner_mod1和inner_mod2
                self.inner_mod1 = InnerModule()
                self.inner_mod2 = InnerModule()

            # 前向传播函数
            def forward(self, x):
                # 使用checkpoint函数对inner_mod1进行检查点操作，禁用重入
                a = checkpoint(self.inner_mod1, x, use_reentrant=False)
                # 对结果a应用余弦函数
                a = torch.cos(a)
                # 使用checkpoint函数对inner_mod2进行检查点操作，禁用重入
                a = checkpoint(self.inner_mod2, a, use_reentrant=False)
                # 对结果a再次应用余弦函数
                a = torch.cos(a)
                return a

        # 创建MockModule的实例mod，并将其移动到CUDA设备上
        mod = MockModule().cuda()
        # 使用DDP对模型进行分布式数据并行处理，设置桶的容量为1MB
        mod = DDP(mod, bucket_cap_mb=1)
        # 生成一个随机张量x，形状为N x N，位于CUDA设备上，需要梯度计算
        x = torch.randn(N, N, device="cuda", requires_grad=True)
        # 准备模型输入参数
        args = (x,)

        # 设置后端为"aot_eager"
        backend = "aot_eager"
        # 创建一个编译计数器实例cnt，使用指定的后端
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        # 使用assertRaisesRegex上下文管理器断言编译过程中是否抛出BackendCompilerFailed异常
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "DDPOptimizer backend: Found a higher order op in the graph",
        ):
            # 编译模型mod，使用cnt指定的后端，传入参数args
            torch.compile(mod, backend=cnt)(*args)

    # 定义一个测试函数，用于测试FSDP包装原始参数的断言
    def test_fsdp_orig_params_assert(self):
        # 测试使用基本的FSDP包装（将整个模型外部包装）
        # 调用get_model函数获取模型m、输入inputs和正确输出correct_outputs，位于特定CUDA设备上
        m, inputs, correct_outputs = get_model(f"cuda:{self.rank}")
        # 使用FSDP对模型m进行包装，禁用使用原始参数
        fsdp_m = FSDP(m, use_orig_params=False)
        # 对包装后的模型应用torch._dynamo.optimize()进行优化
        fsdp_m = torch._dynamo.optimize()(fsdp_m)
        # 使用assertRaisesRegex断言是否抛出断言错误异常
        self.assertRaisesRegex(
            AssertionError,
            "Dynamo only supports FSDP with use_orig_params=True",
            fsdp_m,
            inputs,
        )
    def test_fsdp_skip_guards(self):
        """
        It's currently difficult to test dynamo guards.  Most guards tests are indirect- modify something and
        observe that the guard in question failed. In this case, since the FSDP guards were already deemed
        useless and skipping them is expected to have no practical effect, it's pretty contrived to even try to
        make those guards fail.  Instead, we observe the 'guard source' printed by dynamo's comptime print_guards
        function.

        Note: comptime prints the guards before the time they get installed or not installed, so in both cases
        (skip or no skip) the same guards get printed.  The difference is that in the skip case, they show up
        with a special 'guard source' which will cuase them to not be installed.  So all we check for is the expected
        guard source 'local_fsdp_module'.
        """
        global GUARDS_FILE
        # 初始化一个全局的 StringIO 对象，用于存储测试期间输出的 guards 信息
        GUARDS_FILE = StringIO()

        # 对于每一组 skip_guards 和 expected_guard_source 的组合进行测试
        for skip_guards, expected_guard_source in (
            (True, "local_fsdp_module"),  # 第一组测试，期望的 guard source 为 "local_fsdp_module"
            (False, "local"),  # 第二组测试，期望的 guard source 为 "local"
        ):
            torch._dynamo.reset()  # 重置 torch._dynamo 状态，确保测试的独立性

            # 定义一个简单的 ToyModel 作为测试模型
            class ToyModel(nn.Module):
                def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
                    super().__init__()
                    # 定义一个简单的神经网络结构
                    self.net = nn.Sequential(
                        *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
                    )

                def forward(self, inputs):
                    out = self.net(inputs)

                    @comptime
                    def _(ctx):
                        ctx.print_guards(file=GUARDS_FILE)  # 在 forward 运行时打印 guards 到 GUARDS_FILE

                    return out

            device = f"cuda:{self.rank}"  # 确定设备的 CUDA 设备
            m = ToyModel(
                in_feat=10,
                hidden_feat=5000,
                out_feat=5,
            ).to(device)  # 将 ToyModel 移动到指定的 CUDA 设备上
            inputs = torch.rand(20, 10).to(device)  # 创建输入张量并移动到 CUDA 设备
            m.apply(init_weights)  # 初始化模型的权重
            correct_outputs = m(inputs)  # 获取模型在输入上的正确输出

            fsdp_m = FSDP(m, use_orig_params=True)  # 使用 FSDP 对象封装模型

            # 使用 torch._dynamo.config.patch 来设置 skip_fsdp_guards 的值为 skip_guards
            with torch._dynamo.config.patch(skip_fsdp_guards=skip_guards):
                opt_m = torch._dynamo.optimize("aot_eager")(fsdp_m)  # 对 fsdp_m 进行优化
                outputs = opt_m(inputs)  # 获取优化后模型在输入上的输出

            # 使用 FileCheck 对输出的 guards 信息进行检查，确保输出符合预期
            FileCheck().check("""local "L['self']" TYPE_MATCH""").check(
                f"""{expected_guard_source} "L['self']._modules['net']" TYPE_MATCH"""
            ).check(
                f"""{expected_guard_source} "L['self']._modules['net']._modules['0']" TYPE_MATCH"""
            ).run(
                GUARDS_FILE.getvalue()  # 运行 FileCheck 检查 GUARDS_FILE 中的内容
            )

            self.assertTrue(same(correct_outputs, outputs))  # 断言模型输出和正确输出相同
    def test_fsdp_skip_register_attr_or_module(self):
        """
        ensure FSDP module is not registered as attrbutes
        in the fx graph
        see `not source.guard_source().is_fsdp_module()`
        before calling `register_attr_or_module`
        in variables/builder.py
        """

        # 定义一个测试函数，确保 FSDP 模块不被注册为属性在 fx 图中
        class ToyModel(nn.Module):
            def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
                super().__init__()
                # 创建一个神经网络模型，包含两个线性层和ReLU激活函数
                self.net = nn.Sequential(
                    *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
                    + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                )

            def forward(self, inputs):
                # 前向传播函数
                out = self.net(inputs)
                return out

        # 重置 Torch 的 Dynamo 状态
        torch._dynamo.reset()

        # 设置设备为 CUDA
        device = f"cuda:{self.rank}"
        # 创建 ToyModel 实例，并移动到指定设备上
        m = ToyModel(
            in_feat=10,
            hidden_feat=5000,
            out_feat=5,
        ).to(device)
        # 创建输入数据，并移动到指定设备上
        inputs = torch.rand(20, 10).to(device)
        # 对模型应用初始化权重函数
        m.apply(init_weights)
        # 计算正确的输出
        correct_outputs = m(inputs)
        # 使用 FSDP 封装模型，使用原始参数
        fsdp_m = FSDP(m, use_orig_params=True)

        # 定义一个调试编译器函数
        def debug_compiler(gm, _):
            # 遍历图中的节点
            for node in gm.graph.nodes:
                # 如果节点操作为 "get_attr"
                if node.op == "get_attr":
                    # 检查特定的名称是否在节点名称中
                    for name in [
                        "l__self___net_0_weight",
                        "l__self___net_0_bias",
                        "l__self___net_2_weight",
                        "l__self___net_2_bias",
                    ]:
                        # 断言 FSDP 模块的名称不应该在节点名称中
                        self.assertFalse(
                            name in node.name,
                            f"FSDP module {name} should not be registered as attributes",
                        )
            return gm

        # 优化使用调试编译器后的模型
        opt_m = torch._dynamo.optimize(backend=debug_compiler)(fsdp_m)
        # 计算优化模型的输出
        outputs = opt_m(inputs)

        # 断言正确的输出和优化模型的输出相同
        self.assertTrue(same(correct_outputs, outputs))
    # 定义一个测试方法，用于验证 FSDP 管理的模块中，具有相同来源的参数和缓冲是否被去重，
    # 即它们是否只作为图的输入传递一次。
    def test_fsdp_dup_tensors_same_source(self):
        """
        Tests that FSDP-managed modules' parameters and buffers with the same
        source are de-duplicated, meaning that they are each only passed once
        as a graph input.
        """

        # 定义一个继承自 nn.Module 的内部类 DuplicateModule
        class DuplicateModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 创建一个名为 _param 的参数，其值为在 CUDA 设备上生成的随机张量
                self._param = torch.randn((3,), device="cuda")
                # 注册一个名为 _buf 的缓冲区，其值为在 CUDA 设备上生成的随机张量，
                # 并指定不需要梯度
                self.register_buffer(
                    "_buf", torch.randn((3,), requires_grad=False, device="cuda")
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 在这个编译后的前向传播中，使用 `_param` 和 `_buf` 各两次，
                # 以检验它们是否被 TorchDynamo 去重
                z = x + self._buf + self._buf
                z += self._param + self._param
                return z

        # 创建 DuplicateModule 类的实例 model
        model = DuplicateModule()
        # 使用深拷贝的方式创建 FSDP 模型 fsdp_model，同时保留原始参数
        fsdp_model = FSDP(copy.deepcopy(model), use_orig_params=True)
        # 对 fsdp_model 进行 TorchDynamo 的即时编译优化
        fsdp_model = torch._dynamo.optimize("aot_eager")(fsdp_model)
        # 创建一个在 CUDA 设备上生成的随机张量 inp
        inp = torch.randn((2, 3), device="cuda")
        # 在本地模型上应用输入 inp，获得 local_out
        local_out = model(inp)
        # 在 FSDP 模型上应用输入 inp，获得 fsdp_out
        fsdp_out = fsdp_model(inp)
        # 断言 local_out 和 fsdp_out 是否相等
        self.assertEqual(local_out, fsdp_out)

    # 使用 patch.object 方法，将 config.guard_nn_modules 设置为 True
    @patch.object(config, "guard_nn_modules", True)
    def test_fsdp_dup_tensors_diff_source(self):
        """
        Tests that FSDP-managed modules' parameters and buffers with different
        source do not result in incorrect AOTAutograd de-dup guards like
        ``a is b``, where ``a`` and ``b`` are certainly not the same. We check
        this by checking for per-invocation recompiles.
        """

        # 定义一个测试类，用于测试不同源的 FSDP 管理模块的参数和缓冲区，
        # 确保不会导致类似于 ``a is b`` 的错误 AOTAutograd 去重保护，其中 ``a`` 和 ``b`` 显然不相同。
        # 我们通过检查每次调用重新编译来验证这一点。

        class BufModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 注册一个不需要梯度的 CUDA 上的随机张量作为缓冲区 "_buf"
                self.register_buffer(
                    "_buf", torch.randn((3,), requires_grad=False, device="cuda")
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 返回输入张量 x 加上缓冲区 "_buf" 的结果
                return x + self._buf

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # 注册一个 CUDA 上的随机张量作为参数 "_param"
                self._param = nn.Parameter(torch.randn((1,), device="cuda"))
                # 创建一个 BufModule 实例作为模块的属性 "_buf_module"
                self._buf_module = BufModule()
                # 共享缓冲区，意味着使用相同的张量但来源不同
                self.register_buffer("_buf", self._buf_module._buf)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # 在编译后的 forward 方法中两次使用相同的缓冲区张量，
                # 包括数据突变以触发去重逻辑
                self._buf.mul_(2)
                z = x + self._buf
                z = self._buf_module(z)
                z += self._param
                return z

        # 创建一个使用原始参数的 FSDP 模型
        fsdp_model = FSDP(Model(), use_orig_params=True)
        # 创建一个带有后端的编译计数器
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        # 对 fsdp_model 进行优化
        fsdp_model = torch._dynamo.optimize(cnt)(fsdp_model)
        # 创建一个随机输入张量
        inp = torch.randn((2, 3), device="cuda")
        # 多次调用模型的 forward 方法
        for _ in range(15):
            fsdp_model(inp)
        # 检查是否没有重新编译（如果存在错误的去重保护，则帧计数将等于 forward 调用的次数）
        self.assertEqual(cnt.frame_count, 1)
    def test_fsdp_staticmethod(self):
        """
        Tests that Dynamo compiles staticmethods for FSDP-managed modules
        correctly both when the staticmethod is invoked from the class and from
        the object itself.
        """

        class ModuleWithStaticMethod(nn.Module):
            def __init__(self, use_self: bool):
                super().__init__()
                self._use_self = use_self
                torch.manual_seed(42)  # 强制 `_param` 是确定性的
                self._param = nn.Parameter(torch.randn((3,), device="cuda"))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self._use_self:
                    # 如果使用 self，则从对象自身调用 _add 静态方法
                    z = self._add(x, self._param)
                else:
                    # 如果不使用 self，则从类名调用 _add 静态方法
                    z = ModuleWithStaticMethod._add(x, self._param)
                z *= 2  # 结果乘以 2
                return z

            @staticmethod
            def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # 静态方法，将两个张量相加并返回结果
                return x + y

        model = ModuleWithStaticMethod(False)
        x = torch.randn((2, 3), device="cuda")
        ref_out = model(x)
        test_outs: List[torch.Tensor] = []

        for use_self in (False, True):
            model = ModuleWithStaticMethod(use_self)
            # 使用 FSDP 对模型进行管理，确保优化后的参数
            fsdp_model = FSDP(model, use_orig_params=True)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            # 使用 Torch 动态编译优化模型
            fsdp_model = torch._dynamo.optimize(cnt)(fsdp_model)
            test_outs.append(fsdp_model(x))
            # 检查是否有重新编译，如果方法静态调用错误（例如重复传递 `self` 参数），可能会发生重新编译
            # 这里期望的是 1，因为只有 1 次 forward 调用
            self.assertEqual(cnt.frame_count, 1)
        for test_out in test_outs:
            # 检查测试输出是否与参考输出相等
            self.assertEqual(test_out, ref_out)

    def test_async_subclass_no_specialize(self):
        # 创建计数器以记录编译次数
        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")

        # 使用 Torch 的编译装饰器，定义一个简单的函数
        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x):
            return x + 1

        # 调用 f 函数两次，应该只编译一次，因为静态图只有一个 forward
        f(_maybe_wrap_tensor(torch.randn(10)))
        f(_maybe_wrap_tensor(torch.randn(12)))

        # 检查编译次数是否为 1
        self.assertEqual(cnt.frame_count, 1)
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._dynamo.test_case 模块中导入 run_tests 函数
    from torch._dynamo.test_case import run_tests

    # 运行测试函数，这个函数通常用于执行单元测试或集成测试
    run_tests()
```