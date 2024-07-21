# `.\pytorch\benchmarks\fastrnns\test_bench.py`

```py
# 导入 pytest 库，用于编写和运行测试
import pytest

# 导入 PyTorch 库，用于深度学习模型和计算
import torch

# 从当前包中导入相关模块
from .fuser import set_fuser
from .runner import get_nn_runners

# 定义一个 pytest 的 fixture，作用域为 class
@pytest.fixture(scope="class")
def modeldef(request, net_name, executor, fuser):
    # 调用 set_fuser 函数，设置融合器和执行器
    set_fuser(fuser, executor)

    # 从 get_nn_runners 函数返回的结果中获取模型名称、创建函数和上下文信息
    name, rnn_creator, context = get_nn_runners(net_name)[0]

    # 定义创建函数的参数字典
    creator_args = {
        "seqLength": 100,
        "numLayers": 1,
        "inputSize": 512,
        "hiddenSize": 512,
        "miniBatch": 64,
        "device": "cuda",
        "seed": None,
    }

    # 使用创建函数和参数字典创建 RNN 模型并返回
    return rnn_creator(**creator_args)


# 定义一个函数 cuda_sync，用于执行带有 CUDA 同步的函数调用
def cuda_sync(func, *args, **kwargs):
    # 调用传入的函数和参数执行计算
    out = func(*args, **kwargs)
    
    # 在 CUDA 上同步操作
    torch.cuda.synchronize()
    
    # 返回执行结果
    return out


# 使用 pytest 的 benchmark 标记定义一个性能测试类 TestBenchNetwork
@pytest.mark.benchmark(
    warmup=True,  # 运行性能测试前进行预热
    warmup_iterations=3,  # 预热迭代次数
    disable_gc=True,  # 禁用垃圾回收以减少干扰
    max_time=0.1,  # 每个性能测试的最大运行时间
    group="fastrnns",  # 性能测试分组名称
)
class TestBenchNetwork:
    # 测试模型正向传播的性能
    # 使用 'modeldef' fixture 提供的模型定义进行测试
    def test_forward(self, modeldef, benchmark):
        # 使用 benchmark 工具对 cuda_sync 函数执行模型的正向传播，并记录性能数据
        forward_output = benchmark(cuda_sync, modeldef.forward, *modeldef.inputs)

    # 测试模型反向传播的性能
    # 使用 'modeldef' fixture 提供的模型定义进行测试
    def test_backward(self, modeldef, benchmark):
        # 执行模型的正向传播获取反向传播的输入数据
        backward_input = modeldef.forward(*modeldef.inputs)
        
        # 如果存在反向传播的设置函数，则对反向传播输入数据进行配置
        if modeldef.backward_setup is not None:
            backward_input = modeldef.backward_setup(backward_input)

        # 如果定义了反向传播函数，则使用 benchmark 工具对其执行，并保持计算图
        if modeldef.backward is not None:
            benchmark(cuda_sync, modeldef.backward, *backward_input, retain_graph=True)

            # 确保梯度计算正确性，同时将参数梯度置零
            with torch.no_grad():
                for param in modeldef.params:
                    assert param.grad is not None
                    param.grad.zero_()
```