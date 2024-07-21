# `.\pytorch\benchmarks\fastrnns\runner.py`

```py
from collections import namedtuple  # 导入 namedtuple 模块
from functools import partial  # 导入 partial 函数

import torchvision.models as cnn  # 导入 torchvision 中的模型定义

import torch  # 导入 PyTorch 库

from .factory import (  # 从当前包中导入多个工厂函数
    dropoutlstm_creator,
    imagenet_cnn_creator,
    layernorm_pytorch_lstm_creator,
    lnlstm_creator,
    lstm_creator,
    lstm_multilayer_creator,
    lstm_premul_bias_creator,
    lstm_premul_creator,
    lstm_simple_creator,
    pytorch_lstm_creator,
    varlen_lstm_creator,
    varlen_pytorch_lstm_creator,
)

# 禁用 CuDNN 的上下文管理器类
class DisableCuDNN:
    def __enter__(self):
        self.saved = torch.backends.cudnn.enabled  # 保存当前 CuDNN 状态
        torch.backends.cudnn.enabled = False  # 禁用 CuDNN

    def __exit__(self, *args, **kwargs):
        torch.backends.cudnn.enabled = self.saved  # 恢复原始 CuDNN 状态

# 空上下文管理器类
class DummyContext:
    def __enter__(self):
        pass  # 无操作

    def __exit__(self, *args, **kwargs):
        pass  # 无操作

# 断言不使用 JIT 的上下文管理器类
class AssertNoJIT:
    def __enter__(self):
        import os  # 导入 os 模块

        enabled = os.environ.get("PYTORCH_JIT", 1)  # 获取环境变量 PYTORCH_JIT，若未设置默认为 1
        assert not enabled  # 断言 JIT 功能未启用

    def __exit__(self, *args, **kwargs):
        pass  # 无操作

# 命名元组 RNNRunner，用于表示神经网络模型的运行器
RNNRunner = namedtuple(
    "RNNRunner",
    [
        "name",
        "creator",
        "context",
    ],
)

# 根据名称获取神经网络模型运行器的函数
def get_nn_runners(*names):
    return [nn_runners[name] for name in names]  # 返回指定名称的神经网络模型运行器列表

# 定义各种神经网络模型运行器的字典
nn_runners = {
    "cudnn": RNNRunner("cudnn", pytorch_lstm_creator, DummyContext),
    "cudnn_dropout": RNNRunner(
        "cudnn_dropout", partial(pytorch_lstm_creator, dropout=0.4), DummyContext
    ),
    "cudnn_layernorm": RNNRunner(
        "cudnn_layernorm", layernorm_pytorch_lstm_creator, DummyContext
    ),
    "vl_cudnn": RNNRunner("vl_cudnn", varlen_pytorch_lstm_creator, DummyContext),
    "vl_jit": RNNRunner(
        "vl_jit", partial(varlen_lstm_creator, script=True), DummyContext
    ),
    "vl_py": RNNRunner("vl_py", varlen_lstm_creator, DummyContext),
    "aten": RNNRunner("aten", pytorch_lstm_creator, DisableCuDNN),
    "jit": RNNRunner("jit", lstm_creator, DummyContext),
    "jit_premul": RNNRunner("jit_premul", lstm_premul_creator, DummyContext),
    "jit_premul_bias": RNNRunner(
        "jit_premul_bias", lstm_premul_bias_creator, DummyContext
    ),
    "jit_simple": RNNRunner("jit_simple", lstm_simple_creator, DummyContext),
    "jit_multilayer": RNNRunner(
        "jit_multilayer", lstm_multilayer_creator, DummyContext
    ),
    "jit_layernorm": RNNRunner("jit_layernorm", lnlstm_creator, DummyContext),
    "jit_layernorm_decom": RNNRunner(
        "jit_layernorm_decom",
        partial(lnlstm_creator, decompose_layernorm=True),
        DummyContext,
    ),
    "jit_dropout": RNNRunner("jit_dropout", dropoutlstm_creator, DummyContext),
    "py": RNNRunner("py", partial(lstm_creator, script=False), DummyContext),
    "resnet18": RNNRunner(
        "resnet18", imagenet_cnn_creator(cnn.resnet18, jit=False), DummyContext
    ),
    "resnet18_jit": RNNRunner(
        "resnet18_jit", imagenet_cnn_creator(cnn.resnet18), DummyContext
    ),
    "resnet50": RNNRunner(
        "resnet50", imagenet_cnn_creator(cnn.resnet50, jit=False), DummyContext
    ),
}
    # 创建一个名为"resnet50_jit"的RNNRunner对象，使用imagenet_cnn_creator函数创建一个ResNet50模型，使用DummyContext作为上下文
    "resnet50_jit": RNNRunner(
        "resnet50_jit", imagenet_cnn_creator(cnn.resnet50), DummyContext
    ),
}



# 这行代码表示一个单独的右大括号 '}'，用于结束一个代码块或者字典的定义。
```