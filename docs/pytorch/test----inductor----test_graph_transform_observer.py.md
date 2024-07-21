# `.\pytorch\test\inductor\test_graph_transform_observer.py`

```py
# Owner(s): ["module: inductor"]

# 导入所需的库
import glob
import math
import os
import shutil
import tempfile

# 导入 PyTorch 相关模块
import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA

try:
    import pydot  # noqa: F401

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False

# 检查是否有 'dot' 命令可用
HAS_DOT = True if shutil.which("dot") is not None else False

# 定义测试类 TestGraphTransformObserver，继承自 TestCase
class TestGraphTransformObserver(TestCase):
    # 装饰器，用于跳过 ROCm 平台的测试
    @skipIfRocm
    def test_sdpa_rewriter(self):
        # 如果不满足测试条件，则直接返回
        if not (
            HAS_CUDA and PLATFORM_SUPPORTS_FUSED_ATTENTION and HAS_PYDOT and HAS_DOT
        ):
            return

        # 定义点积注意力函数，接受三个张量输入
        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            # 计算点积注意力的输出张量
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        # 创建临时目录来存储日志文件
        log_url = tempfile.mkdtemp()
        # 设置图形转换观察器的日志 URL
        inductor_config.trace.log_url_for_graph_xform = log_url
        # 强制禁用缓存
        inductor_config.force_disable_caches = True
        # 编译点积注意力函数，生成完整的计算图
        compiled_fn = torch.compile(dot_prod_attention, fullgraph=True)

        # 定义张量的形状
        tensor_shape = (4, 2, 16, 32)
        # 在 CUDA 设备上生成随机张量
        q = torch.randn(tensor_shape, device="cuda")
        k = torch.randn(tensor_shape, device="cuda")
        v = torch.randn(tensor_shape, device="cuda")
        # 执行编译后的函数
        compiled_fn(q, k, v)

        # 初始化变量，用于检查是否找到输入和输出的 SVG 文件
        found_input_svg = False
        found_output_svg = False
        # 遍历临时目录下的所有文件和文件夹
        for filepath_object in glob.glob(log_url + "/*"):
            # 如果是文件
            if os.path.isfile(filepath_object):
                # 检查是否是输入图形的 DOT 文件
                if filepath_object.endswith("input_graph.dot"):
                    found_input_svg = True
                # 检查是否是输出图形的 DOT 文件
                elif filepath_object.endswith("output_graph.dot"):
                    found_output_svg = True

        # 断言找到了输入和输出的 SVG 文件
        self.assertTrue(found_input_svg)
        self.assertTrue(found_output_svg)


if __name__ == "__main__":
    # 如果运行在 Linux 环境下，运行测试
    if IS_LINUX:
        run_tests()
```