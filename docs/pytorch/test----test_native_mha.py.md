# `.\pytorch\test\test_native_mha.py`

```py
# 导入 math 和 copy 模块
import math
import copy

# 导入 torch 库和测试相关模块
import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase, TEST_WITH_ROCM

# 定义测试类 TestMHADeviceType，继承自 TestCase 类
class TestMHADeviceType(TestCase):

    # 使用 torch.no_grad() 修饰器，禁用梯度计算的上下文管理器
    @torch.no_grad()
    # 定义内部方法 _test_transform_bias_rescale_qkv_impl，接受 device、dtype、use_nt 和 use_padding 参数
    def _test_transform_bias_rescale_qkv_impl(
        self, device, dtype, use_nt, use_padding=False
    ):
        # 省略方法内部实现，不在当前代码块中

    # 使用 dtypesIfCUDA(torch.float) 装饰器，用于指定 CUDA 环境下的数据类型为 torch.float
    # 使用 dtypes(torch.float) 装饰器，用于指定非 CUDA 环境下的数据类型为 torch.float
    # 使用 skipMeta 装饰器，跳过测试的元信息
    def test_transform_bias_rescale_qkv(self, device, dtype):
        # 遍历 use_padding 参数的两个取值 False 和 True
        for use_padding in (False, True):
            # 使用 self.subTest 方法创建子测试用例，用于分别测试 use_padding 参数为 False 和 True 时的情况
            with self.subTest(use_padding=use_padding):
                # 调用 _test_transform_bias_rescale_qkv_impl 方法，传入参数 device、dtype、use_nt 和 use_padding
                self._test_transform_bias_rescale_qkv_impl(
                    device, dtype, use_nt=False, use_padding=use_padding
                )

    # 使用 dtypesIfCUDA(torch.float) 装饰器，指定 CUDA 环境下的数据类型为 torch.float
    # 使用 dtypes(torch.float) 装饰器，指定非 CUDA 环境下的数据类型为 torch.float
    # 使用 skipMeta 装饰器，跳过测试的元信息
    # 使用 onlyCUDA 装饰器，仅在 CUDA 环境下运行测试
    def test_transform_bias_rescale_qkv_nested(self, device, dtype):
        # 遍历 use_padding 参数的两个取值 False 和 True
        for use_padding in (False, True):
            # 使用 self.subTest 方法创建子测试用例，用于分别测试 use_padding 参数为 False 和 True 时的情况
            with self.subTest(use_padding=use_padding):
                # 调用 _test_transform_bias_rescale_qkv_impl 方法，传入参数 device、dtype、use_nt 和 use_padding
                self._test_transform_bias_rescale_qkv_impl(
                    device, dtype, use_nt=True, use_padding=use_padding
                )

    # 定义内部方法 _test_multihead_attention_impl，接受 device、dtype、mode、use_nt、need_weights、average_attn_weights、use_padding 和 pad_all 参数
    @torch.no_grad()
    # 使用 dtypesIfCUDA(torch.float, torch.half) 装饰器，指定 CUDA 环境下的数据类型为 torch.float 和 torch.half
    # 使用 dtypes(torch.float) 装饰器，指定非 CUDA 环境下的数据类型为 torch.float
    # 使用 skipMeta 装饰器，跳过测试的元信息
    # 使用 parametrize 装饰器，为方法参数 parametrize 不同的取值组合
    def _test_multihead_attention_impl(
        self, device, dtype, mode, use_nt, need_weights, average_attn_weights, use_padding=False, pad_all=False
    ):
        # 省略方法内部实现，不在当前代码块中

    # 使用 parametrize 装饰器，为方法参数 parametrize 不同的取值组合
    # 使用 torch.no_grad() 修饰器，禁用梯度计算的上下文管理器
    @parametrize("use_nt", [False, True])
    @parametrize("use_padding, pad_all", [(False, False), (True, False), (True, True)])
    @parametrize("need_weights", [False])
    @parametrize("average_attn_weights", [False, True])
    @parametrize("fused", [False, True])
    @torch.no_grad()
    # 定义多头注意力测试方法
    def test_multihead_attention(self, device, dtype, mode, use_nt, need_weights, average_attn_weights, use_padding=False, pad_all=False):
        # 省略方法内部实现，不在当前代码块中
    # 测试多头自注意力机制的函数
    def test_native_multihead_self_attention(self, device, dtype, use_nt,
                                             need_weights, average_attn_weights, use_padding, pad_all, fused):
        # 如果使用 ROCM 并且需要嵌套张量，则跳过测试
        if TEST_WITH_ROCM and use_nt:
            self.skipTest("ROCM does not support nested tensors for Flash Attention for now.")
        
        # 对每一种权重需求情况进行测试
        for need_weights in (False, not pad_all):
            # 使用子测试框架，测试带有指定参数的情况
            with self.subTest(use_padding=use_padding, pad_all=pad_all,
                              use_nt=use_nt, need_weights=need_weights,
                              average_attn_weights=average_attn_weights):
                
                # 根据不同的融合方式选择不同的 CUDA 内核配置
                with torch.backends.cuda.sdp_kernel(
                        enable_flash=False, enable_mem_efficient=False
                ) if not fused else torch.backends.cuda.sdp_kernel(
                        enable_flash=True, enable_mem_efficient=True
                ):
                    # 调用测试多头注意力机制的具体实现函数
                    self._test_multihead_attention_impl(
                        device,
                        dtype,
                        "self",  # 指定使用自注意力
                        use_nt=use_nt,
                        use_padding=use_padding,
                        pad_all=pad_all,
                        need_weights=need_weights,
                        average_attn_weights=average_attn_weights,
                    )

    # 测试多头编码-解码注意力机制的函数
    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    @torch.no_grad()
    def test_native_multihead_encoder_decoder_attention(self, device, dtype):
        # 调用测试多头注意力机制的具体实现函数，使用编码-解码注意力
        self._test_multihead_attention_impl(
            device,
            dtype,
            "encdec",  # 指定使用编码-解码注意力
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )

    # 测试通用多头注意力机制的函数
    @dtypesIfCUDA(torch.float, torch.half)
    @dtypes(torch.float)
    @skipMeta
    @torch.no_grad()
    def test_native_multihead_attention(self, device, dtype):
        # 调用测试多头注意力机制的具体实现函数，使用通用多头注意力
        self._test_multihead_attention_impl(
            device,
            dtype,
            "generic",  # 指定使用通用多头注意力
            use_nt=False,
            need_weights=False,
            average_attn_weights=False,
        )
# 使用给定的测试类 TestMHADeviceType 和全局变量，实例化设备类型测试
instantiate_device_type_tests(TestMHADeviceType, globals())

# 如果当前脚本被直接执行（而不是被导入到其他模块中执行），则运行测试
if __name__ == "__main__":
    run_tests()
```