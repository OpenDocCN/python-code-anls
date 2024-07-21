# `.\pytorch\test\distributed\pipelining\test_microbatch.py`

```
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]

# 从模块中导入需要的类和函数
from model_registry import ModelWithKwargs

# 导入 PyTorch 库
import torch

# 从 torch.distributed.pipelining 中导入 pipeline 函数
from torch.distributed.pipelining import pipeline

# 从 torch.distributed.pipelining.microbatch 中导入相关函数和类
from torch.distributed.pipelining.microbatch import (
    merge_chunks,
    split_args_kwargs_into_chunks,
    TensorChunkSpec,
)

# 从 torch.testing._internal.common_utils 中导入测试相关的函数和类
from torch.testing._internal.common_utils import run_tests, TestCase

# 定义隐藏层维度
d_hid = 512

# 设定随机种子
torch.manual_seed(0)

# 定义测试类 MicrobatchTests，继承自 TestCase 类
class MicrobatchTests(TestCase):
    
    # 定义测试方法 test_split_and_merge
    def test_split_and_merge(self):
        # 创建三个张量 x0, x1, x2，形状分别为 (128, 512), (256, 512), (512, 512)
        x0 = torch.randn(128, d_hid)
        x1 = torch.randn(256, d_hid)
        x2 = torch.randn(512, d_hid)

        # 将张量 x0, x1, x2 组成元组 args
        args = (x0, x1, x2)

        # 将张量 x0, x1, x2 组成字典 kwargs
        kwargs = {"x0": x0, "x1": x1, "x2": x2}

        # 使用默认的参数对 args 和 kwargs 进行分块，每个块的大小为 2
        arg_chunks, kwarg_chunks = split_args_kwargs_into_chunks(args, kwargs, 2)

        # 断言分块后的张量列表和字典的长度为 2
        assert len(arg_chunks) == 2
        assert len(kwarg_chunks) == 2

        # 断言分块后的第一个 arg_chunks 的第一个元素形状为 [64, 512]
        assert arg_chunks[0][0].shape == torch.Size([64, d_hid])

        # 断言分块后的第二个 arg_chunks 的第一个元素形状为 [64, 512]
        assert arg_chunks[1][0].shape == torch.Size([64, d_hid])

        # 断言分块后的第一个 arg_chunks 的第二个元素形状为 [128, 512]
        assert arg_chunks[0][1].shape == torch.Size([128, d_hid])

        # 断言分块后的第一个 arg_chunks 的第三个元素形状为 [256, 512]
        assert arg_chunks[0][2].shape == torch.Size([256, d_hid])

        # 断言分块后的第一个 kwarg_chunks 的 "x0" 元素形状为 [64, 512]
        assert kwarg_chunks[0]["x0"].shape == torch.Size([64, d_hid])

        # 断言分块后的第一个 kwarg_chunks 的 "x1" 元素形状为 [128, 512]
        assert kwarg_chunks[0]["x1"].shape == torch.Size([128, d_hid])

        # 断言分块后的第二个 kwarg_chunks 的 "x2" 元素形状为 [256, 512]
        assert kwarg_chunks[1]["x2"].shape == torch.Size([256, d_hid])

        # 将分块后的参数再次合并
        merged_args = merge_chunks(
            arg_chunks,
            (TensorChunkSpec(0), TensorChunkSpec(0), TensorChunkSpec(0)),
        )

        # 使用测试工具函数验证合并后的参数是否与原始参数相等
        torch.testing.assert_close(merged_args, args)

        # 将分块后的关键字参数再次合并
        merged_kwargs = merge_chunks(
            kwarg_chunks,
            {
                "x0": TensorChunkSpec(0),
                "x1": TensorChunkSpec(0),
                "x2": TensorChunkSpec(0),
            },
        )

        # 使用测试工具函数验证合并后的关键字参数是否与原始关键字参数相等
        torch.testing.assert_close(merged_kwargs, kwargs)

        # 打印测试通过消息
        print("Microbatch test passed")

    # 定义测试方法 test_chunk_spec
    def test_chunk_spec(self):
        # 创建 ModelWithKwargs 的实例 mod
        mod = ModelWithKwargs()

        # 获取 ModelWithKwargs 类的默认批处理大小
        batch_size = ModelWithKwargs.DEFAULT_BATCH_SIZE

        # 创建形状为 (batch_size, 512) 的张量 x 和 y
        x = torch.randn(batch_size, d_hid)
        y = torch.randn(batch_size, d_hid)

        # 设定分块数目为 4
        num_chunks = 4

        # 创建基于元组的 TensorChunkSpec 参数规范
        args_chunk_spec = TensorChunkSpec.from_tuple((0,))

        # 创建基于字典的 TensorChunkSpec 参数规范
        kwargs_chunk_spec = TensorChunkSpec.from_dict({"y": 0})

        # 使用给定的参数规范将输入的张量 x 和 y 进行分块
        args_split, kwargs_split = split_args_kwargs_into_chunks(
            (x,),
            {"y": y},
            num_chunks,
            args_chunk_spec,
            kwargs_chunk_spec,
        )

        # 使用 pipeline 函数创建流水线
        pipe = pipeline(
            mod,
            mb_args=args_split[0],
            mb_kwargs=kwargs_split[0],
        )

        # 调用模型的正向传播，并赋值给 ref
        ref = mod(x, y)

        # 在流水线上执行正向传播，并取第一个输出
        out = pipe(x, y)[0]

        # 使用测试工具函数验证流水线的输出是否与模型的输出相等
        torch.testing.assert_close(out, ref)

        # 打印测试通过消息，显示输出的总和和参考值的总和
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")

# 如果脚本作为主程序执行，则运行测试
if __name__ == "__main__":
    run_tests()
```