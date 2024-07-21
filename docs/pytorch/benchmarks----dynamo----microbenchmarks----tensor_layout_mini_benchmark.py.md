# `.\pytorch\benchmarks\dynamo\microbenchmarks\tensor_layout_mini_benchmark.py`

```
import torch  # 导入PyTorch库
from torch._inductor import ir  # 从torch._inductor导入ir模块
from torch._inductor.runtime.runtime_utils import do_bench  # 从torch._inductor.runtime.runtime_utils导入do_bench函数


def to_channels_last(x):
    assert x.dim() == 4  # 断言x的维度为4

    # NCHW -> NHWC的转换顺序
    stride_order = [3, 0, 2, 1]
    y = x.clone().as_strided(
        x.shape,
        ir.FlexibleLayout.stride_ordered(x.shape, stride_order),  # 根据指定顺序生成NHWC布局的数据
    )
    y.copy_(x)  # 将x的数据复制到y中
    assert torch.allclose(x, y)  # 断言x和y的数据在数值上相等
    return y


def bench_conv(with_stack=True):
    x = torch.rand(256, 3, 224, 224).cuda()  # 创建一个大小为(256, 3, 224, 224)的随机张量，并放置在CUDA上
    weight = torch.rand(64, 3, 7, 7).cuda()  # 创建一个大小为(64, 3, 7, 7)的随机张量，并放置在CUDA上

    x_chan = to_channels_last(x)  # 将输入张量x转换为通道最后的布局
    weight_chan = to_channels_last(weight)  # 将权重张量weight转换为通道最后的布局
    kwargs = {
        "stride": [2, 2],  # 卷积操作的步幅
        "padding": [3, 3],  # 卷积操作的填充
        "dilation": [1, 1],  # 卷积操作的膨胀率
        "transposed": False,  # 是否是转置卷积
        "output_padding": [0, 0],  # 输出填充
        "groups": 1,  # 卷积操作的组数
    }

    def baseline_fn():
        return torch.convolution(x, weight, bias=None, **kwargs)  # 执行基准卷积操作

    def test_fn():
        return torch.convolution(x_chan, weight_chan, bias=None, **kwargs)  # 执行通道最后布局下的卷积操作

    # 预热
    baseline_fn()  # 执行基准卷积操作
    test_fn()  # 执行通道最后布局下的卷积操作

    torch.cuda.synchronize()  # 等待CUDA操作完成
    with torch.profiler.profile(with_stack=with_stack) as p:  # 使用profiler进行性能分析
        baseline_out = baseline_fn()  # 记录基准卷积操作的输出
        test_out = test_fn()  # 记录通道最后布局下卷积操作的输出
        torch.cuda.synchronize()  # 等待CUDA操作完成

    p.export_chrome_trace("/tmp/chrome.json")  # 将性能分析结果导出为Chrome跟踪文件
    assert torch.allclose(baseline_out, test_out, atol=1e-3, rtol=1e-3), (  # 断言基准和测试卷积操作的输出在数值上接近
        baseline_out[0][0][0][:32],  # 输出基准卷积操作的部分数据
        test_out[0][0][0][:32],  # 输出测试卷积操作的部分数据
    )

    baseline_ms = do_bench(baseline_fn, rep=40)  # 使用bench函数进行基准卷积操作的性能评估
    test_ms = do_bench(test_fn, rep=40)  # 使用bench函数进行测试卷积操作的性能评估
    print(f"baseline {baseline_ms} test {test_ms} speedup {baseline_ms / test_ms:.3f}x")  # 打印性能评估结果


def main():
    bench_conv()  # 执行卷积性能评估


if __name__ == "__main__":
    main()  # 如果作为主程序运行，则调用main函数执行
```