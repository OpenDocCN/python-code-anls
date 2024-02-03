# `bitsandbytes\tests\test_modules.py`

```py
# 导入数学库
import math

# 导入 einops 库
import einops

# 导入 pytest 库
import pytest

# 导入 torch 库
import torch

# 从 torch 库中导入 nn 模块
from torch import nn

# 导入 bitsandbytes 库并重命名为 bnb
import bitsandbytes as bnb

# 从 tests.helpers 中导入 id_formatter 函数
from tests.helpers import id_formatter

# 定义 MockArgs 类
class MockArgs:
    def __init__(self, initial_data):
        # 遍历初始数据的键值对，将其设置为 MockArgs 对象的属性
        for key in initial_data:
            setattr(self, key, initial_data[key])

# 定义 MLP8bit 类，继承自 torch.nn.Module
class MLP8bit(torch.nn.Module):
    def __init__(self, dim1, dim2, has_fp16_weights=True, memory_efficient_backward=False, threshold=0.0):
        super().__init__()
        # 初始化第一个全连接层，使用 bnb.nn.Linear8bitLt 类
        self.fc1 = bnb.nn.Linear8bitLt(
            dim1, dim2, has_fp16_weights=has_fp16_weights, memory_efficient_backward=memory_efficient_backward,
            threshold=threshold
        )
        # 初始化第二个全连接层，使用 bnb.nn.Linear8bitLt 类
        self.fc2 = bnb.nn.Linear8bitLt(
            dim2, dim1, has_fp16_weights=has_fp16_weights, memory_efficient_backward=memory_efficient_backward,
            threshold=threshold
        )

    # 前向传播函数
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 获取参数的函数
def get_args():
    # 创建 MockArgs 对象
    args = MockArgs([])
    # 设置参数
    args.quant_type = "vector"
    args.use_8bit_training = "full"
    args.clip_freq = 9999
    return args

# 断言两个张量的值近似相等的函数
def assert_all_approx_close(a, b, atol=1e-8, rtol=1e-5, count=10):
    # 找到不近似相等的值的索引
    idx = torch.isclose(a, b, rtol, atol)
    # 统计不近似相等的值的数量
    sumval = (idx == 0).sum().item()
    # 如果不近似相等的值数量超过阈值，则打印提示信息
    if sumval > count:
        print(f"Too many values not close: assert {sumval} < {count}")
        # 使用 torch.testing.assert_close 函数进行断言

# 定义 LinearFunction 类，继承自 torch.autograd.Function
class LinearFunction(torch.autograd.Function):
    # 静态方法，获取修剪后的 8 位线性值
    @staticmethod
    def get_8bit_linear_trimmed(x, stochastic=False, trim_value=3.0):
        # 如果 stochastic 为 True，则使用 LinearFunction.round_stoachastic 函数，否则使用 torch.round 函数
        round_func = (
            LinearFunction.round_stoachastic if stochastic else torch.round
        )
        # 计算标准差
        norm = math.sqrt(math.pi) / math.sqrt(2.0)
        std = torch.std(x)
        max1 = std * trim_value
        # 对 x 进行归一化处理
        x = x / max1 * 127
        # 对 x 进行四舍五入
        x = round_func(x)
        # 将大于 127 的值设置为 127
        x[x > 127] = 127
        # 将小于 -127 的值设置为 -127
        x[x < -127] = -127
        # 对 x 进行反归一化处理
        x = x / 127 * max1

        return x
    # 定义量化函数，根据不同的量化类型对输入进行量化处理
    def quant(x, quant_type, dim=1):
        # 如果量化类型为线性
        if quant_type == "linear":
            # 计算输入张量的绝对值的最大值
            max1 = torch.abs(x).max().float()
            # 对输入张量进行线性量化，将结果四舍五入为整数，并转换为int8类型
            xq = torch.round(x / max1 * 127).to(torch.int8)
            return xq, max1
        # 如果量化类型为向量
        elif quant_type == "vector":
            # 计算输入张量在指定维度上的绝对值的最大值
            max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
            # 对输入张量进行向量量化，将结果四舍五入为整数，并转换为int8类型
            xq = torch.round(x / max1 * 127).to(torch.int8)
            return xq, max1
        # 如果量化类型为最小-最大
        elif quant_type == "min-max":
            # 计算输入张量在指定维度上的最大值和最小值
            maxA = torch.amax(x, dim=dim, keepdim=True).float()
            minA = torch.amin(x, dim=dim, keepdim=True).float()
            # 计算缩放比例
            scale = (maxA - minA) / 2.0
            # 对输入张量进行最小-最大量化，将结果四舍五入为整数，并转换为int8类型
            xq = torch.round(127 * (x - minA - scale) / scale).to(torch.int8)
            return xq, (minA.float(), scale.float())
        else:
            return None

    # 定义反量化函数，根据不同的量化类型对输入进行反量化处理
    def dequant(xq, S1, S2, dtype, quant_type):
        # 如果量化类型为线性
        if quant_type == "linear":
            # 计算归一化系数
            norm = S1 * S2 / (127 * 127)
            # 执行反量化操作，并转换为指定数据类型
            return (xq.float() * norm).to(dtype)
        # 如果量化类型为向量
        elif quant_type == "vector":
            x = xq.float()
            # 处理特定情况下的维度匹配
            if len(xq.shape) == 2 and len(S1.shape) == 3:
                S1 = S1.squeeze(0)
            if len(xq.shape) == 2 and len(S2.shape) == 3:
                S2 = S2.squeeze(0)
            # 根据量化参数进行反量化操作
            if len(S1.shape) == 2:
                x *= S1.t() / 127
            else:
                x *= S1 / 127
            x *= S2 / 127
            return x.to(dtype)
        else:
            return None

    # 定义最小-最大反量化函数
    def dequant_min_max(xq, A, B, SA, SB, dtype):
        # 计算偏移量
        offset = B.float().t().sum(0) * (SA[0] + SA[1])
        x = xq.float()
        # 处理特定情况下的维度匹配
        if len(xq.shape) == 2 and len(SB.shape) == 3:
            SB = SB.squeeze(0)
        if len(xq.shape) == 2 and len(SA.shape) == 3:
            SA = SA.squeeze(0)
        # 根据量化参数进行反量化操作
        if len(SB.shape) == 2:
            x *= SB.t() / 127
        else:
            x *= SB / 127
        x *= SA[1] / 127
        x += offset
        return x.to(dtype)
    # 获取输入张量 x 的线性量化结果，将值缩放到 [-127, 127] 范围内
    def get_8bit_linear(x, stochastic=False):
        # 根据 stochastic 参数选择使用随机舍入函数或标准舍入函数
        round_func = (
            LinearFunction.round_stoachastic if stochastic else torch.round
        )
        # 计算输入张量 x 的绝对值的最大值
        max1 = torch.abs(x).max()
        # 将输入张量 x 缩放到 [-127, 127] 范围内
        x = x / max1 * 127
        # 对缩放后的张量进行舍入操作
        x = round_func(x) / 127 * max1
        # 返回量化结果
        return x

    # 获取输入张量 x 沿指定维度 dim 的向量线性量化结果
    @staticmethod
    def get_8bit_vector_wise(x, dim, stochastic=False):
        # 根据 stochastic 参数选择使用随机舍入函数或标准舍入函数
        round_func = (
            LinearFunction.round_stoachastic if stochastic else torch.round
        )
        # 计算输入张量 x 沿指定维度 dim 的绝对值的最大值
        max1 = torch.amax(torch.abs(x), dim=dim, keepdim=True)
        # 将输入张量 x 沿指定维度 dim 缩放到 [-127, 127] 范围内
        x = (x * 127) / max1
        # 对缩放后的张量进行舍入操作
        x = round_func(x) / 127 * max1
        # 返回量化结果
        return x

    # 实现随机舍入函数
    @staticmethod
    def round_stoachastic(x):
        # 计算输入张量 x 的符号
        sign = torch.sign(x)
        # 计算输入张量 x 的绝对值
        absx = torch.abs(x)
        # 计算输入张量 x 的小数部分
        decimal = absx - torch.floor(absx)
        # 生成与 decimal 相同形状的随机张量
        rdm = torch.rand_like(decimal)
        # 返回随机舍入结果
        return sign * (torch.floor(absx) + (rdm < decimal).to(x.dtype))

    # 对权重张量进行伪 8 位存储
    @staticmethod
    def fake_8bit_storage(w, exponent_bits):
        # 创建动态映射
        code = bnb.functional.create_dynamic_map(n=exponent_bits).to(w.device)
        # 对权重张量 w 进行分块量化
        absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
        # 对量化结果进行反量化
        out = bnb.functional.dequantize_blockwise(absmax, C, code)
        # 将反量化结果转换为半精度浮点数
        out = out.half()
        # 将反量化结果拷贝回原权重张量 w
        w.copy_(out)
        # 返回反量化结果
        return out

    # 对权重张量进行伪 8 位存储，并基于分位数估计量化参数
    @staticmethod
    def fake_8bit_storage_quantile(w, args):
        # 估计权重张量 w 的分位数
        code = bnb.functional.estimate_quantiles(w.data, offset=args.offset)
        # 将分位数参数归一化
        code /= torch.max(torch.abs(code))
        # 对权重张量 w 进行分块量化
        absmax, C = bnb.functional.quantize_blockwise(w.data, code=code)
        # 对量化结果进行反量化
        out = bnb.functional.dequantize_blockwise(absmax, C, code)
        # 将反量化结果转换为半精度浮点数
        out = out.half()
        # 将反量化结果拷贝回原权重张量 w
        w.copy_(out)
        # 返回反量化结果
        return out

    @staticmethod
    # 使用随机数生成器生成一个大小为1024的随机张量，设备为输入张量w所在设备
    rand = torch.rand(1024, device=w.device)
    # 对输入张量w进行分块量化，返回绝对值最大值和量化参数
    absmax, C = bnb.functional.quantize_blockwise(w.data, rand=rand)
    # 对量化后的张量进行反量化
    out = bnb.functional.dequantize_blockwise(absmax, C)
    # 将张量转换为半精度浮点数
    out = out.half()
    # 将处理后的张量复制回输入张量w
    w.copy_(out)
    # 返回处理后的张量
    return out

# 静态方法，用于处理8位假存储和最大值的情况
@staticmethod
def fake_8bit_storage_with_max(w, topk=8):
    # 将输入张量展平后重新排列成指定形状
    blocked_w = einops.rearrange(w.flatten(), "(h b) -> h b", b=256)
    # 对每个分块的绝对值进行排序，返回排序后的值和索引
    max_val, idx = torch.sort(torch.abs(blocked_w), dim=1, descending=True)
    # 保留每个分块中绝对值最大的topk个值的索引
    idx = idx[:, :topk]
    max_val = max_val[:, :topk]

    # 创建一个与blocked_w形状相同的全零张量，并根据索引将指定位置置为1
    mask = torch.zeros_like(blocked_w)
    mask.scatter_(dim=1, index=idx, src=torch.ones_like(max_val))
    mask = mask.bool()

    # 1. 将最大值置零
    # 2. 量化 + 反量化
    # 3. 恢复最大值
    # 4. 将矩阵复制回权重

    # 提取被置零的值
    values = blocked_w[mask]
    blocked_w[mask] = 0

    # 创建一个动态映射
    code = bnb.functional.create_dynamic_map()
    code = code.to(w.device)
    # 对被置零后的张量进行分块量化
    absmax, C = bnb.functional.quantize_blockwise(blocked_w.data)
    # 对量化后的张量进行反量化
    bnb.functional.dequantize_blockwise(absmax, C, out=blocked_w)

    # 恢复被置零的值
    blocked_w[mask] = values

    # 将处理后的张量展平后重新排列成原始形状
    unblocked_w = blocked_w.flatten().view(w.shape)

    # 将处理后的张量复制回输入张量w
    w.copy_(unblocked_w)
    # 返回处理后的张量
    return unblocked_w
    # 定义前向传播函数，接受输入 x、权重 weight、偏置 bias 和其他参数 args
    def forward(ctx, x, weight, bias=None, args=None):
        # 如果使用 8 位训练
        if args.use_8bit_training != "off":
            # 对权重进行量化，返回量化后的权重和缩放因子 S1
            weight8, S1 = LinearFunction.quant(weight, args.quant_type, dim=1)
            # 对输入 x 进行量化，返回量化后的输入和缩放因子 S2
            x8, S2 = LinearFunction.quant(x, args.quant_type, dim=2)
            # 使用量化后的输入和权重进行整数矩阵乘法
            outputq = bnb.functional.igemm(x8, weight8.t())
            # 对输出进行反量化，返回反量化后的输出
            output = LinearFunction.dequant(
                outputq, S1, S2, x.dtype, args.quant_type
            )
            # 如果随机数小于 0.01，计算输出的误差和相对误差
            # output32 = torch.matmul(x, weight.t())
            # err = torch.abs(output-output32).float()
            # relerr = err/(torch.abs(output32).float()+1e-8)
            # print(f'{err.mean().item():.4f}, {relerr.mean().item():.4f}', args.quant_type, 'forward', proxy)
        else:
            # 如果不使用 8 位训练，直接使用矩阵乘法计算输出
            # output = torch.matmul(x, weight.t())
            output = torch.einsum("bsi,oi->bso", x, weight)

        # 保存计算图中需要的变量
        ctx.save_for_backward(x, weight, bias)
        ctx.args = args

        # 如果存在偏置，将偏置加到输出上
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        # 返回输出
        return output

    @staticmethod
# 定义一个名为 Linear8bit 的类，继承自 nn.Module 类
class Linear8bit(nn.Module):
    # 初始化函数，接受输入特征数、输出特征数、是否包含偏置、其他参数
    def __init__(self, input_features, output_features, bias=True, args=None):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化输入特征数、输出特征数、其他参数
        self.input_features = input_features
        self.output_features = output_features
        self.args = args

        # 初始化权重参数为可训练参数
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        # 如果包含偏置，则初始化偏置参数为可训练参数
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # 否则注册偏置参数为 None
            self.register_parameter("bias", None)

        # 使用 Xavier 初始化方法初始化权重参数
        torch.nn.init.xavier_uniform_(self.weight)
        # 如果存在偏置参数，则使用零初始化方法初始化偏置参数
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    # 前向传播函数，接受输入 x
    def forward(self, x):
        # 设置参数的训练状态为当前模型的训练状态
        self.args.training = self.training

        # 调用 LinearFunction 的 apply 方法进行前向传播计算
        return LinearFunction.apply(x, self.weight, self.bias, self.args)


# 使用 pytest 的 parametrize 装饰器，参数化测试用例，测试 threshold 参数为 0.0 和 3.0 时的情况
@pytest.mark.parametrize("threshold", [0.0, 3.0], ids=id_formatter("threshold"))
def test_linear8bitlt_inference(threshold):
    # 创建一个 Linear8bitLt 实例，设置输入特征数为 32，输出特征数为 64，阈值为 threshold，放在 GPU 上，使用半精度浮点数
    l1 = bnb.nn.Linear8bitLt(32, 64, threshold=threshold).cuda().half()
    # 断言权重参数在 GPU 上
    assert l1.weight.device.type == "cuda"
    # 断言权重参数的数据类型为半精度浮点数
    assert l1.weight.dtype == torch.float16

    # 将模型设置为评估模式
    l1.eval()
    # 循环 100 次
    for i in range(100):
        # 生成随机输入数据，放在 GPU 上，使用半精度浮点数
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 进行前向传播计算
        o1 = l1(b1)
        # 如果是第一次迭代，断言模型状态中的 CxB 不为 None


# 测试 Linear8bitLt 类的梯度累积情况
def test_linear8bitlt_accumulated_gradient():
    # 创建两个包含两个 Linear8bitLt 实例的 Sequential 模型，放在 GPU 上，使用半精度浮点数
    l1 = torch.nn.Sequential(*[bnb.nn.Linear8bitLt(32, 32).cuda().half() for i in range(2)])
    l2 = torch.nn.Sequential(*[torch.nn.Linear(32, 32).cuda().half() for i in range(2)])
    # 将 l2 的权重和偏置数据复制给 l1
    l1[0].weight.data.copy_(l2[0].weight.data)
    l1[1].weight.data.copy_(l2[1].weight.data)
    l1[0].bias.data.copy_(l2[0].bias.data)
    l1[1].bias.data.copy_(l2[1].bias.data)

    # 创建两个 Adam32bit 优化器，分别用于 l1 和 l2 的参数优化
    opt1 = bnb.optim.Adam32bit(l1.parameters(), lr=0.001)
    opt2 = bnb.optim.Adam32bit(l2.parameters(), lr=0.001)

    # 设置梯度累积的步数为 10
    acc_steps = 10
    # 循环10次
    for i in range(10):
        # 生成一个16x8x32的张量，使用GPU加速，数据类型为半精度
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 将b1传入l1模型，得到输出o1
        o1 = l1(b1)
        # 将b1传入l2模型，得到输出o2
        o2 = l2(b1)
        # 计算o1的均值作为损失值loss1
        loss1 = o1.mean()
        # 计算o2的均值作为损失值loss2
        loss2 = o2.mean()
        # 反向传播计算loss1的梯度
        loss1.backward()
        # 反向传播计算loss2的梯度
        loss2.backward()
        
        # 当i等于2时
        if i == 2:
            # 断言l1的第一个元素的状态CxB不为None
            assert l1[0].state.CxB is not None
            # 断言l1的第二个元素的状态CxB不为None
            assert l1[1].state.CxB is not None

        # 当i大于0且i能整除acc_steps时
        if i > 0 and i % acc_steps == 0:
            # 对opt1进行一步优化
            opt1.step()
            # 清空opt1的梯度
            opt1.zero_grad(True)
            # 对opt2进行一步优化
            opt2.step()
            # 清空opt2的梯度
            opt2.zero_grad(True)
            # 断言l1和l2的第一个元素的权重近似相等
            assert_all_approx_close(
                l1[0].weight, l2[0].weight, rtol=1.05, atol=0.01, count=2
            )
            # 断言l1和l2的第二个元素的权重近似相等
            assert_all_approx_close(
                l1[1].weight, l2[1].weight, rtol=1.05, atol=0.01, count=2
            )
            # 复制l2的权重到l1，避免随时间累积的小差异
            l1[0].weight.data.copy_(l2[0].weight.data)
            l1[1].weight.data.copy_(l2[1].weight.data)
            l1[0].bias.data.copy_(l2[0].bias.data)
            l1[1].bias.data.copy_(l2[1].bias.data)
        else:
            # 断言l1和l2的第一个元素的权重梯度近似相等
            torch.testing.assert_close(l1[0].weight.grad, l2[0].weight.grad, atol=1e-3, rtol=1e-3)
            # 断言l1和l2的第二个元素的权重梯度近似相等
            torch.testing.assert_close(l1[1].weight.grad, l2[1].weight.grad, atol=1e-3, rtol=1e-3)
# 使用 pytest.mark.parametrize 装饰器为测试用例 test_linear8bitlt_no_fp16_weights 添加参数化测试
@pytest.mark.parametrize("threshold", [0.0, 2.0])
@pytest.mark.parametrize("memory_efficient_backward", [False])
def test_linear8bitlt_no_fp16_weights(threshold, memory_efficient_backward):
    # 创建 Linear8bitLt 模型对象 l1，设置参数并移动到 GPU，数据类型为半精度浮点数
    l1 = (bnb.nn.Linear8bitLt( 32, 64, threshold=threshold, has_fp16_weights=False, memory_efficient_backward=memory_efficient_backward).cuda().half())
    # 断言权重数据类型为 int8
    assert l1.weight.dtype == torch.int8

    # 将模型设置为评估模式
    l1.eval()
    # 循环进行前向传播测试
    for i in range(100):
        # 生成随机输入数据 b1，数据类型为半精度浮点数，移动到 GPU
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 模型前向传播
        o1 = l1(b1)
        # 断言输出数据类型为 float16
        assert o1.dtype == torch.float16

    # 创建 MLP8bit 模型对象 mlp，设置参数并移动到 GPU
    mlp = MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False).cuda()
    # 断言第一层全连接层权重数据类型为 int8
    assert mlp.fc1.weight.dtype == torch.int8
    # 断言第二层全连接层权重数据类型为 int8
    assert mlp.fc2.weight.dtype == torch.int8

    # 循环进行前向传播测试
    for i in range(100):
        # 生成随机输入数据 b1，数据类型为半精度浮点数，移动到 GPU
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 模型前向传播
        o1 = mlp(b1)
        # 断言输出数据类型为 float16
        assert o1.dtype == torch.float16
        # 如果阈值大于 0，则断言第一层全连接层状态索引不为 None
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        # 如果阈值大于 0，则断言第二层全连接层状态索引不为 None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None

    # 创建 MLP8bit 模型对象 mlp，设置参数并移动到 GPU，数据类型为半精度浮点数
    mlp = (
        MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False)
        .cuda()
        .half()
    )
    # 断言第一层全连接层权重数据类型为 int8
    assert mlp.fc1.weight.dtype == torch.int8
    # 断言第二层全连接层权重数据类型为 int8
    assert mlp.fc2.weight.dtype == torch.int8

    # 循环进行前向传播测试
    for i in range(100):
        # 生成随机输入数据 b1，数据类型为半精度浮点数，移动到 GPU
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 模型前向传播
        o1 = mlp(b1)
        # 断言输出数据类型为 float16
        assert o1.dtype == torch.float16
        # 如果阈值大于 0，则断言第一层全连接层状态索引不为 None
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        # 如果阈值大于 0，则断言第二层全连接层状态索引不为 None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None

    # 创建 MLP8bit 模型对象 mlp，设置参数并数据类型为半精度浮点数，移动到 GPU
    mlp = (
        MLP8bit(32, 64, threshold=threshold, has_fp16_weights=False)
        .half()
        .cuda()
    )

    # 循环进行前向传播测试
    for i in range(100):
        # 生成随机输入数据 b1，数据类型为半精度浮点数，移动到 GPU
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 模型前向传播
        o1 = mlp(b1)
        # 断言输出数据类型为 float16
        assert o1.dtype == torch.float16
        # 如果阈值大于 0，则断言第一层全连接层状态索引不为 None
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        # 如果阈值大于 0，则断言第二层全连接层状态索引不为 None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    # 断言第一层全连接层权重数据类型为 int8
    assert mlp.fc1.weight.dtype == torch.int8
    # 断言多层感知机（MLP）的第二个全连接层的权重数据类型为 torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    
    # 创建一个具有指定参数的 8 位 MLP 模型，并将其转换为半精度浮点数，然后移动到 GPU 上
    mlp = ( MLP8bit( 32, 64, threshold=threshold, has_fp16_weights=False, memory_efficient_backward=memory_efficient_backward).half().to("cuda"))
    
    # 循环执行 100 次
    for i in range(100):
        # 生成一个半精度浮点数的随机张量，形状为 (16, 8, 32)，并移动到 GPU 上
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 将输入张量传入 MLP 模型，得到输出张量 o1
        o1 = mlp(b1)
        # 断言 o1 的数据类型为 torch.float16
        assert o1.dtype == torch.float16
        # 如果阈值大于 0，则断言 MLP 的第一个全连接层的状态索引不为 None
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        # 如果阈值大于 0，则断言 MLP 的第二个全连接层的状态索引不为 None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    # 断言 MLP 的第一个全连接层的权重数据类型为 torch.int8
    assert mlp.fc1.weight.dtype == torch.int8
    # 断言 MLP 的第二个全连接层的权重数据类型为 torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    # 断言 MLP 的第一个全连接层的权重设备类型为 "cuda"
    assert mlp.fc1.weight.device.type == "cuda"
    # 断言 MLP 的第二个全连接层的权重设备类型为 "cuda"
    
    # 创建一个具有指定参数的 8 位 MLP 模型
    mlp = MLP8bit(
            32, 64, threshold=threshold, has_fp16_weights=False, memory_efficient_backward=memory_efficient_backward
        )
    # 复制 MLP 的第一个全连接层和第二个全连接层的权重，并将其移动到 GPU 上
    w1, w2 = mlp.fc1.weight.clone().cuda(), mlp.fc2.weight.clone().cuda()  # grab weights before quantization,
    # 将 MLP 模型移动到 GPU 上并转换为半精度浮点数，这一行触发了量化操作
    mlp = mlp.cuda().half()
    
    # 循环执行 100 次
    for i in range(100):
        # 生成一个半精度浮点数的随机张量，形状为 (16, 8, 32)，并移动到 GPU 上
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 将输入张量传入 MLP 模型，得到输出张量 o1
        o1 = mlp(b1)
        # 断言 o1 的数据类型为 torch.float16
        assert o1.dtype == torch.float16
        # 如果阈值大于 0，则断言 MLP 的第一个全连接层的状态索引不为 None
        if threshold > 0:
            assert mlp.fc1.state.idx is not None
        # 如果阈值大于 0，则断言 MLP 的第二个全连接层的状态索引不为 None
        if threshold > 0:
            assert mlp.fc2.state.idx is not None
    
    # 断言 MLP 的第一个全连接层的权重数据类型为 torch.int8
    assert mlp.fc1.weight.dtype == torch.int8
    # 断言 MLP 的第二个全连接层的权重数据类型为 torch.int8
    assert mlp.fc2.weight.dtype == torch.int8
    # 断言 MLP 的第一个全连接层的权重设备类型为 "cuda"
    assert mlp.fc1.weight.device.type == "cuda"
    # 断言 MLP 的第二个全连接层的权重设备类型为 "cuda"
    # 如果选择内存效率向后传播
    if memory_efficient_backward:
        # 创建一个16x8x32的张量，使用半精度数据类型，在GPU上运行，需要梯度计算
        b1 = torch.randn(16, 8, 32, device="cuda", requires_grad=True, dtype=torch.half)
        # 将张量传入MLP模型中进行前向传播
        o1 = mlp(b1)
        # 断言输出o1的数据类型为torch.float16
        assert o1.dtype == torch.float16
        # 断言输出o1需要梯度计算
        assert o1.requires_grad
        # 创建一个与o1相同大小的随机梯度张量
        grad_proj = torch.randn_like(o1)

        # 将MLP模型的梯度清零
        mlp.zero_grad()
        # 计算(o1 * grad_proj)的和，并进行反向传播
        (o1 * grad_proj).sum().backward()
        # 计算梯度的参考值
        grad_ref = grad_proj.flatten(2) @ w2.half() @ w1.half()
        # 计算梯度的绝对值的平均值
        scale = grad_ref.abs().mean()

        # 断言b1的梯度与grad_ref的值在一定范围内接近
        torch.testing.assert_close(b1.grad, grad_ref, rtol=0, atol=0.05 * scale)
        # 判断b1的梯度与grad_ref的值是否在一定范围内接近
        idx = torch.isclose(b1.grad, grad_ref, atol=0.01 * scale, rtol=0.1)
        # 断言不接近的元素数量不超过总元素数量的0.005倍
        assert (idx == 0).sum().item() <= b1.numel() * 0.005
# 使用 pytest.mark.parametrize 装饰器为 test_linear_kbit_fp32_bias 函数添加参数化测试
@pytest.mark.parametrize("module", [lambda nin, nout, bias=True: bnb.nn.Linear8bitLt(nin, nout, bias=bias, has_fp16_weights=False), bnb.nn.LinearFP4], ids=['Int8Lt', 'FP4'])
# 定义测试函数 test_linear_kbit_fp32_bias，测试线性模型的 k 位精度和浮点数偏置
def test_linear_kbit_fp32_bias(module):
    # 将模型转换为 fp16 -> int8 自动转换
    l1 = module(32, 64).cuda()
    # 断言权重的数据类型为 int8 或 uint8
    assert l1.weight.dtype in [torch.int8, torch.uint8]
    # 断言偏置的数据类型为 float32
    assert l1.bias.dtype == torch.float32

    for i in range(100):
        # 生成随机输入数据 b1，并转换为半精度
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        # 将偏置转换为 float32
        o1 = l1(b1)
        # 断言偏置的数据类型为 float16
        assert l1.bias.dtype == torch.float16

    # 将模型转换为 fp16 -> int8 自动转换，不包含偏置
    l1 = module(32, 64, bias=False).cuda()
    # 断言权重的数据类型为 int8 或 uint8
    assert l1.weight.dtype in [torch.int8, torch.uint8]
    # 断言偏置为 None
    assert l1.bias is None

    for i in range(100):
        # 生成随机输入数据 b1，并转换为半精度
        b1 = torch.randn(16, 8, 32, device="cuda").half()
        o1 = l1(b1)
        # 断言偏置为 None
        assert l1.bias is None

# 定义模型字典 module_dict，包含不同类型的线性模型
module_dict = {
    "Int8Lt": bnb.nn.Linear8bitLt,
    "4bit": bnb.nn.Linear4bit,
    "FP4": bnb.nn.LinearFP4,
    "NF4": bnb.nn.LinearNF4,
    "FP4+C": lambda d1, d2: bnb.nn.LinearFP4(d1, d2, compress_statistics=True),
    "NF4+C": lambda d1, d2: bnb.nn.LinearNF4(d1, d2, compress_statistics=True),
    "NF4+fp32": lambda d1, d2: bnb.nn.LinearFP4(d1, d2, compute_dtype=torch.float32),
    "NF4+fp16": lambda d1, d2: bnb.nn.LinearFP4(d1, d2, compute_dtype=torch.float16),
    "NF4+bf16": lambda d1, d2: bnb.nn.LinearFP4(d1, d2, compute_dtype=torch.bfloat16),
}

# 使用 pytest.mark.parametrize 装饰器为 test_kbit_backprop 函数添加参数化测试
@pytest.mark.parametrize("module", module_dict.values(), ids=module_dict.keys())
# 定义测试函数 test_kbit_backprop，测试 k 位精度模型的反向传播
def test_kbit_backprop(module):
    # 初始化参数
    b = 17
    dim1 = 37
    dim2 = 83

    # 创建参考模型 ref，包含两个线性层
    ref = nn.Sequential(*[torch.nn.Linear(dim1, dim2), torch.nn.Linear(dim2, 10)])
    ref[1].weight.requires_grad = False
    torch.nn.init.kaiming_normal_(ref[0].weight)
    torch.nn.init.kaiming_normal_(ref[1].weight)
    # 创建 k 位精度模型 kbit，包含两个线性层
    kbit = nn.Sequential(*[torch.nn.Linear(dim1, dim2), module(dim2, 10)])
    kbit[0].weight.detach().copy_(ref[0].weight)
    kbit[1].weight.detach().copy_(ref[1].weight)
    # 将第一个模型的偏置参数复制到第二个模型的偏置参数
    kbit[0].bias.detach().copy_(ref[0].bias)
    # 将第二个模型的偏置参数转换为半精度浮点数，并移动到 GPU 上
    kbit[1].bias.detach().copy_(ref[1].bias)
    # 将参考模型转换为半精度浮点数，并移动到 GPU 上
    ref = ref.half().cuda()
    # 将第二个模型转换为半精度浮点数，并移动到 GPU 上
    kbit = kbit.half().cuda()
    # 将第二个模型转换为半精度浮点数，并移动到 GPU 上
    kbit = kbit.half().to('cuda')

    # 初始化用于存储误差的列表
    errs1 = []
    errs2 = []
    relerrs1 = []
    relerrs2 = []
    # 循环执行100次
    for i in range(100):
        # 生成一个半精度浮点数的随机张量，并移动到 GPU 上
        batch = torch.randn(b, dim1).half().cuda()
        # 使用参考模型和第二个模型对输入进行前向传播
        out1 = ref(batch)
        out2 = kbit(batch)
        # 计算输出的均值并进行反向传播
        out1.mean().backward()
        out2.mean().backward()

        # 获取参考模型和第二个模型的权重梯度和偏置梯度
        grad1 = ref[0].weight.grad
        grad2 = kbit[0].weight.grad
        bgrad1 = ref[0].bias.grad
        bgrad2 = kbit[0].bias.grad

        # 计算输出误差和梯度误差
        err1 = (out1-out2).abs().float()
        err2 = (grad1-grad2).abs().float()
        # 计算相对误差
        relerr1 = (err1/(out1.abs().float()+1e-9))
        relerr2 = (err2/(grad1.abs().float()+1e-9))
        # 将误差和相对误差的均值添加到对应的列表中
        errs1.append(err1.mean().item())
        errs2.append(err2.mean().item())
        relerrs1.append(relerr1.mean().item())
        relerrs2.append(relerr2.mean().item()

        # 根据模型类型进行误差检查
        if isinstance(module, bnb.nn.Linear8bitLt):
            assert_all_approx_close(grad1, grad2, atol=0.008, rtol=0.05, count=1)
            torch.testing.assert_close(bgrad1, bgrad2, atol=0.008, rtol=0.05)
        else:
            assert_all_approx_close(grad1, grad2, atol=0.015, rtol=0.05, count=1)
            torch.testing.assert_close(bgrad1, bgrad2, atol=0.02, rtol=0.05)
        # 清空模型的梯度
        ref.zero_grad()
        kbit.zero_grad()

        # 断言第二个模型的权重梯度和偏置梯度为 None 或者梯度和为 0
        assert kbit[0].weight.grad is None or kbit[0].weight.grad.sum().item() == 0
        assert kbit[0].weight.grad is None or kbit[0].bias.grad.sum().item() == 0
    # 打印输出误差、梯度误差、相对输出误差和相对梯度误差的均值
    #print('out', sum(errs1)/len(errs1))
    #print('grad', sum(errs2)/len(errs2))
    #print('rel out', sum(relerrs1)/len(relerrs1))
    #print('rel grad', sum(relerrs2)/len(relerrs2))
# 测试使用8位整数和32位浮点数线性层的性能
def test_fp8linear():

    # 定义批量大小和隐藏层大小
    b = 10
    h = 1024
    # 生成随机输入数据并移动到GPU
    inp = torch.randn(b, h).cuda()
    # 创建32位浮点数线性层
    fp32 = torch.nn.Linear(h, h*2).cuda()
    # 创建8位整数和32位浮点数混合线性层
    fp8 = bnb.research.nn.LinearFP8Mixed(h, h*2).cuda()
    # 创建32位浮点数线性层
    fp32b = torch.nn.Linear(h*2, h).cuda()
    # 创建8位整数和32位浮点数混合线性层
    fp8b = bnb.research.nn.LinearFP8Mixed(h*2, h).cuda()

    # 将32位浮点数线性层的权重复制到8位整数和32位浮点数混合线性层
    fp8.weight.data.copy_(fp32.weight.data)
    fp8.bias.data.copy_(fp32.bias.data)
    fp8b.weight.data.copy_(fp32b.weight.data)
    fp8b.bias.data.copy_(fp32b.bias.data)

    # 计算32位浮点数线性层和激活函数后的输出
    a = fp32b(torch.nn.functional.gelu(fp32(inp)))
    # 计算8位整数和32位浮点数混合线性层和激活函数后的输出
    b = fp8b(torch.nn.functional.gelu(fp8(inp)))

    # 计算两种输出之间的绝对误差的平均值
    err = (a-b).abs().mean()

    # 计算32位浮点数线性层的梯度
    a.mean().backward()
    # 计算8位整数和32位浮点数混合线性层的梯度
    b.mean().backward()

    # 计算权重梯度的绝对误差的平均值
    graderr = (fp8.weight.grad-fp32.weight.grad).abs().mean()
    # 计算偏置梯度的绝对误差的平均值
    bgraderr = (fp8.bias.grad-fp32.bias.grad).abs().mean()

    # 断言误差小于0.05
    assert err < 0.05
    # 断言权重梯度的误差小于0.00002
    assert graderr < 0.00002
    # 断言偏置梯度的误差小于0.00002

# 测试4位整数线性层的警告
def test_4bit_warnings():
    # 定义隐藏层大小
    dim1 = 64

    # 检查是否会发出用户警告，匹配'inference or training'
    with pytest.warns(UserWarning, match=r'inference or training'):
        # 创建包含10个4位整数线性层的神经网络
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        # 生成随机输入数据并移动到GPU
        inp = torch.rand(10, dim1).cuda().half()
        # 运行神经网络
        net(inp)
    # 检查是否会发出用户警告，匹配'inference.'
    with pytest.warns(UserWarning, match=r'inference.'):
        # 创建包含10个4位整数线性层的神经网络
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        # 生成随机输入数据并移动到GPU
        inp = torch.rand(1, dim1).cuda().half()
        # 运行神经网络
        net(inp)

    # 检查是否会发出用户警告，并记录警告信息
    with pytest.warns(UserWarning) as record:

        # 创建包含10个4位整数线性层的神经网络
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        # 生成随机输入数据并移动到GPU
        inp = torch.rand(10, dim1).cuda().half()
        # 运行神经网络
        net(inp)

        # 创建包含10个4位整数线性层的神经网络
        net = nn.Sequential(*[bnb.nn.Linear4bit(dim1, dim1, compute_dtype=torch.float32) for i in range(10)])
        net = net.cuda()
        # 生成随机输入数据并移动到GPU
        inp = torch.rand(1, dim1).cuda().half()
        # 运行神经网络
        net(inp)

    # 断言记录的警告数量为2
    assert len(record) == 2
```