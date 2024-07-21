# `.\pytorch\torch\nn\modules\_functions.py`

```py
# mypy: allow-untyped-defs
# 导入PyTorch库
import torch
# 导入分布式通信模块
import torch.distributed as dist
# 从PyTorch自动求导函数库中导入Function类
from torch.autograd.function import Function


# 定义SyncBatchNorm类，继承自Function类
class SyncBatchNorm(Function):
    # 前向传播方法，静态方法
    @staticmethod
    def forward(
        self,
        input,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        momentum,
        process_group,
        world_size,
    ):
        # 省略了具体实现的前向传播逻辑
        pass


# 定义CrossMapLRN2d类，继承自Function类
class CrossMapLRN2d(Function):
    # 前向传播方法，静态方法
    @staticmethod
    def forward(ctx, input, size, alpha=1e-4, beta=0.75, k=1):
        # 保存超参数和上下文信息到ctx对象中
        ctx.size = size
        ctx.alpha = alpha
        ctx.beta = beta
        ctx.k = k
        ctx.scale = None

        # 检查输入是否为4维张量，否则引发异常
        if input.dim() != 4:
            raise ValueError(
                f"CrossMapLRN2d: Expected input to be 4D, got {input.dim()}D instead."
            )

        # 创建新的张量用作输出
        ctx.scale = ctx.scale or input.new()
        output = input.new()

        # 获取输入张量的尺寸信息
        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        # 调整输出和缩放张量的尺寸以匹配输入
        output.resize_as_(input)
        ctx.scale.resize_as_(input)

        # 使用输出存储作为临时缓冲区
        input_square = output
        # 计算输入的平方，并将结果存储在input_square中
        torch.pow(input, 2, out=input_square)

        # 计算前填充量
        pre_pad = int((ctx.size - 1) / 2 + 1)
        pre_pad_crop = min(pre_pad, channels)

        # 获取缩放张量的第一个特征图
        scale_first = ctx.scale.select(1, 0)
        scale_first.zero_()
        # 计算第一个特征图的归一化
        for c in range(pre_pad_crop):
            scale_first.add_(input_square.select(1, c))

        # 通过添加下一个特征图并移除上一个特征图来重复计算
        for c in range(1, channels):
            scale_previous = ctx.scale.select(1, c - 1)
            scale_current = ctx.scale.select(1, c)
            scale_current.copy_(scale_previous)
            if c < channels - pre_pad + 1:
                square_next = input_square.select(1, c + pre_pad - 1)
                scale_current.add_(square_next, alpha=1)

            if c > pre_pad:
                square_previous = input_square.select(1, c - pre_pad)
                scale_current.add_(square_previous, alpha=-1)

        # 缩放乘以常数因子并加上偏置项
        ctx.scale.mul_(ctx.alpha / ctx.size).add_(ctx.k)

        # 计算最终输出
        torch.pow(ctx.scale, -ctx.beta, out=output)
        output.mul_(input)

        # 保存输入和输出张量用于反向传播
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    # 定义一个反向传播函数，接受上下文和梯度输出作为输入
    def backward(ctx, grad_output):
        # 从上下文中获取保存的张量输入和输出
        input, output = ctx.saved_tensors
        # 创建一个新的梯度输入张量
        grad_input = grad_output.new()

        # 获取输入张量的批量大小、通道数、高度和宽度
        batch_size = input.size(0)
        channels = input.size(1)
        input_height = input.size(2)
        input_width = input.size(3)

        # 创建一个新的填充后的比例张量，尺寸为（channels + ctx.size - 1, input_height, input_width）
        paddded_ratio = input.new(channels + ctx.size - 1, input_height, input_width)
        # 创建一个累积比例张量，尺寸为（input_height, input_width）
        accum_ratio = input.new(input_height, input_width)

        # 计算缓存比例值，公式为 2 * ctx.alpha * ctx.beta / ctx.size
        cache_ratio_value = 2 * ctx.alpha * ctx.beta / ctx.size
        # 计算预填充值
        inversePrePad = int(ctx.size - (ctx.size - 1) / 2)

        # 调整梯度输入张量的大小与输入张量相同
        grad_input.resize_as_(input)
        # 计算 ctx.scale 的 -ctx.beta 次幂，结果保存到 grad_input 中，并与 grad_output 相乘
        torch.pow(ctx.scale, -ctx.beta, out=grad_input).mul_(grad_output)

        # 将填充后的比例张量置零
        paddded_ratio.zero_()
        # 获取填充后比例的中心部分，尺寸为 (channels, input_height, input_width)
        padded_ratio_center = paddded_ratio.narrow(0, inversePrePad, channels)
        # 遍历每个批次中的样本
        for n in range(batch_size):
            # 将 grad_output[n] 与 output[n] 相乘，结果保存到 padded_ratio_center 中
            torch.mul(grad_output[n], output[n], out=padded_ratio_center)
            # 将 padded_ratio_center 按 ctx.scale[n] 进行除法
            padded_ratio_center.div_(ctx.scale[n])
            # 计算填充后比例张量中的累积值，保存到 accum_ratio 中
            torch.sum(
                paddded_ratio.narrow(0, 0, ctx.size - 1),
                0,
                keepdim=False,
                out=accum_ratio,
            )
            # 遍历每个通道
            for c in range(channels):
                # 累积计算梯度输入张量中的每个通道值
                accum_ratio.add_(paddded_ratio[c + ctx.size - 1])
                grad_input[n][c].addcmul_(
                    input[n][c], accum_ratio, value=-cache_ratio_value
                )
                accum_ratio.add_(paddded_ratio[c], alpha=-1)

        # 返回梯度输入张量和空值（None），其余的输出值在此函数中并未使用
        return grad_input, None, None, None, None
class BackwardHookFunction(torch.autograd.Function):
    # 定义静态方法 forward，用于前向传播计算
    @staticmethod
    def forward(ctx, *args):
        # 在上下文中标记不需要梯度的参数
        ctx.mark_non_differentiable(*[arg for arg in args if not arg.requires_grad])
        # 返回参数 args，不进行任何修改
        return args

    # 定义静态方法 backward，用于反向传播计算
    @staticmethod
    def backward(ctx, *args):
        # 直接返回传入的参数 args，因为此处不需要进行反向传播的操作
        return args
```