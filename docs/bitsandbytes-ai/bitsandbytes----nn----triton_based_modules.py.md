# `bitsandbytes\bitsandbytes\nn\triton_based_modules.py`

```py
# 从 functools 模块导入 partial 函数
from functools import partial

# 导入 torch 和 torch.nn 模块
import torch
import torch.nn as nn

# 导入自定义模块中的函数
from bitsandbytes.triton.dequantize_rowwise import dequantize_rowwise
from bitsandbytes.triton.int8_matmul_mixed_dequantize import (
    int8_matmul_mixed_dequantize,
)
from bitsandbytes.triton.int8_matmul_rowwise_dequantize import (
    int8_matmul_rowwise_dequantize,
)
from bitsandbytes.triton.quantize_columnwise_and_transpose import (
    quantize_columnwise_and_transpose,
)
from bitsandbytes.triton.quantize_global import (
    quantize_global,
    quantize_global_transpose,
)
from bitsandbytes.triton.quantize_rowwise import quantize_rowwise
from bitsandbytes.triton.triton_utils import is_triton_available

# 定义一个自定义的 torch.autograd.Function 类
class _switchback_global(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X_3D, W, bias):
        # 将输入 X_3D 重塑为 [N * L, D] 的形状
        X = X_3D.view(-1, X_3D.size(-1))

        # 对 X 进行行向量量化，对 W 进行全局量化
        X_int8, state_X = quantize_rowwise(X)
        W_int8, state_W = quantize_global(W)

        # 保存用于反向传播的变量
        ctx.save_for_backward = X, W

        # 矩阵乘法，融合去量化和添加偏置
        # 这里称为 "mixed"，因为我们混合了行向量量化和全局量化
        return int8_matmul_mixed_dequantize(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)

    @staticmethod
    # 定义反向传播函数，接收上下文和3D梯度张量作为输入
    def backward(ctx, G_3D):
        # 将3D梯度张量重塑为二维张量，形状为[N_out * L, D]
        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        # 从上下文中获取保存的输入张量 X 和权重张量 W
        X, W = ctx.save_for_backward
        # 如果需要计算输入张量 X 的梯度
        if ctx.needs_input_grad[0]:
            # 对 G 进行按行量化，对 W 进行全局量化并转置
            # 对于 W，我们还融合了转置操作，因为只支持 A @ B^T 运算
            # 所以我们先转置一次，然后在矩阵乘法中调用 .t()
            G_int8, state_G = quantize_rowwise(G)
            W_int8, state_W = quantize_global_transpose(W)
            # 计算梯度并反量化
            grad_X = int8_matmul_mixed_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1], -1
            )
        # 如果需要计算权重张量 W 的梯度
        if ctx.needs_input_grad[1]:
            # 反向传播使用标准的权重梯度计算
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        # 如果需要计算偏置项的梯度
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        # 返回计算得到的梯度：输入张量 X 的梯度、权重张量 W 的梯度、偏置项的梯度
        return grad_X, grad_W, grad_bias
# 定义一个自定义的 PyTorch 自动求导函数 _switchback_vectorrize
class _switchback_vectorrize(torch.autograd.Function):

    @staticmethod
    # 前向传播函数，接收输入 X_3D、权重 W 和偏置 bias
    def forward(ctx, X_3D, W, bias):
        # 将输入 X_3D 重塑为 [N * L, D] 的形状
        X = X_3D.view(-1, X_3D.size(-1))

        # 保存计算图中的变量 X 和 W
        ctx.save_for_backward = X, W
        # 对 X 进行按行量化
        # 对 W 进行按行量化（首先按行，后转置）
        X_int8, state_X = quantize_rowwise(X)
        W_int8, state_W = quantize_rowwise(W)

        # 矩阵乘法，融合反量化和添加偏置
        # 调用期望按行量化的 X 和 W 的内核
        return int8_matmul_rowwise_dequantize(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)

    @staticmethod
    # 反向传播函数
    def backward(ctx, G_3D):
        # 从上下文中获取保存的 X 和 W
        X, W = ctx.save_for_backward

        # 将梯度 G_3D 重塑为 [-1, G_3D.size(-1)] 的形状
        G = G_3D.reshape(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        if ctx.needs_input_grad[0]:
            # 对 G 进行按行量化，对 W 进行按列量化并转置
            # 我们稍后对权重调用 .t()，因为只支持 A @ B^T
            G_int8, state_G = quantize_rowwise(G)
            W_int8, state_W = quantize_columnwise_and_transpose(W)
            grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            # 反向传播使用标准的权重梯度
            grad_W = torch.matmul(G.t(), X.to(G.dtype))
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias

# 定义另一个自定义的 PyTorch 自动求导函数 _switchback_global_mem_efficient
class _switchback_global_mem_efficient(torch.autograd.Function):

    @staticmethod
    # 前向传播函数，接受输入 X_3D，权重 W 和偏置 bias
    def forward(ctx, X_3D, W, bias):
        # 将输入 X_3D 重塑为 [N * L, D] 的形状
        X = X_3D.view(-1, X_3D.size(-1))
        # 获取输入 X_3D 的大小
        X_3D_sz = X_3D.size()

        # 对 X 进行逐行量化，对 W 进行全局量化
        X_int8, state_X = quantize_rowwise(X)
        # 释放 X 的内存
        del X
        W_int8, state_W = quantize_global(W)

        # 保存用于反向传播的数据
        ctx.save_for_backward = X_int8, state_X, W_int8, state_W

        # 矩阵乘法，融合反量化和添加偏置
        # 命名为 "mixed" 是因为我们混合了逐行量化和全局量化
        return int8_matmul_mixed_dequantize(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D_sz[:-1], -1)

    @staticmethod
    # 反向传播函数，接受梯度 G_3D
    def backward(ctx, G_3D):
        # 将输入 G_3D 重塑为 [N_out * L, D] 的形状
        G = G_3D.reshape(-1, G_3D.size(-1))
        # 获取输入 G_3D 的大小
        G_3D_sz = G_3D.size()

        # 初始化梯度变量
        grad_X = grad_W = grad_bias = None

        # 从保存的数据中获取 X_int8, state_X, W_int8, state_W
        X_int8, state_X, W_int8, state_W = ctx.save_for_backward
        # 如果需要计算权重的梯度
        if ctx.needs_input_grad[1]:
            # 反量化 X_int8 得到 real_X
            real_X = dequantize_rowwise(X_int8, state_X)
            # 释放 X_int8 的内存
            del X_int8
            # 计算权重的梯度
            grad_W = torch.matmul(G.t(), real_X.to(G.dtype))
            # 释放 real_X 的内存
            del real_X
        # 如果需要计算偏置的梯度
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)
        # 如果需要计算输入 X 的梯度
        if ctx.needs_input_grad[0]:
            # 对 G 进行逐行量化
            G_int8, state_G = quantize_rowwise(G)
            # 释放 G 的内存
            del G
            # 转置 W_int8 并确保连续性
            W_int8 = W_int8.t().contiguous()
            # 计算输入 X 的梯度
            grad_X = int8_matmul_mixed_dequantize(G_int8, W_int8.t(), state_G, state_W, None).view(
                *G_3D_sz[:-1], -1
            )

        return grad_X, grad_W, grad_bias
# 自定义的线性层，继承自 nn.Linear
class SwitchBackLinear(nn.Linear):
    # 初始化函数，接受输入特征数、输出特征数、是否包含偏置等参数
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None,
            vector_wise_quantization: bool = False,
            mem_efficient : bool = False,
        ):
        # 调用父类的初始化函数
        super().__init__(in_features, out_features, bias, device, dtype)

        # 检查是否安装了 triton 库，如果没有则抛出 ImportError
        if not is_triton_available():
            raise ImportError('''Could not import triton. Please install triton to use SwitchBackLinear.
                               Alternatively, you can use bnb.nn.SwitchBackLinearBnb, but it will be slower''')

        # 默认情况下，使用全局量化
        self.vector_wise_quantization = vector_wise_quantization
        # 如果启用了向量化量化
        if self.vector_wise_quantization:
            # 使用向量化量化函数
            self._fn = _switchback_vectorrize
            # 如果启用了内存高效模式
            if mem_efficient:
                # 输出提示信息并退出程序
                print('mem efficient is not supported for vector-wise quantization.')
                exit(1)
        else:
            # 如果启用了内存高效模式
            if mem_efficient:
                # 使用全局内存高效量化函数
                self._fn = _switchback_global_mem_efficient
            else:
                # 使用全局量化函数
                self._fn = _switchback_global

    # 准备进行评估
    def prepare_for_eval(self):
        # 如果只是进行评估，可以预先量化权重而不是在前向传播中进行
        # 注意这是实验性质的，没有经过充分测试
        # 需要显式调用，例如：
        # def cond_prepare(m):
        #     if hasattr(m, "prepare_for_eval"):
        #         m.prepare_for_eval()
        # model.apply(cond_prepare)
        print('=> preparing for eval.')
        # 如果启用了向量化量化
        if self.vector_wise_quantization:
            # 对权重进行按行量化
            W_int8, state_W = quantize_rowwise(self.weight)
        else:
            # 对权重进行全局量化
            W_int8, state_W = quantize_global(self.weight)

        # 注册量化后的权重和状态
        self.register_buffer("W_int8", W_int8)
        self.register_buffer("state_W", state_W)

        # 删除原始权重
        del self.weight
    # 定义一个前向传播函数，接受输入 x
    def forward(self, x):
        # 如果处于训练状态
        if self.training:
            # 调用自定义的函数 apply，传入输入 x、权重和偏置
            return self._fn.apply(x, self.weight, self.bias)
        else:
            # 如果没有为评估准备好，运行标准的前向传播
            if not hasattr(self, "W_int8"):
                return self._fn.apply(x, self.weight, self.bias)

            # 否则，使用预先计算的权重
            # 将输入 x 转换为二维张量
            X = x.view(-1, x.size(-1))
            # 对 X 进行逐行量化
            X_int8, state_X = quantize_rowwise(X)

            # 如果启用向量级量化
            if self.vector_wise_quantization:
                # 执行整数矩阵乘法并反量化
                return int8_matmul_rowwise_dequantize(
                    X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
                ).view(*x.size()[:-1], -1)
            else:
                # 执行混合整数矩阵乘法并反量化
                return int8_matmul_mixed_dequantize(
                    X_int8, self.W_int8.t(), state_X, self.state_W, self.bias
                ).view(*x.size()[:-1], -1)
# 使用 SwitchBackLinear 函数的偏函数，关闭向量智能量化
SwitchBackLinearGlobal = partial(SwitchBackLinear, vector_wise_quantization=False)
# 使用 SwitchBackLinear 函数的偏函数，关闭向量智能量化和内存高效模式
SwitchBackLinearGlobalMemEfficient = partial(SwitchBackLinear, vector_wise_quantization=False, mem_efficient=True)
# 使用 SwitchBackLinear 函数的偏函数，开启向量智能量化
SwitchBackLinearVectorwise = partial(SwitchBackLinear, vector_wise_quantization=True)

# 这是标准的线性函数类
class StandardLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # 将输入数据展平成二维张量
        X = input.view(-1, input.size(-1))

        # 保存计算中需要的张量
        ctx.save_for_backward(X, weight, bias)
        # 计算线性函数的输出
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output.view(*input.size()[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output_3D):
        input, weight, bias = ctx.saved_tensors

        # 将梯度输出展平成二维张量
        grad_output = grad_output_3D.reshape(-1, grad_output_3D.size(-1))

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight.to(grad_output.dtype)).view(*grad_output_3D.size()[:-1], -1)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input.to(grad_output.dtype))
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

# 继承自 nn.Linear 类的标准线性层
class StandardLinear(nn.Linear):

    def forward(self, x):
        return StandardLinearFunction.apply(x, self.weight, self.bias)
```