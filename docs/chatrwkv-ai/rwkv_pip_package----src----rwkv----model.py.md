# `ChatRWKV\rwkv_pip_package\src\rwkv\model.py`

```py
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
# 导入所需的库
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
# 设置一些 CUDA 相关的参数
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# 获取当前文件所在目录的路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 根据环境变量判断是否开启 JIT 编译
if os.environ.get('RWKV_JIT_ON') != '0':
    os.environ["RWKV_JIT_ON"] = '1'
    # 如果开启 JIT 编译，则使用 torch.jit 提供的相关方法
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    # 如果未开启 JIT 编译，则使用 torch.nn 提供的相关方法
    MyModule = torch.nn.Module
    # 定义一个空函数，用于占位
    def __nop(ob):
        return ob
    MyFunction = __nop
    MyStatic = __nop

# 根据环境变量判断是否开启 CUDA
if os.environ.get('RWKV_CUDA_ON') == '1':
    # 导入 CUDA 相关的扩展
    from torch.utils.cpp_extension import load
    try:
        # 加载 CUDA 相关的扩展模块
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
            verbose=True,
            extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            is_python_module=False)
        # 设置是否禁用 CUBLAS GEMM
        DISABLE_CUBLAS_GEMM = False
    # 捕获所有异常并打印错误信息
    except:
        print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
        # 载入 CUDA 模块
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
            verbose=True,
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
            is_python_module=False)
        # 设置禁用 cuBLAS GEMM 标志
        DISABLE_CUBLAS_GEMM = True

    # 定义一个名为 cuda_wkv 的静态方法
    @MyStatic
    def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
        # 断言条件，确保满足特定的条件
        assert 1 * C % min(C, 32) == 0
        assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
        assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
        # 使张量连续
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # 创建一个空张量 y
        y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=k.dtype)
        # 调用 torch.ops.rwkv.wkv_forward 方法
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
        # 返回结果张量
        return y, aa, bb, pp

    # 定义一个名为 cuda_mm8_seq 的静态方法
    @MyStatic
    def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
        # 断言条件，确保满足特定的条件
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        assert w.dtype == torch.uint8
        assert x.shape == (B, N)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        # 创建一个空张量 y
        y = torch.empty((B, M), device=w.device, dtype=x.dtype)
        # 调用 torch.ops.rwkv.mm8_seq 方法
        torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
        # 返回结果张量
        return y
    # 定义一个名为 cuda_mm8_seq 的静态方法
    @MyStatic
    # 定义一个函数，用于执行CUDA加速的矩阵乘法运算
    def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
        # 断言输入的张量数据类型都相同
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        # 断言输入的张量数据类型为32位浮点数或16位浮点数
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        # 断言权重张量的数据类型为无符号8位整数
        assert w.dtype == torch.uint8
        # 断言输入张量的形状为(N,)
        assert x.shape == (N,)
        # 断言权重张量的形状为(N, M)
        assert w.shape == (N, M)
        # 断言反向传播的输入张量的形状为(M,)
        assert rx.shape == mx.shape == (M,)
        # 断言反向传播的输出张量的形状为(N, 1)
        assert ry.shape == my.shape == (N, 1)
        # 创建一个全零张量y，形状为(M,)，存储在与权重张量相同的设备上，数据类型为32位浮点数
        y = torch.zeros((M,), device=w.device, dtype=torch.float32)
        # 调用CUDA操作执行矩阵乘法运算
        torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
        # 将结果张量y的数据类型转换为输入张量x的数据类型，并返回
        return y.to(dtype=x.dtype)
# 如果条件不满足，则将环境变量"RWKV_CUDA_ON"设置为'0'
else:
    os.environ["RWKV_CUDA_ON"] = '0'

# 定义一个静态方法，用于执行矩阵乘法操作
@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# 定义一个静态方法，用于执行矩阵乘法操作
@MyStatic
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# 如果环境变量"RWKV_CUDA_ON"的值为'1'，则执行以下代码块
if os.environ.get('RWKV_CUDA_ON') == '1':
    # 定义一个静态方法，用于执行矩阵乘法操作
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        # 如果输入的张量w在cuda设备上，并且数据类型为torch.float16，则执行以下代码块
        if w.device.type == 'cuda' and x.dtype == torch.float16:
            B, N, M = x.shape[0], w.shape[0], w.shape[1]
            # 调用cuda_mm8_seq函数执行矩阵乘法操作
            return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
        else:
            # 否则调用torch_mm8_seq函数执行矩阵乘法操作
            return torch_mm8_seq(x, w, mx, rx, my, ry)
    # 定义一个静态方法，用于执行矩阵乘法操作
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        # 如果输入的张量w在cuda设备上，则执行以下代码块
        if w.device.type == 'cuda':
            N, M = w.shape[0], w.shape[1]
            # 调用cuda_mm8_one函数执行矩阵乘法操作
            return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
        else:
            # 否则调用torch_mm8_one函数执行矩阵乘法操作
            return torch_mm8_one(x, w, mx, rx, my, ry)
# 如果环境变量"RWKV_CUDA_ON"的值不为'1'，则执行以下代码块
else:
    # 定义一个静态方法，用于执行矩阵乘法操作
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        # 调用torch_mm8_seq函数执行矩阵乘法操作
        return torch_mm8_seq(x, w, mx, rx, my, ry)
    # 定义一个静态方法，用于执行矩阵乘法操作
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        # 调用torch_mm8_one函数执行矩阵乘法操作
        return torch_mm8_one(x, w, mx, rx, my, ry)

# 定义一个函数，用于执行矩阵乘法操作
def mm8(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor, rx: torch.Tensor, my: torch.Tensor, ry: torch.Tensor):
    # 如果输入张量x的维度为1，则执行以下代码块
    if len(x.shape) == 1:
        # 调用mm8_one函数执行矩阵乘法操作
        return mm8_one(x, w, mx, rx, my, ry)
    # 否则执行以下代码块
    return mm8_seq(x, w, mx, rx, my, ry)

# 定义一个函数，用于执行矩阵乘法操作
def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    # 如果输出数据类型未指定，则将输出数据类型设置为输入张量a的数据类型
    if output_dtype is None:
        output_dtype = a.dtype
    # 如果输入张量b的数据类型为torch.float16、torch.bfloat16或torch.float32，则执行以下代码块
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        # 断言输入张量a的数据类型与输入张量b的数据类型相同
        assert a.dtype == b.dtype
        # 调用matmul_float函数执行矩阵乘法操作
        return matmul_float(a, b, output_dtype=output_dtype)
    # 如果输入张量b的数据类型为torch.uint8，则执行以下代码块
    elif b.dtype == torch.uint8:
        # 断言输入张量mx、rx、my、ry不为空
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        # 调用mm8函数执行矩阵乘法操作，并将结果转换为指定的输出数据类型
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    # 如果不支持的数据类型，则抛出数值错误异常
    else:
        raise ValueError("Unsupported dtype")
# 检查环境变量 RWKV_CUDA_ON 是否为 '1'，并且 DISABLE_CUBLAS_GEMM 为假
if os.environ.get('RWKV_CUDA_ON') == '1' and not DISABLE_CUBLAS_GEMM:
    # 定义一个函数 matmul_float，用于矩阵乘法，支持输出数据类型的指定
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        # 如果未指定输出数据类型，则使用 a 的数据类型
        if output_dtype is None:
            output_dtype = a.dtype
        # 如果 a 和 b 的数据类型都为 torch.float16，且设备类型为 'cuda'
        if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda':
            # 如果 a 的维度为 1
            if len(a.shape) == 1:
                assert len(b.shape) == 2
                # 创建一个空的张量 c，数据类型为 output_dtype，设备为 a 的设备
                c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
                # 将 a 扩展为二维张量
                a = a.unsqueeze(0)
            else:
                assert len(a.shape) == len(b.shape)
                assert len(a.shape) == 2 or len(a.shape) == 3
                # 如果 a 的维度为 2，则创建一个空的张量 c，形状为 (a.shape[0], b.shape[-1])
                if len(a.shape) == 2:
                    c = torch.empty((a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device)
                else:
                    # 如果 a 的维度为 3，则创建一个空的张量 c，形状为 (a.shape[0], a.shape[1], b.shape[-1])
                    c = torch.empty((a.shape[0], a.shape[1], b.shape[-1]), dtype=output_dtype, device=a.device)
            # 调用 rwkv 模块中的 gemm_fp16_cublas 函数进行矩阵乘法
            torch.ops.rwkv.gemm_fp16_cublas(a, b, c)
            # 返回结果张量 c
            return c
        else:
            # 如果不满足条件，则使用默认的矩阵乘法，并将结果转换为指定的输出数据类型
            return (a @ b).to(output_dtype)
# 如果环境变量 RWKV_CUDA_ON 不为 '1'，或者 DISABLE_CUBLAS_GEMM 为真
else:
    # 定义一个函数 matmul_float，用于矩阵乘法，支持输出数据类型的指定
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        # 使用默认的矩阵乘法，并将结果转换为指定的输出数据类型
        return (a @ b).to(output_dtype)

# 检查环境变量 RWKV_DML_ON 是否为 '1'
if os.environ.get('RWKV_DML_ON') == '1':
    # 导入 torch_directml 模块
    import torch_directml
    # 打印信息：PyTorch with DirectML Enabled

# 定义一个类 RWKV，继承自 MyModule
class RWKV(MyModule):
    # 定义方法 RUN_RWKV_5，用于执行 RWKV_5 操作
    def RUN_RWKV_5(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_5.apply(B, T, C, H, state, r, k, v, w, u)

    # 定义方法 RUN_RWKV_6，用于执行 RWKV_6 操作
    def RUN_RWKV_6(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_6.apply(B, T, C, H, state, r, k, v, w, u)

    # 定义一个装饰器 MyFunction
    @MyFunction
    # 定义一个函数，对输入进行一系列操作并返回结果
    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算加权和
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算 sigmoid 函数
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 计算 relu 函数
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        # 计算加权和
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        # 返回结果
        return x + out, xx

    # 使用自定义函数修饰器
    @MyFunction
    # 定义一个函数，对输入序列进行一系列操作并返回结果
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 在输入序列前添加一个元素
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算加权和
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算 sigmoid 函数
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 计算 relu 函数
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        # 计算加权和
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        # 返回结果
        return x + out, xx[-1,:]

    # 使用自定义函数修饰器
    @MyFunction
    # 定义一个函数，对输入进行一系列操作并返回结果
    def ffn_one_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 对输入进行一系列操作
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        # 计算 sigmoid 函数
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 计算 relu 函数
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        # 计算加权和
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        # 返回结果
        return x + out, xx

    # 使用自定义函数修饰器
    @MyFunction
    # 定义一个函数，对输入进行一系列操作，返回处理后的结果和中间变量
    def ffn_seq_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将输入的历史信息与当前信息拼接起来
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算输入的变化量
        sx = sx - xx
        # 根据输入和变化量计算新的值
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        # 对 rx 进行 Sigmoid 操作
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 对 kx 进行 ReLU 操作并平方
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        # 计算最终输出
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        # 返回处理后的结果和中间变量
        return x + out, xx[-1,:]

    ########################################################################################################

    # 定义一个装饰器修饰的函数
    @MyFunction
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 根据输入和权重计算新的值
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 对 rx 进行 Sigmoid 操作
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 对 kx 和 vx 进行矩阵乘法操作
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        # 根据一系列计算得到新的值
        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        # 对最终结果进行矩阵乘法操作
        out = matmul(r * wkv, ow, omx, orx, omy, ory)
        # 返回处理后的结果和中间变量
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

    @MyFunction
    # 定义一个名为 att_seq 的方法，接受多个参数
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入数据 x 进行 layer normalization，使用 ln_w 和 ln_b 权重和偏置
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将输入数据 xx 添加到历史数据 sx 中
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算加权后的键、值、查询
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 对查询进行 Sigmoid 激活
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 计算键和值
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        # 获取输入数据的时间步数
        T = x.shape[0]
        # 遍历每个时间步
        for t in range(T):
            # 获取当前时间步的键和值
            kk = k[t]
            vv = v[t]
            # 计算权重和概率
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            # 更新历史数据 sx
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        # 计算输出
        out = matmul(r * sx, ow, omx, orx, omy, ory)
        # 返回更新后的输入数据、最后一个时间步的输入数据、更新后的 aa、bb、pp
        return x + out, xx[-1,:], aa, bb, pp

    ########################################################################################################

    # 装饰器，用于修饰下面的函数或方法
    @MyFunction
    # 定义一个名为 att_one_v5 的方法，接受多个参数
    def att_one_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入 x 进行 layer normalization，使用 ln_w 和 ln_b 权重和偏置
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算加权后的 kx、vx、rx
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算 H 和 N 的值
        H = t_decay.shape[0]
        N = x.shape[-1] // H

        # 计算 r、k、v
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        
        # 计算 a 和 out
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        # 对 out 进行 flatten、group normalization 和矩阵乘法
        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回 x 加上 out、xx 和 s
        return x + out, xx, s

    # 装饰器，用于修饰下面的函数
    @MyFunction
    # 定义一个方法，接受多个参数
    def att_seq_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入数据进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将输入数据添加到历史数据中
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算加权后的输入数据
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 获取参数的维度信息
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # 对参数进行重塑和计算
        w = t_decay.reshape(-1, 1)
        u = t_first.reshape(-1, 1)
        ws = w.pow(T).reshape(H, 1, 1)
        ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)
        wk = w.reshape(H, 1, T)
        wb = wk.transpose(-2, -1).flip(1)
        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(H, T, T)

        # 矩阵相乘计算
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)

        # 计算输出
        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v
        
        # 对输出进行处理
        out = out.transpose(0, 1).contiguous().reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回结果
        return x + out, xx[-1,:], s

    ########################################################################################################

    # 装饰器函数
    @MyFunction
    # 定义一个名为 att_one_v5_1 的方法，接受多个参数
    def att_one_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入 x 进行 layer normalization，使用 ln_w 和 ln_b 权重和偏置
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算混合后的 kx，vx，rx，gx
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        # 获取 t_decay 的长度和 x 的最后一个维度的大小
        H = t_decay.shape[0]
        N = x.shape[-1] // H

        # 计算 r，k，v，g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 计算 a 和 out
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        # 对 out 进行扁平化，然后进行 group normalization，使用 lx_w 和 lx_b 权重和偏置，分组数为 H，eps 为 64e-5
        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps=64e-5).squeeze(0)
        # 将 out 转换为与 x 相同的数据类型，并乘以 g
        out = out.to(dtype=x.dtype) * g
        # 使用 ow，omx，orx，omy，ory 进行矩阵相乘
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回 x 加上 out，以及 xx 和 s
        return x + out, xx, s

    # 装饰器，用于修饰下面的函数
    @MyFunction
    # 定义一个方法，接受多个参数
    def att_seq_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入数据进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将输入数据的历史信息与当前数据拼接起来
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算加权后的键、值、查询和门控信息
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        # 获取参数的维度信息
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # 对时间衰减和时间首位进行处理
        w = t_decay.reshape(-1, 1)
        u = t_first.reshape(-1, 1)
        ws = w.pow(T).reshape(H, 1, 1)
        ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)
        wk = w.reshape(H, 1, T)
        wb = wk.transpose(-2, -1).flip(1)
        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(H, T, T)

        # 计算注意力权重
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 计算输出和新的历史信息
        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v
        
        # 对输出进行处理
        out = out.transpose(0, 1).contiguous().reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回处理后的结果
        return x + out, xx[-1,:], s

    ########################################################################################################

    # 装饰器，用于修饰下面的函数
    @MyFunction
    # 定义一个名为 att_seq_v5_2 的方法，接受多个参数
    def att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入数据进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将输入数据添加到历史数据中
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算注意力机制中的 k、v、r、g
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        # 计算矩阵的维度
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # 计算注意力机制中的 r、k、v、g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 创建一个空的张量用于存储输出
        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        # 循环计算注意力机制的输出
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        # 重塑输出的形状
        out = out.reshape(T, H*N)
        # 对输出进行 group normalization
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        # 将输出转换为输入数据的类型，并乘以 g
        out = out.to(dtype=x.dtype) * g
        # 计算最终输出
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回输入数据与输出数据的和，以及 xx 的最后一行和 s
        return x + out, xx[-1,:], s

    ########################################################################################################

    # 装饰器，用于修饰下面的函数
    @MyFunction
    # 定义一个方法，接受多个参数
    def att_one_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入数据进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        
        # 计算输入数据与另一个参数的差值
        sx = sx - xx
        # 计算新的变量
        xxx = xx + sx * x_maa
        # 对变量进行 tanh 激活函数处理
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        # 对变量进行矩阵乘法运算
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        # 将结果按维度拆分成多个变量
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # 计算新的变量
        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        # 计算矩阵的维度
        H = t_decay.shape[0]
        N = x.shape[-1] // H

        # 对变量进行矩阵乘法运算
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
        # 对变量进行计算
        w = t_decay + (torch.tanh(wx @ td_w1) @ td_w2).float().view(H, N, 1)
        w = torch.exp(-torch.exp(w.float()))

        # 对变量进行矩阵乘法运算
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + w * s

        # 对变量进行处理
        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回处理后的结果
        return x + out, xx, s

    # 装饰器，用于修饰下面的函数
    @MyFunction
    # 定义一个方法，接受多个参数
    def att_seq_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 获取 t_decay 的行数
        H = t_decay.shape[0]
        # 获取 x 的最后一个维度的大小，除以 H 得到 N
        N = x.shape[-1] // H
        # 获取 x 的第一个维度的大小
        T = x.shape[0]

        # 对 x 进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算 sx 的增量
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx
        # 计算 xxx
        xxx = xx + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # 计算 wx, kx, vx, rx, gx
        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        # 计算 r, k, v, g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 计算 w
        w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)
        w = torch.exp(-torch.exp(w.float()))
        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        # 循环计算 out
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + w[t] * s

        # 重塑 out
        out = out.reshape(T, H*N)
        # 对 out 进行 group normalization
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        # 转换 out 的数据类型并乘以 g
        out = out.to(dtype=x.dtype) * g
        # 计算最终输出
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回 x 加上 out，以及 xx 的最后一行和 s
        return x + out, xx[-1,:], s
    # 这里是空白的代码块，没有需要注释的内容
```