# `.\rwkv\model.py`

```
# 导入所需的库和模块
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
# 设置一些 Torch 的后端参数
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
# 获取当前文件所在路径
current_path = os.path.dirname(os.path.abspath(__file__))

# 根据环境变量设置是否启用 Torch JIT
if os.environ.get('RWKV_JIT_ON') != '0':
    os.environ["RWKV_JIT_ON"] = '1'
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script
else:
    MyModule = torch.nn.Module
    # 定义一个空函数
    def __nop(ob):
        return ob
    MyFunction = __nop
    MyStatic = __nop

# 根据环境变量设置是否启用 CUDA
if os.environ.get('RWKV_CUDA_ON') == '1':
    # 导入 Torch 的 CUDA 扩展
    from torch.utils.cpp_extension import load
    try:
        # 尝试加载 CUDA 扩展模块
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
            verbose=True,
            extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            is_python_module=False)
        DISABLE_CUBLAS_GEMM = False
    except:
        # 加载失败时回退到使用 torch.matmul
        print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
        load(
            name=f"wkv_cuda",
            sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
            verbose=True,
            extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
            extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
            is_python_module=False)
        DISABLE_CUBLAS_GEMM = True

    # 定义一个 CUDA 函数
    @MyStatic
    def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp):
        # 断言条件
        assert 1 * C % min(C, 32) == 0
        assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
        assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
        # 确保张量是连续的
        w = w.contiguous()
        u = u.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        # 创建一个空张量
        y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=k.dtype)
        # 调用 CUDA 操作
        torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
        return y, aa, bb, pp
    @MyStatic
    # 定义一个函数，用于在 CUDA 上执行矩阵乘法操作，输入参数包括矩阵的维度和数据，以及相关的偏置和缩放参数
    def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
        # 断言输入数据的数据类型
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        # 断言输入数据的数据类型为 float32 或 float16
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        # 断言输入数据的形状
        assert x.shape == (B, N)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        # 创建一个空的输出张量 y，指定设备和数据类型
        y = torch.empty((B, M), device=w.device, dtype=x.dtype)
        # 调用自定义的 CUDA 操作 mm8_seq 进行矩阵乘法操作
        torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
        # 返回计算结果 y
        return y
    
    # 使用装饰器 MyStatic 定义一个静态方法，用于在 CUDA 上执行单个矩阵乘法操作
    @MyStatic
    def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry):
        # 断言输入数据的数据类型
        assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
        # 断言输入数据的数据类型为 float32 或 float16
        assert x.dtype == torch.float32 or x.dtype == torch.float16
        # 断言输入数据的形状
        assert x.shape == (N,)
        assert w.shape == (N, M)
        assert rx.shape == mx.shape == (M,)
        assert ry.shape == my.shape == (N, 1)
        # 创建一个全零的输出张量 y，指定设备和数据类型
        y = torch.zeros((M,), device=w.device, dtype=torch.float32)
        # 调用自定义的 CUDA 操作 mm8_one 进行单个矩阵乘法操作
        torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
        # 将计算结果 y 转换为输入数据 x 的数据类型，并返回
        return y.to(dtype=x.dtype)
# 如果条件不成立，则将环境变量"RWKV_CUDA_ON"设置为'0'
else:
    os.environ["RWKV_CUDA_ON"] = '0'

# 定义一个静态方法，执行矩阵乘法运算
@MyStatic
def torch_mm8_seq(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# 定义一个静态方法，执行矩阵乘法运算
@MyStatic
def torch_mm8_one(x, w, mx, rx, my, ry):
    return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# 如果环境变量"RWKV_CUDA_ON"的值为'1'，则定义两个静态方法
if os.environ.get('RWKV_CUDA_ON') == '1':
    # 定义一个静态方法，根据条件选择使用CUDA加速或普通计算
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        if w.device.type == 'cuda' and x.dtype == torch.float16:
            B, N, M = x.shape[0], w.shape[0], w.shape[1]
            return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_seq(x, w, mx, rx, my, ry)
    # 定义一个静态方法，根据条件选择使用CUDA加速或普通计算
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        if w.device.type == 'cuda':
            N, M = w.shape[0], w.shape[1]
            return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
        else:
            return torch_mm8_one(x, w, mx, rx, my, ry)
else:
    # 如果环境变量"RWKV_CUDA_ON"的值不为'1'，则定义两个静态方法
    @MyStatic
    def mm8_seq(x, w, mx, rx, my, ry):
        return torch_mm8_seq(x, w, mx, rx, my, ry)
    @MyStatic
    def mm8_one(x, w, mx, rx, my, ry):
        return torch_mm8_one(x, w, mx, rx, my, ry)

# 定义一个函数，根据输入的张量执行矩阵乘法运算
def mm8(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor, rx: torch.Tensor, my: torch.Tensor, ry: torch.Tensor):
    if len(x.shape) == 1:
        return mm8_one(x, w, mx, rx, my, ry)
    return mm8_seq(x, w, mx, rx, my, ry)

# 定义一个函数，执行矩阵乘法运算，支持不同数据类型和参数
def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
    if output_dtype is None:
        output_dtype = a.dtype
    if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
        assert a.dtype == b.dtype
        return matmul_float(a, b, output_dtype=output_dtype)
    elif b.dtype == torch.uint8:
        assert mx is not None
        assert rx is not None
        assert my is not None
        assert ry is not None
        return mm8(a, b, mx, rx, my, ry).to(output_dtype)
    else:
        raise ValueError("Unsupported dtype")

# 如果环境变量"RWKV_CUDA_ON"的值为'1'且未禁用CUBLAS GEMM，则执行以下代码
if os.environ.get('RWKV_CUDA_ON') == '1' and not DISABLE_CUBLAS_GEMM:
    # 对两个浮点数矩阵进行矩阵乘法运算，可以指定输出数据类型
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        # 如果未指定输出数据类型，则使用输入矩阵a的数据类型作为输出数据类型
        if output_dtype is None:
            output_dtype = a.dtype
        # 如果输入矩阵a和b的数据类型都为torch.float16，并且在cuda设备上
        if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda':
            # 如果矩阵a的维度为1
            if len(a.shape) == 1:
                assert len(b.shape) == 2
                # 创建一个空的张量c，用于存储结果，指定数据类型和设备
                c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
                # 将矩阵a扩展为2维
                a = a.unsqueeze(0)
            else:
                assert len(a.shape) == len(b.shape)
                assert len(a.shape) == 2 or len(a.shape) == 3
                # torch.empty((*a.shape[:-1], b.shape[-1]))无法与jit一起使用
                # 根据矩阵a和b的维度创建空的张量c，指定数据类型和设备
                if len(a.shape) == 2:
                    c = torch.empty((a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device)
                else:
                    c = torch.empty((a.shape[0], a.shape[1], b.shape[-1]), dtype=output_dtype, device=a.device)
            # 调用torch.ops.rwkv.gemm_fp16_cublas函数进行矩阵乘法运算
            torch.ops.rwkv.gemm_fp16_cublas(a, b, c)
            # 返回结果张量c
            return c
        else:
            # 如果输入矩阵a和b的数据类型不满足条件，则使用普通矩阵乘法运算，并转换为指定的输出数据类型
            return (a @ b).to(output_dtype)
# 如果条件不成立，则定义一个矩阵乘法函数，将结果转换为指定的数据类型
else:
    def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
        return (a @ b).to(output_dtype)

# 如果环境变量 RWKV_DML_ON 的值为 '1'，则导入 torch_directml 模块并打印信息
if os.environ.get('RWKV_DML_ON') == '1':
    import torch_directml
    print("PyTorch with DirectML Enabled")

########################################################################################################

# 定义一个类 RWKV，继承自 MyModule
class RWKV(MyModule):
    # 定义一个方法 RUN_RWKV_5，调用 RWKV_5 的 apply 方法
    def RUN_RWKV_5(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_5.apply(B, T, C, H, state, r, k, v, w, u)

    # 定义一个方法 RUN_RWKV_6，调用 RWKV_6 的 apply 方法
    def RUN_RWKV_6(self, B, T, C, H, state, r, k, v, w, u):
        return self.RWKV_6.apply(B, T, C, H, state, r, k, v, w, u)

    ########################################################################################################

    # 定义一个装饰器为 MyFunction 的方法 ffn_one
    @MyFunction
    def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入 x 进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算 kx 和 rx
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算 r、vx 和输出结果 out
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    # 定义一个装饰器为 MyFunction 的方法 ffn_seq
    @MyFunction
    def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入 x 进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将 sx 与 xx 进行拼接
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算 kx 和 rx
        kx = xx * k_mix + sx * (1 - k_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算 r、vx 和输出结果 out
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1,:]

    # 定义一个装饰器为 MyFunction 的方法 ffn_one_v6
    @MyFunction
    def ffn_one_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入 x 进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算 kx 和 rx
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        # 计算 r、vx 和输出结果 out
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx

    # 定义一个装饰器为 MyFunction 的方法 ffn_seq_v6
    @MyFunction
    def ffn_seq_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
        # 对输入 x 进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将 sx 与 xx 进行拼接
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算 kx 和 rx
        sx = sx - xx
        kx = xx + sx * k_maa
        rx = xx + sx * r_maa

        # 计算 r、vx 和输出结果 out
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
        out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
        return x + out, xx[-1,:]

    ########################################################################################################

    # 定义一个装饰器为 MyFunction 的方法
    @MyFunction
    # 定义一个函数，实现单个注意力机制
    def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算加权后的键、值、查询
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算激活函数
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 矩阵乘法操作
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        # 计算加权和
        ww = t_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
        ww = t_decay + pp
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)

        # 矩阵乘法操作
        out = matmul(r * wkv, ow, omx, orx, omy, ory)
        return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

    # 使用装饰器定义一个函数，实现序列级别的注意力机制
    @MyFunction
    def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将上一步的结果与输入序列进行拼接
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算加权后的键、值、查询
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 计算激活函数
        r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
        # 矩阵乘法操作
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

        # 循环处理序列中的每个时间步
        T = x.shape[0]
        for t in range(T):
            kk = k[t]
            vv = v[t]
            ww = t_first + kk
            p = torch.maximum(pp, ww)
            e1 = torch.exp(pp - p)
            e2 = torch.exp(ww - p)
            # 更新序列加权和
            sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
            ww = t_decay + pp
            p = torch.maximum(ww, kk)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(kk - p)
            aa = e1 * aa + e2 * vv
            bb = e1 * bb + e2
            pp = p
        # 矩阵乘法操作
        out = matmul(r * sx, ow, omx, orx, omy, ory)
        return x + out, xx[-1,:], aa, bb, pp

    ########################################################################################################

    # 使用装饰器定义一个函数
    @MyFunction
    # 定义一个函数，实现一层注意力机制
    def att_one_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入数据进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算加权后的键、值、查询
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 获取隐藏单元数和特征维度
        H = t_decay.shape[0]
        N = x.shape[-1] // H

        # 计算注意力分数
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        # 对输出进行 Group Normalization
        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx, s

    # 使用自定义函数装饰器
    @MyFunction
    # 定义一个函数，实现序列级别的注意力机制
    def att_seq_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
        # 对输入数据进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将上一步的输出添加到输入序列的开头
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算加权后的键、值、查询
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)

        # 获取隐藏单元数、特征维度和序列长度
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # 对时间衰减和时间偏置进行处理
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

        # 计算注意力分数
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)

        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v
        
        # 对输出进行 Group Normalization
        out = out.transpose(0, 1).contiguous().reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype)
        out = matmul(out, ow, omx, orx, omy, ory)

        return x + out, xx[-1,:], s

    ########################################################################################################

    # 使用自定义函数装饰器
    @MyFunction
    # 定义一个函数，接受多个参数
    def att_one_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入 x 进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算 kx, vx, rx, gx
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        # 获取 H 和 N 的值
        H = t_decay.shape[0]
        N = x.shape[-1] // H

        # 计算 r, k, v, g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
        # 计算 a 和 out
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + t_decay * s

        # 对 out 进行一系列操作
        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回更新后的 x, xx, s
        return x + out, xx, s

    # 使用自定义函数装饰器
    @MyFunction
    # 定义一个方法，用于执行序列注意力机制的计算
    def att_seq_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入数据进行 Layer Normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将历史信息与当前输入数据拼接
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算注意力机制中的 k、v、r、g
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

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

        # 计算注意力机制中的 r、k、v、g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 计算输出结果
        out = ((r @ k) * w) @ v + (r @ s) * wb
        s = ws * s + (k * wk) @ v
        
        # 对输出结果进行处理
        out = out.transpose(0, 1).contiguous().reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回更新后的数据
        return x + out, xx[-1,:], s

    ########################################################################################################

    # 装饰器函数
    @MyFunction
    # 定义一个函数，实现序列的注意力机制
    def att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入数据进行 Layer Norm 处理
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 将输入数据添加到历史序列中
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
        # 计算注意力机制中的键、值、记忆和门控信息
        kx = xx * k_mix + sx * (1 - k_mix)
        vx = xx * v_mix + sx * (1 - v_mix)
        rx = xx * r_mix + sx * (1 - r_mix)
        gx = xx * g_mix + sx * (1 - g_mix)

        # 获取参数的维度信息
        H = t_decay.shape[0]
        N = x.shape[-1] // H
        T = x.shape[0]

        # 计算注意力机制中的记忆、键、值和门控信息
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 初始化输出结果的张量
        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        # 遍历时间步，计算输出结果
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + t_decay * s

        # 重塑输出结果的形状
        out = out.reshape(T, H*N)
        # 对输出结果进行 Group Norm 处理
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        # 将输出结果转换为输入数据的数据类型，并乘以门控信息
        out = out.to(dtype=x.dtype) * g
        # 使用权重矩阵计算最终输出结果
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回最终结果和一些中间变量
        return x + out, xx[-1,:], s

    ########################################################################################################

    # 装饰器，用于自定义函数
    @MyFunction
    # 定义一个方法，接收多个参数
    def att_one_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 对输入数据进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        
        # 计算 sx 与 xx 的差值
        sx = sx - xx
        # 计算新的 xx
        xxx = xx + sx * x_maa
        # 对 xxx 进行矩阵乘法和激活函数处理
        xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
        xxx = torch.bmm(xxx, tm_w2).view(5, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # 根据不同的权重参数计算 wx, kx, vx, rx, gx
        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        # 计算 H 和 N
        H = t_decay.shape[0]
        N = x.shape[-1] // H

        # 计算 r, k, v, g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
        # 计算 w
        w = t_decay + (torch.tanh(wx @ td_w1) @ td_w2).float().view(H, N, 1)
        w = torch.exp(-torch.exp(w.float()))

        # 计算 a, out, s
        a = matmul(k, v)
        out = r @ (t_first * a + s)
        s = a + w * s

        # 对 out 进行处理
        out = out.flatten()
        out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回处理后的结果
        return x + out, xx, s

    # 装饰器
    @MyFunction
    # 定义一个方法，用于执行特定的操作
    def att_seq_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
        # 获取 t_decay 的形状的第一个维度大小
        H = t_decay.shape[0]
        # 获取 x 的最后一个维度大小
        N = x.shape[-1] // H
        # 获取 x 的第一个维度大小
        T = x.shape[0]

        # 对输入 x 进行 layer normalization
        xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        # 计算 sx 的更新值
        sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx
        # 对 xx 进行一系列操作
        xxx = xx + sx * x_maa
        xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        # 对不同的变量进行更新
        wx = xx + sx * (w_maa + mw)
        kx = xx + sx * (k_maa + mk)
        vx = xx + sx * (v_maa + mv)
        rx = xx + sx * (r_maa + mr)
        gx = xx + sx * (g_maa + mg)

        # 计算 r、k、v、g
        r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
        v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
        g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

        # 计算权重 w
        w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)
        w = torch.exp(-torch.exp(w.float()))
        out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
        # 循环计算输出
        for t in range(T):
            rt = r[:,t:t+1,:]
            kt = k[:,:,t:t+1]
            vt = v[:,t:t+1,:]
            at = matmul(kt, vt)
            out[t] = (rt @ (t_first * at + s)).squeeze(1)
            s = at + w[t] * s

        # 对输出进行处理
        out = out.reshape(T, H*N)
        out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
        out = out.to(dtype=x.dtype) * g
        out = matmul(out, ow, omx, orx, omy, ory)

        # 返回最终结果
        return x + out, xx[-1,:], s

    ########################################################################################################

    ########################################################################################################
```